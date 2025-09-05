# api/routes.py
from flask import Flask, request, jsonify, Response
from functools import wraps
import uuid
from typing import Dict, Any, Optional
import yaml
from datetime import datetime

from .errors import (
    APIError, schema_validation_error, not_found_error, 
    rate_limit_error, ErrorCode
)
from .schemas import TaskRequestSchema, ApprovalRequestSchema
from .security import validate_tenant
from execution.queue import enqueue_task
from storage.manifest import ManifestStore

app = Flask(__name__)
manifest_store = ManifestStore()

# In-memory task store (replace with actual storage in production)
tasks_db: Dict[str, Dict[str, Any]] = {}
idempotency_cache: Dict[str, str] = {}

def require_headers(f):
    """Decorator to enforce required headers"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check for required headers
        idempotency_key = request.headers.get('Idempotency-Key')
        tenant_id = request.headers.get('X-Tenant-ID')
        
        if not idempotency_key:
            raise schema_validation_error(
                "Missing required header: Idempotency-Key",
                details={"missing_header": "Idempotency-Key"}
            )
        
        if not tenant_id:
            raise schema_validation_error(
                "Missing required header: X-Tenant-ID",
                details={"missing_header": "X-Tenant-ID"}
            )
        
        # Validate tenant
        if not validate_tenant(tenant_id):
            raise schema_validation_error(
                "Invalid tenant ID format",
                details={"header": "X-Tenant-ID", "value": tenant_id}
            )
        
        # Store in request context
        request.idempotency_key = idempotency_key
        request.tenant_id = tenant_id
        
        return f(*args, **kwargs)
    
    return decorated_function

@app.errorhandler(APIError)
def handle_api_error(error: APIError):
    """Global error handler for API errors"""
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    
    # Add any additional headers
    headers = error.get_headers()
    for key, value in headers.items():
        response.headers[key] = value
    
    return response

@app.route('/tasks', methods=['POST'])
@require_headers
def create_task():
    """Create and enqueue a new task"""
    try:
        # Check idempotency
        if request.idempotency_key in idempotency_cache:
            # Return cached response
            task_id = idempotency_cache[request.idempotency_key]
            task = tasks_db.get(task_id)
            if task:
                return jsonify(task), 201
        
        # Validate request body
        schema = TaskRequestSchema()
        try:
            data = schema.load(request.json)
        except Exception as e:
            raise schema_validation_error(
                f"Invalid request body: {str(e)}",
                details={"validation_errors": str(e)}
            )
        
        # Create task
        task_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        
        task = {
            "task_id": task_id,
            "trace_id": trace_id,
            "tenant_id": request.tenant_id,
            "status": "queued",
            "operation": data["operation"],
            "payload": data.get("payload", {}),
            "metadata": data.get("metadata", {}),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "events": []
        }
        
        # Store task
        tasks_db[task_id] = task
        idempotency_cache[request.idempotency_key] = task_id
        
        # Enqueue for execution
        enqueue_task({
            "task_id": task_id,
            "trace_id": trace_id,
            "tenant_id": request.tenant_id,
            **data
        })
        
        # Return response
        response = {
            "task_id": task_id,
            "trace_id": trace_id,
            "status": "queued",
            "created_at": task["created_at"]
        }
        
        return jsonify(response), 201
        
    except APIError:
        raise
    except Exception as e:
        raise APIError(
            error_code=ErrorCode.INTERNAL_ERROR,
            message=f"Failed to create task: {str(e)}",
            status_code=500
        )

@app.route('/tasks/<task_id>', methods=['GET'])
@require_headers
def get_task(task_id: str):
    """Get task status and last event"""
    try:
        # Validate task_id format
        try:
            uuid.UUID(task_id)
        except ValueError:
            raise schema_validation_error(
                "Invalid task ID format",
                details={"task_id": task_id}
            )
        
        # Get task
        task = tasks_db.get(task_id)
        if not task:
            raise not_found_error("Task", task_id)
        
        # Check tenant access
        if task["tenant_id"] != request.tenant_id:
            raise not_found_error("Task", task_id)
        
        # Prepare response
        response = {
            "task_id": task["task_id"],
            "trace_id": task["trace_id"],
            "status": task["status"],
            "created_at": task["created_at"],
            "updated_at": task["updated_at"]
        }
        
        # Add last event if exists
        if task.get("events"):
            last_event = task["events"][-1]
            response["last_event"] = last_event
            
            # Include approval link if HITL required
            if last_event.get("event_type") == "HITL_REQUIRED":
                response["last_event"]["approval_link"] = last_event.get("approval_link")
        
        return jsonify(response), 200
        
    except APIError:
        raise
    except Exception as e:
        raise APIError(
            error_code=ErrorCode.INTERNAL_ERROR,
            message=f"Failed to get task: {str(e)}",
            status_code=500
        )

@app.route('/approvals/<task_id>', methods=['POST'])
@require_headers
def approve_task(task_id: str):
    """Record HITL approval for a task"""
    try:
        # Validate task_id format
        try:
            uuid.UUID(task_id)
        except ValueError:
            raise schema_validation_error(
                "Invalid task ID format",
                details={"task_id": task_id}
            )
        
        # Get task
        task = tasks_db.get(task_id)
        if not task:
            raise not_found_error("Task", task_id)
        
        # Check tenant access
        if task["tenant_id"] != request.tenant_id:
            raise not_found_error("Task", task_id)
        
        # Validate request body
        schema = ApprovalRequestSchema()
        try:
            data = schema.load(request.json)
        except Exception as e:
            raise schema_validation_error(
                f"Invalid request body: {str(e)}",
                details={"validation_errors": str(e)}
            )
        
        # Record approval in manifest
        manifest_store.record_approval(
            task_id=task_id,
            approval_token=data["approval_token"],
            decision=data["decision"],
            reason=data.get("reason"),
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Update task status
        if data["decision"] == "approve":
            task["status"] = "processing"
            task["events"].append({
                "event_type": "APPROVED",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"decision": "approve", "reason": data.get("reason")}
            })
        else:
            task["status"] = "rejected"
            task["events"].append({
                "event_type": "REJECTED",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"decision": "reject", "reason": data.get("reason")}
            })
        
        task["updated_at"] = datetime.utcnow().isoformat()
        
        return '', 204
        
    except APIError:
        raise
    except Exception as e:
        raise APIError(
            error_code=ErrorCode.INTERNAL_ERROR,
            message=f"Failed to record approval: {str(e)}",
            status_code=500
        )

@app.route('/openapi', methods=['GET'])
def get_openapi_spec():
    """Return OpenAPI specification"""
    try:
        with open('api/openapi.yaml', 'r') as f:
            spec = f.read()
        return Response(spec, mimetype='application/yaml')
    except Exception as e:
        raise APIError(
            error_code=ErrorCode.INTERNAL_ERROR,
            message=f"Failed to load OpenAPI spec: {str(e)}",
            status_code=500
        )

# Helper function to simulate rate limiting
def check_rate_limit(tenant_id: str) -> bool:
    """Check if tenant has exceeded rate limit"""
    # Implement actual rate limiting logic here
    # For now, return False (no rate limit)
    return False

# Add rate limit check to create_task
@app.before_request
def rate_limit_check():
    """Check rate limits before processing requests"""
    if request.method == 'POST' and request.path == '/tasks':
        tenant_id = request.headers.get('X-Tenant-ID')
        if tenant_id and check_rate_limit(tenant_id):
            raise rate_limit_error(retry_after=60)
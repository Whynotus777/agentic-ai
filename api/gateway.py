# api/gateway.py - Enterprise API Gateway

import asyncio
import json
import uuid
import time
import hashlib
import hmac
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from functools import wraps
import re

from fastapi import FastAPI, HTTPException, Request, Response, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
import uvicorn
import aioredis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import structlog
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

from orchestrator.core import Orchestrator, Task, AgentCapability, Priority
from execution.layer import TaskQueueManager, QueuedTask, DeliveryGuarantee

logger = structlog.get_logger()

# Metrics
api_requests = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
api_latency = Histogram('api_request_duration_seconds', 'API request latency', ['method', 'endpoint'])
active_connections = Gauge('api_active_connections', 'Active API connections')
rate_limit_hits = Counter('api_rate_limit_hits_total', 'Rate limit hits', ['tenant'])

# Create FastAPI app
app = FastAPI(
    title="Agentic AI Platform API",
    description="Enterprise Multi-Agent AI System",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Security
security = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Request/Response Models
class TaskRequest(BaseModel):
    """Task submission request"""
    description: str = Field(..., min_length=1, max_length=10000)
    capabilities: List[str] = Field(default_factory=list)
    priority: str = Field(default="normal")
    input_data: Dict[str, Any] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    idempotency_key: Optional[str] = None
    callback_url: Optional[str] = None
    timeout_seconds: int = Field(default=300, ge=1, le=3600)
    
    @validator('priority')
    def validate_priority(cls, v):
        valid = ['critical', 'high', 'normal', 'low', 'batch']
        if v.lower() not in valid:
            raise ValueError(f'Priority must be one of {valid}')
        return v.lower()
    
    @validator('capabilities')
    def validate_capabilities(cls, v):
        valid = [c.value for c in AgentCapability]
        for cap in v:
            if cap not in valid:
                raise ValueError(f'Invalid capability: {cap}')
        return v

class TaskResponse(BaseModel):
    """Task submission response"""
    task_id: str
    status: str
    estimated_cost_usd: float
    estimated_duration_seconds: int
    message: str

class TaskStatusResponse(BaseModel):
    """Task status response"""
    task_id: str
    status: str
    progress_percentage: float
    current_step: Optional[str]
    output_data: Optional[Dict[str, Any]]
    artifacts: List[str]
    cost_usd: float
    error: Optional[str]
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]

class ErrorResponse(BaseModel):
    """Error response"""
    code: str
    message: str
    details: Optional[Dict[str, Any]]
    trace_id: Optional[str]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    uptime_seconds: float
    active_tasks: int
    queue_size: int

class MetricsQuery(BaseModel):
    """Metrics query parameters"""
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    tenant_id: Optional[str]
    aggregation: str = Field(default="1h")
    metrics: List[str] = Field(default_factory=lambda: ["all"])

# Authentication & Authorization
class AuthContext:
    """Authentication context"""
    
    def __init__(self, tenant_id: str, user_id: str, roles: List[str], scopes: List[str]):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.roles = roles
        self.scopes = scopes
        self.rate_limit_multiplier = 1.0
        
        # Premium tiers get higher limits
        if 'premium' in roles:
            self.rate_limit_multiplier = 2.0
        elif 'enterprise' in roles:
            self.rate_limit_multiplier = 5.0
    
    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes or '*' in self.scopes
    
    def has_role(self, role: str) -> bool:
        return role in self.roles or 'admin' in self.roles

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> AuthContext:
    """Verify JWT token and return auth context"""
    
    token = credentials.credentials
    
    try:
        # Verify JWT
        payload = jwt.decode(
            token,
            app.state.jwt_secret,
            algorithms=["HS256"]
        )
        
        # Check expiration
        exp = payload.get('exp')
        if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        
        # Create auth context
        return AuthContext(
            tenant_id=payload.get('tenant_id'),
            user_id=payload.get('user_id'),
            roles=payload.get('roles', []),
            scopes=payload.get('scopes', [])
        )
        
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}"
        )

async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)) -> Optional[AuthContext]:
    """Verify API key and return auth context"""
    
    if not api_key:
        return None
    
    # Look up API key in Redis
    key_data = await app.state.redis.get(f"api_key:{api_key}")
    
    if not key_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    key_info = json.loads(key_data)
    
    # Check if key is active
    if not key_info.get('active'):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is inactive"
        )
    
    return AuthContext(
        tenant_id=key_info.get('tenant_id'),
        user_id=key_info.get('user_id', 'api_user'),
        roles=key_info.get('roles', []),
        scopes=key_info.get('scopes', [])
    )

async def get_auth_context(
    token_auth: Optional[AuthContext] = Depends(verify_token),
    api_key_auth: Optional[AuthContext] = Depends(verify_api_key)
) -> AuthContext:
    """Get authentication context from either JWT or API key"""
    
    # Prefer token auth over API key
    if token_auth:
        return token_auth
    
    if api_key_auth:
        return api_key_auth
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required"
    )

def require_scope(scope: str):
    """Decorator to require specific scope"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, auth: AuthContext = Depends(get_auth_context), **kwargs):
            if not auth.has_scope(scope):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Scope required: {scope}"
                )
            return await func(*args, auth=auth, **kwargs)
        return wrapper
    return decorator

# Rate limiting
async def check_rate_limit(request: Request, auth: AuthContext) -> bool:
    """Check rate limit for tenant"""
    
    tenant_id = auth.tenant_id
    key = f"rate_limit:{tenant_id}:{request.url.path}"
    
    # Get current count
    current = await app.state.redis.incr(key)
    
    if current == 1:
        # Set expiry on first request
        await app.state.redis.expire(key, 60)  # 1 minute window
    
    # Calculate limit based on tier
    base_limit = app.state.default_rate_limit
    limit = int(base_limit * auth.rate_limit_multiplier)
    
    if current > limit:
        rate_limit_hits.labels(tenant=tenant_id).inc()
        return False
    
    return True

# Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to responses"""
    
    response = await call_next(request)
    
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    
    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log and meter all requests"""
    
    start_time = time.time()
    trace_id = request.headers.get('X-Trace-Id', str(uuid.uuid4()))
    
    # Add trace ID to context
    request.state.trace_id = trace_id
    
    # Log request
    logger.info("Request received",
               method=request.method,
               path=request.url.path,
               trace_id=trace_id,
               client=request.client.host if request.client else None)
    
    active_connections.inc()
    
    try:
        response = await call_next(request)
        
        # Add trace ID to response
        response.headers["X-Trace-Id"] = trace_id
        
        # Record metrics
        duration = time.time() - start_time
        api_requests.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        api_latency.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        # Log response
        logger.info("Request completed",
                   method=request.method,
                   path=request.url.path,
                   status=response.status_code,
                   duration_ms=duration * 1000,
                   trace_id=trace_id)
        
        return response
        
    except Exception as e:
        logger.error("Request failed",
                    method=request.method,
                    path=request.url.path,
                    error=str(e),
                    trace_id=trace_id)
        
        api_requests.labels(
            method=request.method,
            endpoint=request.url.path,
            status=500
        ).inc()
        
        raise
    
    finally:
        active_connections.dec()

# API Endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    
    # Get system metrics
    active_tasks = await app.state.redis.get("metrics:active_tasks")
    queue_size = await app.state.redis.get("metrics:queue_size")
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=time.time() - app.state.start_time,
        active_tasks=int(active_tasks) if active_tasks else 0,
        queue_size=int(queue_size) if queue_size else 0
    )

@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness check endpoint"""
    
    # Check dependencies
    try:
        # Check Redis
        await app.state.redis.ping()
        
        # Check database
        # await app.state.db.execute("SELECT 1")
        
        return {"status": "ready"}
        
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )

@app.post("/api/v1/tasks", response_model=TaskResponse, tags=["Tasks"])
@limiter.limit("10/minute")
async def submit_task(
    request: Request,
    task_request: TaskRequest,
    auth: AuthContext = Depends(get_auth_context),
    idempotency_key: Optional[str] = Header(None)
):
    """Submit a new task for processing"""
    
    # Check rate limit
    if not await check_rate_limit(request, auth):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    # Check scope
    if not auth.has_scope("tasks:write"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    # Use provided idempotency key or from header
    idempotency_key = task_request.idempotency_key or idempotency_key
    
    # Create task
    task = Task(
        trace_id=request.state.trace_id,
        tenant_id=auth.tenant_id,
        description=task_request.description,
        input_data=task_request.input_data,
        required_capabilities=[AgentCapability[cap] for cap in task_request.capabilities],
        constraints=task_request.constraints,
        metadata={
            **task_request.metadata,
            'user_id': auth.user_id,
            'submitted_via': 'api'
        },
        idempotency_key=idempotency_key
    )
    
    # Create queued task
    queued_task = QueuedTask(
        id=task.id,
        idempotency_key=idempotency_key,
        tenant_id=auth.tenant_id,
        priority=Priority[task_request.priority.upper()],
        payload={
            'task': task.__dict__,
            'callback_url': task_request.callback_url
        },
        timeout_seconds=task_request.timeout_seconds,
        metadata={
            'trace_id': request.state.trace_id,
            'user_id': auth.user_id
        }
    )
    
    # Enqueue task
    task_id = await app.state.queue_manager.enqueue(
        queued_task,
        DeliveryGuarantee.EXACTLY_ONCE if idempotency_key else DeliveryGuarantee.AT_LEAST_ONCE
    )
    
    logger.info("Task submitted",
               task_id=task_id,
               tenant_id=auth.tenant_id,
               trace_id=request.state.trace_id)
    
    return TaskResponse(
        task_id=task_id,
        status="queued",
        estimated_cost_usd=0.1,  # Would calculate based on task
        estimated_duration_seconds=60,
        message="Task queued for processing"
    )

@app.get("/api/v1/tasks/{task_id}", response_model=TaskStatusResponse, tags=["Tasks"])
async def get_task_status(
    task_id: str,
    auth: AuthContext = Depends(get_auth_context)
):
    """Get task status and results"""
    
    # Check scope
    if not auth.has_scope("tasks:read"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    # Get task from database
    task_data = await app.state.redis.get(f"task:data:{task_id}")
    
    if not task_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    task = json.loads(task_data)
    
    # Check tenant access
    if task.get('tenant_id') != auth.tenant_id and not auth.has_role('admin'):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Get task result if available
    result_data = await app.state.redis.get(f"task:result:{task_id}")
    
    if result_data:
        result = json.loads(result_data)
        status = "completed"
        output_data = result.get('output')
        artifacts = result.get('artifacts', [])
        cost_usd = result.get('cost', 0.0)
        error = result.get('error')
        completed_at = result.get('completed_at')
    else:
        status = task.get('status', 'processing')
        output_data = None
        artifacts = []
        cost_usd = 0.0
        error = None
        completed_at = None
    
    return TaskStatusResponse(
        task_id=task_id,
        status=status,
        progress_percentage=50.0 if status == 'processing' else 100.0,
        current_step=task.get('current_step'),
        output_data=output_data,
        artifacts=artifacts,
        cost_usd=cost_usd,
        error=error,
        created_at=datetime.fromisoformat(task['created_at']),
        updated_at=datetime.utcnow(),
        completed_at=datetime.fromisoformat(completed_at) if completed_at else None
    )

@app.delete("/api/v1/tasks/{task_id}", tags=["Tasks"])
async def cancel_task(
    task_id: str,
    auth: AuthContext = Depends(get_auth_context)
):
    """Cancel a running task"""
    
    # Check scope
    if not auth.has_scope("tasks:write"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    # Get task
    task_data = await app.state.redis.get(f"task:data:{task_id}")
    
    if not task_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    task = json.loads(task_data)
    
    # Check tenant access
    if task.get('tenant_id') != auth.tenant_id and not auth.has_role('admin'):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Cancel task
    # This would send a cancellation message to the executor
    await app.state.redis.set(f"task:cancel:{task_id}", "true", ex=300)
    
    logger.info("Task cancelled",
               task_id=task_id,
               tenant_id=auth.tenant_id)
    
    return {"message": "Task cancellation requested"}

@app.get("/api/v1/tasks", tags=["Tasks"])
async def list_tasks(
    auth: AuthContext = Depends(get_auth_context),
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """List tasks for tenant"""
    
    # Check scope
    if not auth.has_scope("tasks:read"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    # Get task IDs for tenant
    pattern = f"task:data:*"
    cursor = offset
    tasks = []
    
    # Scan Redis for tasks
    cursor, keys = await app.state.redis.scan(
        cursor,
        match=pattern,
        count=limit
    )
    
    for key in keys:
        task_data = await app.state.redis.get(key)
        if task_data:
            task = json.loads(task_data)
            
            # Filter by tenant
            if task.get('tenant_id') == auth.tenant_id or auth.has_role('admin'):
                # Filter by status if specified
                if not status or task.get('status') == status:
                    tasks.append({
                        'task_id': task['id'],
                        'status': task.get('status', 'unknown'),
                        'created_at': task.get('created_at'),
                        'description': task.get('description', '')[:100]
                    })
    
    return {
        'tasks': tasks,
        'total': len(tasks),
        'limit': limit,
        'offset': offset
    }

@app.get("/api/v1/artifacts/{artifact_id}", tags=["Artifacts"])
async def get_artifact(
    artifact_id: str,
    auth: AuthContext = Depends(get_auth_context)
):
    """Download an artifact"""
    
    # Check scope
    if not auth.has_scope("artifacts:read"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    # Get artifact metadata
    artifact_meta = await app.state.redis.get(f"artifact:{artifact_id}")
    
    if not artifact_meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifact not found"
        )
    
    meta = json.loads(artifact_meta)
    
    # Check tenant access
    if meta.get('tenant_id') != auth.tenant_id and not auth.has_role('admin'):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Get artifact from S3
    try:
        response = await app.state.s3.get_object(
            Bucket=app.state.artifact_bucket,
            Key=meta['s3_key']
        )
        
        content = await response['Body'].read()
        
        return Response(
            content=content,
            media_type=meta.get('content_type', 'application/octet-stream'),
            headers={
                'Content-Disposition': f'attachment; filename="{meta.get("filename", artifact_id)}"'
            }
        )
        
    except Exception as e:
        logger.error("Failed to retrieve artifact",
                    artifact_id=artifact_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve artifact"
        )

@app.post("/api/v1/models/{model_name}/invoke", tags=["Models"])
async def invoke_model(
    model_name: str,
    request_body: Dict[str, Any],
    auth: AuthContext = Depends(get_auth_context)
):
    """Direct model invocation endpoint"""
    
    # Check scope
    if not auth.has_scope("models:invoke"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    # Check if model is allowed for tenant
    allowed_models = await app.state.redis.smembers(f"tenant:models:{auth.tenant_id}")
    
    if model_name not in allowed_models and not auth.has_role('admin'):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Model {model_name} not available for your tenant"
        )
    
    # Create a simple task for model invocation
    task = Task(
        tenant_id=auth.tenant_id,
        description=f"Direct invocation of {model_name}",
        input_data=request_body,
        metadata={
            'type': 'direct_invocation',
            'model': model_name,
            'user_id': auth.user_id
        }
    )
    
    # Process synchronously (with timeout)
    try:
        result = await asyncio.wait_for(
            app.state.orchestrator.process_task(task),
            timeout=30.0
        )
        
        return {
            'model': model_name,
            'output': result.output_data,
            'tokens_used': result.tokens_used,
            'cost_usd': result.cost_usd
        }
        
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Model invocation timed out"
        )

@app.get("/api/v1/metrics", tags=["Metrics"])
async def get_metrics(
    query: MetricsQuery = Depends(),
    auth: AuthContext = Depends(get_auth_context)
):
    """Get platform metrics"""
    
    # Check scope
    if not auth.has_scope("metrics:read"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    # Admin can see all tenants, others only their own
    if auth.has_role('admin') and query.tenant_id:
        tenant_id = query.tenant_id
    else:
        tenant_id = auth.tenant_id
    
    metrics = {}
    
    # Get task metrics
    if 'tasks' in query.metrics or 'all' in query.metrics:
        metrics['tasks'] = {
            'total': await app.state.redis.get(f"metrics:{tenant_id}:tasks:total") or 0,
            'completed': await app.state.redis.get(f"metrics:{tenant_id}:tasks:completed") or 0,
            'failed': await app.state.redis.get(f"metrics:{tenant_id}:tasks:failed") or 0,
            'avg_duration_ms': await app.state.redis.get(f"metrics:{tenant_id}:tasks:avg_duration") or 0
        }
    
    # Get cost metrics
    if 'costs' in query.metrics or 'all' in query.metrics:
        month_key = datetime.utcnow().strftime('%Y-%m')
        metrics['costs'] = {
            'current_month': await app.state.redis.get(f"cost:{tenant_id}:{month_key}") or 0,
            'budget': await app.state.redis.get(f"config:budget:{tenant_id}") or 1000.0
        }
    
    # Get model usage
    if 'models' in query.metrics or 'all' in query.metrics:
        model_usage = {}
        pattern = f"metrics:{tenant_id}:model:*:calls"
        cursor = 0
        
        cursor, keys = await app.state.redis.scan(cursor, match=pattern, count=100)
        
        for key in keys:
            model_name = key.split(':')[3]
            calls = await app.state.redis.get(key)
            model_usage[model_name] = int(calls) if calls else 0
        
        metrics['model_usage'] = model_usage
    
    return metrics

@app.get("/metrics", tags=["Monitoring"])
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    
    # Generate Prometheus metrics
    metrics_output = generate_latest()
    
    return Response(
        content=metrics_output,
        media_type="text/plain"
    )

@app.post("/api/v1/admin/tenants", tags=["Admin"])
async def create_tenant(
    tenant_data: Dict[str, Any],
    auth: AuthContext = Depends(get_auth_context)
):
    """Create a new tenant (admin only)"""
    
    if not auth.has_role('admin'):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    tenant_id = tenant_data.get('tenant_id', str(uuid.uuid4()))
    
    # Store tenant configuration
    await app.state.redis.set(
        f"tenant:config:{tenant_id}",
        json.dumps({
            **tenant_data,
            'created_at': datetime.utcnow().isoformat(),
            'created_by': auth.user_id
        })
    )
    
    # Set default budget
    budget = tenant_data.get('budget', 1000.0)
    await app.state.redis.set(f"config:budget:{tenant_id}", budget)
    
    # Set default rate limit
    rate_limit = tenant_data.get('rate_limit', 100)
    await app.state.redis.set(f"config:rate_limit:{tenant_id}:default", rate_limit)
    
    # Set allowed models
    models = tenant_data.get('models', ['o4-mini'])
    for model in models:
        await app.state.redis.sadd(f"tenant:models:{tenant_id}", model)
    
    logger.info("Tenant created",
               tenant_id=tenant_id,
               created_by=auth.user_id)
    
    return {
        'tenant_id': tenant_id,
        'message': 'Tenant created successfully'
    }

@app.websocket("/ws/tasks/{task_id}")
async def task_websocket(websocket, task_id: str):
    """WebSocket endpoint for real-time task updates"""
    
    await websocket.accept()
    
    try:
        # Verify authentication
        auth_header = websocket.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            await websocket.close(code=1008, reason="Unauthorized")
            return
        
        # Subscribe to task updates
        pubsub = app.state.redis.pubsub()
        await pubsub.subscribe(f"task:updates:{task_id}")
        
        # Send updates to client
        async for message in pubsub.listen():
            if message['type'] == 'message':
                await websocket.send_text(message['data'].decode())
        
    except Exception as e:
        logger.error("WebSocket error", task_id=task_id, error=str(e))
    
    finally:
        await websocket.close()

# Startup and shutdown
@app.on_event("startup")
async def startup():
    """Initialize application state"""
    
    logger.info("Starting API Gateway")
    
    # Initialize Redis
    app.state.redis = await aioredis.create_redis_pool('redis://localhost:6379')
    
    # Initialize S3
    import aioboto3
    session = aioboto3.Session()
    app.state.s3 = await session.client('s3').__aenter__()
    
    # Initialize components
    from orchestrator.core import create_orchestrator
    from execution.layer import create_execution_layer
    
    # Create orchestrator
    orchestrator_config = {
        'database_url': 'postgresql://user:pass@localhost/agentic',
        'redis_url': 'redis://localhost:6379'
    }
    app.state.orchestrator = await create_orchestrator(orchestrator_config)
    
    # Create execution layer
    execution_config = {
        'redis_url': 'redis://localhost:6379',
        'nats_url': 'nats://localhost:4222',
        'queue_url': 'https://sqs.us-east-1.amazonaws.com/123456/tasks',
        'dlq_url': 'https://sqs.us-east-1.amazonaws.com/123456/dlq',
        'num_workers': 10
    }
    execution_layer = await create_execution_layer(execution_config)
    app.state.queue_manager = execution_layer['queue_manager']
    
    # Configuration
    app.state.jwt_secret = "your-secret-key"  # Load from env
    app.state.default_rate_limit = 100
    app.state.artifact_bucket = "agentic-ai-artifacts"
    app.state.start_time = time.time()
    
    # Enable instrumentation
    FastAPIInstrumentor.instrument_app(app)
    RequestsInstrumentor().instrument()
    
    logger.info("API Gateway started successfully")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    
    logger.info("Shutting down API Gateway")
    
    # Close connections
    if hasattr(app.state, 'redis'):
        app.state.redis.close()
        await app.state.redis.wait_closed()
    
    if hasattr(app.state, 's3'):
        await app.state.s3.__aexit__(None, None, None)
    
    logger.info("API Gateway shut down")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Trace-Id"]
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*.agentic-ai.com", "localhost"]
)

if __name__ == "__main__":
    uvicorn.run(
        "api.gateway:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )
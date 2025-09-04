# api/gateway.py
"""
Enhanced API Gateway with canonical error codes, required headers,
HITL flows, and comprehensive request handling.
"""

import uuid
import json
import time
import jwt
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass, field, asdict

from fastapi import FastAPI, HTTPException, Request, Response, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentator

tracer = trace.get_tracer(__name__)


class ErrorCode(Enum):
    """Canonical error codes for API responses"""
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    CONFLICT = "CONFLICT"
    RETRY_LATER = "RETRY_LATER"
    INVARIANT_VIOLATION = "INVARIANT_VIOLATION"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    HITL_REQUIRED = "HITL_REQUIRED"
    NOT_FOUND = "NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    INVALID_IDEMPOTENCY_KEY = "INVALID_IDEMPOTENCY_KEY"
    TENANT_SUSPENDED = "TENANT_SUSPENDED"


@dataclass
class APIError:
    """Structured API error response"""
    code: ErrorCode
    message: str
    trace_id: str
    details: Optional[Dict[str, Any]] = None
    retry_after: Optional[int] = None
    approval_link: Optional[str] = None
    
    def to_response(self) -> JSONResponse:
        """Convert to JSON response"""
        content = {
            "error": {
                "code": self.code.value,
                "message": self.message,
                "trace_id": self.trace_id
            }
        }
        
        if self.details:
            content["error"]["details"] = self.details
            
        if self.approval_link:
            content["error"]["approval_link"] = self.approval_link
            
        headers = {}
        if self.retry_after:
            headers["Retry-After"] = str(self.retry_after)
            
        status_code = self._get_status_code()
        
        return JSONResponse(
            status_code=status_code,
            content=content,
            headers=headers
        )
    
    def _get_status_code(self) -> int:
        """Map error code to HTTP status code"""
        mapping = {
            ErrorCode.RATE_LIMIT_EXCEEDED: 429,
            ErrorCode.BUDGET_EXCEEDED: 429,
            ErrorCode.SCHEMA_VALIDATION_FAILED: 400,
            ErrorCode.UNAUTHORIZED: 401,
            ErrorCode.FORBIDDEN: 403,
            ErrorCode.CONFLICT: 409,
            ErrorCode.RETRY_LATER: 503,
            ErrorCode.INVARIANT_VIOLATION: 409,
            ErrorCode.POLICY_BLOCKED: 403,
            ErrorCode.HITL_REQUIRED: 403,
            ErrorCode.NOT_FOUND: 404,
            ErrorCode.INTERNAL_ERROR: 500,
            ErrorCode.SERVICE_UNAVAILABLE: 503,
            ErrorCode.TIMEOUT: 504,
            ErrorCode.INVALID_IDEMPOTENCY_KEY: 400,
            ErrorCode.TENANT_SUSPENDED: 403
        }
        return mapping.get(self.code, 500)


# Pydantic models for request/response

class TaskRequest(BaseModel):
    """Request model for task creation"""
    description: str = Field(..., min_length=1, max_length=10000)
    capabilities: List[str] = Field(default_factory=list)
    priority: str = Field(default="normal", pattern="^(low|normal|high|critical)$")
    input_data: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: Optional[int] = Field(default=30, ge=1, le=3600)
    max_cost_usd: Optional[float] = Field(default=10.0, ge=0.01, le=1000)
    delivery_semantics: Optional[str] = Field(
        default="exactly_once",
        pattern="^(at_most_once|at_least_once|exactly_once)$"
    )
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class TaskResponse(BaseModel):
    """Response model for task operations"""
    task_id: str
    status: str
    created_at: datetime
    estimated_completion: Optional[datetime] = None
    cost_estimate_usd: Optional[float] = None
    approval_required: bool = False
    approval_link: Optional[str] = None


class HITLApprovalRequest(BaseModel):
    """Request model for HITL approval"""
    approval_id: str
    decision: str = Field(..., pattern="^(approve|reject)$")
    reason: Optional[str] = None
    modifications: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: datetime
    services: Dict[str, str]


# Dependencies for request validation

async def validate_idempotency_key(
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key")
) -> Optional[str]:
    """Validate idempotency key format"""
    if idempotency_key:
        # Must be UUID format
        try:
            uuid.UUID(idempotency_key)
            return idempotency_key
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": ErrorCode.INVALID_IDEMPOTENCY_KEY.value,
                        "message": "Idempotency-Key must be a valid UUID"
                    }
                }
            )
    return None


async def validate_tenant_id(
    x_tenant_id: str = Header(..., alias="X-Tenant-ID")
) -> str:
    """Validate tenant ID"""
    if not x_tenant_id:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": ErrorCode.SCHEMA_VALIDATION_FAILED.value,
                    "message": "X-Tenant-ID header is required"
                }
            }
        )
    
    # Check if tenant is active (would query database in production)
    # For demo, blacklist some tenant IDs
    if x_tenant_id in ["suspended-tenant", "blocked-tenant"]:
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "code": ErrorCode.TENANT_SUSPENDED.value,
                    "message": f"Tenant {x_tenant_id} is suspended"
                }
            }
        )
    
    return x_tenant_id


async def validate_auth_token(
    authorization: str = Header(...)
) -> Dict[str, Any]:
    """Validate JWT token"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "code": ErrorCode.UNAUTHORIZED.value,
                    "message": "Invalid authorization header format"
                }
            }
        )
    
    token = authorization[7:]
    
    try:
        # In production, verify with proper secret
        payload = jwt.decode(
            token, 
            "secret_key", 
            algorithms=["HS256"],
            options={"verify_signature": False}  # Demo only!
        )
        return payload
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "code": ErrorCode.UNAUTHORIZED.value,
                    "message": f"Invalid token: {str(e)}"
                }
            }
        )


class RequestContext:
    """Context for request processing"""
    def __init__(self):
        self.trace_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.tenant_id = ""
        self.user_id = ""
        self.idempotency_key = ""
        
        
class APIGateway:
    """
    Main API Gateway application
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app = FastAPI(
            title="Agentic AI API Gateway",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        self.request_cache: Dict[str, Any] = {}
        self.hitl_approvals: Dict[str, Dict] = {}
        self.rate_limiter = GatewayRateLimiter()
        self.budget_tracker = BudgetTracker()
        
        self._setup_middleware()
        self._setup_routes()
        
        # Instrument with OpenTelemetry
        FastAPIInstrumentator.instrument_app(self.app)
    
    def _setup_middleware(self):
        """Configure middleware"""
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get("cors_origins", ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Request context middleware
        @self.app.middleware("http")
        async def add_request_context(request: Request, call_next):
            # Generate trace ID
            trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
            
            # Add to request state
            request.state.trace_id = trace_id
            request.state.start_time = time.time()
            
            # Process request
            response = await call_next(request)
            
            # Add response headers
            response.headers["X-Trace-ID"] = trace_id
            response.headers["X-Response-Time"] = str(
                int((time.time() - request.state.start_time) * 1000)
            )
            
            return response
    
    def _setup_routes(self):
        """Configure API routes"""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint"""
            return HealthResponse(
                status="healthy",
                version="1.0.0",
                timestamp=datetime.utcnow(),
                services={
                    "orchestrator": "healthy",
                    "control_plane": "healthy",
                    "execution_layer": "healthy"
                }
            )
        
        @self.app.post("/api/v1/tasks", response_model=TaskResponse)
        async def create_task(
            task: TaskRequest,
            request: Request,
            idempotency_key: str = Depends(validate_idempotency_key),
            tenant_id: str = Depends(validate_tenant_id),
            auth: Dict = Depends(validate_auth_token)
        ):
            """Create a new task"""
            trace_id = request.state.trace_id
            
            with tracer.start_as_current_span("create_task") as span:
                span.set_attributes({
                    "tenant_id": tenant_id,
                    "user_id": auth.get("sub", ""),
                    "idempotency_key": idempotency_key or "",
                    "priority": task.priority
                })
                
                try:
                    # Check idempotency
                    if idempotency_key:
                        cached = await self._check_idempotency(idempotency_key)
                        if cached:
                            span.add_event("idempotency_cache_hit")
                            return cached
                    
                    # Check rate limits
                    rate_ok = await self.rate_limiter.check_rate(
                        tenant_id, 
                        auth.get("sub", "")
                    )
                    if not rate_ok:
                        raise APIError(
                            code=ErrorCode.RATE_LIMIT_EXCEEDED,
                            message="Rate limit exceeded for tenant",
                            trace_id=trace_id,
                            retry_after=60
                        )
                    
                    # Check budget
                    budget_ok = await self.budget_tracker.check_budget(
                        tenant_id,
                        task.max_cost_usd or 10.0
                    )
                    if not budget_ok:
                        raise APIError(
                            code=ErrorCode.BUDGET_EXCEEDED,
                            message="Budget exceeded for tenant",
                            trace_id=trace_id,
                            details={"remaining_budget_usd": 0}
                        )
                    
                    # Check if HITL approval needed
                    needs_approval = await self._check_hitl_requirement(task, auth)
                    
                    if needs_approval:
                        approval_id = str(uuid.uuid4())
                        approval_link = f"https://approvals.example.com/req/{approval_id}"
                        
                        # Store approval request
                        self.hitl_approvals[approval_id] = {
                            "task": task.dict(),
                            "tenant_id": tenant_id,
                            "user_id": auth.get("sub", ""),
                            "created_at": datetime.utcnow(),
                            "expires_at": datetime.utcnow() + timedelta(minutes=10)
                        }
                        
                        response = TaskResponse(
                            task_id=str(uuid.uuid4()),
                            status="pending_approval",
                            created_at=datetime.utcnow(),
                            approval_required=True,
                            approval_link=approval_link
                        )
                        
                        span.add_event("hitl_approval_required")
                        
                    else:
                        # Process task normally
                        task_id = str(uuid.uuid4())
                        
                        # In production, this would submit to orchestrator
                        response = TaskResponse(
                            task_id=task_id,
                            status="submitted",
                            created_at=datetime.utcnow(),
                            estimated_completion=datetime.utcnow() + timedelta(seconds=task.timeout_seconds or 30),
                            cost_estimate_usd=await self._estimate_cost(task)
                        )
                        
                        # Track budget
                        await self.budget_tracker.record_usage(
                            tenant_id,
                            response.cost_estimate_usd or 0
                        )
                    
                    # Cache for idempotency
                    if idempotency_key:
                        await self._cache_response(idempotency_key, response)
                    
                    return response
                    
                except APIError as e:
                    return e.to_response()
                except Exception as e:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    error = APIError(
                        code=ErrorCode.INTERNAL_ERROR,
                        message=f"Internal error: {str(e)}",
                        trace_id=trace_id
                    )
                    return error.to_response()
        
        @self.app.get("/api/v1/tasks/{task_id}", response_model=TaskResponse)
        async def get_task(
            task_id: str,
            request: Request,
            tenant_id: str = Depends(validate_tenant_id),
            auth: Dict = Depends(validate_auth_token)
        ):
            """Get task status"""
            # In production, query task status from execution layer
            return TaskResponse(
                task_id=task_id,
                status="in_progress",
                created_at=datetime.utcnow() - timedelta(minutes=5),
                estimated_completion=datetime.utcnow() + timedelta(minutes=2)
            )
        
        @self.app.post("/api/v1/approvals", status_code=200)
        async def process_approval(
            approval: HITLApprovalRequest,
            request: Request,
            tenant_id: str = Depends(validate_tenant_id),
            auth: Dict = Depends(validate_auth_token)
        ):
            """Process HITL approval"""
            trace_id = request.state.trace_id
            
            if approval.approval_id not in self.hitl_approvals:
                error = APIError(
                    code=ErrorCode.NOT_FOUND,
                    message="Approval request not found",
                    trace_id=trace_id
                )
                return error.to_response()
            
            approval_data = self.hitl_approvals[approval.approval_id]
            
            # Check expiration
            if datetime.utcnow() > approval_data["expires_at"]:
                del self.hitl_approvals[approval.approval_id]
                error = APIError(
                    code=ErrorCode.CONFLICT,
                    message="Approval request expired",
                    trace_id=trace_id
                )
                return error.to_response()
            
            # Process approval
            if approval.decision == "approve":
                # Submit task with modifications if any
                task_data = approval_data["task"]
                if approval.modifications:
                    task_data.update(approval.modifications)
                
                # In production, submit to orchestrator
                response = {
                    "status": "approved",
                    "task_id": str(uuid.uuid4()),
                    "message": "Task approved and submitted"
                }
            else:
                response = {
                    "status": "rejected",
                    "reason": approval.reason or "No reason provided"
                }
            
            # Clean up
            del self.hitl_approvals[approval.approval_id]
            
            return response
        
        @self.app.websocket("/ws/tasks/{task_id}")
        async def task_websocket(
            websocket,
            task_id: str,
            tenant_id: str = Header(..., alias="X-Tenant-ID")
        ):
            """WebSocket for real-time task updates"""
            await websocket.accept()
            
            try:
                # Send initial status
                await websocket.send_json({
                    "type": "status",
                    "task_id": task_id,
                    "status": "connected"
                })
                
                # In production, subscribe to task events
                # Simulate updates for demo
                import asyncio
                for i in range(5):
                    await asyncio.sleep(2)
                    await websocket.send_json({
                        "type": "progress",
                        "task_id": task_id,
                        "progress": (i + 1) * 20,
                        "message": f"Processing step {i + 1}/5"
                    })
                
                await websocket.send_json({
                    "type": "complete",
                    "task_id": task_id,
                    "result": {"status": "success"}
                })
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
            finally:
                await websocket.close()
    
    async def _check_idempotency(self, key: str) -> Optional[Any]:
        """Check idempotency cache"""
        if key in self.request_cache:
            cached = self.request_cache[key]
            # Check if still valid (24 hour TTL)
            if (datetime.utcnow() - cached["timestamp"]).seconds < 86400:
                return cached["response"]
        return None
    
    async def _cache_response(self, key: str, response: Any):
        """Cache response for idempotency"""
        self.request_cache[key] = {
            "response": response,
            "timestamp": datetime.utcnow()
        }
    
    async def _check_hitl_requirement(
        self, 
        task: TaskRequest, 
        auth: Dict
    ) -> bool:
        """Check if task requires human approval"""
        # High-cost tasks require approval
        if task.max_cost_usd and task.max_cost_usd > 100:
            return True
        
        # Critical priority requires approval
        if task.priority == "critical":
            return True
        
        # Certain capabilities require approval
        dangerous_capabilities = ["database_delete", "production_deploy", "financial_transaction"]
        if any(cap in dangerous_capabilities for cap in task.capabilities):
            return True
        
        return False
    
    async def _estimate_cost(self, task: TaskRequest) -> float:
        """Estimate task cost"""
        # Simple estimation based on capabilities and priority
        base_cost = 0.1
        
        capability_costs = {
            "llm_generate": 0.5,
            "code_generation": 1.0,
            "web_search": 0.05,
            "database_query": 0.02
        }
        
        for cap in task.capabilities:
            base_cost += capability_costs.get(cap, 0.1)
        
        priority_multipliers = {
            "low": 0.5,
            "normal": 1.0,
            "high": 2.0,
            "critical": 5.0
        }
        
        return base_cost * priority_multipliers.get(task.priority, 1.0)
    
    def run(self):
        """Run the API gateway"""
        uvicorn.run(
            self.app,
            host=self.config.get("host", "0.0.0.0"),
            port=self.config.get("port", 8000),
            log_level=self.config.get("log_level", "info")
        )


class GatewayRateLimiter:
    """Rate limiting for API gateway"""
    
    def __init__(self):
        self.limits = {
            "default": {"rpm": 60, "burst": 10},
            "premium": {"rpm": 600, "burst": 50}
        }
        self.requests: Dict[str, List[datetime]] = {}
    
    async def check_rate(self, tenant_id: str, user_id: str) -> bool:
        """Check if request is within rate limits"""
        key = f"{tenant_id}:{user_id}"
        now = datetime.utcnow()
        
        # Get tenant tier (would query database in production)
        tier = "premium" if "premium" in tenant_id else "default"
        limit = self.limits[tier]
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Clean old requests (older than 1 minute)
        self.requests[key] = [
            ts for ts in self.requests[key]
            if (now - ts).seconds < 60
        ]
        
        # Check rate limit
        if len(self.requests[key]) >= limit["rpm"] / 60:  # Per second rate
            return False
        
        self.requests[key].append(now)
        return True


class BudgetTracker:
    """Track and enforce budget limits"""
    
    def __init__(self):
        self.usage: Dict[str, float] = {}
        self.limits: Dict[str, float] = {
            "default": 100.0,  # $100 daily limit
            "premium": 1000.0  # $1000 daily limit
        }
    
    async def check_budget(self, tenant_id: str, cost: float) -> bool:
        """Check if tenant has budget for request"""
        # Get tenant tier
        tier = "premium" if "premium" in tenant_id else "default"
        limit = self.limits[tier]
        
        current_usage = self.usage.get(tenant_id, 0.0)
        return (current_usage + cost) <= limit
    
    async def record_usage(self, tenant_id: str, cost: float):
        """Record cost usage"""
        if tenant_id not in self.usage:
            self.usage[tenant_id] = 0.0
        self.usage[tenant_id] += cost
    
    async def reset_daily(self):
        """Reset daily usage (called by scheduler)"""
        self.usage = {}


# Main entry point
def create_app(config: Optional[Dict] = None) -> FastAPI:
    """Create and configure the API gateway application"""
    if config is None:
        config = {
            "host": "0.0.0.0",
            "port": 8000,
            "cors_origins": ["http://localhost:3000"],
            "log_level": "info"
        }
    
    gateway = APIGateway(config)
    return gateway.app


if __name__ == "__main__":
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "cors_origins": ["*"],
        "log_level": "debug"
    }
    
    gateway = APIGateway(config)
    gateway.run()
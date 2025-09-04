# api/gateway_complete.py
"""
Complete API Gateway implementation with X-Tenant-ID header enforcement,
canonical error enums, Retry-After headers, and comprehensive request validation.
"""

import asyncio
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
from fastapi import FastAPI, Request, Response, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx

from opentelemetry import trace
tracer = trace.get_tracer(__name__)


class ErrorCode(Enum):
    """Canonical error codes for consistent error handling"""
    # Authentication & Authorization
    UNAUTHORIZED = ("AUTH_001", 401, "Authentication required")
    INVALID_CREDENTIALS = ("AUTH_002", 401, "Invalid credentials")
    TOKEN_EXPIRED = ("AUTH_003", 401, "Authentication token expired")
    FORBIDDEN = ("AUTH_004", 403, "Access forbidden")
    INSUFFICIENT_PERMISSIONS = ("AUTH_005", 403, "Insufficient permissions")
    
    # Validation
    INVALID_REQUEST = ("VAL_001", 400, "Invalid request format")
    MISSING_REQUIRED_FIELD = ("VAL_002", 400, "Missing required field")
    INVALID_FIELD_VALUE = ("VAL_003", 400, "Invalid field value")
    REQUEST_TOO_LARGE = ("VAL_004", 413, "Request entity too large")
    
    # Rate Limiting
    RATE_LIMIT_EXCEEDED = ("RATE_001", 429, "Rate limit exceeded")
    QUOTA_EXCEEDED = ("RATE_002", 429, "Quota exceeded")
    CONCURRENT_LIMIT = ("RATE_003", 429, "Concurrent request limit exceeded")
    
    # Resource Errors
    NOT_FOUND = ("RES_001", 404, "Resource not found")
    ALREADY_EXISTS = ("RES_002", 409, "Resource already exists")
    CONFLICT = ("RES_003", 409, "Resource conflict")
    GONE = ("RES_004", 410, "Resource no longer available")
    
    # Processing Errors
    INTERNAL_ERROR = ("PROC_001", 500, "Internal server error")
    SERVICE_UNAVAILABLE = ("PROC_002", 503, "Service temporarily unavailable")
    TIMEOUT = ("PROC_003", 504, "Request timeout")
    DEPENDENCY_FAILURE = ("PROC_004", 502, "Dependency service failure")
    
    # Business Logic
    INVALID_STATE = ("BIZ_001", 400, "Invalid state for operation")
    PRECONDITION_FAILED = ("BIZ_002", 412, "Precondition failed")
    OPERATION_NOT_SUPPORTED = ("BIZ_003", 405, "Operation not supported")
    TENANT_SUSPENDED = ("BIZ_004", 403, "Tenant account suspended")
    
    # Cost & Budget
    BUDGET_EXCEEDED = ("COST_001", 402, "Budget limit exceeded")
    PAYMENT_REQUIRED = ("COST_002", 402, "Payment required")
    INSUFFICIENT_CREDITS = ("COST_003", 402, "Insufficient credits")
    
    def __init__(self, code: str, status: int, message: str):
        self.code = code
        self.status = status
        self.message = message


@dataclass
class TenantInfo:
    """Tenant information"""
    tenant_id: str
    name: str
    tier: str  # free, standard, premium, enterprise
    status: str  # active, suspended, deleted
    rate_limit: int
    quota_limit: int
    budget_limit: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitInfo:
    """Rate limiting information"""
    requests_remaining: int
    requests_limit: int
    reset_time: datetime
    retry_after: int  # seconds


class APIError(Exception):
    """Custom API error with canonical error code"""
    
    def __init__(
        self,
        error_code: ErrorCode,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        self.error_code = error_code
        self.details = details or {}
        self.request_id = request_id or str(uuid.uuid4())
        super().__init__(error_code.message)
    
    def to_response(self) -> JSONResponse:
        """Convert to JSON response"""
        content = {
            "error": {
                "code": self.error_code.code,
                "message": self.error_code.message,
                "details": self.details,
                "request_id": self.request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        headers = {
            "X-Request-ID": self.request_id,
            "X-Error-Code": self.error_code.code
        }
        
        # Add Retry-After header for rate limiting errors
        if self.error_code in [ErrorCode.RATE_LIMIT_EXCEEDED, ErrorCode.QUOTA_EXCEEDED]:
            headers["Retry-After"] = str(self.details.get("retry_after", 60))
        
        # Add Retry-After for service unavailable
        if self.error_code == ErrorCode.SERVICE_UNAVAILABLE:
            headers["Retry-After"] = str(self.details.get("retry_after", 30))
        
        return JSONResponse(
            status_code=self.error_code.status,
            content=content,
            headers=headers
        )


class TenantManager:
    """Manages tenant information and validation"""
    
    def __init__(self):
        self.tenants: Dict[str, TenantInfo] = {}
        self._init_default_tenants()
    
    def _init_default_tenants(self):
        """Initialize default tenants"""
        self.tenants = {
            "tenant-free": TenantInfo(
                tenant_id="tenant-free",
                name="Free Tier Tenant",
                tier="free",
                status="active",
                rate_limit=10,
                quota_limit=1000,
                budget_limit=0.0
            ),
            "tenant-standard": TenantInfo(
                tenant_id="tenant-standard",
                name="Standard Tenant",
                tier="standard",
                status="active",
                rate_limit=100,
                quota_limit=10000,
                budget_limit=100.0
            ),
            "tenant-premium": TenantInfo(
                tenant_id="tenant-premium",
                name="Premium Tenant",
                tier="premium",
                status="active",
                rate_limit=1000,
                quota_limit=100000,
                budget_limit=1000.0
            ),
            "tenant-enterprise": TenantInfo(
                tenant_id="tenant-enterprise",
                name="Enterprise Tenant",
                tier="enterprise",
                status="active",
                rate_limit=10000,
                quota_limit=1000000,
                budget_limit=100000.0
            )
        }
    
    def validate_tenant(self, tenant_id: str) -> TenantInfo:
        """Validate tenant ID and return tenant info"""
        if not tenant_id:
            raise APIError(
                ErrorCode.MISSING_REQUIRED_FIELD,
                {"field": "X-Tenant-ID"}
            )
        
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            raise APIError(
                ErrorCode.NOT_FOUND,
                {"resource": "tenant", "tenant_id": tenant_id}
            )
        
        if tenant.status == "suspended":
            raise APIError(
                ErrorCode.TENANT_SUSPENDED,
                {"tenant_id": tenant_id}
            )
        
        if tenant.status == "deleted":
            raise APIError(
                ErrorCode.GONE,
                {"resource": "tenant", "tenant_id": tenant_id}
            )
        
        return tenant


class RateLimiter:
    """Rate limiting implementation"""
    
    def __init__(self):
        self.request_counts: Dict[str, List[datetime]] = {}
    
    def check_rate_limit(
        self,
        tenant_id: str,
        limit: int
    ) -> RateLimitInfo:
        """Check if request is within rate limit"""
        now = datetime.utcnow()
        
        # Get request history
        if tenant_id not in self.request_counts:
            self.request_counts[tenant_id] = []
        
        # Remove old requests (older than 1 minute)
        cutoff = now - timedelta(minutes=1)
        self.request_counts[tenant_id] = [
            ts for ts in self.request_counts[tenant_id]
            if ts > cutoff
        ]
        
        # Check limit
        current_count = len(self.request_counts[tenant_id])
        
        if current_count >= limit:
            # Calculate reset time
            oldest_request = min(self.request_counts[tenant_id])
            reset_time = oldest_request + timedelta(minutes=1)
            retry_after = int((reset_time - now).total_seconds())
            
            raise APIError(
                ErrorCode.RATE_LIMIT_EXCEEDED,
                {
                    "limit": limit,
                    "current": current_count,
                    "reset_at": reset_time.isoformat(),
                    "retry_after": retry_after
                }
            )
        
        # Add current request
        self.request_counts[tenant_id].append(now)
        
        return RateLimitInfo(
            requests_remaining=limit - current_count - 1,
            requests_limit=limit,
            reset_time=now + timedelta(minutes=1),
            retry_after=0
        )


# Initialize services
app = FastAPI(title="Agentic AI Gateway")
tenant_manager = TenantManager()
rate_limiter = RateLimiter()


# Dependency for tenant validation
async def validate_tenant_header(
    x_tenant_id: str = Header(..., description="Tenant identifier")
) -> TenantInfo:
    """Validate X-Tenant-ID header"""
    return tenant_manager.validate_tenant(x_tenant_id)


# Dependency for request ID
async def get_request_id(
    x_request_id: Optional[str] = Header(None, description="Request ID for tracing")
) -> str:
    """Get or generate request ID"""
    return x_request_id or str(uuid.uuid4())


# Middleware for global error handling
@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    """Handle API errors"""
    return exc.to_response()


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    error_code = ErrorCode.INTERNAL_ERROR
    
    if exc.status_code == 404:
        error_code = ErrorCode.NOT_FOUND
    elif exc.status_code == 400:
        error_code = ErrorCode.INVALID_REQUEST
    elif exc.status_code == 401:
        error_code = ErrorCode.UNAUTHORIZED
    elif exc.status_code == 403:
        error_code = ErrorCode.FORBIDDEN
    
    api_error = APIError(error_code, {"detail": exc.detail})
    return api_error.to_response()


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)
    
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response


# Request/Response models
class TaskRequest(BaseModel):
    """Task creation request"""
    task_type: str = Field(..., description="Type of task to execute")
    payload: Dict[str, Any] = Field(..., description="Task payload")
    priority: int = Field(default=5, ge=1, le=10, description="Task priority")
    timeout_seconds: int = Field(default=300, ge=1, le=3600)
    require_approval: bool = Field(default=False)
    estimated_cost: float = Field(default=0.0, ge=0)
    
    class Config:
        schema_extra = {
            "example": {
                "task_type": "code_generation",
                "payload": {
                    "language": "python",
                    "requirements": "Create a REST API"
                },
                "priority": 5,
                "timeout_seconds": 300
            }
        }


class TaskResponse(BaseModel):
    """Task creation response"""
    task_id: str
    status: str
    created_at: str
    estimated_completion: str
    estimated_cost: float
    requires_approval: bool


@app.post("/api/v1/tasks", response_model=TaskResponse)
@tracer.start_as_current_span("create_task")
async def create_task(
    task: TaskRequest,
    tenant: TenantInfo = Depends(validate_tenant_header),
    request_id: str = Depends(get_request_id),
    response: Response = Response()
):
    """Create a new task"""
    span = trace.get_current_span()
    
    # Rate limiting
    rate_info = rate_limiter.check_rate_limit(tenant.tenant_id, tenant.rate_limit)
    
    # Add rate limit headers
    response.headers["X-RateLimit-Limit"] = str(rate_info.requests_limit)
    response.headers["X-RateLimit-Remaining"] = str(rate_info.requests_remaining)
    response.headers["X-RateLimit-Reset"] = rate_info.reset_time.isoformat()
    
    # Budget check
    if task.estimated_cost > tenant.budget_limit:
        raise APIError(
            ErrorCode.BUDGET_EXCEEDED,
            {
                "estimated_cost": task.estimated_cost,
                "budget_limit": tenant.budget_limit,
                "tenant_id": tenant.tenant_id
            },
            request_id=request_id
        )
    
    # Validate task type
    valid_task_types = ["code_generation", "code_review", "web_search", "analysis"]
    if task.task_type not in valid_task_types:
        raise APIError(
            ErrorCode.INVALID_FIELD_VALUE,
            {
                "field": "task_type",
                "value": task.task_type,
                "valid_values": valid_task_types
            },
            request_id=request_id
        )
    
    # Create task
    task_id = f"task-{uuid.uuid4().hex[:12]}"
    
    span.set_attributes({
        "tenant_id": tenant.tenant_id,
        "request_id": request_id,
        "task_id": task_id,
        "task_type": task.task_type
    })
    
    # Add response headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Task-ID"] = task_id
    response.headers["X-Tenant-ID"] = tenant.tenant_id
    
    return TaskResponse(
        task_id=task_id,
        status="pending",
        created_at=datetime.utcnow().isoformat(),
        estimated_completion=(datetime.utcnow() + timedelta(seconds=task.timeout_seconds)).isoformat(),
        estimated_cost=task.estimated_cost,
        requires_approval=task.require_approval
    )


@app.get("/api/v1/tasks/{task_id}")
async def get_task(
    task_id: str,
    tenant: TenantInfo = Depends(validate_tenant_header),
    request_id: str = Depends(get_request_id),
    response: Response = Response()
):
    """Get task status"""
    # Add headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Tenant-ID"] = tenant.tenant_id
    
    # Mock task lookup
    if not task_id.startswith("task-"):
        raise APIError(
            ErrorCode.NOT_FOUND,
            {"resource": "task", "task_id": task_id},
            request_id=request_id
        )
    
    return {
        "task_id": task_id,
        "status": "completed",
        "result": {"message": "Task completed successfully"}
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "api-gateway"
    }


@app.get("/api/v1/tenant/info")
async def get_tenant_info(
    tenant: TenantInfo = Depends(validate_tenant_header),
    request_id: str = Depends(get_request_id),
    response: Response = Response()
):
    """Get tenant information"""
    response.headers["X-Request-ID"] = request_id
    
    return {
        "tenant_id": tenant.tenant_id,
        "name": tenant.name,
        "tier": tenant.tier,
        "status": tenant.status,
        "limits": {
            "rate_limit": tenant.rate_limit,
            "quota_limit": tenant.quota_limit,
            "budget_limit": tenant.budget_limit
        }
    }


# Error code documentation endpoint
@app.get("/api/v1/errors")
async def list_error_codes():
    """List all error codes"""
    return {
        "error_codes": [
            {
                "code": error.code,
                "status": error.status,
                "message": error.message,
                "name": error.name
            }
            for error in ErrorCode
        ]
    }


# Egress Proxy Implementation
class EgressProxy:
    """
    Egress proxy with domain allow-list and request filtering
    """
    
    def __init__(self):
        self.allowed_domains = self._init_allowed_domains()
        self.blocked_patterns = self._init_blocked_patterns()
        self.request_log = []
    
    def _init_allowed_domains(self) -> Set[str]:
        """Initialize allowed domains"""
        return {
            # AI/ML APIs
            "api.openai.com",
            "api.anthropic.com",
            "generativelanguage.googleapis.com",
            "api.cohere.ai",
            "api.mistral.ai",
            
            # Cloud providers
            "*.amazonaws.com",
            "*.azure.com",
            "*.googleapis.com",
            
            # Data sources
            "api.github.com",
            "api.gitlab.com",
            "*.wikipedia.org",
            
            # Internal services
            "*.svc.cluster.local",
            
            # Monitoring
            "*.datadoghq.com",
            "*.newrelic.com",
            "ingest.sentry.io"
        }
    
    def _init_blocked_patterns(self) -> List[str]:
        """Initialize blocked URL patterns"""
        return [
            r".*\.(exe|dll|sh|bat|ps1)$",  # Executables
            r".*\/\.\.",  # Path traversal
            r".*[;<>|`]",  # Command injection
            r"^file:\/\/",  # Local file access
            r"^ftp:\/\/",  # FTP protocol
            r".*\b(password|secret|key|token)\b.*",  # Sensitive data in URL
        ]
    
    def validate_request(
        self,
        url: str,
        tenant_id: str,
        request_type: str = "GET"
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate egress request
        
        Returns:
            Tuple of (is_allowed, error_message)
        """
        import re
        from urllib.parse import urlparse
        
        # Parse URL
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
        except Exception:
            return False, "Invalid URL format"
        
        # Check protocol
        if parsed.scheme not in ["http", "https"]:
            return False, f"Protocol {parsed.scheme} not allowed"
        
        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if re.match(pattern, url, re.IGNORECASE):
                return False, f"URL matches blocked pattern: {pattern}"
        
        # Check allowed domains
        allowed = False
        for allowed_domain in self.allowed_domains:
            if allowed_domain.startswith("*"):
                # Wildcard domain
                suffix = allowed_domain[1:]
                if domain.endswith(suffix):
                    allowed = True
                    break
            elif domain == allowed_domain:
                allowed = True
                break
        
        if not allowed:
            return False, f"Domain {domain} not in allow-list"
        
        # Log request
        self.request_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "tenant_id": tenant_id,
            "url": url,
            "request_type": request_type,
            "allowed": True
        })
        
        return True, None
    
    async def make_request(
        self,
        url: str,
        tenant_id: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Any] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Make validated egress request"""
        # Validate request
        is_allowed, error = self.validate_request(url, tenant_id, method)
        
        if not is_allowed:
            raise APIError(
                ErrorCode.FORBIDDEN,
                {"reason": error, "url": url}
            )
        
        # Add security headers
        safe_headers = headers or {}
        safe_headers.update({
            "X-Tenant-ID": tenant_id,
            "X-Request-ID": str(uuid.uuid4()),
            "User-Agent": "AgenticAI/1.0"
        })
        
        # Make request
        async with httpx.AsyncClient() as client:
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=safe_headers,
                    json=data if method in ["POST", "PUT", "PATCH"] else None,
                    timeout=timeout
                )
                
                return {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
                }
                
            except httpx.TimeoutException:
                raise APIError(
                    ErrorCode.TIMEOUT,
                    {"url": url, "timeout": timeout}
                )
            except Exception as e:
                raise APIError(
                    ErrorCode.DEPENDENCY_FAILURE,
                    {"url": url, "error": str(e)}
                )


# Initialize egress proxy
egress_proxy = EgressProxy()


@app.post("/api/v1/egress")
async def egress_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    data: Optional[Any] = None,
    tenant: TenantInfo = Depends(validate_tenant_header),
    request_id: str = Depends(get_request_id)
):
    """Make egress request through proxy"""
    result = await egress_proxy.make_request(
        url=url,
        tenant_id=tenant.tenant_id,
        method=method,
        headers=headers,
        data=data
    )
    
    return result


# Circuit breaker implementation
class CircuitBreaker:
    """Circuit breaker for downstream services"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func, *args, **kwargs):
        """Call function with circuit breaker"""
        if self.state == "open":
            if self.last_failure_time:
                if (datetime.utcnow() - self.last_failure_time).seconds > self.recovery_timeout:
                    self.state = "half-open"
                else:
                    raise APIError(
                        ErrorCode.SERVICE_UNAVAILABLE,
                        {
                            "service": func.__name__,
                            "retry_after": self.recovery_timeout
                        }
                    )
        
        try:
            result = await func(*args, **kwargs)
            
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            
            return result
            
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                raise APIError(
                    ErrorCode.SERVICE_UNAVAILABLE,
                    {
                        "service": func.__name__,
                        "retry_after": self.recovery_timeout,
                        "error": str(e)
                    }
                )
            
            raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# api/errors.py
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass

class ErrorCode(Enum):
    """Canonical error codes for API responses"""
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    AUTHORIZATION_FAILED = "AUTHORIZATION_FAILED"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    HITL_REQUIRED = "HITL_REQUIRED"
    EXECUTION_TIMEOUT = "EXECUTION_TIMEOUT"
    DEPENDENCY_FAILED = "DEPENDENCY_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"

@dataclass
class APIError(Exception):
    """Structured API error with code, message, and optional details"""
    error_code: ErrorCode
    message: str
    status_code: int = 400
    details: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None
    retry_after: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to response dictionary"""
        response = {
            "error_code": self.error_code.value,
            "message": self.message
        }
        
        if self.details:
            response["details"] = self.details
        
        if self.trace_id:
            response["trace_id"] = self.trace_id
            
        return response
    
    def get_headers(self) -> Dict[str, str]:
        """Get additional HTTP headers for the error response"""
        headers = {}
        if self.retry_after:
            headers["Retry-After"] = str(self.retry_after)
        return headers

# Error factory functions
def schema_validation_error(message: str, details: Optional[Dict] = None) -> APIError:
    """Create a schema validation error"""
    return APIError(
        error_code=ErrorCode.SCHEMA_VALIDATION_FAILED,
        message=message,
        status_code=400,
        details=details
    )

def authentication_error(message: str = "Authentication failed") -> APIError:
    """Create an authentication error"""
    return APIError(
        error_code=ErrorCode.AUTHENTICATION_FAILED,
        message=message,
        status_code=401
    )

def authorization_error(message: str = "Authorization failed") -> APIError:
    """Create an authorization error"""
    return APIError(
        error_code=ErrorCode.AUTHORIZATION_FAILED,
        message=message,
        status_code=403
    )

def not_found_error(resource: str, resource_id: str) -> APIError:
    """Create a resource not found error"""
    return APIError(
        error_code=ErrorCode.RESOURCE_NOT_FOUND,
        message=f"{resource} with id '{resource_id}' not found",
        status_code=404,
        details={"resource": resource, "resource_id": resource_id}
    )

def rate_limit_error(retry_after: int = 60) -> APIError:
    """Create a rate limit exceeded error"""
    return APIError(
        error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
        message="Rate limit exceeded. Please retry after specified time.",
        status_code=429,
        retry_after=retry_after
    )

def hitl_required_error(task_id: str, approval_link: str) -> APIError:
    """Create a HITL required error/event"""
    return APIError(
        error_code=ErrorCode.HITL_REQUIRED,
        message="Human approval required for this operation",
        status_code=202,
        details={
            "task_id": task_id,
            "approval_link": approval_link
        }
    )

def internal_error(message: str = "Internal server error", trace_id: Optional[str] = None) -> APIError:
    """Create an internal server error"""
    return APIError(
        error_code=ErrorCode.INTERNAL_ERROR,
        message=message,
        status_code=500,
        trace_id=trace_id
    )
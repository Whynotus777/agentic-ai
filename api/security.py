# api/security.py
import uuid
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import hashlib
import hmac

# In-memory tenant store (replace with actual database in production)
TENANT_STORE: Dict[str, Dict[str, Any]] = {
    # Example tenants for testing
    "550e8400-e29b-41d4-a716-446655440000": {
        "name": "Test Tenant 1",
        "status": "active",
        "created_at": "2024-01-01T00:00:00Z"
    }
}

# Rate limit tracking (replace with Redis in production)
RATE_LIMIT_STORE: Dict[str, Dict[str, Any]] = {}

def validate_tenant(tenant_id: str) -> bool:
    """
    Validate tenant ID format and existence
    
    Args:
        tenant_id: The tenant ID from X-Tenant-ID header
        
    Returns:
        True if tenant is valid, False otherwise
    """
    # Validate UUID format
    try:
        uuid.UUID(tenant_id)
    except ValueError:
        return False
    
    # In production, check against actual tenant database
    # For now, accept any valid UUID format
    # Uncomment below to enable strict tenant validation:
    
    # if tenant_id not in TENANT_STORE:
    #     return False
    # 
    # tenant = TENANT_STORE[tenant_id]
    # if tenant.get("status") != "active":
    #     return False
    
    return True

def check_tenant_rate_limit(tenant_id: str, limit: int = 100, window_seconds: int = 3600) -> tuple[bool, Optional[int]]:
    """
    Check if tenant has exceeded rate limit
    
    Args:
        tenant_id: The tenant ID to check
        limit: Maximum requests per window
        window_seconds: Time window in seconds
        
    Returns:
        Tuple of (is_within_limit, retry_after_seconds)
    """
    now = datetime.utcnow()
    window_start = now - timedelta(seconds=window_seconds)
    
    if tenant_id not in RATE_LIMIT_STORE:
        RATE_LIMIT_STORE[tenant_id] = {
            "requests": [],
            "window_start": now
        }
    
    tenant_limits = RATE_LIMIT_STORE[tenant_id]
    
    # Clean old requests outside window
    tenant_limits["requests"] = [
        req_time for req_time in tenant_limits["requests"]
        if req_time > window_start
    ]
    
    # Check if limit exceeded
    if len(tenant_limits["requests"]) >= limit:
        # Calculate retry after
        oldest_request = min(tenant_limits["requests"])
        retry_after = int((oldest_request + timedelta(seconds=window_seconds) - now).total_seconds())
        return False, max(1, retry_after)
    
    # Add current request
    tenant_limits["requests"].append(now)
    return True, None

def verify_idempotency_signature(idempotency_key: str, request_body: str, signature: Optional[str]) -> bool:
    """
    Verify idempotency key signature to prevent replay attacks
    
    Args:
        idempotency_key: The idempotency key from header
        request_body: The request body as string
        signature: Optional signature to verify
        
    Returns:
        True if signature is valid or not required
    """
    if not signature:
        # Signature not required in current implementation
        return True
    
    # In production, implement HMAC signature verification
    # Example implementation:
    # secret_key = get_tenant_secret_key(tenant_id)
    # expected_signature = hmac.new(
    #     secret_key.encode(),
    #     f"{idempotency_key}:{request_body}".encode(),
    #     hashlib.sha256
    # ).hexdigest()
    # return hmac.compare_digest(signature, expected_signature)
    
    return True

def get_tenant_permissions(tenant_id: str) -> Dict[str, bool]:
    """
    Get tenant permissions for operations
    
    Args:
        tenant_id: The tenant ID
        
    Returns:
        Dictionary of operation permissions
    """
    # In production, fetch from database
    # For now, return default permissions
    return {
        "analyze": True,
        "transform": True,
        "commit": True,  # May require additional validation
        "actuate": True  # May require additional validation
    }

def validate_operation_permission(tenant_id: str, operation: str) -> bool:
    """
    Check if tenant has permission for operation
    
    Args:
        tenant_id: The tenant ID
        operation: The operation to check
        
    Returns:
        True if tenant has permission
    """
    permissions = get_tenant_permissions(tenant_id)
    return permissions.get(operation, False)

def generate_approval_token(task_id: str, tenant_id: str) -> str:
    """
    Generate secure approval token for HITL
    
    Args:
        task_id: The task ID
        tenant_id: The tenant ID
        
    Returns:
        Secure approval token
    """
    # In production, use proper JWT or cryptographic token
    # For now, use simple hash
    token_data = f"{task_id}:{tenant_id}:{datetime.utcnow().isoformat()}"
    return hashlib.sha256(token_data.encode()).hexdigest()

def validate_approval_token(token: str, task_id: str, tenant_id: str) -> bool:
    """
    Validate approval token
    
    Args:
        token: The approval token
        task_id: The task ID
        tenant_id: The tenant ID
        
    Returns:
        True if token is valid
    """
    # In production, validate against stored tokens or JWT
    # For now, accept any non-empty token
    return bool(token and len(token) > 0)
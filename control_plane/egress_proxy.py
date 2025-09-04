# control_plane/egress_proxy.py
"""
Egress Proxy with domain allow-list, tool-level RBAC, and content filtering.
Enforces security boundaries for all external tool calls.
"""

import re
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
from urllib.parse import urlparse
import asyncio
import aiohttp

from opentelemetry import trace
from cryptography.fernet import Fernet
import tiktoken

tracer = trace.get_tracer(__name__)


class FilterAction(Enum):
    """Actions for content filtering"""
    PASS = "pass"
    BLOCK = "block"
    REDACT = "redact"
    SANITIZE = "sanitize"


@dataclass
class ToolPermission:
    """Permission definition for a tool"""
    tool_name: str
    allowed_roles: Set[str]
    required_scopes: Set[str] = field(default_factory=set)
    rate_limit_per_minute: int = 60
    max_cost_per_call_usd: float = 1.0
    allowed_domains: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=list)
    require_approval: bool = False
    sandbox_required: bool = False
    data_filters: List[str] = field(default_factory=list)


@dataclass
class ScopedToken:
    """Temporary scoped token for tool access"""
    token_id: str
    tool_name: str
    user_id: str
    tenant_id: str
    scopes: Set[str]
    expires_at: datetime
    max_uses: int = 1
    used_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EgressRequest:
    """Request to make through egress proxy"""
    tool_name: str
    method: str  # GET, POST, etc.
    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[Any] = None
    user_id: str = ""
    tenant_id: str = ""
    trace_id: str = ""
    scoped_token: Optional[str] = None


@dataclass
class EgressResponse:
    """Response from egress proxy"""
    status_code: int
    headers: Dict[str, str]
    body: Any
    filtered: bool = False
    redactions: List[str] = field(default_factory=list)
    blocked_content: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    cost_usd: float = 0.0


class ContentFilter:
    """
    Content filtering for egress responses
    """
    
    def __init__(self):
        self.pii_patterns = {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone": re.compile(r'\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b'),
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "credit_card": re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
            "ip_address": re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
        }
        
        self.jailbreak_patterns = [
            re.compile(r'ignore\s+previous\s+instructions', re.IGNORECASE),
            re.compile(r'disregard\s+all\s+prior', re.IGNORECASE),
            re.compile(r'new\s+instructions?\s*:', re.IGNORECASE),
            re.compile(r'system\s+prompt\s*:', re.IGNORECASE),
            re.compile(r'<\|im_start\|>', re.IGNORECASE),
            re.compile(r'\[\[INST\]\]', re.IGNORECASE)
        ]
        
        self.blocked_keywords = {
            "violence": ["murder", "kill", "assault", "torture"],
            "illegal": ["cocaine", "heroin", "meth", "crack"],
            "malware": ["ransomware", "trojan", "keylogger", "botnet"]
        }
        
    async def filter_response(
        self,
        content: str,
        filters: List[str]
    ) -> Tuple[str, List[str], List[str]]:
        """
        Filter response content
        
        Returns:
            Tuple of (filtered_content, redactions, blocked_items)
        """
        filtered = content
        redactions = []
        blocked = []
        
        if "pii_redactor" in filters:
            filtered, pii_redactions = await self._redact_pii(filtered)
            redactions.extend(pii_redactions)
            
        if "jailbreak_detector" in filters:
            jailbreaks = await self._detect_jailbreaks(filtered)
            if jailbreaks:
                blocked.extend(jailbreaks)
                # Remove jailbreak attempts
                for pattern in self.jailbreak_patterns:
                    filtered = pattern.sub("[REMOVED]", filtered)
                    
        if "content_blocker" in filters:
            filtered, content_blocked = await self._block_harmful_content(filtered)
            blocked.extend(content_blocked)
            
        return filtered, redactions, blocked
    
    async def _redact_pii(self, content: str) -> Tuple[str, List[str]]:
        """Redact PII from content"""
        redacted = content
        redactions = []
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(content)
            if matches:
                for match in matches:
                    # Create consistent hash for audit
                    hash_val = hashlib.sha256(match.encode()).hexdigest()[:8]
                    replacement = f"[REDACTED_{pii_type.upper()}_{hash_val}]"
                    redacted = redacted.replace(match, replacement)
                    redactions.append(f"{pii_type}:{hash_val}")
                    
        return redacted, redactions
    
    async def _detect_jailbreaks(self, content: str) -> List[str]:
        """Detect potential jailbreak attempts"""
        detected = []
        
        for pattern in self.jailbreak_patterns:
            if pattern.search(content):
                detected.append(f"jailbreak_pattern:{pattern.pattern[:30]}")
                
        return detected
    
    async def _block_harmful_content(self, content: str) -> Tuple[str, List[str]]:
        """Block harmful content"""
        filtered = content
        blocked = []
        
        content_lower = content.lower()
        for category, keywords in self.blocked_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    filtered = filtered.replace(keyword, "[BLOCKED]")
                    blocked.append(f"{category}:{keyword}")
                    
        return filtered, blocked


class VaultManager:
    """
    Manages scoped temporary tokens for tool access
    """
    
    def __init__(self, encryption_key: bytes):
        self.fernet = Fernet(encryption_key)
        self.tokens: Dict[str, ScopedToken] = {}
        self.token_usage: Dict[str, List[datetime]] = {}
        
    async def issue_scoped_token(
        self,
        tool_name: str,
        user_id: str,
        tenant_id: str,
        scopes: Set[str],
        ttl_seconds: int = 300
    ) -> str:
        """
        Issue a temporary scoped token for tool access
        """
        token_id = hashlib.sha256(
            f"{tool_name}{user_id}{tenant_id}{time.time()}".encode()
        ).hexdigest()[:16]
        
        token = ScopedToken(
            token_id=token_id,
            tool_name=tool_name,
            user_id=user_id,
            tenant_id=tenant_id,
            scopes=scopes,
            expires_at=datetime.utcnow() + timedelta(seconds=ttl_seconds)
        )
        
        self.tokens[token_id] = token
        
        # Encrypt token data
        token_data = json.dumps({
            "id": token_id,
            "tool": tool_name,
            "exp": token.expires_at.isoformat()
        })
        
        encrypted = self.fernet.encrypt(token_data.encode())
        return encrypted.decode()
    
    async def validate_token(self, encrypted_token: str) -> Optional[ScopedToken]:
        """Validate and return scoped token"""
        try:
            decrypted = self.fernet.decrypt(encrypted_token.encode())
            token_data = json.loads(decrypted.decode())
            
            token_id = token_data["id"]
            if token_id not in self.tokens:
                return None
                
            token = self.tokens[token_id]
            
            # Check expiration
            if datetime.utcnow() > token.expires_at:
                del self.tokens[token_id]
                return None
                
            # Check usage limit
            if token.used_count >= token.max_uses:
                return None
                
            # Increment usage
            token.used_count += 1
            
            # Track usage for rate limiting
            if token_id not in self.token_usage:
                self.token_usage[token_id] = []
            self.token_usage[token_id].append(datetime.utcnow())
            
            return token
            
        except Exception:
            return None
    
    async def revoke_token(self, token_id: str):
        """Revoke a token immediately"""
        if token_id in self.tokens:
            del self.tokens[token_id]
            
    async def cleanup_expired(self):
        """Clean up expired tokens"""
        now = datetime.utcnow()
        expired = [
            tid for tid, token in self.tokens.items()
            if now > token.expires_at
        ]
        
        for tid in expired:
            del self.tokens[tid]
            if tid in self.token_usage:
                del self.token_usage[tid]


class EgressProxy:
    """
    Main egress proxy for controlling external tool access
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tool_permissions: Dict[str, ToolPermission] = {}
        self.content_filter = ContentFilter()
        self.vault = VaultManager(config["encryption_key"].encode())
        self.rate_limiter = RateLimiter()
        self.domain_allowlist = set(config.get("global_allowed_domains", []))
        self.domain_blocklist = set(config.get("global_blocked_domains", []))
        self._load_permissions()
        
    def _load_permissions(self):
        """Load tool permissions from config"""
        for tool_config in self.config.get("tools", []):
            permission = ToolPermission(**tool_config)
            self.tool_permissions[permission.tool_name] = permission
            
    @tracer.start_as_current_span("egress_request")
    async def request(
        self,
        req: EgressRequest,
        user_roles: Set[str]
    ) -> EgressResponse:
        """
        Make a request through the egress proxy
        """
        start_time = time.time()
        span = trace.get_current_span()
        
        span.set_attributes({
            "egress.tool": req.tool_name,
            "egress.method": req.method,
            "egress.url": req.url,
            "egress.user_id": req.user_id,
            "egress.tenant_id": req.tenant_id
        })
        
        try:
            # Validate token
            if req.scoped_token:
                token = await self.vault.validate_token(req.scoped_token)
                if not token:
                    return EgressResponse(
                        status_code=401,
                        headers={},
                        body={"error": "Invalid or expired token"}
                    )
                    
                # Ensure token matches request
                if token.tool_name != req.tool_name:
                    return EgressResponse(
                        status_code=403,
                        headers={},
                        body={"error": "Token not valid for this tool"}
                    )
            else:
                # Issue token for this request
                token = await self.vault.issue_scoped_token(
                    req.tool_name,
                    req.user_id,
                    req.tenant_id,
                    set(),
                    ttl_seconds=60
                )
                
            # Check permissions
            permission_check = await self._check_permissions(
                req, user_roles
            )
            if not permission_check["allowed"]:
                return EgressResponse(
                    status_code=403,
                    headers={},
                    body={"error": permission_check["reason"]}
                )
                
            # Check domain allowlist/blocklist
            domain_check = await self._check_domain(req.url)
            if not domain_check["allowed"]:
                span.add_event("domain_blocked", {"domain": domain_check["domain"]})
                return EgressResponse(
                    status_code=403,
                    headers={},
                    body={"error": f"Domain not allowed: {domain_check['domain']}"}
                )
                
            # Check rate limits
            rate_limit_ok = await self.rate_limiter.check_rate_limit(
                req.tool_name,
                req.user_id,
                req.tenant_id
            )
            if not rate_limit_ok:
                return EgressResponse(
                    status_code=429,
                    headers={"Retry-After": "60"},
                    body={"error": "Rate limit exceeded"}
                )
                
            # Make the actual request
            response = await self._make_request(req)
            
            # Apply content filters
            if req.tool_name in self.tool_permissions:
                permission = self.tool_permissions[req.tool_name]
                if permission.data_filters and isinstance(response.body, str):
                    filtered, redactions, blocked = await self.content_filter.filter_response(
                        response.body,
                        permission.data_filters
                    )
                    
                    response.body = filtered
                    response.filtered = True
                    response.redactions = redactions
                    response.blocked_content = blocked
                    
                    if blocked:
                        span.add_event("content_blocked", {
                            "blocked_count": len(blocked),
                            "blocked_types": blocked[:5]  # First 5 for telemetry
                        })
                        
            # Calculate cost
            response.cost_usd = await self._calculate_cost(req, response)
            
            # Record latency
            response.latency_ms = (time.time() - start_time) * 1000
            
            span.set_attributes({
                "egress.status_code": response.status_code,
                "egress.filtered": response.filtered,
                "egress.redactions_count": len(response.redactions),
                "egress.blocked_count": len(response.blocked_content),
                "egress.latency_ms": response.latency_ms,
                "egress.cost_usd": response.cost_usd
            })
            
            return response
            
        except Exception as e:
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            return EgressResponse(
                status_code=500,
                headers={},
                body={"error": f"Egress proxy error: {str(e)}"}
            )
    
    async def _check_permissions(
        self,
        req: EgressRequest,
        user_roles: Set[str]
    ) -> Dict[str, Any]:
        """Check if user has permission for tool"""
        if req.tool_name not in self.tool_permissions:
            return {
                "allowed": False,
                "reason": f"Unknown tool: {req.tool_name}"
            }
            
        permission = self.tool_permissions[req.tool_name]
        
        # Check role-based access
        if not (user_roles & permission.allowed_roles):
            return {
                "allowed": False,
                "reason": f"Missing required roles: {permission.allowed_roles}"
            }
            
        # Check if approval required
        if permission.require_approval:
            # In production, this would check for HITL approval
            return {
                "allowed": False,
                "reason": "Human approval required"
            }
            
        return {"allowed": True}
    
    async def _check_domain(self, url: str) -> Dict[str, Any]:
        """Check if domain is allowed"""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Check blocklist first
        if domain in self.domain_blocklist:
            return {
                "allowed": False,
                "domain": domain,
                "reason": "Domain is blocked"
            }
            
        # Check against tool-specific domains
        # (would check tool permissions here)
        
        # Check global allowlist
        if self.domain_allowlist:
            # Check exact match and wildcards
            allowed = False
            for pattern in self.domain_allowlist:
                if pattern.startswith("*."):
                    # Wildcard domain
                    if domain.endswith(pattern[1:]):
                        allowed = True
                        break
                elif domain == pattern:
                    allowed = True
                    break
                    
            if not allowed:
                return {
                    "allowed": False,
                    "domain": domain,
                    "reason": "Domain not in allowlist"
                }
                
        return {"allowed": True, "domain": domain}
    
    async def _make_request(self, req: EgressRequest) -> EgressResponse:
        """Make the actual HTTP request"""
        async with aiohttp.ClientSession() as session:
            # Prepare headers
            headers = req.headers.copy()
            headers["X-Trace-ID"] = req.trace_id
            
            # Make request with timeout
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with session.request(
                method=req.method,
                url=req.url,
                headers=headers,
                json=req.body if req.body else None,
                timeout=timeout
            ) as response:
                # Read response
                if "application/json" in response.headers.get("content-type", ""):
                    body = await response.json()
                else:
                    body = await response.text()
                    
                return EgressResponse(
                    status_code=response.status,
                    headers=dict(response.headers),
                    body=body
                )
    
    async def _calculate_cost(
        self,
        req: EgressRequest,
        response: EgressResponse
    ) -> float:
        """Calculate cost of the request"""
        # Simple token-based cost calculation
        if isinstance(response.body, str):
            # Estimate tokens (rough approximation)
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = len(encoding.encode(response.body))
            
            # Cost per 1k tokens (example rates)
            cost_per_1k = 0.002
            return (tokens / 1000) * cost_per_1k
            
        return 0.0


class RateLimiter:
    """
    Rate limiting for tool access
    """
    
    def __init__(self):
        self.request_history: Dict[str, List[datetime]] = {}
        self.limits: Dict[str, int] = {
            "default": 60,  # 60 requests per minute
            "web.search": 30,
            "llm.generate": 10,
            "db.query": 100
        }
        
    async def check_rate_limit(
        self,
        tool_name: str,
        user_id: str,
        tenant_id: str
    ) -> bool:
        """Check if request is within rate limit"""
        key = f"{tenant_id}:{user_id}:{tool_name}"
        now = datetime.utcnow()
        
        # Get or create history
        if key not in self.request_history:
            self.request_history[key] = []
            
        # Clean old entries (older than 1 minute)
        self.request_history[key] = [
            ts for ts in self.request_history[key]
            if (now - ts).seconds < 60
        ]
        
        # Get limit for tool
        limit = self.limits.get(tool_name, self.limits["default"])
        
        # Check if within limit
        if len(self.request_history[key]) >= limit:
            return False
            
        # Add current request
        self.request_history[key].append(now)
        return True


# Example usage
async def main():
    """Example usage of egress proxy"""
    config = {
        "encryption_key": Fernet.generate_key().decode(),
        "global_allowed_domains": [
            "*.wikipedia.org",
            "api.openai.com",
            "*.github.com",
            "stackoverflow.com"
        ],
        "global_blocked_domains": [
            "malware.com",
            "phishing-site.net"
        ],
        "tools": [
            {
                "tool_name": "web.search",
                "allowed_roles": ["agent_runtime", "developer"],
                "rate_limit_per_minute": 30,
                "allowed_domains": ["*.google.com", "*.bing.com"],
                "data_filters": ["pii_redactor", "jailbreak_detector"]
            },
            {
                "tool_name": "llm.generate",
                "allowed_roles": ["agent_runtime"],
                "rate_limit_per_minute": 10,
                "max_cost_per_call_usd": 5.0,
                "data_filters": ["content_blocker"]
            }
        ]
    }
    
    proxy = EgressProxy(config)
    
    # Make a request
    request = EgressRequest(
        tool_name="web.search",
        method="GET",
        url="https://en.wikipedia.org/wiki/Python_(programming_language)",
        user_id="user123",
        tenant_id="tenant456",
        trace_id="trace-xyz-789"
    )
    
    response = await proxy.request(
        request,
        user_roles={"agent_runtime", "developer"}
    )
    
    print(f"Response: {response.status_code}")
    print(f"Filtered: {response.filtered}")
    print(f"Cost: ${response.cost_usd:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
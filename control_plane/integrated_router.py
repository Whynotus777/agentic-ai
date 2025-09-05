# control_plane/integrated_router.py
import yaml
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PolicyMatch:
    """Result of policy matching"""
    policy_name: str
    priority: int
    matched: bool
    match_criteria: Dict[str, Any]
    guardrails: Dict[str, Any]
    fallbacks: List[str]
    policy_mode: str

class IntegratedRouter:
    """
    Router with integrated policy evaluation and guardrail enforcement.
    Includes rate limiting and decision tracing.
    """
    
    def __init__(self, policy_file: str = "control_plane/router_policy.yaml"):
        self.policies = self._load_policies(policy_file)
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.decision_trace_enabled = True
    
    def _load_policies(self, policy_file: str) -> Dict[str, Any]:
        """Load policies from YAML file"""
        try:
            with open(policy_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load policies: {e}")
            return {"policies": [], "guardrail_definitions": {}}
    
    def route(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route request based on policies with full decision trace.
        
        Args:
            request: Incoming request with operation, tags, etc.
            
        Returns:
            Routing decision with trace
        """
        decision_trace = {
            "timestamp": datetime.utcnow().isoformat(),
            "request": request,
            "evaluated_policies": [],
            "matched_policy": None,
            "guardrails": {},
            "fallbacks": [],
            "rate_limit_status": {},
            "decision": None,
            "policy_mode": None
        }
        
        # Evaluate each policy
        for policy in sorted(self.policies.get("policies", []), 
                           key=lambda p: p.get("priority", 0), 
                           reverse=True):
            
            match_result = self._evaluate_policy(policy, request)
            decision_trace["evaluated_policies"].append({
                "name": policy["name"],
                "priority": policy.get("priority", 0),
                "matched": match_result.matched,
                "criteria": match_result.match_criteria
            })
            
            if match_result.matched:
                decision_trace["matched_policy"] = policy["name"]
                decision_trace["guardrails"] = match_result.guardrails
                decision_trace["fallbacks"] = match_result.fallbacks
                decision_trace["policy_mode"] = match_result.policy_mode
                
                # Check rate limit
                if "rate_limit_hz" in match_result.guardrails:
                    rate_limit_hz = match_result.guardrails["rate_limit_hz"]
                    rate_limiter = self._get_rate_limiter(policy["name"], rate_limit_hz)
                    
                    allowed, wait_time = rate_limiter.check()
                    decision_trace["rate_limit_status"] = {
                        "limit_hz": rate_limit_hz,
                        "allowed": allowed,
                        "wait_time_ms": wait_time * 1000 if wait_time else 0,
                        "current_rate": rate_limiter.get_current_rate()
                    }
                    
                    if not allowed and match_result.policy_mode == "enforce":
                        decision_trace["decision"] = "rate_limited"
                        return self._create_response(decision_trace, "rate_limited")
                
                # Check HITL requirement
                if match_result.guardrails.get("require_hitl", False):
                    if not request.get("approvals"):
                        if match_result.policy_mode == "enforce":
                            decision_trace["decision"] = "hitl_required"
                            return self._create_response(decision_trace, "hitl_required")
                        else:
                            decision_trace["guardrails"]["hitl_bypassed"] = True
                
                # Check other guardrails
                guardrail_violations = self._check_guardrails(
                    match_result.guardrails, 
                    request
                )
                
                if guardrail_violations:
                    decision_trace["guardrail_violations"] = guardrail_violations
                    
                    if match_result.policy_mode == "enforce":
                        decision_trace["decision"] = "blocked"
                        return self._create_response(decision_trace, "blocked")
                    elif match_result.policy_mode == "monitor":
                        logger.warning(f"Guardrail violations in monitor mode: {guardrail_violations}")
                
                # All checks passed
                decision_trace["decision"] = "allowed"
                return self._create_response(decision_trace, "allowed")
        
        # No policy matched
        decision_trace["decision"] = "no_policy_match"
        return self._create_response(decision_trace, "default")
    
    def _evaluate_policy(self, policy: Dict[str, Any], 
                        request: Dict[str, Any]) -> PolicyMatch:
        """
        Evaluate if a policy matches the request.
        
        Args:
            policy: Policy definition
            request: Incoming request
            
        Returns:
            PolicyMatch result
        """
        match_criteria = policy.get("match", {})
        matched = True
        
        # Check operation match
        if "operation" in match_criteria:
            operations = match_criteria["operation"]
            if isinstance(operations, list):
                if request.get("operation") not in operations:
                    matched = False
            elif request.get("operation") != operations:
                matched = False
        
        # Check tags match
        if matched and "tags" in match_criteria:
            required_tags = set(match_criteria["tags"])
            request_tags = set(request.get("tags", []))
            if not required_tags.issubset(request_tags):
                matched = False
        
        # Check environment match
        if matched and "environment" in match_criteria:
            environments = match_criteria["environment"]
            if isinstance(environments, list):
                if request.get("environment") not in environments:
                    matched = False
            elif request.get("environment") != environments:
                matched = False
        
        # Check numeric ranges
        for field in ["data_size_gb"]:
            if matched and field in match_criteria:
                value = request.get(field)
                if value is not None:
                    range_spec = match_criteria[field]
                    if "min" in range_spec and value < range_spec["min"]:
                        matched = False
                    if "max" in range_spec and value > range_spec["max"]:
                        matched = False
        
        return PolicyMatch(
            policy_name=policy["name"],
            priority=policy.get("priority", 0),
            matched=matched,
            match_criteria=match_criteria,
            guardrails=policy.get("guardrails", {}),
            fallbacks=policy.get("fallbacks", []),
            policy_mode=policy.get("policy_mode", "enforce")
        )
    
    def _check_guardrails(self, guardrails: Dict[str, Any], 
                         request: Dict[str, Any]) -> List[str]:
        """
        Check guardrail violations.
        
        Args:
            guardrails: Guardrail specifications
            request: Request to check
            
        Returns:
            List of violations
        """
        violations = []
        
        # Check approval count
        if "require_approval_count" in guardrails:
            required = guardrails["require_approval_count"]
            actual = len(request.get("approvals", []))
            if actual < required:
                violations.append(f"insufficient_approvals: {actual}/{required}")
        
        # Check timeout
        if "timeout_seconds" in guardrails:
            # This would be checked during execution
            pass
        
        # Check safety checks
        if "safety_checks" in guardrails:
            required_checks = set(guardrails["safety_checks"])
            completed_checks = set(request.get("safety_checks", []))
            missing = required_checks - completed_checks
            if missing:
                violations.append(f"missing_safety_checks: {list(missing)}")
        
        # Check memory limit
        if "memory_limit_gb" in guardrails:
            requested = request.get("memory_gb", 0)
            limit = guardrails["memory_limit_gb"]
            if requested > limit:
                violations.append(f"memory_exceeded: {requested}GB > {limit}GB")
        
        return violations
    
    def _get_rate_limiter(self, policy_name: str, rate_hz: int) -> 'RateLimiter':
        """Get or create rate limiter for policy"""
        if policy_name not in self.rate_limiters:
            self.rate_limiters[policy_name] = RateLimiter(rate_hz)
        return self.rate_limiters[policy_name]
    
    def _create_response(self, trace: Dict[str, Any], 
                        decision: str) -> Dict[str, Any]:
        """
        Create router response with decision trace.
        
        Args:
            trace: Decision trace
            decision: Final decision
            
        Returns:
            Complete response
        """
        response = {
            "decision": decision,
            "timestamp": trace["timestamp"],
            "policy_matched": trace.get("matched_policy"),
            "policy_mode": trace.get("policy_mode", "enforce")
        }
        
        # Add key information based on decision
        if decision == "allowed":
            response["guardrails"] = trace.get("guardrails", {})
            response["fallbacks"] = trace.get("fallbacks", [])
            
        elif decision == "rate_limited":
            response["rate_limit"] = trace.get("rate_limit_status", {})
            response["retry_after_ms"] = trace.get("rate_limit_status", {}).get("wait_time_ms", 0)
            
        elif decision == "hitl_required":
            response["approval_required"] = True
            response["required_approvals"] = trace.get("guardrails", {}).get("require_approval_count", 1)
            
        elif decision == "blocked":
            response["violations"] = trace.get("guardrail_violations", [])
        
        # Include full trace if enabled
        if self.decision_trace_enabled:
            response["decision_trace"] = trace
        
        return response
    
    def get_policy_for_operation(self, operation: str, 
                                 tags: List[str] = None) -> Optional[Dict[str, Any]]:
        """Get matching policy for an operation"""
        request = {
            "operation": operation,
            "tags": tags or []
        }
        
        result = self.route(request)
        if result.get("policy_matched"):
            for policy in self.policies.get("policies", []):
                if policy["name"] == result["policy_matched"]:
                    return policy
        return None

class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, rate_hz: int):
        self.rate_hz = rate_hz
        self.max_tokens = rate_hz
        self.tokens = float(rate_hz)
        self.last_update = time.time()
        self.request_times = []
    
    def check(self) -> tuple[bool, Optional[float]]:
        """
        Check if request is allowed.
        
        Returns:
            Tuple of (allowed, wait_time_seconds)
        """
        now = time.time()
        
        # Refill tokens
        elapsed = now - self.last_update
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate_hz)
        self.last_update = now
        
        # Clean old request times (keep last second)
        self.request_times = [t for t in self.request_times if now - t < 1.0]
        
        if self.tokens >= 1:
            # Allow request
            self.tokens -= 1
            self.request_times.append(now)
            return True, None
        else:
            # Calculate wait time
            wait_time = (1 - self.tokens) / self.rate_hz
            return False, wait_time
    
    def get_current_rate(self) -> float:
        """Get current request rate (requests per second)"""
        now = time.time()
        recent_requests = [t for t in self.request_times if now - t < 1.0]
        return len(recent_requests)
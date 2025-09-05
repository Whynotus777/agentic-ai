"""
Policy Engine for multi-agent system control plane.
Evaluates requests against policies with enforce/monitor/dry_run modes.
"""
import yaml
import json
import hashlib
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


class PolicyMode(Enum):
    ENFORCE = "enforce"
    MONITOR = "monitor" 
    DRY_RUN = "dry_run"


class PolicyDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    HITL_REQUIRED = "hitl_required"


@dataclass
class RequestContext:
    """Context for policy evaluation."""
    tenant_id: str
    service: str
    capability: str
    domain: str
    user_id: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    budget_used: float = 0.0
    request_size_bytes: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class PolicyEvalResult:
    """Result of policy evaluation."""
    decision: PolicyDecision
    reasons: List[str]
    mode: PolicyMode
    policy_ref: str
    matched_rules: List[str]
    
    def to_dict(self):
        return {
            "decision": self.decision.value,
            "reasons": self.reasons,
            "mode": self.mode.value,
            "policy_ref": self.policy_ref,
            "matched_rules": self.matched_rules
        }


class PolicyEngine:
    """Main policy engine for request evaluation."""
    
    def __init__(self, policy_path: str = "control_plane/policy_engine/policy.yaml"):
        self.policy_path = Path(policy_path)
        self.policies = self._load_policies()
        self._cache = {}  # Simple in-memory cache
        
    def _load_policies(self) -> Dict:
        """Load policy configuration from YAML."""
        if not self.policy_path.exists():
            # Default policies if file doesn't exist
            return self._get_default_policies()
            
        with open(self.policy_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_default_policies(self) -> Dict:
        """Default policy configuration."""
        return {
            "version": "1.0",
            "mode": "enforce",
            "rules": {
                "capability_rules": {
                    "repo.commit": {
                        "hitl_required": True,
                        "reason": "Code commits require human review"
                    },
                    "robot.actuate": {
                        "hitl_required": True,
                        "reason": "Robot actuation requires human approval"
                    },
                    "llm.generate": {
                        "allowed": True,
                        "budget_limit_usd": 100.0
                    },
                    "data.read": {
                        "allowed": True,
                        "pii_check": True
                    }
                },
                "domain_rules": {
                    "robotics": {
                        "default_hitl": True,
                        "budget_limit_usd": 500.0
                    },
                    "simulation": {
                        "default_hitl": False,
                        "budget_limit_usd": 1000.0
                    },
                    "appdev": {
                        "default_hitl": False,
                        "budget_limit_usd": 50.0
                    }
                },
                "budget_thresholds": {
                    "daily_limit_usd": 1000.0,
                    "per_request_limit_usd": 50.0,
                    "alert_threshold_pct": 80
                },
                "egress_scopes": {
                    "allowed_domains": ["*.internal.com", "api.openai.com"],
                    "blocked_domains": ["*.suspicious.com"]
                }
            }
        }
    
    def evaluate(self, request_ctx: RequestContext) -> PolicyEvalResult:
        """
        Evaluate request against policies.
        
        Returns PolicyEvalResult with decision, reasons, mode, and references.
        """
        # Check cache
        cache_key = self._get_cache_key(request_ctx)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        mode = PolicyMode(self.policies.get("mode", "enforce"))
        rules = self.policies.get("rules", {})
        
        decision = PolicyDecision.ALLOW
        reasons = []
        matched_rules = []
        
        # 1. Check capability-specific rules
        cap_rules = rules.get("capability_rules", {}).get(request_ctx.capability, {})
        if cap_rules:
            if cap_rules.get("hitl_required"):
                decision = PolicyDecision.HITL_REQUIRED
                reasons.append(cap_rules.get("reason", f"HITL required for {request_ctx.capability}"))
                matched_rules.append(f"capability:{request_ctx.capability}")
                
            elif not cap_rules.get("allowed", True):
                decision = PolicyDecision.DENY
                reasons.append(f"Capability {request_ctx.capability} is denied")
                matched_rules.append(f"capability:{request_ctx.capability}:deny")
                
            # Check capability budget
            if "budget_limit_usd" in cap_rules:
                if request_ctx.budget_used > cap_rules["budget_limit_usd"]:
                    decision = PolicyDecision.DENY
                    reasons.append(f"Budget exceeded for {request_ctx.capability}")
                    matched_rules.append(f"capability:{request_ctx.capability}:budget")
        
        # 2. Check domain-specific rules
        domain_rules = rules.get("domain_rules", {}).get(request_ctx.domain, {})
        if domain_rules:
            if domain_rules.get("default_hitl") and decision != PolicyDecision.DENY:
                decision = PolicyDecision.HITL_REQUIRED
                reasons.append(f"Domain {request_ctx.domain} requires HITL by default")
                matched_rules.append(f"domain:{request_ctx.domain}:hitl")
                
            # Check domain budget
            if "budget_limit_usd" in domain_rules:
                if request_ctx.budget_used > domain_rules["budget_limit_usd"]:
                    decision = PolicyDecision.DENY
                    reasons.append(f"Domain budget exceeded for {request_ctx.domain}")
                    matched_rules.append(f"domain:{request_ctx.domain}:budget")
        
        # 3. Check global budget thresholds
        budget_rules = rules.get("budget_thresholds", {})
        if request_ctx.budget_used > budget_rules.get("per_request_limit_usd", float('inf')):
            decision = PolicyDecision.DENY
            reasons.append("Per-request budget limit exceeded")
            matched_rules.append("global:budget:per_request")
        
        # 4. Check egress scopes (simplified)
        if request_ctx.resource:
            egress = rules.get("egress_scopes", {})
            if any(domain in request_ctx.resource for domain in egress.get("blocked_domains", [])):
                decision = PolicyDecision.DENY
                reasons.append(f"Blocked egress domain: {request_ctx.resource}")
                matched_rules.append("egress:blocked_domain")
        
        # If no specific rules matched, allow by default
        if not reasons:
            reasons.append("No policy violations found")
            
        # In monitor/dry_run mode, log but don't enforce denies
        if mode in [PolicyMode.MONITOR, PolicyMode.DRY_RUN] and decision == PolicyDecision.DENY:
            reasons.append(f"[{mode.value}] Would deny but mode is {mode.value}")
            if mode == PolicyMode.DRY_RUN:
                decision = PolicyDecision.ALLOW  # Override deny in dry_run
        
        result = PolicyEvalResult(
            decision=decision,
            reasons=reasons,
            mode=mode,
            policy_ref=f"policy:v{self.policies.get('version', '1.0')}",
            matched_rules=matched_rules
        )
        
        # Cache result
        self._cache[cache_key] = result
        
        return result
    
    def _get_cache_key(self, ctx: RequestContext) -> str:
        """Generate cache key for request context."""
        key_parts = [
            ctx.tenant_id,
            ctx.service,
            ctx.capability,
            ctx.domain,
            str(ctx.budget_used)
        ]
        return hashlib.md5(":".join(key_parts).encode()).hexdigest()
    
    def reload_policies(self):
        """Reload policies from file and clear cache."""
        self.policies = self._load_policies()
        self._cache.clear()
    
    def get_policy_version(self) -> str:
        """Get current policy version."""
        return self.policies.get("version", "unknown")
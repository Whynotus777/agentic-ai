"""
Integrated Router with policy, flags, and capability integration.
Routes requests to appropriate models based on policies and feature flags.
"""
import yaml
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import hashlib

from control_plane.policy_engine.engine import PolicyEngine, RequestContext, PolicyDecision
from control_plane.flags.store import FeatureFlagStore
from control_plane.cap_registry.registry import CapabilityRegistry


@dataclass
class RoutingRequest:
    """Request for routing decision."""
    tenant_id: str
    service: str
    domain: str
    capability: str
    user_id: Optional[str] = None
    environment: str = "production"
    tier: str = "standard"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RoutingDecision:
    """Routing decision with trace information."""
    chosen_primary: str
    fallbacks: List[str]
    policy_match: Dict[str, Any]
    flag_overrides: Dict[str, Any]
    guardrails: List[Dict[str, Any]]
    requires_hitl: bool
    decision_trace: Dict[str, Any]
    timestamp: str
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


class IntegratedRouter:
    """
    Integrated router combining policies, flags, and capabilities.
    """
    
    def __init__(self,
                 router_policy_path: str = "control_plane/router_policy.yaml",
                 policy_engine_path: str = "control_plane/policy_engine/policy.yaml",
                 flags_path: str = "control_plane/flags/flags.yaml",
                 capabilities_path: str = "control_plane/cap_registry/capabilities.yaml"):
        
        # Load router policy
        self.router_policy_path = Path(router_policy_path)
        self.router_policy = self._load_router_policy()
        
        # Initialize components
        self.policy_engine = PolicyEngine(policy_engine_path)
        self.flag_store = FeatureFlagStore(flags_path)
        self.capability_registry = CapabilityRegistry(capabilities_path)
        
        # Cache for routing decisions
        self._routing_cache = {}
        
    def _load_router_policy(self) -> Dict:
        """Load router policy configuration."""
        if not self.router_policy_path.exists():
            return self._get_default_router_policy()
            
        with open(self.router_policy_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_default_router_policy(self) -> Dict:
        """Default router policy if file doesn't exist."""
        return {
            "defaults": {
                "primary_model": "o4-mini",
                "timeout_seconds": 30,
                "retry_attempts": 3,
                "fallback_enabled": True
            },
            "domain_policies": {},
            "capability_routing": {},
            "fallback_policies": {},
            "guardrails": {}
        }
    
    def route(self, request: RoutingRequest) -> RoutingDecision:
        """
        Make routing decision based on policies, flags, and capabilities.
        
        Returns RoutingDecision with chosen model and trace information.
        """
        timestamp = datetime.utcnow().isoformat()
        
        # 1. Check policy engine
        policy_ctx = RequestContext(
            tenant_id=request.tenant_id,
            service=request.service,
            capability=request.capability,
            domain=request.domain,
            user_id=request.user_id
        )
        
        policy_result = self.policy_engine.evaluate(policy_ctx)
        
        # Early return if policy denies
        if policy_result.decision == PolicyDecision.DENY:
            return RoutingDecision(
                chosen_primary="none",
                fallbacks=[],
                policy_match={
                    "decision": policy_result.decision.value,
                    "reasons": policy_result.reasons
                },
                flag_overrides={},
                guardrails=[],
                requires_hitl=False,
                decision_trace={
                    "blocked_by_policy": True,
                    "policy_ref": policy_result.policy_ref
                },
                timestamp=timestamp
            )
        
        # 2. Check feature flags
        flag_context = {
            "tenant": request.tenant_id,
            "tenant_id": request.tenant_id,
            "service": request.service,
            "environment": request.environment,
            "user_id": request.user_id
        }
        
        flags = self.flag_store.evaluate_all(flag_context)
        
        # 3. Validate capabilities
        cap_validation = self.capability_registry.validate_agent_capabilities(
            request.service,
            [request.capability]
        )
        
        if request.capability in cap_validation["missing"]:
            return RoutingDecision(
                chosen_primary="none",
                fallbacks=[],
                policy_match={"decision": "deny", "reasons": ["Capability not registered"]},
                flag_overrides={},
                guardrails=[],
                requires_hitl=False,
                decision_trace={
                    "blocked_by_capability": True,
                    "missing_capability": request.capability
                },
                timestamp=timestamp
            )
        
        # 4. Determine primary model
        primary_model = self._select_primary_model(request, flags, policy_result)
        
        # 5. Determine fallbacks
        fallbacks = self._select_fallbacks(request, primary_model)
        
        # 6. Apply guardrails
        guardrails = self._get_guardrails(request)
        
        # 7. Check HITL requirement
        requires_hitl = (
            policy_result.decision == PolicyDecision.HITL_REQUIRED or
            self._domain_requires_hitl(request.domain) or
            self._capability_requires_hitl(request.capability)
        )
        
        # 8. Build decision trace
        decision_trace = {
            "request_id": self._generate_request_id(request),
            "domain": request.domain,
            "capability": request.capability,
            "tier": request.tier,
            "policy_evaluation": {
                "decision": policy_result.decision.value,
                "mode": policy_result.mode.value,
                "matched_rules": policy_result.matched_rules
            },
            "flag_evaluation": {
                "use_deepseek_v3": flags.get("use_deepseek_v3", False),
                "enable_rt2_robotics": flags.get("enable_rt2_robotics", False),
                "appdev_tier_routing": flags.get("appdev_tier_routing", False)
            },
            "capability_validation": cap_validation,
            "model_selection": {
                "primary": primary_model,
                "fallbacks": fallbacks,
                "selection_reason": self._get_selection_reason(request, flags)
            },
            "guardrails_applied": len(guardrails),
            "hitl_required": requires_hitl
        }
        
        return RoutingDecision(
            chosen_primary=primary_model,
            fallbacks=fallbacks,
            policy_match={
                "decision": policy_result.decision.value,
                "reasons": policy_result.reasons,
                "mode": policy_result.mode.value
            },
            flag_overrides={k: v for k, v in flags.items() if v},
            guardrails=guardrails,
            requires_hitl=requires_hitl,
            decision_trace=decision_trace,
            timestamp=timestamp
        )
    
    def _select_primary_model(self, 
                            request: RoutingRequest,
                            flags: Dict[str, bool],
                            policy_result: Any) -> str:
        """Select primary model based on domain, flags, and policies."""
        
        # Check feature flag overrides first
        if flags.get("use_deepseek_v3") and request.service == "orchestrator":
            return "deepseek-v3"
        
        # Domain-specific routing
        domain_policy = self.router_policy.get("domain_policies", {}).get(request.domain, {})
        
        if request.domain == "robotics" and flags.get("enable_rt2_robotics"):
            return "rt-2"
        
        if request.domain == "simulation" and flags.get("gemini_25_simulation"):
            return "gemini-2.5-pro"
        
        if request.domain == "appdev" and flags.get("appdev_tier_routing"):
            # Tier-based routing for appdev
            tiers = domain_policy.get("tiers", {})
            tier_config = tiers.get(request.tier, tiers.get("standard", {}))
            return tier_config.get("primary_model", "o4-mini")
        
        # Use domain policy if available
        if "primary_model" in domain_policy:
            return domain_policy["primary_model"]
        
        # Capability-based routing
        cap_routing = self.router_policy.get("capability_routing", {}).get(request.capability, {})
        if "preferred_models" in cap_routing and cap_routing["preferred_models"]:
            return cap_routing["preferred_models"][0]
        
        # Default
        return self.router_policy.get("defaults", {}).get("primary_model", "o4-mini")
    
    def _select_fallbacks(self, request: RoutingRequest, primary: str) -> List[str]:
        """Select fallback models."""
        domain_policy = self.router_policy.get("domain_policies", {}).get(request.domain, {})
        
        # Domain-specific fallbacks
        if "secondary_models" in domain_policy:
            return [m for m in domain_policy["secondary_models"] if m != primary]
        
        # Tier-based fallbacks for appdev
        if request.domain == "appdev" and "tiers" in domain_policy:
            tier_config = domain_policy["tiers"].get(request.tier, {})
            if "secondary_models" in tier_config:
                return [m for m in tier_config["secondary_models"] if m != primary]
        
        # Default fallbacks based on primary
        default_fallbacks = {
            "o4": ["o4-mini", "gpt-4"],
            "o4-mini": ["gpt-4", "claude-3"],
            "deepseek-v3": ["o4-mini", "gpt-4"],
            "rt-2": ["o4", "gemini-2.5-pro"],
            "gemini-2.5-pro": ["gemini-2-flash", "claude-3"]
        }
        
        return default_fallbacks.get(primary, ["gpt-4"])
    
    def _get_guardrails(self, request: RoutingRequest) -> List[Dict[str, Any]]:
        """Get applicable guardrails for request."""
        guardrails = []
        
        # Global guardrails
        global_guardrails = self.router_policy.get("guardrails", {}).get("global", [])
        guardrails.extend(global_guardrails)
        
        # Domain-specific guardrails
        domain_policy = self.router_policy.get("domain_policies", {}).get(request.domain, {})
        if "guardrails" in domain_policy:
            guardrails.extend(domain_policy["guardrails"])
        
        # Robotics-specific guardrails
        if request.domain == "robotics":
            robotics_guardrails = self.router_policy.get("guardrails", {}).get("robotics_specific", [])
            guardrails.extend(robotics_guardrails)
        
        return guardrails
    
    def _domain_requires_hitl(self, domain: str) -> bool:
        """Check if domain requires HITL."""
        domain_policy = self.router_policy.get("domain_policies", {}).get(domain, {})
        requirements = domain_policy.get("requirements", {})
        return requirements.get("hitl_gate", False)
    
    def _capability_requires_hitl(self, capability: str) -> bool:
        """Check if capability requires HITL."""
        cap_routing = self.router_policy.get("capability_routing", {}).get(capability, {})
        return cap_routing.get("require_hitl", False)
    
    def _get_selection_reason(self, request: RoutingRequest, flags: Dict[str, bool]) -> str:
        """Get human-readable reason for model selection."""
        if flags.get("use_deepseek_v3") and request.service == "orchestrator":
            return "Feature flag: use_deepseek_v3 enabled for orchestrator"
        
        if request.domain == "robotics" and flags.get("enable_rt2_robotics"):
            return "Domain policy: RT-2 for robotics with flag enabled"
        
        if request.domain == "simulation":
            return "Domain policy: Gemini 2.5 Pro for simulation"
        
        if request.domain == "appdev":
            return f"Tier-based routing: {request.tier} tier for appdev"
        
        return "Default routing policy"
    
    def _generate_request_id(self, request: RoutingRequest) -> str:
        """Generate unique request ID."""
        data = f"{request.tenant_id}:{request.service}:{request.domain}:{request.capability}:{datetime.utcnow().isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get router metrics."""
        return {
            "cache_size": len(self._routing_cache),
            "policy_version": self.policy_engine.get_policy_version(),
            "active_flags": self.flag_store.get_metrics()["enabled_flags"],
            "registered_capabilities": self.capability_registry.export_metrics()["total_capabilities"]
        }
    
    def reload_all(self):
        """Reload all configurations."""
        self.router_policy = self._load_router_policy()
        self.policy_engine.reload_policies()
        self.flag_store.reload_flags()
        self._routing_cache.clear()
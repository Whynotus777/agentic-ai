# control_plane/integrated_router.py
"""
Integrated intelligent router with telemetry-driven auto-tuning,
capability scoring, HITL integration, and domain-specific routing policies.
This connects all components together.
"""

import asyncio
import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict, deque

from opentelemetry import trace, metrics
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)


class ModelTier(Enum):
    """Model tiers for routing"""
    TIER_1_FRONTIER = "tier_1_frontier"  # GPT-5, Claude-4.1, Gemini-2.5
    TIER_2_BALANCED = "tier_2_balanced"  # GPT-4o, Claude-Sonnet, Gemini-Pro
    TIER_3_FAST = "tier_3_fast"  # GPT-4o-mini, Claude-Haiku, Gemini-Flash
    SPECIALIZED_SLM = "specialized_slm"  # Domain-specific small models


@dataclass
class ModelCapability:
    """Model capability definition with performance metrics"""
    model_id: str
    tier: ModelTier
    domains: List[str]
    capabilities: List[str]
    avg_latency_ms: float
    p95_latency_ms: float
    cost_per_1k_tokens: float
    success_rate: float
    context_window: int
    supports_tools: bool
    supports_vision: bool
    supports_streaming: bool
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RoutingPolicy:
    """Routing policy configuration"""
    policy_id: str
    name: str
    domain: str
    rules: List[Dict[str, Any]]
    tier_preferences: List[ModelTier]
    fallback_chain: List[str]
    cost_threshold: float
    latency_slo_ms: float
    require_hitl_approval: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Result of routing decision"""
    request_id: str
    selected_model: str
    reasoning: str
    estimated_latency_ms: float
    estimated_cost: float
    confidence_score: float
    fallback_models: List[str]
    hitl_required: bool
    telemetry_factors: Dict[str, float]


class IntegratedRouter:
    """
    Production-grade router with telemetry integration and auto-tuning
    """
    
    def __init__(self, config_path: str = "config/router.yaml"):
        self.config = self._load_config(config_path)
        self.model_registry = self._init_model_registry()
        self.routing_policies = self._init_routing_policies()
        self.telemetry_buffer = deque(maxlen=10000)
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.routing_weights: Dict[str, float] = {}
        self.capability_scores: Dict[str, Dict[str, float]] = {}
        self.hitl_queue: Dict[str, RoutingDecision] = {}
        self._init_metrics()
        
        # Start auto-tuning loop
        asyncio.create_task(self._auto_tuning_loop())
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load router configuration from YAML"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Return default config if file doesn't exist
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default router configuration"""
        return {
            "model_tiers": {
                "tier_1_frontier": {
                    "models": ["gpt-5", "claude-4.1-opus", "gemini-2.5-ultra"],
                    "cost_multiplier": 10.0,
                    "quality_score": 0.95
                },
                "tier_2_balanced": {
                    "models": ["gpt-4o", "claude-4.1-sonnet", "gemini-2.5-pro"],
                    "cost_multiplier": 5.0,
                    "quality_score": 0.85
                },
                "tier_3_fast": {
                    "models": ["gpt-4o-mini", "claude-4.1-haiku", "gemini-2.5-flash"],
                    "cost_multiplier": 1.0,
                    "quality_score": 0.75
                }
            },
            "domain_models": {
                "code": ["deepseek-coder-6.7b", "yi-coder-9b", "codestral-22b"],
                "robotics": ["rt-2", "pi-0-small"],
                "bioengineering": ["esm3", "alphafold-3"],
                "simulation": ["minerva", "galactica"]
            },
            "routing_rules": {
                "cost_optimization": {
                    "enabled": True,
                    "max_cost_per_request": 1.0
                },
                "latency_optimization": {
                    "enabled": True,
                    "max_latency_ms": 1000
                },
                "quality_threshold": {
                    "min_success_rate": 0.9
                }
            },
            "telemetry": {
                "sampling_rate": 0.1,
                "error_sampling_rate": 1.0,
                "auto_tuning_interval": 60,
                "performance_window": 300
            }
        }
    
    def _init_model_registry(self) -> Dict[str, ModelCapability]:
        """Initialize model capability registry"""
        registry = {}
        
        # Tier 1 - Frontier Models (Sep 2025 capabilities)
        frontier_models = [
            ModelCapability(
                model_id="gpt-5",
                tier=ModelTier.TIER_1_FRONTIER,
                domains=["general", "reasoning", "code", "creative"],
                capabilities=["chain_of_thought", "tool_use", "vision", "long_context"],
                avg_latency_ms=150,
                p95_latency_ms=300,
                cost_per_1k_tokens=0.10,
                success_rate=0.98,
                context_window=200000,
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True
            ),
            ModelCapability(
                model_id="claude-4.1-opus",
                tier=ModelTier.TIER_1_FRONTIER,
                domains=["general", "analysis", "code", "safety"],
                capabilities=["constitutional_ai", "long_form", "technical"],
                avg_latency_ms=140,
                p95_latency_ms=280,
                cost_per_1k_tokens=0.09,
                success_rate=0.97,
                context_window=200000,
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True
            ),
            ModelCapability(
                model_id="gemini-2.5-ultra",
                tier=ModelTier.TIER_1_FRONTIER,
                domains=["general", "multimodal", "search", "creative"],
                capabilities=["multimodal", "grounding", "long_context"],
                avg_latency_ms=160,
                p95_latency_ms=320,
                cost_per_1k_tokens=0.08,
                success_rate=0.96,
                context_window=2000000,
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True
            )
        ]
        
        # Tier 2 - Balanced Models
        balanced_models = [
            ModelCapability(
                model_id="gpt-4o",
                tier=ModelTier.TIER_2_BALANCED,
                domains=["general", "code", "analysis"],
                capabilities=["reasoning", "tool_use", "structured_output"],
                avg_latency_ms=80,
                p95_latency_ms=150,
                cost_per_1k_tokens=0.03,
                success_rate=0.94,
                context_window=128000,
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True
            ),
            ModelCapability(
                model_id="claude-4.1-sonnet",
                tier=ModelTier.TIER_2_BALANCED,
                domains=["general", "writing", "code"],
                capabilities=["balanced", "efficient", "safe"],
                avg_latency_ms=70,
                p95_latency_ms=140,
                cost_per_1k_tokens=0.025,
                success_rate=0.93,
                context_window=200000,
                supports_tools=True,
                supports_vision=True,
                supports_streaming=True
            )
        ]
        
        # Tier 3 - Fast Models
        fast_models = [
            ModelCapability(
                model_id="gpt-4o-mini",
                tier=ModelTier.TIER_3_FAST,
                domains=["general", "simple_tasks"],
                capabilities=["fast", "efficient"],
                avg_latency_ms=30,
                p95_latency_ms=60,
                cost_per_1k_tokens=0.002,
                success_rate=0.90,
                context_window=128000,
                supports_tools=True,
                supports_vision=False,
                supports_streaming=True
            ),
            ModelCapability(
                model_id="claude-4.1-haiku",
                tier=ModelTier.TIER_3_FAST,
                domains=["general", "classification"],
                capabilities=["ultrafast", "lightweight"],
                avg_latency_ms=25,
                p95_latency_ms=50,
                cost_per_1k_tokens=0.0015,
                success_rate=0.89,
                context_window=200000,
                supports_tools=False,
                supports_vision=False,
                supports_streaming=True
            )
        ]
        
        # Specialized SLMs
        specialized_models = [
            ModelCapability(
                model_id="deepseek-coder-6.7b",
                tier=ModelTier.SPECIALIZED_SLM,
                domains=["code"],
                capabilities=["code_generation", "code_completion"],
                avg_latency_ms=40,
                p95_latency_ms=80,
                cost_per_1k_tokens=0.001,
                success_rate=0.92,
                context_window=16000,
                supports_tools=False,
                supports_vision=False,
                supports_streaming=True
            ),
            ModelCapability(
                model_id="yi-coder-9b",
                tier=ModelTier.SPECIALIZED_SLM,
                domains=["code"],
                capabilities=["code_understanding", "refactoring"],
                avg_latency_ms=45,
                p95_latency_ms=90,
                cost_per_1k_tokens=0.0012,
                success_rate=0.91,
                context_window=32000,
                supports_tools=False,
                supports_vision=False,
                supports_streaming=True
            ),
            ModelCapability(
                model_id="rt-2",
                tier=ModelTier.SPECIALIZED_SLM,
                domains=["robotics"],
                capabilities=["robot_control", "path_planning"],
                avg_latency_ms=20,
                p95_latency_ms=40,
                cost_per_1k_tokens=0.005,
                success_rate=0.94,
                context_window=4096,
                supports_tools=True,
                supports_vision=True,
                supports_streaming=False
            ),
            ModelCapability(
                model_id="pi-0-small",
                tier=ModelTier.SPECIALIZED_SLM,
                domains=["robotics", "vision"],
                capabilities=["scene_understanding", "object_detection"],
                avg_latency_ms=15,
                p95_latency_ms=30,
                cost_per_1k_tokens=0.003,
                success_rate=0.93,
                context_window=2048,
                supports_tools=False,
                supports_vision=True,
                supports_streaming=False
            ),
            ModelCapability(
                model_id="esm3",
                tier=ModelTier.SPECIALIZED_SLM,
                domains=["bioengineering"],
                capabilities=["protein_folding", "sequence_analysis"],
                avg_latency_ms=100,
                p95_latency_ms=200,
                cost_per_1k_tokens=0.02,
                success_rate=0.88,
                context_window=8192,
                supports_tools=False,
                supports_vision=False,
                supports_streaming=False
            )
        ]
        
        # Add all models to registry
        for model in frontier_models + balanced_models + fast_models + specialized_models:
            registry[model.model_id] = model
        
        return registry
    
    def _init_routing_policies(self) -> Dict[str, RoutingPolicy]:
        """Initialize routing policies"""
        policies = {}
        
        # General purpose policy
        policies["general"] = RoutingPolicy(
            policy_id="policy-general",
            name="General Purpose Routing",
            domain="general",
            rules=[
                {"if": "complexity == 'high'", "then": "use_tier_1"},
                {"if": "cost_sensitive == true", "then": "use_tier_3"},
                {"if": "latency_critical == true", "then": "use_tier_3"}
            ],
            tier_preferences=[ModelTier.TIER_2_BALANCED, ModelTier.TIER_1_FRONTIER],
            fallback_chain=["gpt-4o", "claude-4.1-sonnet", "gpt-4o-mini"],
            cost_threshold=1.0,
            latency_slo_ms=1000
        )
        
        # Code generation policy
        policies["code"] = RoutingPolicy(
            policy_id="policy-code",
            name="Code Generation Routing",
            domain="code",
            rules=[
                {"if": "task == 'generation'", "then": "use_deepseek"},
                {"if": "task == 'review'", "then": "use_tier_1"},
                {"if": "task == 'refactor'", "then": "use_yi_coder"}
            ],
            tier_preferences=[ModelTier.SPECIALIZED_SLM, ModelTier.TIER_1_FRONTIER],
            fallback_chain=["deepseek-coder-6.7b", "yi-coder-9b", "gpt-5"],
            cost_threshold=0.5,
            latency_slo_ms=500
        )
        
        # Robotics policy
        policies["robotics"] = RoutingPolicy(
            policy_id="policy-robotics",
            name="Robotics Control Routing",
            domain="robotics",
            rules=[
                {"if": "task == 'planning'", "then": "use_rt2"},
                {"if": "task == 'vision'", "then": "use_pi0"},
                {"if": "safety_critical == true", "then": "require_hitl"}
            ],
            tier_preferences=[ModelTier.SPECIALIZED_SLM],
            fallback_chain=["rt-2", "pi-0-small"],
            cost_threshold=0.1,
            latency_slo_ms=50,
            require_hitl_approval=True
        )
        
        # Bioengineering policy
        policies["bioengineering"] = RoutingPolicy(
            policy_id="policy-bio",
            name="Bioengineering Routing",
            domain="bioengineering",
            rules=[
                {"if": "task == 'protein_analysis'", "then": "use_esm3"},
                {"if": "task == 'general'", "then": "use_tier_1"}
            ],
            tier_preferences=[ModelTier.SPECIALIZED_SLM, ModelTier.TIER_1_FRONTIER],
            fallback_chain=["esm3", "gpt-5"],
            cost_threshold=5.0,
            latency_slo_ms=5000
        )
        
        return policies
    
    def _init_metrics(self):
        """Initialize OpenTelemetry metrics"""
        self.routing_counter = meter.create_counter(
            "routing_decisions",
            description="Number of routing decisions made"
        )
        
        self.latency_histogram = meter.create_histogram(
            "routing_latency",
            description="Routing decision latency",
            unit="ms"
        )
        
        self.cost_histogram = meter.create_histogram(
            "routing_cost",
            description="Estimated cost per request",
            unit="usd"
        )
        
        self.success_rate_gauge = meter.create_gauge(
            "model_success_rate",
            description="Model success rate"
        )
    
    @tracer.start_as_current_span("route_request")
    async def route_request(
        self,
        request_type: str,
        domain: str,
        context: Dict[str, Any]
    ) -> RoutingDecision:
        """
        Route a request to the optimal model based on telemetry and policies
        """
        span = trace.get_current_span()
        request_id = context.get("request_id", str(uuid.uuid4()))
        
        # Get relevant policy
        policy = self.routing_policies.get(domain, self.routing_policies["general"])
        
        # Calculate capability scores for all models
        capability_scores = await self._calculate_capability_scores(
            request_type,
            domain,
            context
        )
        
        # Apply routing rules
        selected_model = await self._apply_routing_rules(
            policy,
            capability_scores,
            context
        )
        
        # Get telemetry-based adjustments
        telemetry_weights = await self._get_telemetry_weights(selected_model)
        
        # Check if HITL approval required
        hitl_required = await self._check_hitl_requirement(
            policy,
            selected_model,
            context
        )
        
        # Build fallback chain
        fallback_models = self._build_fallback_chain(
            policy,
            selected_model,
            capability_scores
        )
        
        # Create routing decision
        model_cap = self.model_registry[selected_model]
        decision = RoutingDecision(
            request_id=request_id,
            selected_model=selected_model,
            reasoning=self._generate_reasoning(selected_model, capability_scores),
            estimated_latency_ms=model_cap.avg_latency_ms * telemetry_weights.get("latency_factor", 1.0),
            estimated_cost=self._estimate_cost(selected_model, context),
            confidence_score=capability_scores[selected_model]["total"],
            fallback_models=fallback_models,
            hitl_required=hitl_required,
            telemetry_factors=telemetry_weights
        )
        
        # Queue for HITL if required
        if hitl_required:
            self.hitl_queue[request_id] = decision
        
        # Record telemetry
        await self._record_routing_telemetry(decision)
        
        # Update metrics
        self.routing_counter.add(1, attributes={
            "domain": domain,
            "model": selected_model,
            "tier": model_cap.tier.value
        })
        
        span.set_attributes({
            "request_id": request_id,
            "domain": domain,
            "selected_model": selected_model,
            "estimated_cost": decision.estimated_cost,
            "hitl_required": hitl_required
        })
        
        return decision
    
    async def _calculate_capability_scores(
        self,
        request_type: str,
        domain: str,
        context: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate capability scores for all models"""
        scores = {}
        
        for model_id, capability in self.model_registry.items():
            # Domain match score
            domain_score = 1.0 if domain in capability.domains else 0.5
            
            # Performance score based on telemetry
            perf_data = self._get_performance_data(model_id)
            success_score = perf_data.get("success_rate", capability.success_rate)
            
            # Latency score (inverse - lower is better)
            latency_target = context.get("latency_slo_ms", 1000)
            latency_score = min(1.0, latency_target / max(capability.avg_latency_ms, 1))
            
            # Cost score (inverse - lower is better)
            cost_budget = context.get("cost_budget", 1.0)
            cost_score = min(1.0, cost_budget / max(capability.cost_per_1k_tokens * 10, 0.01))
            
            # Feature match score
            feature_score = 1.0
            if context.get("requires_tools") and not capability.supports_tools:
                feature_score *= 0.5
            if context.get("requires_vision") and not capability.supports_vision:
                feature_score *= 0.5
            if context.get("requires_streaming") and not capability.supports_streaming:
                feature_score *= 0.8
            
            # Context window score
            required_context = context.get("context_size", 4000)
            context_score = 1.0 if capability.context_window >= required_context else 0.5
            
            # Weighted total score
            weights = {
                "domain": 0.25,
                "success": 0.25,
                "latency": 0.20,
                "cost": 0.15,
                "features": 0.10,
                "context": 0.05
            }
            
            total_score = (
                domain_score * weights["domain"] +
                success_score * weights["success"] +
                latency_score * weights["latency"] +
                cost_score * weights["cost"] +
                feature_score * weights["features"] +
                context_score * weights["context"]
            )
            
            scores[model_id] = {
                "domain": domain_score,
                "success": success_score,
                "latency": latency_score,
                "cost": cost_score,
                "features": feature_score,
                "context": context_score,
                "total": total_score
            }
        
        # Store for analysis
        self.capability_scores[context.get("request_id", "")] = scores
        
        return scores
    
    def _get_performance_data(self, model_id: str) -> Dict[str, float]:
        """Get performance data from telemetry"""
        if model_id not in self.performance_history:
            return {}
        
        recent_data = list(self.performance_history[model_id])
        if not recent_data:
            return {}
        
        # Calculate aggregated metrics
        success_count = sum(1 for d in recent_data if d.get("success", False))
        success_rate = success_count / len(recent_data) if recent_data else 0
        
        latencies = [d.get("latency_ms", 0) for d in recent_data if d.get("latency_ms")]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        costs = [d.get("cost", 0) for d in recent_data if d.get("cost")]
        avg_cost = sum(costs) / len(costs) if costs else 0
        
        return {
            "success_rate": success_rate,
            "avg_latency_ms": avg_latency,
            "avg_cost": avg_cost,
            "sample_count": len(recent_data)
        }
    
    async def _apply_routing_rules(
        self,
        policy: RoutingPolicy,
        scores: Dict[str, Dict[str, float]],
        context: Dict[str, Any]
    ) -> str:
        """Apply routing rules to select model"""
        # Filter models by tier preference
        tier_models = []
        for tier in policy.tier_preferences:
            tier_models.extend([
                model_id for model_id, cap in self.model_registry.items()
                if cap.tier == tier
            ])
        
        # Sort by total score
        sorted_models = sorted(
            tier_models,
            key=lambda m: scores.get(m, {}).get("total", 0),
            reverse=True
        )
        
        # Apply policy rules
        for rule in policy.rules:
            condition = rule.get("if", "")
            action = rule.get("then", "")
            
            # Simple rule evaluation (in production, use proper rule engine)
            if self._evaluate_rule_condition(condition, context):
                if "use_tier_1" in action:
                    for model in sorted_models:
                        if self.model_registry[model].tier == ModelTier.TIER_1_FRONTIER:
                            return model
                elif "use_tier_3" in action:
                    for model in sorted_models:
                        if self.model_registry[model].tier == ModelTier.TIER_3_FAST:
                            return model
                elif "use_deepseek" in action:
                    if "deepseek-coder-6.7b" in sorted_models:
                        return "deepseek-coder-6.7b"
                elif "use_rt2" in action:
                    if "rt-2" in sorted_models:
                        return "rt-2"
        
        # Return highest scoring model
        return sorted_models[0] if sorted_models else "gpt-4o"
    
    def _evaluate_rule_condition(
        self,
        condition: str,
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate a routing rule condition"""
        # Simple evaluation - in production use proper expression evaluator
        if "==" in condition:
            parts = condition.split("==")
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip().strip("'\"")
                
                if key in context:
                    return str(context[key]) == value
        
        return False
    
    async def _get_telemetry_weights(
        self,
        model_id: str
    ) -> Dict[str, float]:
        """Get telemetry-based weight adjustments"""
        perf_data = self._get_performance_data(model_id)
        
        weights = {}
        
        # Adjust based on recent performance
        if perf_data.get("success_rate", 1.0) < 0.8:
            weights["success_penalty"] = 0.8
        else:
            weights["success_penalty"] = 1.0
        
        # Adjust for latency variance
        if model_id in self.performance_history:
            recent = list(self.performance_history[model_id])
            if recent:
                latencies = [d.get("latency_ms", 0) for d in recent[-10:]]
                if latencies:
                    variance = np.var(latencies)
                    if variance > 1000:  # High variance
                        weights["latency_factor"] = 1.2
                    else:
                        weights["latency_factor"] = 1.0
        
        return weights
    
    async def _check_hitl_requirement(
        self,
        policy: RoutingPolicy,
        model_id: str,
        context: Dict[str, Any]
    ) -> bool:
        """Check if HITL approval is required"""
        # Policy-based requirement
        if policy.require_hitl_approval:
            return True
        
        # Context-based requirement
        if context.get("safety_critical", False):
            return True
        
        if context.get("production_change", False):
            return True
        
        # Cost-based requirement
        estimated_cost = self._estimate_cost(model_id, context)
        if estimated_cost > policy.cost_threshold:
            return True
        
        return False
    
    def _build_fallback_chain(
        self,
        policy: RoutingPolicy,
        primary_model: str,
        scores: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Build fallback chain of models"""
        fallback = []
        
        # Add policy-defined fallbacks
        for model in policy.fallback_chain:
            if model != primary_model and model in self.model_registry:
                fallback.append(model)
        
        # Add high-scoring alternatives
        sorted_by_score = sorted(
            scores.items(),
            key=lambda x: x[1]["total"],
            reverse=True
        )
        
        for model_id, _ in sorted_by_score[:5]:
            if model_id != primary_model and model_id not in fallback:
                fallback.append(model_id)
        
        return fallback[:3]  # Limit to 3 fallbacks
    
    def _estimate_cost(
        self,
        model_id: str,
        context: Dict[str, Any]
    ) -> float:
        """Estimate cost for request"""
        if model_id not in self.model_registry:
            return 0.0
        
        capability = self.model_registry[model_id]
        
        # Estimate token count
        estimated_tokens = context.get("estimated_tokens", 1000)
        
        # Calculate cost
        cost = (estimated_tokens / 1000) * capability.cost_per_1k_tokens
        
        # Add multiplier for complex tasks
        if context.get("complexity") == "high":
            cost *= 1.5
        
        return cost
    
    def _generate_reasoning(
        self,
        model_id: str,
        scores: Dict[str, Dict[str, float]]
    ) -> str:
        """Generate reasoning for model selection"""
        model_scores = scores.get(model_id, {})
        capability = self.model_registry.get(model_id)
        
        reasons = []
        
        if model_scores.get("domain", 0) > 0.8:
            reasons.append("Strong domain match")
        
        if model_scores.get("success", 0) > 0.9:
            reasons.append("High success rate")
        
        if model_scores.get("latency", 0) > 0.8:
            reasons.append("Meets latency requirements")
        
        if model_scores.get("cost", 0) > 0.7:
            reasons.append("Cost effective")
        
        if capability:
            reasons.append(f"Tier: {capability.tier.value}")
        
        return "; ".join(reasons) if reasons else "Default selection"
    
    async def _record_routing_telemetry(
        self,
        decision: RoutingDecision
    ):
        """Record routing decision telemetry"""
        telemetry_data = {
            "timestamp": datetime.utcnow(),
            "request_id": decision.request_id,
            "model": decision.selected_model,
            "estimated_cost": decision.estimated_cost,
            "estimated_latency": decision.estimated_latency_ms,
            "confidence": decision.confidence_score,
            "hitl_required": decision.hitl_required
        }
        
        self.telemetry_buffer.append(telemetry_data)
    
    async def record_execution_result(
        self,
        request_id: str,
        model_id: str,
        success: bool,
        actual_latency_ms: float,
        actual_cost: float
    ):
        """Record actual execution results for learning"""
        result = {
            "timestamp": datetime.utcnow(),
            "request_id": request_id,
            "success": success,
            "latency_ms": actual_latency_ms,
            "cost": actual_cost
        }
        
        self.performance_history[model_id].append(result)
        
        # Update success rate metric
        perf_data = self._get_performance_data(model_id)
        self.success_rate_gauge.set(
            perf_data.get("success_rate", 0),
            attributes={"model": model_id}
        )
    
    async def _auto_tuning_loop(self):
        """Continuous auto-tuning based on telemetry"""
        while True:
            try:
                await asyncio.sleep(self.config.get("telemetry", {}).get("auto_tuning_interval", 60))
                
                # Analyze performance for each model
                for model_id in self.model_registry:
                    perf_data = self._get_performance_data(model_id)
                    
                    if perf_data.get("sample_count", 0) < 10:
                        continue
                    
                    # Update routing weights based on performance
                    success_rate = perf_data.get("success_rate", 1.0)
                    
                    if model_id not in self.routing_weights:
                        self.routing_weights[model_id] = 1.0
                    
                    # Adjust weight based on success rate
                    if success_rate < 0.8:
                        self.routing_weights[model_id] *= 0.9
                    elif success_rate > 0.95:
                        self.routing_weights[model_id] *= 1.1
                    
                    # Keep weights in reasonable range
                    self.routing_weights[model_id] = max(0.1, min(2.0, self.routing_weights[model_id]))
                
                print(f"Auto-tuning complete. Updated weights: {self.routing_weights}")
                
            except Exception as e:
                print(f"Auto-tuning error: {e}")
    
    async def approve_hitl_request(
        self,
        request_id: str
    ) -> bool:
        """Approve HITL request"""
        if request_id in self.hitl_queue:
            decision = self.hitl_queue.pop(request_id)
            decision.hitl_required = False
            return True
        return False
    
    async def get_routing_metrics(self) -> Dict[str, Any]:
        """Get routing metrics"""
        metrics = {
            "total_requests": len(self.telemetry_buffer),
            "models_registered": len(self.model_registry),
            "policies_active": len(self.routing_policies),
            "hitl_pending": len(self.hitl_queue),
            "routing_weights": dict(self.routing_weights),
            "model_performance": {}
        }
        
        for model_id in self.model_registry:
            perf = self._get_performance_data(model_id)
            if perf.get("sample_count", 0) > 0:
                metrics["model_performance"][model_id] = perf
        
        return metrics


# Example usage
async def main():
    import uuid
    
    # Initialize router
    router = IntegratedRouter()
    
    # Route a general request
    decision = await router.route_request(
        request_type="chat",
        domain="general",
        context={
            "request_id": str(uuid.uuid4()),
            "complexity": "high",
            "requires_tools": True,
            "latency_slo_ms": 200,
            "cost_budget": 0.5
        }
    )
    
    print(f"Selected model: {decision.selected_model}")
    print(f"Reasoning: {decision.reasoning}")
    print(f"Estimated cost: ${decision.estimated_cost:.4f}")
    print(f"Fallback chain: {decision.fallback_models}")
    
    # Route a code generation request
    code_decision = await router.route_request(
        request_type="code_generation",
        domain="code",
        context={
            "request_id": str(uuid.uuid4()),
            "task": "generation",
            "language": "python",
            "complexity": "medium"
        }
    )
    
    print(f"\nCode generation model: {code_decision.selected_model}")
    
    # Route a robotics request
    robotics_decision = await router.route_request(
        request_type="robot_control",
        domain="robotics",
        context={
            "request_id": str(uuid.uuid4()),
            "task": "planning",
            "safety_critical": True
        }
    )
    
    print(f"\nRobotics model: {robotics_decision.selected_model}")
    print(f"HITL required: {robotics_decision.hitl_required}")
    
    # Simulate execution and record results
    await router.record_execution_result(
        request_id=decision.request_id,
        model_id=decision.selected_model,
        success=True,
        actual_latency_ms=150,
        actual_cost=0.045
    )
    
    # Get metrics
    metrics = await router.get_routing_metrics()
    print(f"\nRouting metrics: {json.dumps(metrics, indent=2, default=str)}")


if __name__ == "__main__":
    asyncio.run(main())
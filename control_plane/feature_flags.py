# control_plane/feature_flags.py
"""
Complete feature flag system with progressive rollouts, A/B testing,
and capability registry for dynamic agent discovery.
"""

import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
import random
import aioredis

from opentelemetry import trace
tracer = trace.get_tracer(__name__)


class FeatureFlagType(Enum):
    """Types of feature flags"""
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    VARIANT = "variant"
    GRADUAL = "gradual"
    SCHEDULED = "scheduled"


class RolloutStrategy(Enum):
    """Rollout strategies"""
    ALL_USERS = "all_users"
    PERCENTAGE = "percentage"
    SPECIFIC_TENANTS = "specific_tenants"
    RING_BASED = "ring_based"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"


@dataclass
class FeatureFlag:
    """Feature flag configuration"""
    name: str
    description: str
    flag_type: FeatureFlagType
    enabled: bool
    rollout_strategy: RolloutStrategy
    rollout_percentage: float = 0.0
    allowed_tenants: List[str] = field(default_factory=list)
    blocked_tenants: List[str] = field(default_factory=list)
    variants: Dict[str, Any] = field(default_factory=dict)
    schedule: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    
@dataclass
class AgentCapability:
    """Agent capability definition"""
    capability_id: str
    name: str
    description: str
    required_models: List[str]
    required_resources: Dict[str, Any]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    cost_per_invocation: float = 0.0
    average_latency_ms: float = 0.0
    success_rate: float = 1.0
    
    
@dataclass
class RegisteredAgent:
    """Registered agent in the system"""
    agent_id: str
    name: str
    version: str
    capabilities: List[str]
    endpoint: str
    health_check: str
    status: str  # healthy, degraded, unavailable
    last_heartbeat: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureFlagManager:
    """
    Comprehensive feature flag management with progressive rollouts
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.flags: Dict[str, FeatureFlag] = {}
        self.flag_evaluations: Dict[str, int] = {}
        self.redis_client = None
        self._load_default_flags()
        
    def _load_default_flags(self):
        """Load default feature flags"""
        default_flags = [
            FeatureFlag(
                name="enable_robotics",
                description="Enable robotics capabilities",
                flag_type=FeatureFlagType.BOOLEAN,
                enabled=False,
                rollout_strategy=RolloutStrategy.ALL_USERS
            ),
            FeatureFlag(
                name="enable_bioengineering",
                description="Enable bioengineering models",
                flag_type=FeatureFlagType.BOOLEAN,
                enabled=False,
                rollout_strategy=RolloutStrategy.SPECIFIC_TENANTS,
                allowed_tenants=["research-labs", "pharma-corp"]
            ),
            FeatureFlag(
                name="enable_hitl_production",
                description="Require HITL approval in production",
                flag_type=FeatureFlagType.BOOLEAN,
                enabled=True,
                rollout_strategy=RolloutStrategy.ALL_USERS
            ),
            FeatureFlag(
                name="enable_auto_routing",
                description="Enable telemetry-based auto-routing",
                flag_type=FeatureFlagType.PERCENTAGE,
                enabled=True,
                rollout_strategy=RolloutStrategy.PERCENTAGE,
                rollout_percentage=50.0
            ),
            FeatureFlag(
                name="model_selection_strategy",
                description="Model selection strategy variant",
                flag_type=FeatureFlagType.VARIANT,
                enabled=True,
                rollout_strategy=RolloutStrategy.PERCENTAGE,
                variants={
                    "control": 40,
                    "cost_optimized": 30,
                    "quality_optimized": 30
                }
            ),
            FeatureFlag(
                name="advanced_caching",
                description="Gradual rollout of advanced caching",
                flag_type=FeatureFlagType.GRADUAL,
                enabled=True,
                rollout_strategy=RolloutStrategy.RING_BASED,
                rollout_percentage=0.0,
                schedule={
                    "start_date": datetime.utcnow().isoformat(),
                    "end_date": (datetime.utcnow() + timedelta(days=14)).isoformat(),
                    "daily_increment": 10.0
                }
            ),
            FeatureFlag(
                name="enable_slsa_verification",
                description="Enforce SLSA verification",
                flag_type=FeatureFlagType.BOOLEAN,
                enabled=True,
                rollout_strategy=RolloutStrategy.CANARY,
                rollout_percentage=10.0
            )
        ]
        
        for flag in default_flags:
            self.flags[flag.name] = flag
    
    async def initialize(self):
        """Initialize feature flag manager"""
        self.redis_client = await aioredis.create_redis_pool(
            self.config.get("redis_url", "redis://localhost")
        )
        
        # Load flags from persistent storage
        await self._load_flags_from_storage()
        
        # Start background tasks
        asyncio.create_task(self._gradual_rollout_updater())
        asyncio.create_task(self._metrics_collector())
    
    @tracer.start_as_current_span("evaluate_flag")
    async def evaluate(
        self,
        flag_name: str,
        context: Dict[str, Any]
    ) -> Any:
        """
        Evaluate a feature flag
        
        Args:
            flag_name: Name of the flag
            context: Evaluation context (tenant_id, user_id, etc.)
            
        Returns:
            Flag value (bool, string variant, or percentage)
        """
        span = trace.get_current_span()
        
        if flag_name not in self.flags:
            span.set_attribute("flag.found", False)
            return False
        
        flag = self.flags[flag_name]
        
        # Track evaluation
        self.flag_evaluations[flag_name] = self.flag_evaluations.get(flag_name, 0) + 1
        
        # Check if globally disabled
        if not flag.enabled:
            span.set_attribute("flag.enabled", False)
            return False
        
        tenant_id = context.get("tenant_id", "")
        user_id = context.get("user_id", "")
        
        # Check blocked tenants
        if tenant_id in flag.blocked_tenants:
            span.set_attribute("flag.blocked_tenant", True)
            return False
        
        # Evaluate based on strategy
        result = await self._evaluate_strategy(flag, context)
        
        span.set_attributes({
            "flag.name": flag_name,
            "flag.type": flag.flag_type.value,
            "flag.strategy": flag.rollout_strategy.value,
            "flag.result": str(result)
        })
        
        # Log evaluation for analysis
        await self._log_evaluation(flag_name, context, result)
        
        return result
    
    async def _evaluate_strategy(
        self,
        flag: FeatureFlag,
        context: Dict[str, Any]
    ) -> Any:
        """Evaluate flag based on rollout strategy"""
        tenant_id = context.get("tenant_id", "")
        user_id = context.get("user_id", "")
        
        if flag.rollout_strategy == RolloutStrategy.ALL_USERS:
            return self._get_flag_value(flag)
        
        elif flag.rollout_strategy == RolloutStrategy.SPECIFIC_TENANTS:
            if tenant_id in flag.allowed_tenants:
                return self._get_flag_value(flag)
            return False
        
        elif flag.rollout_strategy == RolloutStrategy.PERCENTAGE:
            # Use consistent hashing for stickiness
            hash_input = f"{flag.name}:{tenant_id}:{user_id}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            percentage = hash_value % 100
            
            if percentage < flag.rollout_percentage:
                return self._get_flag_value(flag)
            return False
        
        elif flag.rollout_strategy == RolloutStrategy.RING_BASED:
            # Determine ring
            ring = self._determine_ring(tenant_id)
            ring_percentages = {
                "ring0": 100,  # Internal
                "ring1": 75,   # Beta
                "ring2": 50,   # Early adopters
                "ring3": flag.rollout_percentage  # General
            }
            
            if random.random() * 100 < ring_percentages.get(ring, 0):
                return self._get_flag_value(flag)
            return False
        
        elif flag.rollout_strategy == RolloutStrategy.CANARY:
            # Canary deployment - small percentage first
            hash_input = f"{flag.name}:{tenant_id}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            
            if (hash_value % 100) < flag.rollout_percentage:
                return self._get_flag_value(flag)
            return False
        
        elif flag.rollout_strategy == RolloutStrategy.BLUE_GREEN:
            # Blue-green deployment
            if context.get("deployment_group") == "green":
                return self._get_flag_value(flag)
            return False
        
        return False
    
    def _get_flag_value(self, flag: FeatureFlag) -> Any:
        """Get the actual value of a flag based on its type"""
        if flag.flag_type == FeatureFlagType.BOOLEAN:
            return True
        
        elif flag.flag_type == FeatureFlagType.PERCENTAGE:
            return flag.rollout_percentage
        
        elif flag.flag_type == FeatureFlagType.VARIANT:
            # Select variant based on weighted distribution
            if flag.variants:
                rand = random.random() * 100
                cumulative = 0
                for variant, weight in flag.variants.items():
                    cumulative += weight
                    if rand < cumulative:
                        return variant
            return "control"
        
        elif flag.flag_type == FeatureFlagType.GRADUAL:
            # Calculate current percentage based on schedule
            if flag.schedule:
                return self._calculate_gradual_percentage(flag)
            return flag.rollout_percentage
        
        elif flag.flag_type == FeatureFlagType.SCHEDULED:
            # Check if within schedule
            if flag.schedule:
                now = datetime.utcnow()
                start = datetime.fromisoformat(flag.schedule.get("start_date"))
                end = datetime.fromisoformat(flag.schedule.get("end_date"))
                return start <= now <= end
            return False
        
        return False
    
    def _calculate_gradual_percentage(self, flag: FeatureFlag) -> float:
        """Calculate percentage for gradual rollout"""
        if not flag.schedule:
            return flag.rollout_percentage
        
        now = datetime.utcnow()
        start = datetime.fromisoformat(flag.schedule.get("start_date"))
        end = datetime.fromisoformat(flag.schedule.get("end_date"))
        
        if now < start:
            return 0.0
        if now > end:
            return 100.0
        
        days_elapsed = (now - start).days
        daily_increment = flag.schedule.get("daily_increment", 10.0)
        
        return min(100.0, days_elapsed * daily_increment)
    
    def _determine_ring(self, tenant_id: str) -> str:
        """Determine deployment ring for tenant"""
        # In production, this would be based on tenant configuration
        internal_tenants = ["internal", "testing"]
        beta_tenants = ["beta1", "beta2"]
        early_adopters = ["early1", "early2"]
        
        if tenant_id in internal_tenants:
            return "ring0"
        elif tenant_id in beta_tenants:
            return "ring1"
        elif tenant_id in early_adopters:
            return "ring2"
        else:
            return "ring3"
    
    async def update_flag(
        self,
        flag_name: str,
        updates: Dict[str, Any]
    ):
        """Update a feature flag"""
        if flag_name not in self.flags:
            raise ValueError(f"Flag {flag_name} not found")
        
        flag = self.flags[flag_name]
        
        # Update fields
        for key, value in updates.items():
            if hasattr(flag, key):
                setattr(flag, key, value)
        
        flag.updated_at = datetime.utcnow()
        
        # Persist to storage
        await self._persist_flag(flag)
        
        # Emit telemetry
        span = trace.get_current_span()
        span.add_event("flag_updated", {
            "flag_name": flag_name,
            "updates": json.dumps(updates, default=str)
        })
    
    async def create_flag(self, flag: FeatureFlag):
        """Create a new feature flag"""
        self.flags[flag.name] = flag
        await self._persist_flag(flag)
    
    async def delete_flag(self, flag_name: str):
        """Delete a feature flag"""
        if flag_name in self.flags:
            del self.flags[flag_name]
            # Remove from storage
            if self.redis_client:
                await self.redis_client.delete(f"flag:{flag_name}")
    
    async def _persist_flag(self, flag: FeatureFlag):
        """Persist flag to storage"""
        if self.redis_client:
            key = f"flag:{flag.name}"
            value = json.dumps({
                "name": flag.name,
                "description": flag.description,
                "flag_type": flag.flag_type.value,
                "enabled": flag.enabled,
                "rollout_strategy": flag.rollout_strategy.value,
                "rollout_percentage": flag.rollout_percentage,
                "allowed_tenants": flag.allowed_tenants,
                "blocked_tenants": flag.blocked_tenants,
                "variants": flag.variants,
                "schedule": flag.schedule,
                "metadata": flag.metadata,
                "created_at": flag.created_at.isoformat(),
                "updated_at": flag.updated_at.isoformat()
            })
            await self.redis_client.set(key, value)
    
    async def _load_flags_from_storage(self):
        """Load flags from persistent storage"""
        if self.redis_client:
            # Scan for flag keys
            cursor = b'0'
            while cursor:
                cursor, keys = await self.redis_client.scan(
                    cursor, match="flag:*"
                )
                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        flag_data = json.loads(data)
                        # Reconstruct flag
                        # (Implementation details omitted for brevity)
    
    async def _log_evaluation(
        self,
        flag_name: str,
        context: Dict[str, Any],
        result: Any
    ):
        """Log flag evaluation for analysis"""
        if self.redis_client:
            # Store evaluation in time series
            key = f"flag_eval:{flag_name}:{datetime.utcnow().strftime('%Y%m%d')}"
            await self.redis_client.hincrby(key, str(result), 1)
            await self.redis_client.expire(key, 86400 * 7)  # 7 days retention
    
    async def _gradual_rollout_updater(self):
        """Update gradual rollout percentages"""
        while True:
            try:
                for flag in self.flags.values():
                    if flag.flag_type == FeatureFlagType.GRADUAL and flag.enabled:
                        old_percentage = flag.rollout_percentage
                        new_percentage = self._calculate_gradual_percentage(flag)
                        
                        if new_percentage != old_percentage:
                            flag.rollout_percentage = new_percentage
                            await self._persist_flag(flag)
                            print(f"Updated {flag.name} rollout to {new_percentage}%")
                
            except Exception as e:
                print(f"Gradual rollout updater error: {e}")
            
            await asyncio.sleep(3600)  # Check hourly
    
    async def _metrics_collector(self):
        """Collect flag usage metrics"""
        while True:
            try:
                metrics = {
                    "total_evaluations": sum(self.flag_evaluations.values()),
                    "flag_evaluations": dict(self.flag_evaluations),
                    "enabled_flags": sum(1 for f in self.flags.values() if f.enabled),
                    "rollout_progress": {
                        name: flag.rollout_percentage
                        for name, flag in self.flags.items()
                        if flag.flag_type in [FeatureFlagType.PERCENTAGE, FeatureFlagType.GRADUAL]
                    }
                }
                
                # Store metrics
                if self.redis_client:
                    await self.redis_client.set(
                        "flag_metrics",
                        json.dumps(metrics),
                        expire=3600
                    )
                
            except Exception as e:
                print(f"Metrics collector error: {e}")
            
            await asyncio.sleep(60)  # Collect every minute
    
    async def get_all_flags(self) -> List[FeatureFlag]:
        """Get all feature flags"""
        return list(self.flags.values())
    
    async def get_flag_metrics(self, flag_name: str) -> Dict[str, Any]:
        """Get metrics for a specific flag"""
        if flag_name not in self.flags:
            return {}
        
        return {
            "evaluations": self.flag_evaluations.get(flag_name, 0),
            "enabled": self.flags[flag_name].enabled,
            "rollout_percentage": self.flags[flag_name].rollout_percentage,
            "last_updated": self.flags[flag_name].updated_at.isoformat()
        }


class CapabilityRegistry:
    """
    Registry for agent capabilities and dynamic discovery
    """
    
    def __init__(self):
        self.capabilities: Dict[str, AgentCapability] = {}
        self.agents: Dict[str, RegisteredAgent] = {}
        self.capability_to_agents: Dict[str, List[str]] = {}
        self._register_default_capabilities()
    
    def _register_default_capabilities(self):
        """Register default agent capabilities"""
        default_capabilities = [
            AgentCapability(
                capability_id="code_generation",
                name="Code Generation",
                description="Generate code in various languages",
                required_models=["gpt-5", "deepseek-coder", "yi-coder"],
                required_resources={"memory_gb": 8, "cpu_cores": 2},
                input_schema={"language": "string", "requirements": "string"},
                output_schema={"code": "string", "tests": "array"},
                performance_metrics={"avg_latency_ms": 2000, "success_rate": 0.95},
                cost_per_invocation=0.10
            ),
            AgentCapability(
                capability_id="code_review",
                name="Code Review",
                description="Review code for quality and security",
                required_models=["claude-4.1", "gpt-5"],
                required_resources={"memory_gb": 4, "cpu_cores": 1},
                input_schema={"code": "string", "language": "string"},
                output_schema={"issues": "array", "score": "number"},
                performance_metrics={"avg_latency_ms": 1500, "success_rate": 0.98},
                cost_per_invocation=0.05
            ),
            AgentCapability(
                capability_id="web_search",
                name="Web Search",
                description="Search the web for information",
                required_models=["o4-mini"],
                required_resources={"memory_gb": 2, "cpu_cores": 1},
                input_schema={"query": "string", "max_results": "integer"},
                output_schema={"results": "array"},
                performance_metrics={"avg_latency_ms": 500, "success_rate": 0.99},
                cost_per_invocation=0.01
            ),
            AgentCapability(
                capability_id="robot_control",
                name="Robot Control",
                description="Control robotic systems",
                required_models=["rt-2", "pi-0"],
                required_resources={"memory_gb": 16, "gpu": "required"},
                input_schema={"command": "string", "parameters": "object"},
                output_schema={"status": "string", "result": "object"},
                performance_metrics={"avg_latency_ms": 100, "success_rate": 0.99},
                cost_per_invocation=0.50
            ),
            AgentCapability(
                capability_id="protein_folding",
                name="Protein Folding",
                description="Predict protein structures",
                required_models=["alphafold-3", "esm3"],
                required_resources={"memory_gb": 32, "gpu": "required"},
                input_schema={"sequence": "string", "constraints": "object"},
                output_schema={"structure": "object", "confidence": "number"},
                performance_metrics={"avg_latency_ms": 30000, "success_rate": 0.85},
                cost_per_invocation=5.00
            )
        ]
        
        for cap in default_capabilities:
            self.capabilities[cap.capability_id] = cap
    
    async def register_capability(self, capability: AgentCapability):
        """Register a new capability"""
        self.capabilities[capability.capability_id] = capability
        
        # Update agent mappings
        for agent_id, agent in self.agents.items():
            if capability.capability_id in agent.capabilities:
                if capability.capability_id not in self.capability_to_agents:
                    self.capability_to_agents[capability.capability_id] = []
                if agent_id not in self.capability_to_agents[capability.capability_id]:
                    self.capability_to_agents[capability.capability_id].append(agent_id)
    
    async def register_agent(
        self,
        name: str,
        version: str,
        capabilities: List[str],
        endpoint: str,
        health_check: str = "/health"
    ) -> str:
        """Register an agent with its capabilities"""
        agent_id = f"{name}-{version}-{hashlib.md5(endpoint.encode()).hexdigest()[:8]}"
        
        agent = RegisteredAgent(
            agent_id=agent_id,
            name=name,
            version=version,
            capabilities=capabilities,
            endpoint=endpoint,
            health_check=health_check,
            status="healthy",
            last_heartbeat=datetime.utcnow()
        )
        
        self.agents[agent_id] = agent
        
        # Update capability mappings
        for cap_id in capabilities:
            if cap_id not in self.capability_to_agents:
                self.capability_to_agents[cap_id] = []
            self.capability_to_agents[cap_id].append(agent_id)
        
        return agent_id
    
    async def discover_agents(
        self,
        capability: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[RegisteredAgent]:
        """Discover agents with a specific capability"""
        agent_ids = self.capability_to_agents.get(capability, [])
        
        agents = []
        for agent_id in agent_ids:
            agent = self.agents.get(agent_id)
            if agent and agent.status == "healthy":
                # Apply constraints if provided
                if constraints:
                    if "version" in constraints and agent.version != constraints["version"]:
                        continue
                    if "min_success_rate" in constraints:
                        cap = self.capabilities.get(capability)
                        if cap and cap.success_rate < constraints["min_success_rate"]:
                            continue
                
                agents.append(agent)
        
        return agents
    
    async def get_capability_info(
        self,
        capability_id: str
    ) -> Optional[AgentCapability]:
        """Get information about a capability"""
        return self.capabilities.get(capability_id)
    
    async def update_agent_health(
        self,
        agent_id: str,
        status: str,
        heartbeat: Optional[datetime] = None
    ):
        """Update agent health status"""
        if agent_id in self.agents:
            self.agents[agent_id].status = status
            if heartbeat:
                self.agents[agent_id].last_heartbeat = heartbeat
    
    async def get_capability_metrics(self) -> Dict[str, Any]:
        """Get metrics for all capabilities"""
        metrics = {}
        
        for cap_id, capability in self.capabilities.items():
            agents = self.capability_to_agents.get(cap_id, [])
            healthy_agents = [
                a for a in agents
                if self.agents.get(a) and self.agents[a].status == "healthy"
            ]
            
            metrics[cap_id] = {
                "name": capability.name,
                "total_agents": len(agents),
                "healthy_agents": len(healthy_agents),
                "avg_latency_ms": capability.performance_metrics.get("avg_latency_ms", 0),
                "success_rate": capability.performance_metrics.get("success_rate", 0),
                "cost_per_invocation": capability.cost_per_invocation
            }
        
        return metrics


# Example usage
async def main():
    config = {"redis_url": "redis://localhost"}
    
    # Initialize feature flags
    ff_manager = FeatureFlagManager(config)
    await ff_manager.initialize()
    
    # Evaluate flags
    context = {
        "tenant_id": "tenant-123",
        "user_id": "user-456"
    }
    
    robotics_enabled = await ff_manager.evaluate("enable_robotics", context)
    print(f"Robotics enabled: {robotics_enabled}")
    
    model_strategy = await ff_manager.evaluate("model_selection_strategy", context)
    print(f"Model strategy: {model_strategy}")
    
    # Update flag
    await ff_manager.update_flag(
        "enable_robotics",
        {"enabled": True, "rollout_percentage": 25.0}
    )
    
    # Initialize capability registry
    registry = CapabilityRegistry()
    
    # Register an agent
    agent_id = await registry.register_agent(
        name="code-generator",
        version="1.0.0",
        capabilities=["code_generation", "code_review"],
        endpoint="http://code-gen-service:8080"
    )
    
    print(f"Registered agent: {agent_id}")
    
    # Discover agents
    code_agents = await registry.discover_agents("code_generation")
    print(f"Code generation agents: {[a.name for a in code_agents]}")
    
    # Get capability metrics
    metrics = await registry.get_capability_metrics()
    print(f"Capability metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
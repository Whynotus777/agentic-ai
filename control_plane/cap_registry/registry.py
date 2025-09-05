"""
Capability Registry for multi-agent system.
Manages capability definitions, validation, and access control.
"""
import yaml
import json
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import hashlib


@dataclass
class Capability:
    """Capability definition."""
    id: str
    owner: str
    scopes: List[str]
    rate_limits: Dict[str, int]
    budget_usd_day: float
    tools: List[str]
    pii: bool
    description: str = ""
    version: str = "1.0"
    created_at: str = ""
    updated_at: str = ""
    tags: List[str] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = []
            
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Capability':
        return cls(**data)


class CapabilityRegistry:
    """Registry for managing system capabilities."""
    
    def __init__(self, registry_path: str = "control_plane/cap_registry/capabilities.yaml"):
        self.registry_path = Path(registry_path)
        self.capabilities: Dict[str, Capability] = {}
        self.capability_cache: Dict[str, Any] = {}
        self._load_capabilities()
        
    def _load_capabilities(self):
        """Load capabilities from YAML file."""
        if not self.registry_path.exists():
            # Load defaults if file doesn't exist
            self._init_default_capabilities()
            return
            
        with open(self.registry_path, 'r') as f:
            data = yaml.safe_load(f)
            
        for cap_id, cap_data in data.get('capabilities', {}).items():
            cap_data['id'] = cap_id
            self.capabilities[cap_id] = Capability.from_dict(cap_data)
    
    def _init_default_capabilities(self):
        """Initialize with default capabilities."""
        defaults = [
            Capability(
                id="repo.commit",
                owner="vcs-team",
                scopes=["write", "modify"],
                rate_limits={"per_min": 10, "per_hour": 100},
                budget_usd_day=10.0,
                tools=["git", "diff"],
                pii=False,
                description="Commit code to repository",
                tags=["vcs", "critical"]
            ),
            Capability(
                id="repo.push",
                owner="vcs-team",
                scopes=["write", "publish"],
                rate_limits={"per_min": 5, "per_hour": 50},
                budget_usd_day=10.0,
                tools=["git"],
                pii=False,
                description="Push code to remote repository",
                dependencies=["repo.commit"]
            ),
            Capability(
                id="robot.actuate",
                owner="robotics-team",
                scopes=["control", "physical"],
                rate_limits={"per_min": 1, "per_hour": 60},
                budget_usd_day=500.0,
                tools=["ros2", "moveit"],
                pii=False,
                description="Control physical robot actuators",
                tags=["safety-critical", "robotics"]
            ),
            Capability(
                id="robot.sense",
                owner="robotics-team",
                scopes=["read", "monitor"],
                rate_limits={"per_min": 100, "per_hour": 6000},
                budget_usd_day=100.0,
                tools=["ros2", "opencv"],
                pii=False,
                description="Read robot sensor data"
            ),
            Capability(
                id="llm.generate",
                owner="ai-team",
                scopes=["inference", "generate"],
                rate_limits={"per_min": 60, "per_hour": 1000},
                budget_usd_day=100.0,
                tools=["openai", "anthropic"],
                pii=True,
                description="Generate text using LLM"
            ),
            Capability(
                id="data.read",
                owner="data-team",
                scopes=["read"],
                rate_limits={"per_min": 1000, "per_hour": 50000},
                budget_usd_day=20.0,
                tools=["postgres", "redis"],
                pii=True,
                description="Read data from storage"
            ),
            Capability(
                id="sim.run",
                owner="simulation-team",
                scopes=["execute", "compute"],
                rate_limits={"per_min": 10, "per_hour": 100},
                budget_usd_day=200.0,
                tools=["mujoco", "isaac"],
                pii=False,
                description="Run simulation environment"
            )
        ]
        
        for cap in defaults:
            self.capabilities[cap.id] = cap
    
    def register(self, capability: Capability) -> bool:
        """
        Register a new capability.
        
        Returns True if successful, False if already exists.
        """
        if capability.id in self.capabilities:
            return False
            
        # Validate dependencies exist
        for dep in capability.dependencies or []:
            if dep not in self.capabilities:
                raise ValueError(f"Dependency {dep} not found for {capability.id}")
                
        capability.created_at = datetime.utcnow().isoformat()
        capability.updated_at = capability.created_at
        self.capabilities[capability.id] = capability
        self._invalidate_cache(capability.id)
        return True
    
    def get(self, capability_id: str) -> Optional[Capability]:
        """Get capability by ID."""
        return self.capabilities.get(capability_id)
    
    def validate_agent_capabilities(self, 
                                  agent_id: str, 
                                  requested_capabilities: List[str]) -> Dict[str, Any]:
        """
        Validate that an agent can use requested capabilities.
        
        Returns validation result with allowed/denied capabilities.
        """
        result = {
            "agent_id": agent_id,
            "allowed": [],
            "denied": [],
            "missing": [],
            "errors": []
        }
        
        for cap_id in requested_capabilities:
            if cap_id not in self.capabilities:
                result["missing"].append(cap_id)
                result["errors"].append(f"Capability {cap_id} not registered")
                continue
                
            cap = self.capabilities[cap_id]
            
            # Check dependencies
            deps_met = True
            for dep in cap.dependencies or []:
                if dep not in requested_capabilities:
                    deps_met = False
                    result["errors"].append(
                        f"Capability {cap_id} requires {dep}"
                    )
                    break
                    
            if deps_met:
                result["allowed"].append(cap_id)
            else:
                result["denied"].append(cap_id)
                
        return result
    
    def check_rate_limit(self, capability_id: str, 
                        current_rate: Dict[str, int]) -> bool:
        """
        Check if current rate is within limits for capability.
        
        Args:
            capability_id: Capability to check
            current_rate: Dict with 'per_min' and 'per_hour' counts
            
        Returns:
            True if within limits, False otherwise
        """
        cap = self.capabilities.get(capability_id)
        if not cap:
            return False
            
        limits = cap.rate_limits
        
        if current_rate.get('per_min', 0) > limits.get('per_min', float('inf')):
            return False
        if current_rate.get('per_hour', 0) > limits.get('per_hour', float('inf')):
            return False
            
        return True
    
    def check_budget(self, capability_id: str, 
                    current_spend: float) -> bool:
        """
        Check if current spend is within daily budget.
        
        Returns True if within budget, False otherwise.
        """
        cap = self.capabilities.get(capability_id)
        if not cap:
            return False
            
        return current_spend <= cap.budget_usd_day
    
    def get_tools_for_capability(self, capability_id: str) -> List[str]:
        """Get tools required for a capability."""
        cap = self.capabilities.get(capability_id)
        return cap.tools if cap else []
    
    def get_capabilities_by_owner(self, owner: str) -> List[Capability]:
        """Get all capabilities owned by a team/owner."""
        return [
            cap for cap in self.capabilities.values() 
            if cap.owner == owner
        ]
    
    def get_capabilities_by_scope(self, scope: str) -> List[Capability]:
        """Get all capabilities with a specific scope."""
        return [
            cap for cap in self.capabilities.values()
            if scope in cap.scopes
        ]
    
    def get_pii_capabilities(self) -> List[Capability]:
        """Get all capabilities that handle PII."""
        return [
            cap for cap in self.capabilities.values()
            if cap.pii
        ]
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get dependency graph of all capabilities."""
        graph = {}
        for cap_id, cap in self.capabilities.items():
            graph[cap_id] = cap.dependencies or []
        return graph
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export registry metrics."""
        return {
            "total_capabilities": len(self.capabilities),
            "owners": len(set(cap.owner for cap in self.capabilities.values())),
            "pii_capabilities": len(self.get_pii_capabilities()),
            "total_daily_budget": sum(
                cap.budget_usd_day for cap in self.capabilities.values()
            ),
            "capability_list": list(self.capabilities.keys())
        }
    
    def _invalidate_cache(self, capability_id: str):
        """Invalidate cache entries for a capability."""
        keys_to_remove = [
            k for k in self.capability_cache.keys() 
            if capability_id in k
        ]
        for key in keys_to_remove:
            del self.capability_cache[key]
    
    def save(self):
        """Save current registry to file."""
        data = {
            'capabilities': {
                cap_id: cap.to_dict() 
                for cap_id, cap in self.capabilities.items()
            }
        }
        
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
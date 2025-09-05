"""
Feature Flags Store for multi-agent system.
Supports percentage rollout, targeting by env/tenant/service.
"""
import yaml
import json
import random
import hashlib
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from enum import Enum


class TargetType(Enum):
    ENVIRONMENT = "environment"
    TENANT = "tenant"
    SERVICE = "service"
    ROUTE = "route"
    USER = "user"
    GROUP = "group"


@dataclass
class FlagTarget:
    """Target specification for feature flag."""
    type: TargetType
    values: List[str]
    enabled: bool = True
    
    def matches(self, context: Dict[str, Any]) -> bool:
        """Check if context matches this target."""
        context_key = self.type.value
        context_value = context.get(context_key)
        
        if context_value is None:
            return False
            
        return context_value in self.values


@dataclass
class FeatureFlag:
    """Feature flag configuration."""
    name: str
    description: str
    enabled: bool
    rollout_percentage: float  # 0.0 to 100.0
    targets: List[FlagTarget]
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str
    version: str = "1.0"
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['targets'] = [
            {'type': t.type.value, 'values': t.values, 'enabled': t.enabled}
            for t in self.targets
        ]
        return d
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FeatureFlag':
        targets = []
        for t in data.get('targets', []):
            targets.append(FlagTarget(
                type=TargetType(t['type']),
                values=t['values'],
                enabled=t.get('enabled', True)
            ))
        
        data['targets'] = targets
        return cls(**data)


class FeatureFlagStore:
    """Store and evaluate feature flags."""
    
    def __init__(self, flags_path: str = "control_plane/flags/flags.yaml"):
        self.flags_path = Path(flags_path)
        self.flags: Dict[str, FeatureFlag] = {}
        self.evaluation_cache: Dict[str, bool] = {}
        self._load_flags()
        
    def _load_flags(self):
        """Load feature flags from YAML file."""
        if not self.flags_path.exists():
            self._init_default_flags()
            return
            
        with open(self.flags_path, 'r') as f:
            data = yaml.safe_load(f)
            
        for flag_name, flag_data in data.get('flags', {}).items():
            flag_data['name'] = flag_name
            self.flags[flag_name] = FeatureFlag.from_dict(flag_data)
    
    def _init_default_flags(self):
        """Initialize with default feature flags."""
        now = datetime.utcnow().isoformat()
        
        defaults = [
            FeatureFlag(
                name="use_deepseek_v3",
                description="Use DeepSeek-V3 as default orchestrator model",
                enabled=False,
                rollout_percentage=0.0,
                targets=[
                    FlagTarget(
                        type=TargetType.ENVIRONMENT,
                        values=["dev", "staging"],
                        enabled=True
                    )
                ],
                metadata={"model": "deepseek-v3", "previous": "o4-mini"},
                created_at=now,
                updated_at=now
            ),
            FeatureFlag(
                name="enable_rt2_robotics",
                description="Enable RT-2 model for robotics domain",
                enabled=True,
                rollout_percentage=100.0,
                targets=[
                    FlagTarget(
                        type=TargetType.SERVICE,
                        values=["robotics-service", "robot-planner"],
                        enabled=True
                    )
                ],
                metadata={"model": "rt-2", "domain": "robotics"},
                created_at=now,
                updated_at=now
            ),
            FeatureFlag(
                name="hitl_auto_approve",
                description="Auto-approve HITL requests for non-critical operations",
                enabled=False,
                rollout_percentage=10.0,
                targets=[
                    FlagTarget(
                        type=TargetType.TENANT,
                        values=["test-tenant", "dev-tenant"],
                        enabled=True
                    )
                ],
                metadata={"risk_level": "low"},
                created_at=now,
                updated_at=now
            ),
            FeatureFlag(
                name="enhanced_monitoring",
                description="Enable enhanced monitoring and tracing",
                enabled=True,
                rollout_percentage=50.0,
                targets=[
                    FlagTarget(
                        type=TargetType.SERVICE,
                        values=["api-gateway", "orchestrator"],
                        enabled=True
                    )
                ],
                metadata={"sampling_rate": 0.1},
                created_at=now,
                updated_at=now
            ),
            FeatureFlag(
                name="budget_alerts_v2",
                description="Use v2 budget alerting system",
                enabled=True,
                rollout_percentage=75.0,
                targets=[
                    FlagTarget(
                        type=TargetType.TENANT,
                        values=["premium-*", "enterprise-*"],
                        enabled=True
                    )
                ],
                metadata={"alert_channels": ["email", "slack"]},
                created_at=now,
                updated_at=now
            )
        ]
        
        for flag in defaults:
            self.flags[flag.name] = flag
    
    def is_enabled(self, flag_name: str, context: Dict[str, Any]) -> bool:
        """
        Check if a feature flag is enabled for the given context.
        
        Args:
            flag_name: Name of the feature flag
            context: Dict containing evaluation context (tenant, service, env, etc.)
            
        Returns:
            True if flag is enabled, False otherwise
        """
        # Check cache
        cache_key = self._get_cache_key(flag_name, context)
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        flag = self.flags.get(flag_name)
        if not flag:
            return False
        
        # Global flag check
        if not flag.enabled:
            self.evaluation_cache[cache_key] = False
            return False
        
        # Check targeted rules first (higher priority)
        for target in flag.targets:
            if target.matches(context):
                result = target.enabled
                self.evaluation_cache[cache_key] = result
                return result
        
        # Percentage rollout (using stable hash)
        if flag.rollout_percentage > 0:
            result = self._check_rollout(flag_name, context, flag.rollout_percentage)
            self.evaluation_cache[cache_key] = result
            return result
        
        # Default to flag's enabled state
        result = flag.enabled
        self.evaluation_cache[cache_key] = result
        return result
    
    def _check_rollout(self, flag_name: str, 
                      context: Dict[str, Any], 
                      percentage: float) -> bool:
        """
        Stable percentage rollout based on context hash.
        
        Uses a stable hash so the same context always gets the same result.
        """
        # Create stable identifier from context
        identifier = f"{flag_name}:{context.get('tenant_id', '')}:{context.get('user_id', '')}:{context.get('service', '')}"
        hash_value = int(hashlib.md5(identifier.encode()).hexdigest(), 16)
        
        # Use modulo to get value between 0-100
        rollout_value = hash_value % 100
        
        return rollout_value < percentage
    
    def _get_cache_key(self, flag_name: str, context: Dict[str, Any]) -> str:
        """Generate cache key for flag evaluation."""
        ctx_str = json.dumps(context, sort_keys=True)
        return f"{flag_name}:{hashlib.md5(ctx_str.encode()).hexdigest()}"
    
    def get_flag(self, flag_name: str) -> Optional[FeatureFlag]:
        """Get flag configuration."""
        return self.flags.get(flag_name)
    
    def list_flags(self) -> List[str]:
        """List all flag names."""
        return list(self.flags.keys())
    
    def get_all_flags(self) -> Dict[str, FeatureFlag]:
        """Get all flags."""
        return self.flags.copy()
    
    def evaluate_all(self, context: Dict[str, Any]) -> Dict[str, bool]:
        """
        Evaluate all flags for a given context.
        
        Returns dict of flag_name -> enabled status
        """
        result = {}
        for flag_name in self.flags:
            result[flag_name] = self.is_enabled(flag_name, context)
        return result
    
    def update_flag(self, flag_name: str, 
                   enabled: Optional[bool] = None,
                   rollout_percentage: Optional[float] = None) -> bool:
        """
        Update flag configuration (runtime update).
        
        Returns True if successful.
        """
        flag = self.flags.get(flag_name)
        if not flag:
            return False
            
        if enabled is not None:
            flag.enabled = enabled
            
        if rollout_percentage is not None:
            flag.rollout_percentage = max(0.0, min(100.0, rollout_percentage))
            
        flag.updated_at = datetime.utcnow().isoformat()
        
        # Clear cache for this flag
        self._invalidate_flag_cache(flag_name)
        
        return True
    
    def _invalidate_flag_cache(self, flag_name: str):
        """Invalidate cache entries for a flag."""
        keys_to_remove = [
            k for k in self.evaluation_cache.keys() 
            if k.startswith(f"{flag_name}:")
        ]
        for key in keys_to_remove:
            del self.evaluation_cache[key]
    
    def reload_flags(self):
        """Reload flags from file and clear cache."""
        self.flags.clear()
        self.evaluation_cache.clear()
        self._load_flags()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get feature flag metrics."""
        return {
            "total_flags": len(self.flags),
            "enabled_flags": sum(1 for f in self.flags.values() if f.enabled),
            "flags_with_targets": sum(1 for f in self.flags.values() if f.targets),
            "cache_size": len(self.evaluation_cache),
            "flags_summary": {
                name: {
                    "enabled": flag.enabled,
                    "rollout": flag.rollout_percentage,
                    "has_targets": len(flag.targets) > 0
                }
                for name, flag in self.flags.items()
            }
        }
    
    def save(self):
        """Save current flags to file."""
        data = {
            'flags': {
                flag_name: flag.to_dict()
                for flag_name, flag in self.flags.items()
            }
        }
        
        self.flags_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.flags_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
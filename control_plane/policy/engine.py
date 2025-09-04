# control_plane/policy/engine.py
"""
Policy Engine with dry-run mode, RBAC, and comprehensive audit logging.
Enforces security, data governance, and operational policies.
"""

import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio
import re

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)


class PolicyAction(Enum):
    """Policy decision outcomes"""
    ALLOW = "allow"
    DENY = "deny"
    ALLOW_WITH_REDACTION = "allow_with_redaction"
    REQUIRE_HITL = "require_hitl"
    REQUIRE_ADDITIONAL_AUTH = "require_additional_auth"


class DataClassification(Enum):
    """Data classification tags"""
    PII = "PII"
    SENSITIVE = "SENSITIVE"
    CONFIDENTIAL = "CONFIDENTIAL"
    EXPORT_OK = "EXPORT_OK"
    PUBLIC = "PUBLIC"
    HIPAA_PHI = "HIPAA_PHI"
    FINANCIAL = "FINANCIAL"


@dataclass
class PolicyContext:
    """Context for policy evaluation"""
    user_id: str
    tenant_id: str
    roles: Set[str]
    action: str
    resource: str
    data_tags: Set[DataClassification] = field(default_factory=set)
    request_metadata: Dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""
    environment: str = "production"
    ip_address: str = ""
    user_agent: str = ""
    cost_usd: float = 0.0


@dataclass
class PolicyResult:
    """Result of policy evaluation"""
    action: PolicyAction
    reasons: List[str]
    applied_policies: List[str]
    redactions: Optional[Dict[str, str]] = None
    hitl_requirement: Optional[Dict[str, Any]] = None
    audit_record: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass
class PolicyRule:
    """Individual policy rule"""
    id: str
    name: str
    description: str
    conditions: Dict[str, Any]
    effect: PolicyAction
    priority: int = 100
    enabled: bool = True
    dry_run: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class PolicyEngine:
    """
    Main policy engine with dry-run support and comprehensive logging
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rules: List[PolicyRule] = []
        self.rbac_map: Dict[str, Set[str]] = defaultdict(set)
        self.separation_of_duties: List[Tuple[str, str]] = []
        self.data_retention_matrix: Dict[DataClassification, Dict] = {}
        self.audit_logger = AuditLogger()
        self.metrics_collector = MetricsCollector()
        self._load_policies()
        
    def _load_policies(self):
        """Load policies from configuration"""
        # Load RBAC mappings
        rbac_config = self.config.get("rbac", {})
        for capability, roles in rbac_config.get("capability_roles", {}).items():
            self.rbac_map[capability] = set(roles)
            
        # Load separation of duties
        self.separation_of_duties = [
            tuple(pair) for pair in rbac_config.get("separation_of_duties", [])
        ]
        
        # Load data retention policies
        retention_config = self.config.get("data_retention", {})
        for tag, policy in retention_config.items():
            if tag in DataClassification.__members__:
                self.data_retention_matrix[DataClassification[tag]] = policy
                
        # Load policy rules
        for rule_config in self.config.get("rules", []):
            self.rules.append(PolicyRule(**rule_config))
            
        # Sort rules by priority
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    @tracer.start_as_current_span("evaluate_policy")
    async def evaluate(
        self,
        context: PolicyContext,
        dry_run: bool = False
    ) -> PolicyResult:
        """
        Evaluate policies for a given context
        
        Args:
            context: The context to evaluate
            dry_run: If True, log what would happen without enforcing
            
        Returns:
            PolicyResult with decision and metadata
        """
        start_time = time.time()
        span = trace.get_current_span()
        
        # Add context to span
        span.set_attributes({
            "policy.user_id": context.user_id,
            "policy.tenant_id": context.tenant_id,
            "policy.action": context.action,
            "policy.resource": context.resource,
            "policy.dry_run": dry_run,
            "policy.trace_id": context.trace_id
        })
        
        result = PolicyResult(
            action=PolicyAction.DENY,  # Default deny
            reasons=[],
            applied_policies=[]
        )
        
        try:
            # Check RBAC
            rbac_result = await self._check_rbac(context)
            if rbac_result.action == PolicyAction.DENY:
                result = rbac_result
                if not dry_run:
                    return result
                    
            # Check separation of duties
            sod_result = await self._check_separation_of_duties(context)
            if sod_result.action == PolicyAction.DENY:
                result = sod_result
                if not dry_run:
                    return result
                    
            # Check data governance policies
            data_result = await self._check_data_policies(context)
            if data_result.action != PolicyAction.ALLOW:
                result = data_result
                if not dry_run and result.action == PolicyAction.DENY:
                    return result
                    
            # Evaluate custom rules
            for rule in self.rules:
                if not rule.enabled:
                    continue
                    
                if await self._evaluate_rule(rule, context):
                    result.applied_policies.append(rule.id)
                    
                    if rule.dry_run or dry_run:
                        # In dry-run, log what would happen
                        self._log_dry_run(rule, context, result)
                    else:
                        # Apply the rule effect
                        if rule.effect == PolicyAction.DENY:
                            result.action = PolicyAction.DENY
                            result.reasons.append(f"Denied by rule: {rule.name}")
                            span.set_status(Status(StatusCode.ERROR, "Policy denied"))
                            return result
                        elif rule.effect == PolicyAction.REQUIRE_HITL:
                            result.action = PolicyAction.REQUIRE_HITL
                            result.hitl_requirement = self._generate_hitl_requirement(
                                rule, context
                            )
                            
            # If no deny rules matched, allow
            if result.action != PolicyAction.DENY:
                result.action = PolicyAction.ALLOW
                
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            result.action = PolicyAction.DENY
            result.reasons.append(f"Policy evaluation error: {str(e)}")
            
        finally:
            # Calculate latency
            result.latency_ms = (time.time() - start_time) * 1000
            
            # Audit log
            result.audit_record = await self.audit_logger.log(
                context, result, dry_run
            )
            
            # Collect metrics
            await self.metrics_collector.record(context, result)
            
            span.set_attributes({
                "policy.result": result.action.value,
                "policy.latency_ms": result.latency_ms,
                "policy.applied_count": len(result.applied_policies)
            })
            
        return result
    
    async def _check_rbac(self, context: PolicyContext) -> PolicyResult:
        """Check role-based access control"""
        required_roles = self.rbac_map.get(context.action, set())
        
        if not required_roles:
            # No RBAC requirement for this action
            return PolicyResult(
                action=PolicyAction.ALLOW,
                reasons=["No RBAC requirement"],
                applied_policies=["rbac_check"]
            )
            
        if context.roles & required_roles:
            return PolicyResult(
                action=PolicyAction.ALLOW,
                reasons=[f"User has required role from: {required_roles}"],
                applied_policies=["rbac_check"]
            )
            
        return PolicyResult(
            action=PolicyAction.DENY,
            reasons=[f"Missing required roles: {required_roles - context.roles}"],
            applied_policies=["rbac_check"]
        )
    
    async def _check_separation_of_duties(self, context: PolicyContext) -> PolicyResult:
        """Check separation of duties constraints"""
        for role1, role2 in self.separation_of_duties:
            if role1 in context.roles and role2 in context.roles:
                return PolicyResult(
                    action=PolicyAction.DENY,
                    reasons=[f"Separation of duties violation: {role1} and {role2}"],
                    applied_policies=["separation_of_duties"]
                )
                
        return PolicyResult(
            action=PolicyAction.ALLOW,
            reasons=["No separation of duties violation"],
            applied_policies=["separation_of_duties"]
        )
    
    async def _check_data_policies(self, context: PolicyContext) -> PolicyResult:
        """Check data classification and governance policies"""
        result = PolicyResult(
            action=PolicyAction.ALLOW,
            reasons=[],
            applied_policies=["data_governance"]
        )
        
        # Check PII handling
        if DataClassification.PII in context.data_tags:
            if context.environment != "production":
                # Require redaction in non-prod
                result.action = PolicyAction.ALLOW_WITH_REDACTION
                result.redactions = self._generate_redactions(context)
                result.reasons.append("PII redacted in non-production")
                
            # Check if user has PII access permission
            if "pii_reader" not in context.roles:
                result.action = PolicyAction.DENY
                result.reasons.append("User lacks PII reader role")
                
        # Check HIPAA PHI
        if DataClassification.HIPAA_PHI in context.data_tags:
            if "hipaa_certified" not in context.roles:
                result.action = PolicyAction.DENY
                result.reasons.append("User not HIPAA certified")
                
            # Require additional audit
            result.audit_record["hipaa_access"] = True
            
        # Check data retention
        for tag in context.data_tags:
            if tag in self.data_retention_matrix:
                retention = self.data_retention_matrix[tag]
                result.audit_record[f"retention_{tag.value}"] = retention
                
        return result
    
    async def _evaluate_rule(self, rule: PolicyRule, context: PolicyContext) -> bool:
        """Evaluate if a rule matches the context"""
        conditions = rule.conditions
        
        # Check user conditions
        if "user_id" in conditions:
            if context.user_id not in conditions["user_id"]:
                return False
                
        # Check role conditions
        if "roles" in conditions:
            required = set(conditions["roles"].get("any", []))
            if required and not (context.roles & required):
                return False
                
            required_all = set(conditions["roles"].get("all", []))
            if required_all and not required_all.issubset(context.roles):
                return False
                
        # Check resource patterns
        if "resource_pattern" in conditions:
            pattern = conditions["resource_pattern"]
            if not re.match(pattern, context.resource):
                return False
                
        # Check data tags
        if "data_tags" in conditions:
            required_tags = set(DataClassification[t] for t in conditions["data_tags"])
            if not required_tags.issubset(context.data_tags):
                return False
                
        # Check cost threshold
        if "max_cost_usd" in conditions:
            if context.cost_usd > conditions["max_cost_usd"]:
                return False
                
        # Check time-based conditions
        if "time_window" in conditions:
            current_hour = datetime.now().hour
            allowed_hours = conditions["time_window"].get("hours", [])
            if allowed_hours and current_hour not in allowed_hours:
                return False
                
        return True
    
    def _generate_hitl_requirement(
        self, 
        rule: PolicyRule, 
        context: PolicyContext
    ) -> Dict[str, Any]:
        """Generate HITL approval requirement"""
        return {
            "approval_id": hashlib.sha256(
                f"{context.trace_id}{rule.id}".encode()
            ).hexdigest()[:12],
            "rule_id": rule.id,
            "rule_name": rule.name,
            "user_id": context.user_id,
            "action": context.action,
            "resource": context.resource,
            "required_approvers": rule.metadata.get("approvers", ["security_team"]),
            "timeout_seconds": rule.metadata.get("approval_timeout", 300),
            "approval_link": f"https://approvals.example.com/req/{context.trace_id}",
            "created_at": datetime.utcnow().isoformat()
        }
    
    def _generate_redactions(self, context: PolicyContext) -> Dict[str, str]:
        """Generate redaction mappings for sensitive data"""
        redactions = {}
        
        # Generate consistent hashes for PII fields
        pii_fields = ["email", "phone", "ssn", "credit_card", "name", "address"]
        for field in pii_fields:
            if field in context.request_metadata:
                value = str(context.request_metadata[field])
                redaction_hash = hashlib.sha256(
                    f"{value}{context.tenant_id}".encode()
                ).hexdigest()[:8]
                redactions[field] = f"REDACTED_{field.upper()}_{redaction_hash}"
                
        return redactions
    
    def _log_dry_run(self, rule: PolicyRule, context: PolicyContext, result: PolicyResult):
        """Log what would happen in dry-run mode"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "dry_run": True,
            "rule_id": rule.id,
            "rule_name": rule.name,
            "would_action": rule.effect.value,
            "user_id": context.user_id,
            "tenant_id": context.tenant_id,
            "action": context.action,
            "resource": context.resource,
            "trace_id": context.trace_id
        }
        
        # Emit to telemetry
        span = trace.get_current_span()
        span.add_event("policy_dry_run", attributes=event)
        
        # Log for analysis
        print(f"DRY_RUN: {json.dumps(event)}")


class AuditLogger:
    """Handles audit logging for policy decisions"""
    
    async def log(
        self, 
        context: PolicyContext, 
        result: PolicyResult, 
        dry_run: bool
    ) -> Dict[str, Any]:
        """Create audit log entry"""
        audit_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "trace_id": context.trace_id,
            "user_id": context.user_id,
            "tenant_id": context.tenant_id,
            "action": context.action,
            "resource": context.resource,
            "decision": result.action.value,
            "reasons": result.reasons,
            "applied_policies": result.applied_policies,
            "dry_run": dry_run,
            "latency_ms": result.latency_ms,
            "environment": context.environment,
            "ip_address": context.ip_address,
            "cost_usd": context.cost_usd
        }
        
        # Add data tags if present
        if context.data_tags:
            audit_record["data_tags"] = [tag.value for tag in context.data_tags]
            
        # Add HITL requirement if present
        if result.hitl_requirement:
            audit_record["hitl_requirement"] = result.hitl_requirement
            
        # Add redactions info if present
        if result.redactions:
            audit_record["redaction_fields"] = list(result.redactions.keys())
            # Store hashes for audit trail
            audit_record["redaction_hashes"] = {
                k: hashlib.sha256(v.encode()).hexdigest()[:8] 
                for k, v in result.redactions.items()
            }
            
        # In production, this would write to a persistent audit log
        # For now, emit to OpenTelemetry
        span = trace.get_current_span()
        span.add_event("policy_audit", attributes={
            k: str(v) if not isinstance(v, (str, int, float, bool)) else v
            for k, v in audit_record.items()
        })
        
        return audit_record


class MetricsCollector:
    """Collects metrics for policy decisions"""
    
    def __init__(self):
        self.decision_counter = defaultdict(int)
        self.latency_histogram = defaultdict(list)
        
    async def record(self, context: PolicyContext, result: PolicyResult):
        """Record metrics for a policy decision"""
        # Count decisions by action
        self.decision_counter[result.action.value] += 1
        
        # Record latency
        self.latency_histogram[context.action].append(result.latency_ms)
        
        # Emit metrics to OpenTelemetry
        span = trace.get_current_span()
        span.set_attributes({
            f"policy.metrics.{result.action.value}_count": 
                self.decision_counter[result.action.value],
            "policy.metrics.latency_ms": result.latency_ms
        })
        
        # Check for anomalies
        if result.latency_ms > 100:  # Alert on slow evaluations
            span.add_event("policy_slow_evaluation", attributes={
                "latency_ms": result.latency_ms,
                "action": context.action,
                "policy_count": len(result.applied_policies)
            })


# Example configuration loader
def load_policy_config() -> Dict[str, Any]:
    """Load policy configuration from file or environment"""
    return {
        "rbac": {
            "capability_roles": {
                "web.search": ["agent_runtime", "developer", "analyst"],
                "fs.read": ["agent_runtime", "developer"],
                "fs.write": ["agent_runtime", "developer", "admin"],
                "repo.commit": ["tech_lead", "sre"],
                "robot.actuate": ["sre", "ops_controller"],
                "db.mutate": ["dba", "admin"],
                "model.deploy": ["ml_engineer", "sre"]
            },
            "separation_of_duties": [
                ["developer", "approver"],
                ["agent_runtime", "approver"],
                ["requester", "reviewer"]
            ]
        },
        "data_retention": {
            "PII": {
                "retention_days": 30,
                "purge_method": "delete",
                "backup_retention_days": 7
            },
            "SENSITIVE": {
                "retention_days": 90,
                "purge_method": "delete",
                "backup_retention_days": 30
            },
            "EXPORT_OK": {
                "retention_days": 365,
                "purge_method": "tombstone",
                "backup_retention_days": 90
            },
            "HIPAA_PHI": {
                "retention_days": 2555,  # 7 years
                "purge_method": "archive",
                "backup_retention_days": 2555
            }
        },
        "rules": [
            {
                "id": "prod_db_access",
                "name": "Production Database Access Control",
                "description": "Restrict production database mutations",
                "conditions": {
                    "resource_pattern": r"^db:production:.*",
                    "roles": {"any": ["developer", "agent_runtime"]}
                },
                "effect": "require_hitl",
                "priority": 200,
                "metadata": {
                    "approvers": ["dba", "tech_lead"],
                    "approval_timeout": 600
                }
            },
            {
                "id": "cost_limit",
                "name": "Cost Threshold Alert",
                "description": "Require approval for expensive operations",
                "conditions": {
                    "max_cost_usd": 100
                },
                "effect": "require_hitl",
                "priority": 150,
                "metadata": {
                    "approvers": ["finance", "tech_lead"]
                }
            },
            {
                "id": "off_hours_restriction",
                "name": "Off-Hours Access Restriction",
                "description": "Restrict certain operations outside business hours",
                "conditions": {
                    "time_window": {
                        "hours": list(range(9, 18))  # 9 AM to 6 PM
                    },
                    "resource_pattern": r"^(prod|staging):.*"
                },
                "effect": "deny",
                "priority": 100,
                "dry_run": True  # Testing in dry-run first
            }
        ]
    }


# Usage example
async def main():
    """Example usage of the policy engine"""
    config = load_policy_config()
    engine = PolicyEngine(config)
    
    # Example context
    context = PolicyContext(
        user_id="user123",
        tenant_id="tenant456",
        roles={"developer", "agent_runtime"},
        action="db.mutate",
        resource="db:production:users",
        data_tags={DataClassification.PII, DataClassification.SENSITIVE},
        trace_id="trace-abc-123",
        environment="production",
        cost_usd=150.0
    )
    
    # Evaluate in dry-run mode first
    dry_run_result = await engine.evaluate(context, dry_run=True)
    print(f"Dry-run result: {dry_run_result}")
    
    # Evaluate for real
    result = await engine.evaluate(context, dry_run=False)
    print(f"Policy result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
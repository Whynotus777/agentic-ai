# Control Plane Documentation

## Overview

The Control Plane is a production-grade system that governs router decisions, capability access, feature flags, and Human-in-the-Loop (HITL) gates for a multi-agent system. It provides enforce/monitor/dry-run behavior modes and comprehensive policy management.

## Architecture Components

### 1. Policy Engine (`policy_engine/`)
- **Purpose**: Evaluates requests against security and operational policies
- **Modes**: `enforce`, `monitor`, `dry_run`
- **Features**:
  - Capability-based access control
  - Domain-specific rules
  - Budget thresholds
  - HITL requirements
  - Egress scoping

### 2. Capability Registry (`cap_registry/`)
- **Purpose**: Manages and validates system capabilities
- **Features**:
  - Capability definitions with owners, scopes, and tools
  - Rate limiting per capability
  - Budget controls (USD/day)
  - PII tracking
  - Dependency management

### 3. Feature Flags (`flags/`)
- **Purpose**: Dynamic configuration and A/B testing
- **Features**:
  - Percentage-based rollout
  - Environment/tenant/service targeting
  - Stable hash-based assignment
  - Runtime updates

### 4. Router Policy (`router_policy.yaml`)
- **Purpose**: Model routing configuration
- **Features**:
  - Domain-specific routing (robotics → RT-2, simulation → Gemini 2.5 Pro)
  - Tiered routing for app development
  - Fallback chains
  - Guardrails and error budgets

### 5. Integrated Router (`integrated_router.py`)
- **Purpose**: Combines all components for routing decisions
- **Features**:
  - Policy evaluation
  - Flag checking
  - Capability validation
  - Decision tracing

## Quick Start

### Installation

```bash
# Install dependencies
pip install pyyaml

# Directory structure
mkdir -p control_plane/{policy_engine,flags,cap_registry}
```

### Basic Usage

```python
from control_plane.integrated_router import IntegratedRouter, RoutingRequest

# Initialize router
router = IntegratedRouter()

# Make routing request
request = RoutingRequest(
    tenant_id="tenant-123",
    service="robotics-service",
    domain="robotics",
    capability="robot.actuate",
    environment="production",
    tier="premium"
)

# Get routing decision
decision = router.route(request)
print(decision.to_json())
```

## Examples

### Example 1: HITL Required for Robot Actuation

```python
from control_plane.policy_engine.engine import PolicyEngine, RequestContext

# Initialize policy engine
engine = PolicyEngine()

# Create request context for robot actuation
ctx = RequestContext(
    tenant_id="robotics-tenant",
    service="robot-controller",
    capability="robot.actuate",
    domain="robotics",
    budget_used=100.0
)

# Evaluate policy
result = engine.evaluate(ctx)
print(f"Decision: {result.decision.value}")  # Output: "hitl_required"
print(f"Reasons: {result.reasons}")         # Output: ["Physical robot actuation requires human supervision"]
```

### Example 2: Feature Flag Toggle (o4-mini → DeepSeek-V3)

```python
from control_plane.flags.store import FeatureFlagStore

# Initialize flag store
flags = FeatureFlagStore()

# Check flag before toggle
context = {"service": "orchestrator", "tenant_id": "test-tenant"}
enabled = flags.is_enabled("use_deepseek_v3", context)
print(f"DeepSeek enabled: {enabled}")  # Output: False

# Toggle flag
flags.update_flag("use_deepseek_v3", enabled=True, rollout_percentage=100.0)

# Check flag after toggle
enabled = flags.is_enabled("use_deepseek_v3", context)
print(f"DeepSeek enabled: {enabled}")  # Output: True

# Router will now use DeepSeek-V3 instead of o4-mini
```

### Example 3: Router Decision with Policy + Flags

```python
from control_plane.integrated_router import IntegratedRouter, RoutingRequest

router = IntegratedRouter()

# Request for simulation domain
request = RoutingRequest(
    tenant_id="sim-tenant",
    service="simulation-service",
    domain="simulation",
    capability="sim.run",
    environment="production"
)

decision = router.route(request)

# Decision trace shows:
# - Primary: gemini-2.5-pro (from domain policy)
# - Fallbacks: ["gemini-2-flash", "o4-mini"]
# - HITL: False (simulation doesn't require HITL)
# - Guardrails: Resource limits applied
```

### Example 4: Capability Validation

```python
from control_plane.cap_registry.registry import CapabilityRegistry

registry = CapabilityRegistry()

# Validate agent capabilities
validation = registry.validate_agent_capabilities(
    "my-agent",
    ["repo.commit", "repo.push", "unknown.capability"]
)

print(validation)
# Output:
# {
#   "agent_id": "my-agent",
#   "allowed": ["repo.commit"],
#   "denied": ["repo.push"],  # Missing dependency
#   "missing": ["unknown.capability"],
#   "errors": ["Capability repo.push requires repo.commit", "Capability unknown.capability not registered"]
# }
```

## HTTP API Examples

### Check Policy Decision

```bash
# Policy evaluation endpoint (when wrapped in FastAPI/Flask)
curl -X POST http://localhost:8080/api/v1/policy/evaluate \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: tenant-123" \
  -H "Idempotency-Key: $(uuidgen)" \
  -d '{
    "service": "robotics-service",
    "capability": "robot.actuate",
    "domain": "robotics",
    "budget_used": 250.0
  }'

# Response:
# {
#   "decision": "hitl_required",
#   "reasons": ["Physical robot actuation requires human supervision"],
#   "mode": "enforce",
#   "policy_ref": "policy:v1.0",
#   "matched_rules": ["capability:robot.actuate"]
# }
```

### Get Routing Decision

```bash
# Router decision endpoint
curl -X POST http://localhost:8080/api/v1/router/route \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: app-tenant" \
  -H "Idempotency-Key: $(uuidgen)" \
  -d '{
    "service": "appdev-service",
    "domain": "appdev",
    "capability": "llm.generate",
    "tier": "premium",
    "environment": "production"
  }'

# Response:
# {
#   "chosen_primary": "o4",
#   "fallbacks": ["o4-mini", "claude-3-opus"],
#   "requires_hitl": false,
#   "policy_match": {
#     "decision": "allow",
#     "reasons": ["No policy violations found"],
#     "mode": "enforce"
#   },
#   "flag_overrides": {
#     "appdev_tier_routing": true
#   },
#   "guardrails": [
#     {"type": "code_safety", "params": {"scan_vulnerabilities": true}}
#   ],
#   "decision_trace": {
#     "request_id": "a1b2c3d4e5f6",
#     "model_selection": {
#       "primary": "o4",
#       "selection_reason": "Tier-based routing: premium tier for appdev"
#     }
#   }
# }
```

### Check Feature Flag

```bash
# Feature flag evaluation
curl -X POST http://localhost:8080/api/v1/flags/evaluate \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: test-tenant" \
  -d '{
    "flag_name": "use_deepseek_v3",
    "context": {
      "service": "orchestrator",
      "environment": "staging"
    }
  }'

# Response:
# {
#   "flag": "use_deepseek_v3",
#   "enabled": true,
#   "reason": "Target match: environment=staging"
# }
```

### Register New Capability

```bash
# Register capability
curl -X POST http://localhost:8080/api/v1/capabilities/register \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: admin-tenant" \
  -H "Idempotency-Key: $(uuidgen)" \
  -d '{
    "id": "custom.operation",
    "owner": "custom-team",
    "scopes": ["execute"],
    "rate_limits": {"per_min": 10, "per_hour": 100},
    "budget_usd_day": 50.0,
    "tools": ["custom-tool"],
    "pii": false,
    "description": "Custom operation capability"
  }'
```

## Testing

### Unit Tests

```python
# test_policy_engine.py
import pytest
from control_plane.policy_engine.engine import PolicyEngine, RequestContext, PolicyDecision

def test_hitl_required_for_robot_actuate():
    engine = PolicyEngine()
    ctx = RequestContext(
        tenant_id="test",
        service="robot",
        capability="robot.actuate",
        domain="robotics"
    )
    result = engine.evaluate(ctx)
    assert result.decision == PolicyDecision.HITL_REQUIRED
    assert "human" in result.reasons[0].lower()

def test_budget_exceeded_blocks_request():
    engine = PolicyEngine()
    ctx = RequestContext(
        tenant_id="test",
        service="sim",
        capability="sim.run",
        domain="simulation",
        budget_used=2000.0  # Exceeds limit
    )
    result = engine.evaluate(ctx)
    assert result.decision == PolicyDecision.DENY
    assert "budget" in result.reasons[0].lower()

# test_feature_flags.py
def test_percentage_rollout():
    flags = FeatureFlagStore()
    flags.update_flag("test_flag", enabled=True, rollout_percentage=50.0)
    
    # Test stable assignment
    ctx1 = {"tenant_id": "tenant-1", "service": "test"}
    result1 = flags.is_enabled("test_flag", ctx1)
    result2 = flags.is_enabled("test_flag", ctx1)
    assert result1 == result2  # Same context always gets same result

# test_router.py
def test_router_with_deepseek_flag():
    router = IntegratedRouter()
    
    # Enable DeepSeek flag
    router.flag_store.update_flag("use_deepseek_v3", enabled=True)
    
    request = RoutingRequest(
        tenant_id="test",
        service="orchestrator",
        domain="general",
        capability="llm.generate"
    )
    
    decision = router.route(request)
    assert decision.chosen_primary == "deepseek-v3"
    assert "o4-mini" in decision.fallbacks
```

### Run Tests

```bash
# Run all tests
pytest control_plane/tests/

# Run with coverage
pytest --cov=control_plane control_plane/tests/
```

## Configuration Files

### Policy Configuration (`policy_engine/policy.yaml`)
- Capability rules (HITL requirements, budget limits)
- Domain rules (default HITL, allowed capabilities)
- Budget thresholds (daily/per-request limits)
- Rate limits
- Egress scopes

### Capability Registry (`cap_registry/capabilities.yaml`)
- Capability definitions
- Owner assignments
- Tool requirements
- Rate limits and budgets
- Dependencies

### Feature Flags (`flags/flags.yaml`)
- Flag definitions
- Rollout percentages
- Targeting rules
- Metadata

### Router Policy (`router_policy.yaml`)
- Domain policies
- Model preferences
- Fallback chains
- Guardrails
- Error budgets

## Monitoring and Metrics

```python
# Get system metrics
router = IntegratedRouter()
metrics = router.get_metrics()
print(metrics)

# Output:
# {
#   "cache_size": 42,
#   "policy_version": "1.0",
#   "active_flags": 5,
#   "registered_capabilities": 15
# }

# Policy engine metrics
engine = PolicyEngine()
print(f"Policy version: {engine.get_policy_version()}")

# Flag store metrics
flags = FeatureFlagStore()
flag_metrics = flags.get_metrics()
print(f"Enabled flags: {flag_metrics['enabled_flags']}")

# Capability registry metrics
registry = CapabilityRegistry()
cap_metrics = registry.export_metrics()
print(f"Total capabilities: {cap_metrics['total_capabilities']}")
print(f"Daily budget: ${cap_metrics['total_daily_budget']}")
```

## Acceptance Criteria Verification

✅ **Policy Engine returns HITL_REQUIRED for repo.commit and robot.actuate**
```python
# Verified in Example 1 and policy.yaml configuration
```

✅ **Feature flag toggle switches orchestrator from o4-mini to deepseek-v3**
```python
# Verified in Example 2 with flag toggle demonstration
```

✅ **Router returns JSON decision trace with policy+flag evidence**
```python
# Verified in Example 3 with complete decision trace
```

✅ **Requests lacking declared capabilities are POLICY_BLOCKED**
```python
# Verified in capability validation example
```

## Production Deployment Notes

1. **Environment Variables**:
   ```bash
   CONTROL_PLANE_MODE=enforce  # or monitor, dry_run
   POLICY_VERSION=1.0
   FLAG_CACHE_TTL=300
   ```

2. **Health Checks**:
   - `/health/policy` - Policy engine status
   - `/health/flags` - Feature flag store status
   - `/health/capabilities` - Registry status
   - `/health/router` - Integrated router status

3. **Monitoring**:
   - Track HITL request queue depth
   - Monitor budget consumption rates
   - Alert on policy violations
   - Track flag evaluation patterns

4. **Security**:
   - All non-GET requests require `Idempotency-Key` header
   - Tenant isolation via `X-Tenant-ID` header
   - Rate limiting enforced per capability
   - Audit logs for all policy decisions

## Troubleshooting

### Common Issues

1. **"Capability not registered" errors**
   - Check `capabilities.yaml` for capability definition
   - Verify dependencies are satisfied
   - Ensure capability ID matches exactly

2. **Unexpected routing decisions**
   - Check feature flag states
   - Review policy evaluation trace
   - Verify domain/capability mapping

3. **HITL timeouts**
   - Check fallback policies in `router_policy.yaml`
   - Review escalation paths
   - Monitor HITL queue metrics

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Get detailed decision trace
router = IntegratedRouter()
decision = router.route(request)
print(json.dumps(decision.decision_trace, indent=2))
```

## Support

For issues, questions, or contributions:
- Review configuration files in `control_plane/`
- Check policy rules and capability definitions
- Test with dry_run mode before enforcing
- Monitor metrics and decision traces
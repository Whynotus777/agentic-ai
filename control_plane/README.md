# control_plane/README.md
# EXTENDED: Decision trace examples with policy_match, flag_overrides, chosen_primary, fallbacks, and guardrails

## Control Plane Overview

The control plane manages routing decisions, policy enforcement, feature flags, and failover logic across the system.

## Decision Trace Examples

### What is a Decision Trace?

A decision trace captures the complete decision-making path taken by the control plane when processing a request. It includes policy evaluations, feature flag checks, primary/fallback selections, and guardrail enforcement.

### Example 1: Standard Request with Policy Match

```json
{
  "decision_trace": {
    "trace_id": "8e4f5a6b7c8d9e0f1a2b3c4d5e6f7890",
    "timestamp": "2024-01-15T10:30:00.123Z",
    "request_id": "req_789",
    "tenant_id": "tenant_456",
    "decision_path": [
      {
        "stage": "policy_evaluation",
        "timestamp": "2024-01-15T10:30:00.124Z",
        "policies_checked": [
          {
            "policy_id": "pol_rate_limit",
            "result": "PASS",
            "details": {
              "current_rate": 45,
              "limit": 100,
              "window": "1m"
            }
          },
          {
            "policy_id": "pol_data_classification",
            "result": "PASS",
            "details": {
              "required_clearance": "L2",
              "user_clearance": "L3"
            }
          },
          {
            "policy_id": "pol_cost_budget",
            "result": "PASS",
            "details": {
              "daily_spent": 234.56,
              "daily_limit": 1000.00,
              "remaining": 765.44
            }
          }
        ],
        "policy_match": true,
        "matched_policy": "pol_standard_routing",
        "cost_usd": 0.00001
      },
      {
        "stage": "feature_flags",
        "timestamp": "2024-01-15T10:30:00.125Z",
        "flags_evaluated": [
          {
            "flag_name": "enable_new_llm_model",
            "value": true,
            "source": "tenant_override"
          },
          {
            "flag_name": "use_caching_layer",
            "value": true,
            "source": "global_default"
          },
          {
            "flag_name": "enable_cost_optimization",
            "value": false,
            "source": "user_override"
          }
        ],
        "flag_overrides": {
          "enable_new_llm_model": {
            "default": false,
            "override": true,
            "reason": "tenant_beta_program"
          }
        },
        "cost_usd": 0.00001
      },
      {
        "stage": "routing_decision",
        "timestamp": "2024-01-15T10:30:00.126Z",
        "available_backends": [
          {
            "name": "primary_llm_v2",
            "health": "healthy",
            "load": 0.65,
            "latency_p50_ms": 120
          },
          {
            "name": "primary_llm_v1",
            "health": "healthy",
            "load": 0.80,
            "latency_p50_ms": 150
          },
          {
            "name": "fallback_llm",
            "health": "healthy",
            "load": 0.30,
            "latency_p50_ms": 200
          }
        ],
        "chosen_primary": "primary_llm_v2",
        "reason": "feature_flag_override",
        "fallback_order": ["primary_llm_v1", "fallback_llm"],
        "cost_usd": 0.00002
      },
      {
        "stage": "guardrails",
        "timestamp": "2024-01-15T10:30:00.127Z",
        "guardrails_checked": [
          {
            "guardrail_id": "gr_prompt_injection",
            "result": "PASS",
            "confidence": 0.98
          },
          {
            "guardrail_id": "gr_pii_detection",
            "result": "PASS",
            "pii_found": false
          },
          {
            "guardrail_id": "gr_content_filter",
            "result": "PASS",
            "categories_checked": ["violence", "harassment", "illegal"]
          }
        ],
        "all_passed": true,
        "cost_usd": 0.00003
      }
    ],
    "final_decision": {
      "action": "ROUTE",
      "target": "primary_llm_v2",
      "policies_applied": ["pol_standard_routing"],
      "flags_applied": ["enable_new_llm_model"],
      "guardrails_passed": 3,
      "total_decision_time_ms": 3,
      "total_cost_usd": 0.00007
    }
  }
}
```

### Example 2: Fallback Triggered by Primary Failure

```json
{
  "decision_trace": {
    "trace_id": "9f5a6b7c8d9e0f1a2b3c4d5e6f789012",
    "timestamp": "2024-01-15T11:00:00.456Z",
    "request_id": "req_890",
    "tenant_id": "tenant_789",
    "decision_path": [
      {
        "stage": "policy_evaluation",
        "timestamp": "2024-01-15T11:00:00.457Z",
        "policies_checked": [
          {
            "policy_id": "pol_high_availability",
            "result": "PASS",
            "details": {
              "required_availability": 0.99,
              "current_availability": 0.995
            }
          }
        ],
        "policy_match": true,
        "matched_policy": "pol_high_availability",
        "cost_usd": 0.00001
      },
      {
        "stage": "routing_decision",
        "timestamp": "2024-01-15T11:00:00.458Z",
        "available_backends": [
          {
            "name": "primary_service",
            "health": "degraded",
            "load": 0.95,
            "latency_p50_ms": 500,
            "error_rate": 0.15
          },
          {
            "name": "secondary_service",
            "health": "healthy",
            "load": 0.60,
            "latency_p50_ms": 180
          }
        ],
        "chosen_primary": "primary_service",
        "initial_attempt": {
          "status": "failed",
          "error": "timeout_exceeded",
          "duration_ms": 5000
        },
        "fallback_triggered": true,
        "fallback_reason": "primary_timeout",
        "chosen_fallback": "secondary_service",
        "cost_usd": 0.00005
      },
      {
        "stage": "guardrails",
        "timestamp": "2024-01-15T11:00:05.459Z",
        "guardrails_checked": [
          {
            "guardrail_id": "gr_response_validation",
            "result": "PASS",
            "validated_fields": ["status", "data", "metadata"]
          }
        ],
        "all_passed": true,
        "cost_usd": 0.00002
      }
    ],
    "final_decision": {
      "action": "ROUTE_WITH_FALLBACK",
      "primary_target": "primary_service",
      "actual_target": "secondary_service",
      "fallback_reason": "primary_timeout",
      "retry_attempts": 1,
      "total_decision_time_ms": 5003,
      "total_cost_usd": 0.00008
    }
  }
}
```

### Example 3: Request Blocked by Guardrails

```json
{
  "decision_trace": {
    "trace_id": "af6b7c8d9e0f1a2b3c4d5e6f78901234",
    "timestamp": "2024-01-15T12:00:00.789Z",
    "request_id": "req_901",
    "tenant_id": "tenant_012",
    "decision_path": [
      {
        "stage": "policy_evaluation",
        "timestamp": "2024-01-15T12:00:00.790Z",
        "policies_checked": [
          {
            "policy_id": "pol_rate_limit",
            "result": "PASS"
          }
        ],
        "policy_match": true,
        "cost_usd": 0.00001
      },
      {
        "stage": "guardrails",
        "timestamp": "2024-01-15T12:00:00.791Z",
        "guardrails_checked": [
          {
            "guardrail_id": "gr_prompt_injection",
            "result": "FAIL",
            "confidence": 0.92,
            "detected_patterns": [
              "ignore_previous_instructions",
              "system_prompt_override"
            ]
          },
          {
            "guardrail_id": "gr_pii_detection",
            "result": "FAIL",
            "pii_found": true,
            "pii_types": ["ssn", "credit_card"],
            "locations": ["input.text[45:56]", "input.text[78:94]"]
          }
        ],
        "all_passed": false,
        "blocking_guardrails": ["gr_prompt_injection", "gr_pii_detection"],
        "cost_usd": 0.00003
      }
    ],
    "final_decision": {
      "action": "BLOCK",
      "reason": "guardrails_failed",
      "failed_guardrails": ["gr_prompt_injection", "gr_pii_detection"],
      "error_response": {
        "code": "GUARDRAIL_VIOLATION",
        "message": "Request blocked due to security concerns",
        "details": {
          "prompt_injection_detected": true,
          "pii_detected": true
        }
      },
      "total_decision_time_ms": 1,
      "total_cost_usd": 0.00004,
      "siem_event_sent": true
    }
  }
}
```

### Example 4: Complex Multi-Stage Decision with Override

```json
{
  "decision_trace": {
    "trace_id": "bf7c8d9e0f1a2b3c4d5e6f7890123456",
    "timestamp": "2024-01-15T13:00:00.123Z",
    "request_id": "req_234",
    "tenant_id": "tenant_vip",
    "user_id": "user_567",
    "decision_path": [
      {
        "stage": "policy_evaluation",
        "timestamp": "2024-01-15T13:00:00.124Z",
        "policies_checked": [
          {
            "policy_id": "pol_vip_tenant",
            "result": "MATCH",
            "priority": 100,
            "details": {
              "tenant_tier": "platinum",
              "sla_guaranteed": 0.999,
              "dedicated_resources": true
            }
          },
          {
            "policy_id": "pol_cost_budget",
            "result": "OVERRIDE",
            "details": {
              "daily_spent": 2456.78,
              "daily_limit": 1000.00,
              "override_reason": "vip_tenant_unlimited"
            }
          }
        ],
        "policy_match": true,
        "matched_policy": "pol_vip_tenant",
        "policy_overrides": ["pol_cost_budget"],
        "cost_usd": 0.00001
      },
      {
        "stage": "feature_flags",
        "timestamp": "2024-01-15T13:00:00.125Z",
        "flags_evaluated": [
          {
            "flag_name": "enable_premium_models",
            "value": true,
            "source": "tenant_tier"
          },
          {
            "flag_name": "bypass_rate_limits",
            "value": true,
            "source": "vip_override"
          },
          {
            "flag_name": "enable_dedicated_pool",
            "value": true,
            "source": "policy_requirement"
          }
        ],
        "flag_overrides": {
          "bypass_rate_limits": {
            "default": false,
            "override": true,
            "reason": "vip_tenant_policy"
          }
        },
        "cost_usd": 0.00001
      },
      {
        "stage": "routing_decision",
        "timestamp": "2024-01-15T13:00:00.126Z",
        "routing_strategy": "dedicated_pool",
        "available_backends": [
          {
            "name": "vip_dedicated_pool_1",
            "health": "healthy",
            "load": 0.20,
            "reserved_for": "tenant_vip"
          },
          {
            "name": "vip_dedicated_pool_2",
            "health": "healthy",
            "load": 0.15,
            "reserved_for": "tenant_vip"
          }
        ],
        "chosen_primary": "vip_dedicated_pool_1",
        "reason": "lowest_load_dedicated",
        "fallback_order": ["vip_dedicated_pool_2", "premium_shared_pool"],
        "guaranteed_capacity": true,
        "cost_usd": 0.0001
      },
      {
        "stage": "guardrails",
        "timestamp": "2024-01-15T13:00:00.127Z",
        "guardrails_mode": "audit_only",
        "guardrails_checked": [
          {
            "guardrail_id": "gr_content_filter",
            "result": "WARN",
            "action": "LOG_ONLY",
            "reason": "vip_tenant_audit_mode"
          }
        ],
        "all_passed": true,
        "audit_logged": true,
        "cost_usd": 0.00002
      }
    ],
    "final_decision": {
      "action": "ROUTE_PREMIUM",
      "target": "vip_dedicated_pool_1",
      "routing_class": "dedicated",
      "policies_applied": ["pol_vip_tenant"],
      "policies_overridden": ["pol_cost_budget"],
      "flags_applied": ["enable_premium_models", "bypass_rate_limits", "enable_dedicated_pool"],
      "guardrails_mode": "audit_only",
      "sla_guarantee": 0.999,
      "total_decision_time_ms": 3,
      "total_cost_usd": 0.00014
    }
  }
}
```

## Decision Trace Schema

### Core Fields

| Field | Type | Description |
|-------|------|-------------|
| `trace_id` | string | Unique identifier for the decision trace |
| `timestamp` | ISO8601 | When the decision process started |
| `request_id` | string | Original request being processed |
| `tenant_id` | string | Tenant making the request |
| `decision_path` | array | Ordered list of decision stages |
| `final_decision` | object | Summary of the final routing decision |

### Decision Stage Fields

| Field | Type | Description |
|-------|------|-------------|
| `stage` | string | Stage name (policy_evaluation, feature_flags, routing_decision, guardrails) |
| `timestamp` | ISO8601 | When this stage started |
| `cost_usd` | float | Cost incurred in this stage |
| Additional fields vary by stage type |

### Policy Evaluation Fields

| Field | Type | Description |
|-------|------|-------------|
| `policies_checked` | array | List of policies evaluated |
| `policy_match` | boolean | Whether a policy matched |
| `matched_policy` | string | ID of the matched policy |
| `policy_overrides` | array | Policies that were overridden |

### Routing Decision Fields

| Field | Type | Description |
|-------|------|-------------|
| `available_backends` | array | List of available backend services |
| `chosen_primary` | string | Selected primary backend |
| `fallback_order` | array | Ordered list of fallback options |
| `fallback_triggered` | boolean | Whether fallback was used |
| `fallback_reason` | string | Why fallback was triggered |

### Guardrails Fields

| Field | Type | Description |
|-------|------|-------------|
| `guardrails_checked` | array | List of guardrails evaluated |
| `all_passed` | boolean | Whether all guardrails passed |
| `blocking_guardrails` | array | Guardrails that blocked the request |

## Querying Decision Traces

### Find Requests Using Fallbacks
```sql
SELECT 
  trace_id,
  request_id,
  tenant_id,
  decision_path->>'$.final_decision.fallback_reason' as fallback_reason,
  decision_path->>'$.final_decision.total_cost_usd' as cost
FROM decision_traces
WHERE decision_path->>'$.final_decision.action' = 'ROUTE_WITH_FALLBACK'
  AND timestamp > NOW() - INTERVAL '1 hour';
```

### Find Blocked Requests by Guardrails
```sql
SELECT 
  trace_id,
  tenant_id,
  decision_path->>'$.final_decision.failed_guardrails' as failed_guardrails,
  COUNT(*) as blocked_count
FROM decision_traces
WHERE decision_path->>'$.final_decision.action' = 'BLOCK'
  AND timestamp > NOW() - INTERVAL '24 hours'
GROUP BY tenant_id, failed_guardrails;
```

### Analyze Policy Override Patterns
```sql
SELECT 
  tenant_id,
  decision_path->>'$.decision_path[0].policies_overridden' as overridden_policies,
  COUNT(*) as override_count,
  AVG(CAST(decision_path->>'$.final_decision.total_cost_usd' AS FLOAT)) as avg_cost
FROM decision_traces
WHERE decision_path->>'$.decision_path[0].policy_overrides' IS NOT NULL
GROUP BY tenant_id, overridden_policies
ORDER BY override_count DESC;
```
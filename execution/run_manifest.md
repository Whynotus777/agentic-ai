# Run Manifest Schema

## Overview
The run manifest is an immutable record of a task execution, enabling replay without side-effect duplication and providing audit trail for compliance.

## Schema Definition

```typescript
interface RunManifest {
  // Identifiers
  run_id: string;                    // Unique execution identifier
  tenant_id: string;                 // Tenant isolation boundary
  idempotency_key: string;          // Deduplication key
  trace_id: string;                 // Distributed trace correlation
  correlation_id: string;           // Request correlation
  
  // Execution Details
  inputs_sha: string;               // SHA256 of input payload
  executed_at: string;              // ISO8601 timestamp
  attempt: number;                  // Retry attempt number (0-based)
  replayed: boolean;                // True if replayed from manifest
  
  // Tools & Side Effects
  tools: ToolInvocation[];          // External tool calls made
  artifacts: Artifact[];            // Generated artifacts
  outputs: Record<string, any>;     // Execution outputs
  
  // Security & Compliance
  signatures: Signature[];          // Cryptographic signatures
  policy_mode: PolicyMode;          // Execution policy applied
  approvals: Approval[];            // Required approvals
  
  // Metadata
  metadata: {
    message_id: string;
    priority: number;
    submitted_at: string;
    execution_duration_ms?: number;
    worker_id?: string;
  };
}

interface ToolInvocation {
  tool_name: string;
  invoked_at: string;
  parameters: Record<string, any>;
  result: {
    status: "success" | "failure";
    data?: any;
    error?: string;
  };
  idempotency_token?: string;      // For tool-level idempotency
}

interface Artifact {
  artifact_id: string;
  type: "file" | "database" | "api_response" | "state_change";
  location: string;                 // URI or path
  size_bytes?: number;
  checksum?: string;
  created_at: string;
  metadata?: Record<string, any>;
}

interface Signature {
  signer: string;                   // Identity of signer
  algorithm: "RS256" | "ES256";
  signature: string;                 // Base64 encoded
  signed_at: string;
  scope: string[];                  // What was signed
}

type PolicyMode = "standard" | "strict" | "audit" | "bypass";

interface Approval {
  approver: string;
  approved_at: string;
  approval_type: "manual" | "automated" | "policy";
  reason?: string;
}
```

## JSON Example - Standard Execution

```json
{
  "run_id": "run_a1b2c3d4e5f6",
  "tenant_id": "tenant-001",
  "idempotency_key": "process-order-12345",
  "trace_id": "trace_9f8e7d6c5b4a",
  "correlation_id": "req_xyz789",
  
  "inputs_sha": "3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c",
  "executed_at": "2025-09-05T14:30:00.000Z",
  "attempt": 0,
  "replayed": false,
  
  "tools": [
    {
      "tool_name": "payment_processor",
      "invoked_at": "2025-09-05T14:30:01.123Z",
      "parameters": {
        "amount": 99.99,
        "currency": "USD",
        "payment_method": "card_ending_4242"
      },
      "result": {
        "status": "success",
        "data": {
          "transaction_id": "tx_abc123",
          "authorization_code": "AUTH789"
        }
      },
      "idempotency_token": "pay_idem_xyz"
    },
    {
      "tool_name": "inventory_service",
      "invoked_at": "2025-09-05T14:30:02.456Z",
      "parameters": {
        "action": "reserve",
        "items": [
          {"sku": "WIDGET-001", "quantity": 2}
        ]
      },
      "result": {
        "status": "success",
        "data": {
          "reservation_id": "res_def456",
          "expires_at": "2025-09-05T15:30:02.456Z"
        }
      }
    }
  ],
  
  "artifacts": [
    {
      "artifact_id": "art_invoice_789",
      "type": "file",
      "location": "s3://artifacts/tenant-001/invoices/INV-2025-09-12345.pdf",
      "size_bytes": 245632,
      "checksum": "sha256:4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f",
      "created_at": "2025-09-05T14:30:03.789Z",
      "metadata": {
        "invoice_number": "INV-2025-09-12345",
        "format": "PDF/A-2b"
      }
    }
  ],
  
  "outputs": {
    "order_id": "ORD-2025-09-12345",
    "status": "confirmed",
    "estimated_delivery": "2025-09-08",
    "tracking_number": "1Z999AA1234567890"
  },
  
  "signatures": [
    {
      "signer": "system-executor",
      "algorithm": "RS256",
      "signature": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
      "signed_at": "2025-09-05T14:30:04.000Z",
      "scope": ["inputs", "outputs", "tools"]
    }
  ],
  
  "policy_mode": "standard",
  "approvals": [],
  
  "metadata": {
    "message_id": "msg_qrs456",
    "priority": 5,
    "submitted_at": "2025-09-05T14:29:55.000Z",
    "execution_duration_ms": 4125,
    "worker_id": "worker-2"
  }
}
```

## JSON Example - Replayed Execution

```json
{
  "run_id": "run_replay_7g8h9i",
  "tenant_id": "tenant-001",
  "idempotency_key": "process-order-12345",
  "trace_id": "trace_replay_3c4d5e",
  "correlation_id": "req_replay_mno",
  
  "inputs_sha": "3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c",
  "executed_at": "2025-09-05T15:45:00.000Z",
  "attempt": 2,
  "replayed": true,
  
  "tools": [],
  
  "artifacts": [
    {
      "artifact_id": "art_invoice_789",
      "type": "file",
      "location": "s3://artifacts/tenant-001/invoices/INV-2025-09-12345.pdf",
      "size_bytes": 245632,
      "checksum": "sha256:4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f",
      "created_at": "2025-09-05T14:30:03.789Z",
      "metadata": {
        "note": "Artifact from original execution",
        "original_run_id": "run_a1b2c3d4e5f6"
      }
    }
  ],
  
  "outputs": {
    "order_id": "ORD-2025-09-12345",
    "status": "confirmed",
    "replayed_from": "run_a1b2c3d4e5f6",
    "note": "Execution replayed from manifest - no side effects repeated"
  },
  
  "signatures": [
    {
      "signer": "replay-validator",
      "algorithm": "ES256",
      "signature": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9...",
      "signed_at": "2025-09-05T15:45:01.000Z",
      "scope": ["replay", "manifest_integrity"]
    }
  ],
  
  "policy_mode": "standard",
  "approvals": [],
  
  "metadata": {
    "message_id": "msg_replay_tuv",
    "priority": 5,
    "submitted_at": "2025-09-05T15:44:58.000Z",
    "execution_duration_ms": 125,
    "worker_id": "worker-1",
    "replay_reason": "worker_failure_recovery"
  }
}
```

## JSON Example - Failed Execution with Compensating Action

```json
{
  "run_id": "run_failed_jkl789",
  "tenant_id": "tenant-002",
  "idempotency_key": "complex-workflow-67890",
  "trace_id": "trace_fail_6f7g8h",
  "correlation_id": "req_fail_pqr",
  
  "inputs_sha": "9f8e7d6c5b4a3b2c1d0e9f8e7d6c5b4a3b2c1d0e9f8e7d6c5b4a3b2c1d0e9f8e",
  "executed_at": "2025-09-05T16:00:00.000Z",
  "attempt": 3,
  "replayed": false,
  
  "tools": [
    {
      "tool_name": "database_transaction",
      "invoked_at": "2025-09-05T16:00:01.234Z",
      "parameters": {
        "operation": "update",
        "table": "inventory",
        "conditions": {"sku": "ITEM-999"}
      },
      "result": {
        "status": "success",
        "data": {
          "rows_affected": 1,
          "transaction_id": "txn_db_456"
        }
      }
    },
    {
      "tool_name": "external_api",
      "invoked_at": "2025-09-05T16:00:02.567Z",
      "parameters": {
        "endpoint": "/v1/process",
        "method": "POST"
      },
      "result": {
        "status": "failure",
        "error": "Connection timeout after 30000ms"
      }
    }
  ],
  
  "artifacts": [],
  
  "outputs": {
    "status": "failed",
    "error": "External API timeout",
    "partial_completion": true,
    "rollback_required": true,
    "compensating_action_ref": "compensate/tenant-002/msg_fail_xyz"
  },
  
  "signatures": [],
  
  "policy_mode": "strict",
  "approvals": [
    {
      "approver": "system-policy",
      "approved_at": "2025-09-05T15:59:59.000Z",
      "approval_type": "policy",
      "reason": "Automatic approval for retry attempt"
    }
  ],
  
  "metadata": {
    "message_id": "msg_fail_xyz",
    "priority": 3,
    "submitted_at": "2025-09-05T15:59:30.000Z",
    "execution_duration_ms": 31800,
    "worker_id": "worker-4",
    "failure_classification": "transient_network",
    "dlq_entry": true
  }
}
```

## Manifest Usage Patterns

### 1. Replay Detection
Before executing, check for existing manifest:
```python
manifest_path = get_manifest_path(tenant_id, run_id)
if manifest_path.exists():
    # Load and return cached results
    return load_manifest(manifest_path)
```

### 2. Audit Trail
All manifests are immutable once written:
```python
# Manifests are write-once
manifest = create_manifest(execution_result)
save_manifest(manifest)  # No updates allowed
```

### 3. Compensating Actions
For failed executions in DLQ:
```python
if execution.failed and execution.attempts >= max_retries:
    manifest["outputs"]["compensating_action_ref"] = create_compensating_action(
        execution, 
        manifest["tools"]  # Include partial progress
    )
```

### 4. Integrity Verification
Signatures ensure manifest hasn't been tampered:
```python
def verify_manifest(manifest):
    for signature in manifest["signatures"]:
        if not verify_signature(signature, manifest):
            raise IntegrityError("Manifest signature invalid")
```

## Storage Layout

```
/var/lib/execution/manifests/
├── tenant-001/
│   ├── run_a1b2c3d4e5f6/
│   │   └── manifest.json
│   ├── run_replay_7g8h9i/
│   │   └── manifest.json
│   └── ...
├── tenant-002/
│   ├── run_failed_jkl789/
│   │   └── manifest.json
│   └── ...
└── compensating_actions/
    ├── tenant-001/
    │   └── msg_xyz.json
    └── tenant-002/
        └── msg_fail_xyz.json
```

## Retention Policy

- **Active Manifests**: Retained for idempotency window (24 hours minimum)
- **Completed Manifests**: 90 days standard retention
- **Failed/DLQ Manifests**: 30 days + compensating action period
- **Compliance Mode**: 7 years with tamper-proof storage
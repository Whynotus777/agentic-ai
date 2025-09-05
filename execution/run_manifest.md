# Run Manifest Specification

## Overview
The run manifest is a persistent record of each execution run, providing audit trail, replay capability, and exactly-once guarantees.

## Schema

```json
{
  "run_id": "string (UUID)",
  "idempotency_key": "string",
  "operation": "string",
  "payload": "object",
  "approvals": "array",
  "policy_mode": "string",
  "created_at": "ISO 8601 timestamp",
  "status": "string",
  "result": "object (optional)",
  "error": "string (optional)",
  "applied_at": "ISO 8601 timestamp (optional)",
  "committed_at": "ISO 8601 timestamp (optional)",
  "failed_at": "ISO 8601 timestamp (optional)",
  "written_at": "ISO 8601 timestamp"
}
```

## Field Descriptions

### Core Fields
- **run_id**: Unique identifier for this execution run (UUID v4)
- **idempotency_key**: Client-provided key for exactly-once semantics
- **operation**: The operation type (commit, actuate, analyze, etc.)
- **payload**: Operation-specific parameters and data
- **created_at**: When the run was initiated
- **written_at**: When the manifest was last written to disk

### Policy Fields (NEW)
- **approvals**: Array of approval records for HITL operations
  - Each approval contains:
    - `approver`: User/system that approved
    - `decision`: approve/reject
    - `timestamp`: When approved
    - `reason`: Optional approval reason
    - `token`: Approval token used
- **policy_mode**: Policy enforcement mode
  - `enforce`: Strict policy enforcement (default)
  - `monitor`: Log violations but allow execution
  - `bypass`: Skip policy checks (requires elevated privileges)

### Status Fields
- **status**: Current execution status
  - `dispatched`: Initial state, queued for execution
  - `applied`: Side effects have been applied
  - `committed`: Successfully committed to ledger
  - `failed`: Execution failed
  - `rolled_back`: Applied but then rolled back

### Result Fields
- **result**: Execution result (when status is committed)
- **error**: Error message (when status is failed)
- **applied_at**: When side effects were applied
- **committed_at**: When execution was committed
- **failed_at**: When execution failed

## Example Manifest

```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "idempotency_key": "client-key-12345",
  "operation": "commit",
  "payload": {
    "repository": "main-repo",
    "branch": "feature/update",
    "files": ["src/main.py", "tests/test_main.py"],
    "message": "Update main functionality"
  },
  "approvals": [
    {
      "approver": "user@example.com",
      "decision": "approve",
      "timestamp": "2024-01-15T10:30:00Z",
      "reason": "Code review passed",
      "token": "approval-token-abc123"
    }
  ],
  "policy_mode": "enforce",
  "created_at": "2024-01-15T10:00:00Z",
  "status": "committed",
  "result": {
    "operation": "commit",
    "executed_at": "2024-01-15T10:31:00Z",
    "side_effects": [
      {
        "type": "git_commit",
        "commit_id": "abc550e8",
        "files_changed": ["src/main.py", "tests/test_main.py"]
      }
    ]
  },
  "applied_at": "2024-01-15T10:31:00Z",
  "committed_at": "2024-01-15T10:31:05Z",
  "written_at": "2024-01-15T10:31:05Z"
}
```

## Replay Semantics

### Exactly-Once Guarantee
When a request with the same `idempotency_key` is received:
1. System checks for existing manifest with that key
2. If found and status is `committed`: Return cached result (no re-execution)
3. If found and status is `failed`: Return cached error
4. If found and status is `dispatched` or `applied`: Attempt recovery

### Crash Recovery
If the system crashes during execution:
1. On restart, scan for incomplete manifests (status: `dispatched` or `applied`)
2. Check apply ledger for commit status
3. If committed in ledger: Update manifest and return result
4. If not committed: Safe to retry operation (no side effects were persisted)

### Write-Through Policy
Manifests are written at multiple points:
1. **On dispatch**: Initial manifest with status `dispatched`
2. **After apply**: Updated with result and status `applied`
3. **After commit**: Final update with status `committed`
4. **On failure**: Updated with error and status `failed`

This ensures the manifest always reflects the current state, even if the process crashes.

## Approval Integration

### HITL Approval Flow
1. Operation requires approval (based on policy)
2. System returns `HITL_REQUIRED` status with approval link
3. Human approver reviews and approves/rejects
4. Approval record added to manifest `approvals` array
5. Operation proceeds if approved

### Approval Record Schema
```json
{
  "approver": "string (email or system ID)",
  "decision": "approve | reject",
  "timestamp": "ISO 8601 timestamp",
  "reason": "string (optional)",
  "token": "string (approval token)"
}
```

## Policy Mode Behavior

### Enforce Mode (Default)
- All policies strictly enforced
- Violations block execution
- Approvals required for risky operations

### Monitor Mode
- Policy violations logged but not blocking
- Useful for testing and gradual rollout
- Approvals still recorded but not required

### Bypass Mode
- Skip all policy checks
- Requires elevated privileges
- Should be used sparingly and audited

## File Storage

Manifests are stored as JSON files:
- Location: `/tmp/manifests/manifest_{run_id}.json`
- Format: Pretty-printed JSON with 2-space indentation
- Retention: Configurable (default: 7 days)

## Usage Examples

### Reading a Manifest
```python
from execution.state_ledger import StateLedger

ledger = StateLedger()
manifest = ledger.get_run_manifest("550e8400-e29b-41d4-a716-446655440000")
print(f"Status: {manifest['status']}")
print(f"Approvals: {manifest['approvals']}")
```

### Writing a Manifest
```python
manifest = {
    "run_id": run_id,
    "idempotency_key": idempotency_key,
    "operation": "commit",
    "payload": payload,
    "approvals": [approval_record],
    "policy_mode": "enforce",
    "created_at": datetime.utcnow().isoformat(),
    "status": "dispatched"
}

ledger.write_run_manifest(run_id, manifest)
```

## Best Practices

1. **Always include approvals**: Even if empty array
2. **Set appropriate policy_mode**: Default to `enforce` for production
3. **Write manifest early**: Before any side effects
4. **Update status atomically**: Use ledger transactions
5. **Include timestamps**: For all state transitions
6. **Preserve idempotency_key**: Essential for replay
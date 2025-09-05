# DSAR (Data Subject Access Request) Runbook

## Overview
This runbook outlines the process for handling Data Subject Access Requests including data export, rectification, and deletion requests under GDPR, CCPA, and other privacy regulations.

## Request Types

### 1. Access Request
User requests a copy of all their personal data.

### 2. Rectification Request
User requests correction of inaccurate personal data.

### 3. Deletion Request (Right to Erasure)
User requests complete removal of their personal data.

### 4. Portability Request
User requests data in a portable format for transfer.

### 5. Restriction Request
User requests limitation on processing of their data.

## Initial Request Handling

### Step 1: Verify Identity
```bash
# Use the identity verification tool
./scripts/verify_identity.sh \
  --email="${USER_EMAIL}" \
  --request_id="${REQUEST_ID}" \
  --verification_method="email_token"
```

### Step 2: Log Request
```yaml
# Create request manifest
request_manifest:
  request_id: "DSAR-2024-001234"
  timestamp: "2024-01-01T00:00:00Z"
  user_id: "usr_abc123"
  email: "user@example.com"
  type: "deletion"  # access|rectification|deletion|portability|restriction
  jurisdiction: "GDPR"  # GDPR|CCPA|other
  deadline: "2024-01-31T23:59:59Z"  # 30 days for GDPR
  status: "verified"
  assigned_to: "privacy_team"
```

### Step 3: Legal Hold Check
```sql
-- Check for active legal holds
SELECT * FROM legal_holds 
WHERE user_id = :user_id 
  AND status = 'active'
  AND end_date > NOW();
```

## Data Discovery Process

### Automated Discovery Script
```bash
#!/bin/bash
# discover_user_data.sh

REQUEST_ID=$1
USER_ID=$2

# Create discovery manifest
cat > /tmp/discovery_${REQUEST_ID}.yaml << EOF
discovery_manifest:
  request_id: "${REQUEST_ID}"
  user_id: "${USER_ID}"
  scan_started: "$(date -Iseconds)"
  systems_to_scan:
    - primary_database
    - analytics_warehouse
    - object_storage
    - cache_layers
    - backup_systems
    - audit_logs
EOF

# Scan each system
for system in ${SYSTEMS[@]}; do
  echo "Scanning $system..."
  ./scanners/${system}_scanner.sh \
    --user_id="${USER_ID}" \
    --output="/tmp/scan_${system}_${REQUEST_ID}.json"
done

# Compile results
./scripts/compile_discovery.sh \
  --request_id="${REQUEST_ID}" \
  --output="purge_manifest_${REQUEST_ID}.yaml"
```

## Purge Manifest Format

```yaml
# purge_manifest_DSAR-2024-001234.yaml
purge_manifest:
  version: "1.0"
  request_id: "DSAR-2024-001234"
  user_id: "usr_abc123"
  created_at: "2024-01-01T12:00:00Z"
  created_by: "privacy_officer@company.com"
  
  data_locations:
    - system: "postgresql_primary"
      database: "users_db"
      tables:
        - name: "users"
          rows: 1
          columns: ["id", "email", "name", "created_at"]
          where_clause: "id = 'usr_abc123'"
        - name: "user_preferences"
          rows: 15
          columns: ["*"]
          where_clause: "user_id = 'usr_abc123'"
      
    - system: "mongodb_analytics"
      database: "analytics"
      collections:
        - name: "events"
          documents: 1847
          filter: {"userId": "usr_abc123"}
        - name: "sessions"
          documents: 234
          filter: {"userId": "usr_abc123"}
    
    - system: "s3_storage"
      bucket: "user-uploads"
      objects:
        - key: "usr_abc123/profile.jpg"
          size_bytes: 1048576
        - key: "usr_abc123/documents/*"
          count: 12
          total_size_bytes: 5242880
    
    - system: "redis_cache"
      keys:
        - pattern: "user:usr_abc123:*"
          count: 8
          ttl_seconds: 3600
    
    - system: "elasticsearch_logs"
      indices:
        - name: "logs-2024.01"
          documents: 523
          query: {"match": {"user_id": "usr_abc123"}}
  
  derived_data:
    - description: "ML model features"
      location: "feature_store"
      retention_override: "anonymize_only"
    
    - description: "Aggregated analytics"
      location: "analytics_warehouse"
      retention_override: "keep_aggregates"
  
  execution_plan:
    order:
      - "redis_cache"        # Clear cache first
      - "elasticsearch_logs" # Remove logs
      - "s3_storage"        # Delete files
      - "mongodb_analytics" # Clean analytics
      - "postgresql_primary" # Finally, remove core data
    
    rollback_enabled: true
    dry_run: true
    parallel_execution: false
    
  approvals:
    - role: "data_protection_officer"
      approved_by: null
      approved_at: null
    - role: "legal_counsel"
      approved_by: null
      approved_at: null
```

## Execution Procedures

### Access Request Execution
```bash
# Generate data export
./scripts/export_user_data.sh \
  --manifest="purge_manifest_${REQUEST_ID}.yaml" \
  --format="json" \
  --encryption="pgp" \
  --recipient="${USER_EMAIL}"
```

### Deletion Request Execution
```bash
# Dry run first
./scripts/execute_deletion.sh \
  --manifest="purge_manifest_${REQUEST_ID}.yaml" \
  --dry-run \
  --verbose

# Review dry run results
cat deletion_dry_run_${REQUEST_ID}.log

# Execute with approval
./scripts/execute_deletion.sh \
  --manifest="purge_manifest_${REQUEST_ID}.yaml" \
  --approval-token="${APPROVAL_TOKEN}" \
  --backup-first \
  --verify-completion
```

### Verification Script
```python
#!/usr/bin/env python3
# verify_deletion.py

import yaml
import sys
from datetime import datetime

def verify_deletion(manifest_file):
    with open(manifest_file, 'r') as f:
        manifest = yaml.safe_load(f)
    
    verification_report = {
        'request_id': manifest['purge_manifest']['request_id'],
        'verification_time': datetime.utcnow().isoformat(),
        'results': []
    }
    
    for location in manifest['purge_manifest']['data_locations']:
        # Check each system
        result = verify_system_deletion(location)
        verification_report['results'].append(result)
    
    # Generate cryptographic proof
    proof = generate_deletion_proof(verification_report)
    verification_report['cryptographic_proof'] = proof
    
    return verification_report

def verify_system_deletion(location):
    # System-specific verification logic
    if location['system'] == 'postgresql_primary':
        return verify_postgresql(location)
    elif location['system'] == 's3_storage':
        return verify_s3(location)
    # ... other systems
    
def generate_deletion_proof(report):
    # Generate SHA-256 hash of deletion confirmation
    import hashlib
    content = json.dumps(report, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()
```

## Compliance Timelines

| Regulation | Request Type | Response Time | Completion Time |
|------------|-------------|---------------|-----------------|
| GDPR | Access | 30 days | 30 days (max 90) |
| GDPR | Deletion | Without delay | 30 days |
| GDPR | Rectification | 30 days | 30 days |
| CCPA | Access | 45 days | 45 days (max 90) |
| CCPA | Deletion | 45 days | 45 days |
| LGPD | All types | 15 days | 15 days |

## Exception Handling

### Legal Hold Exception
```yaml
exception:
  type: "legal_hold"
  request_id: "DSAR-2024-001234"
  reason: "Active litigation - Case #12345"
  data_preserved:
    - All data specified in hold order
  user_notification:
    sent: true
    template: "legal_hold_exception"
  review_date: "2024-06-01"
```

### Technical Impossibility
```yaml
exception:
  type: "technical_impossibility"
  request_id: "DSAR-2024-001234"
  reason: "Data in immutable backup system"
  partial_completion: true
  completed_actions:
    - Deleted from primary systems
    - Deleted from mutable backups
  pending_actions:
    - Immutable backup expiry in 90 days
  user_notification:
    sent: true
    explanation: "detailed_technical_explanation"
```

## Post-Execution Tasks

### 1. User Notification
```bash
# Send completion notification
./scripts/notify_user.sh \
  --request_id="${REQUEST_ID}" \
  --status="completed" \
  --proof="${CRYPTOGRAPHIC_PROOF}" \
  --template="deletion_complete"
```

### 2. Audit Log Entry
```json
{
  "event": "dsar_completed",
  "request_id": "DSAR-2024-001234",
  "user_id": "usr_abc123",
  "type": "deletion",
  "completed_at": "2024-01-15T14:30:00Z",
  "executed_by": "privacy_officer@company.com",
  "systems_affected": ["postgresql", "mongodb", "s3", "redis"],
  "data_removed": {
    "records": 2617,
    "files": 13,
    "size_bytes": 6291456
  },
  "verification_proof": "sha256:abcd1234...",
  "compliance": {
    "regulation": "GDPR",
    "deadline": "2024-01-31T23:59:59Z",
    "sla_met": true
  }
}
```

### 3. Metrics Update
```bash
# Update DSAR metrics
curl -X POST ${METRICS_ENDPOINT}/dsar \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "'${REQUEST_ID}'",
    "completion_time_hours": 72,
    "data_systems_touched": 5,
    "success": true
  }'
```

## Emergency Procedures

### Bulk Request Handling
If receiving >100 requests/day:
1. Activate emergency response team
2. Enable automated processing pipeline
3. Prioritize by regulatory deadline
4. Consider legal consultation for extensions

### System Failure During Deletion
1. Immediately pause execution
2. Capture current state
3. Initiate rollback if configured
4. Document partial completion
5. Resume from checkpoint when resolved

### Accidental Over-Deletion
1. Stop all deletion processes
2. Initiate recovery from backup
3. Document incident
4. Review and update safeguards
5. Notify affected parties if required

## Testing & Validation

### Monthly Test Execution
```bash
# Test with synthetic user
./scripts/dsar_test.sh \
  --user_id="test_usr_synthetic" \
  --request_type="deletion" \
  --validate_only
```

### Compliance Audit
- Quarterly review of completion times
- Annual third-party audit
- Continuous monitoring of SLA compliance
- Regular updates for new regulations
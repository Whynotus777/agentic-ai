# Data Governance Policy

## Overview
This document defines data classification, access control, and governance policies for all system data.

## Data Classification Tags

### PII (Personally Identifiable Information)
**Definition:** Any data that can identify a specific individual.

**Examples:**
- User IDs, emails, names
- IP addresses
- Device identifiers
- Location data
- Biometric data

**Access Rules:**
- **Read:** Requires `pii:read` permission + audit logging
- **Write:** Requires `pii:write` permission + approval workflow
- **Export:** Prohibited without DSAR request or legal requirement
- **Encryption:** AES-256 at rest, TLS 1.3 in transit
- **Retention:** 30 days maximum, then hard delete

### SENSITIVE
**Definition:** Business-critical or security-sensitive data.

**Examples:**
- Authentication tokens
- API keys
- Internal system metrics
- Model weights
- Cost data
- Robot control commands

**Access Rules:**
- **Read:** Requires `sensitive:read` permission
- **Write:** Requires `sensitive:write` permission + MFA
- **Export:** Requires manager approval + export audit
- **Encryption:** AES-256 at rest, mTLS in transit
- **Retention:** 90 days maximum, then soft delete with tombstone

### EXPORT_OK
**Definition:** Non-sensitive data approved for external sharing.

**Examples:**
- Aggregated metrics
- Public documentation
- Anonymized telemetry
- System health indicators
- Error rates (without PII)

**Access Rules:**
- **Read:** Default allowed with basic auth
- **Write:** Requires `data:write` permission
- **Export:** Allowed with rate limiting
- **Encryption:** Optional (recommended TLS)
- **Retention:** 365 days, then tombstone

## Access Control Matrix

| Role | PII Read | PII Write | SENSITIVE Read | SENSITIVE Write | EXPORT_OK |
|------|----------|-----------|----------------|-----------------|-----------|
| Admin | ✓ | ✓ | ✓ | ✓ | Full |
| Developer | Audit Required | ✗ | ✓ | Approval Required | Full |
| Operator | ✗ | ✗ | ✓ | ✗ | Read |
| Auditor | ✓ | ✗ | ✓ | ✗ | Read |
| Service Account | Per-service ACL | ✗ | Per-service ACL | ✗ | Read |

## Data Processing Rules

### Collection Minimization
- Only collect data necessary for stated purpose
- Default to EXPORT_OK classification
- Escalate to SENSITIVE/PII only when justified
- Document purpose for all PII collection

### Purpose Limitation
- Data must be used only for declared purposes
- Cross-purpose usage requires re-consent (PII) or approval (SENSITIVE)
- Maintain purpose registry in `governance/purpose_registry.yaml`

### Accuracy Requirement
- Implement data validation at ingestion
- Provide user correction mechanisms for PII
- Regular accuracy audits for SENSITIVE data
- Version control for configuration data

### Storage Limitation
- Enforce retention policies automatically
- Implement progressive data minimization:
  - Day 1-7: Full fidelity
  - Day 8-30: Aggregated/sampled
  - Day 31+: Statistical summaries only
- Exception process for legal holds

## Compliance Requirements

### GDPR Compliance
- Right to Access: Implement via DSAR API
- Right to Rectification: User portal for PII correction
- Right to Erasure: 48-hour SLA for deletion requests
- Right to Portability: Export in JSON/CSV formats
- Data Protection by Design: Default privacy settings

### CCPA Compliance
- Consumer request portal
- 45-day response window
- Opt-out for data sales (not applicable)
- Annual privacy audit requirement

### SOC2 Type II
- Continuous monitoring of access controls
- Quarterly access reviews
- Annual penetration testing
- Incident response within 4 hours

## Enforcement Mechanisms

### Technical Controls
```yaml
enforcement:
  ingestion:
    - classify_on_entry: true
    - reject_unclassified: true
    - validate_schema: true
  
  access:
    - enforce_rbac: true
    - audit_all_pii: true
    - mfa_for_sensitive: true
  
  retention:
    - automated_deletion: true
    - legal_hold_api: true
    - tombstone_recovery: 30d
```

### Administrative Controls
- Quarterly data governance review
- Monthly access audit
- Annual policy update
- Incident response team activation

### Monitoring & Alerting
- Real-time PII access monitoring
- Anomaly detection for bulk exports
- Retention policy violation alerts
- Unauthorized access attempts

## Incident Response

### Data Breach Procedure
1. **Detect:** Automated alerts + manual reports
2. **Contain:** Revoke affected credentials immediately
3. **Assess:** Determine scope and impact within 4 hours
4. **Notify:** 
   - Internal stakeholders: Immediately
   - Affected users: Within 72 hours (GDPR)
   - Regulators: As required by jurisdiction
5. **Remediate:** Patch vulnerabilities, enhance controls
6. **Review:** Post-incident review within 7 days

### Violation Handling
- First violation: Warning + retraining
- Second violation: Privilege reduction
- Third violation: Access revocation + HR review
- Malicious violation: Immediate termination + legal action

## Audit Requirements

### Continuous Auditing
- All PII access logged to immutable audit trail
- Daily automated compliance scans
- Weekly access pattern analysis
- Monthly privilege attestation

### External Audits
- Annual SOC2 Type II audit
- Bi-annual penetration testing
- Quarterly vendor security assessments
- Ad-hoc regulatory audits

## Contact Information
- Data Protection Officer: dpo@company.internal
- Security Team: security@company.internal
- Privacy Team: privacy@company.internal
- Emergency Hotline: +1-555-SEC-URITY
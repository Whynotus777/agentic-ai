# Kubernetes Guardrails Changelog

## [1.0.0] - 2024-01-15

### Added - Security Guardrails (Step 4A)

#### New Files
- `egress-proxy.yaml`: Squid proxy deployment with domain allow-listing
- `networkpolicies.yaml`: Zero-trust network policies with default deny
- `podsecurity.yaml`: Pod Security Standards enforcement for restricted profiles
- `README.guardrails.md`: Complete documentation for applying and testing guardrails
- `../tests/test_k8s_guardrails.py`: Validation suite for manifest linting and policy simulation

#### Features
- **Egress Proxy System**:
  - Squid-based proxy with configurable domain allow-list
  - High availability with 3 replicas and auto-scaling
  - Resource limits and pod disruption budgets
  - Session affinity for consistent routing

- **Network Isolation**:
  - Default deny-all NetworkPolicy for all namespaces
  - Explicit allows for DNS, egress proxy, and internal services
  - Agent namespace restricted to proxy-only external access
  - Separate policies for ingress controllers and monitoring

- **Pod Security**:
  - Enforced `restricted` profile for production and agents namespaces
  - Required security context: non-root, read-only root filesystem
  - Capability dropping (ALL capabilities removed)
  - Resource quotas and limits to prevent resource exhaustion

- **Shared Contracts**:
  - Required headers: `Idempotency-Key` and `X-Tenant-ID` for non-GET requests
  - `Retry-After` header for retry responses
  - Canonical error codes enum for consistent error handling

#### Security Improvements
- All external traffic must go through controlled proxy
- Direct egress attempts are blocked by NetworkPolicy
- Pods enforced to run as non-root with read-only filesystems
- EmptyDir volumes for necessary writable paths
- Comprehensive security context requirements

#### Testing
- Manifest structure validation
- Network policy simulation for egress denial
- Pod security standards enforcement checks
- Header contract validation

### Configuration Required
- Pods needing external access must have label: `networking/allow-egress: "true"`
- Proxy environment variables must be set in pod specs
- Domains must be added to egress proxy allow-list ConfigMap
- Pods must include proper security context settings

### Breaking Changes
- Direct external connectivity is now blocked by default
- All pods must comply with Pod Security Standards
- Root user and privileged containers are prohibited in production

### Migration Notes
1. Apply in order: PodSecurity → Egress Proxy → NetworkPolicies
2. Update pod specs with required labels and security context
3. Configure proxy environment variables in deployments
4. Test with provided validation scripts before production rollout

### Rollback Procedure
1. Delete NetworkPolicies first to restore connectivity
2. Remove Pod Security labels from namespaces if needed
3. Delete egress proxy deployment if required

---

## Future Enhancements
- [ ] Add Envoy as alternative proxy option
- [ ] Implement OPA policies for fine-grained controls
- [ ] Add admission webhooks for runtime validation
- [ ] Integrate with service mesh for advanced traffic management
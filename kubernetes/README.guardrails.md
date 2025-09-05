# Kubernetes Cluster Guardrails

## Overview

This document describes the security guardrails implemented for Kubernetes clusters, including:
- **Egress Proxy**: All external traffic must go through a controlled proxy
- **NetworkPolicies**: Zero-trust networking with default deny
- **PodSecurity**: Enforced security standards with read-only root filesystem

## Quick Start

### 1. Apply Security Guardrails

```bash
# Create namespaces with security labels
kubectl apply -f podsecurity.yaml

# Deploy egress proxy system
kubectl apply -f egress-proxy.yaml

# Apply network policies (WARNING: This will block traffic)
kubectl apply -f networkpolicies.yaml
```

### 2. Verify Installation

```bash
# Check egress proxy is running
kubectl get pods -n egress-system
kubectl get svc -n egress-system

# Verify network policies are in place
kubectl get networkpolicies --all-namespaces

# Check Pod Security Standards
kubectl describe namespace production | grep pod-security
```

### 3. Test Egress Proxy

```bash
# Create a test pod with proper labels
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: test-egress
  namespace: default
  labels:
    networking/allow-egress: "true"
spec:
  containers:
  - name: test
    image: curlimages/curl:8.4.0
    command: ["sleep", "3600"]
    env:
    - name: HTTP_PROXY
      value: "http://egress-proxy.egress-system.svc.cluster.local:3128"
    - name: HTTPS_PROXY
      value: "http://egress-proxy.egress-system.svc.cluster.local:3128"
    - name: NO_PROXY
      value: ".cluster.local,.svc,10.0.0.0/8"
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      runAsNonRoot: true
      runAsUser: 65532
      capabilities:
        drop:
        - ALL
    volumeMounts:
    - name: tmp
      mountPath: /tmp
  volumes:
  - name: tmp
    emptyDir: {}
EOF

# Test proxy connectivity (should succeed for allowed domains)
kubectl exec -it test-egress -- curl -I https://github.com

# Test direct egress without proxy (should fail)
kubectl exec -it test-egress -- sh -c "unset HTTP_PROXY HTTPS_PROXY && curl -I --connect-timeout 5 https://github.com"
```

### 4. Test Network Policy Denial

```bash
# Create pod without egress label (should be denied external access)
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: test-denied
  namespace: default
spec:
  containers:
  - name: test
    image: curlimages/curl:8.4.0
    command: ["sleep", "3600"]
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      runAsNonRoot: true
      runAsUser: 65532
      capabilities:
        drop:
        - ALL
    volumeMounts:
    - name: tmp
      mountPath: /tmp
  volumes:
  - name: tmp
    emptyDir: {}
EOF

# This should timeout (network policy blocks it)
kubectl exec -it test-denied -- curl -I --connect-timeout 5 https://github.com || echo "✓ External access blocked as expected"

# Clean up test pods
kubectl delete pod test-egress test-denied
```

### 5. Test Pod Security Standards

```bash
# Try to create a privileged pod (should fail in production namespace)
cat <<EOF | kubectl apply -f - 2>&1 | grep -q "violates PodSecurity" && echo "✓ Privileged pod blocked"
apiVersion: v1
kind: Pod
metadata:
  name: test-privileged
  namespace: production
spec:
  containers:
  - name: test
    image: busybox
    command: ["sleep", "3600"]
    securityContext:
      privileged: true
EOF

# Try to create pod running as root (should fail)
cat <<EOF | kubectl apply -f - 2>&1 | grep -q "violates PodSecurity" && echo "✓ Root pod blocked"
apiVersion: v1
kind: Pod
metadata:
  name: test-root
  namespace: production
spec:
  containers:
  - name: test
    image: busybox
    command: ["sleep", "3600"]
    securityContext:
      runAsUser: 0
EOF
```

## Architecture

### Network Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Agent     │────▶│ Egress      │────▶│  External   │
│   Pods      │     │ Proxy       │     │  Services   │
└─────────────┘     └─────────────┘     └─────────────┘
      ↓                                         ✗
      └──────────────────X─────────────────────┘
         Direct egress blocked by NetworkPolicy
```

### Security Layers

1. **NetworkPolicies**: Default deny with explicit allows
2. **Egress Proxy**: Domain allow-listing and request logging
3. **PodSecurity**: Enforced security context requirements
4. **Resource Quotas**: Prevent resource exhaustion
5. **Read-only Root FS**: Prevent runtime modifications

## Configuration

### Required Pod Labels

| Label | Value | Purpose |
|-------|-------|---------|
| `networking/allow-egress` | `"true"` | Allow external access via proxy |
| `networking/allow-ingress` | `"true"` | Allow ingress from controllers |
| `networking/internal` | `"true"` | Allow namespace-internal communication |
| `monitoring/prometheus-scrape` | `"true"` | Allow metrics scraping |

### Required Pod Configuration

```yaml
# Minimum security context for production pods
securityContext:
  runAsNonRoot: true
  runAsUser: 65532  # Or any UID > 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop:
    - ALL

# Required environment variables for external access
env:
- name: HTTP_PROXY
  value: "http://egress-proxy.egress-system.svc.cluster.local:3128"
- name: HTTPS_PROXY
  value: "http://egress-proxy.egress-system.svc.cluster.local:3128"
- name: NO_PROXY
  value: ".cluster.local,.svc,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"

# Required volume mounts for writable directories
volumeMounts:
- name: tmp
  mountPath: /tmp
- name: cache
  mountPath: /app/cache

volumes:
- name: tmp
  emptyDir: {}
- name: cache
  emptyDir: {}
```

### Shared Contract Headers

All non-GET requests must include:
- `Idempotency-Key: <uuid4>`
- `X-Tenant-ID: <tenant>`

Retry responses must include:
- `Retry-After: <seconds>`

### Error Codes

Canonical error enum values:
- `RATE_LIMIT_EXCEEDED`
- `BUDGET_EXCEEDED`
- `SCHEMA_VALIDATION_FAILED`
- `UNAUTHORIZED`
- `FORBIDDEN`
- `CONFLICT`
- `RETRY_LATER`
- `INVARIANT_VIOLATION`
- `POLICY_BLOCKED`
- `HITL_REQUIRED`

## Rollback Procedure

If issues arise, rollback in reverse order:

```bash
# 1. Remove network policies first (restores connectivity)
kubectl delete -f networkpolicies.yaml

# 2. Remove pod security policies if needed
kubectl label namespace production pod-security.kubernetes.io/enforce-
kubectl label namespace agents pod-security.kubernetes.io/enforce-

# 3. Remove egress proxy if needed
kubectl delete -f egress-proxy.yaml

# 4. Verify rollback
kubectl get networkpolicies --all-namespaces
kubectl get pods --all-namespaces
```

## Monitoring

### Check Egress Proxy Logs

```bash
# View proxy logs
kubectl logs -n egress-system -l app.kubernetes.io/name=egress-proxy

# Check denied requests
kubectl logs -n egress-system -l app.kubernetes.io/name=egress-proxy | grep "DENIED"
```

### Check NetworkPolicy Violations

```bash
# Look for connection timeouts in pod logs
kubectl logs <pod-name> | grep -i "timeout\|connection refused"

# Check events for policy violations
kubectl get events --all-namespaces | grep NetworkPolicy
```

### Verify Security Compliance

```bash
# Check pods running as non-root
kubectl get pods --all-namespaces -o json | \
  jq '.items[] | select(.spec.securityContext.runAsNonRoot != true) | .metadata.name'

# Check pods with read-only root filesystem
kubectl get pods --all-namespaces -o json | \
  jq '.items[].spec.containers[] | select(.securityContext.readOnlyRootFilesystem != true) | .name'
```

## Troubleshooting

### Common Issues

#### 1. Pod Can't Reach External Services

**Symptom**: Timeouts when accessing external URLs

**Solution**:
- Ensure pod has label `networking/allow-egress: "true"`
- Verify proxy environment variables are set
- Check if domain is in egress proxy allow-list

#### 2. Pod Fails to Start (Security Violations)

**Symptom**: Pod rejected with "violates PodSecurity" error

**Solution**:
- Add proper security context (see required configuration above)
- Ensure running as non-root user
- Add emptyDir volumes for writable paths

#### 3. Internal Service Communication Fails

**Symptom**: Can't reach other services in namespace

**Solution**:
- Add label `networking/internal: "true"` to both pods
- Verify DNS is working: `kubectl exec <pod> -- nslookup kubernetes`

#### 4. Metrics Scraping Fails

**Symptom**: Prometheus can't scrape metrics

**Solution**:
- Add label `monitoring/prometheus-scrape: "true"`
- Verify Prometheus namespace has label `name: monitoring`

### Debug Commands

```bash
# Test DNS resolution
kubectl run -it --rm debug --image=busybox --restart=Never -- nslookup kubernetes

# Test proxy connectivity
kubectl run -it --rm debug --image=curlimages/curl \
  --env="HTTP_PROXY=http://egress-proxy.egress-system.svc.cluster.local:3128" \
  --labels="networking/allow-egress=true" \
  -- curl -I https://github.com

# Check network policy rules
kubectl describe networkpolicy <policy-name> -n <namespace>

# View pod security context
kubectl get pod <pod-name> -o jsonpath='{.spec.securityContext}'
```

## Performance Tuning

### Egress Proxy Scaling

```bash
# Manual scale
kubectl scale deployment egress-proxy -n egress-system --replicas=5

# Check HPA status
kubectl get hpa -n egress-system

# Adjust HPA if needed
kubectl edit hpa egress-proxy -n egress-system
```

### Network Policy Optimization

- Minimize the number of NetworkPolicy rules
- Use namespace selectors instead of pod selectors where possible
- Combine related rules into single policies

## Compliance Checklist

- [ ] All namespaces have Pod Security Standards labels
- [ ] NetworkPolicies implement default-deny
- [ ] Egress proxy is running and accessible
- [ ] Production pods run as non-root
- [ ] Production pods have read-only root filesystem
- [ ] All external traffic goes through egress proxy
- [ ] Resource quotas are in place
- [ ] Monitoring and alerting configured

## References

- [Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)
- [Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
- [Security Context](https://kubernetes.io/docs/tasks/configure-pod-container/security-context/)
- [Resource Quotas](https://kubernetes.io/docs/concepts/policy/resource-quotas/)
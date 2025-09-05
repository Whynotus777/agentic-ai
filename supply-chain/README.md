# Supply Chain Security Documentation

## Overview

This document covers the supply chain security implementation for the agentic-ai system, including:
- Software Bill of Materials (SBOM) generation
- Container image signing with Cosign/Sigstore
- Kubernetes admission control for signature verification
- Hardened Dockerfile with non-root execution
- Security-enhanced entrypoint script

## Components

### 1. CI/CD Pipeline Security (`.github/workflows/ci.yaml`)

The CI pipeline implements comprehensive supply chain security:

- **SBOM Generation**: Creates SBOM in SPDX and CycloneDX formats using Syft
- **Image Signing**: Signs images using Cosign with keyless Sigstore signing
- **Attestation**: Attaches and signs SBOM as attestation
- **Vulnerability Scanning**: Scans with Trivy and uploads results to GitHub Security

### 2. Kubernetes Admission Control (`kubernetes/admission/signature-policy.yaml`)

Enforces signature verification for all container images:

- **ClusterImagePolicy**: Requires valid Cosign signatures
- **ValidatingWebhook**: Blocks unsigned images at admission time
- **Keyless Verification**: Supports GitHub Actions OIDC signing
- **SBOM Attestation**: Requires SBOM for production images

### 3. Hardened Container (`Dockerfile`)

Security-hardened multi-stage build:

- **Non-root User**: Runs as `appuser` (UID 1001)
- **Minimal Attack Surface**: Slim base image with only required packages
- **Pinned Dependencies**: All packages use specific versions
- **Security Scanning**: Built-in Trivy scanning stage

### 4. Secure Entrypoint (`docker/entrypoint.sh`)

Implements security best practices:

- **Strict Error Handling**: `set -euo pipefail`
- **Security Checks**: Validates non-root execution
- **Environment Validation**: Checks required variables
- **Signal Handling**: Graceful shutdown on SIGTERM/SIGINT

## Local Verification Commands

### Verify Image Signature

```bash
# Install Cosign
wget https://github.com/sigstore/cosign/releases/download/v2.2.0/cosign-linux-amd64
chmod +x cosign-linux-amd64
sudo mv cosign-linux-amd64 /usr/local/bin/cosign

# Verify signature (keyless/OIDC signing)
cosign verify \
  --certificate-identity-regexp "https://github.com/YOUR_ORG/agentic-ai/.*" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  ghcr.io/YOUR_ORG/agentic-ai:latest

# Verify with public key (if using key-based signing)
cosign verify --key cosign.pub ghcr.io/YOUR_ORG/agentic-ai:latest
```

### Download and Inspect SBOM

```bash
# Download SBOM from image
cosign download sbom ghcr.io/YOUR_ORG/agentic-ai:latest > sbom.json

# Pretty print SBOM
cat sbom.json | jq .

# Verify SBOM signature
cosign verify-attestation \
  --type spdxjson \
  --certificate-identity-regexp "https://github.com/YOUR_ORG/agentic-ai/.*" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  ghcr.io/YOUR_ORG/agentic-ai:latest
```

### Generate Local SBOM

```bash
# Install Syft
curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin

# Generate SBOM for local image
syft docker:agentic-ai:local -o spdx-json > sbom-local.json

# Generate SBOM for source code
syft dir:. -o spdx-json > sbom-source.json
```

## In-Cluster Verification

### Deploy Sigstore Policy Controller

```bash
# Install policy-controller (official Sigstore admission controller)
kubectl apply -f https://github.com/sigstore/policy-controller/releases/download/v0.8.0/policy-controller.yaml

# Apply our custom policies
kubectl apply -f kubernetes/admission/signature-policy.yaml

# Verify policy is active
kubectl get clusterimagepolicy
kubectl get validatingwebhookconfiguration
```

### Test Signature Enforcement

```bash
# Test 1: Deploy signed image (should succeed)
kubectl run signed-test --image=ghcr.io/YOUR_ORG/agentic-ai:latest

# Test 2: Deploy unsigned image (should fail)
kubectl run unsigned-test --image=busybox:latest
# Expected: Error from server (BadRequest): admission webhook denied the request

# Test 3: Check admission webhook logs
kubectl logs -n cosign-system deployment/policy-controller-webhook

# Test 4: Bypass verification (emergency only)
kubectl run emergency-pod --image=busybox:latest \
  --labels="skip-signature-verification=true"
```

### Simulate Policy Violations

```bash
# 1. Test unsigned image rejection
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: unsigned-test
spec:
  containers:
  - name: app
    image: nginx:latest  # Unsigned public image
EOF
# Expected: Admission webhook should reject

# 2. Test signed image acceptance
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: signed-test
spec:
  containers:
  - name: app
    image: ghcr.io/YOUR_ORG/agentic-ai:latest
EOF
# Expected: Should be created successfully

# 3. Verify webhook is working
kubectl get events --field-selector reason=FailedAdmission
```

## Security Verification Checklist

### Container Security

- [ ] Container runs as non-root user (UID 1001)
- [ ] Dockerfile uses multi-stage build
- [ ] All packages are pinned to specific versions
- [ ] Base image uses SHA256 digest
- [ ] No secrets in image layers
- [ ] Minimal attack surface (slim image)

Verify non-root:
```bash
docker run --rm ghcr.io/YOUR_ORG/agentic-ai:latest id
# Should output: uid=1001(appuser) gid=1001(appuser)
```

### Supply Chain Security

- [ ] SBOM generated for every build
- [ ] Images signed with Cosign
- [ ] Signatures stored in transparency log (Rekor)
- [ ] Admission control enforces signatures
- [ ] Vulnerability scanning in CI

### Verify in CI

Check CI artifacts:
```bash
# Download artifacts from GitHub Actions
gh run download <run-id> -n sbom-artifacts
gh run download <run-id> -n signatures

# Verify SBOM exists
ls -la sbom*.json

# Check signature files
ls -la *.sig *.pem
```

## Key Generation (Optional - for key-based signing)

If you prefer key-based signing over keyless:

```bash
# Generate key pair
cosign generate-key-pair

# This creates:
# - cosign.key (private key - store securely!)
# - cosign.pub (public key - distribute freely)

# Add private key to GitHub Secrets as COSIGN_PRIVATE_KEY
# Add password to GitHub Secrets as COSIGN_PASSWORD

# Sign image with key
cosign sign --key cosign.key ghcr.io/YOUR_ORG/agentic-ai:latest

# Verify with public key
cosign verify --key cosign.pub ghcr.io/YOUR_ORG/agentic-ai:latest
```

## Troubleshooting

### Common Issues

1. **Admission webhook denying all images**
   - Check webhook logs: `kubectl logs -n cosign-system deployment/policy-controller-webhook`
   - Verify policy configuration: `kubectl get clusterimagepolicy -o yaml`
   - Ensure image is properly signed

2. **SBOM not attached to image**
   - Verify Cosign attach command in CI
   - Check registry supports OCI artifacts
   - Try manual attachment: `cosign attach sbom --sbom sbom.json IMAGE`

3. **Container running as root**
   - Check Dockerfile USER directive
   - Verify no runtime user override
   - Ensure base image supports non-root

4. **Signature verification fails**
   - Check certificate identity matches
   - Verify OIDC issuer URL
   - Ensure Rekor/Fulcio connectivity

### Debug Commands

```bash
# Get detailed signature information
cosign tree ghcr.io/YOUR_ORG/agentic-ai:latest

# Check Rekor transparency log
rekor-cli search --rekor_server https://rekor.sigstore.dev --sha <image-digest>

# Verify certificate details
cosign verify --certificate-identity-regexp ".*" \
  --certificate-oidc-issuer-regexp ".*" \
  ghcr.io/YOUR_ORG/agentic-ai:latest | jq .

# Test admission webhook directly
kubectl run test --image=busybox --dry-run=server -o yaml
```

## Compliance

This implementation addresses the following security standards:

- **SLSA Level 3**: Signed provenance, isolated builds
- **NIST 800-190**: Container security guidelines
- **CIS Docker Benchmark**: Hardened container configuration
- **SSDF**: Secure Software Development Framework practices

## Additional Resources

- [Sigstore Documentation](https://docs.sigstore.dev/)
- [Cosign GitHub Actions](https://github.com/sigstore/cosign-installer)
- [SBOM Formats (SPDX)](https://spdx.dev/)
- [Container Security Best Practices](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)
- [Kubernetes Admission Controllers](https://kubernetes.io/docs/reference/access-authn-authz/admission-controllers/)
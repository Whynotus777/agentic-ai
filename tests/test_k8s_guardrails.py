#!/usr/bin/env python3
"""
Test suite for Kubernetes guardrails validation.
Lints manifest structure and simulates network policy enforcement.
"""

import yaml
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class PolicyEffect(Enum):
    """Network policy effects"""
    ALLOW = "allow"
    DENY = "deny"


class ErrorCode(Enum):
    """Canonical error codes from shared contract"""
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    CONFLICT = "CONFLICT"
    RETRY_LATER = "RETRY_LATER"
    INVARIANT_VIOLATION = "INVARIANT_VIOLATION"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    HITL_REQUIRED = "HITL_REQUIRED"


@dataclass
class ValidationResult:
    """Result of a validation check"""
    passed: bool
    message: str
    severity: str = "error"  # error, warning, info
    
    def __str__(self):
        icon = "✓" if self.passed else "✗"
        return f"{icon} [{self.severity.upper()}] {self.message}"


class ManifestLinter:
    """Validates Kubernetes manifest structure and security settings"""
    
    REQUIRED_SECURITY_CONTEXT = {
        "runAsNonRoot": True,
        "allowPrivilegeEscalation": False,
        "readOnlyRootFilesystem": True,
    }
    
    REQUIRED_CAPABILITIES = {
        "drop": ["ALL"]
    }
    
    REQUIRED_HEADERS = ["Idempotency-Key", "X-Tenant-ID"]
    
    def __init__(self, manifest_dir: Path):
        self.manifest_dir = Path(manifest_dir)
        self.results: List[ValidationResult] = []
    
    def lint_all(self) -> bool:
        """Lint all manifest files"""
        files = [
            "egress-proxy.yaml",
            "networkpolicies.yaml",
            "podsecurity.yaml"
        ]
        
        all_passed = True
        for file in files:
            filepath = self.manifest_dir / file
            if not filepath.exists():
                self.results.append(
                    ValidationResult(False, f"Missing required file: {file}")
                )
                all_passed = False
                continue
            
            try:
                with open(filepath) as f:
                    docs = list(yaml.safe_load_all(f))
                    
                for doc in docs:
                    if not doc:
                        continue
                    
                    kind = doc.get("kind", "")
                    if kind == "Deployment":
                        all_passed &= self._lint_deployment(doc, file)
                    elif kind == "NetworkPolicy":
                        all_passed &= self._lint_network_policy(doc, file)
                    elif kind == "Namespace":
                        all_passed &= self._lint_namespace(doc, file)
                    elif kind == "ConfigMap" and "egress" in file:
                        all_passed &= self._lint_egress_config(doc, file)
                        
            except Exception as e:
                self.results.append(
                    ValidationResult(False, f"Failed to parse {file}: {e}")
                )
                all_passed = False
        
        return all_passed
    
    def _lint_deployment(self, deployment: Dict, filename: str) -> bool:
        """Validate deployment security settings"""
        passed = True
        name = deployment["metadata"]["name"]
        
        # Check pod spec
        pod_spec = deployment["spec"]["template"]["spec"]
        
        # Validate pod security context
        pod_security = pod_spec.get("securityContext", {})
        if not pod_security.get("runAsNonRoot"):
            self.results.append(
                ValidationResult(False, f"{filename}/{name}: Pod must have runAsNonRoot: true")
            )
            passed = False
        
        if pod_security.get("runAsUser", 0) == 0:
            self.results.append(
                ValidationResult(False, f"{filename}/{name}: Pod must not run as root (UID 0)")
            )
            passed = False
        
        # Check containers
        for container in pod_spec.get("containers", []):
            container_name = container.get("name", "unknown")
            security_context = container.get("securityContext", {})
            
            # Check required security settings
            for key, expected in self.REQUIRED_SECURITY_CONTEXT.items():
                if security_context.get(key) != expected:
                    self.results.append(
                        ValidationResult(
                            False, 
                            f"{filename}/{name}/{container_name}: Missing or incorrect {key}"
                        )
                    )
                    passed = False
            
            # Check capabilities
            caps = security_context.get("capabilities", {})
            if caps.get("drop") != ["ALL"]:
                self.results.append(
                    ValidationResult(
                        False,
                        f"{filename}/{name}/{container_name}: Must drop ALL capabilities"
                    )
                )
                passed = False
            
            # Check resource limits
            if "resources" not in container:
                self.results.append(
                    ValidationResult(
                        False,
                        f"{filename}/{name}/{container_name}: Missing resource limits",
                        severity="warning"
                    )
                )
            
        if passed:
            self.results.append(
                ValidationResult(True, f"{filename}/{name}: Security settings validated")
            )
        
        return passed
    
    def _lint_network_policy(self, policy: Dict, filename: str) -> bool:
        """Validate network policy structure"""
        passed = True
        name = policy["metadata"]["name"]
        
        # Check for policy types
        policy_types = policy["spec"].get("policyTypes", [])
        if not policy_types:
            self.results.append(
                ValidationResult(
                    False,
                    f"{filename}/{name}: NetworkPolicy must specify policyTypes"
                )
            )
            passed = False
        
        # Check for default deny
        if name == "default-deny-all":
            if policy["spec"].get("podSelector") != {}:
                self.results.append(
                    ValidationResult(
                        False,
                        f"{filename}/{name}: Default deny must have empty podSelector"
                    )
                )
                passed = False
            
            if not all(t in policy_types for t in ["Ingress", "Egress"]):
                self.results.append(
                    ValidationResult(
                        False,
                        f"{filename}/{name}: Default deny must block both Ingress and Egress"
                    )
                )
                passed = False
        
        if passed:
            self.results.append(
                ValidationResult(True, f"{filename}/{name}: NetworkPolicy structure valid")
            )
        
        return passed
    
    def _lint_namespace(self, namespace: Dict, filename: str) -> bool:
        """Validate namespace Pod Security Standards"""
        passed = True
        name = namespace["metadata"]["name"]
        labels = namespace["metadata"].get("labels", {})
        
        # Check for Pod Security Standards
        pss_labels = [
            "pod-security.kubernetes.io/enforce",
            "pod-security.kubernetes.io/audit",
            "pod-security.kubernetes.io/warn"
        ]
        
        for label in pss_labels:
            if label not in labels:
                self.results.append(
                    ValidationResult(
                        False,
                        f"{filename}/{name}: Missing Pod Security label: {label}",
                        severity="warning"
                    )
                )
                passed = False
            elif label == "pod-security.kubernetes.io/enforce":
                level = labels[label]
                if name in ["production", "agents"] and level != "restricted":
                    self.results.append(
                        ValidationResult(
                            False,
                            f"{filename}/{name}: Production namespace must enforce 'restricted'"
                        )
                    )
                    passed = False
        
        if passed:
            self.results.append(
                ValidationResult(True, f"{filename}/{name}: Namespace security labels valid")
            )
        
        return passed
    
    def _lint_egress_config(self, configmap: Dict, filename: str) -> bool:
        """Validate egress proxy configuration"""
        passed = True
        name = configmap["metadata"]["name"]
        
        # Check for squid config
        squid_conf = configmap.get("data", {}).get("squid.conf", "")
        
        if not squid_conf:
            self.results.append(
                ValidationResult(False, f"{filename}/{name}: Missing squid.conf")
            )
            return False
        
        # Check for required settings
        required_settings = [
            "http_access deny all",  # Default deny
            "acl allowed_domains",    # Domain allowlist
            "http_port 3128",         # Standard proxy port
        ]
        
        for setting in required_settings:
            if setting not in squid_conf:
                self.results.append(
                    ValidationResult(
                        False,
                        f"{filename}/{name}: Missing required setting: {setting}"
                    )
                )
                passed = False
        
        if passed:
            self.results.append(
                ValidationResult(True, f"{filename}/{name}: Egress proxy config valid")
            )
        
        return passed


class NetworkPolicySimulator:
    """Simulates network policy effects for testing"""
    
    def __init__(self):
        self.policies: List[Dict] = []
        self.results: List[ValidationResult] = []
    
    def load_policies(self, filepath: Path) -> bool:
        """Load network policies from file"""
        try:
            with open(filepath) as f:
                docs = list(yaml.safe_load_all(f))
                self.policies = [
                    doc for doc in docs 
                    if doc and doc.get("kind") == "NetworkPolicy"
                ]
            return True
        except Exception as e:
            self.results.append(
                ValidationResult(False, f"Failed to load policies: {e}")
            )
            return False
    
    def simulate_egress(self, 
                       namespace: str,
                       pod_labels: Dict[str, str],
                       destination: str) -> PolicyEffect:
        """Simulate egress traffic from a pod"""
        
        # Find applicable policies
        applicable = self._find_applicable_policies(namespace, pod_labels, "Egress")
        
        if not applicable:
            # No policy = allow (Kubernetes default)
            return PolicyEffect.ALLOW
        
        # Check if any policy allows the traffic
        for policy in applicable:
            egress_rules = policy["spec"].get("egress", [])
            
            # Empty egress = deny all
            if not egress_rules:
                continue
            
            # Check each rule
            for rule in egress_rules:
                if self._matches_destination(rule, destination):
                    return PolicyEffect.ALLOW
        
        # No matching allow rule = deny
        return PolicyEffect.DENY
    
    def _find_applicable_policies(self,
                                 namespace: str,
                                 pod_labels: Dict[str, str],
                                 policy_type: str) -> List[Dict]:
        """Find policies that apply to a pod"""
        applicable = []
        
        for policy in self.policies:
            # Check namespace
            if policy["metadata"].get("namespace", "default") != namespace:
                continue
            
            # Check policy type
            if policy_type not in policy["spec"].get("policyTypes", []):
                continue
            
            # Check pod selector
            selector = policy["spec"].get("podSelector", {})
            if self._matches_selector(selector, pod_labels):
                applicable.append(policy)
        
        return applicable
    
    def _matches_selector(self, selector: Dict, labels: Dict[str, str]) -> bool:
        """Check if labels match selector"""
        if selector == {}:
            # Empty selector matches all pods
            return True
        
        match_labels = selector.get("matchLabels", {})
        for key, value in match_labels.items():
            if labels.get(key) != value:
                return False
        
        # TODO: Implement matchExpressions for complete simulation
        
        return True
    
    def _matches_destination(self, rule: Dict, destination: str) -> bool:
        """Check if destination matches egress rule"""
        # Simplified - in reality would need to resolve DNS and check IPs
        
        # Check for external traffic
        if destination.startswith("http"):
            # Check if rule allows external
            for to in rule.get("to", []):
                if "ipBlock" in to:
                    cidr = to["ipBlock"].get("cidr", "")
                    if cidr == "0.0.0.0/0":
                        return True
                
                # Check for egress proxy
                ns_selector = to.get("namespaceSelector", {})
                if ns_selector.get("matchLabels", {}).get("name") == "egress-system":
                    return True
        
        return False
    
    def test_scenarios(self) -> bool:
        """Test common network policy scenarios"""
        all_passed = True
        
        # Test 1: Pod without egress label should be denied external access
        effect = self.simulate_egress(
            namespace="default",
            pod_labels={},
            destination="https://example.com"
        )
        
        if effect == PolicyEffect.DENY:
            self.results.append(
                ValidationResult(True, "Pod without egress label correctly denied external access")
            )
        else:
            self.results.append(
                ValidationResult(False, "Pod without egress label should be denied external access")
            )
            all_passed = False
        
        # Test 2: Pod with egress label should access proxy
        effect = self.simulate_egress(
            namespace="default",
            pod_labels={"networking/allow-egress": "true"},
            destination="egress-proxy.egress-system.svc.cluster.local"
        )
        
        if effect == PolicyEffect.ALLOW:
            self.results.append(
                ValidationResult(True, "Pod with egress label can access proxy")
            )
        else:
            self.results.append(
                ValidationResult(False, "Pod with egress label should access proxy")
            )
            all_passed = False
        
        # Test 3: Agent pods restricted to proxy only
        effect = self.simulate_egress(
            namespace="agents",
            pod_labels={"app.kubernetes.io/component": "agent"},
            destination="https://external-api.com"
        )
        
        # Should be denied direct external access
        if effect == PolicyEffect.DENY:
            self.results.append(
                ValidationResult(True, "Agent pods correctly restricted to proxy-only egress")
            )
        else:
            self.results.append(
                ValidationResult(
                    False, 
                    "Agent pods must be restricted to egress proxy only",
                    severity="error"
                )
            )
            all_passed = False
        
        return all_passed


def validate_headers_contract(sample_request: Dict) -> ValidationResult:
    """Validate that requests include required headers"""
    headers = sample_request.get("headers", {})
    required = ["Idempotency-Key", "X-Tenant-ID"]
    
    missing = [h for h in required if h not in headers]
    
    if missing:
        return ValidationResult(
            False,
            f"Missing required headers: {', '.join(missing)}"
        )
    
    # Validate Idempotency-Key format (should be UUID)
    idempotency = headers.get("Idempotency-Key", "")
    if len(idempotency) != 36 or idempotency.count("-") != 4:
        return ValidationResult(
            False,
            "Idempotency-Key should be a valid UUID",
            severity="warning"
        )
    
    return ValidationResult(True, "Required headers present and valid")


def main():
    """Main test runner"""
    print("=" * 60)
    print("Kubernetes Guardrails Validation")
    print("=" * 60)
    
    # Setup paths
    k8s_dir = Path(__file__).parent.parent / "kubernetes"
    
    # Run manifest linting
    print("\n[1] Manifest Structure Validation")
    print("-" * 40)
    linter = ManifestLinter(k8s_dir)
    lint_passed = linter.lint_all()
    
    for result in linter.results:
        print(result)
    
    # Run network policy simulation
    print("\n[2] Network Policy Simulation")
    print("-" * 40)
    simulator = NetworkPolicySimulator()
    
    if simulator.load_policies(k8s_dir / "networkpolicies.yaml"):
        sim_passed = simulator.test_scenarios()
        for result in simulator.results:
            print(result)
    else:
        sim_passed = False
        print("✗ Failed to load network policies")
    
    # Test header contract
    print("\n[3] Shared Contract Validation")
    print("-" * 40)
    
    # Test valid request
    valid_request = {
        "headers": {
            "Idempotency-Key": "550e8400-e29b-41d4-a716-446655440000",
            "X-Tenant-ID": "tenant-123"
        }
    }
    result = validate_headers_contract(valid_request)
    print(result)
    
    # Test invalid request
    invalid_request = {
        "headers": {
            "X-Tenant-ID": "tenant-123"
        }
    }
    result = validate_headers_contract(invalid_request)
    print(result)
    
    # Summary
    print("\n" + "=" * 60)
    all_passed = lint_passed and sim_passed
    
    if all_passed:
        print("✓ All guardrail validations passed!")
        return 0
    else:
        print("✗ Some validations failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
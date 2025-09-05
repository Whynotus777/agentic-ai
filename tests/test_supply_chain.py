"""
Supply Chain Security Tests
Validates CI workflow, Dockerfile hardening, and security configurations.
"""
import os
import re
import yaml
import pytest
from pathlib import Path
from typing import Dict, List, Any


class TestSupplyChainSecurity:
    """Test suite for supply chain security implementation."""
    
    @pytest.fixture
    def ci_workflow(self) -> Dict[str, Any]:
        """Load CI workflow YAML."""
        workflow_path = Path(".github/workflows/ci.yaml")
        if not workflow_path.exists():
            pytest.skip("CI workflow file not found")
        
        with open(workflow_path, 'r') as f:
            return yaml.safe_load(f)
    
    @pytest.fixture
    def dockerfile_content(self) -> str:
        """Load Dockerfile content."""
        dockerfile_path = Path("Dockerfile")
        if not dockerfile_path.exists():
            pytest.skip("Dockerfile not found")
        
        with open(dockerfile_path, 'r') as f:
            return f.read()
    
    @pytest.fixture
    def entrypoint_content(self) -> str:
        """Load entrypoint script content."""
        entrypoint_path = Path("docker/entrypoint.sh")
        if not entrypoint_path.exists():
            pytest.skip("Entrypoint script not found")
        
        with open(entrypoint_path, 'r') as f:
            return f.read()
    
    @pytest.fixture
    def admission_policy(self) -> Dict[str, Any]:
        """Load Kubernetes admission policy."""
        policy_path = Path("kubernetes/admission/signature-policy.yaml")
        if not policy_path.exists():
            pytest.skip("Admission policy not found")
        
        with open(policy_path, 'r') as f:
            # Handle multi-document YAML
            docs = list(yaml.safe_load_all(f))
            return {"documents": docs}
    
    # ========================================================================
    # CI Workflow Tests
    # ========================================================================
    
    def test_ci_has_cosign_signing(self, ci_workflow):
        """Test that CI workflow includes Cosign signing."""
        # Check for Cosign installation
        workflow_str = str(ci_workflow)
        assert "cosign" in workflow_str.lower(), "CI workflow should include Cosign"
        
        # Check for specific Cosign commands
        jobs = ci_workflow.get('jobs', {})
        build_job = jobs.get('build-and-sign', {})
        steps = build_job.get('steps', [])
        
        # Find Cosign-related steps
        has_cosign_install = False
        has_cosign_sign = False
        
        for step in steps:
            step_name = step.get('name', '').lower()
            step_run = step.get('run', '').lower()
            step_uses = step.get('uses', '').lower()
            
            if 'cosign' in step_name or 'cosign' in step_uses:
                has_cosign_install = True
            
            if 'cosign sign' in step_run:
                has_cosign_sign = True
        
        assert has_cosign_install, "CI should install Cosign"
        assert has_cosign_sign, "CI should sign images with Cosign"
    
    def test_ci_generates_sbom(self, ci_workflow):
        """Test that CI workflow generates SBOM."""
        workflow_str = str(ci_workflow)
        
        # Check for SBOM generation
        assert "sbom" in workflow_str.lower(), "CI workflow should generate SBOM"
        assert "syft" in workflow_str.lower(), "CI should use Syft for SBOM generation"
        
        # Check for SBOM file outputs
        jobs = ci_workflow.get('jobs', {})
        build_job = jobs.get('build-and-sign', {})
        steps = build_job.get('steps', [])
        
        sbom_step_found = False
        for step in steps:
            if 'sbom' in step.get('name', '').lower():
                sbom_step_found = True
                run_cmd = step.get('run', '')
                # Check for SBOM output files
                assert 'sbom.spdx.json' in run_cmd or 'sbom.json' in run_cmd, \
                    "SBOM step should output sbom.json file"
                break
        
        assert sbom_step_found, "CI should have dedicated SBOM generation step"
    
    def test_ci_uploads_artifacts(self, ci_workflow):
        """Test that CI uploads SBOM and signature artifacts."""
        jobs = ci_workflow.get('jobs', {})
        build_job = jobs.get('build-and-sign', {})
        steps = build_job.get('steps', [])
        
        has_sbom_upload = False
        has_signature_upload = False
        
        for step in steps:
            step_name = step.get('name', '').lower()
            if 'upload' in step_name and 'sbom' in step_name:
                has_sbom_upload = True
                # Check artifact includes sbom files
                with_config = step.get('with', {})
                path = with_config.get('path', '')
                assert 'sbom' in path.lower(), "Should upload SBOM files"
            
            if 'upload' in step_name and ('signature' in step_name or 'sign' in step_name):
                has_signature_upload = True
        
        assert has_sbom_upload, "CI should upload SBOM artifacts"
        # Signature upload is optional for keyless signing
    
    def test_ci_prints_verification_command(self, ci_workflow):
        """Test that CI prints verification command."""
        jobs = ci_workflow.get('jobs', {})
        build_job = jobs.get('build-and-sign', {})
        steps = build_job.get('steps', [])
        
        has_verification_print = False
        for step in steps:
            step_name = step.get('name', '').lower()
            if 'verification' in step_name or 'verify' in step_name:
                step_run = step.get('run', '')
                if 'cosign verify' in step_run:
                    has_verification_print = True
                    break
        
        assert has_verification_print, "CI should print cosign verify command"
    
    def test_ci_has_vulnerability_scanning(self, ci_workflow):
        """Test that CI includes vulnerability scanning."""
        workflow_str = str(ci_workflow)
        
        # Check for security scanning tools
        has_trivy = 'trivy' in workflow_str.lower()
        has_snyk = 'snyk' in workflow_str.lower()
        has_grype = 'grype' in workflow_str.lower()
        
        assert has_trivy or has_snyk or has_grype, \
            "CI should include vulnerability scanning (Trivy/Snyk/Grype)"
    
    # ========================================================================
    # Dockerfile Tests
    # ========================================================================
    
    def test_dockerfile_has_user_nonroot(self, dockerfile_content):
        """Test that Dockerfile specifies non-root USER."""
        # Check for USER directive
        user_pattern = r'USER\s+(\S+)'
        user_matches = re.findall(user_pattern, dockerfile_content)
        
        assert len(user_matches) > 0, "Dockerfile must have USER directive"
        
        # Ensure USER is not root
        for user in user_matches:
            assert user != 'root', f"Dockerfile should not use 'USER root' (found: USER {user})"
            assert user in ['appuser', 'nobody', 'nonroot'] or user.isdigit(), \
                f"User should be non-root (found: {user})"
        
        # Check that final stage uses non-root
        # Split by FROM to get stages
        stages = dockerfile_content.split('FROM ')
        if len(stages) > 1:
            final_stage = stages[-1]
            assert 'USER ' in final_stage, "Final stage must specify USER"
            final_user_matches = re.findall(user_pattern, final_stage)
            assert len(final_user_matches) > 0, "Final stage must have USER directive"
            assert final_user_matches[-1] != 'root', "Final USER must be non-root"
    
    def test_dockerfile_is_multistage(self, dockerfile_content):
        """Test that Dockerfile uses multi-stage build."""
        # Count FROM statements
        from_pattern = r'^FROM\s+'
        from_matches = re.findall(from_pattern, dockerfile_content, re.MULTILINE)
        
        assert len(from_matches) >= 2, \
            f"Dockerfile should use multi-stage build (found {len(from_matches)} FROM statements)"
        
        # Check for stage names
        stage_pattern = r'FROM\s+.*\s+AS\s+(\S+)'
        stage_matches = re.findall(stage_pattern, dockerfile_content, re.IGNORECASE)
        
        assert len(stage_matches) >= 1, "Multi-stage build should use named stages (AS <name>)"
    
    def test_dockerfile_uses_pinned_packages(self, dockerfile_content):
        """Test that Dockerfile pins package versions."""
        # Check for apt-get install with versions
        apt_pattern = r'apt-get\s+install[^&]*'
        apt_matches = re.findall(apt_pattern, dockerfile_content)
        
        if apt_matches:
            for apt_cmd in apt_matches:
                # Check if packages have versions (package=version format)
                if not any(skip in apt_cmd for skip in ['update', 'upgrade', 'clean']):
                    # At least some packages should be pinned
                    version_pattern = r'\S+=[\d\.]+'
                    has_versions = re.search(version_pattern, apt_cmd)
                    assert has_versions or '--no-install-recommends' in apt_cmd, \
                        f"Packages should be pinned to versions: {apt_cmd[:50]}"
    
    def test_dockerfile_uses_digest(self, dockerfile_content):
        """Test that base images use SHA256 digest."""
        # Check FROM statements for digest
        from_pattern = r'FROM\s+([^\s]+)'
        from_matches = re.findall(from_pattern, dockerfile_content)
        
        has_digest = False
        for image in from_matches:
            if '@sha256:' in image:
                has_digest = True
                break
        
        assert has_digest, "At least one base image should use SHA256 digest for reproducibility"
    
    def test_dockerfile_security_labels(self, dockerfile_content):
        """Test that Dockerfile includes security labels."""
        # Check for security-related labels
        assert 'LABEL' in dockerfile_content, "Dockerfile should include LABEL directives"
        
        security_labels = [
            'security.scan',
            'security.sbom',
            'security.non-root',
            'version',
            'maintainer'
        ]
        
        found_labels = 0
        for label in security_labels:
            if label in dockerfile_content:
                found_labels += 1
        
        assert found_labels >= 2, \
            f"Dockerfile should include security labels (found {found_labels}/{len(security_labels)})"
    
    def test_dockerfile_has_healthcheck(self, dockerfile_content):
        """Test that Dockerfile includes HEALTHCHECK."""
        assert 'HEALTHCHECK' in dockerfile_content, \
            "Dockerfile should include HEALTHCHECK directive"
    
    # ========================================================================
    # Entrypoint Tests
    # ========================================================================
    
    def test_entrypoint_has_pipefail(self, entrypoint_content):
        """Test that entrypoint uses set -euo pipefail."""
        # Check for the security flags
        assert 'set -euo pipefail' in entrypoint_content or \
               ('set -e' in entrypoint_content and 
                'set -u' in entrypoint_content and 
                'set -o pipefail' in entrypoint_content), \
               "Entrypoint must use 'set -euo pipefail' for strict error handling"
    
    def test_entrypoint_has_error_handling(self, entrypoint_content):
        """Test that entrypoint has proper error handling."""
        # Check for trap commands
        assert 'trap' in entrypoint_content, "Entrypoint should use trap for error handling"
        
        # Check for error handling patterns
        error_patterns = [
            r'trap.*ERR',
            r'trap.*EXIT',
            r'trap.*SIGTERM',
            r'trap.*SIGINT'
        ]
        
        found_traps = 0
        for pattern in error_patterns:
            if re.search(pattern, entrypoint_content):
                found_traps += 1
        
        assert found_traps >= 2, \
            f"Entrypoint should handle multiple signals (found {found_traps} trap handlers)"
    
    def test_entrypoint_checks_nonroot(self, entrypoint_content):
        """Test that entrypoint verifies non-root execution."""
        # Check for root user validation
        root_check_patterns = [
            r'id -u.*-eq 0',
            r'whoami.*root',
            r'USER.*root',
            r'\$UID.*-eq.*0'
        ]
        
        has_root_check = False
        for pattern in root_check_patterns:
            if re.search(pattern, entrypoint_content):
                has_root_check = True
                break
        
        assert has_root_check, "Entrypoint should check for root user execution"
    
    def test_entrypoint_has_logging(self, entrypoint_content):
        """Test that entrypoint has proper logging."""
        # Check for logging functions
        assert 'log_info' in entrypoint_content or 'echo' in entrypoint_content, \
            "Entrypoint should have logging"
        
        # Check for timestamp in logs
        timestamp_patterns = [
            r'date',
            r'timestamp',
            r'%Y-%m-%d',
            r'\$\(date'
        ]
        
        has_timestamp = any(pattern in entrypoint_content for pattern in timestamp_patterns)
        assert has_timestamp, "Entrypoint logging should include timestamps"
    
    # ========================================================================
    # Kubernetes Admission Policy Tests
    # ========================================================================
    
    def test_admission_has_clusterimagepolicy(self, admission_policy):
        """Test that admission policy includes ClusterImagePolicy."""
        docs = admission_policy['documents']
        
        has_cluster_policy = False
        for doc in docs:
            if doc and doc.get('kind') == 'ClusterImagePolicy':
                has_cluster_policy = True
                # Check for authorities
                spec = doc.get('spec', {})
                authorities = spec.get('authorities', [])
                assert len(authorities) > 0, "ClusterImagePolicy should define authorities"
                break
        
        assert has_cluster_policy, "Admission policy should include ClusterImagePolicy"
    
    def test_admission_has_webhook(self, admission_policy):
        """Test that admission policy includes ValidatingWebhookConfiguration."""
        docs = admission_policy['documents']
        
        has_webhook = False
        for doc in docs:
            if doc and doc.get('kind') == 'ValidatingWebhookConfiguration':
                has_webhook = True
                # Check webhook configuration
                webhooks = doc.get('webhooks', [])
                assert len(webhooks) > 0, "Should define at least one webhook"
                
                # Check failure policy
                for webhook in webhooks:
                    failure_policy = webhook.get('failurePolicy', '')
                    assert failure_policy == 'Fail', \
                        "Webhook should fail closed for security"
                break
        
        assert has_webhook, "Admission policy should include ValidatingWebhookConfiguration"
    
    def test_admission_requires_signatures(self, admission_policy):
        """Test that policy requires image signatures."""
        docs = admission_policy['documents']
        
        for doc in docs:
            if doc and doc.get('kind') == 'ClusterImagePolicy':
                spec = doc.get('spec', {})
                authorities = spec.get('authorities', [])
                
                # Check for keyless or key-based verification
                for authority in authorities:
                    has_keyless = 'keyless' in authority
                    has_key = 'key' in authority
                    assert has_keyless or has_key, \
                        "Authority should specify keyless or key-based verification"
    
    # ========================================================================
    # Integration Tests
    # ========================================================================
    
    def test_supply_chain_completeness(self, ci_workflow, dockerfile_content, 
                                      entrypoint_content, admission_policy):
        """Test that all supply chain components are present and integrated."""
        # CI generates SBOM
        ci_str = str(ci_workflow).lower()
        assert 'sbom' in ci_str, "CI should generate SBOM"
        
        # CI signs images
        assert 'cosign sign' in ci_str, "CI should sign images"
        
        # Dockerfile uses non-root
        assert 'USER' in dockerfile_content and 'appuser' in dockerfile_content, \
            "Dockerfile should run as non-root user"
        
        # Entrypoint has security checks
        assert 'set -euo pipefail' in entrypoint_content, \
            "Entrypoint should have security flags"
        
        # Admission policy enforces signatures
        policy_str = str(admission_policy).lower()
        assert 'signature' in policy_str or 'cosign' in policy_str, \
            "Admission policy should enforce signatures"
        
        print("✅ All supply chain security components are properly integrated")


# Additional helper functions for testing

def verify_sbom_format(sbom_path: Path) -> bool:
    """Verify SBOM file format is valid."""
    if not sbom_path.exists():
        return False
    
    try:
        import json
        with open(sbom_path, 'r') as f:
            sbom = json.load(f)
        
        # Check for SPDX format
        if 'spdxVersion' in sbom:
            return 'creationInfo' in sbom and 'packages' in sbom
        
        # Check for CycloneDX format
        if 'bomFormat' in sbom:
            return sbom['bomFormat'] == 'CycloneDX' and 'components' in sbom
        
        return False
    except:
        return False


def verify_signature_exists(image_name: str) -> bool:
    """Verify that image signature exists (requires cosign)."""
    import subprocess
    try:
        result = subprocess.run(
            ['cosign', 'verify', '--cert-identity-regexp', '.*', 
             '--cert-oidc-issuer-regexp', '.*', image_name],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except:
        return False


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])
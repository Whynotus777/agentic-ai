# control_plane/supply_chain_security.py
"""
SLSA Level 3 compliance and Software Bill of Materials (SBOM) verification system.
Ensures supply chain security for all artifacts and dependencies.
"""

import asyncio
import json
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.x509 import load_pem_x509_certificate
from cryptography.hazmat.backends import default_backend

from opentelemetry import trace
tracer = trace.get_tracer(__name__)


class SLSALevel(Enum):
    """SLSA compliance levels"""
    LEVEL_0 = 0  # No guarantees
    LEVEL_1 = 1  # Documentation of build process
    LEVEL_2 = 2  # Signed provenance
    LEVEL_3 = 3  # Hardened builds
    LEVEL_4 = 4  # Two-party review


class AttestationType(Enum):
    """Types of attestations"""
    PROVENANCE = "provenance"
    SBOM = "sbom"
    VULNERABILITY_SCAN = "vulnerability_scan"
    CODE_REVIEW = "code_review"
    TEST_RESULTS = "test_results"


class ArtifactType(Enum):
    """Types of artifacts"""
    CONTAINER_IMAGE = "container_image"
    BINARY = "binary"
    LIBRARY = "library"
    MODEL = "model"
    DATASET = "dataset"
    CONFIGURATION = "configuration"


@dataclass
class SBOMComponent:
    """Component in SBOM"""
    name: str
    version: str
    type: str  # library, binary, container
    supplier: str
    license: str
    hash_value: str
    hash_algorithm: str
    dependencies: List[str] = field(default_factory=list)
    vulnerabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvenanceStatement:
    """SLSA Provenance statement"""
    subject: Dict[str, Any]  # What was built
    predicate_type: str = "https://slsa.dev/provenance/v0.2"
    predicate: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.predicate:
            self.predicate = {
                "builder": {
                    "id": "https://github.com/actions/runner"
                },
                "buildType": "https://github.com/slsa-framework/slsa-github-generator",
                "invocation": {},
                "buildConfig": {},
                "metadata": {
                    "buildStartedOn": datetime.utcnow().isoformat(),
                    "buildFinishedOn": None,
                    "completeness": {
                        "parameters": True,
                        "environment": False,
                        "materials": True
                    },
                    "reproducible": False
                },
                "materials": []
            }


@dataclass
class Attestation:
    """Security attestation for an artifact"""
    attestation_id: str
    artifact_id: str
    artifact_type: ArtifactType
    attestation_type: AttestationType
    statement: Any  # ProvenanceStatement or SBOMStatement
    signature: str
    signer_identity: str
    timestamp: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Result of verification"""
    verified: bool
    artifact_id: str
    slsa_level: SLSALevel
    issues: List[str] = field(default_factory=list)
    attestations: List[Attestation] = field(default_factory=list)
    sbom: Optional[List[SBOMComponent]] = None
    vulnerabilities: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class SLSAVerifier:
    """
    SLSA Level 3 verification and SBOM management
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trusted_builders = self._load_trusted_builders()
        self.attestations: Dict[str, List[Attestation]] = {}
        self.sboms: Dict[str, List[SBOMComponent]] = {}
        self.verification_cache: Dict[str, VerificationResult] = {}
        
        # Key management
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        self.trusted_keys: Dict[str, Any] = {}
        
        # CVE database (mock)
        self.cve_database = self._load_cve_database()
        
    def _load_trusted_builders(self) -> List[str]:
        """Load list of trusted build systems"""
        return [
            "https://github.com/actions/runner",
            "https://gitlab.com/gitlab-runner",
            "https://jenkins.io",
            "https://circleci.com",
            "https://github.com/slsa-framework/slsa-github-generator"
        ]
    
    def _load_cve_database(self) -> Dict[str, List[str]]:
        """Load CVE database (mock)"""
        return {
            "log4j:2.14.0": ["CVE-2021-44228", "CVE-2021-45046"],
            "openssl:1.0.1": ["CVE-2014-0160"],  # Heartbleed
            "struts:2.5.0": ["CVE-2017-5638"]
        }
    
    @tracer.start_as_current_span("verify_slsa_compliance")
    async def verify_slsa_compliance(
        self,
        artifact_id: str,
        artifact_data: bytes,
        required_level: SLSALevel = SLSALevel.LEVEL_3
    ) -> VerificationResult:
        """
        Verify SLSA compliance for an artifact
        """
        span = trace.get_current_span()
        
        # Check cache
        if artifact_id in self.verification_cache:
            cached = self.verification_cache[artifact_id]
            if (datetime.utcnow() - cached.timestamp).seconds < 3600:
                return cached
        
        result = VerificationResult(
            verified=False,
            artifact_id=artifact_id,
            slsa_level=SLSALevel.LEVEL_0
        )
        
        # Calculate artifact hash
        artifact_hash = hashlib.sha256(artifact_data).hexdigest()
        
        # Get attestations
        attestations = await self._get_attestations(artifact_id)
        if not attestations:
            result.issues.append("No attestations found")
            span.set_attribute("slsa.level", 0)
            return result
        
        result.attestations = attestations
        
        # Verify provenance
        provenance_attestations = [
            a for a in attestations 
            if a.attestation_type == AttestationType.PROVENANCE
        ]
        
        if not provenance_attestations:
            result.issues.append("No provenance attestation found")
            result.slsa_level = SLSALevel.LEVEL_0
        else:
            # Verify provenance
            provenance_valid = await self._verify_provenance(
                provenance_attestations[0],
                artifact_hash
            )
            
            if not provenance_valid:
                result.issues.append("Invalid provenance")
                result.slsa_level = SLSALevel.LEVEL_0
            else:
                # Determine SLSA level based on provenance
                result.slsa_level = await self._determine_slsa_level(
                    provenance_attestations[0]
                )
        
        # Verify SBOM
        sbom_attestations = [
            a for a in attestations
            if a.attestation_type == AttestationType.SBOM
        ]
        
        if sbom_attestations:
            sbom = await self._verify_sbom(sbom_attestations[0])
            if sbom:
                result.sbom = sbom
                
                # Check for vulnerabilities
                vulns = await self._scan_vulnerabilities(sbom)
                result.vulnerabilities = vulns
                
                if vulns:
                    result.issues.append(f"Found {len(vulns)} vulnerabilities")
        
        # Check if meets required level
        result.verified = result.slsa_level.value >= required_level.value
        
        if not result.verified:
            result.issues.append(
                f"SLSA Level {result.slsa_level.value} does not meet required Level {required_level.value}"
            )
        
        # Cache result
        self.verification_cache[artifact_id] = result
        
        span.set_attributes({
            "slsa.verified": result.verified,
            "slsa.level": result.slsa_level.value,
            "slsa.issues": len(result.issues),
            "slsa.vulnerabilities": len(result.vulnerabilities)
        })
        
        return result
    
    async def _verify_provenance(
        self,
        attestation: Attestation,
        expected_hash: str
    ) -> bool:
        """Verify provenance attestation"""
        try:
            # Verify signature
            if not await self._verify_signature(attestation):
                return False
            
            # Check builder is trusted
            statement = attestation.statement
            builder_id = statement.predicate.get("builder", {}).get("id")
            
            if builder_id not in self.trusted_builders:
                print(f"Untrusted builder: {builder_id}")
                return False
            
            # Verify subject matches
            subjects = statement.subject
            if isinstance(subjects, dict):
                subjects = [subjects]
            
            for subject in subjects:
                if subject.get("digest", {}).get("sha256") == expected_hash:
                    return True
            
            return False
            
        except Exception as e:
            print(f"Provenance verification error: {e}")
            return False
    
    async def _determine_slsa_level(self, attestation: Attestation) -> SLSALevel:
        """Determine SLSA level from provenance"""
        statement = attestation.statement
        predicate = statement.predicate
        
        # Level 1: Documentation exists
        if predicate.get("buildType") and predicate.get("builder"):
            level = SLSALevel.LEVEL_1
        else:
            return SLSALevel.LEVEL_0
        
        # Level 2: Signed provenance
        if attestation.signature and await self._verify_signature(attestation):
            level = SLSALevel.LEVEL_2
        else:
            return level
        
        # Level 3: Hardened builds
        metadata = predicate.get("metadata", {})
        completeness = metadata.get("completeness", {})
        
        hardened_requirements = [
            completeness.get("parameters", False),
            completeness.get("materials", False),
            predicate.get("buildConfig", {}).get("hermetic", False),
            self._is_build_service_trusted(predicate.get("builder", {}).get("id"))
        ]
        
        if all(hardened_requirements):
            level = SLSALevel.LEVEL_3
        
        # Level 4: Two-party review (requires additional attestations)
        # Check for code review attestation
        # (Would need to check for multiple independent reviews)
        
        return level
    
    def _is_build_service_trusted(self, builder_id: str) -> bool:
        """Check if build service meets trust requirements"""
        # In production, check against policy
        trusted_services = [
            "https://github.com/slsa-framework/slsa-github-generator",
            "https://gitlab.com/gitlab-runner"
        ]
        return builder_id in trusted_services
    
    async def generate_provenance(
        self,
        artifact_id: str,
        artifact_data: bytes,
        build_config: Dict[str, Any],
        materials: List[Dict[str, Any]]
    ) -> Attestation:
        """Generate provenance attestation for an artifact"""
        
        # Calculate artifact digest
        artifact_hash = hashlib.sha256(artifact_data).hexdigest()
        
        # Create provenance statement
        statement = ProvenanceStatement(
            subject={
                "name": artifact_id,
                "digest": {"sha256": artifact_hash}
            }
        )
        
        statement.predicate.update({
            "builder": {
                "id": "https://github.com/your-org/build-system",
                "version": "1.0.0"
            },
            "buildType": "https://your-org.com/build-type/v1",
            "invocation": {
                "configSource": {
                    "uri": build_config.get("source_uri", ""),
                    "digest": {"sha256": build_config.get("source_hash", "")},
                    "entryPoint": build_config.get("entry_point", "")
                },
                "parameters": build_config.get("parameters", {}),
                "environment": build_config.get("environment", {})
            },
            "buildConfig": {
                "steps": build_config.get("steps", []),
                "hermetic": build_config.get("hermetic", False)
            },
            "metadata": {
                "buildStartedOn": build_config.get("started_at", datetime.utcnow().isoformat()),
                "buildFinishedOn": datetime.utcnow().isoformat(),
                "completeness": {
                    "parameters": True,
                    "environment": True,
                    "materials": True
                },
                "reproducible": build_config.get("reproducible", False)
            },
            "materials": materials
        })
        
        # Sign the statement
        statement_json = json.dumps(statement.__dict__, sort_keys=True)
        signature = await self._sign_statement(statement_json)
        
        # Create attestation
        attestation = Attestation(
            attestation_id=str(uuid.uuid4()),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.BINARY,
            attestation_type=AttestationType.PROVENANCE,
            statement=statement,
            signature=signature,
            signer_identity="build-system@your-org.com",
            timestamp=datetime.utcnow()
        )
        
        # Store attestation
        if artifact_id not in self.attestations:
            self.attestations[artifact_id] = []
        self.attestations[artifact_id].append(attestation)
        
        return attestation
    
    async def generate_sbom(
        self,
        artifact_id: str,
        components: List[SBOMComponent]
    ) -> Attestation:
        """Generate SBOM attestation"""
        
        # Create SBOM in SPDX format
        sbom = {
            "spdxVersion": "SPDX-2.3",
            "creationInfo": {
                "created": datetime.utcnow().isoformat(),
                "creators": ["Tool: agentic-ai-sbom-generator-1.0"]
            },
            "name": artifact_id,
            "packages": []
        }
        
        for component in components:
            package = {
                "name": component.name,
                "version": component.version,
                "supplier": component.supplier,
                "downloadLocation": component.metadata.get("download_url", "NOASSERTION"),
                "filesAnalyzed": False,
                "licenseConcluded": component.license,
                "copyrightText": "NOASSERTION",
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": component.type,
                        "referenceLocator": f"{component.type}:{component.name}@{component.version}"
                    }
                ],
                "checksums": [
                    {
                        "algorithm": component.hash_algorithm.upper(),
                        "value": component.hash_value
                    }
                ]
            }
            
            # Add vulnerability references
            for vuln in component.vulnerabilities:
                package["externalRefs"].append({
                    "referenceCategory": "SECURITY",
                    "referenceType": "cpe23Type",
                    "referenceLocator": vuln
                })
            
            sbom["packages"].append(package)
        
        # Add relationships
        sbom["relationships"] = []
        for component in components:
            for dep in component.dependencies:
                sbom["relationships"].append({
                    "spdxElementId": f"SPDXRef-Package-{component.name}",
                    "relationshipType": "DEPENDS_ON",
                    "relatedSpdxElement": f"SPDXRef-Package-{dep}"
                })
        
        # Sign SBOM
        sbom_json = json.dumps(sbom, sort_keys=True)
        signature = await self._sign_statement(sbom_json)
        
        # Create attestation
        attestation = Attestation(
            attestation_id=str(uuid.uuid4()),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.BINARY,
            attestation_type=AttestationType.SBOM,
            statement=sbom,
            signature=signature,
            signer_identity="sbom-generator@your-org.com",
            timestamp=datetime.utcnow()
        )
        
        # Store
        if artifact_id not in self.attestations:
            self.attestations[artifact_id] = []
        self.attestations[artifact_id].append(attestation)
        
        # Store components for quick lookup
        self.sboms[artifact_id] = components
        
        return attestation
    
    async def _scan_vulnerabilities(
        self,
        components: List[SBOMComponent]
    ) -> List[str]:
        """Scan components for known vulnerabilities"""
        vulnerabilities = []
        
        for component in components:
            # Check CVE database
            key = f"{component.name}:{component.version}"
            if key in self.cve_database:
                cves = self.cve_database[key]
                vulnerabilities.extend(cves)
                component.vulnerabilities.extend(cves)
        
        return vulnerabilities
    
    async def verify_before_load(
        self,
        artifact_id: str,
        artifact_data: bytes
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify artifact before loading/execution
        
        Returns:
            Tuple of (is_safe, error_message)
        """
        # Verify SLSA compliance
        result = await self.verify_slsa_compliance(
            artifact_id,
            artifact_data,
            required_level=SLSALevel.LEVEL_2  # Minimum for production
        )
        
        if not result.verified:
            return False, f"SLSA verification failed: {', '.join(result.issues)}"
        
        # Check for critical vulnerabilities
        if result.vulnerabilities:
            critical_vulns = [v for v in result.vulnerabilities if self._is_critical(v)]
            if critical_vulns:
                return False, f"Critical vulnerabilities found: {', '.join(critical_vulns)}"
        
        # Verify signature
        if result.attestations:
            sig_valid = await self._verify_signature(result.attestations[0])
            if not sig_valid:
                return False, "Invalid signature on attestation"
        
        return True, None
    
    def _is_critical(self, cve: str) -> bool:
        """Check if CVE is critical severity"""
        # In production, query CVE database for CVSS score
        critical_cves = ["CVE-2021-44228", "CVE-2014-0160"]
        return cve in critical_cves
    
    async def _sign_statement(self, statement: str) -> str:
        """Sign a statement"""
        signature = self.private_key.sign(
            statement.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode()
    
    async def _verify_signature(self, attestation: Attestation) -> bool:
        """Verify attestation signature"""
        try:
            # Get public key for signer
            public_key = self.trusted_keys.get(
                attestation.signer_identity,
                self.public_key  # Default to our key for testing
            )
            
            # Reconstruct statement
            if isinstance(attestation.statement, str):
                statement = attestation.statement
            else:
                statement = json.dumps(attestation.statement.__dict__, sort_keys=True)
            
            # Verify
            signature = base64.b64decode(attestation.signature)
            public_key.verify(
                signature,
                statement.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
            
        except Exception as e:
            print(f"Signature verification failed: {e}")
            return False
    
    async def _get_attestations(self, artifact_id: str) -> List[Attestation]:
        """Get attestations for an artifact"""
        return self.attestations.get(artifact_id, [])
    
    async def _verify_sbom(self, attestation: Attestation) -> Optional[List[SBOMComponent]]:
        """Verify and extract SBOM from attestation"""
        try:
            # Verify signature
            if not await self._verify_signature(attestation):
                return None
            
            # Parse SBOM
            sbom_data = attestation.statement
            components = []
            
            for package in sbom_data.get("packages", []):
                component = SBOMComponent(
                    name=package["name"],
                    version=package.get("version", "unknown"),
                    type=package.get("externalRefs", [{}])[0].get("referenceType", "unknown"),
                    supplier=package.get("supplier", "unknown"),
                    license=package.get("licenseConcluded", "unknown"),
                    hash_value=package.get("checksums", [{}])[0].get("value", ""),
                    hash_algorithm=package.get("checksums", [{}])[0].get("algorithm", "SHA256")
                )
                components.append(component)
            
            return components
            
        except Exception as e:
            print(f"SBOM verification error: {e}")
            return None
    
    async def monitor_supply_chain(self):
        """Continuously monitor supply chain for new vulnerabilities"""
        while True:
            try:
                # Check all stored SBOMs for new vulnerabilities
                for artifact_id, components in self.sboms.items():
                    new_vulns = await self._scan_vulnerabilities(components)
                    
                    if new_vulns:
                        print(f"New vulnerabilities found for {artifact_id}: {new_vulns}")
                        
                        # Trigger remediation workflow
                        await self._trigger_remediation(artifact_id, new_vulns)
                
            except Exception as e:
                print(f"Supply chain monitoring error: {e}")
            
            await asyncio.sleep(3600)  # Check hourly
    
    async def _trigger_remediation(self, artifact_id: str, vulnerabilities: List[str]):
        """Trigger remediation for vulnerabilities"""
        # In production, this would:
        # 1. Create tickets
        # 2. Notify security team
        # 3. Block deployment if critical
        # 4. Trigger rebuild with patched dependencies
        print(f"Triggering remediation for {artifact_id}: {vulnerabilities}")
    
    async def export_attestations(self, artifact_id: str) -> Dict[str, Any]:
        """Export all attestations for an artifact"""
        attestations = await self._get_attestations(artifact_id)
        
        return {
            "artifact_id": artifact_id,
            "attestations": [
                {
                    "id": att.attestation_id,
                    "type": att.attestation_type.value,
                    "signer": att.signer_identity,
                    "timestamp": att.timestamp.isoformat(),
                    "expires_at": att.expires_at.isoformat() if att.expires_at else None
                }
                for att in attestations
            ],
            "slsa_level": self.verification_cache.get(
                artifact_id,
                VerificationResult(False, artifact_id, SLSALevel.LEVEL_0)
            ).slsa_level.value,
            "sbom_available": artifact_id in self.sboms
        }


# Example usage
async def main():
    import uuid
    
    config = {}
    verifier = SLSAVerifier(config)
    
    # Generate sample artifact
    artifact_id = "app-binary-v1.0"
    artifact_data = b"binary content here"
    
    # Generate provenance
    provenance = await verifier.generate_provenance(
        artifact_id=artifact_id,
        artifact_data=artifact_data,
        build_config={
            "source_uri": "https://github.com/org/repo",
            "source_hash": "abc123",
            "hermetic": True,
            "reproducible": False
        },
        materials=[
            {
                "uri": "https://github.com/org/dep1",
                "digest": {"sha256": "def456"}
            }
        ]
    )
    
    print(f"Generated provenance: {provenance.attestation_id}")
    
    # Generate SBOM
    components = [
        SBOMComponent(
            name="log4j",
            version="2.17.0",  # Patched version
            type="maven",
            supplier="Apache",
            license="Apache-2.0",
            hash_value="abc123",
            hash_algorithm="sha256"
        ),
        SBOMComponent(
            name="commons-codec",
            version="1.15",
            type="maven",
            supplier="Apache",
            license="Apache-2.0",
            hash_value="def456",
            hash_algorithm="sha256"
        )
    ]
    
    sbom_att = await verifier.generate_sbom(artifact_id, components)
    print(f"Generated SBOM: {sbom_att.attestation_id}")
    
    # Verify SLSA compliance
    result = await verifier.verify_slsa_compliance(artifact_id, artifact_data)
    print(f"SLSA Level: {result.slsa_level.value}, Verified: {result.verified}")
    
    # Check before loading
    is_safe, error = await verifier.verify_before_load(artifact_id, artifact_data)
    print(f"Safe to load: {is_safe}, Error: {error}")
    
    # Export attestations
    export = await verifier.export_attestations(artifact_id)
    print(f"Attestations: {json.dumps(export, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
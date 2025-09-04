# execution/artifact_lineage.py
"""
Artifact Store with lineage tracking, signature verification,
and immutable run manifests for reproducibility.
"""

import json
import hashlib
import hmac
import uuid
import pickle
import gzip
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field, asdict
from pathlib import Path
import asyncio

import aioboto3
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature
from opentelemetry import trace

tracer = trace.get_tracer(__name__)


class ArtifactType(Enum):
    """Types of artifacts that can be stored"""
    MODEL_CHECKPOINT = "model_checkpoint"
    DATASET = "dataset"
    CODE = "code"
    CONFIGURATION = "configuration"
    EVALUATION_RESULT = "evaluation_result"
    RUN_MANIFEST = "run_manifest"
    GENERATED_OUTPUT = "generated_output"
    AUDIT_LOG = "audit_log"


class ArtifactStatus(Enum):
    """Status of an artifact"""
    PENDING = "pending"
    VALIDATED = "validated"
    SIGNED = "signed"
    QUARANTINED = "quarantined"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass
class ArtifactMetadata:
    """Metadata for an artifact"""
    artifact_id: str
    artifact_type: ArtifactType
    name: str
    version: str
    size_bytes: int
    content_hash: str
    created_at: datetime
    created_by: str
    tenant_id: str
    tags: Dict[str, str] = field(default_factory=dict)
    data_classification: Set[str] = field(default_factory=set)
    retention_policy: Optional[Dict[str, Any]] = None
    expiry_date: Optional[datetime] = None
    
    
@dataclass 
class LineageRecord:
    """Record of artifact lineage"""
    artifact_id: str
    parent_artifacts: List[str]  # IDs of parent artifacts
    run_manifest_id: str
    transformation_type: str
    transformation_params: Dict[str, Any]
    created_at: datetime
    signature: Optional[str] = None
    verification_status: Optional[str] = None


@dataclass
class RunManifest:
    """
    Immutable manifest of a task execution run.
    Contains all information needed to reproduce the run.
    """
    run_id: str
    task_id: str
    tenant_id: str
    user_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    
    # Component versions
    component_versions: Dict[str, str]
    
    # Model configurations
    models_used: List[Dict[str, Any]]
    
    # Input artifacts
    input_artifacts: List[str]
    
    # Output artifacts
    output_artifacts: List[str]
    
    # Execution settings
    settings: Dict[str, Any]
    
    # Seeds for reproducibility
    random_seeds: Dict[str, int]
    
    # Environment snapshot
    environment: Dict[str, str]
    
    # Metrics and results
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Signature for integrity
    signature: Optional[str] = None
    
    # Cache snapshot for replay
    cache_snapshot: Optional[Dict[str, Any]] = None


class SignatureManager:
    """
    Manages cryptographic signatures for artifacts
    """
    
    def __init__(self):
        # Generate RSA key pair for signing
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        
        # Shared secret for HMAC (in production, from secure vault)
        self.hmac_key = b"artifact_hmac_secret_key_change_in_production"
    
    def sign_artifact(self, content: bytes) -> str:
        """Sign artifact content with private key"""
        signature = self.private_key.sign(
            content,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature.hex()
    
    def verify_signature(self, content: bytes, signature: str) -> bool:
        """Verify artifact signature with public key"""
        try:
            self.public_key.verify(
                bytes.fromhex(signature),
                content,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            return False
            
    def generate_hmac(self, content: bytes) -> str:
        """Generate HMAC for content"""
        h = hmac.new(self.hmac_key, content, hashlib.sha256)
        return h.hexdigest()
    
    def verify_hmac(self, content: bytes, hmac_value: str) -> bool:
        """Verify HMAC for content"""
        expected = self.generate_hmac(content)
        return hmac.compare_digest(expected, hmac_value)


class ArtifactStore:
    """
    Main artifact store with lineage tracking and verification
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.s3_bucket = config["s3_bucket"]
        self.signature_manager = SignatureManager()
        self.metadata_store: Dict[str, ArtifactMetadata] = {}
        self.lineage_store: Dict[str, LineageRecord] = {}
        self.run_manifests: Dict[str, RunManifest] = {}
        self.s3_client = None
        
    async def initialize(self):
        """Initialize S3 client and load metadata"""
        session = aioboto3.Session()
        self.s3_client = await session.client('s3').__aenter__()
        
        # Load metadata from persistent store
        await self._load_metadata()
    
    @tracer.start_as_current_span("store_artifact")
    async def store_artifact(
        self,
        content: bytes,
        metadata: ArtifactMetadata,
        parent_artifacts: List[str] = [],
        run_manifest_id: Optional[str] = None,
        require_signature: bool = True
    ) -> str:
        """
        Store an artifact with lineage tracking
        
        Returns:
            Artifact ID
        """
        span = trace.get_current_span()
        
        try:
            # Validate content hash
            actual_hash = hashlib.sha256(content).hexdigest()
            if actual_hash != metadata.content_hash:
                raise ValueError(f"Content hash mismatch: expected {metadata.content_hash}, got {actual_hash}")
            
            # Check parent artifacts exist and are valid
            for parent_id in parent_artifacts:
                if not await self.verify_artifact(parent_id):
                    raise ValueError(f"Parent artifact {parent_id} failed verification")
            
            # Sign the artifact if required
            signature = None
            if require_signature:
                signature = self.signature_manager.sign_artifact(content)
                span.add_event("artifact_signed", {"artifact_id": metadata.artifact_id})
            
            # Compress content
            compressed = gzip.compress(content)
            
            # Store to S3
            s3_key = self._generate_s3_key(metadata)
            await self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=compressed,
                Metadata={
                    "artifact_id": metadata.artifact_id,
                    "content_hash": metadata.content_hash,
                    "signature": signature or "",
                    "tenant_id": metadata.tenant_id,
                    "classification": ",".join(metadata.data_classification)
                },
                ServerSideEncryption="AES256",
                StorageClass=self._get_storage_class(metadata)
            )
            
            # Create lineage record
            lineage = LineageRecord(
                artifact_id=metadata.artifact_id,
                parent_artifacts=parent_artifacts,
                run_manifest_id=run_manifest_id or "",
                transformation_type="store",
                transformation_params={},
                created_at=datetime.utcnow(),
                signature=signature
            )
            
            # Store metadata and lineage
            self.metadata_store[metadata.artifact_id] = metadata
            self.lineage_store[metadata.artifact_id] = lineage
            
            # Persist metadata
            await self._persist_metadata(metadata, lineage)
            
            span.set_attributes({
                "artifact.id": metadata.artifact_id,
                "artifact.type": metadata.artifact_type.value,
                "artifact.size": metadata.size_bytes,
                "artifact.signed": require_signature
            })
            
            return metadata.artifact_id
            
        except Exception as e:
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
    
    @tracer.start_as_current_span("retrieve_artifact")
    async def retrieve_artifact(
        self,
        artifact_id: str,
        verify_signature: bool = True,
        check_expiry: bool = True
    ) -> Tuple[bytes, ArtifactMetadata]:
        """
        Retrieve an artifact with verification
        
        Returns:
            Tuple of (content, metadata)
        """
        span = trace.get_current_span()
        
        # Get metadata
        if artifact_id not in self.metadata_store:
            raise ValueError(f"Artifact {artifact_id} not found")
        
        metadata = self.metadata_store[artifact_id]
        
        # Check expiry
        if check_expiry and metadata.expiry_date:
            if datetime.utcnow() > metadata.expiry_date:
                span.add_event("artifact_expired")
                raise ValueError(f"Artifact {artifact_id} has expired")
        
        # Get lineage
        lineage = self.lineage_store.get(artifact_id)
        
        # Retrieve from S3
        s3_key = self._generate_s3_key(metadata)
        response = await self.s3_client.get_object(
            Bucket=self.s3_bucket,
            Key=s3_key
        )
        
        compressed = await response['Body'].read()
        content = gzip.decompress(compressed)
        
        # Verify content hash
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != metadata.content_hash:
            span.add_event("hash_mismatch")
            raise ValueError(f"Content hash verification failed for {artifact_id}")
        
        # Verify signature if required
        if verify_signature and lineage and lineage.signature:
            if not self.signature_manager.verify_signature(content, lineage.signature):
                span.add_event("signature_verification_failed")
                raise ValueError(f"Signature verification failed for {artifact_id}")
        
        span.set_attributes({
            "artifact.id": artifact_id,
            "artifact.verified": verify_signature,
            "artifact.size": len(content)
        })
        
        return content, metadata
    
    async def verify_artifact(self, artifact_id: str) -> bool:
        """
        Verify artifact integrity and lineage
        """
        try:
            # Retrieve and verify
            content, metadata = await self.retrieve_artifact(
                artifact_id,
                verify_signature=True,
                check_expiry=True
            )
            
            # Verify lineage chain
            lineage = self.lineage_store.get(artifact_id)
            if lineage:
                # Verify all parent artifacts
                for parent_id in lineage.parent_artifacts:
                    if not await self.verify_artifact(parent_id):
                        return False
            
            return True
            
        except Exception:
            return False
    
    async def create_run_manifest(
        self,
        task_id: str,
        tenant_id: str,
        user_id: str,
        models_used: List[Dict[str, Any]],
        settings: Dict[str, Any]
    ) -> str:
        """
        Create an immutable run manifest for reproducibility
        """
        run_id = str(uuid.uuid4())
        
        manifest = RunManifest(
            run_id=run_id,
            task_id=task_id,
            tenant_id=tenant_id,
            user_id=user_id,
            started_at=datetime.utcnow(),
            completed_at=None,
            component_versions=await self._get_component_versions(),
            models_used=models_used,
            input_artifacts=[],
            output_artifacts=[],
            settings=settings,
            random_seeds=self._generate_seeds(),
            environment=await self._capture_environment()
        )
        
        # Sign the manifest
        manifest_bytes = json.dumps(asdict(manifest), default=str).encode()
        manifest.signature = self.signature_manager.sign_artifact(manifest_bytes)
        
        # Store manifest
        self.run_manifests[run_id] = manifest
        
        # Persist as artifact
        manifest_metadata = ArtifactMetadata(
            artifact_id=f"manifest_{run_id}",
            artifact_type=ArtifactType.RUN_MANIFEST,
            name=f"Run Manifest {run_id}",
            version="1.0",
            size_bytes=len(manifest_bytes),
            content_hash=hashlib.sha256(manifest_bytes).hexdigest(),
            created_at=datetime.utcnow(),
            created_by=user_id,
            tenant_id=tenant_id,
            tags={"type": "run_manifest", "task_id": task_id}
        )
        
        await self.store_artifact(
            manifest_bytes,
            manifest_metadata,
            require_signature=True
        )
        
        return run_id
    
    async def finalize_run_manifest(
        self,
        run_id: str,
        output_artifacts: List[str],
        metrics: Dict[str, Any]
    ):
        """
        Finalize a run manifest with outputs and metrics
        """
        if run_id not in self.run_manifests:
            raise ValueError(f"Run manifest {run_id} not found")
        
        manifest = self.run_manifests[run_id]
        manifest.completed_at = datetime.utcnow()
        manifest.output_artifacts = output_artifacts
        manifest.metrics = metrics
        
        # Re-sign with final data
        manifest_bytes = json.dumps(asdict(manifest), default=str).encode()
        manifest.signature = self.signature_manager.sign_artifact(manifest_bytes)
        
        # Update persisted version
        await self._update_manifest_artifact(run_id, manifest)
    
    async def replay_from_manifest(self, run_id: str) -> Dict[str, Any]:
        """
        Set up environment to replay a run from its manifest
        """
        # Retrieve manifest
        manifest_artifact_id = f"manifest_{run_id}"
        content, _ = await self.retrieve_artifact(manifest_artifact_id)
        manifest_data = json.loads(content.decode())
        
        # Verify manifest signature
        manifest = RunManifest(**manifest_data)
        
        # Prepare replay configuration
        replay_config = {
            "run_id": run_id,
            "original_run_id": manifest.run_id,
            "models": manifest.models_used,
            "settings": manifest.settings,
            "random_seeds": manifest.random_seeds,
            "input_artifacts": manifest.input_artifacts,
            "environment": manifest.environment,
            "cache_snapshot": manifest.cache_snapshot
        }
        
        # Verify all input artifacts are available
        for artifact_id in manifest.input_artifacts:
            if not await self.verify_artifact(artifact_id):
                raise ValueError(f"Input artifact {artifact_id} verification failed")
        
        return replay_config
    
    async def enforce_retention_policies(self):
        """
        Enforce data retention policies on artifacts
        """
        now = datetime.utcnow()
        
        for artifact_id, metadata in list(self.metadata_store.items()):
            if metadata.retention_policy:
                retention_days = metadata.retention_policy.get("retention_days", 365)
                purge_method = metadata.retention_policy.get("purge_method", "delete")
                
                age_days = (now - metadata.created_at).days
                
                if age_days > retention_days:
                    if purge_method == "delete":
                        await self._delete_artifact(artifact_id)
                    elif purge_method == "tombstone":
                        await self._tombstone_artifact(artifact_id)
                    elif purge_method == "archive":
                        await self._archive_artifact(artifact_id)
    
    async def generate_sbom(self, artifact_id: str) -> Dict[str, Any]:
        """
        Generate Software Bill of Materials for an artifact
        """
        if artifact_id not in self.metadata_store:
            raise ValueError(f"Artifact {artifact_id} not found")
        
        metadata = self.metadata_store[artifact_id]
        lineage = self.lineage_store.get(artifact_id, None)
        
        sbom = {
            "artifact_id": artifact_id,
            "name": metadata.name,
            "version": metadata.version,
            "type": metadata.artifact_type.value,
            "created_at": metadata.created_at.isoformat(),
            "created_by": metadata.created_by,
            "content_hash": metadata.content_hash,
            "dependencies": []
        }
        
        # Trace lineage to find all dependencies
        if lineage:
            for parent_id in lineage.parent_artifacts:
                parent_sbom = await self.generate_sbom(parent_id)
                sbom["dependencies"].append(parent_sbom)
        
        # Add signature if present
        if lineage and lineage.signature:
            sbom["signature"] = lineage.signature
            sbom["signature_verified"] = await self.verify_artifact(artifact_id)
        
        return sbom
    
    def _generate_s3_key(self, metadata: ArtifactMetadata) -> str:
        """Generate S3 key for artifact"""
        date_prefix = metadata.created_at.strftime("%Y/%m/%d")
        return f"{metadata.tenant_id}/{date_prefix}/{metadata.artifact_type.value}/{metadata.artifact_id}"
    
    def _get_storage_class(self, metadata: ArtifactMetadata) -> str:
        """Determine S3 storage class based on artifact type"""
        if metadata.artifact_type == ArtifactType.AUDIT_LOG:
            return "GLACIER"  # Long-term archive
        elif metadata.artifact_type == ArtifactType.MODEL_CHECKPOINT:
            return "STANDARD"  # Fast access needed
        else:
            return "STANDARD_IA"  # Infrequent access
    
    async def _get_component_versions(self) -> Dict[str, str]:
        """Get current component versions"""
        return {
            "orchestrator": "1.2.3",
            "execution_layer": "2.1.0",
            "policy_engine": "1.5.2",
            "artifact_store": "1.0.1"
        }
    
    def _generate_seeds(self) -> Dict[str, int]:
        """Generate random seeds for reproducibility"""
        import random
        base_seed = random.randint(0, 2**32 - 1)
        return {
            "numpy": base_seed,
            "torch": base_seed + 1,
            "random": base_seed + 2,
            "tensorflow": base_seed + 3
        }
    
    async def _capture_environment(self) -> Dict[str, str]:
        """Capture environment for reproducibility"""
        import platform
        import sys
        
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.architecture()[0],
            "hostname": platform.node()
        }
    
    async def _load_metadata(self):
        """Load metadata from persistent store"""
        # In production, load from database
        pass
    
    async def _persist_metadata(
        self,
        metadata: ArtifactMetadata,
        lineage: LineageRecord
    ):
        """Persist metadata to database"""
        # In production, save to PostgreSQL
        pass
    
    async def _update_manifest_artifact(
        self,
        run_id: str,
        manifest: RunManifest
    ):
        """Update persisted manifest artifact"""
        manifest_bytes = json.dumps(asdict(manifest), default=str).encode()
        
        # Update in S3
        s3_key = f"{manifest.tenant_id}/manifests/{run_id}"
        await self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=s3_key,
            Body=gzip.compress(manifest_bytes),
            ServerSideEncryption="AES256"
        )
    
    async def _delete_artifact(self, artifact_id: str):
        """Delete an artifact"""
        metadata = self.metadata_store[artifact_id]
        s3_key = self._generate_s3_key(metadata)
        
        await self.s3_client.delete_object(
            Bucket=self.s3_bucket,
            Key=s3_key
        )
        
        del self.metadata_store[artifact_id]
        if artifact_id in self.lineage_store:
            del self.lineage_store[artifact_id]
    
    async def _tombstone_artifact(self, artifact_id: str):
        """Mark artifact as deleted but keep metadata"""
        self.metadata_store[artifact_id].tags["tombstoned"] = "true"
        self.metadata_store[artifact_id].tags["tombstoned_at"] = datetime.utcnow().isoformat()
    
    async def _archive_artifact(self, artifact_id: str):
        """Move artifact to archive storage"""
        metadata = self.metadata_store[artifact_id]
        s3_key = self._generate_s3_key(metadata)
        
        # Change storage class to Glacier
        await self.s3_client.copy_object(
            Bucket=self.s3_bucket,
            CopySource=f"{self.s3_bucket}/{s3_key}",
            Key=s3_key,
            StorageClass="GLACIER"
        )


# Example usage
async def main():
    """Example usage of artifact store"""
    config = {
        "s3_bucket": "agentic-ai-artifacts",
        "region": "us-east-1"
    }
    
    store = ArtifactStore(config)
    await store.initialize()
    
    # Create a run manifest
    run_id = await store.create_run_manifest(
        task_id="task-123",
        tenant_id="tenant-456",
        user_id="user-789",
        models_used=[
            {"name": "gpt-5", "version": "1.0", "provider": "openai"}
        ],
        settings={
            "temperature": 0.7,
            "max_tokens": 2000,
            "timeout": 30
        }
    )
    
    print(f"Created run manifest: {run_id}")
    
    # Store an artifact
    content = b"Generated code output"
    metadata = ArtifactMetadata(
        artifact_id=str(uuid.uuid4()),
        artifact_type=ArtifactType.GENERATED_OUTPUT,
        name="Generated Python Code",
        version="1.0",
        size_bytes=len(content),
        content_hash=hashlib.sha256(content).hexdigest(),
        created_at=datetime.utcnow(),
        created_by="user-789",
        tenant_id="tenant-456",
        data_classification={"PUBLIC"},
        retention_policy={
            "retention_days": 90,
            "purge_method": "delete"
        }
    )
    
    artifact_id = await store.store_artifact(
        content,
        metadata,
        run_manifest_id=run_id
    )
    
    print(f"Stored artifact: {artifact_id}")
    
    # Verify and retrieve
    verified = await store.verify_artifact(artifact_id)
    print(f"Artifact verified: {verified}")
    
    # Generate SBOM
    sbom = await store.generate_sbom(artifact_id)
    print(f"SBOM: {json.dumps(sbom, indent=2, default=str)}")
    
    # Finalize run manifest
    await store.finalize_run_manifest(
        run_id,
        output_artifacts=[artifact_id],
        metrics={"execution_time_ms": 150, "tokens_used": 500}
    )
    
    # Set up replay
    replay_config = await store.replay_from_manifest(run_id)
    print(f"Replay config: {replay_config}")


if __name__ == "__main__":
    asyncio.run(main())
# execution/state_ledger.py
import json
import os
import threading
from typing import Dict, Any, Optional, Set
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)

class StateLedger:
    """
    Distributed state ledger with token invalidation and idempotent apply guards.
    Ensures exactly-once execution at apply phase.
    """
    
    def __init__(self, ledger_path: str = "/tmp/state_ledger"):
        self.ledger_path = ledger_path
        self.lock = threading.Lock()
        self.applied_tokens: Set[str] = set()
        self.invalidated_tokens: Set[str] = set()
        self.run_manifests: Dict[str, Dict[str, Any]] = {}
        
        # Create ledger directory
        os.makedirs(ledger_path, exist_ok=True)
        self._load_state()
    
    def _load_state(self):
        """Load persisted state from disk"""
        try:
            # Load applied tokens
            applied_file = os.path.join(self.ledger_path, "applied_tokens.json")
            if os.path.exists(applied_file):
                with open(applied_file, 'r') as f:
                    self.applied_tokens = set(json.load(f))
            
            # Load invalidated tokens
            invalid_file = os.path.join(self.ledger_path, "invalidated_tokens.json")
            if os.path.exists(invalid_file):
                with open(invalid_file, 'r') as f:
                    self.invalidated_tokens = set(json.load(f))
                    
        except Exception as e:
            logger.error(f"Failed to load ledger state: {e}")
    
    def _persist_state(self):
        """Persist state to disk for crash recovery"""
        try:
            # Save applied tokens
            applied_file = os.path.join(self.ledger_path, "applied_tokens.json")
            with open(applied_file, 'w') as f:
                json.dump(list(self.applied_tokens), f)
            
            # Save invalidated tokens
            invalid_file = os.path.join(self.ledger_path, "invalidated_tokens.json")
            with open(invalid_file, 'w') as f:
                json.dump(list(self.invalidated_tokens), f)
                
        except Exception as e:
            logger.error(f"Failed to persist ledger state: {e}")
    
    def generate_apply_token(self, idempotency_key: str, run_id: str) -> str:
        """
        Generate unique apply token for idempotent execution.
        
        Args:
            idempotency_key: Client-provided idempotency key
            run_id: Unique run identifier
            
        Returns:
            Apply token for this execution
        """
        token_data = f"{idempotency_key}:{run_id}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(token_data.encode()).hexdigest()
    
    def check_idempotency(self, idempotency_key: str) -> Optional[str]:
        """
        Check if idempotency key has been seen before.
        
        Args:
            idempotency_key: Client-provided idempotency key
            
        Returns:
            Existing run_id if found, None otherwise
        """
        with self.lock:
            for run_id, manifest in self.run_manifests.items():
                if manifest.get("idempotency_key") == idempotency_key:
                    logger.info(f"Found existing run {run_id} for idempotency key {idempotency_key}")
                    return run_id
            return None
    
    def begin_apply(self, apply_token: str, run_id: str, operation: str) -> bool:
        """
        Begin apply phase with idempotency guard.
        
        Args:
            apply_token: Token for this apply operation
            run_id: Run identifier
            operation: Operation being applied
            
        Returns:
            True if apply should proceed, False if already applied
        """
        with self.lock:
            # Check if token is invalidated
            if apply_token in self.invalidated_tokens:
                logger.warning(f"Token {apply_token} is invalidated")
                return False
            
            # Check if already applied
            if apply_token in self.applied_tokens:
                logger.info(f"Token {apply_token} already applied - skipping")
                return False
            
            # Mark as in-progress
            logger.info(f"Beginning apply for token {apply_token}, run {run_id}, op {operation}")
            
            # Create apply record
            apply_record = {
                "token": apply_token,
                "run_id": run_id,
                "operation": operation,
                "started_at": datetime.utcnow().isoformat(),
                "status": "in_progress"
            }
            
            # Write to ledger
            apply_file = os.path.join(self.ledger_path, f"apply_{apply_token}.json")
            with open(apply_file, 'w') as f:
                json.dump(apply_record, f)
            
            return True
    
    def commit_apply(self, apply_token: str, result: Dict[str, Any]) -> bool:
        """
        Commit successful apply operation.
        
        Args:
            apply_token: Token for this apply operation
            result: Result of the apply operation
            
        Returns:
            True if committed successfully
        """
        with self.lock:
            if apply_token in self.applied_tokens:
                logger.warning(f"Token {apply_token} already committed")
                return False
            
            # Add to applied set
            self.applied_tokens.add(apply_token)
            
            # Update apply record
            apply_file = os.path.join(self.ledger_path, f"apply_{apply_token}.json")
            if os.path.exists(apply_file):
                with open(apply_file, 'r') as f:
                    record = json.load(f)
                
                record["status"] = "committed"
                record["completed_at"] = datetime.utcnow().isoformat()
                record["result"] = result
                
                with open(apply_file, 'w') as f:
                    json.dump(record, f)
            
            # Persist state
            self._persist_state()
            
            logger.info(f"Committed apply for token {apply_token}")
            return True
    
    def rollback_apply(self, apply_token: str, reason: str) -> bool:
        """
        Rollback failed apply operation.
        
        Args:
            apply_token: Token for this apply operation
            reason: Reason for rollback
            
        Returns:
            True if rolled back successfully
        """
        with self.lock:
            # Invalidate token
            self.invalidated_tokens.add(apply_token)
            
            # Update apply record
            apply_file = os.path.join(self.ledger_path, f"apply_{apply_token}.json")
            if os.path.exists(apply_file):
                with open(apply_file, 'r') as f:
                    record = json.load(f)
                
                record["status"] = "rolled_back"
                record["rolled_back_at"] = datetime.utcnow().isoformat()
                record["rollback_reason"] = reason
                
                with open(apply_file, 'w') as f:
                    json.dump(record, f)
            
            # Persist state
            self._persist_state()
            
            logger.info(f"Rolled back apply for token {apply_token}: {reason}")
            return True
    
    def write_run_manifest(self, run_id: str, manifest: Dict[str, Any]):
        """
        Write run manifest with approvals and policy_mode.
        
        Args:
            run_id: Run identifier
            manifest: Complete run manifest including approvals and policy_mode
        """
        with self.lock:
            # Ensure required fields
            manifest["run_id"] = run_id
            manifest["written_at"] = datetime.utcnow().isoformat()
            
            # Store in memory
            self.run_manifests[run_id] = manifest
            
            # Persist to disk
            manifest_file = os.path.join(self.ledger_path, f"manifest_{run_id}.json")
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"Wrote manifest for run {run_id} with approvals: {manifest.get('approvals', [])} and policy_mode: {manifest.get('policy_mode')}")
    
    def get_run_manifest(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve run manifest.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Run manifest if found
        """
        with self.lock:
            # Check memory cache
            if run_id in self.run_manifests:
                return self.run_manifests[run_id]
            
            # Load from disk
            manifest_file = os.path.join(self.ledger_path, f"manifest_{run_id}.json")
            if os.path.exists(manifest_file):
                with open(manifest_file, 'r') as f:
                    manifest = json.load(f)
                    self.run_manifests[run_id] = manifest
                    return manifest
            
            return None
    
    def cleanup_expired_tokens(self, max_age_seconds: int = 3600):
        """
        Clean up expired tokens and records.
        
        Args:
            max_age_seconds: Maximum age for tokens before cleanup
        """
        with self.lock:
            current_time = datetime.utcnow()
            expired_tokens = []
            
            # Check apply records
            for token in list(self.applied_tokens):
                apply_file = os.path.join(self.ledger_path, f"apply_{token}.json")
                if os.path.exists(apply_file):
                    with open(apply_file, 'r') as f:
                        record = json.load(f)
                    
                    if "completed_at" in record:
                        completed_time = datetime.fromisoformat(record["completed_at"].replace("Z", ""))
                        age = (current_time - completed_time).total_seconds()
                        
                        if age > max_age_seconds:
                            expired_tokens.append(token)
            
            # Remove expired tokens
            for token in expired_tokens:
                self.applied_tokens.discard(token)
                self.invalidated_tokens.discard(token)
                
                # Remove file
                apply_file = os.path.join(self.ledger_path, f"apply_{token}.json")
                if os.path.exists(apply_file):
                    os.remove(apply_file)
            
            if expired_tokens:
                self._persist_state()
                logger.info(f"Cleaned up {len(expired_tokens)} expired tokens")
    
    def get_apply_history(self, run_id: str) -> list[Dict[str, Any]]:
        """
        Get apply history for a run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            List of apply records for this run
        """
        history = []
        
        # Scan apply records
        for filename in os.listdir(self.ledger_path):
            if filename.startswith("apply_"):
                filepath = os.path.join(self.ledger_path, filename)
                with open(filepath, 'r') as f:
                    record = json.load(f)
                    if record.get("run_id") == run_id:
                        history.append(record)
        
        # Sort by start time
        history.sort(key=lambda x: x.get("started_at", ""))
        return history
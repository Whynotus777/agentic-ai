"""
State Ledger for Exactly-Once-At-Apply Semantics

Implements prepare→apply→commit pattern for deterministic execution.
"""
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from threading import Lock
import logging

logger = logging.getLogger(__name__)

class LedgerState(Enum):
    """Execution states in the ledger"""
    PREPARING = "preparing"
    PREPARED = "prepared"
    APPLYING = "applying"
    APPLIED = "applied"
    COMMITTED = "committed"
    FAILED = "failed"

@dataclass
class LedgerEntry:
    """Single entry in the state ledger"""
    run_id: str
    idempotency_key: str
    input_sha: str
    state: LedgerState
    token: Optional[str]
    diff_hash: Optional[str]
    manifest_path: Optional[str]
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    attempts: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['state'] = self.state.value
        d['created_at'] = self.created_at.isoformat()
        d['updated_at'] = self.updated_at.isoformat()
        d['expires_at'] = self.expires_at.isoformat()
        return d

class PrepareToken:
    """Token for apply phase authorization"""
    def __init__(self, token: str, run_id: str, expires_at: datetime):
        self.token = token
        self.run_id = run_id
        self.expires_at = expires_at
        self.used = False

    def is_valid(self) -> bool:
        return not self.used and datetime.utcnow() < self.expires_at

class StateLedger:
    """
    Distributed state ledger for exactly-once-at-apply semantics.
    
    The ledger ensures that each idempotent operation is applied exactly once,
    even in the presence of retries, failures, and concurrent requests.
    """
    
    def __init__(self, 
                 storage_path: Path = Path("/var/lib/execution/ledger"),
                 idempotency_window_hours: int = 24,
                 token_ttl_seconds: int = 300):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.idempotency_window = timedelta(hours=idempotency_window_hours)
        self.token_ttl = timedelta(seconds=token_ttl_seconds)
        
        # In-memory caches with locks for thread safety
        self._entries: Dict[str, LedgerEntry] = {}  # idempotency_key -> entry
        self._tokens: Dict[str, PrepareToken] = {}  # token -> PrepareToken
        self._lock = Lock()
        
        # Load persistent state
        self._load_state()
    
    def prepare(self, idempotency_key: str, input_sha: str, 
                metadata: Optional[Dict[str, Any]] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Prepare phase: Check for existing execution or create new entry.
        
        Returns:
            (token, None) if new execution should proceed
            (None, run_id) if execution already exists
        """
        with self._lock:
            # Clean expired entries
            self._cleanup_expired()
            
            # Check for existing entry
            if idempotency_key in self._entries:
                entry = self._entries[idempotency_key]
                
                # Verify input consistency
                if entry.input_sha != input_sha:
                    raise ValueError(f"Input mismatch for idempotency key {idempotency_key}")
                
                # Return existing run_id if already processed
                if entry.state in [LedgerState.COMMITTED, LedgerState.APPLIED]:
                    logger.info(f"Idempotency hit for {idempotency_key}, run_id={entry.run_id}")
                    return None, entry.run_id
                
                # Handle retry of failed execution
                if entry.state == LedgerState.FAILED:
                    logger.info(f"Retrying failed execution {idempotency_key}")
                    entry.attempts += 1
                else:
                    # In-progress execution
                    logger.warning(f"Execution in progress for {idempotency_key}, state={entry.state}")
                    return None, entry.run_id
            
            # Create new entry
            run_id = f"run_{uuid.uuid4().hex[:12]}"
            token_str = f"tok_{uuid.uuid4().hex}"
            expires_at = datetime.utcnow() + self.idempotency_window
            token_expires = datetime.utcnow() + self.token_ttl
            
            entry = LedgerEntry(
                run_id=run_id,
                idempotency_key=idempotency_key,
                input_sha=input_sha,
                state=LedgerState.PREPARED,
                token=token_str,
                diff_hash=None,
                manifest_path=None,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                expires_at=expires_at,
                metadata=metadata or {}
            )
            
            # Store entry and token
            self._entries[idempotency_key] = entry
            self._tokens[token_str] = PrepareToken(token_str, run_id, token_expires)
            
            # Persist state
            self._persist_entry(entry)
            
            logger.info(f"Prepared execution: run_id={run_id}, token={token_str[:8]}...")
            return token_str, None
    
    def apply(self, token: str, diff_hash: str) -> str:
        """
        Apply phase: Execute the operation once with token validation.
        
        Returns:
            run_id if successful
        
        Raises:
            ValueError if token is invalid or already used
        """
        with self._lock:
            # Validate token
            if token not in self._tokens:
                raise ValueError("Invalid token")
            
            prepare_token = self._tokens[token]
            if not prepare_token.is_valid():
                raise ValueError("Token expired or already used")
            
            # Find entry by run_id
            entry = None
            for e in self._entries.values():
                if e.run_id == prepare_token.run_id:
                    entry = e
                    break
            
            if not entry:
                raise ValueError(f"Entry not found for run_id {prepare_token.run_id}")
            
            # Verify state transition
            if entry.state != LedgerState.PREPARED:
                raise ValueError(f"Invalid state for apply: {entry.state}")
            
            # Mark token as used
            prepare_token.used = True
            
            # Update entry
            entry.state = LedgerState.APPLIED
            entry.diff_hash = diff_hash
            entry.updated_at = datetime.utcnow()
            
            # Persist state
            self._persist_entry(entry)
            
            # Clean up used token
            del self._tokens[token]
            
            logger.info(f"Applied execution: run_id={entry.run_id}, diff_hash={diff_hash[:8]}...")
            return entry.run_id
    
    def commit(self, run_id: str, manifest_path: str) -> bool:
        """
        Commit phase: Finalize the execution with manifest reference.
        
        Returns:
            True if successful
        """
        with self._lock:
            # Find entry by run_id
            entry = None
            for e in self._entries.values():
                if e.run_id == run_id:
                    entry = e
                    break
            
            if not entry:
                raise ValueError(f"Entry not found for run_id {run_id}")
            
            # Verify state transition
            if entry.state != LedgerState.APPLIED:
                raise ValueError(f"Invalid state for commit: {entry.state}")
            
            # Update entry
            entry.state = LedgerState.COMMITTED
            entry.manifest_path = manifest_path
            entry.updated_at = datetime.utcnow()
            
            # Persist state
            self._persist_entry(entry)
            
            logger.info(f"Committed execution: run_id={run_id}, manifest={manifest_path}")
            return True
    
    def get_entry(self, idempotency_key: str) -> Optional[LedgerEntry]:
        """Get ledger entry by idempotency key"""
        with self._lock:
            return self._entries.get(idempotency_key)
    
    def get_entry_by_run_id(self, run_id: str) -> Optional[LedgerEntry]:
        """Get ledger entry by run_id"""
        with self._lock:
            for entry in self._entries.values():
                if entry.run_id == run_id:
                    return entry
            return None
    
    def mark_failed(self, run_id: str, error: str):
        """Mark an execution as failed"""
        with self._lock:
            entry = self.get_entry_by_run_id(run_id)
            if entry:
                entry.state = LedgerState.FAILED
                entry.error = error
                entry.updated_at = datetime.utcnow()
                self._persist_entry(entry)
                
                # Clean up any associated tokens
                tokens_to_remove = [t for t, pt in self._tokens.items() 
                                   if pt.run_id == run_id]
                for token in tokens_to_remove:
                    del self._tokens[token]
    
    def _cleanup_expired(self):
        """Remove expired entries from memory"""
        now = datetime.utcnow()
        expired_keys = [
            key for key, entry in self._entries.items()
            if entry.expires_at < now
        ]
        for key in expired_keys:
            del self._entries[key]
            
        # Clean up expired tokens
        expired_tokens = [
            token for token, pt in self._tokens.items()
            if pt.expires_at < now
        ]
        for token in expired_tokens:
            del self._tokens[token]
    
    def _persist_entry(self, entry: LedgerEntry):
        """Persist entry to disk"""
        file_path = self.storage_path / f"{entry.idempotency_key}.json"
        with open(file_path, 'w') as f:
            json.dump(entry.to_dict(), f, indent=2)
    
    def _load_state(self):
        """Load persistent state from disk"""
        if not self.storage_path.exists():
            return
            
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct entry
                entry = LedgerEntry(
                    run_id=data['run_id'],
                    idempotency_key=data['idempotency_key'],
                    input_sha=data['input_sha'],
                    state=LedgerState(data['state']),
                    token=data.get('token'),
                    diff_hash=data.get('diff_hash'),
                    manifest_path=data.get('manifest_path'),
                    created_at=datetime.fromisoformat(data['created_at']),
                    updated_at=datetime.fromisoformat(data['updated_at']),
                    expires_at=datetime.fromisoformat(data['expires_at']),
                    attempts=data.get('attempts', 0),
                    error=data.get('error'),
                    metadata=data.get('metadata', {})
                )
                
                # Only load non-expired entries
                if entry.expires_at > datetime.utcnow():
                    self._entries[entry.idempotency_key] = entry
                else:
                    # Clean up expired file
                    file_path.unlink()
                    
            except Exception as e:
                logger.error(f"Failed to load ledger entry from {file_path}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ledger statistics"""
        with self._lock:
            state_counts = {}
            for entry in self._entries.values():
                state_counts[entry.state.value] = state_counts.get(entry.state.value, 0) + 1
            
            return {
                "total_entries": len(self._entries),
                "active_tokens": len(self._tokens),
                "state_distribution": state_counts,
                "oldest_entry": min((e.created_at for e in self._entries.values()), 
                                   default=None),
                "newest_entry": max((e.created_at for e in self._entries.values()), 
                                   default=None)
            }
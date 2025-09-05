"""
Tests for State Ledger - Exactly-Once-At-Apply Semantics

Tests the prepare→apply→commit pattern and idempotency guarantees.
"""
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import time
import threading
from unittest.mock import patch, MagicMock

from execution.state_ledger import StateLedger, LedgerState, PrepareToken

class TestStateLedger:
    """Test suite for StateLedger"""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def ledger(self, temp_storage):
        """Create ledger instance with temp storage"""
        return StateLedger(
            storage_path=temp_storage,
            idempotency_window_hours=1,
            token_ttl_seconds=60
        )
    
    # ===== Positive Test Cases =====
    
    def test_prepare_new_execution(self, ledger):
        """Test preparing a new execution returns token"""
        token, run_id = ledger.prepare(
            idempotency_key="test-key-1",
            input_sha="sha123",
            metadata={"test": "data"}
        )
        
        assert token is not None
        assert run_id is None
        assert token.startswith("tok_")
        
        # Verify entry created
        entry = ledger.get_entry("test-key-1")
        assert entry is not None
        assert entry.state == LedgerState.PREPARED
        assert entry.input_sha == "sha123"
        assert entry.metadata == {"test": "data"}
    
    def test_prepare_idempotency_hit(self, ledger):
        """Test prepare with existing key returns run_id"""
        # First execution
        token1, _ = ledger.prepare("test-key-2", "sha456")
        run_id1 = ledger.apply(token1, "diff123")
        ledger.commit(run_id1, "/path/to/manifest")
        
        # Second execution with same key
        token2, run_id2 = ledger.prepare("test-key-2", "sha456")
        
        assert token2 is None
        assert run_id2 == run_id1
    
    def test_apply_with_valid_token(self, ledger):
        """Test apply phase with valid token"""
        token, _ = ledger.prepare("test-key-3", "sha789")
        
        run_id = ledger.apply(token, "diff456")
        
        assert run_id is not None
        assert run_id.startswith("run_")
        
        # Verify state updated
        entry = ledger.get_entry("test-key-3")
        assert entry.state == LedgerState.APPLIED
        assert entry.diff_hash == "diff456"
    
    def test_commit_completes_execution(self, ledger):
        """Test commit phase completes execution"""
        token, _ = ledger.prepare("test-key-4", "sha101")
        run_id = ledger.apply(token, "diff789")
        
        result = ledger.commit(run_id, "/manifests/test.json")
        
        assert result is True
        
        # Verify final state
        entry = ledger.get_entry("test-key-4")
        assert entry.state == LedgerState.COMMITTED
        assert entry.manifest_path == "/manifests/test.json"
    
    def test_exactly_once_guarantee(self, ledger):
        """Test exactly-once-at-apply guarantee"""
        token, _ = ledger.prepare("test-key-5", "sha202")
        
        # First apply succeeds
        run_id = ledger.apply(token, "diff101")
        assert run_id is not None
        
        # Second apply with same token fails
        with pytest.raises(ValueError, match="Token expired or already used"):
            ledger.apply(token, "diff101")
    
    def test_concurrent_prepare_same_key(self, ledger):
        """Test concurrent prepare calls with same key"""
        results = []
        
        def prepare_task():
            token, run_id = ledger.prepare("concurrent-key", "sha303")
            results.append((token, run_id))
        
        # Start multiple threads
        threads = [threading.Thread(target=prepare_task) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Only one should get a token
        tokens = [r[0] for r in results if r[0] is not None]
        run_ids = [r[1] for r in results if r[1] is not None]
        
        assert len(tokens) <= 1  # At most one token issued
    
    def test_retry_failed_execution(self, ledger):
        """Test retrying a failed execution"""
        # First attempt fails
        token1, _ = ledger.prepare("retry-key", "sha404")
        run_id1 = ledger.apply(token1, "diff202")
        ledger.mark_failed(run_id1, "Network error")
        
        # Verify marked as failed
        entry = ledger.get_entry("retry-key")
        assert entry.state == LedgerState.FAILED
        assert entry.error == "Network error"
        
        # Retry should work
        token2, _ = ledger.prepare("retry-key", "sha404")
        assert token2 is not None  # New token for retry
    
    def test_persistence_across_restarts(self, temp_storage):
        """Test state persistence across ledger restarts"""
        # First ledger instance
        ledger1 = StateLedger(storage_path=temp_storage)
        token, _ = ledger1.prepare("persist-key", "sha505")
        run_id = ledger1.apply(token, "diff303")
        ledger1.commit(run_id, "/manifest.json")
        
        # Simulate restart - create new instance
        ledger2 = StateLedger(storage_path=temp_storage)
        
        # Should find existing entry
        entry = ledger2.get_entry("persist-key")
        assert entry is not None
        assert entry.state == LedgerState.COMMITTED
        assert entry.run_id == run_id
    
    def test_token_expiration(self, ledger):
        """Test token expiration after TTL"""
        # Create ledger with short TTL
        ledger = StateLedger(
            storage_path=ledger.storage_path,
            token_ttl_seconds=1
        )
        
        token, _ = ledger.prepare("expire-key", "sha606")
        
        # Wait for token to expire
        time.sleep(1.5)
        
        # Apply should fail with expired token
        with pytest.raises(ValueError, match="Token expired"):
            ledger.apply(token, "diff404")
    
    def test_get_stats(self, ledger):
        """Test statistics collection"""
        # Create some entries
        for i in range(3):
            token, _ = ledger.prepare(f"stats-key-{i}", f"sha{i}")
            if i < 2:  # Complete first two
                run_id = ledger.apply(token, f"diff{i}")
                if i == 0:
                    ledger.commit(run_id, f"/manifest{i}.json")
        
        stats = ledger.get_stats()
        
        assert stats["total_entries"] == 3
        assert stats["state_distribution"][LedgerState.COMMITTED.value] == 1
        assert stats["state_distribution"][LedgerState.APPLIED.value] == 1
        assert stats["state_distribution"][LedgerState.PREPARED.value] == 1
    
    # ===== Negative Test Cases =====
    
    def test_prepare_input_mismatch(self, ledger):
        """Test prepare with different input hash for same key"""
        ledger.prepare("mismatch-key", "sha_original")
        
        # Try with different input hash
        with pytest.raises(ValueError, match="Input mismatch"):
            ledger.prepare("mismatch-key", "sha_different")
    
    def test_apply_invalid_token(self, ledger):
        """Test apply with invalid token"""
        with pytest.raises(ValueError, match="Invalid token"):
            ledger.apply("invalid_token", "diff505")
    
    def test_apply_wrong_state(self, ledger):
        """Test apply when not in PREPARED state"""
        token, _ = ledger.prepare("wrong-state", "sha707")
        run_id = ledger.apply(token, "diff606")
        
        # Try to apply again after already applied
        token2, _ = ledger.prepare("wrong-state-2", "sha708")
        run_id2 = ledger.apply(token2, "diff607")
        ledger.commit(run_id2, "/manifest.json")
        
        # Can't apply to committed entry
        with pytest.raises(ValueError, match="Invalid state for apply"):
            ledger.apply("some_token", "diff608")
    
    def test_commit_invalid_run_id(self, ledger):
        """Test commit with invalid run_id"""
        with pytest.raises(ValueError, match="Entry not found"):
            ledger.commit("invalid_run_id", "/manifest.json")
    
    def test_commit_wrong_state(self, ledger):
        """Test commit when not in APPLIED state"""
        token, _ = ledger.prepare("commit-wrong", "sha808")
        
        # Try to commit before apply
        with pytest.raises(ValueError, match="Entry not found"):
            ledger.commit("some_run_id", "/manifest.json")
    
    def test_token_reuse_prevented(self, ledger):
        """Test that used tokens cannot be reused"""
        token, _ = ledger.prepare("reuse-key", "sha909")
        
        # Use the token
        run_id = ledger.apply(token, "diff707")
        
        # Try to use again
        with pytest.raises(ValueError, match="Invalid token"):
            ledger.apply(token, "diff708")
    
    def test_expired_entries_cleanup(self, ledger):
        """Test expired entries are cleaned up"""
        # Create ledger with very short window
        ledger = StateLedger(
            storage_path=ledger.storage_path,
            idempotency_window_hours=0.0003  # ~1 second
        )
        
        token, _ = ledger.prepare("expire-entry", "sha010")
        
        # Wait for entry to expire
        time.sleep(1.5)
        
        # Trigger cleanup
        ledger._cleanup_expired()
        
        # Entry should be gone
        entry = ledger.get_entry("expire-entry")
        assert entry is None
    
    def test_get_entry_by_run_id(self, ledger):
        """Test retrieving entry by run_id"""
        token, _ = ledger.prepare("lookup-key", "sha111")
        run_id = ledger.apply(token, "diff808")
        
        entry = ledger.get_entry_by_run_id(run_id)
        assert entry is not None
        assert entry.idempotency_key == "lookup-key"
        assert entry.run_id == run_id
        
        # Non-existent run_id
        assert ledger.get_entry_by_run_id("nonexistent") is None
    
    def test_mark_failed_cleans_tokens(self, ledger):
        """Test marking failed cleans up associated tokens"""
        token, _ = ledger.prepare("fail-clean", "sha212")
        
        # Get run_id without applying (simulate failure before apply)
        entry = ledger.get_entry("fail-clean")
        run_id = entry.run_id
        
        # Mark as failed
        ledger.mark_failed(run_id, "Simulated failure")
        
        # Token should be cleaned up
        assert token not in ledger._tokens
    
    def test_concurrent_apply_prevented(self, ledger):
        """Test concurrent apply calls are prevented"""
        token, _ = ledger.prepare("concurrent-apply", "sha313")
        
        apply_results = []
        apply_errors = []
        
        def apply_task():
            try:
                result = ledger.apply(token, "diff909")
                apply_results.append(result)
            except ValueError as e:
                apply_errors.append(str(e))
        
        # Start multiple threads trying to apply
        threads = [threading.Thread(target=apply_task) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Only one should succeed
        assert len(apply_results) == 1
        assert len(apply_errors) == 4
        assert all("Invalid token" in err or "already used" in err for err in apply_errors)
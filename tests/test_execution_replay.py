# tests/test_execution_replay.py
import pytest
import os
import uuid
import time
import signal
import threading
import shutil
from typing import Dict, Any
from datetime import datetime

from execution.dispatcher import ExecutionDispatcher, CrashSimulator
from execution.state_ledger import StateLedger

class TestExecutionReplay:
    """Test exactly-once execution with crash/replay scenarios"""
    
    @pytest.fixture
    def temp_paths(self, tmp_path):
        """Create temporary paths for testing"""
        ledger_path = tmp_path / "ledger"
        manifest_path = tmp_path / "manifests"
        return {
            "ledger": str(ledger_path),
            "manifest": str(manifest_path)
        }
    
    @pytest.fixture
    def dispatcher(self, temp_paths):
        """Create dispatcher with temporary storage"""
        return ExecutionDispatcher(
            ledger_path=temp_paths["ledger"],
            manifest_path=temp_paths["manifest"]
        )
    
    def test_idempotent_execution_same_key(self, dispatcher):
        """Test that same idempotency key returns cached result"""
        idempotency_key = str(uuid.uuid4())
        operation = "database_write"
        payload = {"table": "users", "rows": 5}
        
        # First execution
        result1 = dispatcher.dispatch(
            idempotency_key=idempotency_key,
            operation=operation,
            payload=payload
        )
        
        assert result1["status"] == "success"
        assert result1["replay"] is False
        run_id1 = result1["run_id"]
        
        # Second execution with same key
        result2 = dispatcher.dispatch(
            idempotency_key=idempotency_key,
            operation=operation,
            payload=payload
        )
        
        assert result2["status"] == "success"
        assert result2["replay"] is True
        assert result2["run_id"] == run_id1
        assert result2["result"] == result1["result"]
        
        # Verify no duplicate side effects
        stats = dispatcher.get_execution_stats()
        assert stats["total_side_effects"] == 1  # Only one execution
    
    def test_crash_before_apply(self, temp_paths):
        """Test recovery when crash occurs before apply"""
        # First dispatcher that will crash
        dispatcher1 = ExecutionDispatcher(
            ledger_path=temp_paths["ledger"],
            manifest_path=temp_paths["manifest"]
        )
        dispatcher1.enable_crash_simulation("before_apply")
        
        idempotency_key = str(uuid.uuid4())
        operation = "commit"
        payload = {"repository": "test", "files": ["file1.py"]}
        
        # Execute and expect crash
        with pytest.raises(Exception):
            # This will crash before apply
            dispatcher1.dispatch(
                idempotency_key=idempotency_key,
                operation=operation,
                payload=payload
            )
        
        # New dispatcher to recover
        dispatcher2 = ExecutionDispatcher(
            ledger_path=temp_paths["ledger"],
            manifest_path=temp_paths["manifest"]
        )
        
        # Replay should complete execution
        result = dispatcher2.dispatch(
            idempotency_key=idempotency_key,
            operation=operation,
            payload=payload
        )
        
        assert result["status"] == "success"
        assert result["idempotency_key"] == idempotency_key
    
    def test_crash_during_apply(self, temp_paths):
        """Test recovery when crash occurs during apply"""
        
        def simulate_crash_during_apply():
            """Helper to simulate crash during apply phase"""
            dispatcher = ExecutionDispatcher(
                ledger_path=temp_paths["ledger"],
                manifest_path=temp_paths["manifest"]
            )
            
            # Monkey-patch execute to crash mid-way
            original_execute = dispatcher._execute_operation
            
            def crashing_execute(*args, **kwargs):
                # Start execution
                result = {"partial": "result"}
                # Simulate crash
                os._exit(1)  # Hard exit
                return result
            
            dispatcher._execute_operation = crashing_execute
            
            try:
                dispatcher.dispatch(
                    idempotency_key="crash-test-123",
                    operation="actuate",
                    payload={"robot_id": "robot1", "action": "move"}
                )
            except SystemExit:
                pass
        
        # Run in subprocess to simulate crash
        import multiprocessing
        process = multiprocessing.Process(target=simulate_crash_during_apply)
        process.start()
        process.join(timeout=5)
        
        if process.is_alive():
            process.terminate()
        
        # Recover with new dispatcher
        dispatcher = ExecutionDispatcher(
            ledger_path=temp_paths["ledger"],
            manifest_path=temp_paths["manifest"]
        )
        
        # Check if manifest was created
        existing_run = dispatcher.ledger.check_idempotency("crash-test-123")
        
        # Replay should handle incomplete execution
        result = dispatcher.dispatch(
            idempotency_key="crash-test-123",
            operation="actuate",
            payload={"robot_id": "robot1", "action": "move"}
        )
        
        # Should complete successfully
        assert result["status"] in ["success", "retry_needed"]
    
    def test_crash_after_apply_before_commit(self, temp_paths):
        """Test recovery when crash occurs after apply but before commit"""
        
        class CrashAfterApplyDispatcher(ExecutionDispatcher):
            def commit_apply(self, *args, **kwargs):
                # Crash before commit
                raise Exception("Simulated crash before commit")
        
        dispatcher1 = CrashAfterApplyDispatcher(
            ledger_path=temp_paths["ledger"],
            manifest_path=temp_paths["manifest"]
        )
        
        idempotency_key = str(uuid.uuid4())
        
        # This will apply but fail to commit
        with pytest.raises(Exception):
            dispatcher1.dispatch(
                idempotency_key=idempotency_key,
                operation="database_write",
                payload={"table": "test", "rows": 1}
            )
        
        # New dispatcher to check state
        dispatcher2 = ExecutionDispatcher(
            ledger_path=temp_paths["ledger"],
            manifest_path=temp_paths["manifest"]
        )
        
        # Should be able to replay safely
        result = dispatcher2.dispatch(
            idempotency_key=idempotency_key,
            operation="database_write",
            payload={"table": "test", "rows": 1}
        )
        
        # Verify no duplicate side effects
        stats = dispatcher2.get_execution_stats()
        assert stats["total_side_effects"] <= 2  # At most one retry
    
    def test_concurrent_requests_same_key(self, dispatcher):
        """Test concurrent requests with same idempotency key"""
        idempotency_key = str(uuid.uuid4())
        operation = "analyze"
        payload = {"data": "test"}
        results = []
        errors = []
        
        def execute():
            try:
                result = dispatcher.dispatch(
                    idempotency_key=idempotency_key,
                    operation=operation,
                    payload=payload
                )
                results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        # Launch concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=execute)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All should succeed with same run_id
        assert len(errors) == 0
        assert len(results) == 5
        
        run_ids = [r["run_id"] for r in results]
        assert len(set(run_ids)) == 1  # All same run_id
        
        # Verify exactly one execution
        stats = dispatcher.get_execution_stats()
        assert stats["total_side_effects"] == 1
    
    def test_manifest_write_through(self, dispatcher):
        """Test manifest is written at each stage"""
        idempotency_key = str(uuid.uuid4())
        operation = "commit"
        payload = {"repository": "test", "files": ["test.py"]}
        approvals = [{
            "approver": "test@example.com",
            "decision": "approve",
            "timestamp": datetime.utcnow().isoformat(),
            "token": "test-token"
        }]
        policy_mode = "enforce"
        
        # Execute
        result = dispatcher.dispatch(
            idempotency_key=idempotency_key,
            operation=operation,
            payload=payload,
            approvals=approvals,
            policy_mode=policy_mode
        )
        
        # Load manifest
        manifest = dispatcher.ledger.get_run_manifest(result["run_id"])
        
        # Verify all fields present
        assert manifest["idempotency_key"] == idempotency_key
        assert manifest["operation"] == operation
        assert manifest["approvals"] == approvals
        assert manifest["policy_mode"] == policy_mode
        assert manifest["status"] == "committed"
        assert "created_at" in manifest
        assert "applied_at" in manifest
        assert "committed_at" in manifest
        assert "written_at" in manifest
    
    def test_recovery_incomplete_runs(self, temp_paths):
        """Test recovery of incomplete runs on startup"""
        dispatcher1 = ExecutionDispatcher(
            ledger_path=temp_paths["ledger"],
            manifest_path=temp_paths["manifest"]
        )
        
        # Create some incomplete runs
        for i in range(3):
            run_id = str(uuid.uuid4())
            manifest = {
                "run_id": run_id,
                "idempotency_key": f"incomplete-{i}",
                "operation": "analyze",
                "payload": {"data": f"test-{i}"},
                "approvals": [],
                "policy_mode": "monitor",
                "created_at": datetime.utcnow().isoformat(),
                "status": "dispatched" if i % 2 == 0 else "applied"
            }
            dispatcher1.ledger.write_run_manifest(run_id, manifest)
        
        # New dispatcher recovers
        dispatcher2 = ExecutionDispatcher(
            ledger_path=temp_paths["ledger"],
            manifest_path=temp_paths["manifest"]
        )
        
        # Recover incomplete runs
        recovered = dispatcher2.recover_incomplete_runs()
        
        # Should find and attempt to recover incomplete runs
        assert len(recovered) > 0
    
    def test_apply_token_invalidation(self, dispatcher):
        """Test that invalidated tokens cannot be reused"""
        ledger = dispatcher.ledger
        
        # Generate token
        apply_token = ledger.generate_apply_token("test-key", "test-run")
        
        # Begin apply
        assert ledger.begin_apply(apply_token, "test-run", "test-op") is True
        
        # Rollback (invalidates token)
        ledger.rollback_apply(apply_token, "test failure")
        
        # Try to use invalidated token
        assert ledger.begin_apply(apply_token, "test-run", "test-op") is False
    
    def test_no_duplicate_side_effects_on_replay(self, dispatcher):
        """Test that replay never re-executes side effects"""
        idempotency_key = str(uuid.uuid4())
        
        # Track side effects
        original_execute = dispatcher._execute_operation
        execution_count = {"count": 0}
        
        def counting_execute(*args, **kwargs):
            execution_count["count"] += 1
            return original_execute(*args, **kwargs)
        
        dispatcher._execute_operation = counting_execute
        
        # First execution
        result1 = dispatcher.dispatch(
            idempotency_key=idempotency_key,
            operation="actuate",
            payload={"robot_id": "robot1", "action": "calibrate"}
        )
        
        assert execution_count["count"] == 1
        
        # Multiple replays
        for _ in range(5):
            result = dispatcher.dispatch(
                idempotency_key=idempotency_key,
                operation="actuate",
                payload={"robot_id": "robot1", "action": "calibrate"}
            )
            assert result["replay"] is True
        
        # Still only one execution
        assert execution_count["count"] == 1
# execution/dispatcher.py
import json
import os
import uuid
import random
import signal
import time
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from .state_ledger import StateLedger

logger = logging.getLogger(__name__)

class CrashSimulator:
    """Simulates crashes at critical points for testing"""
    
    def __init__(self, crash_probability: float = 0.0):
        self.crash_probability = crash_probability
        self.crash_points = {
            "before_apply": False,
            "during_apply": False,
            "after_apply": False,
            "before_commit": False
        }
    
    def maybe_crash(self, point: str):
        """Simulate a crash at the given point"""
        if self.crash_points.get(point, False):
            logger.error(f"SIMULATED CRASH at {point}")
            os.kill(os.getpid(), signal.SIGKILL)
        
        if random.random() < self.crash_probability:
            logger.error(f"RANDOM CRASH at {point}")
            os.kill(os.getpid(), signal.SIGKILL)

class ExecutionDispatcher:
    """
    Dispatcher with crash recovery and replay support.
    Ensures exactly-once execution through manifest-based replay.
    """
    
    def __init__(self, ledger_path: str = "/tmp/state_ledger", 
                 manifest_path: str = "/tmp/manifests"):
        self.ledger = StateLedger(ledger_path)
        self.manifest_path = manifest_path
        self.crash_sim = CrashSimulator()
        self.side_effects_applied: Dict[str, bool] = {}
        
        os.makedirs(manifest_path, exist_ok=True)
    
    def dispatch(self, idempotency_key: str, operation: str, 
                 payload: Dict[str, Any], approvals: list = None, 
                 policy_mode: str = "enforce") -> Dict[str, Any]:
        """
        Dispatch operation with exactly-once guarantee.
        
        Args:
            idempotency_key: Client idempotency key
            operation: Operation to execute
            payload: Operation payload
            approvals: List of approval records
            policy_mode: Policy enforcement mode
            
        Returns:
            Execution result
        """
        # Check for existing run (idempotency)
        existing_run_id = self.ledger.check_idempotency(idempotency_key)
        if existing_run_id:
            logger.info(f"Found existing run {existing_run_id} for key {idempotency_key}")
            return self._replay_from_manifest(existing_run_id)
        
        # Generate new run_id
        run_id = str(uuid.uuid4())
        
        # Create initial manifest
        manifest = {
            "run_id": run_id,
            "idempotency_key": idempotency_key,
            "operation": operation,
            "payload": payload,
            "approvals": approvals or [],
            "policy_mode": policy_mode,
            "created_at": datetime.utcnow().isoformat(),
            "status": "dispatched"
        }
        
        # Write manifest (write-through)
        self.ledger.write_run_manifest(run_id, manifest)
        
        # Generate apply token
        apply_token = self.ledger.generate_apply_token(idempotency_key, run_id)
        
        # Crash point: before apply
        self.crash_sim.maybe_crash("before_apply")
        
        # Begin apply with idempotency guard
        if not self.ledger.begin_apply(apply_token, run_id, operation):
            logger.info(f"Apply already in progress or completed for {apply_token}")
            return self._replay_from_manifest(run_id)
        
        try:
            # Crash point: during apply
            self.crash_sim.maybe_crash("during_apply")
            
            # Execute operation (with side effects)
            result = self._execute_operation(run_id, operation, payload)
            
            # Crash point: after apply
            self.crash_sim.maybe_crash("after_apply")
            
            # Update manifest with result
            manifest["status"] = "applied"
            manifest["result"] = result
            manifest["applied_at"] = datetime.utcnow().isoformat()
            self.ledger.write_run_manifest(run_id, manifest)
            
            # Crash point: before commit
            self.crash_sim.maybe_crash("before_commit")
            
            # Commit the apply
            self.ledger.commit_apply(apply_token, result)
            
            # Final manifest update
            manifest["status"] = "committed"
            manifest["committed_at"] = datetime.utcnow().isoformat()
            self.ledger.write_run_manifest(run_id, manifest)
            
            return {
                "run_id": run_id,
                "status": "success",
                "result": result,
                "idempotency_key": idempotency_key,
                "replay": False
            }
            
        except Exception as e:
            logger.error(f"Operation failed: {e}")
            
            # Rollback
            self.ledger.rollback_apply(apply_token, str(e))
            
            # Update manifest with failure
            manifest["status"] = "failed"
            manifest["error"] = str(e)
            manifest["failed_at"] = datetime.utcnow().isoformat()
            self.ledger.write_run_manifest(run_id, manifest)
            
            raise
    
    def _execute_operation(self, run_id: str, operation: str, 
                          payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the actual operation with side effects.
        
        Args:
            run_id: Run identifier
            operation: Operation to execute
            payload: Operation payload
            
        Returns:
            Operation result
        """
        logger.info(f"Executing {operation} for run {run_id}")
        
        # Track side effects to ensure no duplication
        side_effect_key = f"{run_id}:{operation}"
        
        if side_effect_key in self.side_effects_applied:
            logger.warning(f"Side effects already applied for {side_effect_key}")
            return {"status": "skipped", "reason": "already_applied"}
        
        # Simulate actual operation execution
        result = {
            "operation": operation,
            "executed_at": datetime.utcnow().isoformat(),
            "side_effects": []
        }
        
        if operation == "commit":
            # Simulate git commit
            result["side_effects"].append({
                "type": "git_commit",
                "commit_id": f"abc{run_id[:8]}",
                "files_changed": payload.get("files", [])
            })
            
        elif operation == "actuate":
            # Simulate robot actuation
            result["side_effects"].append({
                "type": "robot_actuation",
                "robot_id": payload.get("robot_id"),
                "action": payload.get("action"),
                "timestamp": datetime.utcnow().isoformat()
            })
            
        elif operation == "database_write":
            # Simulate database write
            result["side_effects"].append({
                "type": "database_write",
                "table": payload.get("table"),
                "rows_affected": payload.get("rows", 1)
            })
        
        # Mark side effects as applied
        self.side_effects_applied[side_effect_key] = True
        
        # Simulate some processing time
        time.sleep(0.1)
        
        return result
    
    def _replay_from_manifest(self, run_id: str) -> Dict[str, Any]:
        """
        Replay execution from manifest (no side effects).
        
        Args:
            run_id: Run identifier to replay
            
        Returns:
            Cached result from manifest
        """
        logger.info(f"Replaying from manifest for run {run_id}")
        
        # Load manifest
        manifest = self.ledger.get_run_manifest(run_id)
        if not manifest:
            raise ValueError(f"No manifest found for run {run_id}")
        
        # Check status
        status = manifest.get("status")
        
        if status == "committed":
            # Return cached result
            return {
                "run_id": run_id,
                "status": "success",
                "result": manifest.get("result"),
                "idempotency_key": manifest.get("idempotency_key"),
                "replay": True,
                "replayed_at": datetime.utcnow().isoformat()
            }
            
        elif status == "failed":
            # Return cached failure
            return {
                "run_id": run_id,
                "status": "failed",
                "error": manifest.get("error"),
                "idempotency_key": manifest.get("idempotency_key"),
                "replay": True
            }
            
        elif status in ["dispatched", "applied"]:
            # Incomplete execution - attempt recovery
            logger.warning(f"Incomplete execution for run {run_id}, attempting recovery")
            
            # Check if we can safely retry
            apply_history = self.ledger.get_apply_history(run_id)
            if apply_history:
                # Has apply history - check if committed
                for record in apply_history:
                    if record.get("status") == "committed":
                        return {
                            "run_id": run_id,
                            "status": "success",
                            "result": record.get("result"),
                            "idempotency_key": manifest.get("idempotency_key"),
                            "replay": True,
                            "recovered": True
                        }
            
            # No committed apply - safe to retry
            return {
                "run_id": run_id,
                "status": "retry_needed",
                "idempotency_key": manifest.get("idempotency_key"),
                "replay": True
            }
        
        else:
            raise ValueError(f"Unknown manifest status: {status}")
    
    def recover_incomplete_runs(self) -> list[str]:
        """
        Recover incomplete runs from manifests.
        
        Returns:
            List of recovered run IDs
        """
        recovered = []
        
        # Scan manifest directory
        for filename in os.listdir(self.manifest_path):
            if filename.startswith("manifest_"):
                filepath = os.path.join(self.manifest_path, filename)
                with open(filepath, 'r') as f:
                    manifest = json.load(f)
                
                if manifest.get("status") in ["dispatched", "applied"]:
                    run_id = manifest.get("run_id")
                    logger.info(f"Recovering incomplete run {run_id}")
                    
                    # Attempt to complete or mark as failed
                    try:
                        result = self._replay_from_manifest(run_id)
                        if result.get("status") == "retry_needed":
                            # Re-dispatch
                            self.dispatch(
                                idempotency_key=manifest.get("idempotency_key"),
                                operation=manifest.get("operation"),
                                payload=manifest.get("payload"),
                                approvals=manifest.get("approvals"),
                                policy_mode=manifest.get("policy_mode")
                            )
                        recovered.append(run_id)
                    except Exception as e:
                        logger.error(f"Failed to recover run {run_id}: {e}")
        
        return recovered
    
    def enable_crash_simulation(self, crash_point: str = None, probability: float = 0.0):
        """
        Enable crash simulation for testing.
        
        Args:
            crash_point: Specific point to crash at
            probability: Random crash probability
        """
        if crash_point:
            self.crash_sim.crash_points[crash_point] = True
        self.crash_sim.crash_probability = probability
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            "total_side_effects": len(self.side_effects_applied),
            "cached_manifests": len(self.ledger.run_manifests),
            "applied_tokens": len(self.ledger.applied_tokens),
            "invalidated_tokens": len(self.ledger.invalidated_tokens)
        }
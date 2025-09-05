# orchestrator/brain.py
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from storage.manifest import ManifestStore
from api.errors import APIError, ErrorCode

logger = logging.getLogger(__name__)

class OrchestrationBrain:
    """
    Orchestration brain with minimal HITL enforcement
    Checks for approval tokens before risky operations
    """
    
    def __init__(self):
        self.manifest_store = ManifestStore()
        self.risky_operations = {'commit', 'actuate'}  # Operations requiring HITL
        
    def check_hitl_approval(self, task_id: str, operation: str) -> tuple[bool, Optional[str]]:
        """
        Check if HITL approval exists for risky operations
        
        Args:
            task_id: Task identifier
            operation: Operation type to execute
            
        Returns:
            Tuple of (is_approved, approval_link)
        """
        # Only check for risky operations
        if operation not in self.risky_operations:
            return True, None
            
        # Check manifest for approval record
        approval_record = self.manifest_store.get_approval(task_id)
        
        if not approval_record:
            # Generate approval link placeholder
            approval_link = f"/approvals/{task_id}"
            logger.warning(f"HITL approval required for task {task_id}, operation: {operation}")
            return False, approval_link
            
        # Verify approval is valid and approved
        if approval_record.get("decision") != "approve":
            logger.warning(f"Task {task_id} was rejected in HITL review")
            return False, None
            
        logger.info(f"HITL approval found for task {task_id}")
        return True, None
    
    def execute_operation(self, task_id: str, operation: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute operation with HITL checks
        
        Args:
            task_id: Task identifier
            operation: Operation type
            payload: Operation payload
            
        Returns:
            Operation result
            
        Raises:
            APIError: If HITL approval is required
        """
        # Check HITL approval for risky operations
        is_approved, approval_link = self.check_hitl_approval(task_id, operation)
        
        if not is_approved:
            # Record HITL requirement event
            event = {
                "event_type": "HITL_REQUIRED",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "operation": operation,
                    "reason": f"Human approval required for {operation} operation"
                }
            }
            
            if approval_link:
                event["approval_link"] = approval_link
            
            # Raise HITL required error
            raise APIError(
                error_code=ErrorCode.HITL_REQUIRED,
                message=f"Human approval required for {operation} operation",
                status_code=202,
                details={
                    "task_id": task_id,
                    "approval_link": approval_link,
                    "operation": operation
                }
            )
        
        # Proceed with operation execution
        if operation == "analyze":
            return self._execute_analyze(payload)
        elif operation == "transform":
            return self._execute_transform(payload)
        elif operation == "commit":
            return self._execute_commit(task_id, payload)
        elif operation == "actuate":
            return self._execute_actuate(task_id, payload)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _execute_analyze(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis operation (safe, no HITL needed)"""
        logger.info("Executing analyze operation")
        # Implement actual analysis logic
        return {
            "status": "completed",
            "result": "Analysis completed successfully"
        }
    
    def _execute_transform(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute transformation operation (safe, no HITL needed)"""
        logger.info("Executing transform operation")
        # Implement actual transformation logic
        return {
            "status": "completed",
            "result": "Transformation completed successfully"
        }
    
    def _execute_commit(self, task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute commit operation (risky, requires HITL approval)
        This method is only called after HITL check passes
        """
        logger.info(f"Executing commit operation for task {task_id}")
        
        # Double-check approval before proceeding (defense in depth)
        is_approved, _ = self.check_hitl_approval(task_id, "commit")
        if not is_approved:
            raise APIError(
                error_code=ErrorCode.AUTHORIZATION_FAILED,
                message="Cannot execute commit without approval",
                status_code=403
            )
        
        # Safe to proceed with repository commit
        repo = payload.get("repository")
        branch = payload.get("branch", "main")
        files = payload.get("files", [])
        
        logger.info(f"Committing to {repo}/{branch}: {len(files)} files")
        
        # Actual repo.commit() implementation would go here
        # For now, simulate success
        return {
            "status": "completed",
            "result": {
                "repository": repo,
                "branch": branch,
                "commit_id": "abc123def456",
                "files_changed": len(files)
            }
        }
    
    def _execute_actuate(self, task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute actuation operation (risky, requires HITL approval)
        This method is only called after HITL check passes
        """
        logger.info(f"Executing actuate operation for task {task_id}")
        
        # Double-check approval before proceeding (defense in depth)
        is_approved, _ = self.check_hitl_approval(task_id, "actuate")
        if not is_approved:
            raise APIError(
                error_code=ErrorCode.AUTHORIZATION_FAILED,
                message="Cannot execute actuation without approval",
                status_code=403
            )
        
        # Safe to proceed with robot actuation
        robot_id = payload.get("robot_id")
        action = payload.get("action")
        parameters = payload.get("parameters", {})
        
        logger.info(f"Actuating robot {robot_id}: {action}")
        
        # Actual robot.actuate() implementation would go here
        # For now, simulate success
        return {
            "status": "completed",
            "result": {
                "robot_id": robot_id,
                "action": action,
                "execution_time": datetime.utcnow().isoformat(),
                "parameters": parameters
            }
        }
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for task processing with HITL enforcement
        
        Args:
            task: Task to process
            
        Returns:
            Processing result
        """
        task_id = task["task_id"]
        operation = task["operation"]
        payload = task.get("payload", {})
        
        logger.info(f"Processing task {task_id} with operation: {operation}")
        
        try:
            result = self.execute_operation(task_id, operation, payload)
            
            return {
                "task_id": task_id,
                "status": "completed",
                "result": result,
                "completed_at": datetime.utcnow().isoformat()
            }
            
        except APIError as e:
            # Re-raise API errors (including HITL_REQUIRED)
            raise
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {str(e)}")
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.utcnow().isoformat()
            }

# Storage helper for manifest operations
class ManifestStore:
    """Simplified manifest storage for HITL approvals"""
    
    def __init__(self):
        self.approvals: Dict[str, Dict[str, Any]] = {}
    
    def record_approval(self, task_id: str, approval_token: str, 
                       decision: str, reason: Optional[str], timestamp: str):
        """Record HITL approval decision"""
        self.approvals[task_id] = {
            "approval_token": approval_token,
            "decision": decision,
            "reason": reason,
            "timestamp": timestamp
        }
    
    def get_approval(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get approval record for task"""
        return self.approvals.get(task_id)
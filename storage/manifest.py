# storage/manifest.py
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ManifestStore:
    """
    Storage for HITL approvals and task manifests
    In production, this would use a persistent database
    """
    
    def __init__(self, storage_path: str = "/tmp/manifests"):
        """
        Initialize manifest storage
        
        Args:
            storage_path: Path to store manifest files (for file-based implementation)
        """
        self.storage_path = storage_path
        self.approvals: Dict[str, Dict[str, Any]] = {}  # In-memory cache
        
        # Create storage directory if it doesn't exist
        if not os.path.exists(storage_path):
            try:
                os.makedirs(storage_path)
            except:
                pass  # Use in-memory only if can't create directory
    
    def record_approval(self, task_id: str, approval_token: str, 
                       decision: str, reason: Optional[str] = None, 
                       timestamp: Optional[str] = None) -> bool:
        """
        Record HITL approval decision
        
        Args:
            task_id: Task identifier
            approval_token: Security token for approval
            decision: 'approve' or 'reject'
            reason: Optional reason for decision
            timestamp: ISO format timestamp
            
        Returns:
            True if successfully recorded
        """
        try:
            approval_record = {
                "task_id": task_id,
                "approval_token": approval_token,
                "decision": decision,
                "reason": reason,
                "timestamp": timestamp or datetime.utcnow().isoformat(),
                "recorded_at": datetime.utcnow().isoformat()
            }
            
            # Store in memory
            self.approvals[task_id] = approval_record
            
            # Persist to file (optional)
            try:
                manifest_file = os.path.join(self.storage_path, f"{task_id}_approval.json")
                with open(manifest_file, 'w') as f:
                    json.dump(approval_record, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not persist approval to file: {str(e)}")
            
            logger.info(f"Recorded {decision} approval for task {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record approval: {str(e)}")
            return False
    
    def get_approval(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get approval record for a task
        
        Args:
            task_id: Task identifier
            
        Returns:
            Approval record if exists, None otherwise
        """
        # Check memory first
        if task_id in self.approvals:
            return self.approvals[task_id]
        
        # Check file storage
        try:
            manifest_file = os.path.join(self.storage_path, f"{task_id}_approval.json")
            if os.path.exists(manifest_file):
                with open(manifest_file, 'r') as f:
                    approval = json.load(f)
                    # Cache in memory
                    self.approvals[task_id] = approval
                    return approval
        except Exception as e:
            logger.warning(f"Could not read approval from file: {str(e)}")
        
        return None
    
    def has_approval(self, task_id: str) -> bool:
        """
        Check if task has an approval record
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if approval exists
        """
        return self.get_approval(task_id) is not None
    
    def is_approved(self, task_id: str) -> bool:
        """
        Check if task is approved (not just has record, but approved)
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task is explicitly approved
        """
        approval = self.get_approval(task_id)
        return approval is not None and approval.get("decision") == "approve"
    
    def delete_approval(self, task_id: str) -> bool:
        """
        Delete approval record (for cleanup or testing)
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if deleted successfully
        """
        try:
            # Remove from memory
            if task_id in self.approvals:
                del self.approvals[task_id]
            
            # Remove file
            manifest_file = os.path.join(self.storage_path, f"{task_id}_approval.json")
            if os.path.exists(manifest_file):
                os.remove(manifest_file)
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete approval: {str(e)}")
            return False
    
    def list_approvals(self) -> Dict[str, Dict[str, Any]]:
        """
        List all approval records (for debugging/admin)
        
        Returns:
            Dictionary of all approvals by task_id
        """
        # Load all from files if not in memory
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith("_approval.json"):
                    task_id = filename.replace("_approval.json", "")
                    if task_id not in self.approvals:
                        self.get_approval(task_id)  # This will cache it
        except Exception as e:
            logger.warning(f"Could not list approval files: {str(e)}")
        
        return self.approvals.copy()

# Singleton instance for shared use
_manifest_store_instance = None

def get_manifest_store() -> ManifestStore:
    """Get singleton manifest store instance"""
    global _manifest_store_instance
    if _manifest_store_instance is None:
        _manifest_store_instance = ManifestStore()
    return _manifest_store_instance
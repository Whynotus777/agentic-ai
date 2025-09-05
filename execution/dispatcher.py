"""
Dispatcher with Retry Logic

Coordinates execution with state ledger and implements exponential backoff.
"""
import asyncio
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple
import logging
import uuid

from execution.queue import ExecutionQueue, QueueMessage
from execution.state_ledger import StateLedger

logger = logging.getLogger(__name__)

class RetryPolicy:
    """Exponential backoff retry policy"""
    def __init__(self,
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 300.0,
                 multiplier: float = 4.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number (0-indexed)"""
        if attempt >= self.max_attempts:
            return None  # No more retries
        
        delay = self.base_delay * (self.multiplier ** attempt)
        return min(delay, self.max_delay)

@dataclass
class ExecutionContext:
    """Context passed to execution handlers"""
    run_id: str
    tenant_id: str
    idempotency_key: str
    trace_id: str
    correlation_id: str
    message: QueueMessage
    attempt: int
    manifest_path: Optional[Path] = None
    
    @property
    def is_replay(self) -> bool:
        """Check if this is a replay from existing manifest"""
        return self.manifest_path is not None and self.manifest_path.exists()

class TaskExecutor:
    """Handles actual task execution with proper error handling"""
    
    def __init__(self, handler: Callable[[ExecutionContext], Dict[str, Any]]):
        self.handler = handler
    
    async def execute(self, context: ExecutionContext) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute task with context.
        
        Returns:
            (success, result_or_error)
        """
        try:
            # Check for replay
            if context.is_replay:
                logger.info(f"Replaying from manifest: {context.manifest_path}")
                return self._replay_from_manifest(context)
            
            # Normal execution
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.handler, context
            )
            
            return True, result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}", exc_info=True)
            return False, {"error": str(e), "type": type(e).__name__}
    
    def _replay_from_manifest(self, context: ExecutionContext) -> Tuple[bool, Dict[str, Any]]:
        """Replay execution from existing manifest"""
        try:
            with open(context.manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Verify manifest matches request
            if manifest['run_id'] != context.run_id:
                raise ValueError(f"Manifest run_id mismatch: {manifest['run_id']} != {context.run_id}")
            
            # Return cached result without re-execution
            return True, {
                "replayed": True,
                "run_id": manifest['run_id'],
                "artifacts": manifest.get('artifacts', []),
                "outputs": manifest.get('outputs', {})
            }
            
        except FileNotFoundError:
            logger.warning(f"Manifest not found for replay: {context.manifest_path}")
            return False, {"error": "Manifest not found for replay"}
        except Exception as e:
            logger.error(f"Replay failed: {e}")
            return False, {"error": str(e)}

class Dispatcher:
    """
    Main dispatcher coordinating queue, ledger, and execution.
    
    Implements the prepare→apply→commit pattern with retry logic.
    """
    
    def __init__(self,
                 queue: ExecutionQueue,
                 ledger: StateLedger,
                 executor: TaskExecutor,
                 retry_policy: Optional[RetryPolicy] = None,
                 manifest_base_path: Path = Path("/var/lib/execution/manifests"),
                 max_workers: int = 10):
        
        self.queue = queue
        self.ledger = ledger
        self.executor = executor
        self.retry_policy = retry_policy or RetryPolicy()
        self.manifest_base_path = manifest_base_path
        
        # Thread pool for concurrent execution
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Running state
        self.running = False
        self.worker_tasks: List[asyncio.Task] = []
        
        # Stats
        self.stats = {
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "retried": 0,
            "dlq": 0,
            "replayed": 0
        }
    
    async def start(self, num_workers: int = 5):
        """Start dispatcher workers"""
        self.running = True
        
        # Start worker tasks
        for i in range(num_workers):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self.worker_tasks.append(task)
        
        logger.info(f"Dispatcher started with {num_workers} workers")
    
    async def stop(self):
        """Stop dispatcher workers"""
        self.running = False
        
        # Wait for workers to finish
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        logger.info("Dispatcher stopped")
    
    async def _worker(self, worker_id: str):
        """Worker coroutine processing messages from queue"""
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get tenant list (in production, this would be dynamic)
                tenants = self._get_active_tenants()
                
                for tenant_id in tenants:
                    # Dequeue with short wait
                    messages = self.queue.dequeue(
                        tenant_id=tenant_id,
                        max_messages=1,
                        wait_timeout=1.0
                    )
                    
                    for message in messages:
                        await self._process_message(worker_id, message)
                
                # Short sleep between iterations
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
                await asyncio.sleep(1)
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _process_message(self, worker_id: str, message: QueueMessage):
        """Process a single message through prepare→apply→commit"""
        logger.info(f"[{worker_id}] Processing message: {message.message_id}")
        
        try:
            # 1. PREPARE phase
            input_sha = self._compute_input_hash(message.payload)
            token, existing_run_id = self.ledger.prepare(
                idempotency_key=message.idempotency_key,
                input_sha=input_sha,
                metadata={
                    "tenant_id": message.tenant_id,
                    "trace_id": message.trace_id,
                    "message_id": message.message_id
                }
            )
            
            # Check for existing execution
            if existing_run_id:
                logger.info(f"[{worker_id}] Idempotent execution exists: {existing_run_id}")
                self.queue.acknowledge(message.tenant_id, message.receipt_handle)
                self.stats["replayed"] += 1
                return
            
            # 2. Create execution context
            run_id = f"run_{uuid.uuid4().hex[:12]}"
            manifest_path = self._get_manifest_path(message.tenant_id, run_id)
            
            # Check for existing manifest (replay scenario)
            if manifest_path.exists():
                logger.info(f"[{worker_id}] Found existing manifest for replay")
            
            context = ExecutionContext(
                run_id=run_id,
                tenant_id=message.tenant_id,
                idempotency_key=message.idempotency_key,
                trace_id=message.trace_id,
                correlation_id=message.correlation_id,
                message=message,
                attempt=message.retry_count,
                manifest_path=manifest_path if manifest_path.exists() else None
            )
            
            # 3. APPLY phase - execute with token
            run_id = self.ledger.apply(token, self._compute_diff_hash(context))
            context.run_id = run_id
            
            success, result = await self.executor.execute(context)
            
            if not success:
                # Handle execution failure
                await self._handle_failure(worker_id, message, result.get("error"))
                return
            
            # 4. Create and save manifest
            manifest = self._create_manifest(context, result)
            manifest_path = self._save_manifest(message.tenant_id, run_id, manifest)
            
            # 5. COMMIT phase
            self.ledger.commit(run_id, str(manifest_path))
            
            # 6. Acknowledge message
            self.queue.acknowledge(message.tenant_id, message.receipt_handle)
            
            self.stats["processed"] += 1
            self.stats["succeeded"] += 1
            
            logger.info(f"[{worker_id}] Successfully processed: run_id={run_id}")
            
        except Exception as e:
            logger.error(f"[{worker_id}] Processing failed: {e}", exc_info=True)
            await self._handle_failure(worker_id, message, str(e))
    
    async def _handle_failure(self, worker_id: str, message: QueueMessage, error: str):
        """Handle message processing failure with retry logic"""
        
        # Mark in ledger if we have a run_id
        entry = self.ledger.get_entry(message.idempotency_key)
        if entry:
            self.ledger.mark_failed(entry.run_id, error)
        
        # Calculate retry delay
        retry_delay = self.retry_policy.get_delay(message.retry_count)
        
        if retry_delay is None:
            # Max retries exceeded - send to DLQ
            logger.error(f"[{worker_id}] Max retries exceeded, moving to DLQ: {message.message_id}")
            self.queue.nack(message.tenant_id, message.receipt_handle, retry_delay=None)
            self.stats["dlq"] += 1
            
            # Create compensating action reference
            self._create_compensating_action(message, error)
        else:
            # Retry with backoff
            logger.warning(f"[{worker_id}] Retrying message {message.message_id} "
                          f"(attempt {message.retry_count + 1}) after {retry_delay}s")
            self.queue.nack(message.tenant_id, message.receipt_handle, retry_delay=int(retry_delay))
            self.stats["retried"] += 1
        
        self.stats["failed"] += 1
    
    def _compute_input_hash(self, payload: Dict[str, Any]) -> str:
        """Compute deterministic hash of input payload"""
        # Sort keys for deterministic hash
        canonical = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()
    
    def _compute_diff_hash(self, context: ExecutionContext) -> str:
        """Compute hash representing execution diff"""
        data = {
            "run_id": context.run_id,
            "tenant_id": context.tenant_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        canonical = json.dumps(data, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()
    
    def _get_manifest_path(self, tenant_id: str, run_id: str) -> Path:
        """Get path for run manifest"""
        path = self.manifest_base_path / tenant_id / run_id
        path.parent.mkdir(parents=True, exist_ok=True)
        return path / "manifest.json"
    
    def _create_manifest(self, context: ExecutionContext, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create run manifest with execution details"""
        return {
            "run_id": context.run_id,
            "tenant_id": context.tenant_id,
            "idempotency_key": context.idempotency_key,
            "trace_id": context.trace_id,
            "correlation_id": context.correlation_id,
            "inputs_sha": self._compute_input_hash(context.message.payload),
            "executed_at": datetime.utcnow().isoformat(),
            "attempt": context.attempt,
            "replayed": context.is_replay,
            "tools": result.get("tools", []),
            "artifacts": result.get("artifacts", []),
            "outputs": result.get("outputs", {}),
            "signatures": result.get("signatures", []),
            "policy_mode": result.get("policy_mode", "standard"),
            "approvals": result.get("approvals", []),
            "metadata": {
                "message_id": context.message.message_id,
                "priority": context.message.priority,
                "submitted_at": context.message.submitted_at.isoformat()
            }
        }
    
    def _save_manifest(self, tenant_id: str, run_id: str, manifest: Dict[str, Any]) -> Path:
        """Save run manifest to persistent storage"""
        path = self._get_manifest_path(tenant_id, run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.debug(f"Manifest saved: {path}")
        return path
    
    def _create_compensating_action(self, message: QueueMessage, error: str):
        """Create compensating action reference for DLQ entries"""
        action = {
            "message_id": message.message_id,
            "idempotency_key": message.idempotency_key,
            "tenant_id": message.tenant_id,
            "error": error,
            "failed_at": datetime.utcnow().isoformat(),
            "compensating_action_ref": f"compensate/{message.tenant_id}/{message.message_id}",
            "manual_review_required": True
        }
        
        # Save to compensating actions log
        comp_path = self.manifest_base_path / "compensating_actions" / message.tenant_id
        comp_path.mkdir(parents=True, exist_ok=True)
        
        with open(comp_path / f"{message.message_id}.json", 'w') as f:
            json.dump(action, f, indent=2)
        
        logger.info(f"Compensating action created: {action['compensating_action_ref']}")
    
    def _get_active_tenants(self) -> List[str]:
        """Get list of active tenants (mock implementation)"""
        # In production, this would query tenant registry
        return ["tenant-001", "tenant-002", "tenant-003"]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dispatcher statistics"""
        return {
            **self.stats,
            "queue_stats": self.queue.get_stats(),
            "ledger_stats": self.ledger.get_stats(),
            "workers": len(self.worker_tasks),
            "running": self.running
        }
# execution/runner.py
"""
Execution Layer implementing prepare/apply/commit semantics with
exactly-once guarantees, DLQ, and compensating actions.
"""

import asyncio
import json
import hashlib
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
import pickle
from contextlib import asynccontextmanager

import aioredis
from asyncpg import create_pool
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)


class ExecutionState(Enum):
    """States for task execution"""
    PENDING = "pending"
    PREPARING = "preparing"
    PREPARED = "prepared"
    APPLYING = "applying"
    APPLIED = "applied"
    COMMITTING = "committing"
    COMMITTED = "committed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    DEAD_LETTER = "dead_letter"


class DeliverySemantics(Enum):
    """Delivery guarantee types"""
    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


@dataclass
class TaskDefinition:
    """Definition of a task to execute"""
    task_id: str
    idempotency_key: str
    tenant_id: str
    user_id: str
    task_type: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 30
    max_retries: int = 3
    delivery_semantics: DeliverySemantics = DeliverySemantics.EXACTLY_ONCE
    compensating_action: Optional[str] = None
    trace_id: str = ""


@dataclass
class ExecutionRecord:
    """Record of task execution"""
    task_id: str
    idempotency_key: str
    state: ExecutionState
    prepare_hash: Optional[str] = None
    apply_diff_hash: Optional[str] = None
    commit_run_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    retry_count: int = 0
    side_effects: List[Dict[str, Any]] = field(default_factory=list)
    compensating_actions_applied: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class SideEffectLedger:
    """
    Ledger for tracking side effects to ensure exactly-once semantics
    """
    
    def __init__(self, redis_client, postgres_pool):
        self.redis = redis_client
        self.postgres = postgres_pool
        
    async def record_prepare(
        self,
        idempotency_key: str,
        task_hash: str,
        task_data: Dict[str, Any]
    ) -> bool:
        """
        Record preparation phase - returns False if already prepared
        """
        key = f"prepare:{idempotency_key}"
        
        # Try to set with NX (only if not exists)
        prepared = await self.redis.set(
            key,
            json.dumps({
                "task_hash": task_hash,
                "timestamp": datetime.utcnow().isoformat(),
                "data": task_data
            }),
            nx=True,
            ex=3600  # 1 hour TTL
        )
        
        if prepared:
            # Also persist to PostgreSQL for durability
            async with self.postgres.acquire() as conn:
                await conn.execute("""
                    INSERT INTO side_effect_ledger 
                    (idempotency_key, phase, task_hash, task_data, created_at)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (idempotency_key, phase) DO NOTHING
                """, idempotency_key, "prepare", task_hash, 
                    json.dumps(task_data), datetime.utcnow())
                    
        return bool(prepared)
    
    async def record_apply(
        self,
        idempotency_key: str,
        diff_hash: str,
        side_effects: List[Dict[str, Any]]
    ) -> bool:
        """
        Record application phase with side effects
        """
        key = f"apply:{idempotency_key}"
        
        # Check if already applied
        existing = await self.redis.get(key)
        if existing:
            existing_data = json.loads(existing)
            if existing_data["diff_hash"] == diff_hash:
                # Already applied with same diff
                return False
            else:
                # Different diff - this is an error
                raise ValueError(
                    f"Idempotency conflict: different diff_hash for {idempotency_key}"
                )
                
        # Record application
        applied = await self.redis.set(
            key,
            json.dumps({
                "diff_hash": diff_hash,
                "side_effects": side_effects,
                "timestamp": datetime.utcnow().isoformat()
            }),
            nx=True,
            ex=3600
        )
        
        if applied:
            async with self.postgres.acquire() as conn:
                await conn.execute("""
                    INSERT INTO side_effect_ledger 
                    (idempotency_key, phase, diff_hash, side_effects, created_at)
                    VALUES ($1, $2, $3, $4, $5)
                """, idempotency_key, "apply", diff_hash,
                    json.dumps(side_effects), datetime.utcnow())
                    
        return bool(applied)
    
    async def record_commit(
        self,
        idempotency_key: str,
        run_id: str,
        result: Dict[str, Any]
    ) -> bool:
        """
        Record commit phase - final confirmation
        """
        key = f"commit:{idempotency_key}"
        
        committed = await self.redis.set(
            key,
            json.dumps({
                "run_id": run_id,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }),
            nx=True,
            ex=86400  # 24 hour TTL for commits
        )
        
        if committed:
            async with self.postgres.acquire() as conn:
                await conn.execute("""
                    INSERT INTO side_effect_ledger 
                    (idempotency_key, phase, run_id, result, created_at)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (idempotency_key, phase) 
                    DO UPDATE SET run_id = $3, result = $4
                """, idempotency_key, "commit", run_id,
                    json.dumps(result), datetime.utcnow())
                    
        return bool(committed)
    
    async def get_execution_state(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        """
        Get current execution state for idempotency key
        """
        # Check Redis first
        for phase in ["commit", "apply", "prepare"]:
            key = f"{phase}:{idempotency_key}"
            data = await self.redis.get(key)
            if data:
                return {
                    "phase": phase,
                    "data": json.loads(data)
                }
                
        # Fall back to PostgreSQL
        async with self.postgres.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT phase, task_hash, diff_hash, run_id, result, created_at
                FROM side_effect_ledger
                WHERE idempotency_key = $1
                ORDER BY created_at DESC
                LIMIT 1
            """, idempotency_key)
            
            if row:
                return {
                    "phase": row["phase"],
                    "data": {
                        "task_hash": row["task_hash"],
                        "diff_hash": row["diff_hash"],
                        "run_id": row["run_id"],
                        "result": json.loads(row["result"]) if row["result"] else None
                    }
                }
                
        return None


class ExecutionRunner:
    """
    Main execution runner with prepare/apply/commit pattern
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
        self.postgres_pool = None
        self.side_effect_ledger = None
        self.task_queue = asyncio.Queue()
        self.dead_letter_queue = asyncio.Queue()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.compensating_actions: Dict[str, Callable] = {}
        self.metrics = ExecutionMetrics()
        
    async def initialize(self):
        """Initialize connections and resources"""
        # Redis for fast lookups
        self.redis_client = await aioredis.create_redis_pool(
            self.config["redis_url"],
            minsize=5,
            maxsize=10
        )
        
        # PostgreSQL for durability
        self.postgres_pool = await create_pool(
            self.config["database_url"],
            min_size=5,
            max_size=20
        )
        
        # Create tables if needed
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS side_effect_ledger (
                    idempotency_key VARCHAR(255) NOT NULL,
                    phase VARCHAR(50) NOT NULL,
                    task_hash VARCHAR(64),
                    diff_hash VARCHAR(64),
                    run_id VARCHAR(255),
                    task_data JSONB,
                    side_effects JSONB,
                    result JSONB,
                    created_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (idempotency_key, phase)
                );
                
                CREATE INDEX IF NOT EXISTS idx_ledger_created 
                ON side_effect_ledger(created_at DESC);
                
                CREATE TABLE IF NOT EXISTS execution_records (
                    task_id VARCHAR(255) PRIMARY KEY,
                    idempotency_key VARCHAR(255) NOT NULL,
                    state VARCHAR(50) NOT NULL,
                    tenant_id VARCHAR(255) NOT NULL,
                    user_id VARCHAR(255) NOT NULL,
                    task_type VARCHAR(100),
                    payload JSONB,
                    error TEXT,
                    retry_count INTEGER DEFAULT 0,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    metrics JSONB
                );
                
                CREATE INDEX IF NOT EXISTS idx_exec_idempotency 
                ON execution_records(idempotency_key);
                
                CREATE INDEX IF NOT EXISTS idx_exec_tenant 
                ON execution_records(tenant_id, created_at DESC);
            """)
            
        self.side_effect_ledger = SideEffectLedger(
            self.redis_client, 
            self.postgres_pool
        )
        
        # Initialize circuit breakers
        for service in self.config.get("services", []):
            self.circuit_breakers[service] = CircuitBreaker(
                failure_threshold=self.config.get("circuit_breaker_threshold", 5),
                timeout_seconds=self.config.get("circuit_breaker_timeout", 60)
            )
    
    @tracer.start_as_current_span("execute_task")
    async def execute(self, task: TaskDefinition) -> ExecutionRecord:
        """
        Execute a task with prepare/apply/commit pattern
        """
        span = trace.get_current_span()
        span.set_attributes({
            "task.id": task.task_id,
            "task.type": task.task_type,
            "task.idempotency_key": task.idempotency_key,
            "task.tenant_id": task.tenant_id
        })
        
        record = ExecutionRecord(
            task_id=task.task_id,
            idempotency_key=task.idempotency_key,
            state=ExecutionState.PENDING
        )
        
        try:
            # Check if already executed
            existing_state = await self.side_effect_ledger.get_execution_state(
                task.idempotency_key
            )
            
            if existing_state and existing_state["phase"] == "commit":
                # Already completed
                record.state = ExecutionState.COMMITTED
                record.commit_run_id = existing_state["data"]["run_id"]
                span.add_event("task_already_completed")
                return record
                
            # Start execution
            record.started_at = datetime.utcnow()
            record.state = ExecutionState.PREPARING
            
            # Phase 1: PREPARE
            prepare_success = await self._prepare(task, record)
            if not prepare_success:
                if task.delivery_semantics == DeliverySemantics.EXACTLY_ONCE:
                    # Already prepared by another execution
                    span.add_event("task_already_prepared")
                    return record
                    
            record.state = ExecutionState.PREPARED
            
            # Phase 2: APPLY
            apply_success = await self._apply(task, record)
            if not apply_success:
                # Retry or move to DLQ
                await self._handle_failure(task, record)
                return record
                
            record.state = ExecutionState.APPLIED
            
            # Phase 3: COMMIT
            commit_success = await self._commit(task, record)
            if commit_success:
                record.state = ExecutionState.COMMITTED
                record.completed_at = datetime.utcnow()
                
                # Record metrics
                duration_ms = (
                    record.completed_at - record.started_at
                ).total_seconds() * 1000
                record.metrics["duration_ms"] = duration_ms
                await self.metrics.record_execution(task.task_type, duration_ms, True)
                
        except Exception as e:
            record.state = ExecutionState.FAILED
            record.error = str(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            
            # Apply compensating actions if needed
            if task.compensating_action:
                await self._apply_compensating_action(task, record)
                
            # Handle retry or DLQ
            await self._handle_failure(task, record)
            
        finally:
            # Save execution record
            await self._save_record(record)
            
            span.set_attributes({
                "task.final_state": record.state.value,
                "task.retry_count": record.retry_count,
                "task.duration_ms": record.metrics.get("duration_ms", 0)
            })
            
        return record
    
    async def _prepare(self, task: TaskDefinition, record: ExecutionRecord) -> bool:
        """
        Prepare phase - validate and lock resources
        """
        with tracer.start_as_current_span("prepare_phase") as span:
            # Calculate task hash
            task_hash = hashlib.sha256(
                json.dumps(task.payload, sort_keys=True).encode()
            ).hexdigest()
            
            record.prepare_hash = task_hash
            
            # Check circuit breaker
            if task.task_type in self.circuit_breakers:
                breaker = self.circuit_breakers[task.task_type]
                if not breaker.can_execute():
                    span.add_event("circuit_breaker_open")
                    record.error = "Circuit breaker open"
                    return False
                    
            # Record in ledger
            prepared = await self.side_effect_ledger.record_prepare(
                task.idempotency_key,
                task_hash,
                {
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "tenant_id": task.tenant_id
                }
            )
            
            span.set_attribute("prepare.success", prepared)
            return prepared
    
    async def _apply(self, task: TaskDefinition, record: ExecutionRecord) -> bool:
        """
        Apply phase - execute the actual task
        """
        with tracer.start_as_current_span("apply_phase") as span:
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    self._execute_task_logic(task),
                    timeout=task.timeout_seconds
                )
                
                # Calculate diff hash
                diff_hash = hashlib.sha256(
                    json.dumps(result, sort_keys=True).encode()
                ).hexdigest()
                
                record.apply_diff_hash = diff_hash
                
                # Record side effects
                side_effects = result.get("side_effects", [])
                record.side_effects = side_effects
                
                # Record in ledger
                applied = await self.side_effect_ledger.record_apply(
                    task.idempotency_key,
                    diff_hash,
                    side_effects
                )
                
                span.set_attribute("apply.success", applied)
                return applied
                
            except asyncio.TimeoutError:
                span.add_event("task_timeout")
                record.error = f"Task timed out after {task.timeout_seconds}s"
                return False
                
            except Exception as e:
                span.add_event("task_execution_error", {"error": str(e)})
                record.error = str(e)
                
                # Update circuit breaker
                if task.task_type in self.circuit_breakers:
                    self.circuit_breakers[task.task_type].record_failure()
                    
                return False
    
    async def _commit(self, task: TaskDefinition, record: ExecutionRecord) -> bool:
        """
        Commit phase - finalize and confirm execution
        """
        with tracer.start_as_current_span("commit_phase") as span:
            # Generate run ID
            run_id = str(uuid.uuid4())
            record.commit_run_id = run_id
            
            # Prepare commit result
            result = {
                "task_id": task.task_id,
                "idempotency_key": task.idempotency_key,
                "prepare_hash": record.prepare_hash,
                "apply_diff_hash": record.apply_diff_hash,
                "side_effects": record.side_effects,
                "completed_at": datetime.utcnow().isoformat()
            }
            
            # Record in ledger
            committed = await self.side_effect_ledger.record_commit(
                task.idempotency_key,
                run_id,
                result
            )
            
            if committed and task.task_type in self.circuit_breakers:
                self.circuit_breakers[task.task_type].record_success()
                
            span.set_attribute("commit.success", committed)
            span.set_attribute("commit.run_id", run_id)
            
            return committed
    
    async def _execute_task_logic(self, task: TaskDefinition) -> Dict[str, Any]:
        """
        Actual task execution logic - to be overridden per task type
        """
        # This would be replaced with actual task execution
        # For now, simulate some work
        await asyncio.sleep(0.1)
        
        return {
            "status": "success",
            "result": f"Executed {task.task_type}",
            "side_effects": [
                {
                    "type": "database_write",
                    "table": "task_results",
                    "id": task.task_id
                }
            ]
        }
    
    async def _handle_failure(self, task: TaskDefinition, record: ExecutionRecord):
        """
        Handle task failure - retry or move to DLQ
        """
        record.retry_count += 1
        
        if record.retry_count < task.max_retries:
            # Schedule retry with exponential backoff
            delay = min(2 ** record.retry_count, 300)  # Max 5 minutes
            
            span = trace.get_current_span()
            span.add_event("scheduling_retry", {
                "retry_count": record.retry_count,
                "delay_seconds": delay
            })
            
            await asyncio.sleep(delay)
            await self.task_queue.put(task)
            
        else:
            # Move to dead letter queue
            record.state = ExecutionState.DEAD_LETTER
            await self.dead_letter_queue.put({
                "task": task,
                "record": record,
                "quarantine_until": datetime.utcnow() + timedelta(days=30)
            })
            
            span = trace.get_current_span()
            span.add_event("moved_to_dlq", {
                "task_id": task.task_id,
                "retry_count": record.retry_count
            })
    
    async def _apply_compensating_action(self, task: TaskDefinition, record: ExecutionRecord):
        """
        Apply compensating action for failed task
        """
        if task.compensating_action in self.compensating_actions:
            with tracer.start_as_current_span("compensating_action") as span:
                try:
                    record.state = ExecutionState.COMPENSATING
                    
                    action = self.compensating_actions[task.compensating_action]
                    await action(task, record)
                    
                    record.state = ExecutionState.COMPENSATED
                    record.compensating_actions_applied.append(task.compensating_action)
                    
                    span.set_attribute("compensating_action.success", True)
                    
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.set_attribute("compensating_action.success", False)
    
    async def _save_record(self, record: ExecutionRecord):
        """
        Save execution record to database
        """
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO execution_records 
                (task_id, idempotency_key, state, tenant_id, user_id, 
                 task_type, payload, error, retry_count, started_at, 
                 completed_at, metrics)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (task_id) 
                DO UPDATE SET 
                    state = $3, 
                    error = $8, 
                    retry_count = $9,
                    completed_at = $11,
                    metrics = $12
            """, record.task_id, record.idempotency_key, record.state.value,
                "", "", "", {}, record.error, record.retry_count,
                record.started_at, record.completed_at, 
                json.dumps(record.metrics))
    
    async def process_dlq(self):
        """
        Process dead letter queue - admin triggered
        """
        processed = []
        
        while not self.dead_letter_queue.empty():
            item = await self.dead_letter_queue.get()
            
            if item["quarantine_until"] > datetime.utcnow():
                # Still in quarantine
                await self.dead_letter_queue.put(item)
                break
                
            # Attempt reprocessing
            task = item["task"]
            task.max_retries = 1  # Only one more attempt
            
            result = await self.execute(task)
            processed.append({
                "task_id": task.task_id,
                "result": result.state.value
            })
            
        return processed


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance
    """
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        
    def can_execute(self) -> bool:
        """Check if circuit breaker allows execution"""
        if self.state == "closed":
            return True
            
        if self.state == "open":
            # Check if timeout has passed
            if (datetime.utcnow() - self.last_failure_time).seconds > self.timeout_seconds:
                self.state = "half-open"
                return True
            return False
            
        # Half-open - allow one request
        return True
    
    def record_success(self):
        """Record successful execution"""
        if self.state == "half-open":
            self.state = "closed"
            self.failure_count = 0
            
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class ExecutionMetrics:
    """
    Metrics collection for execution layer
    """
    
    def __init__(self):
        self.execution_times = []
        self.success_count = 0
        self.failure_count = 0
        self.task_type_metrics = {}
        
    async def record_execution(self, task_type: str, duration_ms: float, success: bool):
        """Record execution metrics"""
        self.execution_times.append(duration_ms)
        
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
            
        if task_type not in self.task_type_metrics:
            self.task_type_metrics[task_type] = {
                "count": 0,
                "total_duration_ms": 0,
                "success": 0,
                "failure": 0
            }
            
        metrics = self.task_type_metrics[task_type]
        metrics["count"] += 1
        metrics["total_duration_ms"] += duration_ms
        metrics["success" if success else "failure"] += 1
        
        # Emit to monitoring
        span = trace.get_current_span()
        span.set_attributes({
            f"metrics.{task_type}.count": metrics["count"],
            f"metrics.{task_type}.avg_duration_ms": 
                metrics["total_duration_ms"] / metrics["count"],
            f"metrics.{task_type}.success_rate": 
                metrics["success"] / metrics["count"] if metrics["count"] > 0 else 0
        })


# Example usage
async def main():
    """Example usage of execution runner"""
    config = {
        "redis_url": "redis://localhost",
        "database_url": "postgresql://user:pass@localhost/agentic_db",
        "circuit_breaker_threshold": 5,
        "circuit_breaker_timeout": 60,
        "services": ["model_api", "database", "storage"]
    }
    
    runner = ExecutionRunner(config)
    await runner.initialize()
    
    # Create a task
    task = TaskDefinition(
        task_id=str(uuid.uuid4()),
        idempotency_key="unique-operation-123",
        tenant_id="tenant-456",
        user_id="user-789",
        task_type="code_generation",
        payload={
            "prompt": "Generate a Python function",
            "model": "gpt-5"
        },
        timeout_seconds=30,
        max_retries=3,
        delivery_semantics=DeliverySemantics.EXACTLY_ONCE,
        compensating_action="rollback_code_generation"
    )
    
    # Execute task
    result = await runner.execute(task)
    print(f"Execution result: {result}")
    
    # Process DLQ (admin operation)
    dlq_results = await runner.process_dlq()
    print(f"DLQ processing: {dlq_results}")


if __name__ == "__main__":
    asyncio.run(main())
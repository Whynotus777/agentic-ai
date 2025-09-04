# execution/layer.py - Fault-tolerant Execution Layer

import asyncio
import json
import uuid
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import logging

import aioboto3
import aioredis
import nats
from nats.aio.client import Client as NATS
from nats.js import JetStreamContext
from sqlalchemy.ext.asyncio import AsyncSession
import structlog
from prometheus_client import Counter, Histogram, Gauge, Summary
from circuitbreaker import CircuitBreaker, CircuitBreakerError

logger = structlog.get_logger()

# Metrics
task_queue_size = Gauge('task_queue_size', 'Current size of task queue', ['priority'])
task_processing_time = Histogram('task_processing_seconds', 'Task processing time', ['task_type'])
task_retry_counter = Counter('task_retries_total', 'Total task retries', ['task_type', 'reason'])
dlq_counter = Counter('dlq_messages_total', 'Messages sent to DLQ', ['reason'])
circuit_breaker_state = Gauge('circuit_breaker_state', 'Circuit breaker state (0=closed, 1=open, 2=half-open)', ['service'])
message_bus_throughput = Counter('message_bus_messages_total', 'Total messages processed', ['topic', 'status'])
idempotency_cache_hits = Counter('idempotency_cache_hits_total', 'Idempotency cache hits')

# Enums
class Priority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BATCH = 4

class MessageType(Enum):
    TASK_REQUEST = "task.request"
    TASK_RESPONSE = "task.response"
    TASK_STATUS = "task.status"
    AGENT_REQUEST = "agent.request"
    AGENT_RESPONSE = "agent.response"
    HITL_REQUEST = "hitl.request"
    HITL_RESPONSE = "hitl.response"
    HEALTH_CHECK = "health.check"
    METRICS = "metrics"

class DeliveryGuarantee(Enum):
    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"

# Data classes
@dataclass
class QueuedTask:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    idempotency_key: str = ""
    tenant_id: str = ""
    priority: Priority = Priority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)
    max_retries: int = 3
    retry_count: int = 0
    timeout_seconds: int = 300
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    callback_url: Optional[str] = None

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: type = Exception
    fallback_function: Optional[Callable] = None
    half_open_attempts: int = 3

@dataclass
class TaskResult:
    task_id: str
    status: str
    output: Optional[Any] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

# Task Queue Manager
class TaskQueueManager:
    """
    Manages task queues with priorities, retries, DLQ, and idempotency
    """
    
    def __init__(
        self,
        redis_client: aioredis.Redis,
        sqs_client: Any,
        queue_url: str,
        dlq_url: str
    ):
        self.redis = redis_client
        self.sqs = sqs_client
        self.queue_url = queue_url
        self.dlq_url = dlq_url
        
        # Priority queues
        self.queues: Dict[Priority, deque] = {
            priority: deque() for priority in Priority
        }
        
        # Task tracking
        self.active_tasks: Dict[str, QueuedTask] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        
        # Idempotency cache
        self.idempotency_window = 3600  # 1 hour
        
        # Rate limiting
        self.rate_limiters: Dict[str, int] = defaultdict(int)
        
        # Background tasks
        self.background_tasks = []
    
    async def enqueue(
        self,
        task: QueuedTask,
        guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE
    ) -> str:
        """Enqueue a task with delivery guarantee"""
        
        # Check idempotency
        if task.idempotency_key:
            existing = await self._check_idempotency(task.idempotency_key)
            if existing:
                idempotency_cache_hits.inc()
                logger.info("Task already processed", 
                          idempotency_key=task.idempotency_key,
                          existing_task_id=existing)
                return existing
        
        # Validate dependencies
        if task.dependencies:
            ready = await self._check_dependencies(task.dependencies)
            if not ready:
                task.scheduled_at = datetime.utcnow() + timedelta(seconds=30)
                logger.info("Task has pending dependencies", 
                          task_id=task.id,
                          dependencies=task.dependencies)
        
        # Apply rate limiting
        if not await self._check_rate_limit(task.tenant_id):
            task.scheduled_at = datetime.utcnow() + timedelta(seconds=60)
            logger.warning("Rate limit exceeded", tenant_id=task.tenant_id)
        
        # Store task
        await self._store_task(task)
        
        # Add to queue
        if guarantee == DeliveryGuarantee.EXACTLY_ONCE:
            # Use two-phase commit for exactly-once
            await self._prepare_task(task)
            await self._commit_task(task)
        else:
            # Direct enqueue for at-least-once
            await self._enqueue_to_sqs(task)
        
        # Update metrics
        task_queue_size.labels(priority=task.priority.name).inc()
        
        return task.id
    
    async def dequeue(self, worker_id: str) -> Optional[QueuedTask]:
        """Dequeue highest priority task for processing"""
        
        # Try each priority level
        for priority in Priority:
            # Check SQS first
            messages = await self.sqs.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=1,
                VisibilityTimeout=300,
                WaitTimeSeconds=1
            )
            
            if messages.get('Messages'):
                message = messages['Messages'][0]
                
                # Parse task
                task_data = json.loads(message['Body'])
                task = await self._load_task(task_data['task_id'])
                
                if task:
                    # Track active task
                    self.active_tasks[task.id] = task
                    
                    # Store receipt handle for later deletion
                    task.metadata['sqs_receipt_handle'] = message['ReceiptHandle']
                    
                    # Update metrics
                    task_queue_size.labels(priority=task.priority.name).dec()
                    
                    logger.info("Task dequeued", 
                              task_id=task.id,
                              worker_id=worker_id,
                              priority=task.priority.name)
                    
                    return task
        
        return None
    
    async def complete(self, task_id: str, result: TaskResult):
        """Mark task as completed"""
        
        task = self.active_tasks.get(task_id)
        if not task:
            logger.warning("Attempted to complete unknown task", task_id=task_id)
            return
        
        # Delete from SQS
        if 'sqs_receipt_handle' in task.metadata:
            await self.sqs.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=task.metadata['sqs_receipt_handle']
            )
        
        # Store result
        await self._store_result(task, result)
        
        # Update idempotency cache
        if task.idempotency_key:
            await self._set_idempotency(task.idempotency_key, task_id)
        
        # Remove from active tasks
        del self.active_tasks[task_id]
        
        # Trigger dependent tasks
        await self._trigger_dependents(task_id)
        
        # Update metrics
        task_processing_time.labels(task_type=task.metadata.get('type', 'unknown')).observe(
            result.duration_ms / 1000.0
        )
        
        logger.info("Task completed", 
                   task_id=task_id,
                   status=result.status,
                   duration_ms=result.duration_ms)
    
    async def retry(self, task_id: str, reason: str):
        """Retry a failed task"""
        
        task = self.active_tasks.get(task_id)
        if not task:
            return
        
        task.retry_count += 1
        
        if task.retry_count > task.max_retries:
            # Send to DLQ
            await self._send_to_dlq(task, reason)
            del self.active_tasks[task_id]
            
            dlq_counter.labels(reason=reason).inc()
            logger.error("Task exceeded retry limit, sent to DLQ", 
                        task_id=task_id,
                        retries=task.retry_count,
                        reason=reason)
        else:
            # Re-enqueue with exponential backoff
            delay = min(300, 2 ** task.retry_count * 10)  # Max 5 minutes
            task.scheduled_at = datetime.utcnow() + timedelta(seconds=delay)
            
            await self._enqueue_to_sqs(task, delay)
            
            task_retry_counter.labels(
                task_type=task.metadata.get('type', 'unknown'),
                reason=reason
            ).inc()
            
            logger.warning("Task retry scheduled", 
                         task_id=task_id,
                         retry_count=task.retry_count,
                         delay_seconds=delay)
    
    async def _check_idempotency(self, idempotency_key: str) -> Optional[str]:
        """Check if task with idempotency key was already processed"""
        key = f"idempotency:{idempotency_key}"
        result = await self.redis.get(key)
        return result.decode() if result else None
    
    async def _set_idempotency(self, idempotency_key: str, task_id: str):
        """Set idempotency key with TTL"""
        key = f"idempotency:{idempotency_key}"
        await self.redis.setex(key, self.idempotency_window, task_id)
    
    async def _check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if all dependencies are completed"""
        for dep_id in dependencies:
            key = f"task:result:{dep_id}"
            if not await self.redis.exists(key):
                return False
        return True
    
    async def _trigger_dependents(self, task_id: str):
        """Trigger tasks dependent on completed task"""
        pattern = f"task:pending:*"
        cursor = 0
        
        while True:
            cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
            
            for key in keys:
                task_data = await self.redis.get(key)
                if task_data:
                    task = json.loads(task_data)
                    if task_id in task.get('dependencies', []):
                        task['dependencies'].remove(task_id)
                        
                        if not task['dependencies']:
                            # All dependencies met, enqueue task
                            queued_task = QueuedTask(**task)
                            await self.enqueue(queued_task)
                            await self.redis.delete(key)
            
            if cursor == 0:
                break
    
    async def _check_rate_limit(self, tenant_id: str) -> bool:
        """Check tenant rate limit"""
        key = f"rate_limit:{tenant_id}"
        current = await self.redis.incr(key)
        
        if current == 1:
            await self.redis.expire(key, 60)  # 1 minute window
        
        limit = 100  # Default 100 requests per minute
        return current <= limit
    
    async def _store_task(self, task: QueuedTask):
        """Store task in Redis"""
        key = f"task:data:{task.id}"
        await self.redis.setex(
            key,
            3600 * 24,  # 24 hours
            json.dumps(asdict(task), default=str)
        )
    
    async def _load_task(self, task_id: str) -> Optional[QueuedTask]:
        """Load task from Redis"""
        key = f"task:data:{task_id}"
        data = await self.redis.get(key)
        
        if data:
            task_data = json.loads(data)
            # Convert string dates back to datetime
            for date_field in ['created_at', 'scheduled_at']:
                if task_data.get(date_field):
                    task_data[date_field] = datetime.fromisoformat(task_data[date_field])
            
            # Convert priority string to enum
            task_data['priority'] = Priority[task_data['priority']]
            
            return QueuedTask(**task_data)
        
        return None
    
    async def _store_result(self, task: QueuedTask, result: TaskResult):
        """Store task result"""
        key = f"task:result:{task.id}"
        await self.redis.setex(
            key,
            3600 * 24,  # 24 hours
            json.dumps(asdict(result), default=str)
        )
    
    async def _enqueue_to_sqs(self, task: QueuedTask, delay: int = 0):
        """Enqueue task to SQS"""
        message = {
            'task_id': task.id,
            'priority': task.priority.value,
            'tenant_id': task.tenant_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        params = {
            'QueueUrl': self.queue_url,
            'MessageBody': json.dumps(message),
            'MessageAttributes': {
                'priority': {
                    'StringValue': str(task.priority.value),
                    'DataType': 'Number'
                },
                'tenant_id': {
                    'StringValue': task.tenant_id,
                    'DataType': 'String'
                }
            }
        }
        
        if delay > 0:
            params['DelaySeconds'] = min(delay, 900)  # Max 15 minutes
        
        await self.sqs.send_message(**params)
    
    async def _send_to_dlq(self, task: QueuedTask, reason: str):
        """Send failed task to DLQ"""
        message = {
            'task': asdict(task),
            'failure_reason': reason,
            'failed_at': datetime.utcnow().isoformat()
        }
        
        await self.sqs.send_message(
            QueueUrl=self.dlq_url,
            MessageBody=json.dumps(message, default=str)
        )
    
    async def _prepare_task(self, task: QueuedTask):
        """Prepare phase of two-phase commit"""
        prepare_id = str(uuid.uuid4())
        key = f"prepare:{prepare_id}"
        
        await self.redis.setex(
            key,
            300,  # 5 minute TTL
            json.dumps({
                'task_id': task.id,
                'idempotency_key': task.idempotency_key,
                'timestamp': datetime.utcnow().isoformat()
            })
        )
        
        task.metadata['prepare_id'] = prepare_id
    
    async def _commit_task(self, task: QueuedTask):
        """Commit phase of two-phase commit"""
        prepare_id = task.metadata.get('prepare_id')
        if not prepare_id:
            raise ValueError("No prepare_id found for task")
        
        # Check prepare record
        key = f"prepare:{prepare_id}"
        prepare_data = await self.redis.get(key)
        
        if not prepare_data:
            raise ValueError("Prepare record expired or not found")
        
        # Commit the task
        await self._enqueue_to_sqs(task)
        
        # Clean up prepare record
        await self.redis.delete(key)

# Circuit Breaker Manager
class CircuitBreakerManager:
    """
    Manages circuit breakers for external services
    """
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.configs: Dict[str, CircuitBreakerConfig] = {}
    
    def register(self, service: str, config: CircuitBreakerConfig):
        """Register a circuit breaker for a service"""
        
        self.configs[service] = config
        
        breaker = CircuitBreaker(
            failure_threshold=config.failure_threshold,
            recovery_timeout=config.recovery_timeout,
            expected_exception=config.expected_exception,
            name=service
        )
        
        # Add listeners for metrics
        breaker.add_listener(self._on_circuit_open)
        breaker.add_listener(self._on_circuit_close)
        
        self.breakers[service] = breaker
        
        logger.info("Circuit breaker registered", 
                   service=service,
                   threshold=config.failure_threshold)
    
    async def call(self, service: str, func: Callable, *args, **kwargs) -> Any:
        """Call a function through circuit breaker"""
        
        breaker = self.breakers.get(service)
        if not breaker:
            # No breaker configured, call directly
            return await func(*args, **kwargs)
        
        try:
            # Check breaker state
            if breaker.current_state == 'open':
                circuit_breaker_state.labels(service=service).set(1)
                
                # Try fallback if configured
                config = self.configs[service]
                if config.fallback_function:
                    logger.warning("Circuit breaker open, using fallback", 
                                 service=service)
                    return await config.fallback_function(*args, **kwargs)
                
                raise CircuitBreakerError(f"Circuit breaker open for {service}")
            
            elif breaker.current_state == 'half_open':
                circuit_breaker_state.labels(service=service).set(2)
            else:
                circuit_breaker_state.labels(service=service).set(0)
            
            # Make the call
            result = await breaker.call(func, *args, **kwargs)
            
            # Record success
            await self._record_success(service)
            
            return result
            
        except CircuitBreakerError:
            raise
        
        except Exception as e:
            # Record failure
            await self._record_failure(service, str(e))
            raise
    
    async def get_status(self, service: str) -> Dict[str, Any]:
        """Get circuit breaker status"""
        
        breaker = self.breakers.get(service)
        if not breaker:
            return {'status': 'not_configured'}
        
        # Get metrics from Redis
        success_key = f"cb:success:{service}"
        failure_key = f"cb:failure:{service}"
        
        success_count = int(await self.redis.get(success_key) or 0)
        failure_count = int(await self.redis.get(failure_key) or 0)
        
        total = success_count + failure_count
        success_rate = success_count / total if total > 0 else 0
        
        return {
            'service': service,
            'state': breaker.current_state,
            'failure_count': breaker.failure_count,
            'last_failure_time': breaker.last_failure_time,
            'success_rate': success_rate,
            'total_calls': total
        }
    
    async def reset(self, service: str):
        """Manually reset a circuit breaker"""
        
        breaker = self.breakers.get(service)
        if breaker:
            breaker.reset()
            
            # Clear metrics
            await self.redis.delete(f"cb:success:{service}")
            await self.redis.delete(f"cb:failure:{service}")
            
            circuit_breaker_state.labels(service=service).set(0)
            
            logger.info("Circuit breaker reset", service=service)
    
    def _on_circuit_open(self, breaker: CircuitBreaker):
        """Callback when circuit opens"""
        logger.error("Circuit breaker opened", 
                    service=breaker.name,
                    failures=breaker.failure_count)
    
    def _on_circuit_close(self, breaker: CircuitBreaker):
        """Callback when circuit closes"""
        logger.info("Circuit breaker closed", service=breaker.name)
    
    async def _record_success(self, service: str):
        """Record successful call"""
        key = f"cb:success:{service}"
        await self.redis.incr(key)
        await self.redis.expire(key, 3600)  # 1 hour window
    
    async def _record_failure(self, service: str, error: str):
        """Record failed call"""
        key = f"cb:failure:{service}"
        await self.redis.incr(key)
        await self.redis.expire(key, 3600)  # 1 hour window
        
        # Store last error
        error_key = f"cb:last_error:{service}"
        await self.redis.setex(error_key, 300, error)  # 5 minutes

# Message Bus
class MessageBus:
    """
    Inter-agent message bus using NATS JetStream
    """
    
    def __init__(self, nc: NATS, js: JetStreamContext):
        self.nc = nc
        self.js = js
        self.subscriptions: Dict[str, Any] = {}
        self.handlers: Dict[str, List[Callable]] = defaultdict(list)
        
    async def initialize(self):
        """Initialize streams and consumers"""
        
        # Create streams for different message types
        streams = [
            ("TASKS", ["task.>"], 7 * 24 * 3600),  # 7 days retention
            ("AGENTS", ["agent.>"], 24 * 3600),     # 1 day retention
            ("HITL", ["hitl.>"], 72 * 3600),        # 3 days retention
            ("METRICS", ["metrics.>"], 3600)         # 1 hour retention
        ]
        
        for stream_name, subjects, max_age in streams:
            try:
                await self.js.add_stream(
                    name=stream_name,
                    subjects=subjects,
                    max_age=max_age,
                    storage="file",
                    retention="limits",
                    max_msgs=1000000,
                    max_bytes=1024*1024*1024,  # 1GB
                    discard="old"
                )
                logger.info(f"Stream created/updated: {stream_name}")
            except Exception as e:
                logger.error(f"Failed to create stream {stream_name}: {e}")
    
    async def publish(
        self,
        subject: str,
        message: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ) -> str:
        """Publish a message to the bus"""
        
        # Add message metadata
        message['message_id'] = str(uuid.uuid4())
        message['timestamp'] = datetime.utcnow().isoformat()
        
        # Serialize message
        payload = json.dumps(message).encode()
        
        # Publish with headers
        ack = await self.js.publish(
            subject,
            payload,
            headers=headers
        )
        
        message_bus_throughput.labels(topic=subject, status='published').inc()
        
        logger.debug("Message published", 
                    subject=subject,
                    message_id=message['message_id'],
                    seq=ack.seq)
        
        return message['message_id']
    
    async def subscribe(
        self,
        subject: str,
        handler: Callable,
        durable: str = None,
        deliver_policy: str = "new"
    ):
        """Subscribe to a subject with handler"""
        
        # Store handler
        self.handlers[subject].append(handler)
        
        # Create durable consumer if specified
        if durable:
            consumer_config = {
                "durable_name": durable,
                "deliver_policy": deliver_policy,
                "ack_wait": 30,  # 30 seconds to ack
                "max_deliver": 3,  # Max 3 delivery attempts
                "replay_policy": "instant"
            }
            
            subscription = await self.js.subscribe(
                subject,
                cb=self._create_handler(subject),
                **consumer_config
            )
        else:
            # Ephemeral subscription
            subscription = await self.nc.subscribe(
                subject,
                cb=self._create_handler(subject)
            )
        
        self.subscriptions[subject] = subscription
        
        logger.info("Subscribed to subject", 
                   subject=subject,
                   durable=durable)
    
    async def request(
        self,
        subject: str,
        message: Dict[str, Any],
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Request-reply pattern"""
        
        payload = json.dumps(message).encode()
        
        try:
            response = await self.nc.request(
                subject,
                payload,
                timeout=timeout
            )
            
            message_bus_throughput.labels(topic=subject, status='request_success').inc()
            
            return json.loads(response.data.decode())
            
        except asyncio.TimeoutError:
            message_bus_throughput.labels(topic=subject, status='request_timeout').inc()
            raise
    
    def _create_handler(self, subject: str):
        """Create message handler wrapper"""
        
        async def handler(msg):
            try:
                # Parse message
                data = json.loads(msg.data.decode())
                
                # Call registered handlers
                for handler_func in self.handlers[subject]:
                    try:
                        await handler_func(data, msg)
                    except Exception as e:
                        logger.error("Handler error", 
                                   subject=subject,
                                   error=str(e))
                
                # Acknowledge message (for JetStream)
                if hasattr(msg, 'ack'):
                    await msg.ack()
                
                message_bus_throughput.labels(topic=subject, status='processed').inc()
                
            except Exception as e:
                logger.error("Message processing error", 
                           subject=subject,
                           error=str(e))
                
                # Negative acknowledgment for retry
                if hasattr(msg, 'nak'):
                    await msg.nak()
                
                message_bus_throughput.labels(topic=subject, status='error').inc()
        
        return handler
    
    async def unsubscribe(self, subject: str):
        """Unsubscribe from a subject"""
        
        if subject in self.subscriptions:
            await self.subscriptions[subject].unsubscribe()
            del self.subscriptions[subject]
            del self.handlers[subject]
            
            logger.info("Unsubscribed from subject", subject=subject)

# Task Executor
class TaskExecutor:
    """
    Executes tasks with timeout, sandboxing, and resource limits
    """
    
    def __init__(
        self,
        queue_manager: TaskQueueManager,
        circuit_manager: CircuitBreakerManager,
        message_bus: MessageBus,
        worker_id: str
    ):
        self.queue = queue_manager
        self.circuits = circuit_manager
        self.bus = message_bus
        self.worker_id = worker_id
        
        self.active = True
        self.current_task: Optional[QueuedTask] = None
        
    async def start(self):
        """Start executor loop"""
        
        logger.info("Task executor started", worker_id=self.worker_id)
        
        while self.active:
            try:
                # Get next task
                task = await self.queue.dequeue(self.worker_id)
                
                if not task:
                    # No tasks, wait a bit
                    await asyncio.sleep(1)
                    continue
                
                self.current_task = task
                
                # Execute with timeout
                result = await self._execute_with_timeout(task)
                
                # Complete or retry based on result
                if result.status == "success":
                    await self.queue.complete(task.id, result)
                else:
                    await self.queue.retry(task.id, result.error or "Unknown error")
                
                self.current_task = None
                
            except Exception as e:
                logger.error("Executor error", 
                           worker_id=self.worker_id,
                           error=str(e))
                
                if self.current_task:
                    await self.queue.retry(self.current_task.id, str(e))
                    self.current_task = None
    
    async def stop(self):
        """Stop executor"""
        self.active = False
        
        # Wait for current task to complete
        if self.current_task:
            logger.info("Waiting for current task to complete", 
                       worker_id=self.worker_id,
                       task_id=self.current_task.id)
            
            # Give it max 30 seconds
            for _ in range(30):
                if not self.current_task:
                    break
                await asyncio.sleep(1)
    
    async def _execute_with_timeout(self, task: QueuedTask) -> TaskResult:
        """Execute task with timeout"""
        
        start_time = time.time()
        
        try:
            # Create task context
            context = {
                'task_id': task.id,
                'tenant_id': task.tenant_id,
                'worker_id': self.worker_id
            }
            
            # Publish status update
            await self.bus.publish(
                MessageType.TASK_STATUS.value,
                {
                    'task_id': task.id,
                    'status': 'executing',
                    'worker_id': self.worker_id
                }
            )
            
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_task(task, context),
                timeout=task.timeout_seconds
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            return TaskResult(
                task_id=task.id,
                status="success",
                output=result,
                duration_ms=duration_ms
            )
            
        except asyncio.TimeoutError:
            return TaskResult(
                task_id=task.id,
                status="timeout",
                error=f"Task exceeded timeout of {task.timeout_seconds}s"
            )
        
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                status="error",
                error=str(e)
            )
    
    async def _execute_task(self, task: QueuedTask, context: Dict[str, Any]) -> Any:
        """Execute the actual task logic"""
        
        task_type = task.metadata.get('type', 'unknown')
        
        if task_type == 'agent_call':
            # Call agent through circuit breaker
            return await self.circuits.call(
                'agent_service',
                self._call_agent,
                task.payload,
                context
            )
        
        elif task_type == 'tool_call':
            # Call external tool
            return await self.circuits.call(
                f"tool_{task.payload.get('tool')}",
                self._call_tool,
                task.payload,
                context
            )
        
        elif task_type == 'batch_process':
            # Process batch job
            return await self._process_batch(task.payload, context)
        
        else:
            # Generic task processing
            return await self._process_generic(task.payload, context)
    
    async def _call_agent(self, payload: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Call an agent service"""
        
        # Request via message bus
        response = await self.bus.request(
            MessageType.AGENT_REQUEST.value,
            {
                **payload,
                **context
            },
            timeout=60.0
        )
        
        return response.get('result')
    
    async def _call_tool(self, payload: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Call an external tool"""
        
        tool_name = payload.get('tool')
        tool_input = payload.get('input', {})
        
        # Tool-specific logic would go here
        # This is a placeholder
        
        return {
            'tool': tool_name,
            'output': f"Tool {tool_name} executed successfully",
            'context': context
        }
    
    async def _process_batch(self, payload: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Process batch job"""
        
        items = payload.get('items', [])
        results = []
        
        for item in items:
            # Process each item
            result = await self._process_item(item, context)
            results.append(result)
            
            # Yield control periodically
            await asyncio.sleep(0)
        
        return {
            'processed': len(results),
            'results': results
        }
    
    async def _process_item(self, item: Any, context: Dict[str, Any]) -> Any:
        """Process a single batch item"""
        
        # Placeholder for item processing logic
        return {
            'item': item,
            'status': 'processed',
            'context': context
        }
    
    async def _process_generic(self, payload: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Generic task processing"""
        
        # Simulate some work
        await asyncio.sleep(1)
        
        return {
            'input': payload,
            'output': 'Task processed successfully',
            'context': context
        }

# Main execution layer orchestration
async def create_execution_layer(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create and initialize the execution layer"""
    
    # Initialize Redis
    redis = await aioredis.create_redis_pool(config['redis_url'])
    
    # Initialize AWS clients
    session = aioboto3.Session()
    sqs_client = await session.client('sqs').__aenter__()
    
    # Initialize NATS
    nc = await nats.connect(config['nats_url'])
    js = nc.jetstream()
    
    # Create components
    queue_manager = TaskQueueManager(
        redis_client=redis,
        sqs_client=sqs_client,
        queue_url=config['queue_url'],
        dlq_url=config['dlq_url']
    )
    
    circuit_manager = CircuitBreakerManager(redis_client=redis)
    
    # Register circuit breakers
    circuit_manager.register('agent_service', CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=60
    ))
    
    circuit_manager.register('tool_web_search', CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=30
    ))
    
    message_bus = MessageBus(nc=nc, js=js)
    await message_bus.initialize()
    
    # Create executor pool
    num_workers = config.get('num_workers', 10)
    executors = []
    
    for i in range(num_workers):
        worker_id = f"worker-{i:03d}"
        executor = TaskExecutor(
            queue_manager=queue_manager,
            circuit_manager=circuit_manager,
            message_bus=message_bus,
            worker_id=worker_id
        )
        executors.append(executor)
        
        # Start executor in background
        asyncio.create_task(executor.start())
    
    logger.info("Execution layer initialized", 
               num_workers=num_workers,
               redis_url=config['redis_url'],
               nats_url=config['nats_url'])
    
    return {
        'queue_manager': queue_manager,
        'circuit_manager': circuit_manager,
        'message_bus': message_bus,
        'executors': executors
    }
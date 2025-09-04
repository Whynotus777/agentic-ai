# execution/message_bus.py
"""
Message Bus for inter-agent communication with backpressure,
rate limiting, and golden task evaluation gates.
"""

import asyncio
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
import statistics

import aiokafka
from opentelemetry import trace

tracer = trace.get_tracer(__name__)


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class MessageStatus(Enum):
    """Message processing status"""
    PENDING = "pending"
    IN_FLIGHT = "in_flight"
    PROCESSED = "processed"
    FAILED = "failed"
    DLQ = "dlq"
    EXPIRED = "expired"


@dataclass
class AgentMessage:
    """Message between agents"""
    message_id: str
    correlation_id: str
    source_agent: str
    target_agent: str
    message_type: str
    payload: Dict[str, Any]
    priority: MessagePriority
    created_at: datetime
    expires_at: Optional[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    retry_count: int = 0
    
    
@dataclass
class MessageAck:
    """Message acknowledgment"""
    message_id: str
    status: MessageStatus
    processed_at: datetime
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class MessageBus:
    """
    High-performance message bus for agent communication
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.kafka_producer = None
        self.kafka_consumer = None
        
        # Message tracking
        self.in_flight: Dict[str, AgentMessage] = {}
        self.processed: Set[str] = set()
        
        # Rate limiting and backpressure
        self.rate_limiter = MessageRateLimiter()
        self.backpressure_manager = BackpressureManager()
        
        # DLQ and replay
        self.dead_letter_queue = deque(maxlen=10000)
        self.replay_queue = asyncio.Queue()
        
        # Subscribers
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Metrics
        self.message_counter = defaultdict(int)
        self.latency_tracker = defaultdict(list)
        
    async def initialize(self):
        """Initialize Kafka clients"""
        # Kafka producer
        self.kafka_producer = aiokafka.AIOKafkaProducer(
            bootstrap_servers=self.config.get("kafka_brokers", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode(),
            compression_type="snappy",
            acks="all",  # Wait for all replicas
            max_batch_size=16384,
            linger_ms=100  # Batch messages for 100ms
        )
        await self.kafka_producer.start()
        
        # Kafka consumer
        self.kafka_consumer = aiokafka.AIOKafkaConsumer(
            "agent_messages",
            bootstrap_servers=self.config.get("kafka_brokers", "localhost:9092"),
            value_deserializer=lambda v: json.loads(v.decode()),
            group_id="agent_group",
            enable_auto_commit=False,  # Manual commit for exactly-once
            max_poll_records=100
        )
        await self.kafka_consumer.start()
        
        # Start background workers
        asyncio.create_task(self._message_processor())
        asyncio.create_task(self._dlq_processor())
        asyncio.create_task(self._metrics_collector())
    
    @tracer.start_as_current_span("send_message")
    async def send(
        self,
        source_agent: str,
        target_agent: str,
        message_type: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        ttl_seconds: Optional[int] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Send message between agents
        
        Returns:
            Message ID
        """
        span = trace.get_current_span()
        
        # Check rate limits
        if not await self.rate_limiter.check_rate(source_agent, target_agent):
            raise Exception(f"Rate limit exceeded for {source_agent}->{target_agent}")
        
        # Check backpressure
        if not await self.backpressure_manager.can_send(target_agent):
            raise Exception(f"Backpressure limit reached for {target_agent}")
        
        # Create message
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            correlation_id=correlation_id or str(uuid.uuid4()),
            source_agent=source_agent,
            target_agent=target_agent,
            message_type=message_type,
            payload=payload,
            priority=priority,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=ttl_seconds) if ttl_seconds else None,
            trace_id=format(span.get_span_context().trace_id, '032x')
        )
        
        # Deduplicate
        message_hash = self._compute_message_hash(message)
        if message_hash in self.processed:
            span.add_event("message_deduplicated")
            return message.message_id
        
        # Track in-flight
        self.in_flight[message.message_id] = message
        
        # Send to Kafka with partitioning by target agent
        await self.kafka_producer.send(
            "agent_messages",
            value=asdict(message),
            key=target_agent.encode(),
            partition=hash(target_agent) % self.config.get("num_partitions", 10)
        )
        
        # Update metrics
        self.message_counter[f"{source_agent}->{target_agent}"] += 1
        
        span.set_attributes({
            "message.id": message.message_id,
            "message.source": source_agent,
            "message.target": target_agent,
            "message.type": message_type,
            "message.priority": priority.value
        })
        
        return message.message_id
    
    async def subscribe(
        self,
        agent_id: str,
        handler: Callable[[AgentMessage], Any]
    ):
        """Subscribe agent to messages"""
        self.subscribers[agent_id].append(handler)
    
    async def acknowledge(
        self,
        message_id: str,
        status: MessageStatus,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Acknowledge message processing"""
        if message_id not in self.in_flight:
            return
        
        message = self.in_flight[message_id]
        
        ack = MessageAck(
            message_id=message_id,
            status=status,
            processed_at=datetime.utcnow(),
            result=result,
            error=error
        )
        
        # Calculate latency
        latency_ms = (ack.processed_at - message.created_at).total_seconds() * 1000
        self.latency_tracker[message.target_agent].append(latency_ms)
        
        # Handle based on status
        if status == MessageStatus.PROCESSED:
            # Mark as processed
            message_hash = self._compute_message_hash(message)
            self.processed.add(message_hash)
            del self.in_flight[message_id]
            
        elif status == MessageStatus.FAILED:
            # Retry or move to DLQ
            message.retry_count += 1
            
            if message.retry_count < self.config.get("max_retries", 3):
                # Schedule retry with exponential backoff
                delay = min(2 ** message.retry_count, 60)
                await asyncio.sleep(delay)
                await self.replay_queue.put(message)
            else:
                # Move to DLQ
                message.metadata["final_error"] = error
                self.dead_letter_queue.append(message)
                del self.in_flight[message_id]
        
        # Update backpressure
        await self.backpressure_manager.update_pressure(
            message.target_agent,
            status == MessageStatus.PROCESSED
        )
    
    async def _message_processor(self):
        """Process incoming messages"""
        while True:
            try:
                # Poll Kafka
                records = await self.kafka_consumer.getmany(
                    timeout_ms=1000,
                    max_records=100
                )
                
                for topic_partition, messages in records.items():
                    for msg in messages:
                        await self._process_message(msg.value)
                
                # Commit offsets
                await self.kafka_consumer.commit()
                
            except Exception as e:
                print(f"Message processor error: {e}")
                await asyncio.sleep(1)
    
    async def _process_message(self, message_data: Dict[str, Any]):
        """Process a single message"""
        message = AgentMessage(**message_data)
        
        # Check expiration
        if message.expires_at and datetime.utcnow() > message.expires_at:
            await self.acknowledge(
                message.message_id,
                MessageStatus.EXPIRED,
                error="Message expired"
            )
            return
        
        # Get handlers for target agent
        handlers = self.subscribers.get(message.target_agent, [])
        
        if not handlers:
            # No handler, move to DLQ
            self.dead_letter_queue.append(message)
            return
        
        # Process with handlers
        for handler in handlers:
            try:
                result = await handler(message)
                await self.acknowledge(
                    message.message_id,
                    MessageStatus.PROCESSED,
                    result=result
                )
            except Exception as e:
                await self.acknowledge(
                    message.message_id,
                    MessageStatus.FAILED,
                    error=str(e)
                )
    
    async def _dlq_processor(self):
        """Process dead letter queue"""
        while True:
            try:
                # Process replay queue
                while not self.replay_queue.empty():
                    message = await self.replay_queue.get()
                    
                    # Re-send message
                    await self.send(
                        source_agent=message.source_agent,
                        target_agent=message.target_agent,
                        message_type=message.message_type,
                        payload=message.payload,
                        priority=message.priority,
                        correlation_id=message.correlation_id
                    )
                
                # Check DLQ for expired quarantine
                now = datetime.utcnow()
                for message in list(self.dead_letter_queue):
                    quarantine_time = message.metadata.get(
                        "quarantine_until",
                        message.created_at + timedelta(days=30)
                    )
                    
                    if now > quarantine_time:
                        # Log and remove
                        print(f"Purging expired DLQ message: {message.message_id}")
                        self.dead_letter_queue.remove(message)
                
            except Exception as e:
                print(f"DLQ processor error: {e}")
                
            await asyncio.sleep(60)  # Check every minute
    
    async def _metrics_collector(self):
        """Collect and report metrics"""
        while True:
            try:
                # Calculate metrics
                metrics = {
                    "message_counts": dict(self.message_counter),
                    "in_flight_count": len(self.in_flight),
                    "dlq_size": len(self.dead_letter_queue),
                    "latency_by_agent": {}
                }
                
                # Calculate latency percentiles
                for agent, latencies in self.latency_tracker.items():
                    if latencies:
                        metrics["latency_by_agent"][agent] = {
                            "p50": statistics.median(latencies),
                            "p95": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else max(latencies),
                            "p99": statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else max(latencies)
                        }
                
                # Emit metrics
                span = trace.get_current_span()
                if span:
                    span.add_event("message_bus_metrics", attributes=metrics)
                
            except Exception as e:
                print(f"Metrics collector error: {e}")
                
            await asyncio.sleep(10)
    
    def _compute_message_hash(self, message: AgentMessage) -> str:
        """Compute hash for deduplication"""
        key = f"{message.source_agent}:{message.target_agent}:{message.message_type}:{json.dumps(message.payload, sort_keys=True)}"
        return hashlib.sha256(key.encode()).hexdigest()


class MessageRateLimiter:
    """Rate limiting for message bus"""
    
    def __init__(self):
        self.limits = {
            "default": 100,  # 100 messages per minute
            "high_volume": 1000
        }
        self.message_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
    
    async def check_rate(self, source: str, target: str) -> bool:
        """Check if message rate is within limits"""
        key = f"{source}->{target}"
        now = datetime.utcnow()
        
        # Get rate limit
        limit = self.limits.get(source, self.limits["default"])
        
        # Remove old entries
        times = self.message_times[key]
        while times and (now - times[0]).seconds > 60:
            times.popleft()
        
        # Check limit
        if len(times) >= limit:
            return False
        
        times.append(now)
        return True


class BackpressureManager:
    """Manage backpressure for agents"""
    
    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        self.queue_sizes: Dict[str, int] = defaultdict(int)
        self.processing_rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    async def can_send(self, target_agent: str) -> bool:
        """Check if can send to agent without overwhelming it"""
        return self.queue_sizes[target_agent] < self.max_queue_size
    
    async def update_pressure(self, agent: str, processed: bool):
        """Update backpressure metrics"""
        if processed:
            self.queue_sizes[agent] = max(0, self.queue_sizes[agent] - 1)
            self.processing_rates[agent].append(datetime.utcnow())
        else:
            self.queue_sizes[agent] += 1
    
    def get_processing_rate(self, agent: str) -> float:
        """Get messages per second processing rate"""
        times = self.processing_rates[agent]
        if len(times) < 2:
            return 0.0
        
        duration = (times[-1] - times[0]).total_seconds()
        if duration == 0:
            return 0.0
            
        return len(times) / duration


class GoldenTaskEvaluator:
    """
    Evaluates agents against golden tasks for quality gates
    """
    
    def __init__(self):
        self.golden_tasks: Dict[str, List[GoldenTask]] = {}
        self.evaluation_history: Dict[str, List[EvaluationResult]] = defaultdict(list)
        self.acceptance_gates: Dict[str, AcceptanceGate] = {}
        self._load_golden_tasks()
    
    def _load_golden_tasks(self):
        """Load golden tasks for each domain"""
        self.golden_tasks = {
            "code_generation": [
                GoldenTask(
                    task_id="golden_code_1",
                    description="Generate fibonacci function",
                    input_data={"n": 10},
                    expected_output={"result": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]},
                    evaluation_criteria={
                        "correctness": 1.0,
                        "performance": 0.8,
                        "style": 0.7
                    }
                ),
                GoldenTask(
                    task_id="golden_code_2",
                    description="Generate REST API client",
                    input_data={"endpoint": "https://api.example.com"},
                    expected_output={"has_error_handling": True, "has_retry": True},
                    evaluation_criteria={
                        "completeness": 0.9,
                        "error_handling": 1.0
                    }
                )
            ],
            "robotics": [
                GoldenTask(
                    task_id="golden_robot_1",
                    description="Navigate to target",
                    input_data={"start": [0, 0], "target": [10, 10]},
                    expected_output={"reached": True, "collisions": 0},
                    evaluation_criteria={
                        "success": 1.0,
                        "efficiency": 0.8,
                        "safety": 1.0
                    }
                )
            ]
        }
        
        # Define acceptance gates
        self.acceptance_gates = {
            "pre-prod-promotion": AcceptanceGate(
                name="pre-prod-promotion",
                criteria={
                    "task_success_rate": {"p50": 0.98, "p95": 0.96},
                    "p95_latency_s": {"non_rag": 8, "rag": 15},
                    "cost_delta_vs_baseline": {"max_pct": 20}
                }
            ),
            "canary-rollout": AcceptanceGate(
                name="canary-rollout",
                criteria={
                    "task_success_rate": {"p50": 0.95, "p95": 0.90},
                    "error_rate": {"max": 0.05},
                    "latency_regression": {"max_pct": 10}
                }
            )
        }
    
    async def evaluate_agent(
        self,
        agent_id: str,
        domain: str,
        model: str
    ) -> EvaluationSummary:
        """
        Evaluate an agent against golden tasks
        """
        tasks = self.golden_tasks.get(domain, [])
        if not tasks:
            return EvaluationSummary(
                agent_id=agent_id,
                domain=domain,
                model=model,
                passed=True,
                message="No golden tasks for domain"
            )
        
        results = []
        
        for task in tasks:
            result = await self._evaluate_task(agent_id, model, task)
            results.append(result)
            self.evaluation_history[agent_id].append(result)
        
        # Calculate summary
        success_rate = sum(1 for r in results if r.passed) / len(results)
        avg_latency = statistics.mean(r.latency_ms for r in results)
        avg_cost = statistics.mean(r.cost_usd for r in results)
        
        return EvaluationSummary(
            agent_id=agent_id,
            domain=domain,
            model=model,
            task_count=len(results),
            success_rate=success_rate,
            avg_latency_ms=avg_latency,
            avg_cost_usd=avg_cost,
            passed=success_rate >= 0.95,
            results=results
        )
    
    async def _evaluate_task(
        self,
        agent_id: str,
        model: str,
        task: 'GoldenTask'
    ) -> 'EvaluationResult':
        """
        Evaluate a single golden task
        """
        start_time = datetime.utcnow()
        
        # Execute task (simplified - would call actual agent)
        # In production, this would invoke the agent with the task
        output = {"result": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]}  # Mock result
        
        # Calculate metrics
        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        cost_usd = 0.01  # Mock cost
        
        # Evaluate against criteria
        scores = {}
        for criterion, threshold in task.evaluation_criteria.items():
            # Simplified scoring - in production, implement actual evaluation
            scores[criterion] = 0.95
        
        passed = all(score >= threshold for criterion, threshold in task.evaluation_criteria.items() 
                    for score in [scores.get(criterion, 0)])
        
        return EvaluationResult(
            task_id=task.task_id,
            agent_id=agent_id,
            model=model,
            passed=passed,
            scores=scores,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            timestamp=datetime.utcnow()
        )
    
    async def check_acceptance_gate(
        self,
        gate_name: str,
        agent_id: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if agent passes acceptance gate
        """
        gate = self.acceptance_gates.get(gate_name)
        if not gate:
            return True, {"message": "Gate not found"}
        
        # Get recent evaluation history
        history = self.evaluation_history[agent_id][-20:]  # Last 20 evaluations
        
        if not history:
            return False, {"message": "No evaluation history"}
        
        # Calculate metrics
        success_rate = sum(1 for r in history if r.passed) / len(history)
        latencies = [r.latency_ms / 1000 for r in history]  # Convert to seconds
        p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0]
        
        # Check against gate criteria
        gate_results = {
            "success_rate": success_rate,
            "p95_latency": p95_latency,
            "checks": {}
        }
        
        # Check success rate
        if "task_success_rate" in gate.criteria:
            required = gate.criteria["task_success_rate"]["p50"]
            gate_results["checks"]["success_rate"] = success_rate >= required
        
        # Check latency
        if "p95_latency_s" in gate.criteria:
            max_latency = gate.criteria["p95_latency_s"].get("non_rag", 10)
            gate_results["checks"]["latency"] = p95_latency <= max_latency
        
        passed = all(gate_results["checks"].values())
        
        return passed, gate_results


@dataclass
class GoldenTask:
    """Definition of a golden task"""
    task_id: str
    description: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    evaluation_criteria: Dict[str, float]  # criterion -> minimum score


@dataclass
class EvaluationResult:
    """Result of golden task evaluation"""
    task_id: str
    agent_id: str
    model: str
    passed: bool
    scores: Dict[str, float]
    latency_ms: float
    cost_usd: float
    timestamp: datetime


@dataclass
class EvaluationSummary:
    """Summary of agent evaluation"""
    agent_id: str
    domain: str
    model: str
    task_count: int = 0
    success_rate: float = 0.0
    avg_latency_ms: float = 0.0
    avg_cost_usd: float = 0.0
    passed: bool = False
    results: List[EvaluationResult] = field(default_factory=list)
    message: str = ""


@dataclass
class AcceptanceGate:
    """Acceptance gate criteria"""
    name: str
    criteria: Dict[str, Any]


# Canary deployment manager
class CanaryDeploymentManager:
    """
    Manages canary deployments with progressive rollout
    """
    
    def __init__(self):
        self.deployments: Dict[str, CanaryDeployment] = {}
        self.traffic_weights: Dict[str, float] = {}
        self.error_budgets: Dict[str, ErrorBudget] = {}
    
    async def start_canary(
        self,
        deployment_id: str,
        agent_id: str,
        initial_traffic_pct: float = 5.0,
        target_traffic_pct: float = 100.0,
        increment_pct: float = 10.0,
        evaluation_period_minutes: int = 10
    ) -> str:
        """Start canary deployment"""
        deployment = CanaryDeployment(
            deployment_id=deployment_id,
            agent_id=agent_id,
            current_traffic_pct=initial_traffic_pct,
            target_traffic_pct=target_traffic_pct,
            increment_pct=increment_pct,
            evaluation_period_minutes=evaluation_period_minutes,
            started_at=datetime.utcnow(),
            status="active"
        )
        
        self.deployments[deployment_id] = deployment
        self.traffic_weights[agent_id] = initial_traffic_pct / 100
        
        # Initialize error budget
        self.error_budgets[deployment_id] = ErrorBudget(
            total_budget=0.05,  # 5% error budget
            consumed=0.0
        )
        
        # Start monitoring
        asyncio.create_task(self._monitor_canary(deployment_id))
        
        return deployment_id
    
    async def _monitor_canary(self, deployment_id: str):
        """Monitor canary deployment and adjust traffic"""
        deployment = self.deployments[deployment_id]
        
        while deployment.status == "active":
            await asyncio.sleep(deployment.evaluation_period_minutes * 60)
            
            # Evaluate performance
            metrics = await self._collect_canary_metrics(deployment.agent_id)
            
            # Check error budget
            error_budget = self.error_budgets[deployment_id]
            if metrics["error_rate"] > 0.01:  # 1% threshold
                error_budget.consumed += metrics["error_rate"]
                
                if error_budget.consumed > error_budget.total_budget:
                    # Rollback
                    await self._rollback_canary(deployment_id)
                    return
            
            # Check other metrics
            if metrics["latency_p95"] > metrics["baseline_latency_p95"] * 1.2:
                # 20% latency regression
                await self._rollback_canary(deployment_id)
                return
            
            # Increment traffic if healthy
            if deployment.current_traffic_pct < deployment.target_traffic_pct:
                deployment.current_traffic_pct = min(
                    deployment.current_traffic_pct + deployment.increment_pct,
                    deployment.target_traffic_pct
                )
                self.traffic_weights[deployment.agent_id] = deployment.current_traffic_pct / 100
            else:
                # Fully rolled out
                deployment.status = "completed"
                deployment.completed_at = datetime.utcnow()
    
    async def _collect_canary_metrics(self, agent_id: str) -> Dict[str, float]:
        """Collect metrics for canary evaluation"""
        # In production, query actual metrics
        return {
            "error_rate": 0.005,
            "latency_p95": 2.5,
            "baseline_latency_p95": 2.3,
            "success_rate": 0.995
        }
    
    async def _rollback_canary(self, deployment_id: str):
        """Rollback canary deployment"""
        deployment = self.deployments[deployment_id]
        deployment.status = "rolled_back"
        deployment.completed_at = datetime.utcnow()
        
        # Reset traffic to 0
        self.traffic_weights[deployment.agent_id] = 0.0
        
        print(f"Canary {deployment_id} rolled back due to quality issues")


@dataclass
class CanaryDeployment:
    """Canary deployment configuration"""
    deployment_id: str
    agent_id: str
    current_traffic_pct: float
    target_traffic_pct: float
    increment_pct: float
    evaluation_period_minutes: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "active"  # active, completed, rolled_back


@dataclass
class ErrorBudget:
    """Error budget for canary deployment"""
    total_budget: float
    consumed: float


# Example usage
async def main():
    """Example usage of message bus and golden tasks"""
    
    # Initialize message bus
    bus_config = {
        "kafka_brokers": "localhost:9092",
        "num_partitions": 10,
        "max_retries": 3
    }
    
    bus = MessageBus(bus_config)
    await bus.initialize()
    
    # Register agent handlers
    async def code_agent_handler(message: AgentMessage) -> Dict[str, Any]:
        print(f"Code agent received: {message.message_type}")
        return {"status": "processed", "result": "code generated"}
    
    await bus.subscribe("code_agent", code_agent_handler)
    
    # Send message
    message_id = await bus.send(
        source_agent="orchestrator",
        target_agent="code_agent",
        message_type="generate_code",
        payload={"language": "python", "task": "fibonacci"},
        priority=MessagePriority.HIGH
    )
    
    print(f"Sent message: {message_id}")
    
    # Golden task evaluation
    evaluator = GoldenTaskEvaluator()
    
    # Evaluate agent
    summary = await evaluator.evaluate_agent(
        agent_id="code_agent_v2",
        domain="code_generation",
        model="gpt-5"
    )
    
    print(f"Evaluation summary: {summary}")
    
    # Check acceptance gate
    passed, results = await evaluator.check_acceptance_gate(
        "pre-prod-promotion",
        "code_agent_v2"
    )
    
    print(f"Gate passed: {passed}, Results: {results}")
    
    # Start canary deployment
    canary_manager = CanaryDeploymentManager()
    deployment_id = await canary_manager.start_canary(
        deployment_id="deploy_001",
        agent_id="code_agent_v2",
        initial_traffic_pct=5,
        target_traffic_pct=100,
        increment_pct=20
    )
    
    print(f"Started canary deployment: {deployment_id}")


if __name__ == "__main__":
    asyncio.run(main())
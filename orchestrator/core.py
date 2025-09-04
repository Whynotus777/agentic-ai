# orchestrator/core.py - Multi-Agent Orchestrator System

import asyncio
import json
import uuid
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from contextlib import asynccontextmanager

import aioredis
import aioboto3
from sqlalchemy import create_engine, Column, String, JSON, DateTime, Float, Integer, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
import anthropic
import openai
from openai import AsyncOpenAI
import google.generativeai as genai
from prometheus_client import Counter, Histogram, Gauge
import structlog
from circuitbreaker import circuit
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx
from opentelemetry import trace, metrics
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Initialize structured logging
logger = structlog.get_logger()

# Database models
Base = declarative_base()

class TaskExecution(Base):
    __tablename__ = 'task_executions'
    
    id = Column(String, primary_key=True)
    trace_id = Column(String, nullable=False, index=True)
    tenant_id = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False)
    orchestrator_model = Column(String)
    agent_models = Column(JSON)
    task_plan = Column(JSON)
    input_data = Column(JSON)
    output_data = Column(JSON)
    artifacts = Column(JSON)
    cost_usd = Column(Float, default=0.0)
    tokens_used = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    error_message = Column(String)
    metadata = Column(JSON)

class PolicyRule(Base):
    __tablename__ = 'policy_rules'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    rule_type = Column(String, nullable=False)  # 'access', 'cost', 'rate_limit', 'hitl'
    conditions = Column(JSON)
    actions = Column(JSON)
    priority = Column(Integer, default=0)
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Enums
class TaskStatus(Enum):
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    HITL_REQUIRED = "hitl_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ModelTier(Enum):
    TIER1_FRONTIER = "tier1_frontier"
    TIER2_BALANCED = "tier2_balanced"
    SPECIALIZED_SLM = "specialized_slm"

class AgentCapability(Enum):
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    FILE_SYSTEM = "file_system"
    WEB_SEARCH = "web_search"
    DATABASE = "database"
    VISION = "vision"
    ROBOTICS = "robotics"
    BIO_STRUCTURE = "bio_structure"

# Data classes
@dataclass
class ModelConfig:
    name: str
    tier: ModelTier
    provider: str  # 'openai', 'anthropic', 'google', 'aws_bedrock', 'local'
    endpoint: Optional[str] = None
    capabilities: List[AgentCapability] = field(default_factory=list)
    context_window: int = 8192
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    max_rpm: int = 60
    timeout_seconds: int = 30
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {"max_attempts": 3})

@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    description: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    required_capabilities: List[AgentCapability] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    idempotency_key: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class TaskPlan:
    task_id: str
    steps: List[Dict[str, Any]]
    estimated_cost_usd: float
    estimated_duration_seconds: int
    required_models: List[str]
    hitl_gates: List[str] = field(default_factory=list)

@dataclass
class ExecutionResult:
    task_id: str
    status: TaskStatus
    output_data: Optional[Dict[str, Any]] = None
    artifacts: List[str] = field(default_factory=list)
    cost_usd: float = 0.0
    tokens_used: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None
    duration_seconds: float = 0.0

# Model Registry
class ModelRegistry:
    """Central registry for all available models and their configurations"""
    
    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize with default model configurations"""
        
        # Tier 1 Orchestrator Models
        self.register(ModelConfig(
            name="gpt-5",
            tier=ModelTier.TIER1_FRONTIER,
            provider="openai",
            capabilities=[AgentCapability.CODE_GENERATION, AgentCapability.CODE_REVIEW],
            context_window=200000,
            cost_per_1k_input=0.15,
            cost_per_1k_output=0.60,
            max_rpm=10
        ))
        
        self.register(ModelConfig(
            name="claude-4.1",
            tier=ModelTier.TIER1_FRONTIER,
            provider="anthropic",
            capabilities=[AgentCapability.CODE_GENERATION, AgentCapability.CODE_REVIEW],
            context_window=200000,
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.075,
            max_rpm=20
        ))
        
        self.register(ModelConfig(
            name="gemini-2.5-pro",
            tier=ModelTier.TIER1_FRONTIER,
            provider="google",
            capabilities=[AgentCapability.VISION, AgentCapability.CODE_GENERATION],
            context_window=1000000,
            cost_per_1k_input=0.00125,
            cost_per_1k_output=0.00375,
            max_rpm=30
        ))
        
        # Tier 2 Balanced Models
        self.register(ModelConfig(
            name="o4-mini",
            tier=ModelTier.TIER2_BALANCED,
            provider="openai",
            capabilities=[AgentCapability.CODE_GENERATION],
            context_window=128000,
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
            max_rpm=100
        ))
        
        self.register(ModelConfig(
            name="deepseek-v3",
            tier=ModelTier.TIER2_BALANCED,
            provider="deepseek",
            endpoint="https://api.deepseek.com/v1",
            capabilities=[AgentCapability.CODE_GENERATION],
            context_window=128000,
            cost_per_1k_input=0.00014,
            cost_per_1k_output=0.00028,
            max_rpm=120
        ))
        
        # Specialized SLMs
        self.register(ModelConfig(
            name="yi-coder-9b",
            tier=ModelTier.SPECIALIZED_SLM,
            provider="local",
            capabilities=[AgentCapability.CODE_GENERATION],
            context_window=128000,
            cost_per_1k_input=0.00001,
            cost_per_1k_output=0.00002,
            max_rpm=500
        ))
        
        self.register(ModelConfig(
            name="deepseek-coder-6.7b",
            tier=ModelTier.SPECIALIZED_SLM,
            provider="local",
            capabilities=[AgentCapability.CODE_GENERATION],
            context_window=16000,
            cost_per_1k_input=0.00001,
            cost_per_1k_output=0.00002,
            max_rpm=500
        ))
    
    def register(self, model: ModelConfig):
        """Register a new model configuration"""
        self.models[model.name] = model
    
    def get(self, name: str) -> Optional[ModelConfig]:
        """Get model configuration by name"""
        return self.models.get(name)
    
    def find_by_capability(self, capability: AgentCapability, tier: Optional[ModelTier] = None) -> List[ModelConfig]:
        """Find models with specific capability and optional tier filter"""
        models = [m for m in self.models.values() if capability in m.capabilities]
        if tier:
            models = [m for m in models if m.tier == tier]
        return sorted(models, key=lambda m: m.cost_per_1k_input)

# Model Router
class ModelRouter:
    """Intelligent routing of tasks to appropriate models based on capabilities, cost, and performance"""
    
    def __init__(self, registry: ModelRegistry, redis_client: aioredis.Redis):
        self.registry = registry
        self.redis = redis_client
        self.performance_window = 3600  # 1 hour window for performance metrics
        
    async def select_orchestrator(self, task: Task) -> ModelConfig:
        """Select the best orchestrator model for the task"""
        
        # Check task complexity hints
        complexity = task.metadata.get("complexity", "medium")
        
        if complexity == "high" or len(task.required_capabilities) > 3:
            # Use Tier 1 for complex tasks
            candidates = [m for m in self.registry.models.values() if m.tier == ModelTier.TIER1_FRONTIER]
        else:
            # Use Tier 2 for simpler tasks
            candidates = [m for m in self.registry.models.values() if m.tier == ModelTier.TIER2_BALANCED]
        
        # Score candidates based on recent performance
        scores = await self._score_models(candidates, task)
        
        if not scores:
            # Fallback to default
            return self.registry.get("o4-mini") or candidates[0]
        
        return scores[0][0]
    
    async def select_agent(self, capability: AgentCapability, context: Dict[str, Any]) -> ModelConfig:
        """Select the best agent model for a specific capability"""
        
        candidates = self.registry.find_by_capability(capability)
        
        if not candidates:
            raise ValueError(f"No models found with capability {capability}")
        
        # Prefer SLMs for simple tasks
        if context.get("prefer_local", False):
            slms = [m for m in candidates if m.tier == ModelTier.SPECIALIZED_SLM]
            if slms:
                return slms[0]
        
        # Score based on cost and performance
        scores = await self._score_models(candidates, context)
        
        return scores[0][0] if scores else candidates[0]
    
    async def _score_models(self, models: List[ModelConfig], context: Any) -> List[Tuple[ModelConfig, float]]:
        """Score models based on historical performance and cost"""
        
        scored = []
        for model in models:
            # Get performance metrics from Redis
            success_rate = await self._get_success_rate(model.name)
            avg_latency = await self._get_avg_latency(model.name)
            
            # Calculate composite score (lower is better)
            cost_weight = 0.4
            performance_weight = 0.4
            latency_weight = 0.2
            
            cost_score = model.cost_per_1k_input + model.cost_per_1k_output
            performance_score = 1.0 - success_rate if success_rate else 0.5
            latency_score = avg_latency / 30.0 if avg_latency else 0.5
            
            score = (cost_score * cost_weight + 
                    performance_score * performance_weight + 
                    latency_score * latency_weight)
            
            scored.append((model, score))
        
        return sorted(scored, key=lambda x: x[1])
    
    async def _get_success_rate(self, model_name: str) -> float:
        """Get recent success rate for a model"""
        key = f"model:metrics:{model_name}:success"
        data = await self.redis.get(key)
        return float(data) if data else 0.95
    
    async def _get_avg_latency(self, model_name: str) -> float:
        """Get recent average latency for a model"""
        key = f"model:metrics:{model_name}:latency"
        data = await self.redis.get(key)
        return float(data) if data else 5.0

# Policy Engine
class PolicyEngine:
    """Enforces policies for access control, rate limiting, cost management, and HITL gates"""
    
    def __init__(self, db_session: AsyncSession, redis_client: aioredis.Redis):
        self.db = db_session
        self.redis = redis_client
        self.rules_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    async def evaluate(self, action: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Evaluate if an action is allowed based on policies"""
        
        # Load applicable rules
        rules = await self._get_rules(action)
        
        # Evaluate each rule in priority order
        for rule in sorted(rules, key=lambda r: r.priority, reverse=True):
            if not rule.enabled:
                continue
                
            result = await self._evaluate_rule(rule, context)
            
            if result == "deny":
                return False, f"Denied by policy: {rule.name}"
            elif result == "hitl":
                return False, f"HITL required by policy: {rule.name}"
        
        return True, None
    
    async def check_rate_limit(self, tenant_id: str, resource: str) -> bool:
        """Check if rate limit is exceeded"""
        
        key = f"rate_limit:{tenant_id}:{resource}"
        current = await self.redis.incr(key)
        
        if current == 1:
            await self.redis.expire(key, 60)  # 1 minute window
        
        limit = await self._get_rate_limit(tenant_id, resource)
        
        return current <= limit
    
    async def check_cost_budget(self, tenant_id: str, estimated_cost: float) -> bool:
        """Check if cost budget allows the operation"""
        
        # Get current spend
        month_key = f"cost:{tenant_id}:{datetime.utcnow().strftime('%Y-%m')}"
        current_spend = float(await self.redis.get(month_key) or 0)
        
        # Get budget limit
        budget = await self._get_budget(tenant_id)
        
        return (current_spend + estimated_cost) <= budget
    
    async def require_hitl(self, operation: str, context: Dict[str, Any]) -> bool:
        """Check if human-in-the-loop approval is required"""
        
        hitl_rules = await self._get_rules(f"hitl:{operation}")
        
        for rule in hitl_rules:
            if await self._evaluate_rule(rule, context) == "require":
                return True
        
        return False
    
    async def _get_rules(self, action: str) -> List[PolicyRule]:
        """Get applicable policy rules for an action"""
        
        cache_key = f"rules:{action}"
        
        if cache_key in self.rules_cache:
            cached, timestamp = self.rules_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached
        
        # Query from database
        result = await self.db.execute(
            "SELECT * FROM policy_rules WHERE rule_type LIKE :action AND enabled = true",
            {"action": f"%{action}%"}
        )
        rules = result.fetchall()
        
        # Cache the result
        self.rules_cache[cache_key] = (rules, time.time())
        
        return rules
    
    async def _evaluate_rule(self, rule: PolicyRule, context: Dict[str, Any]) -> str:
        """Evaluate a single policy rule"""
        
        conditions = rule.conditions or {}
        
        # Simple condition evaluation (can be extended with a DSL)
        for key, expected in conditions.items():
            actual = context.get(key)
            
            if isinstance(expected, dict) and "operator" in expected:
                op = expected["operator"]
                value = expected["value"]
                
                if op == ">" and not (actual > value):
                    return "allow"
                elif op == "<" and not (actual < value):
                    return "allow"
                elif op == "in" and actual not in value:
                    return "allow"
                elif op == "==" and actual != value:
                    return "allow"
            elif actual != expected:
                return "allow"
        
        # All conditions met, return the action
        return rule.actions.get("action", "allow")
    
    async def _get_rate_limit(self, tenant_id: str, resource: str) -> int:
        """Get rate limit for tenant and resource"""
        key = f"config:rate_limit:{tenant_id}:{resource}"
        limit = await self.redis.get(key)
        return int(limit) if limit else 100  # Default 100 requests per minute
    
    async def _get_budget(self, tenant_id: str) -> float:
        """Get monthly budget for tenant"""
        key = f"config:budget:{tenant_id}"
        budget = await self.redis.get(key)
        return float(budget) if budget else 1000.0  # Default $1000/month

# Main Orchestrator
class Orchestrator:
    """Main orchestrator that coordinates the entire multi-agent workflow"""
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        model_router: ModelRouter,
        policy_engine: PolicyEngine,
        db_session: AsyncSession,
        redis_client: aioredis.Redis,
        s3_client: Any,
        sqs_client: Any
    ):
        self.registry = model_registry
        self.router = model_router
        self.policy = policy_engine
        self.db = db_session
        self.redis = redis_client
        self.s3 = s3_client
        self.sqs = sqs_client
        
        # Metrics
        self.task_counter = Counter('tasks_total', 'Total tasks processed', ['status'])
        self.task_duration = Histogram('task_duration_seconds', 'Task duration', ['model'])
        self.active_tasks = Gauge('active_tasks', 'Number of active tasks')
        
        # Circuit breakers for each model provider
        self.circuit_breakers = {}
        
        # Initialize tracer
        self.tracer = trace.get_tracer(__name__)
        
    async def process_task(self, task: Task) -> ExecutionResult:
        """Main entry point for processing a task"""
        
        start_time = time.time()
        self.active_tasks.inc()
        
        with self.tracer.start_as_current_span("process_task") as span:
            span.set_attribute("task.id", task.id)
            span.set_attribute("task.tenant_id", task.tenant_id)
            
            try:
                # Check policies
                allowed, reason = await self.policy.evaluate("task.execute", asdict(task))
                if not allowed:
                    return ExecutionResult(
                        task_id=task.id,
                        status=TaskStatus.FAILED,
                        error=reason
                    )
                
                # Check rate limit
                if not await self.policy.check_rate_limit(task.tenant_id, "tasks"):
                    return ExecutionResult(
                        task_id=task.id,
                        status=TaskStatus.FAILED,
                        error="Rate limit exceeded"
                    )
                
                # Create execution plan
                plan = await self._create_plan(task)
                
                # Check cost budget
                if not await self.policy.check_cost_budget(task.tenant_id, plan.estimated_cost_usd):
                    return ExecutionResult(
                        task_id=task.id,
                        status=TaskStatus.FAILED,
                        error="Cost budget exceeded"
                    )
                
                # Check for HITL requirements
                if plan.hitl_gates:
                    approval = await self._request_hitl_approval(task, plan)
                    if not approval:
                        return ExecutionResult(
                            task_id=task.id,
                            status=TaskStatus.HITL_REQUIRED,
                            error="Awaiting human approval"
                        )
                
                # Execute the plan
                result = await self._execute_plan(task, plan)
                
                # Record metrics
                self.task_counter.labels(status=result.status.value).inc()
                self.task_duration.labels(model=plan.required_models[0]).observe(time.time() - start_time)
                
                # Update cost tracking
                await self._update_cost(task.tenant_id, result.cost_usd)
                
                # Save execution record
                await self._save_execution(task, plan, result)
                
                return result
                
            except Exception as e:
                logger.error("Task execution failed", task_id=task.id, error=str(e))
                return ExecutionResult(
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    error=str(e)
                )
            finally:
                self.active_tasks.dec()
    
    async def _create_plan(self, task: Task) -> TaskPlan:
        """Create an execution plan for the task"""
        
        # Select orchestrator model
        orchestrator = await self.router.select_orchestrator(task)
        
        # Generate plan using the selected model
        prompt = self._build_planning_prompt(task)
        
        response = await self._call_model(orchestrator, prompt, task.metadata)
        
        # Parse the response into a structured plan
        plan_data = self._parse_plan_response(response)
        
        return TaskPlan(
            task_id=task.id,
            steps=plan_data["steps"],
            estimated_cost_usd=plan_data.get("estimated_cost", 0.0),
            estimated_duration_seconds=plan_data.get("estimated_duration", 60),
            required_models=plan_data.get("models", [orchestrator.name]),
            hitl_gates=plan_data.get("hitl_gates", [])
        )
    
    async def _execute_plan(self, task: Task, plan: TaskPlan) -> ExecutionResult:
        """Execute the task plan step by step"""
        
        total_cost = 0.0
        total_tokens = {}
        artifacts = []
        results = []
        
        for step in plan.steps:
            with self.tracer.start_as_current_span(f"execute_step_{step['id']}") as span:
                span.set_attribute("step.type", step["type"])
                
                # Select agent for this step
                capability = AgentCapability[step["capability"].upper()]
                agent = await self.router.select_agent(capability, step)
                
                # Execute the step
                step_result = await self._execute_step(task, step, agent)
                
                # Aggregate results
                results.append(step_result)
                total_cost += step_result.get("cost", 0.0)
                
                tokens = step_result.get("tokens", {})
                for key, value in tokens.items():
                    total_tokens[key] = total_tokens.get(key, 0) + value
                
                if step_result.get("artifacts"):
                    artifacts.extend(step_result["artifacts"])
                
                # Check for failures
                if step_result.get("status") == "failed":
                    return ExecutionResult(
                        task_id=task.id,
                        status=TaskStatus.FAILED,
                        output_data={"partial_results": results},
                        artifacts=artifacts,
                        cost_usd=total_cost,
                        tokens_used=total_tokens,
                        error=step_result.get("error")
                    )
        
        # Synthesize final results
        output = await self._synthesize_results(results, task.metadata)
        
        return ExecutionResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            output_data=output,
            artifacts=artifacts,
            cost_usd=total_cost,
            tokens_used=total_tokens
        )
    
    async def _execute_step(self, task: Task, step: Dict[str, Any], agent: ModelConfig) -> Dict[str, Any]:
        """Execute a single step of the plan"""
        
        # Prepare the execution context
        context = {
            "task_id": task.id,
            "step_id": step["id"],
            "input": step.get("input", {}),
            "tools": step.get("tools", []),
            "sandbox": step.get("sandbox", False)
        }
        
        # Call the agent
        prompt = self._build_agent_prompt(step, task.input_data)
        
        try:
            response = await self._call_model(agent, prompt, context)
            
            # Process agent output
            output = self._parse_agent_response(response, step["type"])
            
            # Save artifacts if any
            artifacts = []
            if output.get("code"):
                artifact_id = await self._save_artifact(
                    task.id,
                    step["id"],
                    output["code"],
                    "code"
                )
                artifacts.append(artifact_id)
            
            return {
                "status": "completed",
                "output": output,
                "artifacts": artifacts,
                "cost": self._calculate_cost(agent, response),
                "tokens": self._count_tokens(response)
            }
            
        except Exception as e:
            logger.error("Step execution failed", step_id=step["id"], error=str(e))
            return {
                "status": "failed",
                "error": str(e)
            }
    
    @circuit(failure_threshold=5, recovery_timeout=60)
    async def _call_model(self, model: ModelConfig, prompt: str, context: Dict[str, Any]) -> str:
        """Call a model with circuit breaker protection"""
        
        # Get or create circuit breaker for this provider
        if model.provider not in self.circuit_breakers:
            self.circuit_breakers[model.provider] = circuit(
                failure_threshold=5,
                recovery_timeout=60,
                expected_exception=Exception
            )
        
        breaker = self.circuit_breakers[model.provider]
        
        @breaker
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        async def _make_call():
            if model.provider == "openai":
                client = AsyncOpenAI()
                response = await client.chat.completions.create(
                    model=model.name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=context.get("temperature", 0.7),
                    max_tokens=context.get("max_tokens", 4096)
                )
                return response.choices[0].message.content
                
            elif model.provider == "anthropic":
                client = anthropic.AsyncAnthropic()
                response = await client.messages.create(
                    model=model.name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=context.get("max_tokens", 4096)
                )
                return response.content[0].text
                
            elif model.provider == "google":
                genai.configure()
                model_instance = genai.GenerativeModel(model.name)
                response = await model_instance.generate_content_async(prompt)
                return response.text
                
            elif model.provider == "local":
                # Call local model endpoint
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        model.endpoint or "http://localhost:8000/v1/completions",
                        json={
                            "model": model.name,
                            "prompt": prompt,
                            "max_tokens": context.get("max_tokens", 4096)
                        },
                        timeout=model.timeout_seconds
                    )
                    response.raise_for_status()
                    return response.json()["choices"][0]["text"]
            
            else:
                raise ValueError(f"Unknown provider: {model.provider}")
        
        return await _make_call()
    
    def _build_planning_prompt(self, task: Task) -> str:
        """Build the planning prompt for the orchestrator"""
        
        return f"""You are an expert AI orchestrator. Create a detailed execution plan for the following task:

Task ID: {task.id}
Description: {task.description}
Input Data: {json.dumps(task.input_data, indent=2)}
Required Capabilities: {', '.join([c.value for c in task.required_capabilities])}
Constraints: {json.dumps(task.constraints, indent=2)}

Generate a JSON execution plan with the following structure:
{{
    "steps": [
        {{
            "id": "step_1",
            "type": "code_generation|file_operation|web_search|etc",
            "capability": "capability_name",
            "description": "what this step does",
            "input": {{}},
            "tools": ["tool1", "tool2"],
            "depends_on": [],
            "sandbox": true/false
        }}
    ],
    "estimated_cost": 0.0,
    "estimated_duration": 60,
    "models": ["model1", "model2"],
    "hitl_gates": ["gate1", "gate2"]
}}

Consider:
1. Optimal sequencing of steps
2. Parallelization opportunities
3. Cost optimization
4. Error handling requirements
5. Security considerations
"""
    
    def _build_agent_prompt(self, step: Dict[str, Any], input_data: Dict[str, Any]) -> str:
        """Build the prompt for an agent"""
        
        return f"""You are a specialized AI agent with expertise in {step['capability']}.

Task: {step['description']}
Input: {json.dumps(step.get('input', {}), indent=2)}
Context: {json.dumps(input_data, indent=2)}
Available Tools: {', '.join(step.get('tools', []))}

Execute this task and return the result in a structured format.
Include any code, data, or artifacts generated.
"""
    
    def _parse_plan_response(self, response: str) -> Dict[str, Any]:
        """Parse the orchestrator's planning response"""
        
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback to a simple plan
        return {
            "steps": [
                {
                    "id": "step_1",
                    "type": "generic",
                    "capability": "code_generation",
                    "description": "Execute task",
                    "input": {},
                    "tools": [],
                    "depends_on": [],
                    "sandbox": True
                }
            ],
            "estimated_cost": 0.1,
            "estimated_duration": 60,
            "models": ["o4-mini"],
            "hitl_gates": []
        }
    
    def _parse_agent_response(self, response: str, step_type: str) -> Dict[str, Any]:
        """Parse an agent's response based on step type"""
        
        result = {"raw_response": response}
        
        if step_type == "code_generation":
            # Extract code blocks
            import re
            code_blocks = re.findall(r'```(\w+)?\n(.*?)```', response, re.DOTALL)
            if code_blocks:
                result["code"] = code_blocks[0][1]
                result["language"] = code_blocks[0][0] or "python"
        
        elif step_type == "file_operation":
            # Extract file operations
            result["operations"] = []
            # Parse file operations from response
        
        return result
    
    async def _save_artifact(self, task_id: str, step_id: str, content: str, artifact_type: str) -> str:
        """Save an artifact to S3"""
        
        artifact_id = str(uuid.uuid4())
        key = f"artifacts/{task_id}/{step_id}/{artifact_id}"
        
        # Calculate content hash for integrity
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Save to S3
        await self.s3.put_object(
            Bucket="agentic-ai-artifact-store",
            Key=key,
            Body=content.encode(),
            Metadata={
                "task_id": task_id,
                "step_id": step_id,
                "type": artifact_type,
                "hash": content_hash,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return artifact_id
    
    async def _request_hitl_approval(self, task: Task, plan: TaskPlan) -> bool:
        """Request human-in-the-loop approval"""
        
        # Create approval request
        approval_id = str(uuid.uuid4())
        
        await self.redis.setex(
            f"hitl:{approval_id}",
            3600,  # 1 hour TTL
            json.dumps({
                "task_id": task.id,
                "plan": asdict(plan),
                "requested_at": datetime.utcnow().isoformat()
            })
        )
        
        # Send notification (could be email, Slack, etc.)
        # For now, just log it
        logger.info("HITL approval requested", approval_id=approval_id, task_id=task.id)
        
        # Wait for approval (in production, this would be async)
        return False
    
    async def _synthesize_results(self, results: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize step results into final output"""
        
        return {
            "status": "completed",
            "steps_completed": len(results),
            "results": results,
            "metadata": metadata,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _update_cost(self, tenant_id: str, cost: float):
        """Update cost tracking for tenant"""
        
        month_key = f"cost:{tenant_id}:{datetime.utcnow().strftime('%Y-%m')}"
        await self.redis.incrbyfloat(month_key, cost)
    
    async def _save_execution(self, task: Task, plan: TaskPlan, result: ExecutionResult):
        """Save execution record to database"""
        
        execution = TaskExecution(
            id=task.id,
            trace_id=task.trace_id,
            tenant_id=task.tenant_id,
            status=result.status.value,
            orchestrator_model=plan.required_models[0] if plan.required_models else None,
            agent_models=plan.required_models[1:] if len(plan.required_models) > 1 else [],
            task_plan=asdict(plan),
            input_data=task.input_data,
            output_data=result.output_data,
            artifacts=result.artifacts,
            cost_usd=result.cost_usd,
            tokens_used=result.tokens_used,
            completed_at=datetime.utcnow(),
            error_message=result.error,
            metadata=task.metadata
        )
        
        self.db.add(execution)
        await self.db.commit()
    
    def _calculate_cost(self, model: ModelConfig, response: str) -> float:
        """Calculate cost for a model call"""
        
        # Rough token estimation (would use tiktoken in production)
        input_tokens = len(response.split()) * 1.3
        output_tokens = len(response.split()) * 1.3
        
        input_cost = (input_tokens / 1000) * model.cost_per_1k_input
        output_cost = (output_tokens / 1000) * model.cost_per_1k_output
        
        return input_cost + output_cost
    
    def _count_tokens(self, response: str) -> Dict[str, int]:
        """Count tokens in response"""
        
        # Rough estimation
        tokens = len(response.split()) * 1.3
        
        return {
            "input": int(tokens),
            "output": int(tokens)
        }

# Initialize everything
async def create_orchestrator(config: Dict[str, Any]) -> Orchestrator:
    """Factory function to create and initialize the orchestrator"""
    
    # Initialize database
    engine = create_async_engine(config["database_url"])
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    # Initialize Redis
    redis = await aioredis.create_redis_pool(config["redis_url"])
    
    # Initialize AWS clients
    session = aioboto3.Session()
    async with session.client('s3') as s3_client:
        async with session.client('sqs') as sqs_client:
            
            # Create components
            registry = ModelRegistry()
            router = ModelRouter(registry, redis)
            
            async with async_session() as db_session:
                policy = PolicyEngine(db_session, redis)
                
                orchestrator = Orchestrator(
                    model_registry=registry,
                    model_router=router,
                    policy_engine=policy,
                    db_session=db_session,
                    redis_client=redis,
                    s3_client=s3_client,
                    sqs_client=sqs_client
                )
                
                return orchestrator
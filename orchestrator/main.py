# orchestrator/main.py
"""
Main orchestrator that coordinates the entire agentic AI system,
integrating all control plane components.
"""

import asyncio
import json
import uuid
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
import networkx as nx

from opentelemetry import trace
import aioredis

# Import our components
from policy.engine import PolicyEngine, PolicyContext, DataClassification
from execution.runner import ExecutionRunner, TaskDefinition, DeliverySemantics
from execution.artifact_lineage import ArtifactStore, RunManifest
from egress_proxy import EgressProxy, EgressRequest
from observability.telemetry import ObservabilityHub, EventSeverity, CostCategory

tracer = trace.get_tracer(__name__)


class TaskStatus(Enum):
    """Status of orchestrated tasks"""
    PENDING = "pending"
    PLANNING = "planning"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentCapability(Enum):
    """Capabilities that agents can have"""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    WEB_SEARCH = "web_search"
    DATABASE_QUERY = "database_query"
    FILE_MANIPULATION = "file_manipulation"
    API_CALL = "api_call"
    VISION_PROCESSING = "vision_processing"
    LANGUAGE_TRANSLATION = "language_translation"
    MATH_COMPUTATION = "math_computation"
    ROBOTICS_CONTROL = "robotics_control"
    PROTEIN_FOLDING = "protein_folding"


@dataclass
class TaskPlan:
    """Execution plan for a task"""
    task_id: str
    steps: List['PlanStep']
    dependencies: nx.DiGraph
    estimated_cost_usd: float
    estimated_duration_seconds: float
    required_capabilities: Set[AgentCapability]
    required_approvals: List[str]
    
    
@dataclass
class PlanStep:
    """Individual step in a task plan"""
    step_id: str
    agent_type: str
    model: str
    action: str
    input_data: Dict[str, Any]
    output_schema: Dict[str, Any]
    timeout_seconds: int
    retry_policy: Dict[str, Any]
    dependencies: List[str]  # Step IDs this depends on
    cost_estimate_usd: float


@dataclass
class OrchestratorTask:
    """Task being orchestrated"""
    task_id: str
    tenant_id: str
    user_id: str
    description: str
    input_data: Dict[str, Any]
    status: TaskStatus
    plan: Optional[TaskPlan] = None
    run_manifest_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    total_cost_usd: float = 0.0


class ControlPlaneOrchestrator:
    """
    Main orchestrator that coordinates all control plane components
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tasks: Dict[str, OrchestratorTask] = {}
        
        # Initialize components
        self.policy_engine = PolicyEngine(config["policy"])
        self.execution_runner = ExecutionRunner(config["execution"])
        self.artifact_store = ArtifactStore(config["artifacts"])
        self.egress_proxy = EgressProxy(config["egress"])
        self.observability = ObservabilityHub(config["observability"])
        
        # Load routing configuration
        with open(config["router_config_path"], 'r') as f:
            self.router_config = yaml.safe_load(f)
        
        # Component registry
        self.capability_registry = CapabilityRegistry()
        self.model_router = ModelRouter(self.router_config)
        self.prompt_registry = PromptRegistry()
        self.feature_flags = FeatureFlags()
        
        # State management
        self.redis_client = None
        self.task_queue = asyncio.Queue()
        
    async def initialize(self):
        """Initialize all components"""
        await self.execution_runner.initialize()
        await self.artifact_store.initialize()
        
        # Initialize Redis for state management
        self.redis_client = await aioredis.create_redis_pool(
            self.config["redis_url"]
        )
        
        # Start background workers
        asyncio.create_task(self._task_processor())
        asyncio.create_task(self._health_monitor())
        asyncio.create_task(self._cost_monitor())
    
    @tracer.start_as_current_span("orchestrate_task")
    async def orchestrate(
        self,
        description: str,
        tenant_id: str,
        user_id: str,
        input_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Main entry point for task orchestration
        
        Returns:
            Task ID
        """
        span = trace.get_current_span()
        
        # Create task
        task_id = str(uuid.uuid4())
        task = OrchestratorTask(
            task_id=task_id,
            tenant_id=tenant_id,
            user_id=user_id,
            description=description,
            input_data=input_data,
            status=TaskStatus.PENDING
        )
        
        self.tasks[task_id] = task
        
        span.set_attributes({
            "task.id": task_id,
            "task.tenant_id": tenant_id,
            "task.user_id": user_id
        })
        
        try:
            # Step 1: Policy check
            policy_result = await self._check_policies(task)
            if policy_result["action"] == "deny":
                task.status = TaskStatus.FAILED
                task.error = f"Policy denied: {policy_result['reason']}"
                return task_id
            
            if policy_result["action"] == "require_hitl":
                task.status = TaskStatus.PENDING
                # Queue for HITL approval
                await self._request_hitl_approval(task, policy_result)
                return task_id
            
            # Step 2: Create execution plan
            task.status = TaskStatus.PLANNING
            plan = await self._create_plan(task)
            task.plan = plan
            
            # Step 3: Cost and budget check
            budget_ok = await self.observability.cost_tracker.check_budget(
                tenant_id,
                plan.estimated_cost_usd
            )
            if not budget_ok:
                task.status = TaskStatus.FAILED
                task.error = "Budget exceeded"
                await self._emit_event(
                    "budget_exceeded",
                    EventSeverity.WARNING,
                    task
                )
                return task_id
            
            # Step 4: Create run manifest for reproducibility
            run_manifest_id = await self.artifact_store.create_run_manifest(
                task_id=task_id,
                tenant_id=tenant_id,
                user_id=user_id,
                models_used=[
                    {"name": step.model, "version": "latest"}
                    for step in plan.steps
                ],
                settings={
                    "timeout": max(s.timeout_seconds for s in plan.steps),
                    "max_retries": 3
                }
            )
            task.run_manifest_id = run_manifest_id
            
            # Step 5: Queue for execution
            task.status = TaskStatus.APPROVED
            await self.task_queue.put(task)
            
            span.set_attributes({
                "task.status": task.status.value,
                "task.estimated_cost_usd": plan.estimated_cost_usd,
                "task.steps_count": len(plan.steps)
            })
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            
            await self._emit_event(
                "orchestration_error",
                EventSeverity.ERROR,
                task,
                {"error": str(e)}
            )
        
        return task_id
    
    async def _create_plan(self, task: OrchestratorTask) -> TaskPlan:
        """
        Create execution plan for task using LLM orchestrator
        """
        # Analyze task to determine required capabilities
        capabilities = await self._analyze_capabilities(task.description)
        
        # Build dependency graph
        dep_graph = nx.DiGraph()
        steps = []
        
        # Example planning logic (simplified)
        # In production, this would use the orchestrator LLM to create plan
        
        if AgentCapability.CODE_GENERATION in capabilities:
            # Code generation workflow
            steps.extend([
                PlanStep(
                    step_id="analyze",
                    agent_type="analyzer",
                    model=self.model_router.select_model("code_analysis", task.tenant_id),
                    action="analyze_requirements",
                    input_data=task.input_data,
                    output_schema={"requirements": "list"},
                    timeout_seconds=30,
                    retry_policy={"max_retries": 2},
                    dependencies=[],
                    cost_estimate_usd=0.05
                ),
                PlanStep(
                    step_id="generate",
                    agent_type="code_generator",
                    model=self.model_router.select_model("code_generation", task.tenant_id),
                    action="generate_code",
                    input_data={},  # Will be filled from previous step
                    output_schema={"code": "string"},
                    timeout_seconds=60,
                    retry_policy={"max_retries": 3},
                    dependencies=["analyze"],
                    cost_estimate_usd=0.20
                ),
                PlanStep(
                    step_id="review",
                    agent_type="code_reviewer",
                    model=self.model_router.select_model("code_review", task.tenant_id),
                    action="review_code",
                    input_data={},
                    output_schema={"review": "object"},
                    timeout_seconds=30,
                    retry_policy={"max_retries": 2},
                    dependencies=["generate"],
                    cost_estimate_usd=0.10
                )
            ])
            
            # Build dependency graph
            dep_graph.add_edge("analyze", "generate")
            dep_graph.add_edge("generate", "review")
        
        elif AgentCapability.ROBOTICS_CONTROL in capabilities:
            # Robotics workflow with safety checks
            steps.extend([
                PlanStep(
                    step_id="perception",
                    agent_type="vision",
                    model="pi-0-small",
                    action="perceive_environment",
                    input_data=task.input_data,
                    output_schema={"scene": "object"},
                    timeout_seconds=5,
                    retry_policy={"max_retries": 1},
                    dependencies=[],
                    cost_estimate_usd=0.01
                ),
                PlanStep(
                    step_id="plan_action",
                    agent_type="vla_planner",
                    model="rt-2",
                    action="plan_action",
                    input_data={},
                    output_schema={"action_sequence": "list"},
                    timeout_seconds=10,
                    retry_policy={"max_retries": 2},
                    dependencies=["perception"],
                    cost_estimate_usd=0.05
                ),
                PlanStep(
                    step_id="safety_check",
                    agent_type="safety",
                    model="gpt-5",
                    action="verify_safety",
                    input_data={},
                    output_schema={"safe": "boolean", "risks": "list"},
                    timeout_seconds=5,
                    retry_policy={"max_retries": 1},
                    dependencies=["plan_action"],
                    cost_estimate_usd=0.15
                )
            ])
            
            dep_graph.add_edge("perception", "plan_action")
            dep_graph.add_edge("plan_action", "safety_check")
        
        # Calculate total estimates
        total_cost = sum(s.cost_estimate_usd for s in steps)
        total_duration = self._calculate_critical_path_duration(dep_graph, steps)
        
        return TaskPlan(
            task_id=task.task_id,
            steps=steps,
            dependencies=dep_graph,
            estimated_cost_usd=total_cost,
            estimated_duration_seconds=total_duration,
            required_capabilities=capabilities,
            required_approvals=self._determine_approvals(capabilities)
        )
    
    async def _execute_plan(self, task: OrchestratorTask):
        """
        Execute task plan step by step
        """
        plan = task.plan
        task.status = TaskStatus.EXECUTING
        
        # Track execution state
        step_results = {}
        
        try:
            # Execute in topological order respecting dependencies
            for step_id in nx.topological_sort(plan.dependencies):
                step = next(s for s in plan.steps if s.step_id == step_id)
                
                # Prepare input from dependencies
                step_input = step.input_data.copy()
                for dep_id in step.dependencies:
                    if dep_id in step_results:
                        step_input[f"input_from_{dep_id}"] = step_results[dep_id]
                
                # Execute step
                result = await self._execute_step(task, step, step_input)
                step_results[step.step_id] = result
                
                # Track cost
                await self.observability.cost_tracker.track_cost(
                    trace_id=task.task_id,
                    tenant_id=task.tenant_id,
                    user_id=task.user_id,
                    category=CostCategory.AGENT_EXECUTION,
                    resource_type=step.model,
                    quantity=1,
                    metadata={"step": step.step_id}
                )
                
            # Store final result
            task.result = step_results
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            
            # Finalize run manifest
            if task.run_manifest_id:
                await self.artifact_store.finalize_run_manifest(
                    task.run_manifest_id,
                    output_artifacts=[],  # Would include generated artifacts
                    metrics={
                        "execution_time_seconds": (
                            task.completed_at - task.created_at
                        ).total_seconds(),
                        "total_cost_usd": task.total_cost_usd
                    }
                )
                
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            
            await self._emit_event(
                "execution_failed",
                EventSeverity.ERROR,
                task,
                {"error": str(e), "failed_at_step": step.step_id if 'step' in locals() else "unknown"}
            )
    
    async def _execute_step(
        self,
        task: OrchestratorTask,
        step: PlanStep,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single plan step
        """
        # Create execution task
        exec_task = TaskDefinition(
            task_id=f"{task.task_id}_{step.step_id}",
            idempotency_key=f"{task.task_id}_{step.step_id}",
            tenant_id=task.tenant_id,
            user_id=task.user_id,
            task_type=step.agent_type,
            payload={
                "action": step.action,
                "model": step.model,
                "input": input_data
            },
            timeout_seconds=step.timeout_seconds,
            max_retries=step.retry_policy.get("max_retries", 3),
            delivery_semantics=DeliverySemantics.EXACTLY_ONCE
        )
        
        # Execute through runner
        result = await self.execution_runner.execute(exec_task)
        
        if result.state.value == "committed":
            # Extract actual result from execution
            return {"status": "success", "output": result.metrics}
        else:
            raise Exception(f"Step {step.step_id} failed: {result.error}")
    
    async def _check_policies(self, task: OrchestratorTask) -> Dict[str, Any]:
        """
        Check policies for task execution
        """
        # Determine data classification
        data_tags = set()
        if "personal_data" in str(task.input_data):
            data_tags.add(DataClassification.PII)
        
        context = PolicyContext(
            user_id=task.user_id,
            tenant_id=task.tenant_id,
            roles=await self._get_user_roles(task.user_id),
            action="task.execute",
            resource=f"task:{task.task_id}",
            data_tags=data_tags,
            trace_id=task.task_id,
            cost_usd=0.0  # Will be updated with estimate
        )
        
        result = await self.policy_engine.evaluate(context)
        
        return {
            "action": result.action.value,
            "reason": " ".join(result.reasons),
            "hitl_requirement": result.hitl_requirement
        }
    
    async def _task_processor(self):
        """
        Background task processor
        """
        while True:
            try:
                task = await self.task_queue.get()
                
                span = tracer.start_as_current_span("process_task")
                span.set_attributes({
                    "task.id": task.task_id,
                    "task.status": task.status.value
                })
                
                await self._execute_plan(task)
                
            except Exception as e:
                print(f"Task processor error: {e}")
                
            await asyncio.sleep(0.1)
    
    async def _health_monitor(self):
        """
        Monitor health of all components
        """
        while True:
            try:
                # Check circuit breakers
                for service, breaker in self.execution_runner.circuit_breakers.items():
                    self.observability.circuit_breaker_status.labels(
                        service=service
                    ).set(1 if breaker.state == "open" else 0)
                
                # Check queue depth
                queue_depth = self.task_queue.qsize()
                if queue_depth > 100:
                    await self._emit_event(
                        "queue_depth_high",
                        EventSeverity.WARNING,
                        None,
                        {"depth": queue_depth}
                    )
                
            except Exception as e:
                print(f"Health monitor error: {e}")
                
            await asyncio.sleep(10)
    
    async def _cost_monitor(self):
        """
        Monitor costs and enforce limits
        """
        while True:
            try:
                # Check for cost anomalies
                for tenant_id in self.observability.cost_tracker.tenant_usage.keys():
                    usage = self.observability.cost_tracker.tenant_usage[tenant_id]
                    
                    # Check if approaching budget
                    if tenant_id in self.observability.cost_tracker.tenant_budgets:
                        budget = self.observability.cost_tracker.tenant_budgets[tenant_id]
                        if usage > budget * 0.8:  # 80% threshold
                            await self._emit_event(
                                "budget_warning",
                                EventSeverity.WARNING,
                                None,
                                {
                                    "tenant_id": tenant_id,
                                    "usage": usage,
                                    "budget": budget,
                                    "percentage": (usage / budget) * 100
                                }
                            )
                
            except Exception as e:
                print(f"Cost monitor error: {e}")
                
            await asyncio.sleep(60)
    
    async def _analyze_capabilities(self, description: str) -> Set[AgentCapability]:
        """
        Analyze task description to determine required capabilities
        """
        # Simplified capability detection
        # In production, use NLP or LLM to analyze
        
        capabilities = set()
        
        keywords = {
            AgentCapability.CODE_GENERATION: ["code", "program", "function", "class"],
            AgentCapability.WEB_SEARCH: ["search", "find", "lookup", "google"],
            AgentCapability.DATABASE_QUERY: ["database", "sql", "query", "table"],
            AgentCapability.ROBOTICS_CONTROL: ["robot", "move", "grasp", "navigate"],
            AgentCapability.VISION_PROCESSING: ["image", "video", "detect", "recognize"]
        }
        
        description_lower = description.lower()
        for capability, words in keywords.items():
            if any(word in description_lower for word in words):
                capabilities.add(capability)
        
        # Default to code generation if nothing detected
        if not capabilities:
            capabilities.add(AgentCapability.CODE_GENERATION)
        
        return capabilities
    
    def _calculate_critical_path_duration(
        self,
        graph: nx.DiGraph,
        steps: List[PlanStep]
    ) -> float:
        """
        Calculate critical path duration through dependency graph
        """
        if not graph.nodes():
            return 0
        
        # Find longest path
        step_map = {s.step_id: s for s in steps}
        
        # Topological sort
        topo_order = list(nx.topological_sort(graph))
        
        # Calculate earliest start times
        earliest = {}
        for node in topo_order:
            pred_times = [
                earliest[pred] + step_map[pred].timeout_seconds
                for pred in graph.predecessors(node)
            ]
            earliest[node] = max(pred_times) if pred_times else 0
        
        # Total duration is max earliest time + duration of last task
        if topo_order:
            last_node = max(earliest.keys(), key=lambda k: earliest[k])
            return earliest[last_node] + step_map[last_node].timeout_seconds
        
        return 0
    
    def _determine_approvals(self, capabilities: Set[AgentCapability]) -> List[str]:
        """
        Determine required approvals based on capabilities
        """
        approvals = []
        
        if AgentCapability.ROBOTICS_CONTROL in capabilities:
            approvals.append("safety_officer")
            
        if AgentCapability.DATABASE_QUERY in capabilities:
            approvals.append("dba")
            
        return approvals
    
    async def _request_hitl_approval(
        self,
        task: OrchestratorTask,
        policy_result: Dict[str, Any]
    ):
        """
        Request human-in-the-loop approval
        """
        approval_id = str(uuid.uuid4())
        
        # Store approval request
        await self.redis_client.setex(
            f"approval:{approval_id}",
            600,  # 10 minute TTL
            json.dumps({
                "task_id": task.task_id,
                "tenant_id": task.tenant_id,
                "user_id": task.user_id,
                "policy_result": policy_result,
                "requested_at": datetime.utcnow().isoformat()
            })
        )
        
        await self._emit_event(
            "hitl_approval_requested",
            EventSeverity.INFO,
            task,
            {"approval_id": approval_id}
        )
    
    async def _get_user_roles(self, user_id: str) -> Set[str]:
        """
        Get user roles from identity provider
        """
        # In production, query identity provider
        return {"developer", "agent_runtime"}
    
    async def _emit_event(
        self,
        event_type: str,
        severity: EventSeverity,
        task: Optional[OrchestratorTask],
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Emit telemetry event
        """
        attrs = attributes or {}
        if task:
            attrs.update({
                "task_id": task.task_id,
                "tenant_id": task.tenant_id,
                "user_id": task.user_id
            })
        
        await self.observability.emit_event(
            event_type=event_type,
            severity=severity,
            attributes=attrs,
            tenant_id=task.tenant_id if task else "",
            user_id=task.user_id if task else ""
        )


class CapabilityRegistry:
    """Registry of agent capabilities"""
    
    def __init__(self):
        self.capabilities = {}
        self._register_default_capabilities()
    
    def _register_default_capabilities(self):
        """Register default agent capabilities"""
        self.capabilities = {
            "code_generator": [AgentCapability.CODE_GENERATION],
            "code_reviewer": [AgentCapability.CODE_REVIEW],
            "web_searcher": [AgentCapability.WEB_SEARCH],
            "database_agent": [AgentCapability.DATABASE_QUERY],
            "file_agent": [AgentCapability.FILE_MANIPULATION],
            "vision_agent": [AgentCapability.VISION_PROCESSING],
            "robotics_agent": [AgentCapability.ROBOTICS_CONTROL]
        }
    
    def get_agents_for_capability(
        self,
        capability: AgentCapability
    ) -> List[str]:
        """Get agents that have a capability"""
        return [
            agent for agent, caps in self.capabilities.items()
            if capability in caps
        ]


class ModelRouter:
    """Routes tasks to appropriate models based on policy"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_history = {}
    
    def select_model(
        self,
        task_type: str,
        tenant_id: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> str:
        """Select best model for task"""
        # Get tenant tier
        tier = "tier1_frontier" if "premium" in tenant_id else "tier2_balanced"
        
        # Simple routing based on task type
        routing_map = {
            "code_generation": {
                "tier1_frontier": "gpt-5",
                "tier2_balanced": "deepseek-coder-6.7b"
            },
            "code_analysis": {
                "tier1_frontier": "claude-4.1",
                "tier2_balanced": "o4-mini"
            },
            "code_review": {
                "tier1_frontier": "claude-4.1",
                "tier2_balanced": "yi-coder-9b"
            }
        }
        
        return routing_map.get(task_type, {}).get(tier, "o4-mini")


class PromptRegistry:
    """Registry of prompts and configurations"""
    
    def __init__(self):
        self.prompts = {}
        self._load_default_prompts()
    
    def _load_default_prompts(self):
        """Load default prompt templates"""
        self.prompts = {
            "code_generation": """
                Generate {language} code for the following requirements:
                {requirements}
                
                Constraints:
                - Follow best practices
                - Include error handling
                - Add documentation
            """,
            "code_review": """
                Review the following code for:
                - Correctness
                - Performance
                - Security
                - Best practices
                
                Code:
                {code}
            """
        }
    
    def get_prompt(self, task_type: str, variables: Dict[str, Any]) -> str:
        """Get formatted prompt for task"""
        template = self.prompts.get(task_type, "")
        return template.format(**variables)


class FeatureFlags:
    """Feature flag management"""
    
    def __init__(self):
        self.flags = {
            "enable_robotics": False,
            "enable_bioengineering": False,
            "enable_advanced_routing": True,
            "enable_cost_optimization": True,
            "enable_hitl_for_production": True
        }
    
    def is_enabled(self, flag: str) -> bool:
        """Check if feature flag is enabled"""
        return self.flags.get(flag, False)


# Example usage
async def main():
    """Example orchestrator usage"""
    config = {
        "redis_url": "redis://localhost",
        "router_config_path": "config/router.yaml",
        "policy": load_policy_config(),
        "execution": {
            "redis_url": "redis://localhost",
            "database_url": "postgresql://localhost/agentic"
        },
        "artifacts": {
            "s3_bucket": "agentic-artifacts"
        },
        "egress": {
            "encryption_key": "test_key",
            "global_allowed_domains": ["*.wikipedia.org"]
        },
        "observability": {
            "otlp_endpoint": "localhost:4317"
        }
    }
    
    orchestrator = ControlPlaneOrchestrator(config)
    await orchestrator.initialize()
    
    # Create a task
    task_id = await orchestrator.orchestrate(
        description="Generate a Python function to calculate fibonacci numbers",
        tenant_id="tenant-123",
        user_id="user-456",
        input_data={"max_n": 100}
    )
    
    print(f"Task created: {task_id}")
    
    # Check task status
    task = orchestrator.tasks[task_id]
    print(f"Task status: {task.status.value}")
    
    # Wait for completion
    await asyncio.sleep(5)
    print(f"Final status: {task.status.value}")


if __name__ == "__main__":
    asyncio.run(main()
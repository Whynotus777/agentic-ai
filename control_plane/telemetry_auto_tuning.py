# control_plane/telemetry_auto_tuning.py
"""
Telemetry-driven auto-tuning system for optimizing routing, resource allocation,
and model selection based on real-time performance metrics.
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics

from opentelemetry import trace, metrics
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)


class TuningTarget(Enum):
    """Targets for auto-tuning"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    COST = "cost"
    ACCURACY = "accuracy"
    BALANCED = "balanced"


class ResourceType(Enum):
    """Types of resources to tune"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    REPLICAS = "replicas"
    BATCH_SIZE = "batch_size"
    CACHE_SIZE = "cache_size"


@dataclass
class ModelPerformance:
    """Performance metrics for a model"""
    model_id: str
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    error_rate: float
    cost_per_request: float
    accuracy_score: float
    resource_utilization: Dict[str, float]
    timestamp: datetime
    request_count: int


@dataclass
class RoutingWeight:
    """Routing weight for a model"""
    model_id: str
    weight: float
    min_weight: float = 0.0
    max_weight: float = 1.0
    adjustment_rate: float = 0.1
    last_adjusted: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TuningPolicy:
    """Policy for auto-tuning"""
    target: TuningTarget
    slo_latency_ms: float = 100.0
    slo_error_rate: float = 0.01
    min_accuracy: float = 0.9
    max_cost_per_request: float = 1.0
    evaluation_window: int = 300  # seconds
    adjustment_cooldown: int = 60  # seconds
    max_adjustment_per_cycle: float = 0.2


@dataclass
class ResourceAllocation:
    """Resource allocation for a service"""
    service_id: str
    cpu_cores: float
    memory_gb: float
    gpu_count: int
    replicas: int
    batch_size: int
    cache_size_mb: int
    last_updated: datetime


class TelemetryAutoTuner:
    """
    Auto-tuning system based on telemetry data
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.routing_weights: Dict[str, RoutingWeight] = {}
        self.resource_allocations: Dict[str, ResourceAllocation] = {}
        self.tuning_policies: Dict[str, TuningPolicy] = {}
        self.performance_history: List[ModelPerformance] = []
        self.tuning_decisions: List[Dict[str, Any]] = []
        self._init_default_policies()
        self._init_metrics()
        
    def _init_default_policies(self):
        """Initialize default tuning policies"""
        self.tuning_policies = {
            "default": TuningPolicy(
                target=TuningTarget.BALANCED,
                slo_latency_ms=100.0,
                slo_error_rate=0.01,
                min_accuracy=0.9,
                max_cost_per_request=1.0
            ),
            "latency_optimized": TuningPolicy(
                target=TuningTarget.LATENCY,
                slo_latency_ms=50.0,
                slo_error_rate=0.02,
                min_accuracy=0.85,
                max_cost_per_request=2.0
            ),
            "cost_optimized": TuningPolicy(
                target=TuningTarget.COST,
                slo_latency_ms=200.0,
                slo_error_rate=0.02,
                min_accuracy=0.85,
                max_cost_per_request=0.5
            ),
            "accuracy_optimized": TuningPolicy(
                target=TuningTarget.ACCURACY,
                slo_latency_ms=500.0,
                slo_error_rate=0.01,
                min_accuracy=0.95,
                max_cost_per_request=5.0
            )
        }
    
    def _init_metrics(self):
        """Initialize OpenTelemetry metrics"""
        self.latency_histogram = meter.create_histogram(
            "model_latency",
            description="Model inference latency",
            unit="ms"
        )
        
        self.throughput_counter = meter.create_counter(
            "model_throughput",
            description="Model request throughput"
        )
        
        self.error_counter = meter.create_counter(
            "model_errors",
            description="Model error count"
        )
        
        self.cost_counter = meter.create_counter(
            "model_cost",
            description="Model inference cost",
            unit="usd"
        )
    
    @tracer.start_as_current_span("ingest_telemetry")
    async def ingest_telemetry(
        self,
        model_id: str,
        latency_ms: float,
        success: bool,
        cost: float,
        accuracy: Optional[float] = None,
        resource_usage: Optional[Dict[str, float]] = None
    ):
        """Ingest telemetry data from model execution"""
        span = trace.get_current_span()
        
        # Record metrics
        self.latency_histogram.record(
            latency_ms,
            attributes={"model_id": model_id, "success": str(success)}
        )
        
        if not success:
            self.error_counter.add(1, attributes={"model_id": model_id})
        
        self.throughput_counter.add(1, attributes={"model_id": model_id})
        self.cost_counter.add(cost, attributes={"model_id": model_id})
        
        # Store in buffer
        metric_data = {
            "timestamp": datetime.utcnow(),
            "latency_ms": latency_ms,
            "success": success,
            "cost": cost,
            "accuracy": accuracy,
            "resource_usage": resource_usage or {}
        }
        
        self.model_metrics[model_id].append(metric_data)
        
        span.set_attributes({
            "model_id": model_id,
            "latency_ms": latency_ms,
            "success": success,
            "cost": cost
        })
    
    async def analyze_performance(
        self,
        model_id: str,
        window_seconds: int = 300
    ) -> ModelPerformance:
        """Analyze model performance over time window"""
        if model_id not in self.model_metrics:
            raise ValueError(f"No metrics for model {model_id}")
        
        metrics = self.model_metrics[model_id]
        cutoff_time = datetime.utcnow() - timedelta(seconds=window_seconds)
        
        # Filter metrics within window
        window_metrics = [
            m for m in metrics
            if m["timestamp"] > cutoff_time
        ]
        
        if not window_metrics:
            raise ValueError(f"No recent metrics for model {model_id}")
        
        # Calculate statistics
        latencies = [m["latency_ms"] for m in window_metrics if m["success"]]
        
        if not latencies:
            # All requests failed
            return ModelPerformance(
                model_id=model_id,
                avg_latency_ms=float('inf'),
                p50_latency_ms=float('inf'),
                p95_latency_ms=float('inf'),
                p99_latency_ms=float('inf'),
                throughput_rps=0,
                error_rate=1.0,
                cost_per_request=0,
                accuracy_score=0,
                resource_utilization={},
                timestamp=datetime.utcnow(),
                request_count=len(window_metrics)
            )
        
        # Calculate percentiles
        latencies.sort()
        p50_idx = int(len(latencies) * 0.5)
        p95_idx = int(len(latencies) * 0.95)
        p99_idx = int(len(latencies) * 0.99)
        
        # Calculate metrics
        performance = ModelPerformance(
            model_id=model_id,
            avg_latency_ms=statistics.mean(latencies),
            p50_latency_ms=latencies[p50_idx],
            p95_latency_ms=latencies[p95_idx] if p95_idx < len(latencies) else latencies[-1],
            p99_latency_ms=latencies[p99_idx] if p99_idx < len(latencies) else latencies[-1],
            throughput_rps=len(window_metrics) / window_seconds,
            error_rate=sum(1 for m in window_metrics if not m["success"]) / len(window_metrics),
            cost_per_request=statistics.mean([m["cost"] for m in window_metrics]),
            accuracy_score=statistics.mean([m["accuracy"] for m in window_metrics if m["accuracy"] is not None]) if any(m["accuracy"] for m in window_metrics) else 0.9,
            resource_utilization=self._aggregate_resource_usage(window_metrics),
            timestamp=datetime.utcnow(),
            request_count=len(window_metrics)
        )
        
        # Store in history
        self.performance_history.append(performance)
        
        return performance
    
    def _aggregate_resource_usage(
        self,
        metrics: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Aggregate resource usage from metrics"""
        resource_totals = defaultdict(list)
        
        for m in metrics:
            if m.get("resource_usage"):
                for resource, value in m["resource_usage"].items():
                    resource_totals[resource].append(value)
        
        return {
            resource: statistics.mean(values)
            for resource, values in resource_totals.items()
        }
    
    @tracer.start_as_current_span("auto_tune_routing")
    async def auto_tune_routing(
        self,
        models: List[str],
        policy_name: str = "default"
    ) -> Dict[str, float]:
        """
        Auto-tune routing weights based on performance
        
        Returns:
            Updated routing weights
        """
        span = trace.get_current_span()
        
        policy = self.tuning_policies.get(policy_name, self.tuning_policies["default"])
        
        # Analyze performance for each model
        performances = {}
        for model_id in models:
            try:
                perf = await self.analyze_performance(
                    model_id,
                    window_seconds=policy.evaluation_window
                )
                performances[model_id] = perf
            except ValueError:
                # No metrics available, use default
                if model_id not in self.routing_weights:
                    self.routing_weights[model_id] = RoutingWeight(
                        model_id=model_id,
                        weight=1.0 / len(models)
                    )
        
        if not performances:
            # No performance data available
            return {m: 1.0 / len(models) for m in models}
        
        # Calculate scores based on policy target
        scores = {}
        for model_id, perf in performances.items():
            score = self._calculate_performance_score(perf, policy)
            scores[model_id] = score
        
        # Adjust routing weights
        total_score = sum(scores.values())
        
        for model_id in models:
            if model_id not in self.routing_weights:
                self.routing_weights[model_id] = RoutingWeight(
                    model_id=model_id,
                    weight=1.0 / len(models)
                )
            
            weight_obj = self.routing_weights[model_id]
            
            # Check cooldown
            if (datetime.utcnow() - weight_obj.last_adjusted).seconds < policy.adjustment_cooldown:
                continue
            
            # Calculate new weight
            if model_id in scores and total_score > 0:
                target_weight = scores[model_id] / total_score
                
                # Apply gradual adjustment
                current_weight = weight_obj.weight
                adjustment = (target_weight - current_weight) * weight_obj.adjustment_rate
                
                # Limit adjustment size
                adjustment = max(-policy.max_adjustment_per_cycle, 
                                min(adjustment, policy.max_adjustment_per_cycle))
                
                new_weight = current_weight + adjustment
                
                # Apply bounds
                new_weight = max(weight_obj.min_weight, min(new_weight, weight_obj.max_weight))
                
                weight_obj.weight = new_weight
                weight_obj.last_adjusted = datetime.utcnow()
        
        # Normalize weights
        total_weight = sum(w.weight for w in self.routing_weights.values())
        
        result = {}
        for model_id in models:
            if model_id in self.routing_weights:
                normalized_weight = self.routing_weights[model_id].weight / total_weight
                result[model_id] = normalized_weight
            else:
                result[model_id] = 1.0 / len(models)
        
        # Record tuning decision
        self.tuning_decisions.append({
            "timestamp": datetime.utcnow(),
            "type": "routing",
            "policy": policy_name,
            "weights": result,
            "performances": {
                m: {
                    "latency": p.avg_latency_ms,
                    "error_rate": p.error_rate,
                    "cost": p.cost_per_request
                }
                for m, p in performances.items()
            }
        })
        
        span.set_attributes({
            "tuning.type": "routing",
            "tuning.models": ",".join(models),
            "tuning.policy": policy_name
        })
        
        return result
    
    def _calculate_performance_score(
        self,
        performance: ModelPerformance,
        policy: TuningPolicy
    ) -> float:
        """Calculate performance score based on policy"""
        score = 1.0
        
        # Latency component
        if performance.avg_latency_ms > policy.slo_latency_ms:
            latency_penalty = (performance.avg_latency_ms - policy.slo_latency_ms) / policy.slo_latency_ms
            score *= max(0.1, 1.0 - latency_penalty)
        
        # Error rate component
        if performance.error_rate > policy.slo_error_rate:
            error_penalty = (performance.error_rate - policy.slo_error_rate) / policy.slo_error_rate
            score *= max(0.1, 1.0 - error_penalty)
        
        # Cost component
        if performance.cost_per_request > policy.max_cost_per_request:
            cost_penalty = (performance.cost_per_request - policy.max_cost_per_request) / policy.max_cost_per_request
            score *= max(0.1, 1.0 - cost_penalty)
        
        # Accuracy component
        if performance.accuracy_score < policy.min_accuracy:
            accuracy_penalty = (policy.min_accuracy - performance.accuracy_score) / policy.min_accuracy
            score *= max(0.1, 1.0 - accuracy_penalty)
        
        # Apply target-specific weighting
        if policy.target == TuningTarget.LATENCY:
            score *= (policy.slo_latency_ms / max(1, performance.avg_latency_ms))
        elif policy.target == TuningTarget.COST:
            score *= (policy.max_cost_per_request / max(0.01, performance.cost_per_request))
        elif policy.target == TuningTarget.ACCURACY:
            score *= performance.accuracy_score
        elif policy.target == TuningTarget.THROUGHPUT:
            score *= performance.throughput_rps
        
        return score
    
    @tracer.start_as_current_span("auto_scale_resources")
    async def auto_scale_resources(
        self,
        service_id: str,
        performance: ModelPerformance,
        policy_name: str = "default"
    ) -> ResourceAllocation:
        """
        Auto-scale resources based on performance
        
        Returns:
            Updated resource allocation
        """
        span = trace.get_current_span()
        
        policy = self.tuning_policies.get(policy_name, self.tuning_policies["default"])
        
        # Get current allocation
        if service_id not in self.resource_allocations:
            self.resource_allocations[service_id] = ResourceAllocation(
                service_id=service_id,
                cpu_cores=2.0,
                memory_gb=4.0,
                gpu_count=0,
                replicas=1,
                batch_size=1,
                cache_size_mb=512,
                last_updated=datetime.utcnow()
            )
        
        allocation = self.resource_allocations[service_id]
        
        # Check if scaling is needed
        scaling_decisions = []
        
        # Scale based on latency
        if performance.p95_latency_ms > policy.slo_latency_ms * 1.5:
            # Need more resources
            if performance.resource_utilization.get("cpu", 0) > 0.8:
                allocation.cpu_cores = min(32, allocation.cpu_cores * 1.5)
                scaling_decisions.append("increase_cpu")
            
            if performance.resource_utilization.get("memory", 0) > 0.8:
                allocation.memory_gb = min(128, allocation.memory_gb * 1.5)
                scaling_decisions.append("increase_memory")
            
            allocation.replicas = min(20, allocation.replicas + 1)
            scaling_decisions.append("increase_replicas")
            
        elif performance.p95_latency_ms < policy.slo_latency_ms * 0.5:
            # Can reduce resources
            if performance.resource_utilization.get("cpu", 1.0) < 0.3:
                allocation.cpu_cores = max(0.5, allocation.cpu_cores * 0.75)
                scaling_decisions.append("decrease_cpu")
            
            if performance.resource_utilization.get("memory", 1.0) < 0.3:
                allocation.memory_gb = max(1.0, allocation.memory_gb * 0.75)
                scaling_decisions.append("decrease_memory")
            
            if allocation.replicas > 1:
                allocation.replicas = max(1, allocation.replicas - 1)
                scaling_decisions.append("decrease_replicas")
        
        # Scale based on error rate
        if performance.error_rate > policy.slo_error_rate * 2:
            # Increase resources to reduce errors
            allocation.memory_gb = min(128, allocation.memory_gb * 1.2)
            allocation.batch_size = max(1, allocation.batch_size // 2)
            scaling_decisions.append("reduce_batch_size")
        
        # Optimize batch size for throughput
        if policy.target == TuningTarget.THROUGHPUT:
            if performance.throughput_rps < 100 and allocation.batch_size < 32:
                allocation.batch_size = min(32, allocation.batch_size * 2)
                scaling_decisions.append("increase_batch_size")
        
        allocation.last_updated = datetime.utcnow()
        
        # Record scaling decision
        if scaling_decisions:
            self.tuning_decisions.append({
                "timestamp": datetime.utcnow(),
                "type": "scaling",
                "service_id": service_id,
                "decisions": scaling_decisions,
                "allocation": {
                    "cpu_cores": allocation.cpu_cores,
                    "memory_gb": allocation.memory_gb,
                    "replicas": allocation.replicas,
                    "batch_size": allocation.batch_size
                }
            })
        
        span.set_attributes({
            "tuning.type": "scaling",
            "tuning.service_id": service_id,
            "tuning.decisions": ",".join(scaling_decisions)
        })
        
        return allocation
    
    async def recommend_model_changes(
        self,
        performances: Dict[str, ModelPerformance],
        policy_name: str = "default"
    ) -> List[Dict[str, Any]]:
        """
        Recommend model changes based on performance
        
        Returns:
            List of recommendations
        """
        policy = self.tuning_policies.get(policy_name, self.tuning_policies["default"])
        recommendations = []
        
        for model_id, perf in performances.items():
            # Check for consistent SLO violations
            if perf.error_rate > policy.slo_error_rate * 3:
                recommendations.append({
                    "model_id": model_id,
                    "action": "replace",
                    "reason": f"High error rate: {perf.error_rate:.2%}",
                    "severity": "high"
                })
            
            elif perf.avg_latency_ms > policy.slo_latency_ms * 3:
                recommendations.append({
                    "model_id": model_id,
                    "action": "optimize",
                    "reason": f"High latency: {perf.avg_latency_ms:.0f}ms",
                    "severity": "medium",
                    "suggestions": ["quantization", "distillation", "caching"]
                })
            
            elif perf.cost_per_request > policy.max_cost_per_request * 2:
                recommendations.append({
                    "model_id": model_id,
                    "action": "replace_with_cheaper",
                    "reason": f"High cost: ${perf.cost_per_request:.3f}/request",
                    "severity": "low",
                    "alternatives": self._find_cheaper_alternatives(model_id)
                })
            
            elif perf.accuracy_score < policy.min_accuracy * 0.8:
                recommendations.append({
                    "model_id": model_id,
                    "action": "retrain",
                    "reason": f"Low accuracy: {perf.accuracy_score:.2%}",
                    "severity": "high"
                })
        
        return recommendations
    
    def _find_cheaper_alternatives(self, model_id: str) -> List[str]:
        """Find cheaper alternative models"""
        # In production, query model registry
        alternatives = {
            "gpt-5": ["gpt-4o", "claude-sonnet"],
            "claude-opus": ["claude-sonnet", "gpt-4o-mini"],
            "gemini-ultra": ["gemini-pro", "llama-70b"]
        }
        
        return alternatives.get(model_id, [])
    
    async def export_tuning_report(self) -> Dict[str, Any]:
        """Export comprehensive tuning report"""
        recent_decisions = self.tuning_decisions[-100:]  # Last 100 decisions
        
        # Calculate statistics
        routing_adjustments = [d for d in recent_decisions if d["type"] == "routing"]
        scaling_adjustments = [d for d in recent_decisions if d["type"] == "scaling"]
        
        # Get current state
        current_weights = {
            model_id: weight.weight
            for model_id, weight in self.routing_weights.items()
        }
        
        current_allocations = {
            service_id: {
                "cpu_cores": alloc.cpu_cores,
                "memory_gb": alloc.memory_gb,
                "replicas": alloc.replicas
            }
            for service_id, alloc in self.resource_allocations.items()
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_decisions": len(self.tuning_decisions),
            "recent_decisions": recent_decisions,
            "routing_adjustments": len(routing_adjustments),
            "scaling_adjustments": len(scaling_adjustments),
            "current_routing_weights": current_weights,
            "current_resource_allocations": current_allocations,
            "performance_summary": self._summarize_performance()
        }
    
    def _summarize_performance(self) -> Dict[str, Any]:
        """Summarize recent performance"""
        if not self.performance_history:
            return {}
        
        recent_perfs = self.performance_history[-100:]  # Last 100 measurements
        
        # Group by model
        by_model = defaultdict(list)
        for perf in recent_perfs:
            by_model[perf.model_id].append(perf)
        
        summary = {}
        for model_id, perfs in by_model.items():
            summary[model_id] = {
                "avg_latency_ms": statistics.mean([p.avg_latency_ms for p in perfs]),
                "avg_error_rate": statistics.mean([p.error_rate for p in perfs]),
                "avg_cost": statistics.mean([p.cost_per_request for p in perfs]),
                "avg_accuracy": statistics.mean([p.accuracy_score for p in perfs]),
                "measurement_count": len(perfs)
            }
        
        return summary
    
    async def continuous_tuning_loop(self):
        """Continuous auto-tuning loop"""
        while True:
            try:
                # Get list of active models
                active_models = list(self.model_metrics.keys())
                
                if active_models:
                    # Auto-tune routing
                    weights = await self.auto_tune_routing(active_models)
                    print(f"Updated routing weights: {weights}")
                    
                    # Auto-scale resources for each model
                    for model_id in active_models:
                        try:
                            perf = await self.analyze_performance(model_id)
                            allocation = await self.auto_scale_resources(model_id, perf)
                            print(f"Updated allocation for {model_id}: {allocation.replicas} replicas")
                        except Exception as e:
                            print(f"Scaling error for {model_id}: {e}")
                
            except Exception as e:
                print(f"Auto-tuning error: {e}")
            
            await asyncio.sleep(60)  # Run every minute


# Example usage
async def main():
    config = {}
    tuner = TelemetryAutoTuner(config)
    
    # Simulate telemetry ingestion
    models = ["gpt-5", "claude-4.1", "gemini-2.5"]
    
    for _ in range(100):
        for model_id in models:
            # Simulate different performance characteristics
            if model_id == "gpt-5":
                latency = np.random.normal(50, 10)
                success = np.random.random() > 0.01
                cost = 0.10
            elif model_id == "claude-4.1":
                latency = np.random.normal(75, 15)
                success = np.random.random() > 0.02
                cost = 0.08
            else:
                latency = np.random.normal(100, 20)
                success = np.random.random() > 0.03
                cost = 0.05
            
            await tuner.ingest_telemetry(
                model_id=model_id,
                latency_ms=max(1, latency),
                success=success,
                cost=cost,
                accuracy=0.9 + np.random.normal(0, 0.05),
                resource_usage={
                    "cpu": np.random.random(),
                    "memory": np.random.random()
                }
            )
    
    # Analyze performance
    for model_id in models:
        perf = await tuner.analyze_performance(model_id)
        print(f"{model_id} - Latency: {perf.avg_latency_ms:.1f}ms, Error: {perf.error_rate:.2%}, Cost: ${perf.cost_per_request:.3f}")
    
    # Auto-tune routing
    weights = await tuner.auto_tune_routing(models)
    print(f"Routing weights: {weights}")
    
    # Get recommendations
    performances = {m: await tuner.analyze_performance(m) for m in models}
    recommendations = await tuner.recommend_model_changes(performances)
    print(f"Recommendations: {json.dumps(recommendations, indent=2)}")
    
    # Export report
    report = await tuner.export_tuning_report()
    print(f"Tuning report: {json.dumps(report, indent=2, default=str)}")


if __name__ == "__main__":
    asyncio.run(main())
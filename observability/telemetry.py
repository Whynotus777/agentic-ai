# observability/telemetry.py
"""
Comprehensive observability layer with cost tracking, sampling policies,
and SIEM integration for the agentic AI system.
"""

import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import statistics

from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc import (
    trace_exporter as otlp_trace,
    metric_exporter as otlp_metric
)
from opentelemetry.sdk.trace import TracerProvider, sampling
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary

tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)


class EventSeverity(Enum):
    """Event severity levels for SIEM"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SECURITY = "SECURITY"


class CostCategory(Enum):
    """Categories for cost tracking"""
    LLM_INFERENCE = "llm_inference"
    AGENT_EXECUTION = "agent_execution"
    STORAGE = "storage"
    NETWORK_EGRESS = "network_egress"
    COMPUTE = "compute"
    MONITORING = "monitoring"


@dataclass
class TelemetryEvent:
    """Structured telemetry event"""
    timestamp: datetime
    trace_id: str
    span_id: str
    tenant_id: str
    user_id: str
    event_type: str
    severity: EventSeverity
    attributes: Dict[str, Any]
    cost_usd: float = 0.0
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    
    
@dataclass
class CostRecord:
    """Record of resource usage and cost"""
    timestamp: datetime
    trace_id: str
    tenant_id: str
    user_id: str
    category: CostCategory
    resource_type: str
    quantity: float
    unit: str
    rate_usd: float
    total_cost_usd: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdaptiveSampler(sampling.Sampler):
    """
    Adaptive sampling that samples 1% normally but 100% on errors/retries
    """
    
    def __init__(self, base_rate: float = 0.01):
        self.base_rate = base_rate
        self.error_keywords = ["error", "retry", "failure", "timeout", "circuit_breaker"]
        
    def should_sample(
        self,
        parent_context,
        trace_id: int,
        name: str,
        kind=None,
        attributes=None,
        links=None
    ) -> sampling.SamplingResult:
        """Determine if span should be sampled"""
        
        # Always sample if error-related
        if attributes:
            for key, value in attributes.items():
                key_str = str(key).lower()
                value_str = str(value).lower()
                
                # Check for error indicators
                if any(keyword in key_str for keyword in self.error_keywords):
                    return sampling.SamplingResult(
                        sampling.Decision.RECORD_AND_SAMPLE,
                        attributes={"sampling.reason": "error_detected"}
                    )
                    
                if "status" in key_str and value_str in ["error", "failed"]:
                    return sampling.SamplingResult(
                        sampling.Decision.RECORD_AND_SAMPLE,
                        attributes={"sampling.reason": "error_status"}
                    )
        
        # Sample based on base rate
        if (trace_id % 100) < (self.base_rate * 100):
            return sampling.SamplingResult(
                sampling.Decision.RECORD_AND_SAMPLE,
                attributes={"sampling.reason": "base_rate"}
            )
            
        return sampling.SamplingResult(
            sampling.Decision.DROP,
            attributes={"sampling.reason": "dropped"}
        )


class CostTracker:
    """
    Tracks costs for all operations with per-tenant budgets
    """
    
    def __init__(self):
        self.cost_records: List[CostRecord] = []
        self.tenant_usage: Dict[str, float] = defaultdict(float)
        self.tenant_budgets: Dict[str, float] = {}
        self.cost_rates = self._load_cost_rates()
        
        # Prometheus metrics for cost tracking
        self.cost_counter = Counter(
            'agentic_ai_cost_usd_total',
            'Total cost in USD',
            ['tenant_id', 'category', 'resource_type']
        )
        
        self.budget_gauge = Gauge(
            'agentic_ai_budget_remaining_usd',
            'Remaining budget in USD',
            ['tenant_id']
        )
    
    def _load_cost_rates(self) -> Dict[str, float]:
        """Load cost rates per resource type"""
        return {
            # LLM rates per 1k tokens
            "gpt-5_input": 0.15,
            "gpt-5_output": 0.45,
            "claude-4.1_input": 0.08,
            "claude-4.1_output": 0.24,
            "o4-mini_input": 0.00015,
            "o4-mini_output": 0.0006,
            
            # Compute rates per hour
            "cpu_hour": 0.05,
            "gpu_a100_hour": 2.50,
            "memory_gb_hour": 0.01,
            
            # Storage rates per GB-month
            "s3_standard": 0.023,
            "s3_glacier": 0.004,
            
            # Network rates per GB
            "egress_gb": 0.09,
            
            # Monitoring per million events
            "telemetry_events": 0.50
        }
    
    async def track_cost(
        self,
        trace_id: str,
        tenant_id: str,
        user_id: str,
        category: CostCategory,
        resource_type: str,
        quantity: float,
        unit: str = "unit",
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Track cost for a resource usage
        
        Returns:
            Cost in USD
        """
        rate = self.cost_rates.get(resource_type, 0.0)
        total_cost = quantity * rate
        
        record = CostRecord(
            timestamp=datetime.utcnow(),
            trace_id=trace_id,
            tenant_id=tenant_id,
            user_id=user_id,
            category=category,
            resource_type=resource_type,
            quantity=quantity,
            unit=unit,
            rate_usd=rate,
            total_cost_usd=total_cost,
            metadata=metadata or {}
        )
        
        self.cost_records.append(record)
        self.tenant_usage[tenant_id] += total_cost
        
        # Update Prometheus metrics
        self.cost_counter.labels(
            tenant_id=tenant_id,
            category=category.value,
            resource_type=resource_type
        ).inc(total_cost)
        
        # Update budget gauge
        if tenant_id in self.tenant_budgets:
            remaining = self.tenant_budgets[tenant_id] - self.tenant_usage[tenant_id]
            self.budget_gauge.labels(tenant_id=tenant_id).set(remaining)
        
        # Emit cost as span attribute
        span = trace.get_current_span()
        if span:
            span.set_attribute("cost.usd", total_cost)
            span.set_attribute("cost.category", category.value)
            span.set_attribute("cost.resource_type", resource_type)
        
        return total_cost
    
    async def check_budget(self, tenant_id: str, estimated_cost: float) -> bool:
        """Check if tenant has budget for estimated cost"""
        if tenant_id not in self.tenant_budgets:
            return True  # No budget limit set
            
        current_usage = self.tenant_usage.get(tenant_id, 0.0)
        budget = self.tenant_budgets[tenant_id]
        
        return (current_usage + estimated_cost) <= budget
    
    async def get_cost_breakdown(
        self,
        tenant_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get cost breakdown for a tenant"""
        if not start_time:
            start_time = datetime.utcnow() - timedelta(days=1)
        if not end_time:
            end_time = datetime.utcnow()
            
        relevant_records = [
            r for r in self.cost_records
            if r.tenant_id == tenant_id
            and start_time <= r.timestamp <= end_time
        ]
        
        breakdown = {
            "tenant_id": tenant_id,
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "total_cost_usd": sum(r.total_cost_usd for r in relevant_records),
            "by_category": {},
            "by_resource": {},
            "by_user": {}
        }
        
        # Group by category
        for record in relevant_records:
            cat = record.category.value
            if cat not in breakdown["by_category"]:
                breakdown["by_category"][cat] = 0.0
            breakdown["by_category"][cat] += record.total_cost_usd
            
            # Group by resource
            res = record.resource_type
            if res not in breakdown["by_resource"]:
                breakdown["by_resource"][res] = 0.0
            breakdown["by_resource"][res] += record.total_cost_usd
            
            # Group by user
            user = record.user_id
            if user not in breakdown["by_user"]:
                breakdown["by_user"][user] = 0.0
            breakdown["by_user"][user] += record.total_cost_usd
        
        return breakdown


class ObservabilityHub:
    """
    Central hub for all observability concerns
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cost_tracker = CostTracker()
        self.events: List[TelemetryEvent] = []
        self.siem_queue = asyncio.Queue()
        self.metrics_buffer = defaultdict(list)
        
        # Initialize OpenTelemetry
        self._init_opentelemetry()
        
        # Initialize Prometheus metrics
        self._init_prometheus_metrics()
        
    def _init_opentelemetry(self):
        """Initialize OpenTelemetry providers"""
        # Resource attributes
        resource = Resource(attributes={
            "service.name": "agentic-ai",
            "service.version": "1.0.0",
            "deployment.environment": self.config.get("environment", "production")
        })
        
        # Trace provider with adaptive sampling
        trace_provider = TracerProvider(
            resource=resource,
            sampler=AdaptiveSampler(base_rate=0.01)
        )
        
        # Add OTLP exporter
        otlp_exporter = otlp_trace.OTLPSpanExporter(
            endpoint=self.config.get("otlp_endpoint", "localhost:4317"),
            insecure=True
        )
        trace_provider.add_span_processor(
            trace.export.BatchSpanProcessor(otlp_exporter)
        )
        
        trace.set_tracer_provider(trace_provider)
        
        # Metrics provider
        metric_provider = MeterProvider(
            resource=resource,
            metric_readers=[
                metrics.export.PeriodicExportingMetricReader(
                    otlp_metric.OTLPMetricExporter(
                        endpoint=self.config.get("otlp_endpoint", "localhost:4317"),
                        insecure=True
                    )
                )
            ]
        )
        metrics.set_meter_provider(metric_provider)
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        self.request_counter = Counter(
            'agentic_ai_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'agentic_ai_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
        )
        
        self.active_tasks = Gauge(
            'agentic_ai_active_tasks',
            'Number of active tasks',
            ['task_type', 'tenant_id']
        )
        
        self.model_latency = Summary(
            'agentic_ai_model_latency_seconds',
            'Model inference latency',
            ['model', 'task_type']
        )
        
        self.error_rate = Counter(
            'agentic_ai_errors_total',
            'Total number of errors',
            ['error_type', 'component']
        )
        
        self.circuit_breaker_status = Gauge(
            'agentic_ai_circuit_breaker_open',
            'Circuit breaker status (1=open, 0=closed)',
            ['service']
        )
    
    @tracer.start_as_current_span("emit_event")
    async def emit_event(
        self,
        event_type: str,
        severity: EventSeverity,
        attributes: Dict[str, Any],
        tenant_id: str = "",
        user_id: str = "",
        cost_usd: float = 0.0
    ):
        """
        Emit a telemetry event with cost tracking
        """
        span = trace.get_current_span()
        span_context = span.get_span_context()
        
        event = TelemetryEvent(
            timestamp=datetime.utcnow(),
            trace_id=format(span_context.trace_id, '032x'),
            span_id=format(span_context.span_id, '016x'),
            tenant_id=tenant_id,
            user_id=user_id,
            event_type=event_type,
            severity=severity,
            attributes=attributes,
            cost_usd=cost_usd
        )
        
        self.events.append(event)
        
        # Add cost to every event
        if cost_usd > 0:
            span.set_attribute("cost_usd", cost_usd)
            
        # Send to SIEM for security events
        if severity in [EventSeverity.SECURITY, EventSeverity.CRITICAL]:
            await self.siem_queue.put(event)
        
        # Add event to span
        span.add_event(
            event_type,
            attributes={
                **attributes,
                "severity": severity.value,
                "cost_usd": cost_usd
            }
        )
        
        # Update error metrics
        if severity == EventSeverity.ERROR:
            self.error_rate.labels(
                error_type=attributes.get("error_type", "unknown"),
                component=attributes.get("component", "unknown")
            ).inc()
    
    async def track_model_inference(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        tenant_id: str,
        user_id: str,
        trace_id: str
    ) -> float:
        """
        Track model inference with cost calculation
        """
        # Calculate cost
        input_cost = await self.cost_tracker.track_cost(
            trace_id=trace_id,
            tenant_id=tenant_id,
            user_id=user_id,
            category=CostCategory.LLM_INFERENCE,
            resource_type=f"{model}_input",
            quantity=input_tokens / 1000,
            unit="1k_tokens",
            metadata={"model": model, "type": "input"}
        )
        
        output_cost = await self.cost_tracker.track_cost(
            trace_id=trace_id,
            tenant_id=tenant_id,
            user_id=user_id,
            category=CostCategory.LLM_INFERENCE,
            resource_type=f"{model}_output",
            quantity=output_tokens / 1000,
            unit="1k_tokens",
            metadata={"model": model, "type": "output"}
        )
        
        total_cost = input_cost + output_cost
        
        # Record latency metric
        self.model_latency.labels(
            model=model,
            task_type="inference"
        ).observe(latency_ms / 1000)
        
        # Emit event with cost
        await self.emit_event(
            event_type="model_inference",
            severity=EventSeverity.INFO,
            attributes={
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms
            },
            tenant_id=tenant_id,
            user_id=user_id,
            cost_usd=total_cost
        )
        
        return total_cost
    
    async def check_anomaly(self, metric_name: str, value: float) -> bool:
        """
        Check if a metric value is anomalous
        """
        # Get historical values
        history = self.metrics_buffer.get(metric_name, [])
        
        if len(history) < 10:
            # Not enough data for anomaly detection
            self.metrics_buffer[metric_name].append(value)
            return False
        
        # Calculate statistics
        mean = statistics.mean(history)
        stdev = statistics.stdev(history)
        
        # Check if value is > 2 standard deviations from mean
        if abs(value - mean) > (2 * stdev):
            await self.emit_event(
                event_type="anomaly_detected",
                severity=EventSeverity.WARNING,
                attributes={
                    "metric": metric_name,
                    "value": value,
                    "mean": mean,
                    "stdev": stdev,
                    "threshold": 2 * stdev
                }
            )
            return True
        
        # Update buffer (sliding window of 100)
        self.metrics_buffer[metric_name].append(value)
        if len(self.metrics_buffer[metric_name]) > 100:
            self.metrics_buffer[metric_name].pop(0)
        
        return False
    
    async def export_to_siem(self):
        """
        Export critical events to SIEM system
        """
        while not self.siem_queue.empty():
            event = await self.siem_queue.get()
            
            # Format for SIEM (CEF format example)
            cef_event = self._format_cef_event(event)
            
            # In production, send to SIEM endpoint
            print(f"SIEM Event: {cef_event}")
            
            # Map to MITRE ATT&CK if applicable
            attack_mapping = self._map_to_mitre_attack(event)
            if attack_mapping:
                print(f"MITRE ATT&CK Mapping: {attack_mapping}")
    
    def _format_cef_event(self, event: TelemetryEvent) -> str:
        """Format event in Common Event Format for SIEM"""
        cef_header = f"CEF:0|Anthropic|AgenticAI|1.0|{event.event_type}|{event.event_type}|{event.severity.value}|"
        
        extensions = []
        for key, value in event.attributes.items():
            extensions.append(f"{key}={value}")
        
        extensions.extend([
            f"rt={int(event.timestamp.timestamp() * 1000)}",
            f"duid={event.user_id}",
            f"dvchost=agentic-ai",
            f"cs1Label=TraceID",
            f"cs1={event.trace_id}",
            f"cs2Label=TenantID",
            f"cs2={event.tenant_id}",
            f"cn1Label=CostUSD",
            f"cn1={event.cost_usd}"
        ])
        
        return cef_header + " ".join(extensions)
    
    def _map_to_mitre_attack(self, event: TelemetryEvent) -> Optional[Dict[str, str]]:
        """Map security events to MITRE ATT&CK framework"""
        attack_mappings = {
            "unauthorized_access": {
                "tactic": "Initial Access",
                "technique": "T1190 - Exploit Public-Facing Application"
            },
            "privilege_escalation": {
                "tactic": "Privilege Escalation",
                "technique": "T1068 - Exploitation for Privilege Escalation"
            },
            "data_exfiltration": {
                "tactic": "Exfiltration",
                "technique": "T1041 - Exfiltration Over C2 Channel"
            },
            "prompt_injection": {
                "tactic": "Defense Evasion",
                "technique": "T1027 - Obfuscated Files or Information"
            }
        }
        
        return attack_mappings.get(event.event_type)
    
    async def generate_metrics_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics report"""
        now = datetime.utcnow()
        
        # Get Prometheus metrics
        metrics_data = {
            "timestamp": now.isoformat(),
            "requests": {
                "total": prometheus_client.generate_latest(self.request_counter),
                "duration_p95": self._get_histogram_percentile(self.request_duration, 0.95)
            },
            "models": {
                "latency_p95": self._get_summary_percentile(self.model_latency, 0.95)
            },
            "errors": {
                "total": prometheus_client.generate_latest(self.error_rate)
            },
            "costs": await self._get_cost_summary()
        }
        
        return metrics_data
    
    def _get_histogram_percentile(self, histogram, percentile: float) -> float:
        """Get percentile from Prometheus histogram"""
        # In production, query Prometheus directly
        return 0.0
    
    def _get_summary_percentile(self, summary, percentile: float) -> float:
        """Get percentile from Prometheus summary"""
        # In production, query Prometheus directly
        return 0.0
    
    async def _get_cost_summary(self) -> Dict[str, float]:
        """Get cost summary across all tenants"""
        total_cost = sum(self.cost_tracker.tenant_usage.values())
        
        return {
            "total_usd": total_cost,
            "by_category": {
                cat.value: sum(
                    r.total_cost_usd for r in self.cost_tracker.cost_records
                    if r.category == cat
                )
                for cat in CostCategory
            }
        }


class RedactionManager:
    """
    Manages PII redaction in telemetry data
    """
    
    def __init__(self):
        self.redaction_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
        }
    
    def redact_pii(self, data: Any) -> Tuple[Any, List[str]]:
        """
        Redact PII from data and return redaction hashes
        """
        import re
        import hashlib
        
        redactions = []
        
        if isinstance(data, str):
            redacted = data
            for pii_type, pattern in self.redaction_patterns.items():
                matches = re.findall(pattern, data)
                for match in matches:
                    hash_val = hashlib.sha256(match.encode()).hexdigest()[:8]
                    redacted = redacted.replace(
                        match,
                        f"[REDACTED_{pii_type.upper()}_{hash_val}]"
                    )
                    redactions.append(f"{pii_type}:{hash_val}")
            return redacted, redactions
            
        elif isinstance(data, dict):
            redacted_dict = {}
            for key, value in data.items():
                redacted_value, value_redactions = self.redact_pii(value)
                redacted_dict[key] = redacted_value
                redactions.extend(value_redactions)
            return redacted_dict, redactions
            
        elif isinstance(data, list):
            redacted_list = []
            for item in data:
                redacted_item, item_redactions = self.redact_pii(item)
                redacted_list.append(redacted_item)
                redactions.extend(item_redactions)
            return redacted_list, redactions
            
        return data, redactions


# Example usage
async def main():
    """Example usage of observability layer"""
    config = {
        "otlp_endpoint": "localhost:4317",
        "environment": "production"
    }
    
    hub = ObservabilityHub(config)
    
    # Track model inference
    cost = await hub.track_model_inference(
        model="gpt-5",
        input_tokens=500,
        output_tokens=1000,
        latency_ms=2500,
        tenant_id="tenant-123",
        user_id="user-456",
        trace_id="trace-abc-123"
    )
    
    print(f"Inference cost: ${cost:.4f}")
    
    # Check budget
    hub.cost_tracker.tenant_budgets["tenant-123"] = 100.0
    can_proceed = await hub.cost_tracker.check_budget("tenant-123", 10.0)
    print(f"Budget check: {can_proceed}")
    
    # Emit security event
    await hub.emit_event(
        event_type="unauthorized_access",
        severity=EventSeverity.SECURITY,
        attributes={
            "ip_address": "192.168.1.100",
            "attempted_resource": "/api/admin",
            "user_agent": "suspicious-bot/1.0"
        },
        tenant_id="tenant-123",
        user_id="unknown"
    )
    
    # Export to SIEM
    await hub.export_to_siem()
    
    # Get cost breakdown
    breakdown = await hub.cost_tracker.get_cost_breakdown("tenant-123")
    print(f"Cost breakdown: {json.dumps(breakdown, indent=2, default=str)}")
    
    # Generate metrics report
    report = await hub.generate_metrics_report()
    print(f"Metrics report: {json.dumps(report, indent=2, default=str)}")


if __name__ == "__main__":
    asyncio.run(main())
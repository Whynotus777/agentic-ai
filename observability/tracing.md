# observability/tracing.md
# EXTENDED: Trace propagation examples across ORCH→QUEUE→TOOLS with cost_usd

## Trace Context Propagation

### Overview
This document demonstrates trace_id propagation across the orchestrator → queue → tools pipeline, including cost_usd attribution at each step.

## Trace Flow Example: ORCH → QUEUE → TOOLS

### 1. Orchestrator Initiates Trace

```python
# orchestrator.py
from opentelemetry import trace, baggage
from opentelemetry.propagate import inject, extract
import json

tracer = trace.get_tracer("orchestrator")

def process_request(request_id, task_spec):
    """Orchestrator starts a trace and enqueues work"""
    
    # Start root span with cost attribution
    with tracer.start_as_current_span("orchestrate_task") as span:
        span.set_attribute("request_id", request_id)
        span.set_attribute("cost_usd", 0.0001)  # Base orchestration cost
        span.set_attribute("service", "orchestrator")
        span.set_attribute("capability", "task_planning")
        span.set_attribute("tenant", task_spec.get("tenant_id"))
        
        # Get trace_id for logging
        trace_id = format(span.get_span_context().trace_id, '032x')
        span.set_attribute("trace_id", trace_id)
        
        # Prepare message for queue
        message = {
            "task_id": f"task_{request_id}",
            "spec": task_spec,
            "timestamp": time.time()
        }
        
        # Inject trace context into message headers
        headers = {}
        inject(headers)  # Injects traceparent and tracestate
        
        # Add to queue with trace context
        queue_client.send_message(
            queue_name="task_queue",
            body=json.dumps(message),
            headers=headers,
            attributes={
                "trace_id": trace_id,
                "cost_usd": "0.0001"
            }
        )
        
        print(f"Task enqueued with trace_id: {trace_id}")
        return trace_id
```

### 2. Queue Processor Continues Trace

```python
# queue_processor.py
from opentelemetry import trace
from opentelemetry.propagate import inject, extract
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer("queue_processor")

def process_queue_message(message, headers):
    """Process message from queue, continuing the trace"""
    
    # Extract parent trace context
    ctx = extract(headers)
    
    # Continue trace from parent context
    with tracer.start_as_current_span(
        "process_queue_message",
        context=ctx
    ) as span:
        # Add queue processing cost
        span.set_attribute("cost_usd", 0.00005)  # Queue processing cost
        span.set_attribute("service", "queue_processor")
        span.set_attribute("capability", "message_routing")
        
        # Parse message
        task = json.loads(message)
        span.set_attribute("task_id", task["task_id"])
        span.set_attribute("tenant", task["spec"].get("tenant_id"))
        
        # Get trace_id for correlation
        trace_id = format(span.get_span_context().trace_id, '032x')
        print(f"Processing task {task['task_id']} in trace {trace_id}")
        
        # Route to appropriate tool
        tool_name = determine_tool(task["spec"])
        span.set_attribute("target_tool", tool_name)
        
        # Prepare tool invocation with trace context
        tool_headers = {}
        inject(tool_headers)
        
        # Call tool with trace propagation
        result = invoke_tool(
            tool_name=tool_name,
            task=task,
            headers=tool_headers
        )
        
        span.set_attribute("tool_result_status", result.get("status"))
        
        # Accumulate cost
        total_cost = 0.00005 + result.get("cost_usd", 0)
        span.set_attribute("accumulated_cost_usd", total_cost)
        
        return result
```

### 3. Tool Execution Completes Trace

```python
# tool_executor.py
from opentelemetry import trace
from opentelemetry.propagate import extract
import time

tracer = trace.get_tracer("tool_executor")

def execute_tool(tool_name, task, headers):
    """Execute tool operation, completing the trace"""
    
    # Extract parent context from headers
    ctx = extract(headers)
    
    # Continue trace
    with tracer.start_as_current_span(
        f"execute_{tool_name}",
        context=ctx
    ) as span:
        start_time = time.time()
        
        # Set tool-specific attributes
        span.set_attribute("service", "tool_executor")
        span.set_attribute("capability", tool_name)
        span.set_attribute("tenant", task["spec"].get("tenant_id"))
        span.set_attribute("tool.name", tool_name)
        span.set_attribute("tool.version", "1.0.0")
        
        # Calculate tool-specific cost
        tool_costs = {
            "code_generator": 0.001,
            "test_runner": 0.0005,
            "deployment_tool": 0.002,
            "llm_inference": 0.005,
            "robot_control": 0.01
        }
        
        base_cost = tool_costs.get(tool_name, 0.0001)
        span.set_attribute("cost_usd", base_cost)
        
        try:
            # Execute actual tool logic
            result = tool_implementations[tool_name](task)
            
            # Adjust cost based on execution time
            execution_time = time.time() - start_time
            final_cost = base_cost * (1 + execution_time * 0.01)
            span.set_attribute("cost_usd", round(final_cost, 6))
            span.set_attribute("execution_time_ms", execution_time * 1000)
            
            # Get complete trace_id
            trace_id = format(span.get_span_context().trace_id, '032x')
            
            result["trace_id"] = trace_id
            result["cost_usd"] = final_cost
            
            print(f"Tool {tool_name} completed in trace {trace_id}, cost: ${final_cost}")
            
            return result
            
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("error", True)
            raise
```

## Sample Trace with cost_usd

### Complete Trace Example

```json
{
  "trace_id": "7d3f4e8a9b2c1d5e6f7a8b9c0d1e2f3a",
  "spans": [
    {
      "span_id": "1a2b3c4d5e6f7890",
      "parent_span_id": null,
      "operation_name": "orchestrate_task",
      "service": "orchestrator",
      "capability": "task_planning",
      "tenant": "tenant_123",
      "start_time": "2024-01-15T10:00:00.000Z",
      "end_time": "2024-01-15T10:00:00.100Z",
      "attributes": {
        "request_id": "req_456",
        "cost_usd": 0.0001,
        "trace_id": "7d3f4e8a9b2c1d5e6f7a8b9c0d1e2f3a"
      }
    },
    {
      "span_id": "2b3c4d5e6f789012",
      "parent_span_id": "1a2b3c4d5e6f7890",
      "operation_name": "process_queue_message",
      "service": "queue_processor",
      "capability": "message_routing",
      "tenant": "tenant_123",
      "start_time": "2024-01-15T10:00:00.100Z",
      "end_time": "2024-01-15T10:00:00.150Z",
      "attributes": {
        "task_id": "task_req_456",
        "target_tool": "code_generator",
        "cost_usd": 0.00005,
        "accumulated_cost_usd": 0.00115
      }
    },
    {
      "span_id": "3c4d5e6f78901234",
      "parent_span_id": "2b3c4d5e6f789012",
      "operation_name": "execute_code_generator",
      "service": "tool_executor",
      "capability": "code_generator",
      "tenant": "tenant_123",
      "start_time": "2024-01-15T10:00:00.150Z",
      "end_time": "2024-01-15T10:00:00.250Z",
      "attributes": {
        "tool.name": "code_generator",
        "tool.version": "1.0.0",
        "execution_time_ms": 100,
        "cost_usd": 0.0011,
        "trace_id": "7d3f4e8a9b2c1d5e6f7a8b9c0d1e2f3a"
      }
    }
  ],
  "total_cost_usd": 0.00125,
  "total_duration_ms": 250
}
```

## Trace Context Headers

### W3C Trace Context Format
```
traceparent: 00-7d3f4e8a9b2c1d5e6f7a8b9c0d1e2f3a-1a2b3c4d5e6f7890-01
tracestate: cost_usd=0.0001,tenant=tenant_123,capability=task_planning
```

### Propagation Through Different Transports

#### HTTP Headers
```http
GET /api/v1/tool/execute HTTP/1.1
traceparent: 00-7d3f4e8a9b2c1d5e6f7a8b9c0d1e2f3a-2b3c4d5e6f789012-01
tracestate: cost_usd=0.00015,tenant=tenant_123
X-Trace-Id: 7d3f4e8a9b2c1d5e6f7a8b9c0d1e2f3a
X-Parent-Span: 2b3c4d5e6f789012
X-Cost-USD: 0.00015
```

#### Message Queue Attributes
```json
{
  "message_attributes": {
    "traceparent": "00-7d3f4e8a9b2c1d5e6f7a8b9c0d1e2f3a-1a2b3c4d5e6f7890-01",
    "tracestate": "cost_usd=0.0001",
    "trace_id": "7d3f4e8a9b2c1d5e6f7a8b9c0d1e2f3a",
    "cost_usd": "0.0001"
  }
}
```

#### gRPC Metadata
```python
metadata = [
    ('traceparent', '00-7d3f4e8a9b2c1d5e6f7a8b9c0d1e2f3a-3c4d5e6f78901234-01'),
    ('tracestate', 'cost_usd=0.001'),
    ('x-trace-id', '7d3f4e8a9b2c1d5e6f7a8b9c0d1e2f3a'),
    ('x-cost-usd', '0.001')
]
```

## Cost Aggregation Query Examples

### Total Cost by Trace
```sql
SELECT 
  trace_id,
  SUM(cost_usd) as total_cost,
  COUNT(*) as span_count,
  MAX(end_time) - MIN(start_time) as duration_ms
FROM spans
WHERE trace_id = '7d3f4e8a9b2c1d5e6f7a8b9c0d1e2f3a'
GROUP BY trace_id;
```

### Cost Breakdown by Service
```sql
SELECT 
  trace_id,
  service,
  capability,
  SUM(cost_usd) as service_cost
FROM spans
WHERE trace_id = '7d3f4e8a9b2c1d5e6f7a8b9c0d1e2f3a'
GROUP BY trace_id, service, capability
ORDER BY service_cost DESC;
```

### Top Expensive Traces (Last Hour)
```sql
SELECT 
  trace_id,
  tenant,
  SUM(cost_usd) as total_cost,
  MIN(start_time) as trace_start
FROM spans
WHERE start_time > NOW() - INTERVAL '1 hour'
GROUP BY trace_id, tenant
HAVING SUM(cost_usd) > 0.01
ORDER BY total_cost DESC
LIMIT 10;
```
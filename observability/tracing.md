# observability/tracing.md

## Distributed Tracing Implementation

### Overview
This document extends the tracing implementation to include trace ID propagation, sampling rules, and cost attribution across all services.

## Trace Context Propagation

### W3C Trace Context Standard
We implement the W3C Trace Context standard for trace propagation:

```
traceparent: 00-{trace_id}-{parent_id}-{trace_flags}
tracestate: cost_usd=0.0001,data_tag=PII
```

### Implementation Examples

#### HTTP Headers
```python
# Python/Flask example
from opentelemetry import trace
from opentelemetry.propagate import inject

def make_request(url, data):
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span("http_request") as span:
        # Add cost attribution
        span.set_attribute("cost_usd", 0.0001)
        span.set_attribute("data_tag", "EXPORT_OK")
        
        headers = {}
        inject(headers)  # Injects traceparent and tracestate
        
        response = requests.post(url, json=data, headers=headers)
        return response
```

#### gRPC Metadata
```python
# Python/gRPC example
import grpc
from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient

def call_grpc_service(stub, request):
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span("grpc_call") as span:
        span.set_attribute("cost_usd", 0.0002)
        
        metadata = []
        inject(dict(metadata))  # Propagate trace context
        
        response = stub.ProcessRequest(request, metadata=metadata)
        return response
```

#### Message Queue Headers
```python
# RabbitMQ/Kafka example
def publish_message(channel, message):
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span("publish_message") as span:
        span.set_attribute("cost_usd", 0.00005)
        
        headers = {}
        inject(headers)
        
        # For RabbitMQ
        channel.basic_publish(
            exchange='',
            routing_key='task_queue',
            body=json.dumps(message),
            properties=pika.BasicProperties(headers=headers)
        )
```

## Sampling Rules

### Adaptive Sampling Configuration
```yaml
# sampling_rules.yaml
sampling:
  # Default sampling rate
  default_rate: 0.1  # 10% of traces
  
  # Service-specific rates
  service_rates:
    payment_service: 1.0      # 100% for critical service
    analytics_service: 0.01   # 1% for high-volume service
    robot_control: 1.0        # 100% for safety-critical
  
  # Priority sampling rules
  priority_rules:
    - name: "Error traces"
      condition: "error = true"
      sample_rate: 1.0
    
    - name: "High cost operations"
      condition: "cost_usd > 0.01"
      sample_rate: 1.0
    
    - name: "HITL operations"
      condition: "hitl_required = true"
      sample_rate: 1.0
    
    - name: "PII data access"
      condition: "data_tag = PII"
      sample_rate: 1.0
    
    - name: "Slow requests"
      condition: "duration_ms > 1000"
      sample_rate: 0.5
  
  # Adaptive sampling based on traffic
  adaptive:
    enabled: true
    target_traces_per_second: 100
    min_sampling_rate: 0.001
    max_sampling_rate: 1.0
    adjustment_interval: 60s
```

### Sampling Decision Propagation
```python
# Ensure sampling decision is propagated
def should_sample(span_context, attributes):
    # Check if parent made sampling decision
    if span_context.trace_flags & SAMPLED_FLAG:
        return True
    
    # Apply local sampling rules
    if attributes.get("error"):
        return True
    
    if attributes.get("cost_usd", 0) > 0.01:
        return True
    
    # Default sampling rate
    return random.random() < 0.1
```

## Cost Attribution

### Cost Calculation Model
```python
# cost_calculator.py
class CostCalculator:
    """Calculate cost_usd for different operations"""
    
    # Base costs per operation type
    COSTS = {
        "api_call": 0.00001,
        "database_query": 0.00002,
        "ml_inference": 0.0001,
        "robot_actuation": 0.001,
        "storage_read": 0.000001,
        "storage_write": 0.000005,
    }
    
    @classmethod
    def calculate_span_cost(cls, span):
        """Calculate cost for a span"""
        operation = span.attributes.get("operation_type")
        base_cost = cls.COSTS.get(operation, 0.00001)
        
        # Adjust for duration
        duration_ms = span.end_time - span.start_time
        duration_factor = duration_ms / 100  # Per 100ms
        
        # Adjust for data size
        data_size = span.attributes.get("data_size_bytes", 0)
        data_factor = data_size / (1024 * 1024)  # Per MB
        
        total_cost = base_cost * (1 + duration_factor * 0.1 + data_factor * 0.01)
        
        return round(total_cost, 6)
```

### Cost Aggregation
```python
# Aggregate costs across trace
def aggregate_trace_cost(trace_id):
    spans = get_spans_for_trace(trace_id)
    total_cost = sum(span.attributes.get("cost_usd", 0) for span in spans)
    
    # Store aggregated cost
    store_trace_cost(trace_id, total_cost)
    
    # Alert if exceeds threshold
    if total_cost > 0.1:
        alert_high_cost_operation(trace_id, total_cost)
    
    return total_cost
```

## Trace Storage & Retention

### Storage Strategy
```yaml
trace_storage:
  # Hot storage (immediate access)
  hot:
    duration: 24h
    backend: redis
    full_fidelity: true
  
  # Warm storage (quick retrieval)
  warm:
    duration: 7d
    backend: elasticsearch
    sampling: 0.1  # Keep 10% of traces
    
  # Cold storage (archival)
  cold:
    duration: 30d
    backend: s3
    compression: zstd
    sampling: 0.01  # Keep 1% of traces
```

### Trace Data Classification
```python
def classify_trace_data(span):
    """Classify trace data for retention policy"""
    
    # Check for PII
    if has_pii_attributes(span):
        span.set_attribute("data_tag", "PII")
        span.set_attribute("retention_days", 30)
    
    # Check for sensitive data
    elif has_sensitive_attributes(span):
        span.set_attribute("data_tag", "SENSITIVE")
        span.set_attribute("retention_days", 90)
    
    # Default classification
    else:
        span.set_attribute("data_tag", "EXPORT_OK")
        span.set_attribute("retention_days", 365)
```

## Integration Points

### Service Mesh Integration
```yaml
# Istio/Envoy configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: envoy-tracing-config
data:
  tracing.yaml: |
    tracing:
      http:
        name: envoy.tracers.opentelemetry
        typed_config:
          "@type": type.googleapis.com/envoy.config.trace.v3.OpenTelemetryConfig
          grpc_service:
            envoy_grpc:
              cluster_name: otel-collector
          service_name: "{{ SERVICE_NAME }}"
          # Propagate cost_usd in baggage
          propagation:
            baggage:
              - "cost_usd"
              - "data_tag"
```

### Application Framework Integration

#### Spring Boot (Java)
```java
@Component
public class CostTracingInterceptor implements ClientHttpRequestInterceptor {
    
    @Override
    public ClientHttpResponse intercept(HttpRequest request, byte[] body, 
                                       ClientHttpRequestExecution execution) {
        Span span = tracer.currentSpan();
        
        // Add cost attribution
        span.tag("cost_usd", calculateCost(request, body));
        span.tag("data_tag", classifyData(body));
        
        // Propagate trace context
        tracer.inject(span.context(), Format.HTTP_HEADERS, 
                     new HttpHeadersCarrier(request.getHeaders()));
        
        return execution.execute(request, body);
    }
}
```

#### Express.js (Node.js)
```javascript
const { trace, context, propagation } = require('@opentelemetry/api');

// Middleware for trace propagation and cost attribution
app.use((req, res, next) => {
  const tracer = trace.getTracer('express-app');
  const parentContext = propagation.extract(context.active(), req.headers);
  
  const span = tracer.startSpan('http_request', {
    attributes: {
      'http.method': req.method,
      'http.url': req.url,
      'cost_usd': 0.00001,
      'data_tag': classifyRequest(req)
    }
  }, parentContext);
  
  // Propagate to response
  res.on('finish', () => {
    span.setAttribute('http.status_code', res.statusCode);
    span.setAttribute('cost_usd', calculateFinalCost(req, res));
    span.end();
  });
  
  next();
});
```

## Monitoring & Alerting

### Trace-based Alerts
```yaml
# trace_alerts.yaml
alerts:
  - name: "Orphaned Spans"
    query: "traces without parent_id and not root"
    threshold: 100
    window: 5m
    severity: warning
    
  - name: "Trace Cost Anomaly"
    query: "sum(cost_usd) by trace_id > 0.1"
    threshold: 1
    window: 1m
    severity: critical
    
  - name: "Missing Cost Attribution"
    query: "spans without cost_usd attribute"
    threshold: 1000
    window: 10m
    severity: warning
    
  - name: "Trace Depth Exceeded"
    query: "trace depth > 50"
    threshold: 1
    window: 5m
    severity: warning
```

### Dashboard Queries
```sql
-- Top expensive traces
SELECT 
  trace_id,
  SUM(cost_usd) as total_cost,
  COUNT(*) as span_count,
  MAX(duration_ms) as max_duration
FROM spans
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY trace_id
ORDER BY total_cost DESC
LIMIT 10;

-- Cost by service
SELECT 
  service_name,
  SUM(cost_usd) as total_cost,
  AVG(cost_usd) as avg_cost,
  COUNT(*) as operation_count
FROM spans
WHERE timestamp > NOW() - INTERVAL '1 day'
GROUP BY service_name
ORDER BY total_cost DESC;

-- Data classification distribution
SELECT 
  data_tag,
  COUNT(*) as span_count,
  SUM(cost_usd) as total_cost
FROM spans
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY data_tag;
```

## Troubleshooting

### Common Issues

#### Lost Trace Context
```bash
# Debug script to verify trace propagation
curl -H "traceparent: 00-${TRACE_ID}-${PARENT_ID}-01" \
     -H "tracestate: cost_usd=0.001" \
     -v http://service/api/endpoint

# Check if trace context is logged
grep "trace_id=${TRACE_ID}" /var/log/application.log
```

#### High Sampling Overhead
```python
# Optimize sampling decision caching
@lru_cache(maxsize=10000)
def get_sampling_decision(trace_id):
    """Cache sampling decisions to reduce overhead"""
    return should_sample(trace_id)
```

#### Cost Attribution Drift
```sql
-- Audit query to find cost discrepancies
WITH trace_costs AS (
  SELECT 
    trace_id,
    SUM(cost_usd) as calculated_cost
  FROM spans
  GROUP BY trace_id
)
SELECT 
  t.trace_id,
  t.calculated_cost,
  m.recorded_cost,
  ABS(t.calculated_cost - m.recorded_cost) as drift
FROM trace_costs t
JOIN trace_metadata m ON t.trace_id = m.trace_id
WHERE ABS(t.calculated_cost - m.recorded_cost) > 0.001
ORDER BY drift DESC;
```

## Performance Considerations

### Overhead Targets
- **CPU Overhead:** < 1% for instrumentation
- **Memory Overhead:** < 50MB per service
- **Network Overhead:** < 0.1% additional traffic
- **Latency Impact:** < 1ms per span

### Optimization Strategies
1. **Batch span exports:** Group spans in batches of 100-1000
2. **Compress trace data:** Use protobuf/gzip for wire format
3. **Local span aggregation:** Pre-aggregate metrics locally
4. **Sampling at edge:** Make sampling decisions early
5. **Async processing:** Use background threads for export
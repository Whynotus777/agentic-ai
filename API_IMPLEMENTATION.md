# Public API & Orchestrator HITL Enforcement Implementation

## Overview
This implementation provides a public API contract with minimal orchestrator enforcement for Human-In-The-Loop (HITL) approval workflows. The system enforces header requirements, exposes standardized error codes, and ensures risky operations require human approval before execution.

## Key Components

### 1. API Layer (`api/`)

#### **openapi.yaml**
- Complete OpenAPI 3.0.3 specification
- Documents all endpoints, schemas, and error codes
- Self-contained (no external references)
- Accessible via `GET /openapi`

#### **errors.py**
- Canonical `ErrorCode` enum matching shared contract
- `APIError` class for structured error responses
- Factory functions for common error types
- Automatic HTTP header support (e.g., `Retry-After`)

#### **routes.py**
- Flask-based REST API implementation
- Required header validation (`Idempotency-Key`, `X-Tenant-ID`)
- Idempotency support with request caching
- Four main endpoints:
  - `POST /tasks` - Create and enqueue tasks
  - `GET /tasks/{id}` - Get task status and events
  - `POST /approvals/{task_id}` - Record HITL approvals
  - `GET /openapi` - Retrieve API specification

#### **schemas.py**
- Marshmallow schemas for request/response validation
- Strict validation for operations and payloads
- Header format validators

#### **security.py**
- Tenant validation logic (UUID format check)
- Rate limiting support (configurable per tenant)
- Approval token generation and validation
- Permission checking for operations

### 2. Orchestrator Layer (`orchestrator/`)

#### **brain.py**
- Minimal HITL enforcement logic
- Identifies risky operations (`commit`, `actuate`)
- Blocks risky operations without approval
- Raises `HITL_REQUIRED` error with approval link
- Double-checks approvals before execution (defense in depth)

### 3. Supporting Components

#### **storage/manifest.py**
- HITL approval storage (in-memory + file persistence)
- Records approval decisions with audit trail
- Provides approval lookup for orchestrator checks

#### **execution/queue.py**
- Task queue abstraction (in-memory Queue)
- Async task processing support
- Queue metrics for monitoring

#### **tests/test_api_contract.py**
- Comprehensive test coverage for:
  - Header validation
  - Error code compliance
  - Idempotency behavior
  - HITL approval flow
  - Rate limiting
  - OpenAPI validity

## HITL Flow

1. **Task Creation**: Client sends `POST /tasks` with required headers
2. **Processing**: Task queued for execution
3. **Risk Check**: Orchestrator identifies risky operations
4. **HITL Required**: Returns `HITL_REQUIRED` error with approval link
5. **Human Review**: Human reviews and approves/rejects via `POST /approvals/{task_id}`
6. **Execution**: Orchestrator proceeds with approved operations

## Error Handling

The system uses a standardized error enum:
- `SCHEMA_VALIDATION_FAILED` - Invalid request format/headers
- `AUTHENTICATION_FAILED` - Authentication issues
- `AUTHORIZATION_FAILED` - Permission denied
- `RESOURCE_NOT_FOUND` - Task/resource not found
- `RATE_LIMIT_EXCEEDED` - Too many requests (includes `Retry-After`)
- `HITL_REQUIRED` - Human approval needed (includes `approval_link`)
- `EXECUTION_TIMEOUT` - Operation timeout
- `DEPENDENCY_FAILED` - External dependency failure
- `INTERNAL_ERROR` - Server-side errors

## Security Features

1. **Required Headers**:
   - `Idempotency-Key`: Prevents duplicate operations
   - `X-Tenant-ID`: Multi-tenant isolation

2. **Rate Limiting**:
   - Per-tenant request limits
   - `Retry-After` header on rate limit errors

3. **HITL Approval**:
   - Token-based approval validation
   - Audit trail for all decisions
   - Defense-in-depth checks

## Running the System

```python
# Start Flask API
from api.routes import app
app.run(host='0.0.0.0', port=5000)

# Run tests
pytest tests/test_api_contract.py -v

# Access OpenAPI spec
curl http://localhost:5000/openapi
```

## Production Considerations

1. **Storage**: Replace in-memory stores with persistent databases (PostgreSQL, Redis)
2. **Queue**: Use proper message queue (RabbitMQ, AWS SQS, Redis)
3. **Authentication**: Implement proper JWT/OAuth token validation
4. **Monitoring**: Add metrics, logging, and distributed tracing
5. **Scaling**: Deploy with load balancers and multiple workers
6. **Security**: Add TLS, API keys, and proper secret management

## Acceptance Criteria ✓

- [x] Missing `Idempotency-Key` or `X-Tenant-ID` returns `SCHEMA_VALIDATION_FAILED`
- [x] Retry scenario returns `Retry-After` header
- [x] `errors.py` exposes exact enum from shared contract
- [x] Orchestrator refuses `repo.commit` and `robot.actuate` without approval
- [x] Valid `openapi.yaml` with no external links
- [x] Comprehensive unit tests for contract compliance
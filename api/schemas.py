# api/schemas.py
from marshmallow import Schema, fields, validate, ValidationError
from typing import Dict, Any

class TaskRequestSchema(Schema):
    """Schema for task creation requests"""
    operation = fields.Str(
        required=True,
        validate=validate.OneOf(['analyze', 'transform', 'commit', 'actuate']),
        error_messages={'required': 'Operation is required'}
    )
    payload = fields.Dict(
        required=True,
        error_messages={'required': 'Payload is required'}
    )
    metadata = fields.Dict(
        missing={},
        default={}
    )
    
    class Meta:
        strict = True

class ApprovalRequestSchema(Schema):
    """Schema for HITL approval requests"""
    approval_token = fields.Str(
        required=True,
        validate=validate.Length(min=1, max=256),
        error_messages={'required': 'Approval token is required'}
    )
    decision = fields.Str(
        required=True,
        validate=validate.OneOf(['approve', 'reject']),
        error_messages={'required': 'Decision is required'}
    )
    reason = fields.Str(
        missing=None,
        validate=validate.Length(max=1000)
    )
    
    class Meta:
        strict = True

class TaskResponseSchema(Schema):
    """Schema for task creation response"""
    task_id = fields.UUID(required=True)
    trace_id = fields.UUID(required=True)
    status = fields.Str(
        required=True,
        validate=validate.OneOf(['queued', 'processing', 'hitl_required', 'completed', 'failed'])
    )
    created_at = fields.DateTime(required=True)

class TaskEventSchema(Schema):
    """Schema for task events"""
    event_type = fields.Str(required=True)
    timestamp = fields.DateTime(required=True)
    data = fields.Dict(missing={})
    approval_link = fields.Str(missing=None)

class TaskStatusSchema(Schema):
    """Schema for task status response"""
    task_id = fields.UUID(required=True)
    trace_id = fields.UUID(required=True)
    status = fields.Str(
        required=True,
        validate=validate.OneOf(['queued', 'processing', 'hitl_required', 'completed', 'failed'])
    )
    last_event = fields.Nested(TaskEventSchema, missing=None)
    created_at = fields.DateTime(required=True)
    updated_at = fields.DateTime(required=True)

class ErrorResponseSchema(Schema):
    """Schema for error responses"""
    error_code = fields.Str(
        required=True,
        validate=validate.OneOf([
            'SCHEMA_VALIDATION_FAILED',
            'AUTHENTICATION_FAILED',
            'AUTHORIZATION_FAILED',
            'RESOURCE_NOT_FOUND',
            'RATE_LIMIT_EXCEEDED',
            'HITL_REQUIRED',
            'EXECUTION_TIMEOUT',
            'DEPENDENCY_FAILED',
            'INTERNAL_ERROR'
        ])
    )
    message = fields.Str(required=True)
    details = fields.Dict(missing=None)
    trace_id = fields.UUID(missing=None)

# Header validators
def validate_idempotency_key(key: str) -> bool:
    """Validate idempotency key format"""
    if not key:
        return False
    if len(key) < 1 or len(key) > 256:
        return False
    return True

def validate_tenant_id_format(tenant_id: str) -> bool:
    """Validate tenant ID is valid UUID format"""
    if not tenant_id:
        return False
    try:
        import uuid
        uuid.UUID(tenant_id)
        return True
    except ValueError:
        return False
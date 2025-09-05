# tests/test_api_contract.py
import pytest
import json
import uuid
import yaml
from unittest.mock import Mock, patch
from datetime import datetime

from api.routes import app
from api.errors import ErrorCode, APIError
from orchestrator.brain import OrchestrationBrain

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def valid_headers():
    """Valid request headers"""
    return {
        'Idempotency-Key': str(uuid.uuid4()),
        'X-Tenant-ID': str(uuid.uuid4()),
        'Content-Type': 'application/json'
    }

@pytest.fixture
def valid_task_payload():
    """Valid task creation payload"""
    return {
        "operation": "analyze",
        "payload": {
            "data": "test data"
        }
    }

class TestHeaderValidation:
    """Test header validation requirements"""
    
    def test_missing_idempotency_key(self, client):
        """Test that missing Idempotency-Key returns SCHEMA_VALIDATION_FAILED"""
        headers = {'X-Tenant-ID': str(uuid.uuid4())}
        response = client.post('/tasks', headers=headers, json={})
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error_code'] == ErrorCode.SCHEMA_VALIDATION_FAILED.value
        assert 'Idempotency-Key' in data['message']
    
    def test_missing_tenant_id(self, client):
        """Test that missing X-Tenant-ID returns SCHEMA_VALIDATION_FAILED"""
        headers = {'Idempotency-Key': str(uuid.uuid4())}
        response = client.post('/tasks', headers=headers, json={})
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error_code'] == ErrorCode.SCHEMA_VALIDATION_FAILED.value
        assert 'X-Tenant-ID' in data['message']
    
    def test_invalid_tenant_id_format(self, client):
        """Test that invalid tenant ID format returns error"""
        headers = {
            'Idempotency-Key': str(uuid.uuid4()),
            'X-Tenant-ID': 'not-a-uuid'
        }
        response = client.post('/tasks', headers=headers, json={})
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error_code'] == ErrorCode.SCHEMA_VALIDATION_FAILED.value

class TestTaskEndpoints:
    """Test task creation and retrieval endpoints"""
    
    @patch('api.routes.enqueue_task')
    def test_create_task_success(self, mock_enqueue, client, valid_headers, valid_task_payload):
        """Test successful task creation"""
        response = client.post('/tasks', 
                              headers=valid_headers, 
                              json=valid_task_payload)
        
        assert response.status_code == 201
        data = json.loads(response.data)
        assert 'task_id' in data
        assert 'trace_id' in data
        assert data['status'] == 'queued'
        assert mock_enqueue.called
    
    @patch('api.routes.enqueue_task')
    def test_idempotency_key_returns_same_task(self, mock_enqueue, client, valid_headers, valid_task_payload):
        """Test that same idempotency key returns cached response"""
        # First request
        response1 = client.post('/tasks', 
                               headers=valid_headers, 
                               json=valid_task_payload)
        data1 = json.loads(response1.data)
        
        # Second request with same idempotency key
        response2 = client.post('/tasks', 
                               headers=valid_headers, 
                               json=valid_task_payload)
        data2 = json.loads(response2.data)
        
        assert data1['task_id'] == data2['task_id']
        assert mock_enqueue.call_count == 1  # Only called once
    
    def test_get_task_not_found(self, client, valid_headers):
        """Test getting non-existent task"""
        fake_task_id = str(uuid.uuid4())
        response = client.get(f'/tasks/{fake_task_id}', headers=valid_headers)
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data['error_code'] == ErrorCode.RESOURCE_NOT_FOUND.value
    
    @patch('api.routes.tasks_db')
    def test_get_task_with_hitl_event(self, mock_tasks_db, client, valid_headers):
        """Test getting task with HITL required event"""
        task_id = str(uuid.uuid4())
        tenant_id = valid_headers['X-Tenant-ID']
        
        mock_tasks_db.get.return_value = {
            "task_id": task_id,
            "trace_id": str(uuid.uuid4()),
            "tenant_id": tenant_id,
            "status": "hitl_required",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "events": [{
                "event_type": "HITL_REQUIRED",
                "timestamp": datetime.utcnow().isoformat(),
                "approval_link": f"/approvals/{task_id}"
            }]
        }
        
        response = client.get(f'/tasks/{task_id}', headers=valid_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'hitl_required'
        assert 'last_event' in data
        assert data['last_event']['event_type'] == 'HITL_REQUIRED'
        assert 'approval_link' in data['last_event']

class TestApprovalEndpoint:
    """Test HITL approval endpoint"""
    
    @patch('api.routes.tasks_db')
    @patch('api.routes.manifest_store')
    def test_approve_task_success(self, mock_manifest, mock_tasks_db, client, valid_headers):
        """Test successful task approval"""
        task_id = str(uuid.uuid4())
        tenant_id = valid_headers['X-Tenant-ID']
        
        mock_tasks_db.get.return_value = {
            "task_id": task_id,
            "tenant_id": tenant_id,
            "status": "hitl_required",
            "events": [],
            "updated_at": datetime.utcnow().isoformat()
        }
        
        approval_data = {
            "approval_token": "test-token-123",
            "decision": "approve",
            "reason": "Looks good"
        }
        
        response = client.post(f'/approvals/{task_id}', 
                              headers=valid_headers,
                              json=approval_data)
        
        assert response.status_code == 204
        assert mock_manifest.record_approval.called

class TestRateLimiting:
    """Test rate limiting behavior"""
    
    @patch('api.routes.check_rate_limit')
    def test_rate_limit_exceeded(self, mock_rate_limit, client, valid_headers):
        """Test rate limit exceeded returns proper error and header"""
        mock_rate_limit.return_value = True
        
        response = client.post('/tasks', 
                              headers=valid_headers,
                              json={})
        
        assert response.status_code == 429
        assert 'Retry-After' in response.headers
        data = json.loads(response.data)
        assert data['error_code'] == ErrorCode.RATE_LIMIT_EXCEEDED.value

class TestErrorEnum:
    """Test error enum matches contract"""
    
    def test_error_enum_values(self):
        """Test that all required error codes exist"""
        required_codes = [
            'SCHEMA_VALIDATION_FAILED',
            'AUTHENTICATION_FAILED',
            'AUTHORIZATION_FAILED',
            'RESOURCE_NOT_FOUND',
            'RATE_LIMIT_EXCEEDED',
            'HITL_REQUIRED',
            'EXECUTION_TIMEOUT',
            'DEPENDENCY_FAILED',
            'INTERNAL_ERROR'
        ]
        
        for code_name in required_codes:
            assert hasattr(ErrorCode, code_name)
            assert getattr(ErrorCode, code_name).value == code_name

class TestOpenAPIEndpoint:
    """Test OpenAPI specification endpoint"""
    
    @patch('builtins.open', create=True)
    def test_get_openapi_spec(self, mock_open, client):
        """Test OpenAPI spec retrieval"""
        mock_spec = "openapi: 3.0.3\ninfo:\n  title: Test API"
        mock_open.return_value.__enter__.return_value.read.return_value = mock_spec
        
        response = client.get('/openapi')
        
        assert response.status_code == 200
        assert response.content_type == 'application/yaml'
        assert b'openapi' in response.data
    
    def test_openapi_spec_valid_yaml(self):
        """Test that OpenAPI spec is valid YAML"""
        # Read the actual OpenAPI spec from the artifact
        spec_content = open('api/openapi.yaml', 'r').read()
        parsed = yaml.safe_load(spec_content)
        
        # Validate basic OpenAPI structure
        assert 'openapi' in parsed
        assert 'info' in parsed
        assert 'paths' in parsed
        assert 'components' in parsed
        
        # Validate required endpoints exist
        required_paths = ['/tasks', '/tasks/{taskId}', '/approvals/{taskId}', '/openapi']
        for path in required_paths:
            assert path in parsed['paths']
        
        # Validate error schema includes all enum values
        error_schema = parsed['components']['schemas']['ErrorResponse']
        error_codes = error_schema['properties']['error_code']['enum']
        
        required_codes = [
            'SCHEMA_VALIDATION_FAILED',
            'AUTHENTICATION_FAILED',
            'AUTHORIZATION_FAILED',
            'RESOURCE_NOT_FOUND',
            'RATE_LIMIT_EXCEEDED',
            'HITL_REQUIRED',
            'EXECUTION_TIMEOUT',
            'DEPENDENCY_FAILED',
            'INTERNAL_ERROR'
        ]
        
        for code in required_codes:
            assert code in error_codes

class TestOrchestratorHITLEnforcement:
    """Test orchestrator HITL enforcement for risky operations"""
    
    def test_commit_without_approval_raises_hitl_required(self):
        """Test that commit operation without approval raises HITL_REQUIRED"""
        brain = OrchestrationBrain()
        task_id = str(uuid.uuid4())
        
        with pytest.raises(APIError) as exc_info:
            brain.execute_operation(task_id, "commit", {
                "repository": "test-repo",
                "files": ["file1.py"]
            })
        
        assert exc_info.value.error_code == ErrorCode.HITL_REQUIRED
        assert exc_info.value.status_code == 202
        assert 'approval_link' in exc_info.value.details
    
    def test_actuate_without_approval_raises_hitl_required(self):
        """Test that actuate operation without approval raises HITL_REQUIRED"""
        brain = OrchestrationBrain()
        task_id = str(uuid.uuid4())
        
        with pytest.raises(APIError) as exc_info:
            brain.execute_operation(task_id, "actuate", {
                "robot_id": "robot-123",
                "action": "move"
            })
        
        assert exc_info.value.error_code == ErrorCode.HITL_REQUIRED
    
    @patch('orchestrator.brain.ManifestStore')
    def test_commit_with_approval_proceeds(self, mock_manifest_class):
        """Test that commit with approval proceeds successfully"""
        brain = OrchestrationBrain()
        task_id = str(uuid.uuid4())
        
        # Mock approval exists
        mock_manifest_class.return_value.get_approval.return_value = {
            "decision": "approve",
            "approval_token": "test-token"
        }
        brain.manifest_store = mock_manifest_class.return_value
        
        result = brain.execute_operation(task_id, "commit", {
            "repository": "test-repo",
            "files": ["file1.py"]
        })
        
        assert result['status'] == 'completed'
        assert 'commit_id' in result['result']
    
    def test_analyze_needs_no_approval(self):
        """Test that analyze operation doesn't require approval"""
        brain = OrchestrationBrain()
        task_id = str(uuid.uuid4())
        
        # Should succeed without any approval
        result = brain.execute_operation(task_id, "analyze", {"data": "test"})
        
        assert result['status'] == 'completed'
    
    def test_transform_needs_no_approval(self):
        """Test that transform operation doesn't require approval"""
        brain = OrchestrationBrain()
        task_id = str(uuid.uuid4())
        
        # Should succeed without any approval
        result = brain.execute_operation(task_id, "transform", {"data": "test"})
        
        assert result['status'] == 'completed'
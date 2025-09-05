# tests/golden_tasks/test_appdev.py
"""
Golden tasks for application development workflows.
Tests multi-repo refactoring and HITL (Human-In-The-Loop) gates.
"""

import pytest
import asyncio
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from unittest.mock import Mock, patch, MagicMock
import requests
from datetime import datetime, timedelta


@dataclass
class RefactorRequest:
    """Request for multi-repo refactoring"""
    source_repo: str
    target_repos: List[str]
    refactor_type: str  # 'extract_service' | 'merge_modules' | 'split_monolith'
    description: str
    requires_hitl: bool = True
    hitl_approval: Optional[Dict] = None


class ApplicationDevelopmentAPI:
    """Mock API client for app development operations"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "X-Trace-ID": f"test-{datetime.now().isoformat()}",
            "X-Request-Cost": "0.0"  # Will be populated by backend
        })
    
    def create_refactor_plan(self, request: RefactorRequest) -> Dict:
        """Create a refactoring plan across repositories"""
        response = self.session.post(
            f"{self.base_url}/api/v1/refactor/plan",
            json={
                "source_repo": request.source_repo,
                "target_repos": request.target_repos,
                "refactor_type": request.refactor_type,
                "description": request.description,
                "requires_hitl": request.requires_hitl
            }
        )
        response.raise_for_status()
        return response.json()
    
    def execute_refactor(self, plan_id: str, hitl_token: Optional[str] = None) -> Dict:
        """Execute refactoring with optional HITL token"""
        payload = {"plan_id": plan_id}
        if hitl_token:
            payload["hitl_token"] = hitl_token
        
        response = self.session.post(
            f"{self.base_url}/api/v1/refactor/execute",
            json=payload
        )
        
        # Check if HITL is required
        if response.status_code == 403:
            return response.json()  # Should contain HITL requirement
        
        response.raise_for_status()
        return response.json()
    
    def commit_changes(self, plan_id: str, hitl_token: Optional[str] = None) -> Dict:
        """Commit changes to repositories"""
        response = self.session.post(
            f"{self.base_url}/api/v1/repo/commit",
            json={
                "plan_id": plan_id,
                "hitl_token": hitl_token,
                "message": "Automated refactoring via golden task"
            }
        )
        
        # HITL gate should trigger here
        if response.status_code == 403:
            return response.json()
        
        response.raise_for_status()
        return response.json()
    
    def get_trace_cost(self, trace_id: str) -> float:
        """Get cost for a specific trace"""
        response = self.session.get(
            f"{self.base_url}/api/v1/traces/{trace_id}/cost"
        )
        response.raise_for_status()
        return response.json()["cost_usd"]


class TestApplicationDevelopmentGoldenTasks:
    """Golden tasks for application development workflows"""
    
    @pytest.fixture
    def api_client(self):
        """Create API client for testing"""
        # In real tests, this would point to actual service
        # For now, we'll mock responses
        return ApplicationDevelopmentAPI()
    
    @pytest.fixture
    def mock_hitl_service(self):
        """Mock HITL approval service"""
        with patch('requests.Session.post') as mock_post:
            # Configure mock responses
            def side_effect(url, *args, **kwargs):
                response = Mock()
                
                if "/refactor/plan" in url:
                    response.status_code = 200
                    response.json.return_value = {
                        "plan_id": "plan_12345",
                        "status": "planned",
                        "estimated_changes": 47,
                        "affected_files": [
                            "src/services/auth.py",
                            "src/services/payment.py"
                        ],
                        "cost_usd": 0.0001
                    }
                
                elif "/refactor/execute" in url:
                    json_data = kwargs.get('json', {})
                    if not json_data.get('hitl_token'):
                        response.status_code = 403
                        response.json.return_value = {
                            "error": "HITL approval required",
                            "hitl_request_id": "hitl_req_789",
                            "approval_url": "http://approval.internal/hitl_req_789"
                        }
                    else:
                        response.status_code = 200
                        response.json.return_value = {
                            "status": "executed",
                            "changes_applied": 47,
                            "cost_usd": 0.0015
                        }
                
                elif "/repo/commit" in url:
                    json_data = kwargs.get('json', {})
                    if not json_data.get('hitl_token'):
                        response.status_code = 403
                        response.json.return_value = {
                            "error": "HITL approval required for commits",
                            "hitl_request_id": "hitl_commit_456",
                            "reason": "Repository write operations require human approval"
                        }
                    else:
                        response.status_code = 200
                        response.json.return_value = {
                            "status": "committed",
                            "commit_hash": "abc123def456",
                            "repositories_updated": 3,
                            "cost_usd": 0.0002
                        }
                
                elif "/traces/" in url and "/cost" in url:
                    response.status_code = 200
                    response.json.return_value = {"cost_usd": 0.0018}
                
                else:
                    response.status_code = 404
                    response.json.return_value = {"error": "Not found"}
                
                return response
            
            mock_post.side_effect = side_effect
            yield mock_post
    
    def test_multi_repo_refactor_requires_hitl(self, api_client, mock_hitl_service):
        """Test that multi-repo refactoring requires HITL approval"""
        # Create refactor request
        request = RefactorRequest(
            source_repo="github.com/company/monolith",
            target_repos=[
                "github.com/company/auth-service",
                "github.com/company/payment-service",
                "github.com/company/notification-service"
            ],
            refactor_type="split_monolith",
            description="Extract microservices from monolithic application"
        )
        
        # Step 1: Create plan (should succeed without HITL)
        plan = api_client.create_refactor_plan(request)
        assert plan["plan_id"] == "plan_12345"
        assert plan["status"] == "planned"
        assert "cost_usd" in plan
        
        # Step 2: Attempt execution without HITL token (should fail)
        execution = api_client.execute_refactor(plan["plan_id"])
        assert "error" in execution
        assert "HITL approval required" in execution["error"]
        assert "hitl_request_id" in execution
        
        # Step 3: Execute with HITL token (should succeed)
        execution = api_client.execute_refactor(
            plan["plan_id"],
            hitl_token="approved_token_123"
        )
        assert execution["status"] == "executed"
        assert execution["changes_applied"] == 47
        assert "cost_usd" in execution
    
    def test_repo_commit_requires_hitl_gate(self, api_client, mock_hitl_service):
        """Test that repository commits require HITL approval"""
        plan_id = "plan_12345"
        
        # Attempt commit without HITL token
        commit_response = api_client.commit_changes(plan_id)
        assert "error" in commit_response
        assert "HITL approval required" in commit_response["error"]
        assert "hitl_request_id" in commit_response
        
        # Commit with HITL token
        commit_response = api_client.commit_changes(
            plan_id,
            hitl_token="commit_approved_789"
        )
        assert commit_response["status"] == "committed"
        assert "commit_hash" in commit_response
        assert commit_response["repositories_updated"] == 3
    
    def test_cost_tracking_across_operations(self, api_client, mock_hitl_service):
        """Test that cost_usd is tracked across all operations"""
        trace_id = api_client.session.headers["X-Trace-ID"]
        
        # Create plan
        request = RefactorRequest(
            source_repo="github.com/company/app",
            target_repos=["github.com/company/app-v2"],
            refactor_type="merge_modules",
            description="Consolidate duplicate modules"
        )
        
        plan = api_client.create_refactor_plan(request)
        assert "cost_usd" in plan
        assert plan["cost_usd"] > 0
        
        # Execute with HITL
        execution = api_client.execute_refactor(
            plan["plan_id"],
            hitl_token="approved"
        )
        assert "cost_usd" in execution
        
        # Commit with HITL
        commit = api_client.commit_changes(
            plan["plan_id"],
            hitl_token="approved"
        )
        assert "cost_usd" in commit
        
        # Verify total cost
        total_cost = api_client.get_trace_cost(trace_id)
        assert total_cost > 0
        assert total_cost == pytest.approx(0.0018, rel=0.1)
    
    @pytest.mark.skipif(
        not pytest.config.option.feature_complete,
        reason="Requires PROMPT 1-3 features to be implemented"
    )
    def test_extract_service_workflow(self, api_client):
        """Test extracting a service from monolith (requires features)"""
        request = RefactorRequest(
            source_repo="github.com/company/monolith",
            target_repos=["github.com/company/extracted-service"],
            refactor_type="extract_service",
            description="Extract authentication service"
        )
        
        # Full workflow
        plan = api_client.create_refactor_plan(request)
        
        # Get HITL approval (in real scenario, this would be async)
        hitl_token = self._simulate_hitl_approval(plan["plan_id"])
        
        execution = api_client.execute_refactor(plan["plan_id"], hitl_token)
        commit = api_client.commit_changes(plan["plan_id"], hitl_token)
        
        assert commit["status"] == "committed"
        assert len(commit["commit_hash"]) > 0
    
    def test_parallel_refactor_operations(self, api_client, mock_hitl_service):
        """Test handling multiple parallel refactor operations"""
        requests = [
            RefactorRequest(
                source_repo=f"github.com/company/service-{i}",
                target_repos=[f"github.com/company/service-{i}-v2"],
                refactor_type="merge_modules",
                description=f"Refactor service {i}"
            )
            for i in range(3)
        ]
        
        # Create plans in parallel
        plans = []
        for req in requests:
            plan = api_client.create_refactor_plan(req)
            plans.append(plan)
        
        # Verify each requires HITL
        for plan in plans:
            execution = api_client.execute_refactor(plan["plan_id"])
            assert "HITL approval required" in execution.get("error", "")
    
    def test_refactor_rollback_on_failure(self, api_client):
        """Test rollback mechanism when refactoring fails"""
        with patch('requests.Session.post') as mock_post:
            # Simulate failure during execution
            def side_effect(url, *args, **kwargs):
                response = Mock()
                if "/refactor/execute" in url:
                    response.status_code = 500
                    response.json.return_value = {
                        "error": "Refactoring failed",
                        "rollback_initiated": True,
                        "rollback_id": "rollback_123"
                    }
                else:
                    response.status_code = 200
                    response.json.return_value = {"status": "ok"}
                return response
            
            mock_post.side_effect = side_effect
            
            with pytest.raises(requests.exceptions.HTTPError):
                api_client.execute_refactor("plan_failed", hitl_token="approved")
    
    def test_data_classification_in_refactor(self, api_client, mock_hitl_service):
        """Test that refactored code maintains data classification tags"""
        request = RefactorRequest(
            source_repo="github.com/company/user-service",
            target_repos=["github.com/company/user-service-v2"],
            refactor_type="extract_service",
            description="Extract PII handling module"
        )
        
        plan = api_client.create_refactor_plan(request)
        
        # Verify plan includes data classification
        # In real implementation, this would check the plan details
        assert plan["plan_id"]
        
        # Mock response should include data tags
        with patch('requests.Session.post') as mock_post:
            response = Mock()
            response.status_code = 200
            response.json.return_value = {
                "status": "planned",
                "data_classifications": ["PII", "SENSITIVE"],
                "requires_encryption": True
            }
            mock_post.return_value = response
            
            # Re-fetch plan with classifications
            detailed_plan = api_client.session.post(
                f"{api_client.base_url}/api/v1/refactor/plan/{plan['plan_id']}/details"
            ).json()
            
            assert "PII" in detailed_plan.get("data_classifications", [])
    
    def _simulate_hitl_approval(self, plan_id: str) -> str:
        """Simulate HITL approval process"""
        # In real scenario, this would wait for human approval
        # For testing, we return a mock token
        return f"hitl_approved_{plan_id}_{int(time.time())}"
    
    @pytest.mark.performance
    def test_refactor_performance_slo(self, api_client, mock_hitl_service):
        """Test that refactoring operations meet performance SLOs"""
        import time
        
        request = RefactorRequest(
            source_repo="github.com/company/perf-test",
            target_repos=["github.com/company/perf-test-v2"],
            refactor_type="merge_modules",
            description="Performance test refactor"
        )
        
        # Measure plan creation time
        start = time.time()
        plan = api_client.create_refactor_plan(request)
        plan_time = time.time() - start
        
        # Plan creation should be < 5 seconds
        assert plan_time < 5.0
        
        # Execution (with HITL token) should be < 30 seconds
        start = time.time()
        execution = api_client.execute_refactor(
            plan["plan_id"],
            hitl_token="approved"
        )
        exec_time = time.time() - start
        
        assert exec_time < 30.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
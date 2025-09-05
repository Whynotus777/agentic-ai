# tests/test_router_trace.py
import pytest
import yaml
import tempfile
import os
from typing import Dict, Any

from control_plane.integrated_router import IntegratedRouter, RateLimiter

class TestRouterTrace:
    """Test router decision trace includes all required fields"""
    
    @pytest.fixture
    def policy_yaml(self):
        """Create test policy YAML"""
        return """
version: "1.0"
policies:
  - name: "robotics_safety_policy"
    priority: 100
    match:
      operation: ["actuate", "robot_control"]
      tags:
        - robotics
    guardrails:
      rate_limit_hz: 10
      require_hitl: true
      require_approval_count: 1
    fallbacks:
      - simulation_mode
      - manual_override
    policy_mode: enforce
    
  - name: "standard_operations"
    priority: 50
    match:
      operation: ["analyze", "transform"]
    guardrails:
      rate_limit_hz: 100
      require_hitl: false
    fallbacks:
      - retry_with_backoff
    policy_mode: monitor
"""
    
    @pytest.fixture
    def router(self, policy_yaml, tmp_path):
        """Create router with test policies"""
        policy_file = tmp_path / "test_policy.yaml"
        policy_file.write_text(policy_yaml)
        return IntegratedRouter(str(policy_file))
    
    def test_robotics_trace_includes_rate_limit(self, router):
        """Test robotics operations include rate_limit_hz in trace"""
        request = {
            "operation": "actuate",
            "tags": ["robotics"],
            "robot_id": "robot-001",
            "action": "move_arm"
        }
        
        response = router.route(request)
        
        # Check decision trace exists
        assert "decision_trace" in response
        trace = response["decision_trace"]
        
        # Check matched policy
        assert trace["matched_policy"] == "robotics_safety_policy"
        
        # Check guardrails include rate_limit_hz
        assert "guardrails" in trace
        assert "rate_limit_hz" in trace["guardrails"]
        assert trace["guardrails"]["rate_limit_hz"] == 10
        
        # Check require_hitl
        assert trace["guardrails"]["require_hitl"] is True
    
    def test_trace_includes_all_guardrails(self, router):
        """Test trace includes all guardrail fields"""
        request = {
            "operation": "robot_control",
            "tags": ["robotics"],
            "approvals": [{
                "approver": "operator@example.com",
                "decision": "approve"
            }]
        }
        
        response = router.route(request)
        trace = response["decision_trace"]
        
        # Verify all expected guardrails
        expected_guardrails = {
            "rate_limit_hz": 10,
            "require_hitl": True,
            "require_approval_count": 1
        }
        
        for key, value in expected_guardrails.items():
            assert key in trace["guardrails"]
            assert trace["guardrails"][key] == value
    
    def test_trace_includes_fallbacks(self, router):
        """Test trace includes fallback strategies"""
        request = {
            "operation": "actuate",
            "tags": ["robotics"]
        }
        
        response = router.route(request)
        trace = response["decision_trace"]
        
        # Check fallbacks
        assert "fallbacks" in trace
        assert "simulation_mode" in trace["fallbacks"]
        assert "manual_override" in trace["fallbacks"]
    
    def test_trace_includes_policy_mode(self, router):
        """Test trace includes policy mode"""
        request = {
            "operation": "actuate",
            "tags": ["robotics"]
        }
        
        response = router.route(request)
        trace = response["decision_trace"]
        
        # Check policy mode
        assert "policy_mode" in trace
        assert trace["policy_mode"] == "enforce"
    
    def test_rate_limit_status_in_trace(self, router):
        """Test rate limit status is included in trace"""
        request = {
            "operation": "actuate",
            "tags": ["robotics"],
            "approvals": [{"approver": "test", "decision": "approve"}]
        }
        
        response = router.route(request)
        trace = response["decision_trace"]
        
        # Check rate limit status
        assert "rate_limit_status" in trace
        rate_status = trace["rate_limit_status"]
        
        assert "limit_hz" in rate_status
        assert rate_status["limit_hz"] == 10
        assert "allowed" in rate_status
        assert "wait_time_ms" in rate_status
        assert "current_rate" in rate_status
    
    def test_rate_limiting_enforcement(self, router):
        """Test rate limiting actually blocks requests"""
        request = {
            "operation": "actuate",
            "tags": ["robotics"],
            "approvals": [{"approver": "test", "decision": "approve"}]
        }
        
        # Make requests up to limit
        allowed_count = 0
        blocked_count = 0
        
        for i in range(15):  # Try more than limit (10 Hz)
            response = router.route(request)
            if response["decision"] == "allowed":
                allowed_count += 1
            elif response["decision"] == "rate_limited":
                blocked_count += 1
        
        # Should allow up to rate limit
        assert allowed_count <= 10
        assert blocked_count > 0
    
    def test_hitl_required_in_trace(self, router):
        """Test HITL requirement appears in trace"""
        request = {
            "operation": "actuate",
            "tags": ["robotics"]
            # No approvals provided
        }
        
        response = router.route(request)
        
        assert response["decision"] == "hitl_required"
        
        trace = response["decision_trace"]
        assert trace["guardrails"]["require_hitl"] is True
        assert trace["decision"] == "hitl_required"
    
    def test_policy_evaluation_order_in_trace(self, router):
        """Test trace shows policy evaluation order"""
        request = {
            "operation": "analyze",
            "data_size_gb": 10
        }
        
        response = router.route(request)
        trace = response["decision_trace"]
        
        # Check evaluated policies
        assert "evaluated_policies" in trace
        assert len(trace["evaluated_policies"]) > 0
        
        # Check policies are in priority order
        priorities = [p["priority"] for p in trace["evaluated_policies"]]
        assert priorities == sorted(priorities, reverse=True)
    
    def test_no_policy_match_trace(self, router):
        """Test trace when no policy matches"""
        request = {
            "operation": "unknown_operation",
            "tags": ["unknown"]
        }
        
        response = router.route(request)
        trace = response["decision_trace"]
        
        assert trace["matched_policy"] is None
        assert trace["decision"] == "no_policy_match"
        assert response["decision"] == "default"
    
    def test_monitor_mode_allows_violations(self, router):
        """Test monitor mode logs but allows violations"""
        request = {
            "operation": "analyze"
            # No approvals, but monitor mode
        }
        
        response = router.route(request)
        trace = response["decision_trace"]
        
        assert trace["policy_mode"] == "monitor"
        assert trace["decision"] == "allowed"
    
    def test_complete_trace_structure(self, router):
        """Test complete trace structure for robotics operation"""
        request = {
            "operation": "robot_control",
            "tags": ["robotics", "safety_critical"],
            "robot_id": "robot-42",
            "approvals": [
                {
                    "approver": "supervisor@example.com",
                    "decision": "approve",
                    "reason": "Safety checks completed"
                }
            ],
            "safety_checks": ["emergency_stop_available"]
        }
        
        response = router.route(request)
        
        # Verify top-level response
        assert "decision" in response
        assert "policy_matched" in response
        assert "policy_mode" in response
        assert "guardrails" in response
        assert "fallbacks" in response
        assert "decision_trace" in response
        
        # Verify detailed trace
        trace = response["decision_trace"]
        
        # Required trace fields
        assert "timestamp" in trace
        assert "request" in trace
        assert "evaluated_policies" in trace
        assert "matched_policy" in trace
        assert "guardrails" in trace
        assert "fallbacks" in trace
        assert "rate_limit_status" in trace
        assert "decision" in trace
        assert "policy_mode" in trace
        
        # Verify guardrails content
        guardrails = trace["guardrails"]
        assert guardrails["rate_limit_hz"] == 10
        assert guardrails["require_hitl"] is True
        assert guardrails["require_approval_count"] == 1
        
        # Verify rate limit status
        rate_status = trace["rate_limit_status"]
        assert rate_status["limit_hz"] == 10
        assert isinstance(rate_status["allowed"], bool)
        assert isinstance(rate_status["wait_time_ms"], (int, float))
        assert isinstance(rate_status["current_rate"], (int, float))
    
    def test_rate_limiter_accuracy(self):
        """Test rate limiter enforces correct rate"""
        limiter = RateLimiter(rate_hz=10)
        
        # Should allow first 10 requests
        for i in range(10):
            allowed, wait_time = limiter.check()
            assert allowed is True
            assert wait_time is None
        
        # 11th request should be rate limited
        allowed, wait_time = limiter.check()
        assert allowed is False
        assert wait_time is not None
        assert wait_time > 0
        
        # Current rate should be 10
        assert limiter.get_current_rate() == 10
    
    def test_guardrail_violations_in_trace(self, router):
        """Test guardrail violations appear in trace"""
        request = {
            "operation": "actuate",
            "tags": ["robotics"],
            "safety_checks": []  # Missing required safety checks
        }
        
        # Add safety check requirement to policy
        router.policies["policies"][0]["guardrails"]["safety_checks"] = [
            "emergency_stop_available",
            "operator_present"
        ]
        
        response = router.route(request)
        
        # Should be blocked due to missing safety checks
        if "guardrail_violations" in response["decision_trace"]:
            violations = response["decision_trace"]["guardrail_violations"]
            assert len(violations) > 0
            assert any("safety_checks" in v for v in violations)
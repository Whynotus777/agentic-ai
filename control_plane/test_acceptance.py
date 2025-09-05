"""
Acceptance Criteria Tests for Control Plane
Demonstrates that all requirements are met.
"""
import json
from control_plane.policy_engine.engine import PolicyEngine, RequestContext, PolicyDecision
from control_plane.flags.store import FeatureFlagStore
from control_plane.cap_registry.registry import CapabilityRegistry
from control_plane.integrated_router import IntegratedRouter, RoutingRequest


def test_acceptance_criteria():
    """
    Test all acceptance criteria for the Control Plane.
    """
    print("=" * 60)
    print("CONTROL PLANE ACCEPTANCE CRITERIA TESTS")
    print("=" * 60)
    
    # ========================================================================
    # CRITERIA 1: Policy engine returns HITL_REQUIRED for repo.commit and robot.actuate
    # ========================================================================
    print("\n✅ TEST 1: HITL Required for Critical Capabilities")
    print("-" * 50)
    
    engine = PolicyEngine()
    
    # Test repo.commit
    ctx1 = RequestContext(
        tenant_id="test-tenant",
        service="vcs-service",
        capability="repo.commit",
        domain="development"
    )
    result1 = engine.evaluate(ctx1)
    print(f"repo.commit decision: {result1.decision.value}")
    print(f"Reason: {result1.reasons[0]}")
    assert result1.decision == PolicyDecision.HITL_REQUIRED, "repo.commit should require HITL"
    
    # Test robot.actuate
    ctx2 = RequestContext(
        tenant_id="robotics-tenant",
        service="robot-controller",
        capability="robot.actuate",
        domain="robotics"
    )
    result2 = engine.evaluate(ctx2)
    print(f"\nrobot.actuate decision: {result2.decision.value}")
    print(f"Reason: {result2.reasons[0]}")
    assert result2.decision == PolicyDecision.HITL_REQUIRED, "robot.actuate should require HITL"
    
    print("\n✅ CRITERIA 1 PASSED: HITL required for repo.commit and robot.actuate")
    
    # ========================================================================
    # CRITERIA 2: Feature flag toggles orchestrator from o4-mini to deepseek-v3
    # ========================================================================
    print("\n✅ TEST 2: Feature Flag Switches Orchestrator Model")
    print("-" * 50)
    
    router = IntegratedRouter()
    
    # Before toggling flag (should use o4-mini)
    request = RoutingRequest(
        tenant_id="test-tenant",
        service="orchestrator",
        domain="general",
        capability="llm.generate"
    )
    
    decision_before = router.route(request)
    print(f"Model BEFORE flag toggle: {decision_before.chosen_primary}")
    assert decision_before.chosen_primary == "o4-mini", "Should default to o4-mini"
    
    # Toggle the flag
    router.flag_store.update_flag("use_deepseek_v3", enabled=True, rollout_percentage=100.0)
    
    # After toggling flag (should use deepseek-v3)
    decision_after = router.route(request)
    print(f"Model AFTER flag toggle: {decision_after.chosen_primary}")
    assert decision_after.chosen_primary == "deepseek-v3", "Should switch to deepseek-v3"
    
    print("\n✅ CRITERIA 2 PASSED: Flag successfully switches orchestrator model")
    
    # ========================================================================
    # CRITERIA 3: Router returns JSON decision trace with policy+flag evidence
    # ========================================================================
    print("\n✅ TEST 3: Router Returns Complete Decision Trace")
    print("-" * 50)
    
    router = IntegratedRouter()
    
    request = RoutingRequest(
        tenant_id="sim-tenant",
        service="simulation-service",
        domain="simulation",
        capability="sim.run",
        environment="production",
        tier="standard"
    )
    
    decision = router.route(request)
    
    # Verify decision trace contains all required elements
    trace = decision.decision_trace
    print("Decision Trace contains:")
    print(f"  - Request ID: {trace.get('request_id', 'N/A')}")
    print(f"  - Policy evaluation: {trace.get('policy_evaluation', {}).get('decision', 'N/A')}")
    print(f"  - Flag evaluation: use_deepseek_v3={trace.get('flag_evaluation', {}).get('use_deepseek_v3', False)}")
    print(f"  - Model selection: {trace.get('model_selection', {}).get('primary', 'N/A')}")
    print(f"  - Selection reason: {trace.get('model_selection', {}).get('selection_reason', 'N/A')}")
    print(f"  - HITL required: {trace.get('hitl_required', False)}")
    
    assert "policy_evaluation" in trace, "Should include policy evaluation"
    assert "flag_evaluation" in trace, "Should include flag evaluation"
    assert "model_selection" in trace, "Should include model selection"
    assert "capability_validation" in trace, "Should include capability validation"
    
    print("\n✅ CRITERIA 3 PASSED: Decision trace includes policy+flag evidence")
    
    # ========================================================================
    # CRITERIA 4: Requests lacking declared capabilities are POLICY_BLOCKED
    # ========================================================================
    print("\n✅ TEST 4: Missing Capabilities are Blocked")
    print("-" * 50)
    
    router = IntegratedRouter()
    
    # Request with non-existent capability
    request = RoutingRequest(
        tenant_id="test-tenant",
        service="unknown-service",
        domain="general",
        capability="nonexistent.capability",
        environment="production"
    )
    
    decision = router.route(request)
    
    print(f"Decision for missing capability: {decision.chosen_primary}")
    print(f"Policy match: {decision.policy_match}")
    print(f"Blocked by capability: {decision.decision_trace.get('blocked_by_capability', False)}")
    
    assert decision.chosen_primary == "none", "Should not route when capability missing"
    assert decision.decision_trace.get("blocked_by_capability") == True, "Should be blocked by capability check"
    assert decision.decision_trace.get("missing_capability") == "nonexistent.capability", "Should identify missing capability"
    
    print("\n✅ CRITERIA 4 PASSED: Requests without capabilities are blocked")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 60)
    print("ALL ACCEPTANCE CRITERIA PASSED!")
    print("=" * 60)
    print("\n✅ Policy engine returns HITL_REQUIRED for repo.commit and robot.actuate")
    print("✅ Feature flag toggles orchestrator from o4-mini to deepseek-v3")
    print("✅ Router returns JSON decision trace with policy+flag evidence")
    print("✅ Requests lacking declared capabilities are POLICY_BLOCKED")
    print("\nControl Plane is ready for production deployment! 🚀")
    

if __name__ == "__main__":
    # Run acceptance tests
    test_acceptance()
    
    # Additional demo: Show a complete routing decision
    print("\n" + "=" * 60)
    print("DEMO: Complete Routing Decision for Robotics Domain")
    print("=" * 60)
    
    router = IntegratedRouter()
    
    # Enable RT-2 for robotics
    router.flag_store.update_flag("enable_rt2_robotics", enabled=True)
    
    request = RoutingRequest(
        tenant_id="robotics-tenant",
        service="robot-planner",
        domain="robotics",
        capability="robot.actuate",
        environment="production",
        tier="premium"
    )
    
    decision = router.route(request)
    
    print("\nRouting Decision:")
    print(json.dumps(decision.decision_trace, indent=2))
    
    print(f"\nPrimary Model: {decision.chosen_primary}")
    print(f"Fallbacks: {decision.fallbacks}")
    print(f"Requires HITL: {decision.requires_hitl}")
    print(f"Guardrails Applied: {len(decision.guardrails)}")
    
    if decision.guardrails:
        print("\nActive Guardrails:")
        for guardrail in decision.guardrails:
            print(f"  - {guardrail.get('type', 'unknown')}")
    
    print("\n✅ Demo complete - Control Plane is fully operational!")
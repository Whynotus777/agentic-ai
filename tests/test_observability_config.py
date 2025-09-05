# tests/test_observability_config.py
# NEW: Lints presence of cost_usd processor and alert rules with required labels

import pytest
import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Any


class TestObservabilityConfig:
    """Tests for observability configuration compliance"""
    
    @pytest.fixture
    def otel_config(self):
        """Load OTel collector configuration"""
        config_path = Path("observability/otel-collector.yaml")
        if not config_path.exists():
            pytest.skip(f"Config file {config_path} not found")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @pytest.fixture
    def alerts_config(self):
        """Load monitoring alerts configuration"""
        alerts_path = Path("monitoring/alerts.yaml")
        if not alerts_path.exists():
            pytest.skip(f"Alerts file {alerts_path} not found")
        
        with open(alerts_path, 'r') as f:
            return yaml.safe_load(f)
    
    def test_cost_usd_processor_exists(self, otel_config):
        """Verify cost_usd processor is configured in OTel collector"""
        # Check processors section exists
        assert "processors" in otel_config, "Missing processors section in OTel config"
        
        # Check for cost attribution processor
        assert "attributes/cost" in otel_config["processors"], \
            "Missing attributes/cost processor for cost_usd attribution"
        
        # Verify cost processor configuration
        cost_processor = otel_config["processors"]["attributes/cost"]
        assert "actions" in cost_processor, "Missing actions in cost processor"
        
        # Check for cost_usd attribute
        cost_usd_found = False
        for action in cost_processor["actions"]:
            if action.get("key") == "cost_usd":
                cost_usd_found = True
                assert "value" in action, "cost_usd action missing value"
                assert "action" in action, "cost_usd action missing action type"
                # Verify action is upsert or update
                assert action["action"] in ["upsert", "update", "insert"], \
                    f"Invalid action type for cost_usd: {action['action']}"
        
        assert cost_usd_found, "cost_usd attribute not found in processor actions"
    
    def test_cost_processor_in_pipeline(self, otel_config):
        """Verify cost processor is included in trace pipeline"""
        assert "service" in otel_config, "Missing service section"
        assert "pipelines" in otel_config["service"], "Missing pipelines section"
        assert "traces" in otel_config["service"]["pipelines"], "Missing traces pipeline"
        
        trace_pipeline = otel_config["service"]["pipelines"]["traces"]
        assert "processors" in trace_pipeline, "Missing processors in trace pipeline"
        
        # Check that attributes/cost is in the processors list
        assert "attributes/cost" in trace_pipeline["processors"], \
            "attributes/cost processor not included in trace pipeline"
    
    def test_siem_exporter_exists(self, otel_config):
        """Verify SIEM exporter stub is configured"""
        assert "exporters" in otel_config, "Missing exporters section"
        
        # Check for SIEM exporter
        assert "otlp/siem" in otel_config["exporters"], \
            "Missing otlp/siem exporter for SIEM integration"
        
        siem_exporter = otel_config["exporters"]["otlp/siem"]
        assert "endpoint" in siem_exporter, "SIEM exporter missing endpoint"
        assert "siem" in siem_exporter["endpoint"].lower(), \
            "SIEM exporter endpoint doesn't indicate SIEM destination"
    
    def test_siem_security_pipeline(self, otel_config):
        """Verify security logs pipeline routes to SIEM"""
        pipelines = otel_config["service"]["pipelines"]
        
        # Check for security-specific pipeline
        assert "logs/security" in pipelines, \
            "Missing logs/security pipeline for SIEM events"
        
        security_pipeline = pipelines["logs/security"]
        assert "exporters" in security_pipeline, "Security pipeline missing exporters"
        assert "otlp/siem" in security_pipeline["exporters"], \
            "Security pipeline not routing to SIEM exporter"
    
    def test_alert_required_labels(self, alerts_config):
        """Verify all alerts have required labels"""
        required_labels = ["severity", "owner", "paging_policy"]
        service_labels = ["service", "capability", "tenant"]  # At least one required
        
        assert "groups" in alerts_config, "Missing alert groups"
        
        for group in alerts_config["groups"]:
            assert "rules" in group, f"Alert group {group.get('name')} missing rules"
            
            for rule in group["rules"]:
                alert_name = rule.get("alert", "unnamed")
                
                # Check required labels
                assert "labels" in rule, f"Alert {alert_name} missing labels"
                labels = rule["labels"]
                
                for required_label in required_labels:
                    assert required_label in labels, \
                        f"Alert {alert_name} missing required label: {required_label}"
                
                # Verify owner is an email
                assert "@" in labels["owner"], \
                    f"Alert {alert_name} owner should be an email address"
                
                # Verify paging_policy is valid
                valid_policies = ["P0", "P1", "P2", "P3"]
                assert labels["paging_policy"] in valid_policies, \
                    f"Alert {alert_name} has invalid paging_policy: {labels['paging_policy']}"
                
                # Check for at least one service label (can be template variable)
                has_service_label = any(
                    label in labels or f"{{{{ $labels.{label} }}}}" in str(labels)
                    for label in service_labels
                )
                assert has_service_label, \
                    f"Alert {alert_name} missing service/capability/tenant label"
    
    def test_specific_alerts_exist(self, alerts_config):
        """Verify specific required alerts are configured"""
        required_alerts = {
            "HighToolErrorRate": {
                "description": "Tool error rate > 5% / 10m",
                "check_expr": lambda expr: "0.05" in expr and "tool" in expr.lower()
            },
            "DailyCostAnomaly": {
                "description": "Daily cost > 2x 7d avg / 30m",
                "check_expr": lambda expr: "2 *" in expr and "7d" in expr and "cost_usd" in expr
            }
        }
        
        # Also check for at least one SLO breach alert
        slo_alert_patterns = ["SLOBreach", "StageSLO", "stage_errors"]
        
        found_alerts = set()
        found_slo = False
        
        for group in alerts_config["groups"]:
            for rule in group["rules"]:
                alert_name = rule.get("alert", "")
                found_alerts.add(alert_name)
                
                # Check for SLO alerts
                if any(pattern in alert_name for pattern in slo_alert_patterns):
                    found_slo = True
        
        # Verify required alerts exist
        for alert_name, config in required_alerts.items():
            assert alert_name in found_alerts, \
                f"Missing required alert: {alert_name} ({config['description']})"
            
            # Find and validate the alert expression
            for group in alerts_config["groups"]:
                for rule in group["rules"]:
                    if rule.get("alert") == alert_name:
                        expr = rule.get("expr", "")
                        assert config["check_expr"](expr), \
                            f"Alert {alert_name} expression doesn't match requirements"
        
        assert found_slo, "No SLO breach alerts found (need at least one stage SLO alert)"
    
    def test_alert_annotations(self, alerts_config):
        """Verify alerts have proper annotations"""
        required_annotations = ["summary", "description"]
        
        for group in alerts_config["groups"]:
            for rule in group["rules"]:
                alert_name = rule.get("alert", "unnamed")
                
                assert "annotations" in rule, \
                    f"Alert {alert_name} missing annotations"
                
                annotations = rule["annotations"]
                for required_annotation in required_annotations:
                    assert required_annotation in annotations, \
                        f"Alert {alert_name} missing annotation: {required_annotation}"
                
                # Check that annotations use template variables properly
                if "{{" in annotations["summary"]:
                    assert "}}" in annotations["summary"], \
                        f"Alert {alert_name} has malformed template in summary"
    
    def test_cost_monitoring_alerts(self, alerts_config):
        """Verify cost monitoring alerts are properly configured"""
        cost_alerts_found = []
        
        for group in alerts_config["groups"]:
            if "cost" in group.get("name", "").lower():
                for rule in group["rules"]:
                    alert_name = rule.get("alert", "")
                    expr = rule.get("expr", "")
                    
                    # Check for cost_usd metric
                    if "cost_usd" in expr:
                        cost_alerts_found.append(alert_name)
                        
                        # Verify it has proper labels
                        labels = rule.get("labels", {})
                        assert "owner" in labels, \
                            f"Cost alert {alert_name} missing owner"
                        
                        # Cost alerts should typically be lower priority
                        assert labels.get("paging_policy") in ["P2", "P3"], \
                            f"Cost alert {alert_name} has unexpectedly high priority"
        
        assert len(cost_alerts_found) >= 2, \
            f"Need at least 2 cost alerts, found: {cost_alerts_found}"
    
    def test_security_alerts_siem_forward(self, alerts_config):
        """Verify security alerts are marked for SIEM forwarding"""
        security_keywords = ["bypass", "unauthorized", "security", "policy_violation"]
        
        for group in alerts_config["groups"]:
            for rule in group["rules"]:
                alert_name = rule.get("alert", "")
                
                # Check if this appears to be a security alert
                is_security_alert = any(
                    keyword in alert_name.lower() 
                    for keyword in security_keywords
                )
                
                if is_security_alert:
                    labels = rule.get("labels", {})
                    
                    # Should have high severity
                    assert labels.get("severity") in ["critical", "warning"], \
                        f"Security alert {alert_name} should have high severity"
                    
                    # Should be marked for SIEM forwarding
                    if labels.get("severity") == "critical":
                        assert labels.get("siem_forward") == "true", \
                            f"Critical security alert {alert_name} should forward to SIEM"
    
    @pytest.mark.integration
    def test_config_files_valid_yaml(self):
        """Verify all config files are valid YAML"""
        config_files = [
            "observability/otel-collector.yaml",
            "monitoring/alerts.yaml"
        ]
        
        for config_file in config_files:
            if not Path(config_file).exists():
                continue
                
            try:
                with open(config_file, 'r') as f:
                    yaml.safe_load(f)
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML in {config_file}: {e}")
    
    def test_trace_propagation_documented(self):
        """Verify trace propagation is documented"""
        tracing_doc = Path("observability/tracing.md")
        
        if not tracing_doc.exists():
            pytest.skip("Tracing documentation not found")
        
        with open(tracing_doc, 'r') as f:
            content = f.read().lower()
        
        # Check for required documentation elements
        required_elements = [
            "trace_id",
            "cost_usd",
            "orch",
            "queue",
            "tool",
            "propagat",  # propagation/propagate
            "w3c",  # W3C trace context
        ]
        
        for element in required_elements:
            assert element in content, \
                f"Tracing documentation missing coverage of: {element}"
    
    def test_decision_trace_documented(self):
        """Verify decision trace examples exist"""
        readme_path = Path("control_plane/README.md")
        
        if not readme_path.exists():
            pytest.skip("Control plane README not found")
        
        with open(readme_path, 'r') as f:
            content = f.read()
        
        # Check for decision trace elements
        required_elements = [
            "decision_trace",
            "policy_match",
            "flag_overrides",
            "chosen_primary",
            "fallback",
            "guardrails"
        ]
        
        for element in required_elements:
            assert element in content, \
                f"Control plane README missing decision trace element: {element}"
        
        # Verify JSON examples are present
        assert "```json" in content, "README should contain JSON examples"
        
        # Check for realistic examples
        assert '"trace_id"' in content, "Decision trace examples should include trace_id"
        assert '"cost_usd"' in content, "Decision trace examples should include cost_usd"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
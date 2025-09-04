# observability/siem_integration.py
"""
SIEM (Security Information and Event Management) and SOC (Security Operations Center)
integration for comprehensive security monitoring and incident response.
"""

import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
import re
from collections import defaultdict

from opentelemetry import trace
tracer = trace.get_tracer(__name__)


class SecurityEventType(Enum):
    """Types of security events"""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_VIOLATION = "authz_violation"
    DATA_EXFILTRATION_ATTEMPT = "data_exfil"
    PRIVILEGE_ESCALATION = "priv_escalation"
    ANOMALOUS_ACCESS = "anomalous_access"
    POLICY_BYPASS = "policy_bypass"
    INJECTION_ATTEMPT = "injection"
    DOS_ATTACK = "dos_attack"
    MALWARE_DETECTED = "malware"
    SUSPICIOUS_NETWORK = "suspicious_network"
    COMPLIANCE_VIOLATION = "compliance_violation"
    CONFIGURATION_CHANGE = "config_change"


class SeverityLevel(Enum):
    """Security event severity levels"""
    INFO = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


class MITREAttackTactic(Enum):
    """MITRE ATT&CK Framework Tactics"""
    RECONNAISSANCE = "TA0043"
    RESOURCE_DEVELOPMENT = "TA0042"
    INITIAL_ACCESS = "TA0001"
    EXECUTION = "TA0002"
    PERSISTENCE = "TA0003"
    PRIVILEGE_ESCALATION = "TA0004"
    DEFENSE_EVASION = "TA0005"
    CREDENTIAL_ACCESS = "TA0006"
    DISCOVERY = "TA0007"
    LATERAL_MOVEMENT = "TA0008"
    COLLECTION = "TA0009"
    COMMAND_AND_CONTROL = "TA0011"
    EXFILTRATION = "TA0010"
    IMPACT = "TA0040"


@dataclass
class SecurityEvent:
    """Security event for SIEM"""
    event_id: str
    timestamp: datetime
    event_type: SecurityEventType
    severity: SeverityLevel
    source_ip: str
    source_user: str
    target_resource: str
    tenant_id: str
    description: str
    raw_data: Dict[str, Any]
    mitre_mapping: Optional[List[str]] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)
    remediation_taken: List[str] = field(default_factory=list)
    trace_id: Optional[str] = None
    correlation_id: Optional[str] = None


@dataclass
class SecurityIncident:
    """Security incident (collection of related events)"""
    incident_id: str
    created_at: datetime
    updated_at: datetime
    severity: SeverityLevel
    status: str  # open, investigating, contained, resolved
    events: List[SecurityEvent]
    affected_resources: Set[str]
    affected_users: Set[str]
    mitre_techniques: Set[str]
    response_actions: List[Dict[str, Any]]
    assigned_to: Optional[str] = None
    resolution: Optional[str] = None


@dataclass
class ThreatIndicator:
    """Indicator of Compromise (IoC)"""
    indicator_id: str
    indicator_type: str  # ip, domain, hash, email, url
    value: str
    threat_type: str
    confidence: float  # 0-1
    first_seen: datetime
    last_seen: datetime
    sources: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SIEMConnector:
    """
    SIEM connector for security event management and incident response
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.events_queue = asyncio.Queue(maxsize=10000)
        self.incidents: Dict[str, SecurityIncident] = {}
        self.threat_indicators: Dict[str, ThreatIndicator] = {}
        self.correlation_rules = self._init_correlation_rules()
        self.mitre_mappings = self._init_mitre_mappings()
        self.event_buffer: List[SecurityEvent] = []
        self.metrics = defaultdict(int)
        
    def _init_correlation_rules(self) -> List[Dict[str, Any]]:
        """Initialize event correlation rules"""
        return [
            {
                "name": "brute_force_detection",
                "conditions": {
                    "event_type": SecurityEventType.AUTHENTICATION_FAILURE,
                    "threshold": 5,
                    "time_window": 300,  # 5 minutes
                    "group_by": ["source_ip", "target_resource"]
                },
                "severity": SeverityLevel.HIGH,
                "response": ["block_ip", "notify_soc"]
            },
            {
                "name": "data_exfiltration_pattern",
                "conditions": {
                    "event_types": [
                        SecurityEventType.ANOMALOUS_ACCESS,
                        SecurityEventType.DATA_EXFILTRATION_ATTEMPT
                    ],
                    "sequence": True,
                    "time_window": 3600  # 1 hour
                },
                "severity": SeverityLevel.CRITICAL,
                "response": ["isolate_user", "capture_traffic", "notify_soc"]
            },
            {
                "name": "privilege_escalation_chain",
                "conditions": {
                    "event_types": [
                        SecurityEventType.AUTHORIZATION_VIOLATION,
                        SecurityEventType.PRIVILEGE_ESCALATION
                    ],
                    "threshold": 3,
                    "time_window": 1800  # 30 minutes
                },
                "severity": SeverityLevel.CRITICAL,
                "response": ["revoke_access", "force_mfa", "notify_soc"]
            }
        ]
    
    def _init_mitre_mappings(self) -> Dict[SecurityEventType, List[str]]:
        """Map security events to MITRE ATT&CK techniques"""
        return {
            SecurityEventType.AUTHENTICATION_FAILURE: [
                "T1110",  # Brute Force
                "T1078"   # Valid Accounts
            ],
            SecurityEventType.PRIVILEGE_ESCALATION: [
                "T1068",  # Exploitation for Privilege Escalation
                "T1078.003"  # Valid Accounts: Local Accounts
            ],
            SecurityEventType.DATA_EXFILTRATION_ATTEMPT: [
                "T1041",  # Exfiltration Over C2 Channel
                "T1567"   # Exfiltration Over Web Service
            ],
            SecurityEventType.INJECTION_ATTEMPT: [
                "T1190",  # Exploit Public-Facing Application
                "T1055"   # Process Injection
            ],
            SecurityEventType.MALWARE_DETECTED: [
                "T1105",  # Ingress Tool Transfer
                "T1204"   # User Execution
            ],
            SecurityEventType.POLICY_BYPASS: [
                "T1562",  # Impair Defenses
                "T1484"   # Domain Policy Modification
            ]
        }
    
    @tracer.start_as_current_span("ingest_security_event")
    async def ingest_event(
        self,
        event_type: SecurityEventType,
        severity: SeverityLevel,
        source_ip: str,
        source_user: str,
        target_resource: str,
        tenant_id: str,
        description: str,
        raw_data: Dict[str, Any],
        trace_id: Optional[str] = None
    ) -> str:
        """
        Ingest a security event into SIEM
        
        Returns:
            Event ID
        """
        span = trace.get_current_span()
        
        # Create event
        event_id = f"sec-{hashlib.sha256(f'{datetime.utcnow()}{source_ip}{target_resource}'.encode()).hexdigest()[:12]}"
        
        # Get MITRE mappings
        mitre_techniques = self.mitre_mappings.get(event_type, [])
        
        event = SecurityEvent(
            event_id=event_id,
            timestamp=datetime.utcnow(),
            event_type=event_type,
            severity=severity,
            source_ip=source_ip,
            source_user=source_user,
            target_resource=target_resource,
            tenant_id=tenant_id,
            description=description,
            raw_data=raw_data,
            mitre_mapping=mitre_techniques,
            trace_id=trace_id or format(span.get_span_context().trace_id, '032x')
        )
        
        # Extract indicators
        event.indicators = await self._extract_indicators(event)
        
        # Add to queue
        await self.events_queue.put(event)
        
        # Update metrics
        self.metrics[f"{event_type.value}_count"] += 1
        self.metrics[f"severity_{severity.value}_count"] += 1
        
        # Check for immediate threats
        if severity == SeverityLevel.CRITICAL:
            await self._handle_critical_event(event)
        
        span.set_attributes({
            "siem.event_id": event_id,
            "siem.event_type": event_type.value,
            "siem.severity": severity.value,
            "siem.mitre_techniques": ",".join(mitre_techniques)
        })
        
        return event_id
    
    async def _extract_indicators(self, event: SecurityEvent) -> List[str]:
        """Extract indicators of compromise from event"""
        indicators = []
        
        # Extract IPs
        ip_pattern = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
        for ip in ip_pattern.findall(str(event.raw_data)):
            indicators.append(f"ip:{ip}")
        
        # Extract domains
        domain_pattern = re.compile(r'\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9][a-z0-9-]{0,61}[a-z0-9]\b')
        for domain in domain_pattern.findall(str(event.raw_data)):
            indicators.append(f"domain:{domain}")
        
        # Extract file hashes
        hash_pattern = re.compile(r'\b[a-fA-F0-9]{32,64}\b')
        for hash_val in hash_pattern.findall(str(event.raw_data)):
            if len(hash_val) == 32:
                indicators.append(f"md5:{hash_val}")
            elif len(hash_val) == 64:
                indicators.append(f"sha256:{hash_val}")
        
        return indicators
    
    async def _handle_critical_event(self, event: SecurityEvent):
        """Handle critical security events immediately"""
        # Create incident
        incident = await self._create_incident([event], SeverityLevel.CRITICAL)
        
        # Trigger immediate response
        await self._trigger_incident_response(incident)
        
        # Send alert
        await self._send_critical_alert(event)
    
    async def correlate_events(self):
        """Correlate events to detect patterns and create incidents"""
        while True:
            try:
                # Collect events from queue
                batch = []
                timeout = 10  # seconds
                
                try:
                    while len(batch) < 100:  # Process up to 100 events
                        event = await asyncio.wait_for(
                            self.events_queue.get(),
                            timeout=timeout
                        )
                        batch.append(event)
                        self.event_buffer.append(event)
                except asyncio.TimeoutError:
                    pass
                
                if not batch:
                    continue
                
                # Apply correlation rules
                for rule in self.correlation_rules:
                    incidents = await self._apply_correlation_rule(rule, batch)
                    
                    for incident in incidents:
                        self.incidents[incident.incident_id] = incident
                        await self._trigger_incident_response(incident)
                
                # Clean old events from buffer
                cutoff = datetime.utcnow() - timedelta(hours=1)
                self.event_buffer = [
                    e for e in self.event_buffer
                    if e.timestamp > cutoff
                ]
                
            except Exception as e:
                print(f"Event correlation error: {e}")
                await asyncio.sleep(1)
    
    async def _apply_correlation_rule(
        self,
        rule: Dict[str, Any],
        events: List[SecurityEvent]
    ) -> List[SecurityIncident]:
        """Apply a correlation rule to events"""
        incidents = []
        conditions = rule["conditions"]
        
        if "event_type" in conditions:
            # Single event type correlation
            event_type = conditions["event_type"]
            relevant_events = [e for e in events if e.event_type == event_type]
            
            if "threshold" in conditions:
                # Group events
                groups = defaultdict(list)
                for event in relevant_events:
                    key = tuple(event.source_ip if k == "source_ip" else 
                               event.target_resource if k == "target_resource" else
                               getattr(event, k)
                               for k in conditions.get("group_by", ["source_ip"]))
                    groups[key].append(event)
                
                # Check threshold
                for key, group_events in groups.items():
                    if len(group_events) >= conditions["threshold"]:
                        incident = await self._create_incident(
                            group_events,
                            rule["severity"]
                        )
                        incident.response_actions = [
                            {"action": action, "status": "pending"}
                            for action in rule.get("response", [])
                        ]
                        incidents.append(incident)
        
        elif "event_types" in conditions:
            # Multiple event type correlation
            if conditions.get("sequence"):
                # Look for sequence of events
                sequences = await self._find_event_sequences(
                    events,
                    conditions["event_types"],
                    conditions.get("time_window", 3600)
                )
                
                for seq in sequences:
                    incident = await self._create_incident(seq, rule["severity"])
                    incident.response_actions = [
                        {"action": action, "status": "pending"}
                        for action in rule.get("response", [])
                    ]
                    incidents.append(incident)
        
        return incidents
    
    async def _find_event_sequences(
        self,
        events: List[SecurityEvent],
        event_types: List[SecurityEventType],
        time_window: int
    ) -> List[List[SecurityEvent]]:
        """Find sequences of events matching pattern"""
        sequences = []
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Look for sequences
        for i, start_event in enumerate(sorted_events):
            if start_event.event_type == event_types[0]:
                sequence = [start_event]
                seq_index = 1
                
                for j in range(i + 1, len(sorted_events)):
                    if seq_index >= len(event_types):
                        break
                    
                    next_event = sorted_events[j]
                    time_diff = (next_event.timestamp - start_event.timestamp).seconds
                    
                    if time_diff > time_window:
                        break
                    
                    if next_event.event_type == event_types[seq_index]:
                        sequence.append(next_event)
                        seq_index += 1
                
                if len(sequence) == len(event_types):
                    sequences.append(sequence)
        
        return sequences
    
    async def _create_incident(
        self,
        events: List[SecurityEvent],
        severity: SeverityLevel
    ) -> SecurityIncident:
        """Create security incident from events"""
        incident_id = f"inc-{hashlib.sha256(f'{datetime.utcnow()}{events[0].event_id}'.encode()).hexdigest()[:12]}"
        
        incident = SecurityIncident(
            incident_id=incident_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            severity=severity,
            status="open",
            events=events,
            affected_resources=set(e.target_resource for e in events),
            affected_users=set(e.source_user for e in events),
            mitre_techniques=set(t for e in events for t in e.mitre_mapping),
            response_actions=[]
        )
        
        return incident
    
    async def _trigger_incident_response(self, incident: SecurityIncident):
        """Trigger automated incident response"""
        for action in incident.response_actions:
            if action["status"] == "pending":
                try:
                    await self._execute_response_action(
                        action["action"],
                        incident
                    )
                    action["status"] = "completed"
                    action["completed_at"] = datetime.utcnow().isoformat()
                except Exception as e:
                    action["status"] = "failed"
                    action["error"] = str(e)
    
    async def _execute_response_action(
        self,
        action: str,
        incident: SecurityIncident
    ):
        """Execute a specific response action"""
        if action == "block_ip":
            for event in incident.events:
                await self._block_ip_address(event.source_ip)
        
        elif action == "isolate_user":
            for user in incident.affected_users:
                await self._isolate_user_account(user)
        
        elif action == "revoke_access":
            for resource in incident.affected_resources:
                await self._revoke_resource_access(resource)
        
        elif action == "notify_soc":
            await self._notify_soc_team(incident)
        
        elif action == "capture_traffic":
            await self._start_packet_capture(incident)
        
        elif action == "force_mfa":
            for user in incident.affected_users:
                await self._force_mfa_reset(user)
    
    async def _block_ip_address(self, ip: str):
        """Block IP address at firewall"""
        print(f"Blocking IP: {ip}")
        # In production, integrate with firewall API
    
    async def _isolate_user_account(self, user: str):
        """Isolate user account"""
        print(f"Isolating user: {user}")
        # In production, disable account in identity provider
    
    async def _revoke_resource_access(self, resource: str):
        """Revoke access to resource"""
        print(f"Revoking access to: {resource}")
        # In production, update IAM policies
    
    async def _notify_soc_team(self, incident: SecurityIncident):
        """Notify SOC team of incident"""
        print(f"Notifying SOC of incident: {incident.incident_id}")
        # In production, send to ticketing system / Slack / PagerDuty
    
    async def _start_packet_capture(self, incident: SecurityIncident):
        """Start packet capture for forensics"""
        print(f"Starting packet capture for incident: {incident.incident_id}")
        # In production, trigger network tap
    
    async def _force_mfa_reset(self, user: str):
        """Force MFA reset for user"""
        print(f"Forcing MFA reset for: {user}")
        # In production, trigger MFA reset in identity provider
    
    async def _send_critical_alert(self, event: SecurityEvent):
        """Send critical security alert"""
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_id": event.event_id,
            "severity": "CRITICAL",
            "type": event.event_type.value,
            "description": event.description,
            "affected_resource": event.target_resource,
            "source": event.source_ip,
            "user": event.source_user,
            "mitre_techniques": event.mitre_mapping,
            "action_required": "Immediate investigation required"
        }
        
        # In production, send to alerting system
        print(f"CRITICAL ALERT: {json.dumps(alert, indent=2)}")
    
    def format_cef_event(self, event: SecurityEvent) -> str:
        """Format event in Common Event Format for SIEM"""
        cef_header = f"CEF:0|AgenticAI|SecurityMonitor|1.0|{event.event_type.value}|{event.description}|{event.severity.value}|"
        
        extensions = [
            f"rt={int(event.timestamp.timestamp() * 1000)}",
            f"src={event.source_ip}",
            f"suser={event.source_user}",
            f"dst={event.target_resource}",
            f"cs1Label=TenantID",
            f"cs1={event.tenant_id}",
            f"cs2Label=EventID",
            f"cs2={event.event_id}",
            f"cs3Label=TraceID",
            f"cs3={event.trace_id or 'none'}",
            f"cs4Label=MITRETechniques",
            f"cs4={','.join(event.mitre_mapping)}"
        ]
        
        return cef_header + " ".join(extensions)
    
    def format_leef_event(self, event: SecurityEvent) -> str:
        """Format event in Log Event Extended Format"""
        leef_header = f"LEEF:1.0|AgenticAI|SecurityMonitor|1.0|{event.event_type.value}|"
        
        attributes = [
            f"devTime={event.timestamp.isoformat()}",
            f"severity={event.severity.value}",
            f"src={event.source_ip}",
            f"usrName={event.source_user}",
            f"dst={event.target_resource}",
            f"tenantId={event.tenant_id}",
            f"eventId={event.event_id}",
            f"description={event.description}"
        ]
        
        return leef_header + "|".join(attributes)
    
    async def export_to_siem(
        self,
        format_type: str = "cef"
    ) -> List[str]:
        """Export events to external SIEM"""
        formatted_events = []
        
        for event in self.event_buffer:
            if format_type == "cef":
                formatted = self.format_cef_event(event)
            elif format_type == "leef":
                formatted = self.format_leef_event(event)
            else:
                formatted = json.dumps(event.__dict__, default=str)
            
            formatted_events.append(formatted)
        
        return formatted_events
    
    async def threat_hunt(
        self,
        indicators: List[str],
        time_range: Tuple[datetime, datetime]
    ) -> List[SecurityEvent]:
        """Hunt for threats using indicators"""
        matches = []
        
        for event in self.event_buffer:
            if not (time_range[0] <= event.timestamp <= time_range[1]):
                continue
            
            for indicator in indicators:
                if indicator in event.indicators:
                    matches.append(event)
                    break
                
                # Check in raw data
                if indicator in str(event.raw_data):
                    matches.append(event)
                    break
        
        return matches
    
    async def generate_incident_report(
        self,
        incident_id: str
    ) -> Dict[str, Any]:
        """Generate incident report"""
        if incident_id not in self.incidents:
            return {"error": "Incident not found"}
        
        incident = self.incidents[incident_id]
        
        return {
            "incident_id": incident.incident_id,
            "severity": incident.severity.value,
            "status": incident.status,
            "created_at": incident.created_at.isoformat(),
            "updated_at": incident.updated_at.isoformat(),
            "event_count": len(incident.events),
            "affected_resources": list(incident.affected_resources),
            "affected_users": list(incident.affected_users),
            "mitre_techniques": list(incident.mitre_techniques),
            "timeline": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "type": e.event_type.value,
                    "description": e.description
                }
                for e in sorted(incident.events, key=lambda x: x.timestamp)
            ],
            "response_actions": incident.response_actions,
            "resolution": incident.resolution
        }
    
    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics for dashboard"""
        now = datetime.utcnow()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        # Calculate metrics
        hour_events = [e for e in self.event_buffer if e.timestamp > last_hour]
        day_events = [e for e in self.event_buffer if e.timestamp > last_day]
        
        return {
            "total_events": len(self.event_buffer),
            "events_last_hour": len(hour_events),
            "events_last_day": len(day_events),
            "active_incidents": sum(1 for i in self.incidents.values() if i.status != "resolved"),
            "critical_events": self.metrics.get("severity_5_count", 0),
            "top_event_types": sorted(
                [(k.replace("_count", ""), v) for k, v in self.metrics.items() if k.endswith("_count")],
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "mitre_coverage": len(set(t for i in self.incidents.values() for t in i.mitre_techniques))
        }


# Example usage
async def main():
    config = {}
    siem = SIEMConnector(config)
    
    # Start correlation engine
    asyncio.create_task(siem.correlate_events())
    
    # Ingest some events
    event_id1 = await siem.ingest_event(
        event_type=SecurityEventType.AUTHENTICATION_FAILURE,
        severity=SeverityLevel.MEDIUM,
        source_ip="192.168.1.100",
        source_user="attacker@example.com",
        target_resource="api.example.com",
        tenant_id="tenant-123",
        description="Failed login attempt",
        raw_data={"attempts": 5}
    )
    
    print(f"Ingested event: {event_id1}")
    
    # Simulate multiple failures to trigger correlation
    for i in range(5):
        await siem.ingest_event(
            event_type=SecurityEventType.AUTHENTICATION_FAILURE,
            severity=SeverityLevel.MEDIUM,
            source_ip="192.168.1.100",
            source_user="attacker@example.com",
            target_resource="api.example.com",
            tenant_id="tenant-123",
            description=f"Failed login attempt {i+2}",
            raw_data={"attempt": i+2}
        )
    
    # Wait for correlation
    await asyncio.sleep(2)
    
    # Export to SIEM
    cef_events = await siem.export_to_siem("cef")
    print(f"CEF Events: {cef_events[0] if cef_events else 'None'}")
    
    # Get metrics
    metrics = await siem.get_security_metrics()
    print(f"Security Metrics: {json.dumps(metrics, indent=2)}")
    
    # Generate incident report if any
    if siem.incidents:
        incident_id = list(siem.incidents.keys())[0]
        report = await siem.generate_incident_report(incident_id)
        print(f"Incident Report: {json.dumps(report, indent=2, default=str)}")


if __name__ == "__main__":
    asyncio.run(main()
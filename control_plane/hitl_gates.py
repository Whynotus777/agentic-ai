# control_plane/hitl_gates.py
"""
Human-in-the-Loop (HITL) gates and approval workflows for critical operations.
Implements multi-level approval chains with escalation and timeout handling.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
import hashlib
from collections import defaultdict

from opentelemetry import trace
tracer = trace.get_tracer(__name__)


class ApprovalStatus(Enum):
    """Approval request status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ESCALATED = "escalated"
    AUTO_APPROVED = "auto_approved"
    AUTO_REJECTED = "auto_rejected"


class ApprovalPriority(Enum):
    """Approval priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class OperationType(Enum):
    """Types of operations requiring approval"""
    MODEL_DEPLOYMENT = "model_deployment"
    DATA_ACCESS = "data_access"
    COST_THRESHOLD = "cost_threshold"
    POLICY_CHANGE = "policy_change"
    SECURITY_EXCEPTION = "security_exception"
    PRODUCTION_CHANGE = "production_change"
    DATA_DELETION = "data_deletion"
    PRIVILEGE_ESCALATION = "privilege_escalation"


@dataclass
class ApprovalRequest:
    """Approval request"""
    request_id: str
    operation_type: OperationType
    operation_details: Dict[str, Any]
    requester_id: str
    tenant_id: str
    priority: ApprovalPriority
    status: ApprovalStatus
    created_at: datetime
    expires_at: datetime
    context: Dict[str, Any]
    risk_score: float
    estimated_cost: float
    approval_chain: List[str]  # List of approver IDs
    current_approver_index: int = 0
    approvals: List[Dict[str, Any]] = field(default_factory=list)
    comments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None


@dataclass
class ApprovalPolicy:
    """Policy for approval requirements"""
    operation_type: OperationType
    min_approvers: int
    max_auto_approve_cost: float
    max_auto_approve_risk: float
    required_approver_roles: List[str]
    escalation_timeout: int  # seconds
    approval_timeout: int  # seconds
    auto_approve_conditions: Dict[str, Any]
    auto_reject_conditions: Dict[str, Any]


@dataclass
class Approver:
    """Approver information"""
    approver_id: str
    name: str
    email: str
    roles: List[str]
    approval_limit: float  # Max cost they can approve
    available: bool = True
    notification_preferences: Dict[str, Any] = field(default_factory=dict)


class HITLGateManager:
    """
    Manages Human-in-the-Loop approval gates
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pending_requests: Dict[str, ApprovalRequest] = {}
        self.completed_requests: Dict[str, ApprovalRequest] = {}
        self.approvers: Dict[str, Approver] = {}
        self.policies: Dict[OperationType, ApprovalPolicy] = {}
        self.notification_handlers: List[Callable] = []
        self.metrics = defaultdict(int)
        self._init_default_policies()
        self._init_default_approvers()
        
    def _init_default_policies(self):
        """Initialize default approval policies"""
        self.policies = {
            OperationType.MODEL_DEPLOYMENT: ApprovalPolicy(
                operation_type=OperationType.MODEL_DEPLOYMENT,
                min_approvers=2,
                max_auto_approve_cost=100.0,
                max_auto_approve_risk=0.3,
                required_approver_roles=["ml_engineer", "team_lead"],
                escalation_timeout=3600,
                approval_timeout=86400,
                auto_approve_conditions={"environment": "dev"},
                auto_reject_conditions={"blacklisted_models": ["experimental-v0"]}
            ),
            OperationType.DATA_ACCESS: ApprovalPolicy(
                operation_type=OperationType.DATA_ACCESS,
                min_approvers=1,
                max_auto_approve_cost=0.0,
                max_auto_approve_risk=0.1,
                required_approver_roles=["data_steward"],
                escalation_timeout=1800,
                approval_timeout=7200,
                auto_approve_conditions={"data_classification": "public"},
                auto_reject_conditions={"data_classification": "restricted"}
            ),
            OperationType.COST_THRESHOLD: ApprovalPolicy(
                operation_type=OperationType.COST_THRESHOLD,
                min_approvers=1,
                max_auto_approve_cost=1000.0,
                max_auto_approve_risk=0.5,
                required_approver_roles=["finance", "manager"],
                escalation_timeout=1800,
                approval_timeout=3600,
                auto_approve_conditions={"pre_approved_budget": True},
                auto_reject_conditions={"cost_exceeds": 10000.0}
            ),
            OperationType.PRODUCTION_CHANGE: ApprovalPolicy(
                operation_type=OperationType.PRODUCTION_CHANGE,
                min_approvers=2,
                max_auto_approve_cost=0.0,
                max_auto_approve_risk=0.0,
                required_approver_roles=["sre", "team_lead"],
                escalation_timeout=1800,
                approval_timeout=7200,
                auto_approve_conditions={},
                auto_reject_conditions={"change_freeze": True}
            ),
            OperationType.DATA_DELETION: ApprovalPolicy(
                operation_type=OperationType.DATA_DELETION,
                min_approvers=2,
                max_auto_approve_cost=0.0,
                max_auto_approve_risk=0.0,
                required_approver_roles=["data_steward", "compliance"],
                escalation_timeout=3600,
                approval_timeout=86400,
                auto_approve_conditions={},
                auto_reject_conditions={"protected_data": True}
            )
        }
    
    def _init_default_approvers(self):
        """Initialize default approvers"""
        default_approvers = [
            Approver(
                approver_id="approver-001",
                name="Alice Johnson",
                email="alice@example.com",
                roles=["ml_engineer", "team_lead"],
                approval_limit=5000.0
            ),
            Approver(
                approver_id="approver-002",
                name="Bob Smith",
                email="bob@example.com",
                roles=["data_steward", "compliance"],
                approval_limit=10000.0
            ),
            Approver(
                approver_id="approver-003",
                name="Carol White",
                email="carol@example.com",
                roles=["sre", "manager"],
                approval_limit=25000.0
            ),
            Approver(
                approver_id="approver-004",
                name="David Brown",
                email="david@example.com",
                roles=["finance", "executive"],
                approval_limit=100000.0
            )
        ]
        
        for approver in default_approvers:
            self.approvers[approver.approver_id] = approver
    
    @tracer.start_as_current_span("create_approval_request")
    async def create_approval_request(
        self,
        operation_type: OperationType,
        operation_details: Dict[str, Any],
        requester_id: str,
        tenant_id: str,
        priority: ApprovalPriority = ApprovalPriority.MEDIUM,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create an approval request
        
        Returns:
            Request ID
        """
        span = trace.get_current_span()
        
        # Generate request ID
        request_id = f"apr-{uuid.uuid4().hex[:12]}"
        
        # Get policy
        policy = self.policies.get(operation_type)
        if not policy:
            raise ValueError(f"No policy defined for {operation_type}")
        
        # Calculate risk score
        risk_score = await self._calculate_risk_score(
            operation_type,
            operation_details,
            context or {}
        )
        
        # Calculate estimated cost
        estimated_cost = operation_details.get("estimated_cost", 0.0)
        
        # Check auto-approval conditions
        if await self._check_auto_approval(policy, operation_details, risk_score, estimated_cost):
            status = ApprovalStatus.AUTO_APPROVED
            approval_chain = []
        # Check auto-rejection conditions
        elif await self._check_auto_rejection(policy, operation_details):
            status = ApprovalStatus.AUTO_REJECTED
            approval_chain = []
        else:
            status = ApprovalStatus.PENDING
            # Build approval chain
            approval_chain = await self._build_approval_chain(
                policy,
                estimated_cost,
                priority
            )
        
        # Create request
        request = ApprovalRequest(
            request_id=request_id,
            operation_type=operation_type,
            operation_details=operation_details,
            requester_id=requester_id,
            tenant_id=tenant_id,
            priority=priority,
            status=status,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=policy.approval_timeout),
            context=context or {},
            risk_score=risk_score,
            estimated_cost=estimated_cost,
            approval_chain=approval_chain,
            trace_id=format(span.get_span_context().trace_id, '032x')
        )
        
        # Store request
        if status == ApprovalStatus.PENDING:
            self.pending_requests[request_id] = request
            # Send notification to first approver
            if approval_chain:
                await self._notify_approver(approval_chain[0], request)
            # Start timeout monitor
            asyncio.create_task(self._monitor_timeout(request_id))
        else:
            self.completed_requests[request_id] = request
        
        # Update metrics
        self.metrics[f"{operation_type.value}_requests"] += 1
        self.metrics[f"status_{status.value}"] += 1
        
        span.set_attributes({
            "hitl.request_id": request_id,
            "hitl.operation_type": operation_type.value,
            "hitl.status": status.value,
            "hitl.risk_score": risk_score,
            "hitl.estimated_cost": estimated_cost,
            "hitl.approval_chain_length": len(approval_chain)
        })
        
        return request_id
    
    async def _calculate_risk_score(
        self,
        operation_type: OperationType,
        operation_details: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Calculate risk score for operation"""
        risk_score = 0.0
        
        # Base risk by operation type
        base_risks = {
            OperationType.MODEL_DEPLOYMENT: 0.3,
            OperationType.DATA_ACCESS: 0.2,
            OperationType.COST_THRESHOLD: 0.1,
            OperationType.PRODUCTION_CHANGE: 0.5,
            OperationType.DATA_DELETION: 0.4,
            OperationType.SECURITY_EXCEPTION: 0.6,
            OperationType.PRIVILEGE_ESCALATION: 0.7
        }
        
        risk_score = base_risks.get(operation_type, 0.5)
        
        # Adjust based on context
        if context.get("environment") == "production":
            risk_score += 0.2
        
        if context.get("affects_multiple_tenants"):
            risk_score += 0.3
        
        if operation_details.get("urgent"):
            risk_score += 0.1
        
        # Consider historical data
        if context.get("previous_failures", 0) > 0:
            risk_score += 0.1 * min(context["previous_failures"], 3)
        
        return min(1.0, risk_score)
    
    async def _check_auto_approval(
        self,
        policy: ApprovalPolicy,
        operation_details: Dict[str, Any],
        risk_score: float,
        estimated_cost: float
    ) -> bool:
        """Check if request can be auto-approved"""
        # Check cost threshold
        if estimated_cost > policy.max_auto_approve_cost:
            return False
        
        # Check risk threshold
        if risk_score > policy.max_auto_approve_risk:
            return False
        
        # Check auto-approve conditions
        for key, value in policy.auto_approve_conditions.items():
            if operation_details.get(key) != value:
                return False
        
        return True
    
    async def _check_auto_rejection(
        self,
        policy: ApprovalPolicy,
        operation_details: Dict[str, Any]
    ) -> bool:
        """Check if request should be auto-rejected"""
        for key, value in policy.auto_reject_conditions.items():
            if key == "cost_exceeds":
                if operation_details.get("estimated_cost", 0) > value:
                    return True
            elif key == "blacklisted_models":
                if operation_details.get("model_name") in value:
                    return True
            elif operation_details.get(key) == value:
                return True
        
        return False
    
    async def _build_approval_chain(
        self,
        policy: ApprovalPolicy,
        estimated_cost: float,
        priority: ApprovalPriority
    ) -> List[str]:
        """Build approval chain based on policy"""
        approval_chain = []
        
        # Find approvers with required roles
        eligible_approvers = []
        for approver in self.approvers.values():
            if not approver.available:
                continue
            
            # Check if approver has required role
            has_required_role = any(
                role in approver.roles
                for role in policy.required_approver_roles
            )
            
            if has_required_role:
                # Check approval limit
                if estimated_cost <= approver.approval_limit:
                    eligible_approvers.append(approver)
        
        # Sort by approval limit (lower limit first for efficiency)
        eligible_approvers.sort(key=lambda a: a.approval_limit)
        
        # Select approvers
        for i in range(min(policy.min_approvers, len(eligible_approvers))):
            approval_chain.append(eligible_approvers[i].approver_id)
        
        # Add escalation approver for high priority
        if priority in [ApprovalPriority.CRITICAL, ApprovalPriority.EMERGENCY]:
            # Find executive approver
            for approver in self.approvers.values():
                if "executive" in approver.roles and approver.approver_id not in approval_chain:
                    approval_chain.append(approver.approver_id)
                    break
        
        return approval_chain
    
    async def process_approval(
        self,
        request_id: str,
        approver_id: str,
        decision: str,  # "approve" or "reject"
        comments: Optional[str] = None
    ) -> bool:
        """
        Process an approval decision
        
        Returns:
            True if successful
        """
        if request_id not in self.pending_requests:
            return False
        
        request = self.pending_requests[request_id]
        
        # Verify approver is in chain
        if request.current_approver_index >= len(request.approval_chain):
            return False
        
        if request.approval_chain[request.current_approver_index] != approver_id:
            # Check if approver is later in chain (for parallel approval)
            if approver_id not in request.approval_chain:
                return False
        
        # Record approval
        approval_record = {
            "approver_id": approver_id,
            "decision": decision,
            "timestamp": datetime.utcnow().isoformat(),
            "comments": comments
        }
        request.approvals.append(approval_record)
        
        if comments:
            request.comments.append({
                "author": approver_id,
                "text": comments,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Update metrics
        self.metrics[f"decisions_{decision}"] += 1
        
        if decision == "reject":
            # Rejection ends the process
            request.status = ApprovalStatus.REJECTED
            self._complete_request(request_id)
            await self._notify_requester(request, "rejected")
            return True
        
        # Move to next approver
        request.current_approver_index += 1
        
        if request.current_approver_index >= len(request.approval_chain):
            # All approvals received
            request.status = ApprovalStatus.APPROVED
            self._complete_request(request_id)
            await self._notify_requester(request, "approved")
        else:
            # Notify next approver
            next_approver = request.approval_chain[request.current_approver_index]
            await self._notify_approver(next_approver, request)
        
        return True
    
    def _complete_request(self, request_id: str):
        """Move request to completed"""
        if request_id in self.pending_requests:
            request = self.pending_requests.pop(request_id)
            self.completed_requests[request_id] = request
    
    async def _monitor_timeout(self, request_id: str):
        """Monitor request for timeout"""
        try:
            if request_id not in self.pending_requests:
                return
            
            request = self.pending_requests[request_id]
            policy = self.policies[request.operation_type]
            
            # Wait for escalation timeout
            await asyncio.sleep(policy.escalation_timeout)
            
            # Check if still pending
            if request_id in self.pending_requests:
                if request.status == ApprovalStatus.PENDING:
                    # Escalate
                    await self._escalate_request(request_id)
            
            # Wait for remaining timeout
            remaining = policy.approval_timeout - policy.escalation_timeout
            await asyncio.sleep(remaining)
            
            # Check if still pending
            if request_id in self.pending_requests:
                # Expire request
                request.status = ApprovalStatus.EXPIRED
                self._complete_request(request_id)
                await self._notify_requester(request, "expired")
                
        except Exception as e:
            print(f"Timeout monitor error: {e}")
    
    async def _escalate_request(self, request_id: str):
        """Escalate a request"""
        if request_id not in self.pending_requests:
            return
        
        request = self.pending_requests[request_id]
        request.status = ApprovalStatus.ESCALATED
        
        # Find escalation approver
        for approver in self.approvers.values():
            if "executive" in approver.roles or "manager" in approver.roles:
                if approver.approver_id not in request.approval_chain:
                    request.approval_chain.append(approver.approver_id)
                    await self._notify_approver(
                        approver.approver_id,
                        request,
                        escalation=True
                    )
                    break
    
    async def _notify_approver(
        self,
        approver_id: str,
        request: ApprovalRequest,
        escalation: bool = False
    ):
        """Send notification to approver"""
        approver = self.approvers.get(approver_id)
        if not approver:
            return
        
        notification = {
            "type": "escalation" if escalation else "approval_request",
            "request_id": request.request_id,
            "operation": request.operation_type.value,
            "priority": request.priority.value,
            "requester": request.requester_id,
            "estimated_cost": request.estimated_cost,
            "risk_score": request.risk_score,
            "expires_at": request.expires_at.isoformat(),
            "approve_link": f"https://platform.example.com/approve/{request.request_id}",
            "reject_link": f"https://platform.example.com/reject/{request.request_id}"
        }
        
        # Call notification handlers
        for handler in self.notification_handlers:
            await handler(approver, notification)
        
        # Log notification
        print(f"Notification sent to {approver.name}: {notification['type']} for {request.request_id}")
    
    async def _notify_requester(
        self,
        request: ApprovalRequest,
        outcome: str
    ):
        """Notify requester of outcome"""
        notification = {
            "type": "approval_outcome",
            "request_id": request.request_id,
            "outcome": outcome,
            "operation": request.operation_type.value,
            "approvals": request.approvals,
            "comments": request.comments
        }
        
        print(f"Notified requester {request.requester_id}: {outcome} for {request.request_id}")
    
    def register_notification_handler(self, handler: Callable):
        """Register a notification handler"""
        self.notification_handlers.append(handler)
    
    async def get_pending_approvals(
        self,
        approver_id: str
    ) -> List[ApprovalRequest]:
        """Get pending approvals for an approver"""
        pending = []
        
        for request in self.pending_requests.values():
            if request.current_approver_index < len(request.approval_chain):
                if request.approval_chain[request.current_approver_index] == approver_id:
                    pending.append(request)
        
        return pending
    
    async def get_request_status(
        self,
        request_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get status of an approval request"""
        request = None
        
        if request_id in self.pending_requests:
            request = self.pending_requests[request_id]
        elif request_id in self.completed_requests:
            request = self.completed_requests[request_id]
        
        if not request:
            return None
        
        return {
            "request_id": request.request_id,
            "status": request.status.value,
            "operation": request.operation_type.value,
            "created_at": request.created_at.isoformat(),
            "expires_at": request.expires_at.isoformat(),
            "risk_score": request.risk_score,
            "estimated_cost": request.estimated_cost,
            "current_approver": (
                request.approval_chain[request.current_approver_index]
                if request.current_approver_index < len(request.approval_chain)
                else None
            ),
            "approvals": request.approvals,
            "comments": request.comments
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get HITL metrics"""
        total_pending = len(self.pending_requests)
        total_completed = len(self.completed_requests)
        
        # Calculate average approval time
        approval_times = []
        for request in self.completed_requests.values():
            if request.status == ApprovalStatus.APPROVED:
                if request.approvals:
                    last_approval = datetime.fromisoformat(request.approvals[-1]["timestamp"])
                    approval_time = (last_approval - request.created_at).total_seconds()
                    approval_times.append(approval_time)
        
        avg_approval_time = sum(approval_times) / len(approval_times) if approval_times else 0
        
        return {
            "total_requests": total_pending + total_completed,
            "pending_requests": total_pending,
            "completed_requests": total_completed,
            "auto_approved": self.metrics.get("status_auto_approved", 0),
            "auto_rejected": self.metrics.get("status_auto_rejected", 0),
            "manually_approved": self.metrics.get("decisions_approve", 0),
            "manually_rejected": self.metrics.get("decisions_reject", 0),
            "expired": self.metrics.get("status_expired", 0),
            "escalated": self.metrics.get("status_escalated", 0),
            "average_approval_time_seconds": avg_approval_time,
            "by_operation": {
                op.value: self.metrics.get(f"{op.value}_requests", 0)
                for op in OperationType
            }
        }


# Example usage
async def main():
    config = {}
    hitl_manager = HITLGateManager(config)
    
    # Create approval request
    request_id = await hitl_manager.create_approval_request(
        operation_type=OperationType.MODEL_DEPLOYMENT,
        operation_details={
            "model_name": "gpt-5-turbo",
            "environment": "production",
            "estimated_cost": 5000.0,
            "expected_qps": 1000
        },
        requester_id="user-123",
        tenant_id="tenant-456",
        priority=ApprovalPriority.HIGH,
        context={
            "reason": "New product launch",
            "affects_multiple_tenants": False
        }
    )
    
    print(f"Created approval request: {request_id}")
    
    # Get pending approvals for an approver
    pending = await hitl_manager.get_pending_approvals("approver-001")
    print(f"Pending approvals: {len(pending)}")
    
    # Process approval
    success = await hitl_manager.process_approval(
        request_id,
        "approver-001",
        "approve",
        "Looks good, approved for production deployment"
    )
    
    print(f"Approval processed: {success}")
    
    # Get request status
    status = await hitl_manager.get_request_status(request_id)
    print(f"Request status: {json.dumps(status, indent=2, default=str)}")
    
    # Get metrics
    metrics = await hitl_manager.get_metrics()
    print(f"HITL Metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main()
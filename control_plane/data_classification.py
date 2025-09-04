# control_plane/data_classification.py
"""
Comprehensive data classification and governance system with PII detection,
DSAR (Data Subject Access Request) workflows, and compliance management.
"""

import asyncio
import json
import re
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
import uuid

from opentelemetry import trace
tracer = trace.get_tracer(__name__)


class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"  
    CONFIDENTIAL = "confidential"
    SENSITIVE = "sensitive"
    RESTRICTED = "restricted"
    PII = "pii"
    PHI = "phi"  # Protected Health Information
    PCI = "pci"  # Payment Card Industry


class DataCategory(Enum):
    """Categories of data"""
    PERSONAL = "personal"
    FINANCIAL = "financial"
    HEALTH = "health"
    TECHNICAL = "technical"
    BUSINESS = "business"
    LEGAL = "legal"
    SECURITY = "security"


class ComplianceRegulation(Enum):
    """Compliance regulations"""
    GDPR = "gdpr"  # EU General Data Protection Regulation
    CCPA = "ccpa"  # California Consumer Privacy Act
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    SOX = "sox"  # Sarbanes-Oxley Act
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    SOC2 = "soc2"  # Service Organization Control 2


class PIIType(Enum):
    """Types of PII"""
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    NAME = "name"
    DATE_OF_BIRTH = "dob"
    DRIVER_LICENSE = "driver_license"
    PASSPORT = "passport"
    BANK_ACCOUNT = "bank_account"
    MEDICAL_RECORD = "medical_record"
    IP_ADDRESS = "ip_address"
    DEVICE_ID = "device_id"
    BIOMETRIC = "biometric"


class DSARRequestType(Enum):
    """DSAR request types"""
    ACCESS = "access"  # Right to access
    RECTIFICATION = "rectification"  # Right to rectification
    ERASURE = "erasure"  # Right to be forgotten
    PORTABILITY = "portability"  # Right to data portability
    RESTRICTION = "restriction"  # Right to restrict processing
    OBJECTION = "objection"  # Right to object


@dataclass
class DataAsset:
    """Represents a data asset with classification"""
    asset_id: str
    name: str
    description: str
    classification: DataClassification
    categories: List[DataCategory]
    pii_types: List[PIIType]
    owner: str
    created_at: datetime
    last_accessed: datetime
    retention_days: int
    encryption_required: bool
    access_controls: Dict[str, List[str]]  # role -> permissions
    compliance_requirements: List[ComplianceRegulation]
    tags: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PIIDetectionResult:
    """Result of PII detection scan"""
    contains_pii: bool
    pii_types_found: List[PIIType]
    pii_locations: List[Dict[str, Any]]  # location, type, confidence
    confidence_score: float
    scan_timestamp: datetime
    recommendations: List[str]


@dataclass
class DSARRequest:
    """Data Subject Access Request"""
    request_id: str
    request_type: DSARRequestType
    data_subject_id: str
    data_subject_email: str
    requested_at: datetime
    due_date: datetime
    status: str  # pending, in_progress, completed, rejected
    assigned_to: Optional[str]
    assets_identified: List[str]
    actions_taken: List[Dict[str, Any]]
    completion_date: Optional[datetime]
    notes: List[str] = field(default_factory=list)


@dataclass
class ComplianceReport:
    """Compliance assessment report"""
    report_id: str
    generated_at: datetime
    regulation: ComplianceRegulation
    compliant: bool
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    risk_score: float
    next_review_date: datetime


class DataClassificationEngine:
    """
    Engine for data classification, PII detection, and compliance
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_assets: Dict[str, DataAsset] = {}
        self.dsar_requests: Dict[str, DSARRequest] = {}
        self.pii_patterns = self._init_pii_patterns()
        self.classification_rules = self._init_classification_rules()
        self.compliance_rules = self._init_compliance_rules()
        
    def _init_pii_patterns(self) -> Dict[PIIType, List[re.Pattern]]:
        """Initialize PII detection patterns"""
        return {
            PIIType.SSN: [
                re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
                re.compile(r'\b\d{9}\b')
            ],
            PIIType.CREDIT_CARD: [
                re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
                re.compile(r'\b\d{16}\b')
            ],
            PIIType.EMAIL: [
                re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
            ],
            PIIType.PHONE: [
                re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
                re.compile(r'\b\(\d{3}\)\s?\d{3}-\d{4}\b'),
                re.compile(r'\b\+\d{1,3}[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}\b')
            ],
            PIIType.IP_ADDRESS: [
                re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
            ],
            PIIType.DATE_OF_BIRTH: [
                re.compile(r'\b\d{1,2}/\d{1,2}/\d{4}\b'),
                re.compile(r'\b\d{4}-\d{2}-\d{2}\b')
            ]
        }
    
    def _init_classification_rules(self) -> List[Dict[str, Any]]:
        """Initialize classification rules"""
        return [
            {
                "name": "healthcare_data",
                "patterns": ["medical", "health", "patient", "diagnosis", "treatment"],
                "classification": DataClassification.PHI,
                "categories": [DataCategory.HEALTH],
                "compliance": [ComplianceRegulation.HIPAA]
            },
            {
                "name": "payment_data",
                "patterns": ["credit card", "payment", "billing", "transaction"],
                "classification": DataClassification.PCI,
                "categories": [DataCategory.FINANCIAL],
                "compliance": [ComplianceRegulation.PCI_DSS]
            },
            {
                "name": "personal_data",
                "patterns": ["user", "customer", "profile", "personal"],
                "classification": DataClassification.PII,
                "categories": [DataCategory.PERSONAL],
                "compliance": [ComplianceRegulation.GDPR, ComplianceRegulation.CCPA]
            },
            {
                "name": "security_data",
                "patterns": ["password", "secret", "key", "token", "credential"],
                "classification": DataClassification.RESTRICTED,
                "categories": [DataCategory.SECURITY],
                "compliance": [ComplianceRegulation.SOC2]
            }
        ]
    
    def _init_compliance_rules(self) -> Dict[ComplianceRegulation, Dict[str, Any]]:
        """Initialize compliance requirements"""
        return {
            ComplianceRegulation.GDPR: {
                "max_retention_days": 730,
                "requires_consent": True,
                "requires_encryption": True,
                "dsar_response_days": 30,
                "breach_notification_hours": 72
            },
            ComplianceRegulation.CCPA: {
                "max_retention_days": 365,
                "requires_consent": True,
                "requires_encryption": True,
                "dsar_response_days": 45,
                "breach_notification_hours": 72
            },
            ComplianceRegulation.HIPAA: {
                "max_retention_days": 2190,  # 6 years
                "requires_consent": True,
                "requires_encryption": True,
                "requires_audit_log": True,
                "breach_notification_hours": 60
            },
            ComplianceRegulation.PCI_DSS: {
                "max_retention_days": 365,
                "requires_encryption": True,
                "requires_tokenization": True,
                "requires_audit_log": True
            }
        }
    
    @tracer.start_as_current_span("scan_for_pii")
    async def scan_for_pii(
        self,
        data: str,
        context: Optional[Dict[str, Any]] = None
    ) -> PIIDetectionResult:
        """
        Scan data for PII
        
        Returns:
            PIIDetectionResult
        """
        span = trace.get_current_span()
        
        pii_found = []
        pii_locations = []
        
        # Scan for each PII type
        for pii_type, patterns in self.pii_patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(data)
                for match in matches:
                    pii_found.append(pii_type)
                    pii_locations.append({
                        "type": pii_type.value,
                        "start": match.start(),
                        "end": match.end(),
                        "matched": match.group()[:3] + "***",  # Partial masking for logging
                        "confidence": 0.9
                    })
        
        # Advanced detection using context
        if context:
            # Check for names near keywords
            if "name" in data.lower() or "patient" in data.lower():
                # Simple heuristic for names
                name_pattern = re.compile(r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b')
                for match in name_pattern.finditer(data):
                    pii_found.append(PIIType.NAME)
                    pii_locations.append({
                        "type": PIIType.NAME.value,
                        "start": match.start(),
                        "end": match.end(),
                        "matched": match.group()[:2] + "***",
                        "confidence": 0.7
                    })
        
        # Calculate confidence
        confidence = min(1.0, len(pii_found) * 0.2) if pii_found else 0.0
        
        # Generate recommendations
        recommendations = []
        if pii_found:
            recommendations.append("Encrypt data at rest and in transit")
            recommendations.append("Apply access controls based on principle of least privilege")
            recommendations.append("Implement data retention policies")
            
            if PIIType.CREDIT_CARD in pii_found:
                recommendations.append("Tokenize credit card data")
                recommendations.append("Ensure PCI DSS compliance")
            
            if PIIType.SSN in pii_found:
                recommendations.append("Mask or redact SSN in logs and displays")
                recommendations.append("Use encryption with key management")
        
        result = PIIDetectionResult(
            contains_pii=bool(pii_found),
            pii_types_found=list(set(pii_found)),
            pii_locations=pii_locations,
            confidence_score=confidence,
            scan_timestamp=datetime.utcnow(),
            recommendations=recommendations
        )
        
        span.set_attributes({
            "pii.found": result.contains_pii,
            "pii.types": ",".join([t.value for t in result.pii_types_found]),
            "pii.confidence": result.confidence_score
        })
        
        return result
    
    async def classify_data(
        self,
        name: str,
        description: str,
        data_sample: Optional[str] = None,
        owner: str = "system"
    ) -> DataAsset:
        """
        Classify data and create data asset
        
        Returns:
            DataAsset
        """
        asset_id = f"asset-{uuid.uuid4().hex[:12]}"
        
        # Determine classification based on rules
        classification = DataClassification.INTERNAL
        categories = []
        compliance_requirements = []
        
        combined_text = f"{name} {description} {data_sample or ''}"
        
        for rule in self.classification_rules:
            for pattern in rule["patterns"]:
                if pattern.lower() in combined_text.lower():
                    # Upgrade classification if more restrictive
                    if self._is_more_restrictive(rule["classification"], classification):
                        classification = rule["classification"]
                    
                    categories.extend(rule["categories"])
                    compliance_requirements.extend(rule["compliance"])
        
        # Scan for PII if sample provided
        pii_types = []
        if data_sample:
            pii_result = await self.scan_for_pii(data_sample)
            pii_types = pii_result.pii_types_found
            
            if pii_result.contains_pii:
                # Upgrade to PII classification if PII found
                if self._is_more_restrictive(DataClassification.PII, classification):
                    classification = DataClassification.PII
        
        # Determine retention period based on compliance
        retention_days = 365  # Default
        for reg in compliance_requirements:
            if reg in self.compliance_rules:
                max_retention = self.compliance_rules[reg].get("max_retention_days", 365)
                retention_days = max(retention_days, max_retention)
        
        # Create data asset
        asset = DataAsset(
            asset_id=asset_id,
            name=name,
            description=description,
            classification=classification,
            categories=list(set(categories)) if categories else [DataCategory.BUSINESS],
            pii_types=pii_types,
            owner=owner,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            retention_days=retention_days,
            encryption_required=classification in [
                DataClassification.SENSITIVE,
                DataClassification.RESTRICTED,
                DataClassification.PII,
                DataClassification.PHI,
                DataClassification.PCI
            ],
            access_controls=self._generate_access_controls(classification),
            compliance_requirements=list(set(compliance_requirements)),
            tags={
                "classification": classification.value,
                "export_ok": classification == DataClassification.PUBLIC
            }
        )
        
        self.data_assets[asset_id] = asset
        
        return asset
    
    def _is_more_restrictive(
        self,
        new_class: DataClassification,
        current_class: DataClassification
    ) -> bool:
        """Check if new classification is more restrictive"""
        hierarchy = {
            DataClassification.PUBLIC: 0,
            DataClassification.INTERNAL: 1,
            DataClassification.CONFIDENTIAL: 2,
            DataClassification.SENSITIVE: 3,
            DataClassification.PII: 4,
            DataClassification.PHI: 4,
            DataClassification.PCI: 4,
            DataClassification.RESTRICTED: 5
        }
        
        return hierarchy.get(new_class, 0) > hierarchy.get(current_class, 0)
    
    def _generate_access_controls(
        self,
        classification: DataClassification
    ) -> Dict[str, List[str]]:
        """Generate access controls based on classification"""
        if classification == DataClassification.PUBLIC:
            return {
                "public": ["read"],
                "user": ["read", "write"],
                "admin": ["read", "write", "delete"]
            }
        elif classification == DataClassification.INTERNAL:
            return {
                "user": ["read"],
                "privileged": ["read", "write"],
                "admin": ["read", "write", "delete"]
            }
        elif classification in [DataClassification.CONFIDENTIAL, DataClassification.SENSITIVE]:
            return {
                "privileged": ["read"],
                "admin": ["read", "write"],
                "security": ["read", "write", "delete"]
            }
        else:  # RESTRICTED, PII, PHI, PCI
            return {
                "admin": ["read"],
                "security": ["read", "write"],
                "compliance": ["read", "audit"]
            }
    
    async def create_dsar_request(
        self,
        request_type: DSARRequestType,
        data_subject_id: str,
        data_subject_email: str
    ) -> DSARRequest:
        """
        Create a Data Subject Access Request
        
        Returns:
            DSARRequest
        """
        request_id = f"dsar-{uuid.uuid4().hex[:12]}"
        
        # Determine due date based on regulations
        due_days = 30  # Default GDPR timeline
        
        request = DSARRequest(
            request_id=request_id,
            request_type=request_type,
            data_subject_id=data_subject_id,
            data_subject_email=data_subject_email,
            requested_at=datetime.utcnow(),
            due_date=datetime.utcnow() + timedelta(days=due_days),
            status="pending",
            assigned_to=None,
            assets_identified=[],
            actions_taken=[],
            completion_date=None
        )
        
        self.dsar_requests[request_id] = request
        
        # Start processing
        asyncio.create_task(self._process_dsar_request(request_id))
        
        return request
    
    async def _process_dsar_request(self, request_id: str):
        """Process DSAR request"""
        if request_id not in self.dsar_requests:
            return
        
        request = self.dsar_requests[request_id]
        request.status = "in_progress"
        
        # Find all assets related to data subject
        related_assets = []
        for asset in self.data_assets.values():
            # Check if asset contains data subject's information
            if self._asset_contains_subject_data(asset, request.data_subject_id):
                related_assets.append(asset.asset_id)
        
        request.assets_identified = related_assets
        
        # Process based on request type
        if request.request_type == DSARRequestType.ACCESS:
            await self._handle_access_request(request)
        elif request.request_type == DSARRequestType.ERASURE:
            await self._handle_erasure_request(request)
        elif request.request_type == DSARRequestType.PORTABILITY:
            await self._handle_portability_request(request)
        elif request.request_type == DSARRequestType.RECTIFICATION:
            await self._handle_rectification_request(request)
        
        request.status = "completed"
        request.completion_date = datetime.utcnow()
    
    def _asset_contains_subject_data(
        self,
        asset: DataAsset,
        data_subject_id: str
    ) -> bool:
        """Check if asset contains data subject's information"""
        # In production, this would query the actual data
        # For now, simple heuristic based on metadata
        return (
            asset.owner == data_subject_id or
            asset.metadata.get("data_subject_id") == data_subject_id or
            data_subject_id in str(asset.tags)
        )
    
    async def _handle_access_request(self, request: DSARRequest):
        """Handle data access request"""
        request.actions_taken.append({
            "action": "data_collected",
            "timestamp": datetime.utcnow().isoformat(),
            "assets": request.assets_identified
        })
        
        # Generate report
        request.actions_taken.append({
            "action": "report_generated",
            "timestamp": datetime.utcnow().isoformat(),
            "report_location": f"/reports/{request.request_id}.pdf"
        })
    
    async def _handle_erasure_request(self, request: DSARRequest):
        """Handle right to be forgotten request"""
        for asset_id in request.assets_identified:
            if asset_id in self.data_assets:
                asset = self.data_assets[asset_id]
                
                # Check if erasure is allowed
                if not self._can_erase_asset(asset):
                    request.notes.append(f"Cannot erase {asset_id}: legal retention required")
                    continue
                
                # Mark for deletion
                request.actions_taken.append({
                    "action": "marked_for_deletion",
                    "timestamp": datetime.utcnow().isoformat(),
                    "asset_id": asset_id
                })
                
                # In production, would trigger actual deletion
                asset.tags["marked_for_deletion"] = True
                asset.tags["deletion_date"] = (datetime.utcnow() + timedelta(days=30)).isoformat()
    
    def _can_erase_asset(self, asset: DataAsset) -> bool:
        """Check if asset can be erased"""
        # Check compliance requirements
        for reg in asset.compliance_requirements:
            if reg == ComplianceRegulation.SOX:
                # SOX requires retention for 7 years
                age = (datetime.utcnow() - asset.created_at).days
                if age < 2555:  # 7 years
                    return False
        
        return True
    
    async def _handle_portability_request(self, request: DSARRequest):
        """Handle data portability request"""
        request.actions_taken.append({
            "action": "data_exported",
            "timestamp": datetime.utcnow().isoformat(),
            "format": "JSON",
            "export_location": f"/exports/{request.request_id}.json"
        })
    
    async def _handle_rectification_request(self, request: DSARRequest):
        """Handle data rectification request"""
        request.actions_taken.append({
            "action": "rectification_queued",
            "timestamp": datetime.utcnow().isoformat(),
            "review_required": True
        })
    
    async def generate_compliance_report(
        self,
        regulation: ComplianceRegulation
    ) -> ComplianceReport:
        """Generate compliance assessment report"""
        report_id = f"report-{uuid.uuid4().hex[:12]}"
        violations = []
        recommendations = []
        
        # Check all assets for compliance
        for asset in self.data_assets.values():
            if regulation in asset.compliance_requirements:
                # Check encryption
                if self.compliance_rules[regulation].get("requires_encryption") and not asset.encryption_required:
                    violations.append({
                        "asset_id": asset.asset_id,
                        "violation": "Missing required encryption",
                        "severity": "high"
                    })
                
                # Check retention
                max_retention = self.compliance_rules[regulation].get("max_retention_days")
                if max_retention and asset.retention_days > max_retention:
                    violations.append({
                        "asset_id": asset.asset_id,
                        "violation": f"Retention period exceeds maximum ({asset.retention_days} > {max_retention})",
                        "severity": "medium"
                    })
        
        # Generate recommendations
        if violations:
            recommendations.append("Address all high severity violations immediately")
            recommendations.append("Implement automated compliance monitoring")
            recommendations.append("Schedule regular compliance audits")
        
        # Calculate risk score
        risk_score = min(1.0, len(violations) * 0.1) if violations else 0.0
        
        report = ComplianceReport(
            report_id=report_id,
            generated_at=datetime.utcnow(),
            regulation=regulation,
            compliant=len(violations) == 0,
            violations=violations,
            recommendations=recommendations,
            risk_score=risk_score,
            next_review_date=datetime.utcnow() + timedelta(days=90)
        )
        
        return report
    
    def get_data_tags(self, asset_id: str) -> Dict[str, Any]:
        """Get data classification tags for an asset"""
        if asset_id not in self.data_assets:
            return {}
        
        asset = self.data_assets[asset_id]
        
        return {
            "classification": asset.classification.value,
            "pii": "PII" if asset.pii_types else None,
            "sensitive": "SENSITIVE" if asset.classification in [
                DataClassification.SENSITIVE,
                DataClassification.RESTRICTED,
                DataClassification.PHI
            ] else None,
            "export_ok": "EXPORT_OK" if asset.classification == DataClassification.PUBLIC else None,
            "categories": [c.value for c in asset.categories],
            "compliance": [r.value for r in asset.compliance_requirements],
            "retention_days": asset.retention_days,
            "encryption_required": asset.encryption_required
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get classification metrics"""
        total_assets = len(self.data_assets)
        
        # Count by classification
        by_classification = {}
        for asset in self.data_assets.values():
            class_name = asset.classification.value
            by_classification[class_name] = by_classification.get(class_name, 0) + 1
        
        # Count PII assets
        pii_assets = sum(1 for a in self.data_assets.values() if a.pii_types)
        
        # DSAR metrics
        pending_dsars = sum(1 for r in self.dsar_requests.values() if r.status == "pending")
        completed_dsars = sum(1 for r in self.dsar_requests.values() if r.status == "completed")
        
        return {
            "total_assets": total_assets,
            "by_classification": by_classification,
            "pii_assets": pii_assets,
            "encrypted_assets": sum(1 for a in self.data_assets.values() if a.encryption_required),
            "dsar_requests": {
                "total": len(self.dsar_requests),
                "pending": pending_dsars,
                "completed": completed_dsars
            }
        }


# Example usage
async def main():
    config = {}
    classifier = DataClassificationEngine(config)
    
    # Scan for PII
    test_data = "John Smith's SSN is 123-45-6789 and email is john@example.com"
    pii_result = await classifier.scan_for_pii(test_data)
    print(f"PII Found: {pii_result.contains_pii}")
    print(f"PII Types: {[t.value for t in pii_result.pii_types_found]}")
    
    # Classify data
    asset = await classifier.classify_data(
        name="Customer Database",
        description="Contains customer profiles and payment information",
        data_sample=test_data,
        owner="data-team"
    )
    
    print(f"Asset Classification: {asset.classification.value}")
    print(f"Categories: {[c.value for c in asset.categories]}")
    print(f"Compliance: {[r.value for r in asset.compliance_requirements]}")
    
    # Get data tags
    tags = classifier.get_data_tags(asset.asset_id)
    print(f"Data Tags: {json.dumps(tags, indent=2)}")
    
    # Create DSAR request
    dsar = await classifier.create_dsar_request(
        request_type=DSARRequestType.ACCESS,
        data_subject_id="user-123",
        data_subject_email="user@example.com"
    )
    
    print(f"DSAR Request: {dsar.request_id}")
    print(f"Due Date: {dsar.due_date}")
    
    # Wait for processing
    await asyncio.sleep(1)
    
    # Generate compliance report
    report = await classifier.generate_compliance_report(ComplianceRegulation.GDPR)
    print(f"Compliance Report: {report.report_id}")
    print(f"Compliant: {report.compliant}")
    print(f"Risk Score: {report.risk_score}")
    
    # Get metrics
    metrics = await classifier.get_metrics()
    print(f"Metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
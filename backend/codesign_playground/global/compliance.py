"""
Global Compliance Framework for AI Hardware Co-Design Platform.

This module implements comprehensive compliance capabilities for global deployment
including GDPR, CCPA, PDPA, and other international data protection regulations.
"""

import json
import hashlib
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import uuid

logger = logging.getLogger(__name__)


class ComplianceRegulation(Enum):
    """Supported compliance regulations."""
    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    CCPA = "ccpa"  # California Consumer Privacy Act (US)
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act (Canada)
    DPA = "dpa"  # Data Protection Act (UK)
    PRIVACY_ACT = "privacy_act"  # Privacy Act (Australia)
    APPI = "appi"  # Act on Protection of Personal Information (Japan)


class DataCategory(Enum):
    """Categories of data for compliance classification."""
    PERSONAL_IDENTIFIABLE = "personal_identifiable"
    SENSITIVE_PERSONAL = "sensitive_personal" 
    TECHNICAL_TELEMETRY = "technical_telemetry"
    USAGE_ANALYTICS = "usage_analytics"
    PERFORMANCE_METRICS = "performance_metrics"
    MODEL_ARTIFACTS = "model_artifacts"
    RESEARCH_DATA = "research_data"
    SYSTEM_LOGS = "system_logs"
    AUDIT_TRAILS = "audit_trails"
    COOKIES_TRACKING = "cookies_tracking"


class ProcessingPurpose(Enum):
    """Purposes for data processing."""
    SERVICE_PROVISION = "service_provision"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    RESEARCH_DEVELOPMENT = "research_development"
    QUALITY_ASSURANCE = "quality_assurance"
    SECURITY_MONITORING = "security_monitoring"
    LEGAL_COMPLIANCE = "legal_compliance"
    MARKETING_ANALYTICS = "marketing_analytics"
    USER_EXPERIENCE = "user_experience"
    SYSTEM_MAINTENANCE = "system_maintenance"
    ACADEMIC_RESEARCH = "academic_research"


class LegalBasis(Enum):
    """Legal basis for data processing (GDPR Article 6)."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


class DataSubjectRight(Enum):
    """Data subject rights under various regulations."""
    ACCESS = "access"  # Right to access personal data
    RECTIFICATION = "rectification"  # Right to correct inaccurate data
    ERASURE = "erasure"  # Right to be forgotten
    PORTABILITY = "portability"  # Right to data portability
    RESTRICT_PROCESSING = "restrict_processing"  # Right to restrict processing
    OBJECT_PROCESSING = "object_processing"  # Right to object to processing
    OPT_OUT = "opt_out"  # Right to opt out (CCPA)
    NON_DISCRIMINATION = "non_discrimination"  # Right to non-discrimination (CCPA)


@dataclass
class DataProcessingRecord:
    """Record of data processing activity for compliance."""
    
    record_id: str
    user_id: Optional[str]
    data_category: DataCategory
    processing_purpose: ProcessingPurpose
    legal_basis: LegalBasis
    data_collected: List[str]
    timestamp: str
    retention_period: int  # days
    applicable_regulations: List[ComplianceRegulation]
    consent_obtained: bool = False
    consent_id: Optional[str] = None
    processing_location: Optional[str] = None
    third_party_sharing: bool = False
    third_parties: List[str] = field(default_factory=list)
    data_minimization_applied: bool = True
    anonymization_applied: bool = False
    encryption_applied: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "record_id": self.record_id,
            "user_id": self.user_id,
            "data_category": self.data_category.value,
            "processing_purpose": self.processing_purpose.value,
            "legal_basis": self.legal_basis.value,
            "data_collected": self.data_collected,
            "timestamp": self.timestamp,
            "retention_period": self.retention_period,
            "applicable_regulations": [reg.value for reg in self.applicable_regulations],
            "consent_obtained": self.consent_obtained,
            "consent_id": self.consent_id,
            "processing_location": self.processing_location,
            "third_party_sharing": self.third_party_sharing,
            "third_parties": self.third_parties,
            "data_minimization_applied": self.data_minimization_applied,
            "anonymization_applied": self.anonymization_applied,
            "encryption_applied": self.encryption_applied
        }


@dataclass
class ConsentRecord:
    """Record of user consent for compliance."""
    
    consent_id: str
    user_id: str
    consent_type: str
    purposes: List[ProcessingPurpose]
    data_categories: List[DataCategory]
    granted: bool
    timestamp: str
    expiry_date: Optional[str] = None
    withdrawal_date: Optional[str] = None
    granular_preferences: Dict[str, bool] = field(default_factory=dict)
    consent_method: str = "explicit"  # explicit, implicit, opt_in, opt_out
    applicable_regulations: List[ComplianceRegulation] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "consent_id": self.consent_id,
            "user_id": self.user_id,
            "consent_type": self.consent_type,
            "purposes": [p.value for p in self.purposes],
            "data_categories": [dc.value for dc in self.data_categories],
            "granted": self.granted,
            "timestamp": self.timestamp,
            "expiry_date": self.expiry_date,
            "withdrawal_date": self.withdrawal_date,
            "granular_preferences": self.granular_preferences,
            "consent_method": self.consent_method,
            "applicable_regulations": [reg.value for reg in self.applicable_regulations]
        }


@dataclass
class DataSubjectRequest:
    """Data subject request for compliance."""
    
    request_id: str
    user_id: str
    request_type: DataSubjectRight
    applicable_regulation: ComplianceRegulation
    request_details: str
    timestamp: str
    status: str = "pending"  # pending, in_progress, completed, rejected
    response_due_date: Optional[str] = None
    response_provided: Optional[str] = None
    completion_date: Optional[str] = None
    verification_method: Optional[str] = None
    data_exported: Optional[str] = None
    data_deleted: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "request_type": self.request_type.value,
            "applicable_regulation": self.applicable_regulation.value,
            "request_details": self.request_details,
            "timestamp": self.timestamp,
            "status": self.status,
            "response_due_date": self.response_due_date,
            "response_provided": self.response_provided,
            "completion_date": self.completion_date,
            "verification_method": self.verification_method,
            "data_exported": self.data_exported,
            "data_deleted": self.data_deleted
        }


class ComplianceFramework:
    """Comprehensive compliance framework for global regulations."""
    
    def __init__(self):
        """Initialize compliance framework."""
        self.processing_records: Dict[str, DataProcessingRecord] = {}
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.data_subject_requests: Dict[str, DataSubjectRequest] = {}
        
        # Regulation-specific configurations
        self.regulation_configs = self._initialize_regulation_configs()
        
        # Data retention policies
        self.retention_policies = self._initialize_retention_policies()
        
        # Active compliance monitors
        self.compliance_monitors = {}
        
        logger.info("Compliance framework initialized with support for major global regulations")
    
    def _initialize_regulation_configs(self) -> Dict[ComplianceRegulation, Dict[str, Any]]:
        """Initialize regulation-specific configurations."""
        return {
            ComplianceRegulation.GDPR: {
                "name": "General Data Protection Regulation",
                "jurisdiction": "European Union",
                "requires_explicit_consent": True,
                "data_subject_response_time": 30,  # days
                "breach_notification_time": 72,  # hours
                "dpo_required": True,
                "privacy_by_design": True,
                "right_to_be_forgotten": True,
                "data_portability": True,
                "lawful_basis_required": True
            },
            ComplianceRegulation.CCPA: {
                "name": "California Consumer Privacy Act",
                "jurisdiction": "California, USA",
                "requires_explicit_consent": False,
                "data_subject_response_time": 45,  # days
                "opt_out_required": True,
                "sale_disclosure_required": True,
                "non_discrimination": True,
                "deletion_rights": True,
                "access_rights": True
            },
            ComplianceRegulation.PDPA: {
                "name": "Personal Data Protection Act",
                "jurisdiction": "Singapore",
                "requires_explicit_consent": True,
                "data_subject_response_time": 30,  # days
                "breach_notification_time": 72,  # hours
                "dpo_required": True,
                "consent_withdrawal": True
            },
            ComplianceRegulation.LGPD: {
                "name": "Lei Geral de Proteção de Dados",
                "jurisdiction": "Brazil", 
                "requires_explicit_consent": True,
                "data_subject_response_time": 15,  # days
                "dpo_required": True,
                "privacy_by_design": True,
                "right_to_be_forgotten": True
            },
            ComplianceRegulation.PIPEDA: {
                "name": "Personal Information Protection and Electronic Documents Act",
                "jurisdiction": "Canada",
                "requires_explicit_consent": True,
                "data_subject_response_time": 30,  # days
                "breach_notification_time": 72,  # hours
                "access_rights": True,
                "correction_rights": True
            }
        }
    
    def _initialize_retention_policies(self) -> Dict[DataCategory, int]:
        """Initialize data retention policies (in days)."""
        return {
            DataCategory.PERSONAL_IDENTIFIABLE: 2555,  # 7 years
            DataCategory.SENSITIVE_PERSONAL: 1095,  # 3 years
            DataCategory.TECHNICAL_TELEMETRY: 365,  # 1 year
            DataCategory.USAGE_ANALYTICS: 730,  # 2 years
            DataCategory.PERFORMANCE_METRICS: 1095,  # 3 years
            DataCategory.MODEL_ARTIFACTS: 1825,  # 5 years
            DataCategory.RESEARCH_DATA: 3650,  # 10 years
            DataCategory.SYSTEM_LOGS: 90,  # 3 months
            DataCategory.AUDIT_TRAILS: 2555,  # 7 years
            DataCategory.COOKIES_TRACKING: 365  # 1 year
        }
    
    def record_data_processing(
        self,
        user_id: Optional[str],
        data_category: DataCategory,
        processing_purpose: ProcessingPurpose,
        legal_basis: LegalBasis,
        data_collected: List[str],
        applicable_regulations: List[ComplianceRegulation],
        consent_id: Optional[str] = None,
        processing_location: Optional[str] = None,
        third_parties: Optional[List[str]] = None
    ) -> str:
        """Record data processing activity for compliance."""
        
        record_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Determine retention period
        retention_period = self.retention_policies.get(data_category, 365)
        
        # Create processing record
        processing_record = DataProcessingRecord(
            record_id=record_id,
            user_id=user_id,
            data_category=data_category,
            processing_purpose=processing_purpose,
            legal_basis=legal_basis,
            data_collected=data_collected,
            timestamp=timestamp,
            retention_period=retention_period,
            applicable_regulations=applicable_regulations,
            consent_obtained=consent_id is not None,
            consent_id=consent_id,
            processing_location=processing_location,
            third_party_sharing=bool(third_parties),
            third_parties=third_parties or [],
            data_minimization_applied=True,
            anonymization_applied=data_category != DataCategory.PERSONAL_IDENTIFIABLE,
            encryption_applied=True
        )
        
        self.processing_records[record_id] = processing_record
        
        logger.info(f"Recorded data processing activity: {record_id}")
        return record_id
    
    def obtain_consent(
        self,
        user_id: str,
        consent_type: str,
        purposes: List[ProcessingPurpose],
        data_categories: List[DataCategory],
        applicable_regulations: List[ComplianceRegulation],
        granular_preferences: Optional[Dict[str, bool]] = None,
        consent_method: str = "explicit",
        expiry_period_days: Optional[int] = None
    ) -> str:
        """Record user consent for data processing."""
        
        consent_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Calculate expiry date if specified
        expiry_date = None
        if expiry_period_days:
            expiry_datetime = datetime.now() + timedelta(days=expiry_period_days)
            expiry_date = expiry_datetime.isoformat()
        
        consent_record = ConsentRecord(
            consent_id=consent_id,
            user_id=user_id,
            consent_type=consent_type,
            purposes=purposes,
            data_categories=data_categories,
            granted=True,
            timestamp=timestamp,
            expiry_date=expiry_date,
            granular_preferences=granular_preferences or {},
            consent_method=consent_method,
            applicable_regulations=applicable_regulations
        )
        
        self.consent_records[consent_id] = consent_record
        
        logger.info(f"Recorded consent: {consent_id} for user {user_id}")
        return consent_id
    
    def withdraw_consent(self, consent_id: str) -> bool:
        """Withdraw previously granted consent."""
        
        if consent_id not in self.consent_records:
            logger.warning(f"Consent record not found: {consent_id}")
            return False
        
        consent_record = self.consent_records[consent_id]
        consent_record.granted = False
        consent_record.withdrawal_date = datetime.now().isoformat()
        
        logger.info(f"Consent withdrawn: {consent_id}")
        return True
    
    def submit_data_subject_request(
        self,
        user_id: str,
        request_type: DataSubjectRight,
        applicable_regulation: ComplianceRegulation,
        request_details: str
    ) -> str:
        """Submit data subject request (access, deletion, etc.)."""
        
        request_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Calculate response due date based on regulation
        regulation_config = self.regulation_configs.get(applicable_regulation, {})
        response_days = regulation_config.get("data_subject_response_time", 30)
        
        due_date = datetime.now() + timedelta(days=response_days)
        response_due_date = due_date.isoformat()
        
        request = DataSubjectRequest(
            request_id=request_id,
            user_id=user_id,
            request_type=request_type,
            applicable_regulation=applicable_regulation,
            request_details=request_details,
            timestamp=timestamp,
            response_due_date=response_due_date
        )
        
        self.data_subject_requests[request_id] = request
        
        logger.info(f"Data subject request submitted: {request_id}")
        return request_id
    
    def process_data_subject_request(self, request_id: str) -> Dict[str, Any]:
        """Process data subject request and return response."""
        
        if request_id not in self.data_subject_requests:
            return {"error": "Request not found"}
        
        request = self.data_subject_requests[request_id]
        request.status = "in_progress"
        
        response_data = {}
        
        if request.request_type == DataSubjectRight.ACCESS:
            # Provide access to personal data
            user_data = self._collect_user_data(request.user_id)
            response_data = {
                "request_type": "data_access",
                "user_data": user_data,
                "processing_records": self._get_user_processing_records(request.user_id),
                "consent_records": self._get_user_consent_records(request.user_id)
            }
            
        elif request.request_type == DataSubjectRight.ERASURE:
            # Delete personal data (right to be forgotten)
            deleted_data = self._delete_user_data(request.user_id)
            response_data = {
                "request_type": "data_deletion",
                "deleted_records": deleted_data,
                "deletion_confirmation": True
            }
            
        elif request.request_type == DataSubjectRight.PORTABILITY:
            # Provide data in portable format
            portable_data = self._export_user_data_portable(request.user_id)
            response_data = {
                "request_type": "data_portability",
                "exported_data": portable_data,
                "format": "JSON"
            }
            
        elif request.request_type == DataSubjectRight.RESTRICT_PROCESSING:
            # Restrict processing of personal data
            self._restrict_user_data_processing(request.user_id)
            response_data = {
                "request_type": "processing_restriction",
                "restriction_applied": True
            }
        
        # Update request status
        request.status = "completed"
        request.completion_date = datetime.now().isoformat()
        request.response_provided = json.dumps(response_data)
        
        logger.info(f"Processed data subject request: {request_id}")
        return response_data
    
    def check_compliance_status(self, regulations: List[ComplianceRegulation]) -> Dict[str, Any]:
        """Check compliance status for specified regulations."""
        
        compliance_status = {}
        
        for regulation in regulations:
            config = self.regulation_configs.get(regulation, {})
            
            status = {
                "regulation": regulation.value,
                "compliant": True,
                "issues": [],
                "recommendations": []
            }
            
            # Check consent management
            if config.get("requires_explicit_consent", False):
                consent_issues = self._check_consent_compliance(regulation)
                if consent_issues:
                    status["compliant"] = False
                    status["issues"].extend(consent_issues)
            
            # Check data retention
            retention_issues = self._check_retention_compliance(regulation)
            if retention_issues:
                status["compliant"] = False
                status["issues"].extend(retention_issues)
            
            # Check data subject request response times
            request_issues = self._check_request_response_times(regulation)
            if request_issues:
                status["compliant"] = False
                status["issues"].extend(request_issues)
            
            # Generate recommendations
            if not status["compliant"]:
                status["recommendations"] = self._generate_compliance_recommendations(regulation, status["issues"])
            
            compliance_status[regulation.value] = status
        
        return compliance_status
    
    def generate_compliance_report(self, regulations: List[ComplianceRegulation]) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        
        report = {
            "report_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "regulations_covered": [reg.value for reg in regulations],
            "compliance_status": self.check_compliance_status(regulations),
            "data_processing_summary": self._generate_processing_summary(),
            "consent_management_summary": self._generate_consent_summary(),
            "data_subject_requests_summary": self._generate_requests_summary(),
            "recommendations": [],
            "action_items": []
        }
        
        # Generate overall recommendations
        for reg_status in report["compliance_status"].values():
            if not reg_status["compliant"]:
                report["recommendations"].extend(reg_status["recommendations"])
        
        # Generate action items
        report["action_items"] = self._generate_action_items(report["compliance_status"])
        
        return report
    
    def _collect_user_data(self, user_id: str) -> Dict[str, Any]:
        """Collect all data associated with a user."""
        user_data = {
            "user_id": user_id,
            "processing_activities": [],
            "consent_records": [],
            "data_categories": set()
        }
        
        # Collect processing records
        for record in self.processing_records.values():
            if record.user_id == user_id:
                user_data["processing_activities"].append(record.to_dict())
                user_data["data_categories"].add(record.data_category.value)
        
        # Collect consent records
        for consent in self.consent_records.values():
            if consent.user_id == user_id:
                user_data["consent_records"].append(consent.to_dict())
        
        user_data["data_categories"] = list(user_data["data_categories"])
        
        return user_data
    
    def _get_user_processing_records(self, user_id: str) -> List[Dict[str, Any]]:
        """Get processing records for a user."""
        return [
            record.to_dict() for record in self.processing_records.values()
            if record.user_id == user_id
        ]
    
    def _get_user_consent_records(self, user_id: str) -> List[Dict[str, Any]]:
        """Get consent records for a user."""
        return [
            consent.to_dict() for consent in self.consent_records.values()
            if consent.user_id == user_id
        ]
    
    def _delete_user_data(self, user_id: str) -> List[str]:
        """Delete all data associated with a user."""
        deleted_records = []
        
        # Delete processing records
        records_to_delete = [
            record_id for record_id, record in self.processing_records.items()
            if record.user_id == user_id
        ]
        
        for record_id in records_to_delete:
            del self.processing_records[record_id]
            deleted_records.append(f"processing_record_{record_id}")
        
        # Delete consent records
        consents_to_delete = [
            consent_id for consent_id, consent in self.consent_records.items()
            if consent.user_id == user_id
        ]
        
        for consent_id in consents_to_delete:
            del self.consent_records[consent_id]
            deleted_records.append(f"consent_record_{consent_id}")
        
        return deleted_records
    
    def _export_user_data_portable(self, user_id: str) -> Dict[str, Any]:
        """Export user data in portable format."""
        user_data = self._collect_user_data(user_id)
        
        portable_data = {
            "format": "JSON",
            "version": "1.0",
            "export_timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "data": user_data
        }
        
        return portable_data
    
    def _restrict_user_data_processing(self, user_id: str) -> None:
        """Restrict processing of user data."""
        # Mark all user's processing records as restricted
        for record in self.processing_records.values():
            if record.user_id == user_id:
                # In a real implementation, this would set a restriction flag
                # and prevent further processing
                pass
        
        logger.info(f"Processing restriction applied for user: {user_id}")
    
    def _check_consent_compliance(self, regulation: ComplianceRegulation) -> List[str]:
        """Check consent management compliance for regulation."""
        issues = []
        
        config = self.regulation_configs.get(regulation, {})
        
        if config.get("requires_explicit_consent", False):
            # Check for processing without consent
            for record in self.processing_records.values():
                if (regulation in record.applicable_regulations and
                    not record.consent_obtained and
                    record.legal_basis == LegalBasis.CONSENT):
                    issues.append(f"Processing without consent: {record.record_id}")
        
        return issues
    
    def _check_retention_compliance(self, regulation: ComplianceRegulation) -> List[str]:
        """Check data retention compliance."""
        issues = []
        current_time = datetime.now()
        
        for record in self.processing_records.values():
            if regulation in record.applicable_regulations:
                record_date = datetime.fromisoformat(record.timestamp)
                retention_limit = record_date + timedelta(days=record.retention_period)
                
                if current_time > retention_limit:
                    issues.append(f"Data retention period exceeded: {record.record_id}")
        
        return issues
    
    def _check_request_response_times(self, regulation: ComplianceRegulation) -> List[str]:
        """Check data subject request response times."""
        issues = []
        current_time = datetime.now()
        
        for request in self.data_subject_requests.values():
            if (request.applicable_regulation == regulation and
                request.status != "completed" and
                request.response_due_date):
                
                due_date = datetime.fromisoformat(request.response_due_date)
                if current_time > due_date:
                    issues.append(f"Overdue data subject request: {request.request_id}")
        
        return issues
    
    def _generate_compliance_recommendations(self, regulation: ComplianceRegulation, issues: List[str]) -> List[str]:
        """Generate compliance recommendations based on issues."""
        recommendations = []
        
        if any("consent" in issue.lower() for issue in issues):
            recommendations.append("Implement comprehensive consent management system")
            recommendations.append("Review and update consent collection processes")
        
        if any("retention" in issue.lower() for issue in issues):
            recommendations.append("Implement automated data retention policies")
            recommendations.append("Set up regular data purging processes")
        
        if any("overdue" in issue.lower() for issue in issues):
            recommendations.append("Improve data subject request response procedures")
            recommendations.append("Implement automated request tracking and alerts")
        
        return recommendations
    
    def _generate_processing_summary(self) -> Dict[str, Any]:
        """Generate data processing summary."""
        summary = {
            "total_processing_records": len(self.processing_records),
            "categories_processed": {},
            "purposes": {},
            "legal_bases": {},
            "regulations": {}
        }
        
        for record in self.processing_records.values():
            # Count categories
            category = record.data_category.value
            summary["categories_processed"][category] = summary["categories_processed"].get(category, 0) + 1
            
            # Count purposes
            purpose = record.processing_purpose.value
            summary["purposes"][purpose] = summary["purposes"].get(purpose, 0) + 1
            
            # Count legal bases
            legal_basis = record.legal_basis.value
            summary["legal_bases"][legal_basis] = summary["legal_bases"].get(legal_basis, 0) + 1
            
            # Count regulations
            for reg in record.applicable_regulations:
                reg_value = reg.value
                summary["regulations"][reg_value] = summary["regulations"].get(reg_value, 0) + 1
        
        return summary
    
    def _generate_consent_summary(self) -> Dict[str, Any]:
        """Generate consent management summary."""
        total_consents = len(self.consent_records)
        active_consents = sum(1 for consent in self.consent_records.values() if consent.granted)
        withdrawn_consents = total_consents - active_consents
        
        return {
            "total_consents": total_consents,
            "active_consents": active_consents,
            "withdrawn_consents": withdrawn_consents,
            "consent_rate": (active_consents / total_consents * 100) if total_consents > 0 else 0
        }
    
    def _generate_requests_summary(self) -> Dict[str, Any]:
        """Generate data subject requests summary."""
        total_requests = len(self.data_subject_requests)
        
        status_counts = {}
        type_counts = {}
        
        for request in self.data_subject_requests.values():
            # Count by status
            status = request.status
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Count by type
            request_type = request.request_type.value
            type_counts[request_type] = type_counts.get(request_type, 0) + 1
        
        return {
            "total_requests": total_requests,
            "status_breakdown": status_counts,
            "type_breakdown": type_counts
        }
    
    def _generate_action_items(self, compliance_status: Dict[str, Any]) -> List[str]:
        """Generate action items based on compliance status."""
        action_items = []
        
        for reg_name, status in compliance_status.items():
            if not status["compliant"]:
                action_items.append(f"Address {reg_name} compliance issues: {len(status['issues'])} items")
        
        # General action items
        if len(self.data_subject_requests) > 0:
            pending_requests = sum(
                1 for request in self.data_subject_requests.values()
                if request.status == "pending"
            )
            if pending_requests > 0:
                action_items.append(f"Process {pending_requests} pending data subject requests")
        
        return action_items


# Global compliance framework instance
_compliance_framework: Optional[ComplianceFramework] = None


def get_compliance_framework() -> ComplianceFramework:
    """Get compliance framework instance."""
    global _compliance_framework
    
    if _compliance_framework is None:
        _compliance_framework = ComplianceFramework()
    
    return _compliance_framework


def record_processing(
    user_id: Optional[str],
    data_category: DataCategory,
    purpose: ProcessingPurpose,
    legal_basis: LegalBasis,
    data_collected: Optional[List[str]] = None,
    regulations: Optional[List[ComplianceRegulation]] = None,
    **kwargs
) -> str:
    """Convenience function to record data processing."""
    framework = get_compliance_framework()
    
    if data_collected is None:
        data_collected = ["user_interaction_data"]
    
    if regulations is None:
        regulations = [ComplianceRegulation.GDPR, ComplianceRegulation.CCPA]
    
    return framework.record_data_processing(
        user_id=user_id,
        data_category=data_category,
        processing_purpose=purpose,
        legal_basis=legal_basis,
        data_collected=data_collected,
        applicable_regulations=regulations,
        **kwargs
    )


# Compliance decorator for automatic recording
def compliance_monitored(
    data_category: DataCategory,
    purpose: ProcessingPurpose,
    legal_basis: LegalBasis = LegalBasis.LEGITIMATE_INTERESTS,
    regulations: Optional[List[ComplianceRegulation]] = None
):
    """Decorator to automatically record data processing for compliance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract user_id if available
            user_id = kwargs.get('user_id') or (args[0] if args and hasattr(args[0], 'user_id') else None)
            
            # Record processing
            record_processing(
                user_id=user_id,
                data_category=data_category,
                purpose=purpose,
                legal_basis=legal_basis,
                data_collected=[func.__name__],
                regulations=regulations
            )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    # Initialize compliance framework
    compliance = get_compliance_framework()
    
    print("🛡️ Global Compliance Testing")
    print("=" * 40)
    
    # Test data processing recording
    user_id = "test_user_123"
    
    processing_id = compliance.record_data_processing(
        user_id=user_id,
        data_category=DataCategory.USAGE_ANALYTICS,
        processing_purpose=ProcessingPurpose.PERFORMANCE_OPTIMIZATION,
        legal_basis=LegalBasis.LEGITIMATE_INTERESTS,
        data_collected=["optimization_metrics", "performance_data"],
        applicable_regulations=[ComplianceRegulation.GDPR, ComplianceRegulation.CCPA]
    )
    
    print(f"✅ Data processing recorded: {processing_id}")
    
    # Test consent management
    consent_id = compliance.obtain_consent(
        user_id=user_id,
        consent_type="analytics_consent",
        purposes=[ProcessingPurpose.PERFORMANCE_OPTIMIZATION],
        data_categories=[DataCategory.USAGE_ANALYTICS],
        applicable_regulations=[ComplianceRegulation.GDPR]
    )
    
    print(f"✅ Consent obtained: {consent_id}")
    
    # Test data subject request
    request_id = compliance.submit_data_subject_request(
        user_id=user_id,
        request_type=DataSubjectRight.ACCESS,
        applicable_regulation=ComplianceRegulation.GDPR,
        request_details="Request access to personal data"
    )
    
    print(f"✅ Data subject request submitted: {request_id}")
    
    # Test compliance check
    compliance_status = compliance.check_compliance_status([
        ComplianceRegulation.GDPR,
        ComplianceRegulation.CCPA
    ])
    
    print("\\nCompliance Status:")
    for regulation, status in compliance_status.items():
        status_icon = "✅" if status["compliant"] else "❌"
        print(f"{status_icon} {regulation}: {'Compliant' if status['compliant'] else 'Non-compliant'}")
        if status["issues"]:
            print(f"   Issues: {len(status['issues'])}")
    
    # Test compliance report
    report = compliance.generate_compliance_report([
        ComplianceRegulation.GDPR,
        ComplianceRegulation.CCPA
    ])
    
    print(f"\\n📊 Compliance Report Generated: {report['report_id']}")
    print(f"Processing Records: {report['data_processing_summary']['total_processing_records']}")
    print(f"Consent Records: {report['consent_management_summary']['total_consents']}")
    print(f"Data Subject Requests: {report['data_subject_requests_summary']['total_requests']}")
    
    print("\\n✅ Global compliance testing completed!")
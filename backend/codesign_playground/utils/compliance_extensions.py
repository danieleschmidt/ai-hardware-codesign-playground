"""
Extended compliance features for GDPR, CCPA, PDPA, and other privacy regulations.

This module provides additional compliance functionality including data breach management,
automated compliance monitoring, and region-specific privacy law support.
"""

import json
import time
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import logging

from .compliance import (
    ComplianceManager, DataCategory, ConsentType, ComplianceRegion,
    DataProcessingRecord, AuditLogEntry
)
from .logging import get_logger

logger = get_logger(__name__)


class BreachSeverity(Enum):
    """Data breach severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BreachStatus(Enum):
    """Data breach containment status."""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    RESOLVED = "resolved"
    REPORTED = "reported"


@dataclass
class DataBreach:
    """Data breach incident record."""
    
    id: str
    timestamp: float
    severity: BreachSeverity
    affected_users: int
    data_categories: List[DataCategory]
    description: str
    containment_status: BreachStatus
    notification_status: str
    regulatory_reported: bool = False
    resolution_notes: List[str] = field(default_factory=list)
    affected_jurisdictions: List[ComplianceRegion] = field(default_factory=list)
    estimated_records: Optional[int] = None
    discovery_method: str = "automated_monitoring"
    root_cause: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            **asdict(self),
            "data_categories": [cat.value for cat in self.data_categories],
            "affected_jurisdictions": [region.value for region in self.affected_jurisdictions]
        }


@dataclass
class PrivacyImpactAssessment:
    """Privacy Impact Assessment (PIA) record."""
    
    id: str
    timestamp: float
    project_name: str
    data_categories: List[DataCategory]
    processing_purposes: List[str]
    legal_bases: List[str]
    risk_level: str  # low, medium, high, very_high
    mitigation_measures: List[str]
    approved: bool = False
    approval_date: Optional[float] = None
    review_date: Optional[float] = None
    dpo_review: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            **asdict(self),
            "data_categories": [cat.value for cat in self.data_categories]
        }


class EnhancedComplianceManager(ComplianceManager):
    """Enhanced compliance manager with advanced features."""
    
    def __init__(self, *args, **kwargs):
        """Initialize enhanced compliance manager."""
        super().__init__(*args, **kwargs)
        
        self._breach_incidents: List[DataBreach] = []
        self._privacy_assessments: List[PrivacyImpactAssessment] = []
        self._automated_monitoring_enabled = True
        self._breach_notification_threshold = 72  # hours
        
        # CCPA specific tracking
        self._ccpa_sale_opt_outs: Dict[str, bool] = {}
        self._ccpa_do_not_sell_requests: Dict[str, float] = {}
        
        # PDPA specific tracking
        self._pdpa_notification_consents: Dict[str, bool] = {}
        
        self.logger.info("Enhanced compliance manager initialized with breach monitoring")
    
    def report_data_breach(self,
                          description: str,
                          affected_users: int,
                          data_categories: List[DataCategory],
                          severity: BreachSeverity = BreachSeverity.MEDIUM,
                          discovery_method: str = "automated_monitoring",
                          estimated_records: Optional[int] = None) -> str:
        """
        Report a data breach incident.
        
        Args:
            description: Description of the breach
            affected_users: Number of affected users
            data_categories: Categories of data involved
            severity: Severity level of the breach
            discovery_method: How the breach was discovered
            estimated_records: Estimated number of records affected
            
        Returns:
            Breach incident ID
        """
        breach_id = str(uuid.uuid4())
        timestamp = time.time()
        
        # Determine affected jurisdictions based on user data
        affected_jurisdictions = self._determine_affected_jurisdictions(affected_users)
        
        breach = DataBreach(
            id=breach_id,
            timestamp=timestamp,
            severity=severity,
            affected_users=affected_users,
            data_categories=data_categories,
            description=description,
            containment_status=BreachStatus.DETECTED,
            notification_status="pending",
            affected_jurisdictions=affected_jurisdictions,
            estimated_records=estimated_records,
            discovery_method=discovery_method
        )
        
        self._breach_incidents.append(breach)
        self._store_breach_incident(breach)
        
        # Log critical audit event
        self._log_audit_event(
            user_id=None,
            action="data_breach_reported",
            resource=f"breach_incident:{breach_id}",
            data_category=data_categories[0] if data_categories else DataCategory.SYSTEM_LOGS,
            legal_basis="legal_obligation",
            result="success",
            details={
                "breach_id": breach_id,
                "severity": severity.value,
                "affected_users": affected_users,
                "data_categories": [cat.value for cat in data_categories],
                "discovery_method": discovery_method
            },
            risk_level="critical"
        )
        
        # Initiate automated response
        self._initiate_breach_response(breach)
        
        self.logger.critical(
            f"Data breach reported: {breach_id}",
            extra={
                "breach_id": breach_id,
                "severity": severity.value,
                "affected_users": affected_users,
                "data_categories": [cat.value for cat in data_categories]
            }
        )
        
        return breach_id
    
    def handle_ccpa_request(self,
                           user_id: str,
                           request_type: str,
                           details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle CCPA-specific data subject requests.
        
        Args:
            user_id: User identifier
            request_type: CCPA request type (know, delete, opt_out, access)
            details: Additional request details
            
        Returns:
            Response to the CCPA request
        """
        timestamp = time.time()
        request_id = str(uuid.uuid4())
        
        response = {
            "request_id": request_id,
            "user_id": user_id,
            "request_type": request_type,
            "timestamp": timestamp,
            "status": "processing",
            "ccpa_compliant": True
        }
        
        try:
            if request_type == "know":
                # Right to know (CCPA Section 1798.100)
                response["data"] = self._export_ccpa_data(user_id)
                response["categories_collected"] = self._get_collected_categories(user_id)
                response["purposes"] = self._get_processing_purposes(user_id)
                response["third_parties"] = self._get_third_party_sharing(user_id)
                response["status"] = "completed"
                
            elif request_type == "delete":
                # Right to delete (CCPA Section 1798.105)
                deleted_records = self._delete_user_data(user_id)
                response["deleted_records"] = deleted_records
                response["status"] = "completed"
                
            elif request_type == "opt_out":
                # Opt-out of sale (CCPA Section 1798.120)
                self._ccpa_sale_opt_outs[user_id] = True
                self._ccpa_do_not_sell_requests[user_id] = timestamp
                response["opt_out_status"] = "activated"
                response["status"] = "completed"
                
            elif request_type == "access":
                # Right to access specific pieces of information
                response["data"] = self._export_user_data(user_id)
                response["status"] = "completed"
                
            else:
                response["status"] = "error"
                response["message"] = f"Unknown CCPA request type: {request_type}"
                response["ccpa_compliant"] = False
            
            # Log audit event
            self._log_audit_event(
                user_id=user_id,
                action=f"ccpa_request_{request_type}",
                resource=f"user_data:{user_id}",
                data_category=DataCategory.PERSONAL_IDENTIFYING,
                legal_basis="legal_obligation",
                result="success" if response["status"] != "error" else "failure",
                details={
                    "request_id": request_id,
                    "request_type": request_type,
                    "ccpa_compliant": response["ccpa_compliant"]
                },
                risk_level="medium"
            )
            
        except Exception as e:
            response["status"] = "error"
            response["message"] = str(e)
            response["ccpa_compliant"] = False
            self.logger.error(f"CCPA request processing failed: {e}")
        
        return response
    
    def handle_pdpa_request(self,
                           user_id: str,
                           request_type: str,
                           details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle PDPA (Singapore) specific data subject requests.
        
        Args:
            user_id: User identifier
            request_type: PDPA request type (access, correction, withdrawal)
            details: Additional request details
            
        Returns:
            Response to the PDPA request
        """
        timestamp = time.time()
        request_id = str(uuid.uuid4())
        
        response = {
            "request_id": request_id,
            "user_id": user_id,
            "request_type": request_type,
            "timestamp": timestamp,
            "status": "processing",
            "pdpa_compliant": True
        }
        
        try:
            if request_type == "access":
                # Right to access (PDPA Section 21)
                response["data"] = self._export_user_data(user_id)
                response["status"] = "completed"
                
            elif request_type == "correction":
                # Right to correction (PDPA Section 22)
                response["message"] = "Data correction process initiated"
                response["status"] = "requires_verification"
                
            elif request_type == "withdrawal":
                # Withdrawal of consent (PDPA Section 16)
                self._handle_consent_withdrawal(user_id, details)
                response["status"] = "completed"
                
            else:
                response["status"] = "error"
                response["message"] = f"Unknown PDPA request type: {request_type}"
                response["pdpa_compliant"] = False
            
            # Log audit event
            self._log_audit_event(
                user_id=user_id,
                action=f"pdpa_request_{request_type}",
                resource=f"user_data:{user_id}",
                data_category=DataCategory.PERSONAL_IDENTIFYING,
                legal_basis="legal_obligation",
                result="success" if response["status"] != "error" else "failure",
                details={
                    "request_id": request_id,
                    "request_type": request_type,
                    "pdpa_compliant": response["pdpa_compliant"]
                },
                risk_level="medium"
            )
            
        except Exception as e:
            response["status"] = "error"
            response["message"] = str(e)
            response["pdpa_compliant"] = False
            self.logger.error(f"PDPA request processing failed: {e}")
        
        return response
    
    def conduct_privacy_impact_assessment(self,
                                        project_name: str,
                                        data_categories: List[DataCategory],
                                        processing_purposes: List[str],
                                        legal_bases: List[str]) -> str:
        """
        Conduct a Privacy Impact Assessment (PIA).
        
        Args:
            project_name: Name of the project being assessed
            data_categories: Categories of data to be processed
            processing_purposes: Purposes for data processing
            legal_bases: Legal bases for processing
            
        Returns:
            PIA ID
        """
        pia_id = str(uuid.uuid4())
        timestamp = time.time()
        
        # Assess risk level based on data categories and purposes
        risk_level = self._assess_privacy_risk(data_categories, processing_purposes)
        
        # Generate mitigation measures
        mitigation_measures = self._generate_mitigation_measures(data_categories, risk_level)
        
        pia = PrivacyImpactAssessment(
            id=pia_id,
            timestamp=timestamp,
            project_name=project_name,
            data_categories=data_categories,
            processing_purposes=processing_purposes,
            legal_bases=legal_bases,
            risk_level=risk_level,
            mitigation_measures=mitigation_measures,
            dpo_review=risk_level in ["high", "very_high"]
        )
        
        self._privacy_assessments.append(pia)
        
        # Log audit event
        self._log_audit_event(
            user_id=None,
            action="pia_conducted",
            resource=f"project:{project_name}",
            data_category=data_categories[0] if data_categories else DataCategory.SYSTEM_LOGS,
            legal_basis="legal_obligation",
            result="success",
            details={
                "pia_id": pia_id,
                "project_name": project_name,
                "risk_level": risk_level,
                "requires_dpo_review": pia.dpo_review
            },
            risk_level="medium" if risk_level in ["high", "very_high"] else "low"
        )
        
        self.logger.info(
            f"Privacy Impact Assessment conducted: {pia_id}",
            extra={
                "pia_id": pia_id,
                "project_name": project_name,
                "risk_level": risk_level,
                "requires_dpo_review": pia.dpo_review
            }
        )
        
        return pia_id
    
    def _initiate_breach_response(self, breach: DataBreach) -> None:
        """Initiate automated breach response procedures."""
        
        # Update containment status
        breach.containment_status = BreachStatus.INVESTIGATING
        
        # Check if regulatory notification is required
        if self._requires_regulatory_notification(breach):
            breach.notification_status = "regulatory_required"
            self.logger.warning(f"Breach {breach.id} requires regulatory notification within 72 hours")
        
        # Notify affected users if required
        if self._requires_user_notification(breach):
            breach.notification_status = "user_notification_required"
            self.logger.warning(f"Breach {breach.id} requires user notification")
        
        self.logger.info(f"Breach response initiated for {breach.id}")
    
    def _requires_regulatory_notification(self, breach: DataBreach) -> bool:
        """Check if breach requires regulatory notification."""
        
        # GDPR: High risk breaches must be reported within 72 hours
        if self.region == ComplianceRegion.EU:
            return breach.severity in [BreachSeverity.HIGH, BreachSeverity.CRITICAL]
        
        # CCPA: All breaches involving personal information
        if self.region == ComplianceRegion.US:
            return any(cat in [DataCategory.PERSONAL_IDENTIFYING, DataCategory.SENSITIVE_PERSONAL] 
                      for cat in breach.data_categories)
        
        # Default: High and critical severity
        return breach.severity in [BreachSeverity.HIGH, BreachSeverity.CRITICAL]
    
    def _requires_user_notification(self, breach: DataBreach) -> bool:
        """Check if breach requires user notification."""
        
        # High risk breaches affecting personal data
        sensitive_categories = [
            DataCategory.PERSONAL_IDENTIFYING,
            DataCategory.SENSITIVE_PERSONAL,
            DataCategory.FINANCIAL_DATA,
            DataCategory.HEALTH_DATA,
            DataCategory.BIOMETRIC_DATA
        ]
        
        return (breach.severity in [BreachSeverity.HIGH, BreachSeverity.CRITICAL] and
                any(cat in sensitive_categories for cat in breach.data_categories))
    
    def _determine_affected_jurisdictions(self, affected_users: int) -> List[ComplianceRegion]:
        """Determine affected jurisdictions based on breach scope."""
        # In a real implementation, this would check user locations
        # For now, return the current region
        return [self.region]
    
    def _assess_privacy_risk(self, data_categories: List[DataCategory], purposes: List[str]) -> str:
        """Assess privacy risk level for PIA."""
        
        # High risk categories
        high_risk_categories = [
            DataCategory.SENSITIVE_PERSONAL,
            DataCategory.HEALTH_DATA,
            DataCategory.BIOMETRIC_DATA,
            DataCategory.FINANCIAL_DATA
        ]
        
        # High risk purposes
        high_risk_purposes = ["profiling", "automated_decision", "large_scale_processing"]
        
        if any(cat in high_risk_categories for cat in data_categories):
            return "high"
        
        if any(purpose in high_risk_purposes for purpose in purposes):
            return "high"
        
        if len(data_categories) > 3:
            return "medium"
        
        return "low"
    
    def _generate_mitigation_measures(self, data_categories: List[DataCategory], risk_level: str) -> List[str]:
        """Generate privacy risk mitigation measures."""
        
        measures = ["Data minimization", "Purpose limitation", "Storage limitation"]
        
        if risk_level in ["medium", "high", "very_high"]:
            measures.extend([
                "Encryption at rest and in transit",
                "Access controls and authentication",
                "Regular security assessments"
            ])
        
        if risk_level in ["high", "very_high"]:
            measures.extend([
                "Privacy by design implementation",
                "Data Protection Officer consultation",
                "Regular compliance audits",
                "User consent management"
            ])
        
        sensitive_categories = [
            DataCategory.SENSITIVE_PERSONAL,
            DataCategory.HEALTH_DATA,
            DataCategory.BIOMETRIC_DATA
        ]
        
        if any(cat in sensitive_categories for cat in data_categories):
            measures.extend([
                "Explicit consent required",
                "Enhanced security measures",
                "Anonymization where possible"
            ])
        
        return measures
    
    def _store_breach_incident(self, breach: DataBreach) -> None:
        """Store breach incident in database."""
        with self._db_lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO breach_incidents (
                        id, timestamp, severity, affected_users, data_categories,
                        description, containment_status, notification_status,
                        regulatory_reported, resolution_notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    breach.id,
                    breach.timestamp,
                    breach.severity.value,
                    breach.affected_users,
                    json.dumps([cat.value for cat in breach.data_categories]),
                    breach.description,
                    breach.containment_status.value,
                    breach.notification_status,
                    breach.regulatory_reported,
                    json.dumps(breach.resolution_notes)
                ))
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                self.logger.error(f"Failed to store breach incident: {e}")
    
    def _export_ccpa_data(self, user_id: str) -> Dict[str, Any]:
        """Export user data in CCPA-compliant format."""
        base_data = self._export_user_data(user_id)
        
        # Add CCPA-specific information
        ccpa_data = {
            **base_data,
            "ccpa_opt_out_status": self._ccpa_sale_opt_outs.get(user_id, False),
            "do_not_sell_request_date": self._ccpa_do_not_sell_requests.get(user_id),
            "categories_sold": [],  # Would be populated based on actual data sharing
            "categories_disclosed": []  # Would be populated based on actual disclosures
        }
        
        return ccpa_data
    
    def _get_collected_categories(self, user_id: str) -> List[str]:
        """Get categories of personal information collected for user."""
        user_records = [r for r in self._processing_records if r.user_id == user_id]
        categories = set(r.data_category.value for r in user_records)
        return list(categories)
    
    def _get_processing_purposes(self, user_id: str) -> List[str]:
        """Get processing purposes for user data."""
        user_records = [r for r in self._processing_records if r.user_id == user_id]
        purposes = set(r.processing_purpose for r in user_records)
        return list(purposes)
    
    def _get_third_party_sharing(self, user_id: str) -> List[str]:
        """Get third party sharing information for user."""
        user_records = [r for r in self._processing_records 
                       if r.user_id == user_id and r.third_party_sharing]
        return [f"Data location: {r.data_location}" for r in user_records]
    
    def _handle_consent_withdrawal(self, user_id: str, details: Optional[Dict[str, Any]]) -> None:
        """Handle PDPA consent withdrawal."""
        if user_id in self._user_consents:
            consent = self._user_consents[user_id]
            # Revoke all non-essential consents
            for consent_type in ConsentType:
                if consent_type != ConsentType.NECESSARY:
                    consent.revoke_consent(consent_type)
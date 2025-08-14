"""
Compliance and regulatory support for AI Hardware Co-Design Playground.

Provides GDPR, CCPA, PDPA compliance features, data privacy controls,
and regulatory reporting capabilities.
"""

import json
import hashlib
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

from .exceptions import SecurityError, ValidationError

logger = logging.getLogger(__name__)


class DataCategory(Enum):
    """Categories of data for privacy compliance."""
    PERSONAL_IDENTIFYING = "personal_identifying"
    TECHNICAL_METRICS = "technical_metrics"
    USAGE_ANALYTICS = "usage_analytics"
    PERFORMANCE_DATA = "performance_data"
    SYSTEM_LOGS = "system_logs"
    MODEL_ARTIFACTS = "model_artifacts"
    DESIGN_SPECIFICATIONS = "design_specifications"


class ConsentType(Enum):
    """Types of user consent."""
    NECESSARY = "necessary"          # Required for basic functionality
    ANALYTICS = "analytics"          # Usage analytics and optimization
    MARKETING = "marketing"          # Marketing and promotional content
    RESEARCH = "research"            # Research and development
    THIRD_PARTY = "third_party"      # Third-party integrations


class ComplianceRegion(Enum):
    """Supported compliance regions."""
    EU = "eu"           # European Union (GDPR)
    US = "us"           # United States (CCPA, CPRA)
    SINGAPORE = "sg"    # Singapore (PDPA)
    BRAZIL = "br"       # Brazil (LGPD)
    CANADA = "ca"       # Canada (PIPEDA)
    JAPAN = "jp"        # Japan (APPI)
    SOUTH_KOREA = "kr"  # South Korea (PIPA)
    GLOBAL = "global"   # Global baseline


@dataclass
class DataProcessingRecord:
    """Record of data processing activities."""
    
    timestamp: float
    user_id: Optional[str]
    data_category: DataCategory
    processing_purpose: str
    legal_basis: str
    retention_period: int  # days
    data_location: str
    third_party_sharing: bool = False
    consent_given: bool = False
    consent_type: Optional[ConsentType] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "data_category": self.data_category.value,
            "processing_purpose": self.processing_purpose,
            "legal_basis": self.legal_basis,
            "retention_period": self.retention_period,
            "data_location": self.data_location,
            "third_party_sharing": self.third_party_sharing,
            "consent_given": self.consent_given,
            "consent_type": self.consent_type.value if self.consent_type else None,
        }


@dataclass
class UserConsent:
    """User consent management."""
    
    user_id: str
    timestamp: float
    consents: Dict[ConsentType, bool] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    explicit_consent: bool = False
    
    def has_consent(self, consent_type: ConsentType) -> bool:
        """Check if user has given specific consent."""
        return self.consents.get(consent_type, False)
    
    def grant_consent(self, consent_type: ConsentType) -> None:
        """Grant consent for specific type."""
        self.consents[consent_type] = True
        self.timestamp = time.time()
    
    def revoke_consent(self, consent_type: ConsentType) -> None:
        """Revoke consent for specific type."""
        self.consents[consent_type] = False
        self.timestamp = time.time()


class ComplianceManager:
    """Compliance and privacy management system."""
    
    def __init__(self, region: ComplianceRegion = ComplianceRegion.GLOBAL):
        """
        Initialize compliance manager.
        
        Args:
            region: Primary compliance region
        """
        self.region = region
        self._processing_records: List[DataProcessingRecord] = []
        self._user_consents: Dict[str, UserConsent] = {}
        self._data_retention_policies = {}
        self._anonymization_rules = {}
        
        # Load region-specific compliance rules
        self._load_compliance_rules()
        
        logger.info(f"Initialized ComplianceManager for region: {region.value}")
    
    def record_data_processing(
        self,
        user_id: Optional[str],
        data_category: DataCategory,
        processing_purpose: str,
        legal_basis: str,
        data_location: str = "local",
        consent_required: bool = True
    ) -> bool:
        """
        Record data processing activity.
        
        Args:
            user_id: User identifier (can be anonymized)
            data_category: Category of data being processed
            processing_purpose: Purpose of processing
            legal_basis: Legal basis for processing
            data_location: Where data is processed/stored
            consent_required: Whether consent is required
            
        Returns:
            True if processing is compliant
        """
        # Check consent if required
        if consent_required and user_id:
            if not self._check_consent_compliance(user_id, data_category):
                logger.warning(f"Processing blocked: insufficient consent for user {user_id}")
                return False
        
        # Get retention period from policy
        retention_period = self._get_retention_period(data_category)
        
        # Create processing record
        record = DataProcessingRecord(
            timestamp=time.time(),
            user_id=user_id,
            data_category=data_category,
            processing_purpose=processing_purpose,
            legal_basis=legal_basis,
            retention_period=retention_period,
            data_location=data_location,
            consent_given=consent_required and self._has_valid_consent(user_id, data_category)
        )
        
        self._processing_records.append(record)
        
        logger.info(
            f"Recorded data processing: {data_category.value} for {processing_purpose}",
            extra={"user_id": user_id, "legal_basis": legal_basis}
        )
        
        return True
    
    def manage_user_consent(
        self,
        user_id: str,
        consent_updates: Dict[ConsentType, bool],
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> None:
        """
        Manage user consent preferences.
        
        Args:
            user_id: User identifier
            consent_updates: Dictionary of consent type updates
            ip_address: User's IP address for audit trail
            user_agent: User's browser/client info
        """
        if user_id not in self._user_consents:
            self._user_consents[user_id] = UserConsent(
                user_id=user_id,
                timestamp=time.time(),
                ip_address=ip_address,
                user_agent=user_agent,
                explicit_consent=True
            )
        
        consent_record = self._user_consents[user_id]
        
        for consent_type, granted in consent_updates.items():
            if granted:
                consent_record.grant_consent(consent_type)
            else:
                consent_record.revoke_consent(consent_type)
        
        logger.info(
            f"Updated consent for user {user_id}",
            extra={"consents": consent_updates, "ip": ip_address}
        )
    
    def handle_data_subject_request(
        self,
        user_id: str,
        request_type: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle data subject rights requests (GDPR Article 15-22).
        
        Args:
            user_id: User identifier
            request_type: Type of request (access, rectification, erasure, etc.)
            details: Additional request details
            
        Returns:
            Response to the data subject request
        """
        response = {
            "user_id": user_id,
            "request_type": request_type,
            "timestamp": time.time(),
            "status": "processing"
        }
        
        try:
            if request_type == "access":
                # Right to access (Article 15)
                response["data"] = self._export_user_data(user_id)
                response["status"] = "completed"
                
            elif request_type == "rectification":
                # Right to rectification (Article 16)
                response["message"] = "Data rectification process initiated"
                response["status"] = "requires_action"
                
            elif request_type == "erasure":
                # Right to erasure / "Right to be forgotten" (Article 17)
                deleted_records = self._delete_user_data(user_id)
                response["deleted_records"] = deleted_records
                response["status"] = "completed"
                
            elif request_type == "portability":
                # Right to data portability (Article 20)
                response["data"] = self._export_portable_data(user_id)
                response["format"] = "json"
                response["status"] = "completed"
                
            elif request_type == "object":
                # Right to object (Article 21)
                self._handle_processing_objection(user_id, details)
                response["status"] = "completed"
                
            else:
                response["status"] = "error"
                response["message"] = f"Unknown request type: {request_type}"
            
            logger.info(
                f"Processed data subject request: {request_type} for user {user_id}",
                extra={"status": response["status"]}
            )
            
        except Exception as e:
            response["status"] = "error"
            response["message"] = str(e)
            logger.error(f"Error processing data subject request: {e}")
        
        return response
    
    def anonymize_data(self, data: Dict[str, Any], anonymization_level: str = "standard") -> Dict[str, Any]:
        """
        Anonymize personal data according to compliance requirements.
        
        Args:
            data: Data to anonymize
            anonymization_level: Level of anonymization (basic, standard, strong)
            
        Returns:
            Anonymized data
        """
        anonymized = data.copy()
        
        # Define anonymization rules
        rules = self._anonymization_rules.get(anonymization_level, {})
        
        for field, rule in rules.items():
            if field in anonymized:
                if rule == "hash":
                    anonymized[field] = hashlib.sha256(str(anonymized[field]).encode()).hexdigest()[:16]
                elif rule == "remove":
                    del anonymized[field]
                elif rule == "generalize":
                    anonymized[field] = self._generalize_value(anonymized[field])
                elif rule == "mask":
                    anonymized[field] = "*" * len(str(anonymized[field]))
        
        return anonymized
    
    def generate_compliance_report(self, start_date: float, end_date: float) -> Dict[str, Any]:
        """
        Generate compliance report for audit purposes.
        
        Args:
            start_date: Start timestamp
            end_date: End timestamp
            
        Returns:
            Compliance report
        """
        # Filter records by date range
        period_records = [
            r for r in self._processing_records
            if start_date <= r.timestamp <= end_date
        ]
        
        # Calculate statistics
        total_processing_activities = len(period_records)
        categories_processed = len(set(r.data_category for r in period_records))
        consent_based_processing = sum(1 for r in period_records if r.consent_given)
        
        # Consent statistics
        total_users = len(self._user_consents)
        consent_rates = {}
        for consent_type in ConsentType:
            granted = sum(1 for consent in self._user_consents.values() if consent.has_consent(consent_type))
            consent_rates[consent_type.value] = granted / max(total_users, 1)
        
        # Data retention compliance
        expired_data = self._check_data_retention_compliance()
        
        report = {
            "report_period": {
                "start_date": start_date,
                "end_date": end_date
            },
            "compliance_region": self.region.value,
            "processing_activities": {
                "total": total_processing_activities,
                "categories_processed": categories_processed,
                "consent_based": consent_based_processing,
                "consent_rate": consent_based_processing / max(total_processing_activities, 1)
            },
            "user_consent": {
                "total_users": total_users,
                "consent_rates": consent_rates
            },
            "data_retention": {
                "expired_records": len(expired_data),
                "retention_compliance": len(expired_data) == 0
            },
            "data_subject_requests": self._get_request_statistics(start_date, end_date),
            "compliance_status": "compliant" if len(expired_data) == 0 else "attention_required"
        }
        
        logger.info(f"Generated compliance report for period {start_date} to {end_date}")
        return report
    
    def _check_consent_compliance(self, user_id: str, data_category: DataCategory) -> bool:
        """Check if user consent is valid for data processing."""
        if user_id not in self._user_consents:
            return False
        
        consent = self._user_consents[user_id]
        
        # Map data categories to required consent types
        consent_mapping = {
            DataCategory.PERSONAL_IDENTIFYING: ConsentType.NECESSARY,
            DataCategory.TECHNICAL_METRICS: ConsentType.ANALYTICS,
            DataCategory.USAGE_ANALYTICS: ConsentType.ANALYTICS,
            DataCategory.PERFORMANCE_DATA: ConsentType.ANALYTICS,
            DataCategory.SYSTEM_LOGS: ConsentType.NECESSARY,
            DataCategory.MODEL_ARTIFACTS: ConsentType.RESEARCH,
            DataCategory.DESIGN_SPECIFICATIONS: ConsentType.RESEARCH,
        }
        
        required_consent = consent_mapping.get(data_category, ConsentType.NECESSARY)
        return consent.has_consent(required_consent)
    
    def _has_valid_consent(self, user_id: str, data_category: DataCategory) -> bool:
        """Check if user has valid consent for data category."""
        return user_id and self._check_consent_compliance(user_id, data_category)
    
    def _get_retention_period(self, data_category: DataCategory) -> int:
        """Get retention period for data category."""
        return self._data_retention_policies.get(data_category, 365)  # Default 1 year
    
    def _export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all data associated with a user."""
        user_records = [r for r in self._processing_records if r.user_id == user_id]
        user_consent = self._user_consents.get(user_id)
        
        return {
            "user_id": user_id,
            "processing_records": [r.to_dict() for r in user_records],
            "consent_record": {
                "consents": user_consent.consents if user_consent else {},
                "consent_timestamp": user_consent.timestamp if user_consent else None,
                "explicit_consent": user_consent.explicit_consent if user_consent else False
            } if user_consent else None,
            "export_timestamp": time.time()
        }
    
    def _export_portable_data(self, user_id: str) -> Dict[str, Any]:
        """Export user data in portable format."""
        # Return user's data in a structured, machine-readable format
        return self._export_user_data(user_id)
    
    def _delete_user_data(self, user_id: str) -> int:
        """Delete all data associated with a user."""
        # Remove processing records
        initial_count = len(self._processing_records)
        self._processing_records = [r for r in self._processing_records if r.user_id != user_id]
        deleted_records = initial_count - len(self._processing_records)
        
        # Remove consent records
        if user_id in self._user_consents:
            del self._user_consents[user_id]
        
        return deleted_records
    
    def _handle_processing_objection(self, user_id: str, details: Optional[Dict[str, Any]]) -> None:
        """Handle user objection to data processing."""
        # Mark user as objecting to certain types of processing
        if user_id in self._user_consents:
            consent = self._user_consents[user_id]
            # Revoke non-essential consents
            consent.revoke_consent(ConsentType.ANALYTICS)
            consent.revoke_consent(ConsentType.MARKETING)
            consent.revoke_consent(ConsentType.RESEARCH)
    
    def _check_data_retention_compliance(self) -> List[DataProcessingRecord]:
        """Check for data that has exceeded retention periods."""
        current_time = time.time()
        expired_records = []
        
        for record in self._processing_records:
            retention_seconds = record.retention_period * 24 * 3600
            if current_time - record.timestamp > retention_seconds:
                expired_records.append(record)
        
        return expired_records
    
    def _get_request_statistics(self, start_date: float, end_date: float) -> Dict[str, Any]:
        """Get statistics on data subject requests."""
        # This would track actual requests in a real implementation
        return {
            "access_requests": 0,
            "rectification_requests": 0,
            "erasure_requests": 0,
            "portability_requests": 0,
            "objection_requests": 0
        }
    
    def _generalize_value(self, value: Any) -> Any:
        """Apply generalization to a value."""
        if isinstance(value, str) and len(value) > 3:
            return value[:3] + "*" * (len(value) - 3)
        elif isinstance(value, (int, float)):
            # Round to nearest 10
            return round(value / 10) * 10
        return value
    
    def _load_compliance_rules(self) -> None:
        """Load region-specific compliance rules."""
        
        # Data retention policies (in days)
        if self.region == ComplianceRegion.EU:
            # GDPR requirements
            self._data_retention_policies = {
                DataCategory.PERSONAL_IDENTIFYING: 1095,      # 3 years
                DataCategory.TECHNICAL_METRICS: 365,          # 1 year
                DataCategory.USAGE_ANALYTICS: 730,            # 2 years
                DataCategory.PERFORMANCE_DATA: 365,           # 1 year
                DataCategory.SYSTEM_LOGS: 90,                 # 3 months
                DataCategory.MODEL_ARTIFACTS: 2190,           # 6 years
                DataCategory.DESIGN_SPECIFICATIONS: 2555,     # 7 years
            }
        elif self.region == ComplianceRegion.US:
            # CCPA/CPRA requirements
            self._data_retention_policies = {
                DataCategory.PERSONAL_IDENTIFYING: 730,       # 2 years
                DataCategory.TECHNICAL_METRICS: 365,          # 1 year
                DataCategory.USAGE_ANALYTICS: 730,            # 2 years
                DataCategory.PERFORMANCE_DATA: 365,           # 1 year
                DataCategory.SYSTEM_LOGS: 90,                 # 3 months
                DataCategory.MODEL_ARTIFACTS: 2190,           # 6 years
                DataCategory.DESIGN_SPECIFICATIONS: 2555,     # 7 years
            }
        else:
            # Global baseline
            self._data_retention_policies = {
                DataCategory.PERSONAL_IDENTIFYING: 365,       # 1 year
                DataCategory.TECHNICAL_METRICS: 180,          # 6 months
                DataCategory.USAGE_ANALYTICS: 365,            # 1 year
                DataCategory.PERFORMANCE_DATA: 180,           # 6 months
                DataCategory.SYSTEM_LOGS: 30,                 # 1 month
                DataCategory.MODEL_ARTIFACTS: 1095,           # 3 years
                DataCategory.DESIGN_SPECIFICATIONS: 1825,     # 5 years
            }
        
        # Anonymization rules
        self._anonymization_rules = {
            "basic": {
                "user_id": "hash",
                "ip_address": "mask",
                "email": "hash"
            },
            "standard": {
                "user_id": "hash",
                "ip_address": "remove",
                "email": "hash",
                "name": "generalize",
                "location": "generalize"
            },
            "strong": {
                "user_id": "remove",
                "ip_address": "remove",
                "email": "remove",
                "name": "remove",
                "location": "remove",
                "device_id": "hash"
            }
        }


# Global compliance manager instance
_compliance_manager: Optional[ComplianceManager] = None


def get_compliance_manager(region: ComplianceRegion = ComplianceRegion.GLOBAL) -> ComplianceManager:
    """Get global compliance manager instance."""
    global _compliance_manager
    
    if _compliance_manager is None:
        _compliance_manager = ComplianceManager(region)
    
    return _compliance_manager


def record_processing(
    user_id: Optional[str],
    data_category: DataCategory,
    purpose: str,
    legal_basis: str = "legitimate_interest"
) -> bool:
    """
    Convenience function to record data processing.
    
    Args:
        user_id: User identifier
        data_category: Category of data
        purpose: Processing purpose
        legal_basis: Legal basis for processing
        
    Returns:
        True if processing is compliant
    """
    return get_compliance_manager().record_data_processing(
        user_id, data_category, purpose, legal_basis
    )


def check_consent(user_id: str, consent_type: ConsentType) -> bool:
    """
    Convenience function to check user consent.
    
    Args:
        user_id: User identifier
        consent_type: Type of consent to check
        
    Returns:
        True if consent is granted
    """
    manager = get_compliance_manager()
    if user_id not in manager._user_consents:
        return False
    return manager._user_consents[user_id].has_consent(consent_type)
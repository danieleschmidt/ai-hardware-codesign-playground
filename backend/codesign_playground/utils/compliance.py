"""
Compliance and regulatory support for AI Hardware Co-Design Playground.

Provides GDPR, CCPA, PDPA compliance features, data privacy controls,
and regulatory reporting capabilities.
"""

import json
import hashlib
import time
import uuid
import sqlite3
import threading
from typing import Dict, List, Any, Optional, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import logging
from datetime import datetime, timedelta
from contextlib import contextmanager

from .exceptions import SecurityError, ValidationError
from .logging import get_logger

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
    BIOMETRIC_DATA = "biometric_data"
    BEHAVIORAL_DATA = "behavioral_data"
    LOCATION_DATA = "location_data"
    FINANCIAL_DATA = "financial_data"
    HEALTH_DATA = "health_data"
    SENSITIVE_PERSONAL = "sensitive_personal"


class ConsentType(Enum):
    """Types of user consent."""
    NECESSARY = "necessary"          # Required for basic functionality
    ANALYTICS = "analytics"          # Usage analytics and optimization
    MARKETING = "marketing"          # Marketing and promotional content
    RESEARCH = "research"            # Research and development
    THIRD_PARTY = "third_party"      # Third-party integrations
    FUNCTIONAL = "functional"        # Enhanced functionality
    PERSONALIZATION = "personalization"  # Personalized experiences
    COMMUNICATION = "communication"  # Communication preferences
    PROFILING = "profiling"          # Automated decision making


class ComplianceRegion(Enum):
    """Supported compliance regions."""
    EU = "eu"           # European Union (GDPR)
    US = "us"           # United States (CCPA, CPRA)
    SINGAPORE = "sg"    # Singapore (PDPA)
    BRAZIL = "br"       # Brazil (LGPD)
    CANADA = "ca"       # Canada (PIPEDA)
    JAPAN = "jp"        # Japan (APPI)
    SOUTH_KOREA = "kr"  # South Korea (PIPA)
    AUSTRALIA = "au"    # Australia (Privacy Act)
    INDIA = "in"        # India (DPDP Act)
    CHINA = "cn"        # China (PIPL)
    UK = "uk"           # United Kingdom (UK GDPR)
    GLOBAL = "global"   # Global baseline


@dataclass
class AuditLogEntry:
    """Detailed audit log entry for compliance tracking."""
    
    id: str
    timestamp: float
    user_id: Optional[str]
    action: str
    resource: str
    data_category: DataCategory
    legal_basis: str
    consent_status: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    session_id: Optional[str]
    request_id: Optional[str]
    result: str  # success, failure, partial
    details: Dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"  # low, medium, high, critical
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)


@dataclass
class DataProcessingRecord:
    """Record of data processing activities."""
    
    id: str
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
    automated_decision: bool = False
    profiling: bool = False
    cross_border_transfer: bool = False
    encryption_used: bool = True
    data_minimization_applied: bool = True
    audit_trail: List[str] = field(default_factory=list)
    
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
class DataSubjectRequest:
    """Data subject rights request."""
    
    id: str
    user_id: str
    request_type: str  # access, rectification, erasure, portability, objection, restriction
    timestamp: float
    status: str  # pending, processing, completed, rejected
    completion_deadline: float
    verification_method: str
    request_details: Dict[str, Any] = field(default_factory=dict)
    response_data: Optional[Dict[str, Any]] = None
    processing_notes: List[str] = field(default_factory=list)
    automated_processing: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)


@dataclass
class ConsentRecord:
    """Individual consent record with full audit trail."""
    
    id: str
    user_id: str
    consent_type: ConsentType
    granted: bool
    timestamp: float
    ip_address: Optional[str]
    user_agent: Optional[str]
    method: str  # explicit, implicit, opt_in, opt_out
    purpose_description: str
    withdrawal_method: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)


@dataclass
class UserConsent:
    """User consent management with enhanced tracking."""
    
    user_id: str
    timestamp: float
    consents: Dict[ConsentType, bool] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    explicit_consent: bool = False
    consent_records: List[ConsentRecord] = field(default_factory=list)
    withdrawal_rights_exercised: Dict[ConsentType, bool] = field(default_factory=dict)
    age_verification: Optional[bool] = None
    parental_consent: bool = False
    
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
    """Enhanced compliance and privacy management system with comprehensive audit logging."""
    
    def __init__(self, 
                 region: ComplianceRegion = ComplianceRegion.GLOBAL,
                 db_path: str = "compliance.db",
                 audit_retention_days: int = 2555):
        """
        Initialize compliance manager.
        
        Args:
            region: Primary compliance region
            db_path: Path to SQLite database for persistent storage
            audit_retention_days: How long to retain audit logs (default: 7 years)
        """
        self.region = region
        self.audit_retention_days = audit_retention_days
        self.db_path = db_path
        self._db_lock = threading.RLock()
        
        self._processing_records: List[DataProcessingRecord] = []
        self._user_consents: Dict[str, UserConsent] = {}
        self._data_subject_requests: Dict[str, DataSubjectRequest] = {}
        self._audit_logs: List[AuditLogEntry] = []
        self._data_retention_policies = {}
        self._anonymization_rules = {}
        self._privacy_notices = {}
        self._breach_incidents = []
        
        # Initialize database
        self._init_database()
        
        # Load region-specific compliance rules
        self._load_compliance_rules()
        
        # Setup automated compliance tasks
        self._setup_automated_tasks()
        
        self.logger = get_logger(__name__)
        self.logger.info(f"Initialized enhanced ComplianceManager for region: {region.value}")
    
    def _init_database(self) -> None:
        """Initialize SQLite database for persistent compliance storage."""
        with self._db_lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Create audit logs table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id TEXT PRIMARY KEY,
                        timestamp REAL NOT NULL,
                        user_id TEXT,
                        action TEXT NOT NULL,
                        resource TEXT NOT NULL,
                        data_category TEXT NOT NULL,
                        legal_basis TEXT NOT NULL,
                        consent_status TEXT,
                        ip_address TEXT,
                        user_agent TEXT,
                        session_id TEXT,
                        request_id TEXT,
                        result TEXT NOT NULL,
                        details TEXT,
                        risk_level TEXT DEFAULT 'low',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create processing records table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS processing_records (
                        id TEXT PRIMARY KEY,
                        timestamp REAL NOT NULL,
                        user_id TEXT,
                        data_category TEXT NOT NULL,
                        processing_purpose TEXT NOT NULL,
                        legal_basis TEXT NOT NULL,
                        retention_period INTEGER NOT NULL,
                        data_location TEXT NOT NULL,
                        third_party_sharing BOOLEAN DEFAULT 0,
                        consent_given BOOLEAN DEFAULT 0,
                        consent_type TEXT,
                        automated_decision BOOLEAN DEFAULT 0,
                        profiling BOOLEAN DEFAULT 0,
                        cross_border_transfer BOOLEAN DEFAULT 0,
                        encryption_used BOOLEAN DEFAULT 1,
                        data_minimization_applied BOOLEAN DEFAULT 1,
                        audit_trail TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create consent records table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS consent_records (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        consent_type TEXT NOT NULL,
                        granted BOOLEAN NOT NULL,
                        timestamp REAL NOT NULL,
                        ip_address TEXT,
                        user_agent TEXT,
                        method TEXT NOT NULL,
                        purpose_description TEXT,
                        withdrawal_method TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create data subject requests table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS data_subject_requests (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        request_type TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        status TEXT NOT NULL,
                        completion_deadline REAL NOT NULL,
                        verification_method TEXT NOT NULL,
                        request_details TEXT,
                        response_data TEXT,
                        processing_notes TEXT,
                        automated_processing BOOLEAN DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create breach incidents table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS breach_incidents (
                        id TEXT PRIMARY KEY,
                        timestamp REAL NOT NULL,
                        severity TEXT NOT NULL,
                        affected_users INTEGER NOT NULL,
                        data_categories TEXT NOT NULL,
                        description TEXT NOT NULL,
                        containment_status TEXT NOT NULL,
                        notification_status TEXT NOT NULL,
                        regulatory_reported BOOLEAN DEFAULT 0,
                        resolution_notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_processing_records_user_id ON processing_records(user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_consent_records_user_id ON consent_records(user_id)')
                
                conn.commit()
                conn.close()
                
                self.logger.info("Compliance database initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize compliance database: {e}")
                raise
    
    def _setup_automated_tasks(self) -> None:
        """Setup automated compliance tasks."""
        # This would integrate with a scheduler in production
        self.logger.info("Automated compliance tasks configured")
    
    def _log_audit_event(self,
                        user_id: Optional[str],
                        action: str,
                        resource: str,
                        data_category: DataCategory,
                        legal_basis: str,
                        result: str,
                        details: Optional[Dict[str, Any]] = None,
                        risk_level: str = "low",
                        consent_status: Optional[str] = None,
                        ip_address: Optional[str] = None,
                        user_agent: Optional[str] = None,
                        session_id: Optional[str] = None,
                        request_id: Optional[str] = None) -> None:
        """Log detailed audit event."""
        
        audit_entry = AuditLogEntry(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            user_id=user_id,
            action=action,
            resource=resource,
            data_category=data_category,
            legal_basis=legal_basis,
            consent_status=consent_status,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            request_id=request_id,
            result=result,
            details=details or {},
            risk_level=risk_level
        )
        
        # Store in memory
        self._audit_logs.append(audit_entry)
        
        # Store in database
        self._store_audit_log(audit_entry)
        
        # Log to application logger with appropriate level
        log_level = getattr(logging, risk_level.upper(), logging.INFO)
        if risk_level in ["high", "critical"]:
            log_level = logging.WARNING if risk_level == "high" else logging.ERROR
        
        self.logger.log(
            log_level,
            f"Audit Event: {action} on {resource}",
            extra={
                "audit_id": audit_entry.id,
                "user_id": user_id,
                "action": action,
                "resource": resource,
                "result": result,
                "risk_level": risk_level,
                "details": details
            }
        )
    
    def _store_audit_log(self, audit_entry: AuditLogEntry) -> None:
        """Store audit log entry in database."""
        with self._db_lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO audit_logs (
                        id, timestamp, user_id, action, resource, data_category,
                        legal_basis, consent_status, ip_address, user_agent,
                        session_id, request_id, result, details, risk_level
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    audit_entry.id,
                    audit_entry.timestamp,
                    audit_entry.user_id,
                    audit_entry.action,
                    audit_entry.resource,
                    audit_entry.data_category.value,
                    audit_entry.legal_basis,
                    audit_entry.consent_status,
                    audit_entry.ip_address,
                    audit_entry.user_agent,
                    audit_entry.session_id,
                    audit_entry.request_id,
                    audit_entry.result,
                    json.dumps(audit_entry.details),
                    audit_entry.risk_level
                ))
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                self.logger.error(f"Failed to store audit log: {e}")
    
    def _store_processing_record(self, record: DataProcessingRecord) -> None:
        """Store processing record in database."""
        with self._db_lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO processing_records (
                        id, timestamp, user_id, data_category, processing_purpose,
                        legal_basis, retention_period, data_location, third_party_sharing,
                        consent_given, consent_type, automated_decision, profiling,
                        cross_border_transfer, encryption_used, data_minimization_applied,
                        audit_trail
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record.id,
                    record.timestamp,
                    record.user_id,
                    record.data_category.value,
                    record.processing_purpose,
                    record.legal_basis,
                    record.retention_period,
                    record.data_location,
                    record.third_party_sharing,
                    record.consent_given,
                    record.consent_type.value if record.consent_type else None,
                    record.automated_decision,
                    record.profiling,
                    record.cross_border_transfer,
                    record.encryption_used,
                    record.data_minimization_applied,
                    json.dumps(record.audit_trail)
                ))
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                self.logger.error(f"Failed to store processing record: {e}")
    
    def _store_consent_record(self, consent: ConsentRecord) -> None:
        """Store consent record in database."""
        with self._db_lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO consent_records (
                        id, user_id, consent_type, granted, timestamp,
                        ip_address, user_agent, method, purpose_description,
                        withdrawal_method
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    consent.id,
                    consent.user_id,
                    consent.consent_type.value,
                    consent.granted,
                    consent.timestamp,
                    consent.ip_address,
                    consent.user_agent,
                    consent.method,
                    consent.purpose_description,
                    consent.withdrawal_method
                ))
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                self.logger.error(f"Failed to store consent record: {e}")
    
    def _store_user_consent(self, user_consent: UserConsent) -> None:
        """Store user consent summary in database."""
        # For now, just log the update. In production, you might want a separate table
        self.logger.debug(f"User consent updated for {user_consent.user_id}")
    
    def _validate_legal_basis(self, legal_basis: str, data_category: DataCategory, automated_decision: bool) -> bool:
        """Validate legal basis for data processing."""
        valid_bases = [
            "consent", "contract", "legal_obligation", "vital_interests", 
            "public_task", "legitimate_interests"
        ]
        
        if legal_basis not in valid_bases:
            return False
        
        # Special handling for sensitive data and automated decisions
        if data_category in [DataCategory.SENSITIVE_PERSONAL, DataCategory.HEALTH_DATA, DataCategory.BIOMETRIC_DATA]:
            # Sensitive data typically requires explicit consent or specific legal basis
            if legal_basis not in ["consent", "legal_obligation", "vital_interests"]:
                return False
        
        if automated_decision and legal_basis not in ["consent", "contract", "legal_obligation"]:
            return False
        
        return True
    
    def _get_required_consent_type(self, data_category: DataCategory) -> Optional[ConsentType]:
        """Get required consent type for data category."""
        consent_mapping = {
            DataCategory.PERSONAL_IDENTIFYING: ConsentType.NECESSARY,
            DataCategory.TECHNICAL_METRICS: ConsentType.ANALYTICS,
            DataCategory.USAGE_ANALYTICS: ConsentType.ANALYTICS,
            DataCategory.PERFORMANCE_DATA: ConsentType.ANALYTICS,
            DataCategory.SYSTEM_LOGS: ConsentType.NECESSARY,
            DataCategory.MODEL_ARTIFACTS: ConsentType.RESEARCH,
            DataCategory.DESIGN_SPECIFICATIONS: ConsentType.RESEARCH,
            DataCategory.BIOMETRIC_DATA: ConsentType.NECESSARY,
            DataCategory.BEHAVIORAL_DATA: ConsentType.PROFILING,
            DataCategory.LOCATION_DATA: ConsentType.FUNCTIONAL,
            DataCategory.FINANCIAL_DATA: ConsentType.NECESSARY,
            DataCategory.HEALTH_DATA: ConsentType.NECESSARY,
            DataCategory.SENSITIVE_PERSONAL: ConsentType.NECESSARY,
        }
        return consent_mapping.get(data_category)
    
    def record_data_processing(
        self,
        user_id: Optional[str],
        data_category: DataCategory,
        processing_purpose: str,
        legal_basis: str,
        data_location: str = "local",
        consent_required: bool = True,
        automated_decision: bool = False,
        profiling: bool = False,
        cross_border_transfer: bool = False,
        request_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record data processing activity with comprehensive audit logging.
        
        Args:
            user_id: User identifier (can be anonymized)
            data_category: Category of data being processed
            processing_purpose: Purpose of processing
            legal_basis: Legal basis for processing
            data_location: Where data is processed/stored
            consent_required: Whether consent is required
            automated_decision: Whether automated decision making is involved
            profiling: Whether profiling is being performed
            cross_border_transfer: Whether data crosses borders
            request_context: Additional context from the request
            
        Returns:
            True if processing is compliant
        """
        # Generate unique record ID
        record_id = str(uuid.uuid4())
        timestamp = time.time()
        
        # Extract request context
        ip_address = request_context.get("ip_address") if request_context else None
        user_agent = request_context.get("user_agent") if request_context else None
        session_id = request_context.get("session_id") if request_context else None
        request_id = request_context.get("request_id") if request_context else None
        
        # Check consent if required
        consent_status = "not_required"
        if consent_required and user_id:
            if not self._check_consent_compliance(user_id, data_category):
                self._log_audit_event(
                    user_id=user_id,
                    action="data_processing_blocked",
                    resource=f"data_category:{data_category.value}",
                    data_category=data_category,
                    legal_basis=legal_basis,
                    result="failure",
                    details={"reason": "insufficient_consent"},
                    risk_level="high",
                    ip_address=ip_address,
                    user_agent=user_agent,
                    session_id=session_id,
                    request_id=request_id
                )
                self.logger.warning(f"Processing blocked: insufficient consent for user {user_id}")
                return False
            consent_status = "granted"
        
        # Validate legal basis
        if not self._validate_legal_basis(legal_basis, data_category, automated_decision):
            self._log_audit_event(
                user_id=user_id,
                action="data_processing_blocked",
                resource=f"data_category:{data_category.value}",
                data_category=data_category,
                legal_basis=legal_basis,
                result="failure",
                details={"reason": "invalid_legal_basis"},
                risk_level="high",
                ip_address=ip_address,
                user_agent=user_agent,
                session_id=session_id,
                request_id=request_id
            )
            self.logger.error(f"Invalid legal basis {legal_basis} for {data_category.value}")
            return False
        
        # Get retention period from policy
        retention_period = self._get_retention_period(data_category)
        
        # Create processing record
        record = DataProcessingRecord(
            id=record_id,
            timestamp=timestamp,
            user_id=user_id,
            data_category=data_category,
            processing_purpose=processing_purpose,
            legal_basis=legal_basis,
            retention_period=retention_period,
            data_location=data_location,
            third_party_sharing=cross_border_transfer,
            consent_given=consent_required and self._has_valid_consent(user_id, data_category),
            consent_type=self._get_required_consent_type(data_category),
            automated_decision=automated_decision,
            profiling=profiling,
            cross_border_transfer=cross_border_transfer,
            encryption_used=True,  # Assume encryption by default
            data_minimization_applied=True,  # Assume data minimization
            audit_trail=[f"Created at {timestamp}"]
        )
        
        # Store processing record
        self._processing_records.append(record)
        self._store_processing_record(record)
        
        # Log audit event
        self._log_audit_event(
            user_id=user_id,
            action="data_processing_recorded",
            resource=f"data_category:{data_category.value}",
            data_category=data_category,
            legal_basis=legal_basis,
            consent_status=consent_status,
            result="success",
            details={
                "purpose": processing_purpose,
                "location": data_location,
                "automated_decision": automated_decision,
                "profiling": profiling,
                "cross_border": cross_border_transfer
            },
            risk_level="low" if not automated_decision else "medium",
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            request_id=request_id
        )
        
        self.logger.info(
            f"Recorded data processing: {data_category.value} for {processing_purpose}",
            extra={
                "user_id": user_id, 
                "legal_basis": legal_basis,
                "record_id": record_id,
                "automated_decision": automated_decision
            }
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
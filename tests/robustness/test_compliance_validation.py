"""
Automated Compliance Validation Framework for AI Hardware Co-Design Playground.

This module implements comprehensive automated testing for GDPR, CCPA, PDPA
and other privacy regulation compliance features.
"""

import pytest
import time
import json
import sqlite3
import tempfile
import hashlib
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from backend.codesign_playground.utils.compliance import (
    ComplianceManager, DataCategory, ConsentType, ComplianceRegion,
    DataProcessingRecord, get_compliance_manager
)
from backend.codesign_playground.utils.compliance_extensions import (
    EnhancedComplianceManager, DataBreach, BreachSeverity, BreachStatus
)


class ComplianceTestSuite:
    """Comprehensive compliance testing framework."""
    
    def __init__(self):
        self.test_regions = [
            ComplianceRegion.EU,
            ComplianceRegion.US,
            ComplianceRegion.SINGAPORE,
            ComplianceRegion.GLOBAL
        ]
        self.test_data_categories = list(DataCategory)
        self.test_consent_types = list(ConsentType)
    
    def create_test_compliance_manager(self, region: ComplianceRegion, db_path: str) -> EnhancedComplianceManager:
        """Create test compliance manager instance."""
        return EnhancedComplianceManager(region=region, db_path=db_path)
    
    def generate_test_user_id(self, prefix: str = "test_user") -> str:
        """Generate unique test user ID."""
        timestamp = int(time.time())
        return f"{prefix}_{timestamp}_{hash(str(time.time())) % 10000}"


class TestGDPRCompliance:
    """Test GDPR compliance features."""
    
    @pytest.fixture
    def compliance_suite(self):
        return ComplianceTestSuite()
    
    @pytest.fixture
    def eu_compliance_manager(self, tmp_path, compliance_suite):
        db_path = str(tmp_path / "gdpr_test.db")
        return compliance_suite.create_test_compliance_manager(ComplianceRegion.EU, db_path)
    
    def test_gdpr_data_processing_record(self, eu_compliance_manager, compliance_suite):
        """Test GDPR data processing record creation and validation."""
        user_id = compliance_suite.generate_test_user_id("gdpr_user")
        
        # Test legitimate processing
        success = eu_compliance_manager.record_data_processing(
            user_id=user_id,
            data_category=DataCategory.PERSONAL_IDENTIFYING,
            processing_purpose="user_account_management",
            legal_basis="contract"
        )
        
        assert success, "GDPR compliant processing should succeed"
        
        # Verify record was created
        processing_records = [r for r in eu_compliance_manager._processing_records if r.user_id == user_id]
        assert len(processing_records) == 1
        
        record = processing_records[0]
        assert record.data_category == DataCategory.PERSONAL_IDENTIFYING
        assert record.legal_basis == "contract"
        assert record.encryption_used is True
        assert record.data_minimization_applied is True
    
    def test_gdpr_consent_management(self, eu_compliance_manager, compliance_suite):
        """Test GDPR consent management."""
        user_id = compliance_suite.generate_test_user_id("gdpr_consent")
        
        # Test explicit consent granting
        eu_compliance_manager.manage_user_consent(
            user_id=user_id,
            consent_updates={
                ConsentType.ANALYTICS: True,
                ConsentType.MARKETING: False,
                ConsentType.RESEARCH: True
            },
            method="explicit",
            ip_address="192.168.1.100",
            purpose_descriptions={
                ConsentType.ANALYTICS: "To improve our services and user experience",
                ConsentType.RESEARCH: "For AI research and development purposes"
            }
        )
        
        # Verify consent was recorded
        consent_record = eu_compliance_manager._user_consents[user_id]
        assert consent_record.has_consent(ConsentType.ANALYTICS)
        assert not consent_record.has_consent(ConsentType.MARKETING)
        assert consent_record.has_consent(ConsentType.RESEARCH)
        assert consent_record.explicit_consent is True
        
        # Test consent withdrawal
        eu_compliance_manager.manage_user_consent(
            user_id=user_id,
            consent_updates={
                ConsentType.ANALYTICS: False,
                ConsentType.RESEARCH: False
            },
            method="explicit",
            ip_address="192.168.1.100"
        )
        
        # Verify withdrawal
        assert not consent_record.has_consent(ConsentType.ANALYTICS)
        assert not consent_record.has_consent(ConsentType.RESEARCH)
        assert consent_record.withdrawal_rights_exercised[ConsentType.ANALYTICS] is True
    
    def test_gdpr_data_subject_rights(self, eu_compliance_manager, compliance_suite):
        """Test GDPR data subject rights implementation."""
        user_id = compliance_suite.generate_test_user_id("gdpr_rights")
        
        # Create some test data
        eu_compliance_manager.record_data_processing(
            user_id=user_id,
            data_category=DataCategory.PERSONAL_IDENTIFYING,
            processing_purpose="account_creation",
            legal_basis="contract"
        )
        
        # Test Right to Access (Article 15)
        access_response = eu_compliance_manager.handle_data_subject_request(
            user_id=user_id,
            request_type="access",
            verification_method="email"
        )
        
        assert access_response["status"] == "completed"
        assert "data" in access_response
        assert access_response["data"]["user_id"] == user_id
        
        # Test Right to Rectification (Article 16)
        rectification_response = eu_compliance_manager.handle_data_subject_request(
            user_id=user_id,
            request_type="rectification",
            details={"field": "email", "new_value": "updated@example.com"}
        )
        
        assert rectification_response["status"] in ["requires_action", "completed"]
        
        # Test Right to Erasure (Article 17)
        erasure_response = eu_compliance_manager.handle_data_subject_request(
            user_id=user_id,
            request_type="erasure"
        )
        
        assert erasure_response["status"] == "completed"
        assert erasure_response["deleted_records"] > 0
        
        # Verify data was actually deleted
        remaining_records = [r for r in eu_compliance_manager._processing_records if r.user_id == user_id]
        assert len(remaining_records) == 0
    
    def test_gdpr_data_portability(self, eu_compliance_manager, compliance_suite):
        """Test GDPR data portability (Article 20)."""
        user_id = compliance_suite.generate_test_user_id("gdpr_portability")
        
        # Create test data
        eu_compliance_manager.record_data_processing(
            user_id=user_id,
            data_category=DataCategory.PERSONAL_IDENTIFYING,
            processing_purpose="service_provision",
            legal_basis="contract"
        )
        
        eu_compliance_manager.manage_user_consent(
            user_id=user_id,
            consent_updates={ConsentType.ANALYTICS: True},
            method="explicit"
        )
        
        # Test data portability request
        portability_response = eu_compliance_manager.handle_data_subject_request(
            user_id=user_id,
            request_type="portability"
        )
        
        assert portability_response["status"] == "completed"
        assert "data" in portability_response
        assert portability_response["format"] == "json"
        
        # Verify exported data structure
        exported_data = portability_response["data"]
        assert "user_id" in exported_data
        assert "processing_records" in exported_data
        assert "consent_record" in exported_data
    
    def test_gdpr_automated_decision_making(self, eu_compliance_manager, compliance_suite):
        """Test GDPR automated decision making compliance (Article 22)."""
        user_id = compliance_suite.generate_test_user_id("gdpr_automated")
        
        # Test automated decision with proper legal basis
        success = eu_compliance_manager.record_data_processing(
            user_id=user_id,
            data_category=DataCategory.PERSONAL_IDENTIFYING,
            processing_purpose="automated_credit_scoring",
            legal_basis="contract",
            automated_decision=True,
            profiling=True
        )
        
        assert success, "Automated decision with contract basis should be allowed"
        
        # Test automated decision without proper legal basis
        success = eu_compliance_manager.record_data_processing(
            user_id=user_id,
            data_category=DataCategory.PERSONAL_IDENTIFYING,
            processing_purpose="automated_profiling",
            legal_basis="legitimate_interests",
            automated_decision=True,
            profiling=True
        )
        
        assert not success, "Automated decision without proper basis should be rejected"
    
    def test_gdpr_breach_notification(self, eu_compliance_manager, compliance_suite):
        """Test GDPR breach notification requirements."""
        # Report a high-risk breach
        breach_id = eu_compliance_manager.report_data_breach(
            description="Unauthorized access to user personal data",
            affected_users=1500,
            data_categories=[DataCategory.PERSONAL_IDENTIFYING, DataCategory.SENSITIVE_PERSONAL],
            severity=BreachSeverity.HIGH,
            estimated_records=1500
        )
        
        assert breach_id is not None
        
        # Verify breach was recorded
        breach = next((b for b in eu_compliance_manager._breach_incidents if b.id == breach_id), None)
        assert breach is not None
        assert breach.severity == BreachSeverity.HIGH
        assert breach.notification_status == "regulatory_required"
        
        # For GDPR, high-risk breaches must be reported within 72 hours
        assert "regulatory_required" in breach.notification_status


class TestCCPACompliance:
    """Test CCPA compliance features."""
    
    @pytest.fixture
    def compliance_suite(self):
        return ComplianceTestSuite()
    
    @pytest.fixture
    def us_compliance_manager(self, tmp_path, compliance_suite):
        db_path = str(tmp_path / "ccpa_test.db")
        return compliance_suite.create_test_compliance_manager(ComplianceRegion.US, db_path)
    
    def test_ccpa_consumer_rights(self, us_compliance_manager, compliance_suite):
        """Test CCPA consumer rights implementation."""
        user_id = compliance_suite.generate_test_user_id("ccpa_consumer")
        
        # Create test data
        us_compliance_manager.record_data_processing(
            user_id=user_id,
            data_category=DataCategory.PERSONAL_IDENTIFYING,
            processing_purpose="service_provision",
            legal_basis="business_purpose"
        )
        
        # Test Right to Know (CCPA Section 1798.100)
        know_response = us_compliance_manager.handle_ccpa_request(
            user_id=user_id,
            request_type="know"
        )
        
        assert know_response["status"] == "completed"
        assert know_response["ccpa_compliant"] is True
        assert "categories_collected" in know_response
        assert "purposes" in know_response
        assert "third_parties" in know_response
        
        # Test Right to Delete (CCPA Section 1798.105)
        delete_response = us_compliance_manager.handle_ccpa_request(
            user_id=user_id,
            request_type="delete"
        )
        
        assert delete_response["status"] == "completed"
        assert delete_response["ccpa_compliant"] is True
        assert delete_response["deleted_records"] > 0
    
    def test_ccpa_opt_out_of_sale(self, us_compliance_manager, compliance_suite):
        """Test CCPA opt-out of sale (Section 1798.120)."""
        user_id = compliance_suite.generate_test_user_id("ccpa_opt_out")
        
        # Test opt-out request
        opt_out_response = us_compliance_manager.handle_ccpa_request(
            user_id=user_id,
            request_type="opt_out"
        )
        
        assert opt_out_response["status"] == "completed"
        assert opt_out_response["ccpa_compliant"] is True
        assert opt_out_response["opt_out_status"] == "activated"
        
        # Verify opt-out was recorded
        assert us_compliance_manager._ccpa_sale_opt_outs[user_id] is True
        assert user_id in us_compliance_manager._ccpa_do_not_sell_requests
    
    def test_ccpa_data_categories_tracking(self, us_compliance_manager, compliance_suite):
        """Test CCPA data categories tracking."""
        user_id = compliance_suite.generate_test_user_id("ccpa_categories")
        
        # Record various data processing activities
        data_activities = [
            (DataCategory.PERSONAL_IDENTIFYING, "account_management"),
            (DataCategory.USAGE_ANALYTICS, "service_improvement"),
            (DataCategory.BEHAVIORAL_DATA, "personalization")
        ]
        
        for category, purpose in data_activities:
            us_compliance_manager.record_data_processing(
                user_id=user_id,
                data_category=category,
                processing_purpose=purpose,
                legal_basis="business_purpose"
            )
        
        # Test data export
        export_response = us_compliance_manager.handle_ccpa_request(
            user_id=user_id,
            request_type="know"
        )
        
        # Verify all categories are tracked
        categories_collected = export_response["categories_collected"]
        expected_categories = [cat.value for cat, _ in data_activities]
        
        for expected_cat in expected_categories:
            assert expected_cat in categories_collected


class TestPDPACompliance:
    """Test PDPA (Singapore) compliance features."""
    
    @pytest.fixture
    def compliance_suite(self):
        return ComplianceTestSuite()
    
    @pytest.fixture
    def sg_compliance_manager(self, tmp_path, compliance_suite):
        db_path = str(tmp_path / "pdpa_test.db")
        return compliance_suite.create_test_compliance_manager(ComplianceRegion.SINGAPORE, db_path)
    
    def test_pdpa_consent_requirements(self, sg_compliance_manager, compliance_suite):
        """Test PDPA consent requirements."""
        user_id = compliance_suite.generate_test_user_id("pdpa_consent")
        
        # PDPA requires consent for most personal data collection
        sg_compliance_manager.manage_user_consent(
            user_id=user_id,
            consent_updates={
                ConsentType.NECESSARY: True,
                ConsentType.ANALYTICS: True
            },
            method="explicit"
        )
        
        # Test processing with consent
        success = sg_compliance_manager.record_data_processing(
            user_id=user_id,
            data_category=DataCategory.PERSONAL_IDENTIFYING,
            processing_purpose="service_provision",
            legal_basis="consent"
        )
        
        assert success, "Processing with consent should succeed"
        
        # Test processing without consent (should fail)
        success = sg_compliance_manager.record_data_processing(
            user_id=user_id,
            data_category=DataCategory.MARKETING,
            processing_purpose="marketing_campaigns",
            legal_basis="consent",
            consent_required=True
        )
        
        assert not success, "Processing without consent should fail"
    
    def test_pdpa_data_subject_requests(self, sg_compliance_manager, compliance_suite):
        """Test PDPA data subject requests."""
        user_id = compliance_suite.generate_test_user_id("pdpa_requests")
        
        # Create test data
        sg_compliance_manager.record_data_processing(
            user_id=user_id,
            data_category=DataCategory.PERSONAL_IDENTIFYING,
            processing_purpose="account_management",
            legal_basis="legitimate_interests"
        )
        
        # Test access request (PDPA Section 21)
        access_response = sg_compliance_manager.handle_pdpa_request(
            user_id=user_id,
            request_type="access"
        )
        
        assert access_response["status"] == "completed"
        assert access_response["pdpa_compliant"] is True
        
        # Test correction request (PDPA Section 22)
        correction_response = sg_compliance_manager.handle_pdpa_request(
            user_id=user_id,
            request_type="correction",
            details={"field": "name", "correction": "Updated Name"}
        )
        
        assert correction_response["status"] in ["requires_verification", "completed"]
        assert correction_response["pdpa_compliant"] is True
    
    def test_pdpa_consent_withdrawal(self, sg_compliance_manager, compliance_suite):
        """Test PDPA consent withdrawal (Section 16)."""
        user_id = compliance_suite.generate_test_user_id("pdpa_withdrawal")
        
        # Grant initial consent
        sg_compliance_manager.manage_user_consent(
            user_id=user_id,
            consent_updates={
                ConsentType.ANALYTICS: True,
                ConsentType.MARKETING: True
            },
            method="explicit"
        )
        
        # Test consent withdrawal
        withdrawal_response = sg_compliance_manager.handle_pdpa_request(
            user_id=user_id,
            request_type="withdrawal",
            details={"consent_types": ["analytics", "marketing"]}
        )
        
        assert withdrawal_response["status"] == "completed"
        assert withdrawal_response["pdpa_compliant"] is True
        
        # Verify consent was withdrawn
        consent_record = sg_compliance_manager._user_consents[user_id]
        assert not consent_record.has_consent(ConsentType.ANALYTICS)
        assert not consent_record.has_consent(ConsentType.MARKETING)


class TestDataBreachCompliance:
    """Test data breach notification compliance."""
    
    @pytest.fixture
    def compliance_suite(self):
        return ComplianceTestSuite()
    
    @pytest.fixture
    def enhanced_compliance_manager(self, tmp_path, compliance_suite):
        db_path = str(tmp_path / "breach_test.db")
        return compliance_suite.create_test_compliance_manager(ComplianceRegion.EU, db_path)
    
    def test_breach_severity_classification(self, enhanced_compliance_manager):
        """Test breach severity classification."""
        # Test low severity breach
        low_breach_id = enhanced_compliance_manager.report_data_breach(
            description="Minor configuration exposure",
            affected_users=5,
            data_categories=[DataCategory.TECHNICAL_METRICS],
            severity=BreachSeverity.LOW
        )
        
        low_breach = next(b for b in enhanced_compliance_manager._breach_incidents if b.id == low_breach_id)
        assert low_breach.severity == BreachSeverity.LOW
        assert "regulatory_required" not in low_breach.notification_status
        
        # Test high severity breach
        high_breach_id = enhanced_compliance_manager.report_data_breach(
            description="Database compromise with personal data exposure",
            affected_users=10000,
            data_categories=[DataCategory.PERSONAL_IDENTIFYING, DataCategory.FINANCIAL_DATA],
            severity=BreachSeverity.CRITICAL
        )
        
        high_breach = next(b for b in enhanced_compliance_manager._breach_incidents if b.id == high_breach_id)
        assert high_breach.severity == BreachSeverity.CRITICAL
        assert "regulatory_required" in high_breach.notification_status
    
    def test_breach_notification_timelines(self, enhanced_compliance_manager):
        """Test breach notification timeline compliance."""
        # Report breach affecting sensitive data
        breach_id = enhanced_compliance_manager.report_data_breach(
            description="Healthcare data exposure",
            affected_users=500,
            data_categories=[DataCategory.HEALTH_DATA],
            severity=BreachSeverity.HIGH
        )
        
        breach = next(b for b in enhanced_compliance_manager._breach_incidents if b.id == breach_id)
        
        # GDPR requires notification within 72 hours for high-risk breaches
        notification_deadline = breach.timestamp + (72 * 3600)  # 72 hours in seconds
        current_time = time.time()
        
        # Verify breach is marked for regulatory notification
        assert "regulatory_required" in breach.notification_status
        
        # In a real system, you'd verify actual notification was sent
        # Here we just verify the breach was properly classified
        assert breach.affected_jurisdictions == [ComplianceRegion.EU]
    
    def test_breach_impact_assessment(self, enhanced_compliance_manager):
        """Test breach impact assessment."""
        # Report breach with specific data categories
        breach_id = enhanced_compliance_manager.report_data_breach(
            description="Multi-category data breach",
            affected_users=1000,
            data_categories=[
                DataCategory.PERSONAL_IDENTIFYING,
                DataCategory.FINANCIAL_DATA,
                DataCategory.BEHAVIORAL_DATA
            ],
            severity=BreachSeverity.HIGH,
            estimated_records=2500
        )
        
        breach = next(b for b in enhanced_compliance_manager._breach_incidents if b.id == breach_id)
        
        # Verify breach details
        assert len(breach.data_categories) == 3
        assert DataCategory.PERSONAL_IDENTIFYING in breach.data_categories
        assert DataCategory.FINANCIAL_DATA in breach.data_categories
        assert breach.estimated_records == 2500
        
        # High-impact breach should require user notification
        assert enhanced_compliance_manager._requires_user_notification(breach)


class TestPrivacyImpactAssessment:
    """Test Privacy Impact Assessment (PIA) compliance."""
    
    @pytest.fixture
    def enhanced_compliance_manager(self, tmp_path):
        db_path = str(tmp_path / "pia_test.db")
        return EnhancedComplianceManager(region=ComplianceRegion.EU, db_path=db_path)
    
    def test_pia_creation(self, enhanced_compliance_manager):
        """Test PIA creation and risk assessment."""
        pia_id = enhanced_compliance_manager.conduct_privacy_impact_assessment(
            project_name="AI Model Training Platform",
            data_categories=[
                DataCategory.PERSONAL_IDENTIFYING,
                DataCategory.BEHAVIORAL_DATA,
                DataCategory.MODEL_ARTIFACTS
            ],
            processing_purposes=[
                "machine_learning_training",
                "model_optimization",
                "performance_analysis"
            ],
            legal_bases=["legitimate_interests", "consent"]
        )
        
        assert pia_id is not None
        
        # Find the PIA
        pia = next(p for p in enhanced_compliance_manager._privacy_assessments if p.id == pia_id)
        
        # Verify PIA details
        assert pia.project_name == "AI Model Training Platform"
        assert len(pia.data_categories) == 3
        assert pia.risk_level in ["low", "medium", "high", "very_high"]
        assert len(pia.mitigation_measures) > 0
    
    def test_high_risk_pia_requirements(self, enhanced_compliance_manager):
        """Test high-risk PIA requirements."""
        # Create high-risk PIA with sensitive data
        pia_id = enhanced_compliance_manager.conduct_privacy_impact_assessment(
            project_name="Biometric Authentication System",
            data_categories=[
                DataCategory.BIOMETRIC_DATA,
                DataCategory.SENSITIVE_PERSONAL,
                DataCategory.HEALTH_DATA
            ],
            processing_purposes=[
                "biometric_authentication",
                "large_scale_processing",
                "automated_decision"
            ],
            legal_bases=["consent", "legal_obligation"]
        )
        
        pia = next(p for p in enhanced_compliance_manager._privacy_assessments if p.id == pia_id)
        
        # High-risk PIA should require DPO review
        assert pia.risk_level in ["high", "very_high"]
        assert pia.dpo_review is True
        
        # Should have comprehensive mitigation measures
        assert "Explicit consent required" in pia.mitigation_measures
        assert "Enhanced security measures" in pia.mitigation_measures


class TestComplianceReporting:
    """Test compliance reporting and audit features."""
    
    @pytest.fixture
    def compliance_manager(self, tmp_path):
        db_path = str(tmp_path / "reporting_test.db")
        return EnhancedComplianceManager(region=ComplianceRegion.EU, db_path=db_path)
    
    def test_compliance_report_generation(self, compliance_manager):
        """Test comprehensive compliance report generation."""
        # Create test data over time period
        user_ids = [f"test_user_{i}" for i in range(5)]
        
        start_time = time.time() - 3600  # 1 hour ago
        end_time = time.time()
        
        # Record various processing activities
        for i, user_id in enumerate(user_ids):
            # Grant consent
            compliance_manager.manage_user_consent(
                user_id=user_id,
                consent_updates={
                    ConsentType.NECESSARY: True,
                    ConsentType.ANALYTICS: i % 2 == 0,  # 50% consent rate
                    ConsentType.MARKETING: i % 3 == 0   # 33% consent rate
                },
                method="explicit"
            )
            
            # Record processing
            compliance_manager.record_data_processing(
                user_id=user_id,
                data_category=DataCategory.PERSONAL_IDENTIFYING,
                processing_purpose="account_management",
                legal_basis="contract"
            )
            
            if i % 2 == 0:
                compliance_manager.record_data_processing(
                    user_id=user_id,
                    data_category=DataCategory.USAGE_ANALYTICS,
                    processing_purpose="service_improvement",
                    legal_basis="legitimate_interests"
                )
        
        # Generate compliance report
        report = compliance_manager.generate_compliance_report(start_time, end_time)
        
        # Verify report structure
        assert "compliance_region" in report
        assert "processing_activities" in report
        assert "user_consent" in report
        assert "data_retention" in report
        assert "compliance_status" in report
        
        # Verify processing activities data
        processing_stats = report["processing_activities"]
        assert processing_stats["total"] > 0
        assert processing_stats["categories_processed"] > 0
        assert 0 <= processing_stats["consent_rate"] <= 1
        
        # Verify consent data
        consent_stats = report["user_consent"]
        assert consent_stats["total_users"] == len(user_ids)
        assert 0 <= consent_stats["consent_rates"]["analytics"] <= 1
        assert 0 <= consent_stats["consent_rates"]["marketing"] <= 1
    
    def test_audit_log_integrity(self, compliance_manager):
        """Test audit log integrity and completeness."""
        user_id = "audit_test_user"
        
        # Perform various operations that should be audited
        operations = [
            lambda: compliance_manager.record_data_processing(
                user_id=user_id,
                data_category=DataCategory.PERSONAL_IDENTIFYING,
                processing_purpose="testing",
                legal_basis="consent"
            ),
            lambda: compliance_manager.manage_user_consent(
                user_id=user_id,
                consent_updates={ConsentType.ANALYTICS: True},
                method="explicit"
            ),
            lambda: compliance_manager.handle_data_subject_request(
                user_id=user_id,
                request_type="access"
            )
        ]
        
        initial_audit_count = len(compliance_manager._audit_logs)
        
        # Execute operations
        for operation in operations:
            operation()
        
        # Verify audit logs were created
        final_audit_count = len(compliance_manager._audit_logs)
        assert final_audit_count > initial_audit_count
        
        # Verify audit log entries have required fields
        recent_logs = compliance_manager._audit_logs[initial_audit_count:]
        for log in recent_logs:
            assert log.id is not None
            assert log.timestamp > 0
            assert log.action is not None
            assert log.resource is not None
            assert log.result in ["success", "failure", "partial"]
            assert log.risk_level in ["low", "medium", "high", "critical"]
    
    def test_data_retention_compliance_monitoring(self, compliance_manager):
        """Test data retention compliance monitoring."""
        user_id = "retention_test_user"
        
        # Create processing record with short retention period
        old_timestamp = time.time() - (400 * 24 * 3600)  # 400 days ago
        
        # Manually create old record (simulating aged data)
        from backend.codesign_playground.utils.compliance import DataProcessingRecord
        import uuid
        
        old_record = DataProcessingRecord(
            id=str(uuid.uuid4()),
            timestamp=old_timestamp,
            user_id=user_id,
            data_category=DataCategory.TECHNICAL_METRICS,
            processing_purpose="historical_testing",
            legal_basis="legitimate_interests",
            retention_period=365,  # 1 year retention
            data_location="local",
            audit_trail=[f"Created at {old_timestamp}"]
        )
        
        compliance_manager._processing_records.append(old_record)
        
        # Check retention compliance
        expired_records = compliance_manager._check_data_retention_compliance()
        
        # Should identify the expired record
        expired_user_records = [r for r in expired_records if r.user_id == user_id]
        assert len(expired_user_records) == 1
        assert expired_user_records[0].id == old_record.id
    
    def test_cross_border_transfer_tracking(self, compliance_manager):
        """Test cross-border data transfer tracking."""
        user_id = "transfer_test_user"
        
        # Record cross-border transfer
        success = compliance_manager.record_data_processing(
            user_id=user_id,
            data_category=DataCategory.PERSONAL_IDENTIFYING,
            processing_purpose="global_service_provision",
            legal_basis="contract",
            cross_border_transfer=True,
            data_location="us_east_1"
        )
        
        assert success
        
        # Verify transfer was recorded
        transfer_records = [
            r for r in compliance_manager._processing_records
            if r.user_id == user_id and r.cross_border_transfer
        ]
        
        assert len(transfer_records) == 1
        assert transfer_records[0].data_location == "us_east_1"
        assert transfer_records[0].cross_border_transfer is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
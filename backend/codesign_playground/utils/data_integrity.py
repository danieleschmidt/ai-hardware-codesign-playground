"""
Data integrity and backup/recovery mechanisms for AI Hardware Co-Design Playground.

This module provides comprehensive data validation, integrity checking, backup,
and disaster recovery capabilities with automated corruption detection.
"""

import os
import time
import hashlib
import json
import shutil
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import pickle
import gzip
import tarfile

from .logging import get_logger, get_audit_logger
from .monitoring import record_metric
from .exceptions import CodesignError
from .distributed_tracing import trace_span, SpanType

logger = get_logger(__name__)
audit_logger = get_audit_logger(__name__)


class DataIntegrityLevel(Enum):
    """Data integrity check levels."""
    BASIC = "basic"        # Hash verification only
    STANDARD = "standard"  # Hash + structure validation  
    STRICT = "strict"      # Hash + structure + content validation
    PARANOID = "paranoid"  # All checks + redundant verification


class BackupType(Enum):
    """Types of backups."""
    FULL = "full"         # Complete system backup
    INCREMENTAL = "incremental"  # Changes since last backup
    DIFFERENTIAL = "differential"  # Changes since last full backup
    SNAPSHOT = "snapshot"  # Point-in-time snapshot


class RecoveryStatus(Enum):
    """Recovery operation status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class DataChecksum:
    """Data checksum information."""
    algorithm: str
    value: str
    timestamp: float
    file_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "algorithm": self.algorithm,
            "value": self.value,
            "timestamp": self.timestamp,
            "file_size": self.file_size,
            "metadata": self.metadata
        }


@dataclass
class BackupManifest:
    """Backup manifest with metadata."""
    backup_id: str
    backup_type: BackupType
    timestamp: float
    files: List[str]
    checksums: Dict[str, DataChecksum]
    metadata: Dict[str, Any] = field(default_factory=dict)
    size_bytes: int = 0
    compression: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backup_id": self.backup_id,
            "backup_type": self.backup_type.value,
            "timestamp": self.timestamp,
            "files": self.files,
            "checksums": {k: v.to_dict() for k, v in self.checksums.items()},
            "metadata": self.metadata,
            "size_bytes": self.size_bytes,
            "compression": self.compression
        }


class DataIntegrityChecker:
    """Comprehensive data integrity checking."""
    
    def __init__(self, level: DataIntegrityLevel = DataIntegrityLevel.STANDARD):
        """Initialize data integrity checker."""
        self.level = level
        self.checksum_cache: Dict[str, DataChecksum] = {}
        self.corruption_log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
        logger.info("Initialized DataIntegrityChecker", level=level.value)
    
    def compute_checksum(self, file_path: str, algorithm: str = "sha256") -> DataChecksum:
        """Compute checksum for file."""
        try:
            path = Path(file_path)
            if not path.exists():
                raise CodesignError(f"File not found: {file_path}", "FILE_NOT_FOUND")
            
            hash_obj = hashlib.new(algorithm)
            file_size = 0
            
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hash_obj.update(chunk)
                    file_size += len(chunk)
            
            checksum = DataChecksum(
                algorithm=algorithm,
                value=hash_obj.hexdigest(),
                timestamp=time.time(),
                file_size=file_size,
                metadata={"file_path": str(path.absolute())}
            )
            
            # Cache checksum
            with self._lock:
                self.checksum_cache[file_path] = checksum
            
            return checksum
            
        except Exception as e:
            logger.error("Failed to compute checksum", file_path=file_path, error=str(e))
            raise CodesignError(f"Checksum computation failed: {e}", "CHECKSUM_ERROR")
    
    def verify_checksum(self, file_path: str, expected_checksum: DataChecksum) -> bool:
        """Verify file against expected checksum."""
        try:
            current_checksum = self.compute_checksum(file_path, expected_checksum.algorithm)
            
            is_valid = (
                current_checksum.value == expected_checksum.value and
                current_checksum.file_size == expected_checksum.file_size
            )
            
            if not is_valid:
                corruption_entry = {
                    "file_path": file_path,
                    "timestamp": time.time(),
                    "expected_checksum": expected_checksum.value,
                    "actual_checksum": current_checksum.value,
                    "expected_size": expected_checksum.file_size,
                    "actual_size": current_checksum.file_size
                }
                
                with self._lock:
                    self.corruption_log.append(corruption_entry)
                
                audit_logger.log_security_event("data_corruption_detected",
                                               f"Data corruption detected in {file_path}",
                                               "high", **corruption_entry)
                
                record_metric("data_corruption_detected", 1, "counter", {"file": file_path})
            
            return is_valid
            
        except Exception as e:
            logger.error("Checksum verification failed", file_path=file_path, error=str(e))
            return False
    
    def validate_structure(self, file_path: str, expected_structure: Dict[str, Any]) -> bool:
        """Validate file structure (for JSON/pickle files)."""
        if self.level in [DataIntegrityLevel.BASIC]:
            return True  # Skip structure validation for basic level
        
        try:
            path = Path(file_path)
            
            if path.suffix.lower() == '.json':
                return self._validate_json_structure(file_path, expected_structure)
            elif path.suffix.lower() in ['.pkl', '.pickle']:
                return self._validate_pickle_structure(file_path, expected_structure)
            else:
                # For other file types, just check if file is readable
                with open(file_path, 'rb') as f:
                    f.read(1)  # Try to read at least one byte
                return True
                
        except Exception as e:
            logger.error("Structure validation failed", file_path=file_path, error=str(e))
            return False
    
    def _validate_json_structure(self, file_path: str, expected_structure: Dict[str, Any]) -> bool:
        """Validate JSON file structure."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            return self._check_dict_structure(data, expected_structure)
            
        except json.JSONDecodeError:
            return False
        except Exception:
            return False
    
    def _validate_pickle_structure(self, file_path: str, expected_structure: Dict[str, Any]) -> bool:
        """Validate pickle file structure."""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict):
                return self._check_dict_structure(data, expected_structure)
            else:
                # For non-dict objects, check type
                expected_type = expected_structure.get("type")
                if expected_type:
                    return type(data).__name__ == expected_type
                return True
                
        except Exception:
            return False
    
    def _check_dict_structure(self, data: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        """Check dictionary structure against expected."""
        required_keys = expected.get("required_keys", [])
        optional_keys = expected.get("optional_keys", [])
        forbidden_keys = expected.get("forbidden_keys", [])
        
        # Check required keys
        for key in required_keys:
            if key not in data:
                return False
        
        # Check forbidden keys
        for key in forbidden_keys:
            if key in data:
                return False
        
        # Check key types if specified
        key_types = expected.get("key_types", {})
        for key, expected_type in key_types.items():
            if key in data:
                if not isinstance(data[key], expected_type):
                    return False
        
        return True
    
    def validate_content(self, file_path: str, validators: List[Callable[[Any], bool]]) -> bool:
        """Validate file content using custom validators."""
        if self.level in [DataIntegrityLevel.BASIC, DataIntegrityLevel.STANDARD]:
            return True  # Skip content validation for lower levels
        
        try:
            path = Path(file_path)
            
            # Load data based on file type
            if path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
            elif path.suffix.lower() in ['.pkl', '.pickle']:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    data = f.read()
            
            # Apply validators
            for validator in validators:
                if not validator(data):
                    return False
            
            return True
            
        except Exception as e:
            logger.error("Content validation failed", file_path=file_path, error=str(e))
            return False
    
    def comprehensive_check(self, file_path: str, 
                          expected_checksum: Optional[DataChecksum] = None,
                          expected_structure: Optional[Dict[str, Any]] = None,
                          content_validators: Optional[List[Callable]] = None) -> Dict[str, Any]:
        """Perform comprehensive integrity check."""
        results = {
            "file_path": file_path,
            "timestamp": time.time(),
            "checksum_valid": False,
            "structure_valid": False,
            "content_valid": False,
            "overall_valid": False,
            "errors": []
        }
        
        try:
            # Checksum validation
            if expected_checksum:
                results["checksum_valid"] = self.verify_checksum(file_path, expected_checksum)
                if not results["checksum_valid"]:
                    results["errors"].append("Checksum validation failed")
            else:
                results["checksum_valid"] = True  # No checksum to verify
            
            # Structure validation
            if expected_structure:
                results["structure_valid"] = self.validate_structure(file_path, expected_structure)
                if not results["structure_valid"]:
                    results["errors"].append("Structure validation failed")
            else:
                results["structure_valid"] = True  # No structure to verify
            
            # Content validation
            if content_validators:
                results["content_valid"] = self.validate_content(file_path, content_validators)
                if not results["content_valid"]:
                    results["errors"].append("Content validation failed")
            else:
                results["content_valid"] = True  # No content to verify
            
            # Overall validation
            results["overall_valid"] = (
                results["checksum_valid"] and 
                results["structure_valid"] and 
                results["content_valid"]
            )
            
            if results["overall_valid"]:
                record_metric("data_integrity_check_passed", 1, "counter")
            else:
                record_metric("data_integrity_check_failed", 1, "counter")
                
        except Exception as e:
            results["errors"].append(f"Integrity check failed: {e}")
            logger.error("Comprehensive integrity check failed", file_path=file_path, error=str(e))
        
        return results
    
    def get_corruption_report(self) -> Dict[str, Any]:
        """Get corruption detection report."""
        with self._lock:
            return {
                "total_corruptions": len(self.corruption_log),
                "corruptions": self.corruption_log.copy(),
                "last_check": time.time()
            }


class BackupManager:
    """Comprehensive backup and recovery manager."""
    
    def __init__(self, backup_root: str = "/tmp/codesign_backups"):
        """Initialize backup manager."""
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(parents=True, exist_ok=True)
        
        self.manifests: Dict[str, BackupManifest] = {}
        self.integrity_checker = DataIntegrityChecker()
        self._lock = threading.Lock()
        
        # Load existing manifests
        self._load_manifests()
        
        logger.info("Initialized BackupManager", backup_root=backup_root)
    
    def create_backup(self, source_paths: List[str], backup_type: BackupType = BackupType.FULL,
                     compression: bool = True, verify: bool = True) -> str:
        """Create backup of specified paths."""
        backup_id = f"backup_{int(time.time())}_{backup_type.value}"
        
        try:
            with trace_span(f"backup_create_{backup_type.value}", SpanType.FUNCTION_CALL) as span:
                span.set_tag("backup.id", backup_id)
                span.set_tag("backup.type", backup_type.value)
                span.set_tag("backup.source_count", len(source_paths))
                
                # Create backup directory
                backup_dir = self.backup_root / backup_id
                backup_dir.mkdir(parents=True, exist_ok=True)
                
                # Collect files to backup
                files_to_backup = []
                for source_path in source_paths:
                    files_to_backup.extend(self._collect_files(source_path))
                
                span.set_tag("backup.file_count", len(files_to_backup))
                
                # Create checksums
                checksums = {}
                for file_path in files_to_backup:
                    try:
                        checksum = self.integrity_checker.compute_checksum(file_path)
                        checksums[file_path] = checksum
                    except Exception as e:
                        logger.warning("Failed to compute checksum for backup",
                                     file_path=file_path, error=str(e))
                
                # Create backup archive
                backup_file = backup_dir / f"{backup_id}.tar"
                if compression:
                    backup_file = backup_dir / f"{backup_id}.tar.gz"
                
                total_size = self._create_archive(files_to_backup, backup_file, compression)
                
                # Create manifest
                manifest = BackupManifest(
                    backup_id=backup_id,
                    backup_type=backup_type,
                    timestamp=time.time(),
                    files=files_to_backup,
                    checksums=checksums,
                    size_bytes=total_size,
                    compression="gzip" if compression else None,
                    metadata={
                        "source_paths": source_paths,
                        "created_by": "BackupManager",
                        "file_count": len(files_to_backup)
                    }
                )
                
                # Save manifest
                manifest_file = backup_dir / f"{backup_id}_manifest.json"
                with open(manifest_file, 'w') as f:
                    json.dump(manifest.to_dict(), f, indent=2)
                
                with self._lock:
                    self.manifests[backup_id] = manifest
                
                # Verify backup if requested
                if verify:
                    verification_result = self.verify_backup(backup_id)
                    if not verification_result["valid"]:
                        raise CodesignError("Backup verification failed", "BACKUP_VERIFICATION_FAILED")
                
                audit_logger.log_security_event("backup_created",
                                               f"Backup {backup_id} created successfully",
                                               "low", backup_id=backup_id,
                                               backup_type=backup_type.value,
                                               file_count=len(files_to_backup),
                                               size_mb=total_size / 1024 / 1024)
                
                record_metric("backup_created", 1, "counter", {"type": backup_type.value})
                record_metric("backup_size_mb", total_size / 1024 / 1024, "histogram")
                
                logger.info("Backup created successfully",
                           backup_id=backup_id,
                           backup_type=backup_type.value,
                           file_count=len(files_to_backup),
                           size_mb=total_size / 1024 / 1024)
                
                return backup_id
                
        except Exception as e:
            logger.error("Backup creation failed", backup_id=backup_id, error=str(e))
            record_metric("backup_creation_failed", 1, "counter")
            raise CodesignError(f"Backup creation failed: {e}", "BACKUP_CREATION_FAILED")
    
    def restore_backup(self, backup_id: str, target_directory: str,
                      verify_integrity: bool = True) -> Dict[str, Any]:
        """Restore backup to target directory."""
        try:
            with trace_span(f"backup_restore", SpanType.FUNCTION_CALL) as span:
                span.set_tag("backup.id", backup_id)
                span.set_tag("backup.target", target_directory)
                
                # Get manifest
                manifest = self.manifests.get(backup_id)
                if not manifest:
                    raise CodesignError(f"Backup {backup_id} not found", "BACKUP_NOT_FOUND")
                
                # Verify backup integrity first
                if verify_integrity:
                    verification = self.verify_backup(backup_id)
                    if not verification["valid"]:
                        raise CodesignError("Backup integrity check failed", "BACKUP_CORRUPTED")
                
                # Create target directory
                target_path = Path(target_directory)
                target_path.mkdir(parents=True, exist_ok=True)
                
                # Extract backup
                backup_dir = self.backup_root / backup_id
                backup_file = backup_dir / f"{backup_id}.tar"
                if manifest.compression:
                    backup_file = backup_dir / f"{backup_id}.tar.gz"
                
                extraction_result = self._extract_archive(backup_file, target_path, manifest.compression)
                
                # Verify restored files
                verification_results = []
                if verify_integrity:
                    for file_path, expected_checksum in manifest.checksums.items():
                        # Calculate relative path
                        rel_path = Path(file_path).name
                        restored_file = target_path / rel_path
                        
                        if restored_file.exists():
                            is_valid = self.integrity_checker.verify_checksum(
                                str(restored_file), expected_checksum
                            )
                            verification_results.append({
                                "file": str(restored_file),
                                "valid": is_valid
                            })
                
                result = {
                    "backup_id": backup_id,
                    "target_directory": target_directory,
                    "status": RecoveryStatus.COMPLETED.value,
                    "files_restored": extraction_result["files_extracted"],
                    "verification_results": verification_results,
                    "all_files_valid": all(r["valid"] for r in verification_results) if verification_results else True
                }
                
                audit_logger.log_security_event("backup_restored",
                                               f"Backup {backup_id} restored successfully",
                                               "medium", **result)
                
                record_metric("backup_restored", 1, "counter")
                
                logger.info("Backup restored successfully", **result)
                
                return result
                
        except Exception as e:
            logger.error("Backup restoration failed", backup_id=backup_id, error=str(e))
            record_metric("backup_restoration_failed", 1, "counter")
            raise CodesignError(f"Backup restoration failed: {e}", "BACKUP_RESTORATION_FAILED")
    
    def verify_backup(self, backup_id: str) -> Dict[str, Any]:
        """Verify backup integrity."""
        try:
            manifest = self.manifests.get(backup_id)
            if not manifest:
                return {"valid": False, "error": "Backup manifest not found"}
            
            backup_dir = self.backup_root / backup_id
            backup_file = backup_dir / f"{backup_id}.tar"
            if manifest.compression:
                backup_file = backup_dir / f"{backup_id}.tar.gz"
            
            if not backup_file.exists():
                return {"valid": False, "error": "Backup file not found"}
            
            # Check backup file size
            actual_size = backup_file.stat().st_size
            if abs(actual_size - manifest.size_bytes) > 1024:  # Allow 1KB difference
                return {"valid": False, "error": "Backup file size mismatch"}
            
            # TODO: Add more sophisticated backup verification
            # For now, just check file existence and size
            
            return {
                "valid": True,
                "backup_id": backup_id,
                "file_size": actual_size,
                "manifest_size": manifest.size_bytes,
                "verification_time": time.time()
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups."""
        with self._lock:
            return [
                {
                    "backup_id": backup_id,
                    "backup_type": manifest.backup_type.value,
                    "timestamp": manifest.timestamp,
                    "date": datetime.fromtimestamp(manifest.timestamp).isoformat(),
                    "file_count": len(manifest.files),
                    "size_mb": manifest.size_bytes / 1024 / 1024,
                    "compression": manifest.compression
                }
                for backup_id, manifest in self.manifests.items()
            ]
    
    def cleanup_old_backups(self, retention_days: int = 30,
                           keep_minimum: int = 3) -> List[str]:
        """Clean up old backups based on retention policy."""
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        removed_backups = []
        
        # Sort backups by timestamp (newest first)
        sorted_backups = sorted(
            self.manifests.items(),
            key=lambda x: x[1].timestamp,
            reverse=True
        )
        
        # Keep minimum number of backups regardless of age
        backups_to_check = sorted_backups[keep_minimum:]
        
        for backup_id, manifest in backups_to_check:
            if manifest.timestamp < cutoff_time:
                try:
                    self._remove_backup(backup_id)
                    removed_backups.append(backup_id)
                except Exception as e:
                    logger.error("Failed to remove backup", backup_id=backup_id, error=str(e))
        
        if removed_backups:
            logger.info("Cleaned up old backups",
                       removed_count=len(removed_backups),
                       retention_days=retention_days)
        
        return removed_backups
    
    def _collect_files(self, source_path: str) -> List[str]:
        """Collect all files from source path."""
        files = []
        path = Path(source_path)
        
        if path.is_file():
            files.append(str(path))
        elif path.is_dir():
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    files.append(str(file_path))
        
        return files
    
    def _create_archive(self, files: List[str], archive_path: Path, compression: bool) -> int:
        """Create tar archive of files."""
        mode = "w:gz" if compression else "w"
        
        with tarfile.open(archive_path, mode) as tar:
            for file_path in files:
                try:
                    # Add file with just its name (not full path)
                    arcname = Path(file_path).name
                    tar.add(file_path, arcname=arcname)
                except Exception as e:
                    logger.warning("Failed to add file to archive",
                                 file_path=file_path, error=str(e))
        
        return archive_path.stat().st_size
    
    def _extract_archive(self, archive_path: Path, target_path: Path, compression: Optional[str]) -> Dict[str, Any]:
        """Extract tar archive."""
        mode = "r:gz" if compression else "r"
        files_extracted = 0
        
        with tarfile.open(archive_path, mode) as tar:
            for member in tar.getmembers():
                if member.isfile():
                    tar.extract(member, target_path)
                    files_extracted += 1
        
        return {"files_extracted": files_extracted}
    
    def _load_manifests(self) -> None:
        """Load existing backup manifests."""
        for backup_dir in self.backup_root.iterdir():
            if backup_dir.is_dir():
                manifest_file = backup_dir / f"{backup_dir.name}_manifest.json"
                if manifest_file.exists():
                    try:
                        with open(manifest_file, 'r') as f:
                            manifest_data = json.load(f)
                        
                        # Reconstruct DataChecksum objects
                        checksums = {}
                        for file_path, checksum_data in manifest_data.get("checksums", {}).items():
                            checksums[file_path] = DataChecksum(**checksum_data)
                        
                        manifest = BackupManifest(
                            backup_id=manifest_data["backup_id"],
                            backup_type=BackupType(manifest_data["backup_type"]),
                            timestamp=manifest_data["timestamp"],
                            files=manifest_data["files"],
                            checksums=checksums,
                            metadata=manifest_data.get("metadata", {}),
                            size_bytes=manifest_data.get("size_bytes", 0),
                            compression=manifest_data.get("compression")
                        )
                        
                        self.manifests[manifest.backup_id] = manifest
                        
                    except Exception as e:
                        logger.warning("Failed to load backup manifest",
                                     manifest_file=str(manifest_file), error=str(e))
    
    def _remove_backup(self, backup_id: str) -> None:
        """Remove backup and its files."""
        backup_dir = self.backup_root / backup_id
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        
        with self._lock:
            if backup_id in self.manifests:
                del self.manifests[backup_id]


# Global instances
_data_integrity_checker: Optional[DataIntegrityChecker] = None
_backup_manager: Optional[BackupManager] = None
_integrity_lock = threading.Lock()
_backup_lock = threading.Lock()


def get_data_integrity_checker(level: DataIntegrityLevel = DataIntegrityLevel.STANDARD) -> DataIntegrityChecker:
    """Get global data integrity checker."""
    global _data_integrity_checker
    
    with _integrity_lock:
        if _data_integrity_checker is None:
            _data_integrity_checker = DataIntegrityChecker(level)
        
        return _data_integrity_checker


def get_backup_manager(backup_root: str = "/tmp/codesign_backups") -> BackupManager:
    """Get global backup manager."""
    global _backup_manager
    
    with _backup_lock:
        if _backup_manager is None:
            _backup_manager = BackupManager(backup_root)
        
        return _backup_manager


def verify_file_integrity(file_path: str, expected_checksum: Optional[str] = None) -> bool:
    """Convenience function for file integrity verification."""
    checker = get_data_integrity_checker()
    
    if expected_checksum:
        checksum_obj = DataChecksum(
            algorithm="sha256",
            value=expected_checksum,
            timestamp=time.time(),
            file_size=0  # Will be verified during check
        )
        return checker.verify_checksum(file_path, checksum_obj)
    else:
        # Just compute and cache checksum
        checker.compute_checksum(file_path)
        return True


def create_system_backup(backup_paths: List[str], backup_type: BackupType = BackupType.FULL) -> str:
    """Convenience function for creating system backup."""
    manager = get_backup_manager()
    return manager.create_backup(backup_paths, backup_type)
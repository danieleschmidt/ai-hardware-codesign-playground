"""
Comprehensive unit tests for workflow functionality.

This module tests the Workflow class and related workflow management components
for the AI Hardware Co-Design Playground.
"""

import pytest
import tempfile
import shutil
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional

from codesign_playground.core.workflow import (
    Workflow, WorkflowState, WorkflowStage, WorkflowConfig,
    WorkflowManager, WorkflowMetrics
)
from codesign_playground.core.accelerator import Accelerator
from codesign_playground.utils.exceptions import WorkflowError, ValidationError


class TestWorkflowState:
    """Test WorkflowState class functionality."""
    
    def test_workflow_state_creation(self):
        """Test WorkflowState creation and initialization."""
        state = WorkflowState(
            stage=WorkflowStage.INITIALIZED,
            progress=0.0,
            metadata={"created_at": "2024-01-01T00:00:00Z"}
        )
        
        assert state.stage == WorkflowStage.INITIALIZED
        assert state.progress == 0.0
        assert state.metadata["created_at"] == "2024-01-01T00:00:00Z"
        assert state.errors == []
        assert state.warnings == []
    
    def test_workflow_state_transitions(self):
        """Test workflow state transitions."""
        state = WorkflowState(WorkflowStage.INITIALIZED)
        
        # Valid transitions
        state.transition_to(WorkflowStage.MODEL_IMPORTED)
        assert state.stage == WorkflowStage.MODEL_IMPORTED
        
        state.transition_to(WorkflowStage.HARDWARE_MAPPED)
        assert state.stage == WorkflowStage.HARDWARE_MAPPED
        
        state.transition_to(WorkflowStage.OPTIMIZED)
        assert state.stage == WorkflowStage.OPTIMIZED
        
        state.transition_to(WorkflowStage.COMPLETED)
        assert state.stage == WorkflowStage.COMPLETED
    
    def test_workflow_state_invalid_transitions(self):
        """Test invalid workflow state transitions."""
        state = WorkflowState(WorkflowStage.INITIALIZED)
        
        # Invalid transition (skipping stages)
        with pytest.raises(WorkflowError):
            state.transition_to(WorkflowStage.COMPLETED)
        
        # Transition from completed state
        state.stage = WorkflowStage.COMPLETED
        with pytest.raises(WorkflowError):
            state.transition_to(WorkflowStage.MODEL_IMPORTED)
    
    def test_workflow_state_error_handling(self):
        """Test workflow state error handling."""
        state = WorkflowState(WorkflowStage.MODEL_IMPORTED)
        
        # Add error
        state.add_error("Model validation failed", {"code": "E001"})
        
        assert len(state.errors) == 1
        assert state.errors[0]["message"] == "Model validation failed"
        assert state.errors[0]["metadata"]["code"] == "E001"
        assert state.has_errors()
        
        # Add warning
        state.add_warning("Performance may be suboptimal", {"optimization": "disabled"})
        
        assert len(state.warnings) == 1
        assert state.warnings[0]["message"] == "Performance may be suboptimal"
        assert state.has_warnings()
    
    def test_workflow_state_serialization(self):
        """Test workflow state serialization."""
        state = WorkflowState(
            stage=WorkflowStage.OPTIMIZED,
            progress=75.0,
            metadata={"model_name": "test_model"}
        )
        state.add_error("Test error")
        state.add_warning("Test warning")
        
        state_dict = state.to_dict()
        
        assert isinstance(state_dict, dict)
        assert state_dict["stage"] == "OPTIMIZED"
        assert state_dict["progress"] == 75.0
        assert state_dict["metadata"]["model_name"] == "test_model"
        assert len(state_dict["errors"]) == 1
        assert len(state_dict["warnings"]) == 1
        
        # Test deserialization
        restored_state = WorkflowState.from_dict(state_dict)
        assert restored_state.stage == WorkflowStage.OPTIMIZED
        assert restored_state.progress == 75.0
        assert len(restored_state.errors) == 1
        assert len(restored_state.warnings) == 1


class TestWorkflowConfig:
    """Test WorkflowConfig class functionality."""
    
    def test_workflow_config_creation(self):
        """Test WorkflowConfig creation with default values."""
        config = WorkflowConfig(
            name="test_workflow",
            model_path="/path/to/model.onnx",
            input_shapes={"input": (1, 3, 224, 224)}
        )
        
        assert config.name == "test_workflow"
        assert config.model_path == "/path/to/model.onnx"
        assert config.input_shapes == {"input": (1, 3, 224, 224)}
        assert config.framework == "auto"  # default
        assert config.optimization_level == "moderate"  # default
        assert config.parallel_execution is True  # default
    
    def test_workflow_config_validation(self):
        """Test workflow configuration validation."""
        # Valid configuration
        valid_config = WorkflowConfig(
            name="valid_workflow",
            model_path="/valid/path.onnx",
            input_shapes={"input": (1, 3, 224, 224)},
            framework="onnx",
            optimization_level="aggressive"
        )
        
        assert valid_config.validate()
        
        # Invalid configuration - empty name
        with pytest.raises(ValidationError):
            WorkflowConfig(
                name="",
                model_path="/path/to/model.onnx",
                input_shapes={"input": (1, 3, 224, 224)}
            )
        
        # Invalid configuration - invalid optimization level
        with pytest.raises(ValidationError):
            WorkflowConfig(
                name="test",
                model_path="/path/to/model.onnx",
                input_shapes={"input": (1, 3, 224, 224)},
                optimization_level="invalid_level"
            )
    
    def test_workflow_config_serialization(self):
        """Test workflow configuration serialization."""
        config = WorkflowConfig(
            name="serialization_test",
            model_path="/test/model.pt",
            input_shapes={"input": (1, 3, 224, 224)},
            framework="pytorch",
            hardware_template="systolic_array",
            optimization_constraints={"power_budget": 5.0}
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["name"] == "serialization_test"
        assert config_dict["framework"] == "pytorch"
        assert config_dict["optimization_constraints"]["power_budget"] == 5.0
        
        # Test deserialization
        restored_config = WorkflowConfig.from_dict(config_dict)
        assert restored_config.name == config.name
        assert restored_config.framework == config.framework
        assert restored_config.optimization_constraints == config.optimization_constraints


class TestWorkflow:
    """Test Workflow class functionality."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_config(self, temp_workspace):
        """Sample workflow configuration."""
        return WorkflowConfig(
            name="test_workflow",
            model_path=str(temp_workspace / "test_model.onnx"),
            input_shapes={"input": (1, 3, 224, 224)},
            framework="onnx",
            output_dir=str(temp_workspace / "output")
        )
    
    @pytest.fixture
    def workflow(self, sample_config):
        """Workflow instance for testing."""
        return Workflow(sample_config)
    
    def test_workflow_creation(self, sample_config):
        """Test Workflow creation and initialization."""
        workflow = Workflow(sample_config)
        
        assert workflow.config == sample_config
        assert workflow.state.stage == WorkflowStage.INITIALIZED
        assert workflow.state.progress == 0.0
        assert workflow.metrics.start_time > 0
        assert workflow.workspace_dir.exists()
    
    def test_workflow_model_import(self, workflow, temp_workspace):
        """Test model import functionality."""
        # Create mock model file
        model_file = temp_workspace / "test_model.onnx"
        model_file.write_bytes(b"mock_onnx_model_data")
        
        # Update config with actual file path
        workflow.config.model_path = str(model_file)
        
        # Import model
        workflow.import_model()
        
        assert workflow.state.stage == WorkflowStage.MODEL_IMPORTED
        assert workflow.state.progress > 0
        assert workflow.model is not None
        assert hasattr(workflow.model, 'framework')
        assert hasattr(workflow.model, 'input_shapes')
    
    def test_workflow_model_import_invalid_file(self, workflow):
        """Test model import with invalid file."""
        workflow.config.model_path = "/nonexistent/model.onnx"
        
        with pytest.raises(WorkflowError):
            workflow.import_model()
        
        assert workflow.state.stage == WorkflowStage.INITIALIZED
        assert workflow.state.has_errors()
    
    def test_workflow_hardware_mapping(self, workflow, temp_workspace):
        """Test hardware mapping functionality."""
        # First import a model
        model_file = temp_workspace / "test_model.onnx"
        model_file.write_bytes(b"mock_model_data")
        workflow.config.model_path = str(model_file)
        workflow.import_model()
        
        # Map to hardware
        workflow.map_to_hardware(
            template="systolic_array",
            size=(16, 16),
            precision="int8",
            frequency_mhz=400.0
        )
        
        assert workflow.state.stage == WorkflowStage.HARDWARE_MAPPED
        assert workflow.accelerator is not None
        assert isinstance(workflow.accelerator, Accelerator)
        assert workflow.accelerator.compute_units > 0
    
    def test_workflow_hardware_mapping_invalid_template(self, workflow, temp_workspace):
        """Test hardware mapping with invalid template."""
        # Import model first
        model_file = temp_workspace / "test_model.onnx"
        model_file.write_bytes(b"mock_model_data")
        workflow.config.model_path = str(model_file)
        workflow.import_model()
        
        # Try invalid template
        with pytest.raises(WorkflowError):
            workflow.map_to_hardware(
                template="invalid_template",
                size=(16, 16)
            )
        
        assert workflow.state.stage == WorkflowStage.MODEL_IMPORTED
        assert workflow.state.has_errors()
    
    def test_workflow_optimization(self, workflow, temp_workspace):
        """Test workflow optimization functionality."""
        # Setup workflow through previous stages
        model_file = temp_workspace / "test_model.onnx"
        model_file.write_bytes(b"mock_model_data")
        workflow.config.model_path = str(model_file)
        
        workflow.import_model()
        workflow.map_to_hardware(template="systolic_array", size=(16, 16))
        
        # Run optimization
        optimization_result = workflow.optimize(
            target_fps=30.0,
            power_budget=5.0,
            optimization_level="moderate"
        )
        
        assert workflow.state.stage == WorkflowStage.OPTIMIZED
        assert optimization_result is not None
        assert hasattr(optimization_result, 'final_metrics')
        assert optimization_result.final_metrics["power"] <= 5.0
    
    def test_workflow_rtl_generation(self, workflow, temp_workspace):
        """Test RTL generation functionality."""
        # Setup workflow through all previous stages
        model_file = temp_workspace / "test_model.onnx"
        model_file.write_bytes(b"mock_model_data")
        workflow.config.model_path = str(model_file)
        
        workflow.import_model()
        workflow.map_to_hardware(template="systolic_array", size=(8, 8))
        workflow.optimize(target_fps=30.0, power_budget=5.0)
        
        # Generate RTL
        rtl_file = workflow.generate_rtl(
            output_format="verilog",
            optimization_passes=["constant_folding", "dead_code_elimination"]
        )
        
        assert workflow.state.stage == WorkflowStage.RTL_GENERATED
        assert rtl_file.exists()
        assert rtl_file.suffix == ".v"
        
        # Check RTL content
        rtl_content = rtl_file.read_text()
        assert "module" in rtl_content
        assert "accelerator" in rtl_content.lower()
    
    def test_workflow_complete_pipeline(self, workflow, temp_workspace):
        """Test complete workflow pipeline execution."""
        # Create model file
        model_file = temp_workspace / "complete_test_model.onnx"
        model_file.write_bytes(b"complete_mock_model_data")
        workflow.config.model_path = str(model_file)
        
        # Execute complete pipeline
        final_result = workflow.execute_pipeline(
            hardware_config={
                "template": "systolic_array",
                "size": (16, 16),
                "precision": "int8"
            },
            optimization_config={
                "target_fps": 30.0,
                "power_budget": 5.0,
                "optimization_level": "moderate"
            },
            rtl_config={
                "output_format": "verilog",
                "optimization_passes": ["constant_folding"]
            }
        )
        
        assert workflow.state.stage == WorkflowStage.COMPLETED
        assert workflow.state.progress == 100.0
        assert final_result is not None
        assert "rtl_file" in final_result
        assert "optimization_result" in final_result
        assert "performance_metrics" in final_result
    
    def test_workflow_state_persistence(self, workflow, temp_workspace):
        """Test workflow state persistence and restoration."""
        # Execute partial workflow
        model_file = temp_workspace / "persistence_test_model.onnx"
        model_file.write_bytes(b"persistence_mock_model_data")
        workflow.config.model_path = str(model_file)
        
        workflow.import_model()
        workflow.map_to_hardware(template="systolic_array", size=(8, 8))
        
        # Save state
        state_file = workflow.save_state()
        assert state_file.exists()
        
        # Create new workflow and restore state
        new_workflow = Workflow(workflow.config)
        new_workflow.restore_state(state_file)
        
        assert new_workflow.state.stage == WorkflowStage.HARDWARE_MAPPED
        assert new_workflow.accelerator is not None
        assert new_workflow.model is not None
    
    def test_workflow_error_recovery(self, workflow, temp_workspace):
        """Test workflow error recovery mechanisms."""
        # Create model file
        model_file = temp_workspace / "error_test_model.onnx"
        model_file.write_bytes(b"error_mock_model_data")
        workflow.config.model_path = str(model_file)
        
        workflow.import_model()
        
        # Simulate error during hardware mapping
        with patch.object(workflow, '_validate_hardware_config', side_effect=WorkflowError("Hardware error")):
            with pytest.raises(WorkflowError):
                workflow.map_to_hardware(template="systolic_array", size=(16, 16))
        
        # Workflow should remain in previous valid state
        assert workflow.state.stage == WorkflowStage.MODEL_IMPORTED
        assert workflow.state.has_errors()
        
        # Should be able to retry
        workflow.clear_errors()
        workflow.map_to_hardware(template="systolic_array", size=(16, 16))
        assert workflow.state.stage == WorkflowStage.HARDWARE_MAPPED
    
    def test_workflow_parallel_execution(self, workflow, temp_workspace):
        """Test workflow parallel execution capabilities."""
        # Create model file
        model_file = temp_workspace / "parallel_test_model.onnx"
        model_file.write_bytes(b"parallel_mock_model_data")
        workflow.config.model_path = str(model_file)
        workflow.config.parallel_execution = True
        
        workflow.import_model()
        workflow.map_to_hardware(template="systolic_array", size=(16, 16))
        
        # Test parallel optimization with multiple strategies
        parallel_results = workflow.optimize_parallel(
            strategies=["quantization", "pruning"],
            target_fps=30.0,
            power_budget=5.0,
            max_workers=2
        )
        
        assert len(parallel_results) == 2
        assert "quantization" in parallel_results
        assert "pruning" in parallel_results
        
        for strategy, result in parallel_results.items():
            assert result is not None
            assert hasattr(result, 'final_metrics')
    
    def test_workflow_metrics_tracking(self, workflow, temp_workspace):
        """Test workflow metrics tracking."""
        # Execute workflow with metrics tracking
        model_file = temp_workspace / "metrics_test_model.onnx"
        model_file.write_bytes(b"metrics_mock_model_data")
        workflow.config.model_path = str(model_file)
        
        start_time = time.time()
        
        workflow.import_model()
        workflow.map_to_hardware(template="systolic_array", size=(8, 8))
        workflow.optimize(target_fps=30.0, power_budget=5.0)
        
        end_time = time.time()
        
        # Check metrics
        metrics = workflow.get_metrics()
        
        assert metrics.total_execution_time > 0
        assert metrics.total_execution_time <= (end_time - start_time + 1)  # Allow some tolerance
        assert metrics.stage_times["model_import"] > 0
        assert metrics.stage_times["hardware_mapping"] > 0
        assert metrics.stage_times["optimization"] > 0
        assert metrics.memory_usage_mb > 0
    
    def test_workflow_validation(self, workflow):
        """Test workflow validation at different stages."""
        # Initial validation should pass
        assert workflow.validate()
        
        # Validation after model import
        workflow.state.stage = WorkflowStage.MODEL_IMPORTED
        workflow.model = Mock()
        assert workflow.validate()
        
        # Validation should fail if model is missing
        workflow.model = None
        assert not workflow.validate()
        assert workflow.state.has_errors()
    
    def test_workflow_cleanup(self, workflow, temp_workspace):
        """Test workflow cleanup functionality."""
        # Execute partial workflow to create files
        model_file = temp_workspace / "cleanup_test_model.onnx"
        model_file.write_bytes(b"cleanup_mock_model_data")
        workflow.config.model_path = str(model_file)
        
        workflow.import_model()
        workflow.map_to_hardware(template="systolic_array", size=(8, 8))
        
        # Check that workspace has files
        workspace_files = list(workflow.workspace_dir.glob("*"))
        assert len(workspace_files) > 0
        
        # Cleanup
        workflow.cleanup(remove_outputs=False, remove_workspace=True)
        
        # Workspace should be cleaned but outputs preserved
        assert not workflow.workspace_dir.exists() or len(list(workflow.workspace_dir.glob("*"))) == 0


class TestWorkflowManager:
    """Test WorkflowManager class functionality."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def workflow_manager(self, temp_workspace):
        """WorkflowManager instance for testing."""
        return WorkflowManager(workspace_dir=temp_workspace)
    
    def test_workflow_manager_creation(self, workflow_manager, temp_workspace):
        """Test WorkflowManager creation."""
        assert workflow_manager.workspace_dir == temp_workspace
        assert len(workflow_manager.active_workflows) == 0
        assert len(workflow_manager.completed_workflows) == 0
    
    def test_workflow_manager_create_workflow(self, workflow_manager, temp_workspace):
        """Test workflow creation through manager."""
        config = WorkflowConfig(
            name="manager_test_workflow",
            model_path=str(temp_workspace / "test_model.onnx"),
            input_shapes={"input": (1, 3, 224, 224)}
        )
        
        workflow_id = workflow_manager.create_workflow(config)
        
        assert workflow_id is not None
        assert workflow_id in workflow_manager.active_workflows
        assert len(workflow_manager.active_workflows) == 1
        
        workflow = workflow_manager.get_workflow(workflow_id)
        assert workflow is not None
        assert workflow.config.name == "manager_test_workflow"
    
    def test_workflow_manager_execute_workflow(self, workflow_manager, temp_workspace):
        """Test workflow execution through manager."""
        # Create model file
        model_file = temp_workspace / "manager_execution_test.onnx"
        model_file.write_bytes(b"manager_execution_mock_data")
        
        config = WorkflowConfig(
            name="execution_test",
            model_path=str(model_file),
            input_shapes={"input": (1, 3, 224, 224)}
        )
        
        workflow_id = workflow_manager.create_workflow(config)
        
        # Execute workflow
        result = workflow_manager.execute_workflow(
            workflow_id=workflow_id,
            hardware_config={"template": "systolic_array", "size": (8, 8)},
            optimization_config={"target_fps": 30.0, "power_budget": 5.0}
        )
        
        assert result is not None
        assert workflow_id in workflow_manager.completed_workflows
        assert workflow_id not in workflow_manager.active_workflows
        
        completed_workflow = workflow_manager.get_workflow(workflow_id)
        assert completed_workflow.state.stage == WorkflowStage.COMPLETED
    
    def test_workflow_manager_parallel_execution(self, workflow_manager, temp_workspace):
        """Test parallel workflow execution."""
        # Create multiple workflows
        workflow_ids = []
        for i in range(3):
            model_file = temp_workspace / f"parallel_model_{i}.onnx"
            model_file.write_bytes(f"parallel_mock_data_{i}".encode())
            
            config = WorkflowConfig(
                name=f"parallel_workflow_{i}",
                model_path=str(model_file),
                input_shapes={"input": (1, 3, 224, 224)}
            )
            
            workflow_id = workflow_manager.create_workflow(config)
            workflow_ids.append(workflow_id)
        
        # Execute workflows in parallel
        results = workflow_manager.execute_workflows_parallel(
            workflow_ids=workflow_ids,
            hardware_config={"template": "systolic_array", "size": (8, 8)},
            optimization_config={"target_fps": 30.0, "power_budget": 5.0},
            max_workers=2
        )
        
        assert len(results) == 3
        assert all(workflow_id in results for workflow_id in workflow_ids)
        assert len(workflow_manager.completed_workflows) == 3
        assert len(workflow_manager.active_workflows) == 0
    
    def test_workflow_manager_monitoring(self, workflow_manager, temp_workspace):
        """Test workflow monitoring capabilities."""
        # Create and start workflow
        model_file = temp_workspace / "monitoring_test.onnx"
        model_file.write_bytes(b"monitoring_mock_data")
        
        config = WorkflowConfig(
            name="monitoring_test",
            model_path=str(model_file),
            input_shapes={"input": (1, 3, 224, 224)}
        )
        
        workflow_id = workflow_manager.create_workflow(config)
        
        # Get initial status
        status = workflow_manager.get_workflow_status(workflow_id)
        assert status["stage"] == "INITIALIZED"
        assert status["progress"] == 0.0
        
        # Execute workflow partially
        workflow = workflow_manager.get_workflow(workflow_id)
        workflow.import_model()
        
        # Check updated status
        status = workflow_manager.get_workflow_status(workflow_id)
        assert status["stage"] == "MODEL_IMPORTED"
        assert status["progress"] > 0.0
    
    def test_workflow_manager_cleanup(self, workflow_manager, temp_workspace):
        """Test workflow manager cleanup functionality."""
        # Create multiple workflows
        workflow_ids = []
        for i in range(2):
            model_file = temp_workspace / f"cleanup_model_{i}.onnx"
            model_file.write_bytes(f"cleanup_mock_data_{i}".encode())
            
            config = WorkflowConfig(
                name=f"cleanup_workflow_{i}",
                model_path=str(model_file),
                input_shapes={"input": (1, 3, 224, 224)}
            )
            
            workflow_id = workflow_manager.create_workflow(config)
            workflow_ids.append(workflow_id)
        
        # Execute one workflow
        workflow_manager.execute_workflow(
            workflow_ids[0],
            hardware_config={"template": "systolic_array", "size": (8, 8)},
            optimization_config={"target_fps": 30.0, "power_budget": 5.0}
        )
        
        # Cleanup completed workflows
        cleaned_count = workflow_manager.cleanup_completed_workflows(max_age_hours=0)
        assert cleaned_count == 1
        assert len(workflow_manager.completed_workflows) == 0


class TestWorkflowMetrics:
    """Test WorkflowMetrics class functionality."""
    
    def test_workflow_metrics_creation(self):
        """Test WorkflowMetrics creation and initialization."""
        metrics = WorkflowMetrics()
        
        assert metrics.start_time > 0
        assert metrics.end_time is None
        assert metrics.total_execution_time == 0
        assert len(metrics.stage_times) == 0
        assert metrics.memory_usage_mb == 0
        assert metrics.cpu_usage_percent == 0
    
    def test_workflow_metrics_stage_timing(self):
        """Test stage timing measurement."""
        metrics = WorkflowMetrics()
        
        # Start and end stage timing
        metrics.start_stage("model_import")
        time.sleep(0.1)  # Small delay for testing
        metrics.end_stage("model_import")
        
        assert "model_import" in metrics.stage_times
        assert metrics.stage_times["model_import"] > 0.05  # Should be at least 50ms
        assert metrics.stage_times["model_import"] < 1.0   # Should be less than 1s
    
    def test_workflow_metrics_finalization(self):
        """Test metrics finalization."""
        metrics = WorkflowMetrics()
        time.sleep(0.1)  # Small delay
        
        metrics.finalize()
        
        assert metrics.end_time is not None
        assert metrics.total_execution_time > 0.05
        assert metrics.total_execution_time < 1.0
    
    def test_workflow_metrics_serialization(self):
        """Test metrics serialization."""
        metrics = WorkflowMetrics()
        metrics.start_stage("test_stage")
        time.sleep(0.05)
        metrics.end_stage("test_stage")
        metrics.memory_usage_mb = 128.5
        metrics.cpu_usage_percent = 45.2
        metrics.finalize()
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict["total_execution_time"] > 0
        assert metrics_dict["memory_usage_mb"] == 128.5
        assert metrics_dict["cpu_usage_percent"] == 45.2
        assert "test_stage" in metrics_dict["stage_times"]


class TestWorkflowIntegration:
    """Integration tests for workflow components."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_end_to_end_workflow_integration(self, temp_workspace):
        """Test complete end-to-end workflow integration."""
        # Create workflow manager
        manager = WorkflowManager(workspace_dir=temp_workspace)
        
        # Create model file
        model_file = temp_workspace / "integration_test_model.onnx"
        model_file.write_bytes(b"integration_test_mock_data")
        
        # Define workflow configuration
        config = WorkflowConfig(
            name="integration_test_workflow",
            model_path=str(model_file),
            input_shapes={"input": (1, 3, 224, 224)},
            framework="onnx",
            optimization_level="moderate",
            parallel_execution=True
        )
        
        # Create workflow
        workflow_id = manager.create_workflow(config)
        
        # Execute complete workflow
        result = manager.execute_workflow(
            workflow_id=workflow_id,
            hardware_config={
                "template": "systolic_array",
                "size": (16, 16),
                "precision": "int8",
                "frequency_mhz": 400.0
            },
            optimization_config={
                "target_fps": 30.0,
                "power_budget": 5.0,
                "area_budget": 150.0,
                "optimization_level": "moderate"
            },
            rtl_config={
                "output_format": "verilog",
                "optimization_passes": ["constant_folding", "dead_code_elimination"]
            }
        )
        
        # Verify complete integration
        assert result is not None
        assert "rtl_file" in result
        assert "optimization_result" in result
        assert "performance_metrics" in result
        
        # Verify workflow state
        workflow = manager.get_workflow(workflow_id)
        assert workflow.state.stage == WorkflowStage.COMPLETED
        assert workflow.state.progress == 100.0
        assert not workflow.state.has_errors()
        
        # Verify files were created
        assert result["rtl_file"].exists()
        assert result["rtl_file"].suffix == ".v"
        
        # Verify metrics
        metrics = workflow.get_metrics()
        assert metrics.total_execution_time > 0
        assert "model_import" in metrics.stage_times
        assert "hardware_mapping" in metrics.stage_times
        assert "optimization" in metrics.stage_times
        assert "rtl_generation" in metrics.stage_times


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
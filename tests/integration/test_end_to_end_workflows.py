"""
Advanced integration tests for end-to-end AI Hardware Co-Design workflows.

This module tests complete workflows from model import through RTL generation,
including realistic scenarios and production-like conditions.
"""

import pytest
import tempfile
import shutil
import time
import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from codesign_playground.core.workflow import Workflow, WorkflowManager, WorkflowConfig
from codesign_playground.core.accelerator import AcceleratorDesigner, ModelProfile
from codesign_playground.core.optimizer import ModelOptimizer
from codesign_playground.core.explorer import DesignSpaceExplorer
from codesign_playground.utils.exceptions import WorkflowError, OptimizationError
from codesign_playground.utils.monitoring import get_system_monitor
from codesign_playground.utils.compliance import get_compliance_manager


class TestRealisticWorkflows:
    """Test realistic end-to-end workflow scenarios."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def workflow_manager(self, temp_workspace):
        """WorkflowManager instance with temporary workspace."""
        return WorkflowManager(workspace_dir=temp_workspace)
    
    @pytest.fixture
    def realistic_models(self, temp_workspace):
        """Create realistic model files for testing."""
        models = {}
        
        # CNN model (ImageNet classification)
        cnn_model = temp_workspace / "resnet18.onnx"
        cnn_model.write_bytes(b"mock_resnet18_onnx_data" * 1000)  # Simulate larger file
        models["cnn"] = {
            "path": str(cnn_model),
            "framework": "onnx",
            "input_shapes": {"input": (1, 3, 224, 224)},
            "expected_gflops": 18.0,
            "expected_params": 11000000
        }
        
        # Transformer model (NLP)
        transformer_model = temp_workspace / "bert_base.onnx"
        transformer_model.write_bytes(b"mock_bert_base_onnx_data" * 2000)
        models["transformer"] = {
            "path": str(transformer_model),
            "framework": "onnx",
            "input_shapes": {"input_ids": (1, 512), "attention_mask": (1, 512)},
            "expected_gflops": 45.0,
            "expected_params": 110000000
        }
        
        # Lightweight model (Edge deployment)
        mobile_model = temp_workspace / "mobilenet_v2.onnx"
        mobile_model.write_bytes(b"mock_mobilenet_v2_onnx_data" * 300)
        models["mobile"] = {
            "path": str(mobile_model),
            "framework": "onnx",
            "input_shapes": {"input": (1, 3, 224, 224)},
            "expected_gflops": 3.0,
            "expected_params": 3500000
        }
        
        return models
    
    def test_cnn_classification_workflow(self, workflow_manager, realistic_models):
        """Test complete CNN classification workflow."""
        model_info = realistic_models["cnn"]
        
        # Define workflow configuration
        config = WorkflowConfig(
            name="cnn_classification_e2e",
            model_path=model_info["path"],
            input_shapes=model_info["input_shapes"],
            framework=model_info["framework"],
            optimization_level="moderate",
            target_application="image_classification"
        )
        
        # Create and execute workflow
        workflow_id = workflow_manager.create_workflow(config)
        
        result = workflow_manager.execute_workflow(
            workflow_id=workflow_id,
            hardware_config={
                "template": "systolic_array",
                "size": (32, 32),
                "precision": "int8",
                "frequency_mhz": 400.0,
                "memory_hierarchy": ["sram_128kb", "dram"]
            },
            optimization_config={
                "target_fps": 30.0,
                "power_budget": 8.0,
                "area_budget": 200.0,
                "accuracy_threshold": 0.95,
                "optimization_strategies": ["quantization", "pruning"]
            },
            rtl_config={
                "output_format": "verilog",
                "optimization_passes": ["constant_folding", "dead_code_elimination", "memory_optimization"],
                "clock_frequency": 400.0
            }
        )
        
        # Verify workflow completion
        assert result is not None
        workflow = workflow_manager.get_workflow(workflow_id)
        assert workflow.state.stage.value == "COMPLETED"
        assert workflow.state.progress == 100.0
        
        # Verify performance metrics
        assert result["performance_metrics"]["estimated_fps"] >= 25.0  # Close to target
        assert result["performance_metrics"]["estimated_power"] <= 8.5  # Within budget + margin
        assert result["performance_metrics"]["estimated_area"] <= 220.0  # Within budget + margin
        
        # Verify RTL generation
        assert result["rtl_file"].exists()
        rtl_content = result["rtl_file"].read_text()
        assert "module" in rtl_content
        assert "systolic_array" in rtl_content.lower()
        assert "int8" in rtl_content.lower() or "8" in rtl_content
        
        # Verify optimization results
        opt_result = result["optimization_result"]
        assert opt_result.final_metrics["accuracy"] >= 0.95
        assert opt_result.final_metrics["power"] <= 8.0
    
    def test_transformer_nlp_workflow(self, workflow_manager, realistic_models):
        """Test complete transformer NLP workflow."""
        model_info = realistic_models["transformer"]
        
        config = WorkflowConfig(
            name="transformer_nlp_e2e",
            model_path=model_info["path"],
            input_shapes=model_info["input_shapes"],
            framework=model_info["framework"],
            optimization_level="aggressive",
            target_application="natural_language_processing"
        )
        
        workflow_id = workflow_manager.create_workflow(config)
        
        result = workflow_manager.execute_workflow(
            workflow_id=workflow_id,
            hardware_config={
                "template": "transformer_accelerator",
                "attention_heads": 12,
                "precision": "fp16",
                "frequency_mhz": 600.0,
                "memory_hierarchy": ["sram_256kb", "sram_1mb", "dram"]
            },
            optimization_config={
                "target_latency_ms": 100.0,
                "power_budget": 15.0,
                "area_budget": 500.0,
                "accuracy_threshold": 0.98,
                "optimization_strategies": ["attention_optimization", "layer_fusion"]
            },
            rtl_config={
                "output_format": "verilog",
                "optimization_passes": ["attention_fusion", "memory_optimization"],
                "enable_pipeline": True
            }
        )
        
        # Verify transformer-specific optimizations
        assert result is not None
        workflow = workflow_manager.get_workflow(workflow_id)
        assert workflow.state.stage.value == "COMPLETED"
        
        # Verify transformer performance characteristics
        assert result["performance_metrics"]["estimated_latency_ms"] <= 120.0
        assert result["performance_metrics"]["estimated_power"] <= 16.0
        
        # Verify attention mechanism in RTL
        rtl_content = result["rtl_file"].read_text()
        assert any(keyword in rtl_content.lower() for keyword in ["attention", "transformer"])
    
    def test_mobile_edge_workflow(self, workflow_manager, realistic_models):
        """Test mobile/edge deployment workflow with strict constraints."""
        model_info = realistic_models["mobile"]
        
        config = WorkflowConfig(
            name="mobile_edge_e2e",
            model_path=model_info["path"],
            input_shapes=model_info["input_shapes"],
            framework=model_info["framework"],
            optimization_level="aggressive",
            target_application="edge_inference"
        )
        
        workflow_id = workflow_manager.create_workflow(config)
        
        result = workflow_manager.execute_workflow(
            workflow_id=workflow_id,
            hardware_config={
                "template": "vector_processor",
                "vector_width": 128,
                "precision": "int8",
                "frequency_mhz": 200.0,
                "memory_hierarchy": ["sram_64kb", "dram"]
            },
            optimization_config={
                "target_fps": 15.0,  # Lower for edge
                "power_budget": 2.0,  # Very strict for mobile
                "area_budget": 50.0,  # Very strict for mobile
                "accuracy_threshold": 0.90,
                "optimization_strategies": ["aggressive_quantization", "channel_pruning", "knowledge_distillation"]
            },
            rtl_config={
                "output_format": "verilog",
                "optimization_passes": ["power_optimization", "area_optimization"],
                "low_power_mode": True
            }
        )
        
        # Verify edge constraints are met
        assert result is not None
        workflow = workflow_manager.get_workflow(workflow_id)
        assert workflow.state.stage.value == "COMPLETED"
        
        # Strict edge constraints
        assert result["performance_metrics"]["estimated_power"] <= 2.2  # Within strict budget
        assert result["performance_metrics"]["estimated_area"] <= 55.0  # Within strict budget
        assert result["performance_metrics"]["estimated_fps"] >= 12.0  # Acceptable for edge
        
        # Verify power-optimized RTL
        rtl_content = result["rtl_file"].read_text()
        assert "power" in rtl_content.lower() or "low_power" in rtl_content.lower()
    
    def test_multi_model_batch_workflow(self, workflow_manager, realistic_models):
        """Test batch processing of multiple models."""
        configs = []
        
        for model_type, model_info in realistic_models.items():
            config = WorkflowConfig(
                name=f"batch_{model_type}_workflow",
                model_path=model_info["path"],
                input_shapes=model_info["input_shapes"],
                framework=model_info["framework"],
                optimization_level="moderate"
            )
            configs.append(config)
        
        # Create all workflows
        workflow_ids = []
        for config in configs:
            workflow_id = workflow_manager.create_workflow(config)
            workflow_ids.append(workflow_id)
        
        # Execute in parallel
        hardware_configs = [
            {"template": "systolic_array", "size": (16, 16), "precision": "int8"},
            {"template": "transformer_accelerator", "attention_heads": 8, "precision": "fp16"},
            {"template": "vector_processor", "vector_width": 64, "precision": "int8"}
        ]
        
        optimization_config = {
            "target_fps": 20.0,
            "power_budget": 10.0,
            "area_budget": 300.0
        }
        
        start_time = time.time()
        results = workflow_manager.execute_workflows_parallel(
            workflow_ids=workflow_ids,
            hardware_config=hardware_configs[0],  # Use same config for simplicity
            optimization_config=optimization_config,
            max_workers=3
        )
        parallel_time = time.time() - start_time
        
        # Verify all workflows completed
        assert len(results) == len(workflow_ids)
        assert all(result is not None for result in results.values())
        
        # Verify parallel execution was efficient
        assert parallel_time < 60.0  # Should complete within reasonable time
        
        # Verify each workflow produced valid results
        for workflow_id, result in results.items():
            assert "rtl_file" in result
            assert result["rtl_file"].exists()
            assert "performance_metrics" in result
    
    def test_iterative_design_optimization(self, workflow_manager, realistic_models):
        """Test iterative design optimization workflow."""
        model_info = realistic_models["cnn"]
        
        config = WorkflowConfig(
            name="iterative_optimization_e2e",
            model_path=model_info["path"],
            input_shapes=model_info["input_shapes"],
            framework=model_info["framework"]
        )
        
        workflow_id = workflow_manager.create_workflow(config)
        workflow = workflow_manager.get_workflow(workflow_id)
        
        # Phase 1: Initial design with conservative parameters
        workflow.import_model()
        workflow.map_to_hardware(
            template="systolic_array",
            size=(16, 16),
            precision="fp16"
        )
        
        initial_result = workflow.optimize(
            target_fps=20.0,
            power_budget=10.0,
            optimization_level="conservative"
        )
        
        # Phase 2: Analyze results and adjust
        if initial_result.final_metrics["power"] < 8.0:
            # Power budget allows for more aggressive optimization
            workflow.map_to_hardware(
                template="systolic_array",
                size=(32, 32),
                precision="int8"  # More aggressive quantization
            )
            
            improved_result = workflow.optimize(
                target_fps=30.0,
                power_budget=10.0,
                optimization_level="aggressive"
            )
        else:
            improved_result = initial_result
        
        # Phase 3: Final RTL generation
        rtl_file = workflow.generate_rtl(
            output_format="verilog",
            optimization_passes=["all"]
        )
        
        # Verify iterative improvement
        assert workflow.state.stage.value == "RTL_GENERATED"
        assert rtl_file.exists()
        
        # Performance should meet or exceed targets
        final_metrics = improved_result.final_metrics
        assert final_metrics["estimated_fps"] >= 18.0  # Close to target
        assert final_metrics["power"] <= 10.5  # Within budget + margin
    
    def test_failure_recovery_workflow(self, workflow_manager, realistic_models):
        """Test workflow failure recovery and retry mechanisms."""
        model_info = realistic_models["cnn"]
        
        config = WorkflowConfig(
            name="failure_recovery_e2e",
            model_path=model_info["path"],
            input_shapes=model_info["input_shapes"],
            framework=model_info["framework"]
        )
        
        workflow_id = workflow_manager.create_workflow(config)
        workflow = workflow_manager.get_workflow(workflow_id)
        
        # Execute successful phases
        workflow.import_model()
        workflow.map_to_hardware(template="systolic_array", size=(16, 16))
        
        # Simulate failure during optimization
        with patch.object(workflow, 'optimize', side_effect=OptimizationError("Simulated optimization failure")):
            with pytest.raises(OptimizationError):
                workflow.optimize(target_fps=30.0, power_budget=5.0)
        
        # Verify workflow state after failure
        assert workflow.state.stage.value == "HARDWARE_MAPPED"  # Should remain in previous valid state
        assert workflow.state.has_errors()
        
        # Recovery: Clear errors and retry with different parameters
        workflow.clear_errors()
        
        # Retry with more conservative parameters
        recovery_result = workflow.optimize(
            target_fps=20.0,  # Lower target
            power_budget=8.0,  # Higher budget
            optimization_level="conservative"
        )
        
        # Complete workflow after recovery
        rtl_file = workflow.generate_rtl()
        
        # Verify successful recovery
        assert workflow.state.stage.value == "RTL_GENERATED"
        assert not workflow.state.has_errors()
        assert rtl_file.exists()
        assert recovery_result.final_metrics["power"] <= 8.0


class TestPerformanceScenarios:
    """Test performance-critical scenarios and scaling."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_high_throughput_workflow(self, temp_workspace):
        """Test high-throughput workflow scenario."""
        # Create large model file
        large_model = temp_workspace / "large_model.onnx"
        large_model.write_bytes(b"large_model_data" * 10000)  # Simulate large model
        
        config = WorkflowConfig(
            name="high_throughput_e2e",
            model_path=str(large_model),
            input_shapes={"input": (1, 3, 224, 224)},
            framework="onnx",
            parallel_execution=True
        )
        
        workflow = Workflow(config)
        
        start_time = time.time()
        
        # Execute workflow optimized for throughput
        workflow.import_model()
        workflow.map_to_hardware(
            template="systolic_array",
            size=(64, 64),  # Large array for high throughput
            precision="int8",
            frequency_mhz=800.0  # High frequency
        )
        
        result = workflow.optimize(
            target_fps=120.0,  # Very high target
            power_budget=25.0,  # Higher power budget for performance
            optimization_level="aggressive"
        )
        
        workflow.generate_rtl()
        
        execution_time = time.time() - start_time
        
        # Verify high-throughput characteristics
        assert workflow.state.stage.value == "RTL_GENERATED"
        assert result.final_metrics["estimated_fps"] >= 100.0
        assert execution_time < 30.0  # Should complete efficiently despite large model
    
    def test_low_latency_workflow(self, temp_workspace):
        """Test low-latency real-time workflow scenario."""
        model_file = temp_workspace / "realtime_model.onnx"
        model_file.write_bytes(b"realtime_model_data" * 1000)
        
        config = WorkflowConfig(
            name="low_latency_e2e",
            model_path=str(model_file),
            input_shapes={"input": (1, 3, 64, 64)},  # Smaller input for low latency
            framework="onnx"
        )
        
        workflow = Workflow(config)
        
        # Execute workflow optimized for latency
        workflow.import_model()
        workflow.map_to_hardware(
            template="vector_processor",
            vector_width=256,
            precision="fp16",
            frequency_mhz=1000.0,  # Very high frequency
            pipeline_depth=8  # Deep pipeline for latency
        )
        
        result = workflow.optimize(
            target_latency_ms=5.0,  # Very low latency target
            power_budget=20.0,
            optimization_strategies=["pipeline_optimization", "memory_prefetch"]
        )
        
        rtl_file = workflow.generate_rtl(
            optimization_passes=["pipeline_optimization", "timing_optimization"]
        )
        
        # Verify low-latency characteristics
        assert result.final_metrics["estimated_latency_ms"] <= 7.0  # Close to target
        
        # Verify pipeline optimizations in RTL
        rtl_content = rtl_file.read_text()
        assert any(keyword in rtl_content.lower() for keyword in ["pipeline", "register"])
    
    def test_memory_constrained_workflow(self, temp_workspace):
        """Test workflow with strict memory constraints."""
        model_file = temp_workspace / "memory_constrained_model.onnx"
        model_file.write_bytes(b"memory_constrained_data" * 2000)
        
        config = WorkflowConfig(
            name="memory_constrained_e2e",
            model_path=str(model_file),
            input_shapes={"input": (1, 3, 224, 224)},
            framework="onnx"
        )
        
        workflow = Workflow(config)
        
        # Execute workflow with strict memory constraints
        workflow.import_model()
        workflow.map_to_hardware(
            template="systolic_array",
            size=(16, 16),
            precision="int8",
            memory_hierarchy=["sram_32kb"],  # Very limited memory
            memory_optimization=True
        )
        
        result = workflow.optimize(
            target_fps=15.0,
            power_budget=5.0,
            memory_budget_mb=16,  # Very strict memory limit
            optimization_strategies=["memory_optimization", "weight_sharing", "activation_checkpointing"]
        )
        
        rtl_file = workflow.generate_rtl(
            optimization_passes=["memory_optimization", "buffer_optimization"]
        )
        
        # Verify memory optimizations
        assert result.final_metrics["estimated_memory_mb"] <= 18.0  # Within budget + margin
        
        # Verify memory optimizations in RTL
        rtl_content = rtl_file.read_text()
        assert any(keyword in rtl_content.lower() for keyword in ["memory", "buffer", "cache"])


class TestComplianceIntegration:
    """Test integration with compliance and monitoring systems."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_gdpr_compliant_workflow(self, temp_workspace):
        """Test workflow with GDPR compliance tracking."""
        compliance_manager = get_compliance_manager()
        
        model_file = temp_workspace / "gdpr_model.onnx"
        model_file.write_bytes(b"gdpr_compliant_model_data" * 1000)
        
        config = WorkflowConfig(
            name="gdpr_compliant_e2e",
            model_path=str(model_file),
            input_shapes={"input": (1, 3, 224, 224)},
            framework="onnx",
            compliance_tracking=True
        )
        
        workflow = Workflow(config)
        user_id = "test_user_gdpr_001"
        
        # Record GDPR-relevant data processing
        compliance_manager.record_data_processing(
            user_id=user_id,
            data_category="MODEL_ARTIFACTS",
            processing_purpose="hardware_acceleration_design",
            legal_basis="legitimate_interests"
        )
        
        # Execute workflow with compliance tracking
        workflow.import_model()
        workflow.map_to_hardware(template="systolic_array", size=(16, 16))
        result = workflow.optimize(target_fps=30.0, power_budget=5.0)
        rtl_file = workflow.generate_rtl()
        
        # Log completion in compliance system
        compliance_manager._log_audit_event(
            user_id=user_id,
            action="workflow_completion",
            resource="hardware_design",
            data_category="MODEL_ARTIFACTS",
            legal_basis="legitimate_interests",
            result="success"
        )
        
        # Verify workflow completion
        assert workflow.state.stage.value == "RTL_GENERATED"
        assert rtl_file.exists()
        
        # Verify compliance tracking
        report = compliance_manager.generate_compliance_report(
            start_date=time.time() - 3600,
            end_date=time.time()
        )
        
        assert report["processing_activities"]["total"] >= 1
        assert any("hardware_acceleration_design" in str(activity) 
                 for activity in report["processing_activities"]["details"])
    
    def test_monitored_workflow_performance(self, temp_workspace):
        """Test workflow with comprehensive performance monitoring."""
        monitor = get_system_monitor()
        
        model_file = temp_workspace / "monitored_model.onnx"
        model_file.write_bytes(b"monitored_model_data" * 1500)
        
        config = WorkflowConfig(
            name="monitored_performance_e2e",
            model_path=str(model_file),
            input_shapes={"input": (1, 3, 224, 224)},
            framework="onnx",
            performance_monitoring=True
        )
        
        workflow = Workflow(config)
        
        # Start monitoring
        monitor.start_monitoring("workflow_execution")
        
        try:
            # Execute workflow with monitoring
            workflow.import_model()
            
            # Check intermediate metrics
            import_metrics = monitor.get_current_metrics()
            assert import_metrics["cpu_usage_percent"] >= 0
            assert import_metrics["memory_usage_mb"] > 0
            
            workflow.map_to_hardware(template="systolic_array", size=(32, 32))
            workflow.optimize(target_fps=30.0, power_budget=8.0)
            workflow.generate_rtl()
            
            # Get final metrics
            final_metrics = monitor.get_current_metrics()
            
            # Verify monitoring data
            assert final_metrics["total_execution_time"] > 0
            assert final_metrics["peak_memory_usage_mb"] > import_metrics["memory_usage_mb"]
            
        finally:
            monitor.stop_monitoring("workflow_execution")
        
        # Verify workflow completion
        assert workflow.state.stage.value == "RTL_GENERATED"
        
        # Verify performance metrics are within acceptable ranges
        workflow_metrics = workflow.get_metrics()
        assert workflow_metrics.total_execution_time < 60.0  # Should complete within 1 minute
        assert workflow_metrics.memory_usage_mb < 500.0  # Reasonable memory usage
    
    def test_audit_trail_workflow(self, temp_workspace):
        """Test workflow with complete audit trail."""
        model_file = temp_workspace / "audit_model.onnx"
        model_file.write_bytes(b"audit_trail_model_data" * 800)
        
        config = WorkflowConfig(
            name="audit_trail_e2e",
            model_path=str(model_file),
            input_shapes={"input": (1, 3, 224, 224)},
            framework="onnx",
            audit_logging=True
        )
        
        workflow = Workflow(config)
        
        # Execute workflow with audit trail
        audit_events = []
        
        def audit_callback(event_type, event_data):
            audit_events.append({
                "timestamp": time.time(),
                "event_type": event_type,
                "data": event_data
            })
        
        workflow.set_audit_callback(audit_callback)
        
        workflow.import_model()
        workflow.map_to_hardware(template="systolic_array", size=(16, 16))
        result = workflow.optimize(target_fps=30.0, power_budget=5.0)
        rtl_file = workflow.generate_rtl()
        
        # Verify audit trail
        assert len(audit_events) >= 4  # At least one event per major stage
        
        event_types = [event["event_type"] for event in audit_events]
        assert "model_imported" in event_types
        assert "hardware_mapped" in event_types
        assert "optimization_completed" in event_types
        assert "rtl_generated" in event_types
        
        # Verify audit data integrity
        for event in audit_events:
            assert "timestamp" in event
            assert "event_type" in event
            assert "data" in event
            assert event["timestamp"] > 0
        
        # Save audit trail
        audit_file = temp_workspace / "audit_trail.json"
        with open(audit_file, 'w') as f:
            json.dump(audit_events, f, indent=2)
        
        assert audit_file.exists()
        assert audit_file.stat().st_size > 0


class TestProductionReadiness:
    """Test production readiness scenarios."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_concurrent_user_workflows(self, temp_workspace):
        """Test concurrent workflows from multiple users."""
        manager = WorkflowManager(workspace_dir=temp_workspace)
        
        # Create multiple user scenarios
        user_scenarios = [
            {
                "user_id": "user_001",
                "model_type": "cnn",
                "constraints": {"target_fps": 30.0, "power_budget": 5.0}
            },
            {
                "user_id": "user_002", 
                "model_type": "transformer",
                "constraints": {"target_latency_ms": 50.0, "power_budget": 12.0}
            },
            {
                "user_id": "user_003",
                "model_type": "mobile",
                "constraints": {"target_fps": 15.0, "power_budget": 2.0}
            }
        ]
        
        # Create model files for each user
        workflow_ids = []
        for i, scenario in enumerate(user_scenarios):
            model_file = temp_workspace / f"user_{scenario['user_id']}_model.onnx"
            model_file.write_bytes(f"user_{scenario['user_id']}_model_data".encode() * 500)
            
            config = WorkflowConfig(
                name=f"user_{scenario['user_id']}_workflow",
                model_path=str(model_file),
                input_shapes={"input": (1, 3, 224, 224)},
                framework="onnx",
                user_id=scenario['user_id']
            )
            
            workflow_id = manager.create_workflow(config)
            workflow_ids.append((workflow_id, scenario))
        
        # Execute workflows concurrently
        def execute_user_workflow(workflow_id_scenario):
            workflow_id, scenario = workflow_id_scenario
            
            try:
                result = manager.execute_workflow(
                    workflow_id=workflow_id,
                    hardware_config={
                        "template": "systolic_array",
                        "size": (16, 16),
                        "precision": "int8"
                    },
                    optimization_config=scenario["constraints"]
                )
                
                return {
                    "user_id": scenario["user_id"],
                    "workflow_id": workflow_id,
                    "status": "success",
                    "result": result
                }
                
            except Exception as e:
                return {
                    "user_id": scenario["user_id"],
                    "workflow_id": workflow_id,
                    "status": "error",
                    "error": str(e)
                }
        
        # Run concurrent workflows
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(execute_user_workflow, wf_scenario) 
                      for wf_scenario in workflow_ids]
            
            results = [future.result() for future in as_completed(futures, timeout=120)]
        
        # Verify all workflows completed successfully
        assert len(results) == 3
        successful_results = [r for r in results if r["status"] == "success"]
        assert len(successful_results) == 3
        
        # Verify resource isolation (no interference between users)
        for result in successful_results:
            assert result["result"] is not None
            assert "rtl_file" in result["result"]
            assert result["result"]["rtl_file"].exists()
    
    def test_resource_limits_and_quotas(self, temp_workspace):
        """Test resource limits and quota enforcement."""
        # Create workflow with resource limits
        model_file = temp_workspace / "resource_limited_model.onnx"
        model_file.write_bytes(b"resource_limited_data" * 5000)  # Large model
        
        config = WorkflowConfig(
            name="resource_limited_e2e",
            model_path=str(model_file),
            input_shapes={"input": (1, 3, 512, 512)},  # Large input
            framework="onnx",
            resource_limits={
                "max_memory_mb": 100,
                "max_execution_time_seconds": 30,
                "max_cpu_cores": 2
            }
        )
        
        workflow = Workflow(config)
        
        start_time = time.time()
        
        try:
            # Execute workflow within resource limits
            workflow.import_model()
            workflow.map_to_hardware(
                template="systolic_array",
                size=(16, 16),  # Modest size to stay within limits
                precision="int8"
            )
            
            result = workflow.optimize(
                target_fps=20.0,
                power_budget=8.0,
                optimization_level="moderate"  # Not aggressive to save resources
            )
            
            workflow.generate_rtl()
            
            execution_time = time.time() - start_time
            
            # Verify resource limits were respected
            assert execution_time <= 35.0  # Within time limit + small margin
            
            metrics = workflow.get_metrics()
            assert metrics.memory_usage_mb <= 120.0  # Within memory limit + margin
            
        except Exception as e:
            # If workflow fails due to resource limits, that's acceptable
            execution_time = time.time() - start_time
            assert execution_time <= 35.0  # Should fail fast if resource constrained
            assert "resource" in str(e).lower() or "limit" in str(e).lower()
    
    def test_disaster_recovery_workflow(self, temp_workspace):
        """Test disaster recovery and workflow persistence."""
        model_file = temp_workspace / "disaster_recovery_model.onnx"
        model_file.write_bytes(b"disaster_recovery_data" * 1000)
        
        config = WorkflowConfig(
            name="disaster_recovery_e2e",
            model_path=str(model_file),
            input_shapes={"input": (1, 3, 224, 224)},
            framework="onnx",
            enable_checkpointing=True
        )
        
        workflow = Workflow(config)
        
        # Execute workflow through several stages
        workflow.import_model()
        workflow.map_to_hardware(template="systolic_array", size=(16, 16))
        
        # Save checkpoint after hardware mapping
        checkpoint_file = workflow.save_checkpoint()
        assert checkpoint_file.exists()
        
        # Simulate system failure during optimization
        workflow_state_before = workflow.state.to_dict()
        
        # Simulate complete system restart
        del workflow
        
        # Restore workflow from checkpoint
        recovered_workflow = Workflow(config)
        recovered_workflow.restore_checkpoint(checkpoint_file)
        
        # Verify state was recovered correctly
        assert recovered_workflow.state.stage.value == "HARDWARE_MAPPED"
        assert recovered_workflow.accelerator is not None
        assert recovered_workflow.model is not None
        
        # Continue workflow from where it left off
        result = recovered_workflow.optimize(target_fps=30.0, power_budget=5.0)
        rtl_file = recovered_workflow.generate_rtl()
        
        # Verify successful completion after recovery
        assert recovered_workflow.state.stage.value == "RTL_GENERATED"
        assert rtl_file.exists()
        assert result.final_metrics["power"] <= 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
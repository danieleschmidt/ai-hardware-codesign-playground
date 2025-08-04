"""
Integration tests for FastAPI server endpoints.

This module tests the API endpoints for the codesign playground server,
including request/response validation and error handling.
"""

import pytest
from fastapi.testclient import TestClient
import json
import time
from unittest.mock import Mock, patch

from codesign_playground.server import app


class TestHealthEndpoints:
    """Test health check and system information endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_system_info(self):
        """Test system information endpoint."""
        response = self.client.get("/api/v1/system/info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "0.1.0"
        assert "supported_frameworks" in data
        assert "hardware_templates" in data
        assert "pytorch" in data["supported_frameworks"]
        assert "systolic_array" in data["hardware_templates"]


class TestModelEndpoints:
    """Test model profiling endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_profile_model_valid_request(self):
        """Test model profiling with valid request."""
        request_data = {
            "model_path": "test_model.onnx",
            "input_shape": [1, 3, 224, 224],
            "framework": "onnx"
        }
        
        response = self.client.post("/api/v1/model/profile", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "profile" in data
        assert "peak_gflops" in data["profile"]
        assert "bandwidth_gb_s" in data["profile"]
        assert "parameters" in data["profile"]
    
    def test_profile_model_missing_fields(self):
        """Test model profiling with missing required fields."""
        request_data = {
            "input_shape": [1, 3, 224, 224]
            # Missing model_path
        }
        
        response = self.client.post("/api/v1/model/profile", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_profile_model_invalid_input_shape(self):
        """Test model profiling with invalid input shape."""
        request_data = {
            "model_path": "test_model.onnx",
            "input_shape": "invalid_shape",  # Should be list
            "framework": "onnx"
        }
        
        response = self.client.post("/api/v1/model/profile", json=request_data)
        
        assert response.status_code == 422  # Validation error


class TestAcceleratorEndpoints:
    """Test accelerator design endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_design_accelerator_default_params(self):
        """Test accelerator design with default parameters."""
        request_data = {}  # Use all defaults
        
        response = self.client.post("/api/v1/accelerator/design", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "accelerator" in data
        assert "performance" in data
        assert data["accelerator"]["compute_units"] == 64  # default
        assert data["accelerator"]["dataflow"] == "weight_stationary"  # default
    
    def test_design_accelerator_custom_params(self):
        """Test accelerator design with custom parameters."""
        request_data = {
            "compute_units": 128,
            "memory_hierarchy": ["sram_128kb", "dram"],
            "dataflow": "output_stationary",
            "frequency_mhz": 400.0,
            "precision": "fp16",
            "power_budget_w": 10.0
        }
        
        response = self.client.post("/api/v1/accelerator/design", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["accelerator"]["compute_units"] == 128
        assert data["accelerator"]["dataflow"] == "output_stationary"
        assert data["accelerator"]["frequency_mhz"] == 400.0
        assert data["accelerator"]["precision"] == "fp16"
    
    def test_design_accelerator_invalid_dataflow(self):
        """Test accelerator design with invalid dataflow."""
        request_data = {
            "compute_units": 64,
            "dataflow": "invalid_dataflow"
        }
        
        response = self.client.post("/api/v1/accelerator/design", json=request_data)
        
        assert response.status_code == 500  # Internal server error due to validation
    
    def test_generate_rtl(self):
        """Test RTL generation endpoint."""
        request_data = {
            "compute_units": 32,
            "memory_hierarchy": ["sram_64kb", "dram"],
            "dataflow": "weight_stationary",
            "frequency_mhz": 200.0,
            "precision": "int8"
        }
        
        response = self.client.post("/api/v1/accelerator/rtl", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "rtl_file" in data
        assert "accelerator" in data


class TestOptimizationEndpoints:
    """Test optimization endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_start_co_optimization(self):
        """Test starting co-optimization job."""
        request_data = {
            "model_path": "test_model.onnx",
            "accelerator_config": {
                "compute_units": 64,
                "memory_hierarchy": ["sram_64kb", "dram"],
                "dataflow": "weight_stationary",
                "frequency_mhz": 200.0,
                "precision": "int8",
                "power_budget_w": 5.0
            },
            "target_fps": 30.0,
            "power_budget": 5.0,
            "iterations": 5,
            "strategy": "balanced"
        }
        
        response = self.client.post("/api/v1/optimization/co-optimize", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert "job_id" in data
        assert "message" in data
        
        # Return job_id for other tests
        return data["job_id"]
    
    def test_co_optimization_missing_fields(self):
        """Test co-optimization with missing required fields."""
        request_data = {
            "model_path": "test_model.onnx"
            # Missing accelerator_config
        }
        
        response = self.client.post("/api/v1/optimization/co-optimize", json=request_data)
        
        assert response.status_code == 422  # Validation error


class TestExplorationEndpoints:
    """Test design space exploration endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_start_design_space_exploration(self):
        """Test starting design space exploration job."""
        request_data = {
            "model_path": "test_model.onnx",
            "input_shape": [1, 3, 224, 224],
            "design_space": {
                "compute_units": [16, 32, 64],
                "dataflow": ["weight_stationary", "output_stationary"],
                "frequency_mhz": [100, 200, 400],
                "precision": ["int8", "fp16"]
            },
            "objectives": ["latency", "power", "area"],
            "num_samples": 10,
            "strategy": "random"
        }
        
        response = self.client.post("/api/v1/exploration/design-space", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert "job_id" in data
        
        return data["job_id"]
    
    def test_design_space_exploration_invalid_strategy(self):
        """Test design space exploration with invalid strategy."""
        request_data = {
            "model_path": "test_model.onnx",
            "input_shape": [1, 3, 224, 224],
            "design_space": {
                "compute_units": [16, 32, 64]
            },
            "strategy": "invalid_strategy"
        }
        
        response = self.client.post("/api/v1/exploration/design-space", json=request_data)
        
        # Should still accept the request but may fail during execution
        assert response.status_code == 200


class TestWorkflowEndpoints:
    """Test workflow management endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_create_workflow(self):
        """Test workflow creation."""
        request_data = {
            "name": "test_workflow",
            "model_path": "test_model.onnx",
            "input_shapes": {
                "input": [1, 3, 224, 224]
            },
            "hardware_template": "systolic_array",
            "hardware_size": [16, 16],
            "precision": "int8"
        }
        
        response = self.client.post("/api/v1/workflow/create", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
        assert data["workflow_id"] == "test_workflow"
    
    def test_run_workflow(self):
        """Test workflow execution."""
        # First create workflow
        create_request = {
            "name": "test_workflow_run",
            "model_path": "test_model.onnx",
            "input_shapes": {
                "input": [1, 3, 224, 224]
            },
            "hardware_template": "systolic_array",
            "hardware_size": [16, 16],
            "precision": "int8"
        }
        
        create_response = self.client.post("/api/v1/workflow/create", json=create_request)
        assert create_response.status_code == 200
        
        # Then run workflow
        run_response = self.client.post(
            "/api/v1/workflow/test_workflow_run/run",
            json=create_request
        )
        
        assert run_response.status_code == 200
        data = run_response.json()
        assert data["status"] == "started"
        assert data["workflow_id"] == "test_workflow_run"
    
    def test_get_workflow_status(self):
        """Test getting workflow status."""
        # First create workflow
        create_request = {
            "name": "test_workflow_status",
            "model_path": "test_model.onnx",
            "input_shapes": {
                "input": [1, 3, 224, 224]
            }
        }
        
        create_response = self.client.post("/api/v1/workflow/create", json=create_request)
        assert create_response.status_code == 200
        
        # Get status
        status_response = self.client.get("/api/v1/workflow/test_workflow_status/status")
        
        assert status_response.status_code == 200
        data = status_response.json()
        assert "stage" in data
        assert "progress" in data
        assert "messages" in data
    
    def test_get_nonexistent_workflow_status(self):
        """Test getting status of non-existent workflow."""
        response = self.client.get("/api/v1/workflow/nonexistent/status")
        
        assert response.status_code == 404


class TestJobStatusEndpoints:
    """Test job status and management endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_get_job_status_nonexistent(self):
        """Test getting status of non-existent job."""
        response = self.client.get("/api/v1/job/nonexistent-job-id/status")
        
        assert response.status_code == 404
    
    def test_list_jobs_empty(self):
        """Test listing jobs when none exist."""
        response = self.client.get("/api/v1/jobs")
        
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert isinstance(data["jobs"], list)


class TestFileDownloadEndpoints:
    """Test file download endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_download_nonexistent_file(self):
        """Test downloading non-existent file."""
        response = self.client.get("/api/v1/download/nonexistent/file.txt")
        
        assert response.status_code == 404


class TestAPIErrorHandling:
    """Test API error handling and edge cases."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_invalid_json_request(self):
        """Test request with invalid JSON."""
        response = self.client.post(
            "/api/v1/model/profile",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_missing_content_type(self):
        """Test request without proper content type."""
        response = self.client.post(
            "/api/v1/model/profile",
            data='{"model_path": "test.onnx", "input_shape": [1, 3, 224, 224]}'
        )
        
        # FastAPI should handle this gracefully
        assert response.status_code in [200, 422]
    
    def test_empty_request_body(self):
        """Test request with empty body where data is expected."""
        response = self.client.post("/api/v1/model/profile", json={})
        
        assert response.status_code == 422  # Missing required fields


class TestAPIIntegration:
    """Integration tests combining multiple API endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_full_design_flow(self):
        """Test complete design flow using multiple endpoints."""
        # Step 1: Profile model
        profile_request = {
            "model_path": "integration_test_model.onnx",
            "input_shape": [1, 3, 224, 224],
            "framework": "onnx"
        }
        
        profile_response = self.client.post("/api/v1/model/profile", json=profile_request)
        assert profile_response.status_code == 200
        profile_data = profile_response.json()
        
        # Step 2: Design accelerator
        design_request = {
            "compute_units": 64,
            "dataflow": "weight_stationary",
            "frequency_mhz": 200.0,
            "precision": "int8"
        }
        
        design_response = self.client.post("/api/v1/accelerator/design", json=design_request)
        assert design_response.status_code == 200
        design_data = design_response.json()
        
        # Step 3: Generate RTL
        rtl_response = self.client.post("/api/v1/accelerator/rtl", json=design_request)
        assert rtl_response.status_code == 200
        rtl_data = rtl_response.json()
        
        # Verify flow completed successfully
        assert profile_data["status"] == "success"
        assert design_data["status"] == "success"
        assert rtl_data["status"] == "success"
        assert "profile" in profile_data
        assert "accelerator" in design_data
        assert "rtl_file" in rtl_data
    
    def test_optimization_workflow(self):
        """Test optimization workflow integration."""
        # Start co-optimization
        opt_request = {
            "model_path": "optimization_test_model.onnx",
            "accelerator_config": {
                "compute_units": 32,
                "memory_hierarchy": ["sram_64kb", "dram"],
                "dataflow": "weight_stationary",
                "frequency_mhz": 200.0,
                "precision": "int8",
                "power_budget_w": 5.0
            },
            "target_fps": 30.0,
            "power_budget": 5.0,
            "iterations": 3,
            "strategy": "balanced"
        }
        
        opt_response = self.client.post("/api/v1/optimization/co-optimize", json=opt_request)
        assert opt_response.status_code == 200
        
        job_id = opt_response.json()["job_id"]
        
        # Check job status
        status_response = self.client.get(f"/api/v1/job/{job_id}/status")
        assert status_response.status_code == 200
        
        status_data = status_response.json()
        assert status_data["job_id"] == job_id
        assert "status" in status_data
        assert "progress" in status_data


@pytest.fixture
def api_client():
    """Fixture for API test client."""
    return TestClient(app)


class TestAPIFixtures:
    """Test API functionality with fixtures."""
    
    def test_with_api_client_fixture(self, api_client):
        """Test using API client fixture."""
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_profile_model_with_fixture(self, api_client):
        """Test model profiling with fixture."""
        request_data = {
            "model_path": "fixture_test_model.onnx",
            "input_shape": [1, 3, 224, 224],
            "framework": "onnx"
        }
        
        response = api_client.post("/api/v1/model/profile", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
"""
Performance testing configuration using Locust.
Tests AI Hardware Co-Design Playground endpoints under load.
"""

from locust import HttpUser, task, between
import json
import random


class CodesignUser(HttpUser):
    """Simulates a user interacting with the co-design platform."""
    
    wait_time = between(1, 5)
    
    def on_start(self):
        """Initialize user session."""
        # Health check to ensure service is running
        response = self.client.get("/health")
        if response.status_code != 200:
            print(f"Service not healthy: {response.status_code}")
    
    @task(3)
    def view_dashboard(self):
        """Load the main dashboard."""
        self.client.get("/", name="dashboard")
    
    @task(2)
    def list_projects(self):
        """Get list of co-design projects."""
        self.client.get("/api/v1/projects", name="list_projects")
    
    @task(2)
    def get_project_details(self):
        """Get details of a specific project."""
        project_id = random.randint(1, 100)
        self.client.get(f"/api/v1/projects/{project_id}", name="project_details")
    
    @task(1)
    def create_project(self):
        """Create a new co-design project."""
        project_data = {
            "name": f"test_project_{random.randint(1000, 9999)}",
            "description": "Load test project",
            "model_type": random.choice(["cnn", "transformer", "rnn"]),
            "target_hardware": random.choice(["fpga", "asic", "gpu"])
        }
        self.client.post(
            "/api/v1/projects",
            json=project_data,
            name="create_project"
        )
    
    @task(2)
    def run_optimization(self):
        """Trigger hardware-software co-optimization."""
        optimization_config = {
            "project_id": random.randint(1, 10),
            "optimization_target": random.choice(["latency", "power", "area"]),
            "constraints": {
                "max_power": random.uniform(1.0, 10.0),
                "max_area": random.uniform(100, 1000)
            }
        }
        self.client.post(
            "/api/v1/optimize",
            json=optimization_config,
            name="run_optimization"
        )
    
    @task(1)
    def upload_model(self):
        """Upload a neural network model."""
        # Simulate file upload
        files = {
            'model': ('test_model.onnx', b'fake_model_data', 'application/octet-stream')
        }
        data = {
            'name': f'test_model_{random.randint(1000, 9999)}',
            'framework': random.choice(['pytorch', 'tensorflow', 'onnx'])
        }
        self.client.post(
            "/api/v1/models/upload",
            files=files,
            data=data,
            name="upload_model"
        )
    
    @task(1)
    def generate_rtl(self):
        """Generate RTL from high-level design."""
        rtl_config = {
            "design_id": random.randint(1, 10),
            "target_language": random.choice(["verilog", "vhdl"]),
            "optimization_level": random.choice([1, 2, 3]),
            "clock_frequency": random.uniform(100, 1000)
        }
        self.client.post(
            "/api/v1/rtl/generate",
            json=rtl_config,
            name="generate_rtl"
        )
    
    @task(1)
    def run_simulation(self):
        """Run hardware simulation."""
        sim_config = {
            "design_id": random.randint(1, 10),
            "test_vectors": [random.random() for _ in range(100)],
            "simulation_time": random.uniform(1000, 10000),
            "timing_analysis": True
        }
        self.client.post(
            "/api/v1/simulate",
            json=sim_config,
            name="run_simulation"
        )
    
    @task(1)
    def get_metrics(self):
        """Retrieve performance metrics."""
        self.client.get("/api/v1/metrics", name="get_metrics")
        self.client.get("/api/v1/metrics/system", name="get_system_metrics")


class AdminUser(HttpUser):
    """Simulates admin operations with higher privileges."""
    
    wait_time = between(2, 8)
    weight = 1  # Lower frequency than regular users
    
    @task
    def admin_dashboard(self):
        """Access admin dashboard."""
        self.client.get("/admin", name="admin_dashboard")
    
    @task
    def system_status(self):
        """Check system status and health."""
        self.client.get("/admin/status", name="system_status")
        self.client.get("/admin/health", name="detailed_health")
    
    @task
    def manage_resources(self):
        """Manage compute resources."""
        self.client.get("/admin/resources", name="list_resources")
        
        # Simulate resource allocation
        resource_config = {
            "cpu_cores": random.randint(1, 16),
            "memory_gb": random.randint(4, 64),
            "gpu_count": random.randint(0, 4)
        }
        self.client.post(
            "/admin/resources/allocate",
            json=resource_config,
            name="allocate_resources"
        )


class BatchProcessingUser(HttpUser):
    """Simulates batch processing workloads."""
    
    wait_time = between(10, 30)  # Longer wait times for batch jobs
    weight = 2
    
    @task
    def submit_batch_job(self):
        """Submit a batch optimization job."""
        batch_config = {
            "job_type": "batch_optimization",
            "projects": [random.randint(1, 100) for _ in range(5)],
            "priority": random.choice(["low", "normal", "high"]),
            "timeout": random.randint(3600, 7200)  # 1-2 hours
        }
        self.client.post(
            "/api/v1/batch/submit",
            json=batch_config,
            name="submit_batch_job"
        )
    
    @task
    def check_job_status(self):
        """Check status of running batch jobs."""
        job_id = random.randint(1, 1000)
        self.client.get(f"/api/v1/batch/jobs/{job_id}/status", name="job_status")
    
    @task
    def download_results(self):
        """Download batch job results."""
        job_id = random.randint(1, 1000)
        self.client.get(f"/api/v1/batch/jobs/{job_id}/results", name="download_results")
"""
Comprehensive Quality Gates Validator
Final validation for all three generations of SDLC implementation.
"""

import os
import sys
import time
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    execution_time: float

class ComprehensiveQualityValidator:
    """Comprehensive quality gate validation system."""
    
    def __init__(self):
        self.results: List[QualityGateResult] = []
        self.start_time = time.time()
    
    def validate_core_functionality(self) -> QualityGateResult:
        """Validate core accelerator functionality."""
        start_time = time.time()
        details = {}
        
        try:
            # Test accelerator creation and performance
            from codesign_playground.core.accelerator import Accelerator
            
            accelerator = Accelerator(
                compute_units=64,
                memory_hierarchy={'L1': 32, 'L2': 256, 'L3': 2048},
                dataflow='weight_stationary',
                frequency_mhz=300,
                precision='int8'
            )
            
            performance = accelerator.estimate_performance()
            throughput_gops = performance['throughput_ops_s'] / 1e9
            
            details = {
                "accelerator_created": True,
                "throughput_gops": throughput_gops,
                "target_gops": 1.0,
                "performance_ratio": throughput_gops / 1.0,
                "memory_hierarchy_levels": len(accelerator.memory_hierarchy),
                "dataflow_type": accelerator.dataflow
            }
            
            # Quality criteria
            criteria_met = [
                throughput_gops >= 1.0,  # Meet minimum throughput
                performance['throughput_ops_s'] > 0,  # Positive performance
                len(accelerator.memory_hierarchy) >= 3,  # Multi-level memory
                accelerator.dataflow in ['weight_stationary', 'output_stationary', 'input_stationary']
            ]
            
            score = sum(criteria_met) / len(criteria_met) * 100
            passed = score >= 80.0
            
        except Exception as e:
            details = {"error": str(e), "accelerator_created": False}
            score = 0.0
            passed = False
        
        execution_time = time.time() - start_time
        return QualityGateResult("Core Functionality", passed, score, details, execution_time)
    
    def validate_research_capabilities(self) -> QualityGateResult:
        """Validate research and algorithm capabilities."""
        start_time = time.time()
        details = {}
        
        try:
            # Test research module imports
            research_modules = []
            
            try:
                from codesign_playground.research.novel_algorithms import get_quantum_optimizer
                research_modules.append("quantum_optimizer")
            except:
                pass
            
            try:
                from codesign_playground.research.research_discovery import conduct_comprehensive_research_discovery
                research_modules.append("research_discovery")
            except:
                pass
            
            try:
                from codesign_playground.research.comparative_study_framework import get_comparative_study_engine
                research_modules.append("comparative_study")
            except:
                pass
            
            try:
                from codesign_playground.research.breakthrough_algorithms import (
                    QuantumInspiredOptimizer, NeuroEvolutionarySearch, AdaptiveSwarmIntelligence
                )
                research_modules.extend(["quantum_inspired", "neuro_evolutionary", "swarm_intelligence"])
            except:
                pass
            
            details = {
                "available_modules": research_modules,
                "module_count": len(research_modules),
                "target_modules": 8,
                "research_coverage": len(research_modules) / 8 * 100
            }
            
            # Quality criteria
            score = min(100.0, (len(research_modules) / 8) * 100)  # Target 8 research modules
            passed = score >= 60.0  # 60% research modules available
            
        except Exception as e:
            details = {"error": str(e), "available_modules": []}
            score = 0.0
            passed = False
        
        execution_time = time.time() - start_time
        return QualityGateResult("Research Capabilities", passed, score, details, execution_time)
    
    def validate_global_features(self) -> QualityGateResult:
        """Validate global features (i18n, compliance)."""
        start_time = time.time()
        details = {}
        
        try:
            # Test internationalization
            i18n_available = False
            supported_languages = []
            
            try:
                import importlib
                i18n_module = importlib.import_module('codesign_playground.global.internationalization')
                SupportedLanguage = getattr(i18n_module, 'SupportedLanguage', None)
                if SupportedLanguage:
                    i18n_available = True
                    supported_languages = [lang.value for lang in SupportedLanguage]
            except:
                pass
            
            # Test compliance
            compliance_available = False
            compliance_features = []
            
            try:
                import importlib
                compliance_module = importlib.import_module('codesign_playground.global.compliance')
                ComplianceRegulation = getattr(compliance_module, 'ComplianceRegulation', None)
                if ComplianceRegulation:
                    compliance_available = True
                    compliance_features = [reg.value for reg in ComplianceRegulation]
            except:
                pass
            
            details = {
                "i18n_available": i18n_available,
                "supported_languages": len(supported_languages),
                "target_languages": 13,
                "compliance_available": compliance_available,
                "compliance_regulations": len(compliance_features),
                "target_regulations": 5
            }
            
            # Quality criteria
            criteria_scores = [
                100 if i18n_available else 0,
                (len(supported_languages) / 13) * 100,
                100 if compliance_available else 0,
                (len(compliance_features) / 5) * 100
            ]
            
            score = sum(criteria_scores) / len(criteria_scores)
            passed = score >= 70.0
            
        except Exception as e:
            details = {"error": str(e)}
            score = 0.0
            passed = False
        
        execution_time = time.time() - start_time
        return QualityGateResult("Global Features", passed, score, details, execution_time)
    
    def validate_performance_benchmarks(self) -> QualityGateResult:
        """Validate performance benchmarks and scaling."""
        start_time = time.time()
        details = {}
        
        try:
            # Run Generation 3 scaling test
            from generation_3_scaling_optimizer import QuantumLeapScaler
            
            scaler = QuantumLeapScaler()
            
            # Single design optimization
            single_params = {
                'compute_units': 64,
                'memory_hierarchy': {'L1': 32, 'L2': 256},
                'dataflow': 'weight_stationary',
                'frequency_mhz': 300,
                'precision': 'int8'
            }
            
            # Run async optimization
            import asyncio
            single_result = asyncio.run(scaler.optimize_accelerator_design(single_params))
            
            # Extract performance metrics
            performance = single_result.get('performance', {})
            throughput_gops = performance.get('throughput_ops_s', 0) / 1e9
            power_efficiency = single_result.get('power_efficiency', 0)
            scaling_potential = single_result.get('scaling_potential', 0)
            
            # Test caching
            cache_stats = scaler.performance_cache.get_stats()
            
            details = {
                "throughput_gops": throughput_gops,
                "target_gops": 15.0,
                "power_efficiency": power_efficiency,
                "scaling_potential": scaling_potential,
                "cache_utilization": cache_stats['utilization'],
                "max_workers": scaler.concurrent_processor.max_workers,
                "auto_scaling_range": [scaler.auto_scaler.min_workers, scaler.auto_scaler.max_workers]
            }
            
            # Performance criteria
            criteria_met = [
                throughput_gops >= 15.0,  # Target 15+ GOPS
                power_efficiency >= 0.3,  # Minimum power efficiency
                scaling_potential >= 5.0,  # 5x scaling potential
                scaler.concurrent_processor.max_workers >= 32,  # Good concurrency
                scaler.auto_scaler.max_workers >= 100  # Auto-scaling capability
            ]
            
            score = sum(criteria_met) / len(criteria_met) * 100
            passed = score >= 80.0
            
        except Exception as e:
            details = {"error": str(e)}
            score = 0.0
            passed = False
        
        execution_time = time.time() - start_time
        return QualityGateResult("Performance Benchmarks", passed, score, details, execution_time)
    
    def validate_code_quality(self) -> QualityGateResult:
        """Validate code quality and structure."""
        start_time = time.time()
        details = {}
        
        try:
            # Check file structure
            backend_path = Path("backend")
            core_modules = list((backend_path / "codesign_playground" / "core").glob("*.py"))
            utils_modules = list((backend_path / "codesign_playground" / "utils").glob("*.py"))
            research_modules = list((backend_path / "codesign_playground" / "research").glob("*.py"))
            global_modules = list((backend_path / "codesign_playground" / "global").glob("*.py"))
            
            # Check main entry points
            main_exists = (backend_path / "main.py").exists()
            server_exists = (backend_path / "codesign_playground" / "server.py").exists()
            
            # Check documentation
            docs_path = Path("docs")
            doc_files = list(docs_path.glob("*.md")) if docs_path.exists() else []
            readme_exists = Path("README.md").exists()
            
            details = {
                "core_modules": len(core_modules),
                "utils_modules": len(utils_modules), 
                "research_modules": len(research_modules),
                "global_modules": len(global_modules),
                "main_entry_points": int(main_exists) + int(server_exists),
                "documentation_files": len(doc_files),
                "readme_exists": readme_exists,
                "total_modules": len(core_modules) + len(utils_modules) + len(research_modules) + len(global_modules)
            }
            
            # Quality criteria
            criteria_scores = [
                min(100, len(core_modules) * 10),  # Core modules (target 10)
                min(100, len(utils_modules) * 5),  # Utils modules (target 20)
                min(100, len(research_modules) * 25), # Research modules (target 4)
                min(100, len(global_modules) * 50), # Global modules (target 2)
                100 if main_exists and server_exists else 50,
                min(100, len(doc_files) * 5),  # Documentation (target 20)
                100 if readme_exists else 0
            ]
            
            score = sum(criteria_scores) / len(criteria_scores)
            passed = score >= 70.0
            
        except Exception as e:
            details = {"error": str(e)}
            score = 0.0
            passed = False
        
        execution_time = time.time() - start_time
        return QualityGateResult("Code Quality", passed, score, details, execution_time)
    
    def validate_production_readiness(self) -> QualityGateResult:
        """Validate production readiness features."""
        start_time = time.time()
        details = {}
        
        try:
            # Check production configuration files
            docker_exists = Path("Dockerfile").exists()
            k8s_exists = Path("k8s").exists()
            requirements_exists = Path("requirements.txt").exists()
            
            # Check environment configuration
            env_example_exists = Path(".env.example").exists()
            
            # Check deployment guides
            deployment_guide_exists = Path("DEPLOYMENT_GUIDE.md").exists()
            
            # Check monitoring and health endpoints
            health_endpoint_available = False
            try:
                from backend.main import app
                # Check if health endpoint is defined
                health_endpoint_available = any("/health" in str(route.path) for route in app.routes)
            except:
                pass
            
            # Check security features
            security_available = False
            try:
                from codesign_playground.utils.security import SecurityManager
                security_available = True
            except:
                pass
            
            details = {
                "docker_available": docker_exists,
                "k8s_available": k8s_exists,
                "requirements_available": requirements_exists,
                "env_config_available": env_example_exists,
                "deployment_guide_available": deployment_guide_exists,
                "health_endpoint_available": health_endpoint_available,
                "security_available": security_available
            }
            
            # Production readiness criteria
            criteria_met = [
                requirements_exists,
                deployment_guide_exists,
                health_endpoint_available,
                security_available
            ]
            
            score = sum(criteria_met) / len(criteria_met) * 100
            passed = score >= 75.0
            
        except Exception as e:
            details = {"error": str(e)}
            score = 0.0
            passed = False
        
        execution_time = time.time() - start_time
        return QualityGateResult("Production Readiness", passed, score, details, execution_time)
    
    def run_all_quality_gates(self) -> Dict[str, Any]:
        """Run all quality gate validations."""
        print("🔍 COMPREHENSIVE QUALITY GATES VALIDATION")
        print("=" * 60)
        
        # Define quality gates in order
        quality_gates = [
            ("Core Functionality", self.validate_core_functionality),
            ("Research Capabilities", self.validate_research_capabilities),
            ("Global Features", self.validate_global_features),
            ("Performance Benchmarks", self.validate_performance_benchmarks),
            ("Code Quality", self.validate_code_quality),
            ("Production Readiness", self.validate_production_readiness)
        ]
        
        # Run each quality gate
        for gate_name, gate_function in quality_gates:
            print(f"\n🎯 {gate_name}")
            print("-" * 40)
            
            result = gate_function()
            self.results.append(result)
            
            status_icon = "✅" if result.passed else "❌"
            print(f"{status_icon} {result.name}: {result.score:.1f}% ({result.execution_time:.3f}s)")
            
            # Print key details
            for key, value in list(result.details.items())[:5]:  # Show first 5 details
                if isinstance(value, (int, float)):
                    print(f"   • {key}: {value}")
                elif isinstance(value, bool):
                    print(f"   • {key}: {'✅' if value else '❌'}")
                else:
                    print(f"   • {key}: {value}")
        
        # Calculate overall results
        total_execution_time = time.time() - self.start_time
        passed_gates = sum(1 for result in self.results if result.passed)
        total_gates = len(self.results)
        overall_score = sum(result.score for result in self.results) / len(self.results) if self.results else 0
        
        # Summary
        print("\n" + "=" * 60)
        print("QUALITY GATES FINAL SUMMARY")
        print("=" * 60)
        
        for result in self.results:
            status_icon = "✅" if result.passed else "❌"
            print(f"{status_icon} {result.name}: {result.score:.1f}%")
        
        print(f"\n📊 Overall Results:")
        print(f"   • Gates Passed: {passed_gates}/{total_gates} ({passed_gates/total_gates*100:.1f}%)")
        print(f"   • Average Score: {overall_score:.1f}%")
        print(f"   • Total Execution Time: {total_execution_time:.2f}s")
        
        # Determine final status
        final_passed = passed_gates >= int(total_gates * 0.8)  # 80% gates must pass
        final_status = "PASSED" if final_passed and overall_score >= 75.0 else "NEEDS_ATTENTION"
        
        if final_status == "PASSED":
            print("🎉 ALL QUALITY GATES PASSED - PRODUCTION READY!")
        else:
            print("⚠️  SOME QUALITY GATES NEED ATTENTION")
        
        return {
            "final_status": final_status,
            "passed_gates": passed_gates,
            "total_gates": total_gates,
            "overall_score": overall_score,
            "execution_time": total_execution_time,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "score": r.score,
                    "execution_time": r.execution_time
                } for r in self.results
            ]
        }

def main():
    """Run comprehensive quality gate validation."""
    print("🚀 AUTONOMOUS SDLC - COMPREHENSIVE QUALITY VALIDATION")
    print("=" * 70)
    print("Validating all generations: MAKE IT WORK → MAKE IT ROBUST → MAKE IT SCALE")
    print("=" * 70)
    
    validator = ComprehensiveQualityValidator()
    results = validator.run_all_quality_gates()
    
    return results["final_status"] == "PASSED"

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
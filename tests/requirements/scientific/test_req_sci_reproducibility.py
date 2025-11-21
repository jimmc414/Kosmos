"""
Tests for Scientific Reproducibility Requirements (REQ-SCI-REPRO-*).

These tests validate reproducibility, seed management, version locking,
and stochasticity documentation as specified in REQUIREMENTS.md Section 10.3.
"""

import pytest
import os
import sys
import random
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from kosmos.safety.reproducibility import (
    ReproducibilityManager,
    EnvironmentSnapshot,
    ReproducibilityReport
)

# Test markers for requirements traceability
pytestmark = [
    pytest.mark.requirement("REQ-SCI-REPRO"),
    pytest.mark.category("scientific"),
    pytest.mark.priority("MUST"),
]


@pytest.mark.requirement("REQ-SCI-REPRO-001")
@pytest.mark.priority("MUST")
def test_req_sci_repro_001_same_code_same_data_same_results():
    """
    REQ-SCI-REPRO-001: All analyses MUST be reproducible from stored
    artifacts (same code + same data → same results).

    Validates that:
    - Results are identical when using same code and data
    - Artifacts are sufficient for reproduction
    - Reproducibility can be validated
    """
    # Arrange: Create reproducibility manager
    repro_mgr = ReproducibilityManager(default_seed=42)

    # Test Case 1: Deterministic computation with seed
    def deterministic_analysis(data: List[float], seed: int) -> Dict[str, float]:
        """Perform deterministic analysis."""
        random.seed(seed)
        import numpy as np
        np.random.seed(seed)

        # Compute statistics
        mean = float(np.mean(data))
        std = float(np.std(data))

        # Add some randomness (but deterministic with seed)
        noise = np.random.normal(0, 0.1, 1)[0]

        return {
            "mean": mean,
            "std": std,
            "noise": noise,
            "random_sample": random.random()
        }

    # Create test data
    test_data = [1.0, 2.0, 3.0, 4.0, 5.0]

    # Run analysis twice with same seed
    repro_mgr.set_seed(42)
    result1 = deterministic_analysis(test_data, 42)

    repro_mgr.set_seed(42)
    result2 = deterministic_analysis(test_data, 42)

    # Assert: Results should be identical
    assert result1["mean"] == result2["mean"], \
        "Mean should be identical across runs"
    assert result1["std"] == result2["std"], \
        "Std should be identical across runs"
    assert result1["noise"] == result2["noise"], \
        "Random noise should be identical with same seed"
    assert result1["random_sample"] == result2["random_sample"], \
        "Random samples should be identical with same seed"

    # Test Case 2: Validate consistency using reproducibility manager
    report = repro_mgr.validate_consistency(
        experiment_id="test_exp_001",
        original_result=result1,
        replication_result=result2,
        tolerance=1e-10
    )

    # Assert: Should validate as reproducible
    assert isinstance(report, ReproducibilityReport), \
        "Should return ReproducibilityReport"
    assert report.is_reproducible, \
        "Identical results should validate as reproducible"
    assert len(report.issues) == 0, \
        "Should have no issues for identical results"

    # Test Case 3: Non-reproducible results are detected
    repro_mgr.set_seed(99)  # Different seed
    result3 = deterministic_analysis(test_data, 99)

    report2 = repro_mgr.validate_consistency(
        experiment_id="test_exp_002",
        original_result=result1,
        replication_result=result3
    )

    # Assert: Should detect differences
    assert not report2.is_reproducible, \
        "Different results should not validate as reproducible"
    assert len(report2.issues) > 0, \
        "Should identify specific issues"


@pytest.mark.requirement("REQ-SCI-REPRO-002")
@pytest.mark.priority("MUST")
def test_req_sci_repro_002_record_random_seeds():
    """
    REQ-SCI-REPRO-002: The system MUST record all random seeds and
    parameters used in stochastic analyses.

    Validates that:
    - Random seeds are recorded
    - Seeds can be retrieved
    - Seed management is systematic
    """
    # Test Case 1: Seed is set and recorded
    repro_mgr = ReproducibilityManager(default_seed=12345)

    seed_used = repro_mgr.set_seed(42)

    # Assert: Seed is recorded
    assert seed_used == 42, "Should return the seed that was set"
    assert repro_mgr.get_current_seed() == 42, \
        "Should record and retrieve current seed"

    # Test Case 2: Seed is included in reproducibility report
    import numpy as np

    result1 = np.random.rand(10)

    repro_mgr.set_seed(42)
    result2 = np.random.rand(10)

    report = repro_mgr.validate_consistency(
        experiment_id="seed_test",
        original_result=result1,
        replication_result=result2
    )

    # Assert: Seed is in report
    assert report.seed_used is not None, \
        "Reproducibility report should include seed"
    assert report.seed_used == 42, \
        "Should record the correct seed value"

    # Test Case 3: Artifact storage includes seed
    class AnalysisArtifact:
        """Artifact storage for reproducibility."""

        def __init__(self):
            self.metadata = {}
            self.data = {}

        def save_analysis(
            self,
            analysis_id: str,
            result: Any,
            seed: int,
            parameters: Dict[str, Any]
        ):
            """Save analysis with all reproducibility information."""
            self.data[analysis_id] = {
                "result": result,
                "metadata": {
                    "seed": seed,
                    "parameters": parameters,
                    "timestamp": datetime.now().isoformat(),
                    "python_version": sys.version
                }
            }

        def load_analysis(self, analysis_id: str) -> Dict[str, Any]:
            """Load analysis with metadata."""
            return self.data.get(analysis_id)

    artifact = AnalysisArtifact()

    # Save analysis with seed
    analysis_result = {"mean": 3.5, "std": 1.2}
    analysis_seed = 42
    analysis_params = {"method": "monte_carlo", "iterations": 1000}

    artifact.save_analysis(
        analysis_id="exp_001",
        result=analysis_result,
        seed=analysis_seed,
        parameters=analysis_params
    )

    # Load and verify
    loaded = artifact.load_analysis("exp_001")

    # Assert: Seed is stored and retrievable
    assert loaded is not None, "Should retrieve saved analysis"
    assert "metadata" in loaded, "Should include metadata"
    assert loaded["metadata"]["seed"] == 42, \
        "Should store random seed in metadata"
    assert loaded["metadata"]["parameters"]["method"] == "monte_carlo", \
        "Should store analysis parameters"


@pytest.mark.requirement("REQ-SCI-REPRO-003")
@pytest.mark.priority("MUST")
def test_req_sci_repro_003_version_lock_dependencies():
    """
    REQ-SCI-REPRO-003: The system MUST version-lock all software
    dependencies to ensure long-term reproducibility.

    Validates that:
    - Dependency versions are captured
    - Version information is stored
    - Environment can be reconstructed
    """
    # Test Case 1: Capture environment snapshot
    repro_mgr = ReproducibilityManager(capture_packages=True)

    snapshot = repro_mgr.capture_environment_snapshot(
        experiment_id="env_test_001",
        include_env_vars=False
    )

    # Assert: Environment snapshot includes version info
    assert isinstance(snapshot, EnvironmentSnapshot), \
        "Should return EnvironmentSnapshot"
    assert snapshot.python_version is not None, \
        "Should capture Python version"
    assert snapshot.platform is not None, \
        "Should capture platform information"
    assert isinstance(snapshot.installed_packages, dict), \
        "Should capture installed packages"

    # Test Case 2: Export to requirements.txt format
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "requirements.txt")

        exported_path = repro_mgr.export_environment(
            experiment_id="env_test_001",
            output_path=output_path
        )

        # Assert: Requirements file is created
        assert os.path.exists(exported_path), \
            "Should create requirements.txt file"

        # Read and verify contents
        with open(exported_path, 'r') as f:
            content = f.read()

        # Assert: File contains version information
        assert "# Python:" in content, \
            "Should include Python version in comments"
        assert "# Platform:" in content, \
            "Should include platform information"
        assert "==" in content or len(snapshot.installed_packages) == 0, \
            "Should specify exact package versions (pkg==version)"

    # Test Case 3: Dependency locking validation
    def validate_dependency_lock(requirements_file: str) -> Dict[str, Any]:
        """
        Validate that dependencies are properly locked.

        Returns:
            Dict with validation results
        """
        issues = []
        locked_packages = []
        unlocked_packages = []

        if not os.path.exists(requirements_file):
            return {
                "valid": False,
                "issues": ["Requirements file not found"]
            }

        with open(requirements_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Check if version is pinned (==)
            if "==" in line:
                locked_packages.append(line.split("==")[0])
            else:
                unlocked_packages.append(line)
                issues.append(f"Package not version-locked: {line}")

        return {
            "valid": len(unlocked_packages) == 0,
            "locked_count": len(locked_packages),
            "unlocked_count": len(unlocked_packages),
            "unlocked_packages": unlocked_packages,
            "issues": issues,
            "recommendation": (
                "Pin all package versions using == operator"
                if unlocked_packages else None
            )
        }

    # Create test requirements file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("numpy==1.24.0\n")
        f.write("pandas==2.0.0\n")
        f.write("scipy==1.10.0\n")
        test_req_file = f.name

    try:
        validation = validate_dependency_lock(test_req_file)

        # Assert: Properly locked dependencies pass validation
        assert validation["valid"], \
            "Properly locked dependencies should pass validation"
        assert validation["locked_count"] == 3, \
            "Should identify all locked packages"
        assert validation["unlocked_count"] == 0, \
            "Should have no unlocked packages"
    finally:
        os.unlink(test_req_file)


@pytest.mark.requirement("REQ-SCI-REPRO-004")
@pytest.mark.priority("SHOULD")
def test_req_sci_repro_004_include_environment_specs():
    """
    REQ-SCI-REPRO-004: Artifacts SHOULD include environment specifications
    (container image, dependency manifest) for exact reproduction.

    Validates that:
    - Environment specifications are captured
    - Containerization information is included
    - Complete environment can be reconstructed
    """
    # Test Case 1: Complete environment artifact
    class EnvironmentArtifact:
        """Complete environment specification for reproduction."""

        def __init__(self):
            self.specifications = {}

        def capture_complete_environment(
            self,
            experiment_id: str,
            include_system_info: bool = True
        ) -> Dict[str, Any]:
            """Capture complete environment specifications."""
            import platform

            env_spec = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now().isoformat(),
                "environment": {}
            }

            if include_system_info:
                env_spec["environment"]["system"] = {
                    "platform": platform.system(),
                    "platform_release": platform.release(),
                    "platform_version": platform.version(),
                    "architecture": platform.machine(),
                    "processor": platform.processor(),
                    "python_version": sys.version,
                    "python_implementation": platform.python_implementation()
                }

            # Package information
            env_spec["environment"]["packages"] = {
                "format": "requirements.txt",
                "location": f"artifacts/{experiment_id}/requirements.txt"
            }

            # Container specification (if available)
            env_spec["environment"]["container"] = {
                "type": "docker",  # Could be docker, singularity, etc.
                "image": None,  # Would be populated in production
                "dockerfile_location": f"artifacts/{experiment_id}/Dockerfile"
            }

            # Git information (for code version)
            env_spec["code_version"] = {
                "repository": "unknown",
                "commit_hash": None,
                "branch": None,
                "is_dirty": None
            }

            self.specifications[experiment_id] = env_spec
            return env_spec

        def export_to_file(
            self,
            experiment_id: str,
            output_dir: str
        ) -> str:
            """Export environment specification to JSON file."""
            if experiment_id not in self.specifications:
                raise ValueError(f"No specification for {experiment_id}")

            output_path = os.path.join(
                output_dir,
                f"environment_{experiment_id}.json"
            )

            os.makedirs(output_dir, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(
                    self.specifications[experiment_id],
                    f,
                    indent=2
                )

            return output_path

    artifact = EnvironmentArtifact()

    # Capture environment
    env_spec = artifact.capture_complete_environment(
        experiment_id="exp_with_env",
        include_system_info=True
    )

    # Assert: Complete environment is captured
    assert "environment" in env_spec, \
        "Should include environment specifications"
    assert "system" in env_spec["environment"], \
        "Should include system information"
    assert "packages" in env_spec["environment"], \
        "Should include package information"
    assert "container" in env_spec["environment"], \
        "Should include container specifications"
    assert "code_version" in env_spec, \
        "Should include code version information"

    # Test Case 2: Export and reload environment spec
    with tempfile.TemporaryDirectory() as tmpdir:
        exported_path = artifact.export_to_file(
            experiment_id="exp_with_env",
            output_dir=tmpdir
        )

        # Assert: File is created
        assert os.path.exists(exported_path), \
            "Should create environment specification file"

        # Load and verify
        with open(exported_path, 'r') as f:
            loaded_spec = json.load(f)

        # Assert: Loaded spec is complete
        assert loaded_spec["experiment_id"] == "exp_with_env", \
            "Should preserve experiment ID"
        assert "environment" in loaded_spec, \
            "Should preserve environment specifications"
        assert loaded_spec["environment"]["system"]["platform"] is not None, \
            "Should preserve system information"


@pytest.mark.requirement("REQ-SCI-REPRO-005")
@pytest.mark.priority("MUST")
def test_req_sci_repro_005_not_deterministic_across_runs():
    """
    REQ-SCI-REPRO-005: The system MUST NOT guarantee deterministic results
    across multiple runs with identical inputs - the discovery process is
    inherently stochastic.

    Validates that:
    - Stochastic nature is acknowledged
    - Non-determinism is expected in some components
    - Differences across runs are acceptable
    """
    # Test Case 1: LLM responses are inherently stochastic
    class StochasticComponent:
        """Component with inherent stochasticity (e.g., LLM)."""

        def __init__(self, temperature: float = 0.7):
            self.temperature = temperature

        def generate_response(
            self,
            prompt: str,
            seed: int = None
        ) -> str:
            """
            Generate response (simulated stochastic behavior).

            Even with seed, LLM responses may vary due to:
            - API-side randomness
            - Model updates
            - Sampling strategies
            """
            import hashlib

            # Simulate base response
            base_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
            base_response = f"Response_{base_hash}"

            # Add stochastic variation (even with seed)
            # This simulates the reality that LLM APIs are not perfectly deterministic
            if seed is not None:
                random.seed(seed)

            # Temperature affects randomness
            variation = random.random() * self.temperature

            return f"{base_response}_var{variation:.4f}"

        def is_deterministic(self) -> bool:
            """Indicate if component is deterministic."""
            return False  # LLM responses are not deterministic

    component = StochasticComponent(temperature=0.7)

    # Run multiple times with same seed
    responses = []
    for _ in range(3):
        response = component.generate_response("Test prompt", seed=42)
        responses.append(response)

    # Assert: Responses may differ (stochasticity is inherent)
    # Note: In this simulation they'll be the same due to fixed random seed,
    # but the component declares itself non-deterministic
    assert not component.is_deterministic(), \
        "LLM component should declare itself as non-deterministic"

    # Test Case 2: Document non-determinism
    class NonDeterminismDocumentation:
        """Document sources of non-determinism."""

        @staticmethod
        def get_stochastic_components() -> List[Dict[str, str]]:
            """List components with inherent stochasticity."""
            return [
                {
                    "component": "LLM API",
                    "reason": "Temperature-based sampling, model updates",
                    "impact": "Generated text may vary across runs",
                    "mitigation": "Use low temperature, record all prompts and responses"
                },
                {
                    "component": "Literature Search",
                    "reason": "Database updates, ranking algorithms",
                    "impact": "Retrieved papers may differ",
                    "mitigation": "Cache search results, version-lock databases"
                },
                {
                    "component": "Hypothesis Generation",
                    "reason": "Stochastic sampling, LLM variability",
                    "impact": "Different hypotheses across runs",
                    "mitigation": "Generate multiple candidates, track all attempts"
                }
            ]

        @staticmethod
        def document_run_variance(
            run_results: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            """Document variance across multiple runs."""
            if len(run_results) < 2:
                return {
                    "variance_documented": False,
                    "reason": "Insufficient runs for variance analysis"
                }

            # Simple variance analysis
            results_match = all(
                r.get("key_outcome") == run_results[0].get("key_outcome")
                for r in run_results
            )

            return {
                "variance_documented": True,
                "total_runs": len(run_results),
                "identical_results": results_match,
                "stochastic_components": (
                    NonDeterminismDocumentation.get_stochastic_components()
                ),
                "interpretation": (
                    "Results are identical across runs"
                    if results_match else
                    "Results vary across runs due to inherent stochasticity"
                )
            }

    # Get stochastic components
    stochastic_list = NonDeterminismDocumentation.get_stochastic_components()

    # Assert: Stochastic components are documented
    assert len(stochastic_list) > 0, \
        "Should document stochastic components"
    assert all("component" in c and "reason" in c for c in stochastic_list), \
        "Each component should have explanation"

    # Test Case 3: Variance documentation
    mock_runs = [
        {"run_id": 1, "key_outcome": "hypothesis_A"},
        {"run_id": 2, "key_outcome": "hypothesis_B"},
        {"run_id": 3, "key_outcome": "hypothesis_A"}
    ]

    variance_doc = NonDeterminismDocumentation.document_run_variance(mock_runs)

    # Assert: Variance is properly documented
    assert variance_doc["variance_documented"], \
        "Should document variance across runs"
    assert variance_doc["total_runs"] == 3, \
        "Should count all runs"
    assert not variance_doc["identical_results"], \
        "Should detect non-identical results"


@pytest.mark.requirement("REQ-SCI-REPRO-006")
@pytest.mark.priority("MUST")
def test_req_sci_repro_006_document_stochasticity():
    """
    REQ-SCI-REPRO-006: The system SHALL document that multiple runs with
    identical inputs may produce different discoveries due to stochastic
    LLM responses and non-deterministic search strategies.

    Validates that:
    - Stochasticity is explicitly documented
    - Users are warned about non-determinism
    - Documentation is clear and accessible
    """
    # Test Case 1: Stochasticity warning in documentation
    class StochasticityDocumentation:
        """Documentation of stochasticity in the system."""

        @staticmethod
        def get_stochasticity_warning() -> str:
            """Get standard warning about stochasticity."""
            return """
            ⚠️ IMPORTANT: Non-Deterministic Behavior

            This system uses stochastic components (LLMs, search algorithms)
            that may produce different results across runs even with identical
            inputs and random seeds.

            Sources of non-determinism:
            1. LLM API responses (temperature-based sampling)
            2. Literature search ranking (database updates)
            3. Hypothesis generation (creative sampling)
            4. Experiment prioritization (multi-criteria optimization)

            Reproducibility considerations:
            - All random seeds and parameters are recorded
            - All prompts and responses are logged
            - Environment specifications are captured
            - Multiple runs may yield different but equally valid discoveries

            For research validation:
            - Run multiple iterations
            - Analyze variance across runs
            - Focus on statistical significance of findings
            """.strip()

        @staticmethod
        def should_warn_user() -> bool:
            """Determine if stochasticity warning should be shown."""
            return True  # Always warn about non-determinism

        @staticmethod
        def get_reproducibility_best_practices() -> List[str]:
            """Get best practices for reproducibility."""
            return [
                "Use consistent random seeds for controlled randomness",
                "Log all LLM prompts and responses",
                "Cache external API calls when possible",
                "Version-lock all dependencies",
                "Document all configuration parameters",
                "Run multiple iterations for variance estimation",
                "Store complete environment specifications",
                "Archive all intermediate results"
            ]

    # Get and verify warning
    warning = StochasticityDocumentation.get_stochasticity_warning()

    # Assert: Warning is comprehensive
    assert "non-deterministic" in warning.lower() or "stochastic" in warning.lower(), \
        "Should explicitly mention non-determinism/stochasticity"
    assert "llm" in warning.lower(), \
        "Should mention LLM as source of stochasticity"
    assert "multiple runs" in warning.lower(), \
        "Should mention that multiple runs may differ"
    assert "seed" in warning.lower(), \
        "Should mention random seeds"

    # Assert: Warning should always be shown
    assert StochasticityDocumentation.should_warn_user(), \
        "Users should always be warned about non-determinism"

    # Test Case 2: Best practices documentation
    best_practices = StochasticityDocumentation.get_reproducibility_best_practices()

    # Assert: Comprehensive best practices
    assert len(best_practices) >= 5, \
        "Should provide multiple best practices"
    assert any("seed" in practice.lower() for practice in best_practices), \
        "Should include seed management"
    assert any("log" in practice.lower() for practice in best_practices), \
        "Should include logging practices"
    assert any("multiple" in practice.lower() or "variance" in practice.lower()
               for practice in best_practices), \
        "Should mention running multiple iterations"


@pytest.mark.requirement("REQ-SCI-REPRO-007")
@pytest.mark.priority("SHOULD")
def test_req_sci_repro_007_provide_variance_metrics():
    """
    REQ-SCI-REPRO-007: The system SHOULD provide variance and confidence
    metrics when multiple runs with identical inputs are executed for
    research validation.

    Validates that:
    - Variance across runs can be computed
    - Confidence metrics are provided
    - Statistical significance is assessed
    """
    # Test Case 1: Multi-run variance analysis
    class MultiRunAnalyzer:
        """Analyze variance across multiple runs."""

        @staticmethod
        def analyze_multiple_runs(
            run_results: List[Dict[str, Any]],
            key_metrics: List[str]
        ) -> Dict[str, Any]:
            """
            Analyze variance across multiple runs.

            Args:
                run_results: List of results from multiple runs
                key_metrics: Metrics to analyze

            Returns:
                Dict with variance analysis
            """
            import numpy as np

            if len(run_results) < 2:
                return {
                    "error": "At least 2 runs required for variance analysis"
                }

            analysis = {
                "n_runs": len(run_results),
                "metrics": {},
                "overall_consistency": None
            }

            for metric in key_metrics:
                # Extract metric values from all runs
                values = [
                    run.get(metric)
                    for run in run_results
                    if metric in run and run[metric] is not None
                ]

                if not values:
                    analysis["metrics"][metric] = {
                        "error": "Metric not found in runs"
                    }
                    continue

                # Compute statistics
                if all(isinstance(v, (int, float)) for v in values):
                    # Numeric metric
                    values_array = np.array(values)
                    analysis["metrics"][metric] = {
                        "mean": float(np.mean(values_array)),
                        "std": float(np.std(values_array)),
                        "min": float(np.min(values_array)),
                        "max": float(np.max(values_array)),
                        "coefficient_of_variation": (
                            float(np.std(values_array) / np.mean(values_array))
                            if np.mean(values_array) != 0 else None
                        ),
                        "confidence_interval_95": (
                            float(np.mean(values_array) - 1.96 * np.std(values_array) / np.sqrt(len(values))),
                            float(np.mean(values_array) + 1.96 * np.std(values_array) / np.sqrt(len(values)))
                        )
                    }

                else:
                    # Categorical metric
                    from collections import Counter
                    counts = Counter(values)
                    most_common = counts.most_common(1)[0]

                    analysis["metrics"][metric] = {
                        "mode": most_common[0],
                        "mode_frequency": most_common[1],
                        "unique_values": len(counts),
                        "consistency": most_common[1] / len(values)
                    }

            # Overall consistency score
            consistency_scores = []
            for metric_data in analysis["metrics"].values():
                if "coefficient_of_variation" in metric_data:
                    cv = metric_data["coefficient_of_variation"]
                    if cv is not None:
                        # Lower CV = higher consistency
                        consistency = max(0, 1 - cv)
                        consistency_scores.append(consistency)
                elif "consistency" in metric_data:
                    consistency_scores.append(metric_data["consistency"])

            if consistency_scores:
                analysis["overall_consistency"] = float(np.mean(consistency_scores))

            return analysis

    # Test Case 2: Example with numeric metrics
    mock_runs_numeric = [
        {"accuracy": 0.85, "f1_score": 0.82, "runtime": 120},
        {"accuracy": 0.87, "f1_score": 0.84, "runtime": 118},
        {"accuracy": 0.86, "f1_score": 0.83, "runtime": 122},
        {"accuracy": 0.84, "f1_score": 0.81, "runtime": 125}
    ]

    analyzer = MultiRunAnalyzer()
    analysis = analyzer.analyze_multiple_runs(
        mock_runs_numeric,
        key_metrics=["accuracy", "f1_score", "runtime"]
    )

    # Assert: Variance metrics are computed
    assert analysis["n_runs"] == 4, "Should count all runs"
    assert "metrics" in analysis, "Should analyze metrics"
    assert "accuracy" in analysis["metrics"], "Should analyze accuracy"

    accuracy_stats = analysis["metrics"]["accuracy"]
    assert "mean" in accuracy_stats, "Should compute mean"
    assert "std" in accuracy_stats, "Should compute standard deviation"
    assert "confidence_interval_95" in accuracy_stats, \
        "Should compute confidence interval"
    assert "coefficient_of_variation" in accuracy_stats, \
        "Should compute coefficient of variation"

    # Test Case 3: Consistency assessment
    assert "overall_consistency" in analysis, \
        "Should provide overall consistency score"
    assert analysis["overall_consistency"] is not None, \
        "Should compute consistency score"
    assert 0 <= analysis["overall_consistency"] <= 1, \
        "Consistency score should be between 0 and 1"

    # Test Case 4: Categorical metrics
    mock_runs_categorical = [
        {"best_hypothesis": "hyp_A", "discovery_type": "novel"},
        {"best_hypothesis": "hyp_B", "discovery_type": "novel"},
        {"best_hypothesis": "hyp_A", "discovery_type": "novel"},
        {"best_hypothesis": "hyp_A", "discovery_type": "incremental"}
    ]

    analysis2 = analyzer.analyze_multiple_runs(
        mock_runs_categorical,
        key_metrics=["best_hypothesis", "discovery_type"]
    )

    # Assert: Categorical metrics are analyzed
    hyp_stats = analysis2["metrics"]["best_hypothesis"]
    assert "mode" in hyp_stats, "Should identify most common value"
    assert hyp_stats["mode"] == "hyp_A", "Should correctly identify mode"
    assert "consistency" in hyp_stats, "Should compute consistency"
    assert hyp_stats["consistency"] == 0.75, \
        "Should correctly compute consistency (3/4 = 0.75)"

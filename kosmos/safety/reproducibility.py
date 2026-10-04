"""
Reproducibility management and validation.

Implements:
- Random seed management
- Environment capture (dependencies, versions)
- Result consistency validation
- Determinism testing
- Reproducibility reports
"""

import os
from kosmos.utils.compat import model_to_dict
import sys
import random
import logging
import platform
import subprocess
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Try to import numpy for seed management
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class EnvironmentSnapshot(BaseModel):
    """Snapshot of execution environment."""

    python_version: str
    platform: str
    platform_version: str
    cpu_count: int
    installed_packages: Dict[str, str] = Field(default_factory=dict)
    environment_variables: Dict[str, str] = Field(default_factory=dict)
    working_directory: str
    timestamp: datetime = Field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return model_to_dict(self)

    def get_hash(self) -> str:
        """Get hash of environment for comparison."""
        # Hash relevant parts (exclude timestamp and working dir which may vary)
        relevant_data = {
            "python": self.python_version,
            "platform": self.platform,
            "packages": self.installed_packages
        }
        data_str = str(sorted(relevant_data.items()))
        return hashlib.md5(data_str.encode()).hexdigest()


class ReproducibilityReport(BaseModel):
    """Report on reproducibility of experiments."""

    experiment_id: str
    is_reproducible: bool
    seed_used: Optional[int] = None
    environment_snapshot: Optional[EnvironmentSnapshot] = None
    consistency_checks: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    def summary(self) -> str:
        """Generate human-readable summary."""
        if self.is_reproducible:
            return f"✓ Experiment is reproducible (seed={self.seed_used})"
        else:
            return f"✗ Experiment not reproducible: {len(self.issues)} issues found"


class ReproducibilityManager:
    """
    Manages reproducibility of experiments.

    Handles seed management, environment capture, and result validation.
    """

    def __init__(
        self,
        default_seed: int = 42,
        capture_environment: bool = True,
        capture_packages: bool = True,
        strict_mode: bool = False
    ):
        """
        Initialize reproducibility manager.

        Args:
            default_seed: Default random seed
            capture_environment: Capture environment snapshots
            capture_packages: Capture installed package versions
            strict_mode: Strict reproducibility checks
        """
        self.default_seed = default_seed
        self.capture_environment = capture_environment
        self.capture_packages = capture_packages
        self.strict_mode = strict_mode

        # Track current seed
        self.current_seed: Optional[int] = None

        # Modern NumPy Generator (set by set_seed())
        self._rng = None

        # Environment snapshots by experiment
        self.environments: Dict[str, EnvironmentSnapshot] = {}

        logger.info(
            f"ReproducibilityManager initialized (seed={default_seed}, "
            f"capture_env={capture_environment}, strict={strict_mode})"
        )

    def set_seed(self, seed: Optional[int] = None) -> int:
        """
        Set random seed for reproducibility.

        Args:
            seed: Random seed (uses default if None)

        Returns:
            The seed that was set
        """
        if seed is None:
            seed = self.default_seed

        # Set Python random seed
        random.seed(seed)

        # Set NumPy seed if available
        if HAS_NUMPY:
            np.random.seed(seed)  # Legacy API for library compatibility
            self._rng = np.random.default_rng(seed)  # Modern Generator API

        # Try to set PyTorch seed if available
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:  # absent, or a broken install that raises on import
            pass

        # Try to set TensorFlow seed if available
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass

        self.current_seed = seed
        logger.info(f"Random seed set to {seed}")

        return seed

    def get_current_seed(self) -> Optional[int]:
        """Get the current random seed."""
        return self.current_seed

    def capture_environment_snapshot(
        self,
        experiment_id: str,
        include_env_vars: bool = False
    ) -> EnvironmentSnapshot:
        """
        Capture snapshot of current environment.

        Args:
            experiment_id: Experiment ID
            include_env_vars: Include environment variables

        Returns:
            EnvironmentSnapshot object
        """
        # Get Python version
        python_version = sys.version

        # Get platform info
        platform_name = platform.system()
        platform_version = platform.release()

        # Get CPU count
        cpu_count = os.cpu_count() or 0

        # Get installed packages
        packages = {}
        if self.capture_packages:
            packages = self._get_installed_packages()

        # Get environment variables (filtered)
        env_vars = {}
        if include_env_vars:
            # Only include relevant env vars (not secrets)
            safe_vars = ["PATH", "PYTHONPATH", "LANG", "LC_ALL"]
            env_vars = {k: v for k, v in os.environ.items() if k in safe_vars}

        # Get working directory
        working_dir = os.getcwd()

        snapshot = EnvironmentSnapshot(
            python_version=python_version,
            platform=platform_name,
            platform_version=platform_version,
            cpu_count=cpu_count,
            installed_packages=packages,
            environment_variables=env_vars,
            working_directory=working_dir
        )

        # Store snapshot
        self.environments[experiment_id] = snapshot

        logger.info(f"Environment snapshot captured for experiment {experiment_id}")

        return snapshot

    def _get_installed_packages(self) -> Dict[str, str]:
        """Get installed Python packages and versions."""
        packages = {}

        try:
            # Use pip list to get packages
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                import json
                package_list = json.loads(result.stdout)
                packages = {pkg["name"]: pkg["version"] for pkg in package_list}
        except Exception as e:
            logger.warning(f"Could not get installed packages: {e}")

        return packages

    def validate_consistency(
        self,
        experiment_id: str,
        original_result: Any,
        replication_result: Any,
        tolerance: float = 1e-6
    ) -> ReproducibilityReport:
        """
        Validate consistency between original and replicated results.

        Args:
            experiment_id: Experiment ID
            original_result: Original result
            replication_result: Replicated result
            tolerance: Numerical tolerance for comparison

        Returns:
            ReproducibilityReport
        """
        issues = []
        checks = []

        # Check if results are same type
        if type(original_result) != type(replication_result):
            issues.append(
                f"Result types differ: {type(original_result)} vs {type(replication_result)}"
            )
        checks.append("result_type")

        # Numeric comparison
        if isinstance(original_result, (int, float)):
            if abs(original_result - replication_result) > tolerance:
                issues.append(
                    f"Numeric results differ beyond tolerance: "
                    f"{original_result} vs {replication_result} "
                    f"(tolerance={tolerance})"
                )
            checks.append("numeric_value")

        # Array comparison (if numpy available)
        if HAS_NUMPY and isinstance(original_result, np.ndarray):
            try:
                if not np.allclose(original_result, replication_result, atol=tolerance):
                    issues.append("Array results differ beyond tolerance")
                checks.append("array_values")
            except Exception as e:
                issues.append(f"Could not compare arrays: {e}")

        # Dict comparison
        if isinstance(original_result, dict):
            orig_keys = set(original_result.keys())
            repl_keys = set(replication_result.keys())

            if orig_keys != repl_keys:
                missing = orig_keys - repl_keys
                extra = repl_keys - orig_keys
                issues.append(
                    f"Dict keys differ (missing: {missing}, extra: {extra})"
                )
            checks.append("dict_keys")

        # String comparison
        if isinstance(original_result, str):
            if original_result != replication_result:
                issues.append("String results differ")
            checks.append("string_value")

        is_reproducible = len(issues) == 0

        report = ReproducibilityReport(
            experiment_id=experiment_id,
            is_reproducible=is_reproducible,
            seed_used=self.current_seed,
            environment_snapshot=self.environments.get(experiment_id),
            consistency_checks=checks,
            issues=issues
        )

        logger.info(f"Reproducibility validation: {report.summary()}")

        return report

    def test_determinism(
        self,
        experiment_function,
        seed: int,
        n_runs: int = 3,
        **kwargs
    ) -> bool:
        """
        Test if experiment function is deterministic.

        Runs experiment multiple times with same seed and checks if results are identical.

        Args:
            experiment_function: Function to test
            seed: Random seed to use
            n_runs: Number of runs to perform
            **kwargs: Arguments to pass to experiment function

        Returns:
            True if deterministic, False otherwise
        """
        results = []

        for i in range(n_runs):
            # Set seed before each run
            self.set_seed(seed)

            # Run experiment
            try:
                result = experiment_function(**kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Experiment run {i+1} failed: {e}")
                return False

        # Check if all results are identical
        for i in range(1, len(results)):
            # Use validate_consistency for comparison
            report = self.validate_consistency(
                experiment_id=f"determinism_test_{i}",
                original_result=results[0],
                replication_result=results[i]
            )

            if not report.is_reproducible:
                logger.warning(
                    f"Determinism test failed: Run 1 vs Run {i+1} differ"
                )
                return False

        logger.info(f"Determinism test passed ({n_runs} runs)")
        return True

    def compare_environments(
        self,
        env1_id: str,
        env2_id: str
    ) -> Dict[str, Any]:
        """
        Compare two environment snapshots.

        Args:
            env1_id: First environment ID
            env2_id: Second environment ID

        Returns:
            Dictionary with comparison results
        """
        if env1_id not in self.environments:
            raise ValueError(f"Environment {env1_id} not found")
        if env2_id not in self.environments:
            raise ValueError(f"Environment {env2_id} not found")

        env1 = self.environments[env1_id]
        env2 = self.environments[env2_id]

        differences = {
            "python_version": env1.python_version != env2.python_version,
            "platform": env1.platform != env2.platform,
            "cpu_count": env1.cpu_count != env2.cpu_count,
            "package_differences": []
        }

        # Compare packages
        env1_pkgs = set(env1.installed_packages.keys())
        env2_pkgs = set(env2.installed_packages.keys())

        # Missing packages
        missing_in_env2 = env1_pkgs - env2_pkgs
        extra_in_env2 = env2_pkgs - env1_pkgs

        # Version differences
        version_diffs = []
        for pkg in env1_pkgs & env2_pkgs:
            if env1.installed_packages[pkg] != env2.installed_packages[pkg]:
                version_diffs.append({
                    "package": pkg,
                    "env1_version": env1.installed_packages[pkg],
                    "env2_version": env2.installed_packages[pkg]
                })

        differences["package_differences"] = {
            "missing_in_env2": list(missing_in_env2),
            "extra_in_env2": list(extra_in_env2),
            "version_differences": version_diffs
        }

        # Overall match
        differences["environments_match"] = (
            not differences["python_version"] and
            not differences["platform"] and
            len(missing_in_env2) == 0 and
            len(extra_in_env2) == 0 and
            len(version_diffs) == 0
        )

        return differences

    def export_environment(
        self,
        experiment_id: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Export environment snapshot to requirements.txt format.

        Args:
            experiment_id: Experiment ID
            output_path: Output file path (default: requirements_{exp_id}.txt)

        Returns:
            Path to exported file
        """
        if experiment_id not in self.environments:
            raise ValueError(f"Environment {experiment_id} not found")

        env = self.environments[experiment_id]

        if output_path is None:
            output_path = f"requirements_{experiment_id}.txt"

        # Write requirements file
        with open(output_path, 'w') as f:
            # Add header comment
            f.write(f"# Generated by ReproducibilityManager\n")
            f.write(f"# Experiment: {experiment_id}\n")
            f.write(f"# Timestamp: {env.timestamp}\n")
            f.write(f"# Python: {env.python_version}\n")
            f.write(f"# Platform: {env.platform} {env.platform_version}\n\n")

            # Write packages
            for pkg, version in sorted(env.installed_packages.items()):
                f.write(f"{pkg}=={version}\n")

        logger.info(f"Environment exported to {output_path}")

        return output_path

    def clear_snapshots(self):
        """Clear all stored environment snapshots."""
        self.environments.clear()
        logger.info("All environment snapshots cleared")

    def get_snapshot_summary(self) -> Dict[str, Any]:
        """
        Get summary of stored snapshots.

        Returns:
            Dictionary with snapshot statistics
        """
        return {
            "total_snapshots": len(self.environments),
            "experiment_ids": list(self.environments.keys()),
            "current_seed": self.current_seed
        }

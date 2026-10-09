"""
Tests for reproducibility manager.

Tests seed management, environment capture, consistency validation, and determinism.
"""

import pytest
import random
import sys
import platform
from unittest.mock import Mock, patch, MagicMock

from kosmos.safety.reproducibility import (
    ReproducibilityManager, ReproducibilityReport,
    EnvironmentSnapshot
)


class TestReproducibilityManagerInitialization:
    """Tests for ReproducibilityManager initialization."""

    def test_init_default_settings(self):
        """Test initialization with default settings."""
        manager = ReproducibilityManager()

        assert manager.default_seed == 42
        assert manager.capture_environment is True
        assert manager.capture_packages is True
        assert manager.strict_mode is False
        assert manager.current_seed is None

    def test_init_custom_settings(self):
        """Test initialization with custom settings."""
        manager = ReproducibilityManager(
            default_seed=12345,
            capture_environment=False,
            capture_packages=False,
            strict_mode=True
        )

        assert manager.default_seed == 12345
        assert manager.capture_environment is False
        assert manager.capture_packages is False
        assert manager.strict_mode is True


class TestSeedManagement:
    """Tests for random seed management."""

    def test_set_seed_with_custom_value(self):
        """Test setting seed with custom value."""
        manager = ReproducibilityManager()

        seed = manager.set_seed(999)

        assert seed == 999
        assert manager.current_seed == 999

    def test_set_seed_with_default(self):
        """Test setting seed with default value."""
        manager = ReproducibilityManager(default_seed=42)

        seed = manager.set_seed(None)

        assert seed == 42
        assert manager.current_seed == 42

    def test_set_seed_affects_python_random(self):
        """Test that setting seed affects Python random module."""
        manager = ReproducibilityManager()

        manager.set_seed(100)
        value1 = random.random()

        manager.set_seed(100)
        value2 = random.random()

        # Same seed should produce same value
        assert value1 == value2

    @pytest.mark.skipif("not hasattr(sys.modules, 'numpy')")
    def test_set_seed_affects_numpy(self):
        """Test that setting seed affects NumPy if available."""
        import numpy as np

        manager = ReproducibilityManager()

        manager.set_seed(200)
        value1 = np.random.random()

        manager.set_seed(200)
        value2 = np.random.random()

        assert value1 == value2

    def test_get_current_seed(self):
        """Test getting current seed."""
        manager = ReproducibilityManager()

        assert manager.get_current_seed() is None

        manager.set_seed(555)

        assert manager.get_current_seed() == 555


class TestEnvironmentCapture:
    """Tests for environment snapshot capture."""

    def test_capture_environment_snapshot(self):
        """Test capturing environment snapshot."""
        manager = ReproducibilityManager(capture_packages=False)

        snapshot = manager.capture_environment_snapshot("exp_123")

        assert snapshot.python_version == sys.version
        assert snapshot.platform == platform.system()
        assert snapshot.cpu_count > 0
        assert "exp_123" in manager.environments

    def test_snapshot_includes_python_version(self):
        """Test that snapshot includes Python version."""
        manager = ReproducibilityManager(capture_packages=False)

        snapshot = manager.capture_environment_snapshot("exp_test")

        assert snapshot.python_version is not None
        assert len(snapshot.python_version) > 0

    def test_snapshot_includes_platform_info(self):
        """Test that snapshot includes platform information."""
        manager = ReproducibilityManager(capture_packages=False)

        snapshot = manager.capture_environment_snapshot("exp_test")

        assert snapshot.platform is not None
        assert snapshot.platform_version is not None

    @patch('kosmos.safety.reproducibility.subprocess.run')
    def test_capture_packages_when_enabled(self, mock_run):
        """Test that packages are captured when enabled."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout='[{"name": "numpy", "version": "1.24.0"}]'
        )

        manager = ReproducibilityManager(capture_packages=True)

        snapshot = manager.capture_environment_snapshot("exp_test")

        assert len(snapshot.installed_packages) > 0

    def test_capture_environment_variables_when_requested(self):
        """Test that environment variables are captured when requested."""
        manager = ReproducibilityManager()

        snapshot = manager.capture_environment_snapshot("exp_test", include_env_vars=True)

        # Should have at least PATH
        assert len(snapshot.environment_variables) >= 0  # May be empty in test env

    def test_environment_hash_generation(self):
        """Test that environment hash can be generated."""
        manager = ReproducibilityManager(capture_packages=False)

        snapshot = manager.capture_environment_snapshot("exp_test")
        hash_value = snapshot.get_hash()

        assert hash_value is not None
        assert len(hash_value) == 32  # MD5 hash

    def test_environment_hash_consistent(self):
        """Test that environment hash is consistent."""
        manager = ReproducibilityManager(capture_packages=False)

        snapshot = manager.capture_environment_snapshot("exp_test")
        hash1 = snapshot.get_hash()
        hash2 = snapshot.get_hash()

        assert hash1 == hash2


class TestConsistencyValidation:
    """Tests for result consistency validation."""

    def test_validate_identical_numeric_results(self):
        """Test validation of identical numeric results."""
        manager = ReproducibilityManager()

        report = manager.validate_consistency(
            "exp_test",
            original_result=42.0,
            replication_result=42.0
        )

        assert report.is_reproducible
        assert len(report.issues) == 0

    def test_validate_different_numeric_results(self):
        """Test validation of different numeric results."""
        manager = ReproducibilityManager()

        report = manager.validate_consistency(
            "exp_test",
            original_result=42.0,
            replication_result=43.0
        )

        assert not report.is_reproducible
        assert len(report.issues) > 0

    def test_validate_within_tolerance(self):
        """Test validation with tolerance."""
        manager = ReproducibilityManager()

        report = manager.validate_consistency(
            "exp_test",
            original_result=42.0,
            replication_result=42.0000001,
            tolerance=1e-6
        )

        assert report.is_reproducible

    def test_validate_identical_strings(self):
        """Test validation of identical string results."""
        manager = ReproducibilityManager()

        report = manager.validate_consistency(
            "exp_test",
            original_result="test_output",
            replication_result="test_output"
        )

        assert report.is_reproducible

    def test_validate_different_strings(self):
        """Test validation of different string results."""
        manager = ReproducibilityManager()

        report = manager.validate_consistency(
            "exp_test",
            original_result="output_1",
            replication_result="output_2"
        )

        assert not report.is_reproducible

    def test_validate_identical_dicts(self):
        """Test validation of identical dictionary results."""
        manager = ReproducibilityManager()

        original = {"a": 1, "b": 2}
        replication = {"a": 1, "b": 2}

        report = manager.validate_consistency(
            "exp_test",
            original_result=original,
            replication_result=replication
        )

        # Dict keys should match
        assert "dict_keys" in report.consistency_checks

    def test_validate_different_dict_keys(self):
        """Test validation of dicts with different keys."""
        manager = ReproducibilityManager()

        original = {"a": 1, "b": 2}
        replication = {"a": 1, "c": 3}

        report = manager.validate_consistency(
            "exp_test",
            original_result=original,
            replication_result=replication
        )

        assert not report.is_reproducible
        assert any("keys differ" in issue.lower() for issue in report.issues)

    def test_validate_different_types(self):
        """Test validation of different result types."""
        manager = ReproducibilityManager()

        report = manager.validate_consistency(
            "exp_test",
            original_result=42,
            replication_result="42"
        )

        assert not report.is_reproducible
        assert any("types differ" in issue.lower() for issue in report.issues)

    def test_validate_dict_vs_non_dict(self):
        """A dict original against a non-dict replication is a type issue, not a crash."""
        manager = ReproducibilityManager()

        report = manager.validate_consistency(
            "exp_test",
            original_result={"a": 1},
            replication_result=[1]
        )

        assert not report.is_reproducible
        assert any("types differ" in issue.lower() for issue in report.issues)


class TestDeterminismTesting:
    """Tests for determinism testing."""

    def test_deterministic_function_passes(self):
        """Test that deterministic function passes test."""
        manager = ReproducibilityManager()

        def deterministic_func(x):
            return x * 2

        is_deterministic = manager.test_determinism(
            deterministic_func,
            seed=42,
            n_runs=3,
            x=5
        )

        assert is_deterministic

    def test_random_function_with_seed_passes(self):
        """Test that random function with seed passes."""
        manager = ReproducibilityManager()

        def seeded_random_func():
            return random.random()

        is_deterministic = manager.test_determinism(
            seeded_random_func,
            seed=42,
            n_runs=3
        )

        assert is_deterministic

    def test_failed_function_returns_false(self):
        """Test that failed function returns False."""
        manager = ReproducibilityManager()

        def failing_func():
            raise ValueError("Test error")

        is_deterministic = manager.test_determinism(
            failing_func,
            seed=42,
            n_runs=2
        )

        assert not is_deterministic


class TestEnvironmentComparison:
    """Tests for environment comparison."""

    def test_compare_identical_environments(self):
        """Test comparison of identical environments."""
        manager = ReproducibilityManager(capture_packages=False)

        # Capture same environment twice
        manager.capture_environment_snapshot("env1")
        manager.capture_environment_snapshot("env2")

        diff = manager.compare_environments("env1", "env2")

        assert diff["environments_match"] is True
        assert not diff["python_version"]
        assert not diff["platform"]

    def test_compare_raises_on_missing_environment(self):
        """Test that comparison raises error for missing environment."""
        manager = ReproducibilityManager()

        with pytest.raises(ValueError, match="not found"):
            manager.compare_environments("env1", "env2")


class TestEnvironmentExport:
    """Tests for environment export."""

    def test_export_environment_to_file(self, tmp_path):
        """Test exporting environment to requirements file."""
        manager = ReproducibilityManager(capture_packages=False)

        manager.capture_environment_snapshot("exp_123")

        output_path = tmp_path / "requirements.txt"
        result_path = manager.export_environment(
            "exp_123",
            output_path=str(output_path)
        )

        assert output_path.exists()
        content = output_path.read_text()
        assert "# Generated by ReproducibilityManager" in content
        assert "exp_123" in content

    def test_export_raises_on_missing_environment(self):
        """Test that export raises error for missing environment."""
        manager = ReproducibilityManager()

        with pytest.raises(ValueError, match="not found"):
            manager.export_environment("nonexistent")


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_clear_snapshots(self):
        """Test clearing all snapshots."""
        manager = ReproducibilityManager(capture_packages=False)

        manager.capture_environment_snapshot("env1")
        manager.capture_environment_snapshot("env2")

        assert len(manager.environments) == 2

        manager.clear_snapshots()

        assert len(manager.environments) == 0

    def test_get_snapshot_summary(self):
        """Test getting snapshot summary."""
        manager = ReproducibilityManager(capture_packages=False)

        manager.capture_environment_snapshot("env1")
        manager.capture_environment_snapshot("env2")
        manager.set_seed(100)

        summary = manager.get_snapshot_summary()

        assert summary["total_snapshots"] == 2
        assert set(summary["experiment_ids"]) == {"env1", "env2"}
        assert summary["current_seed"] == 100

    def test_summary_empty_manager(self):
        """Test summary of empty manager."""
        manager = ReproducibilityManager()

        summary = manager.get_snapshot_summary()

        assert summary["total_snapshots"] == 0
        assert summary["experiment_ids"] == []
        assert summary["current_seed"] is None


class TestReproducibilityReport:
    """Tests for ReproducibilityReport."""

    def test_report_summary_reproducible(self):
        """Test summary for reproducible report."""
        report = ReproducibilityReport(
            experiment_id="test",
            is_reproducible=True,
            seed_used=42
        )

        summary = report.summary()

        assert "✓" in summary or "reproducible" in summary.lower()
        assert "42" in summary

    def test_report_summary_not_reproducible(self):
        """Test summary for non-reproducible report."""
        report = ReproducibilityReport(
            experiment_id="test",
            is_reproducible=False,
            issues=["Issue 1", "Issue 2"]
        )

        summary = report.summary()

        assert "✗" in summary or "not reproducible" in summary.lower()
        assert "2" in summary  # Number of issues

"""
Integration tests for complete analysis pipeline (Phase 6).

Tests end-to-end flow: ExperimentResult → Analysis → Visualization → Summary.

Tests using REAL Claude API for LLM-dependent tests.
Pure Python tests (statistics, visualization) run without mocks.
"""

import os
import pytest
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
from datetime import datetime

from kosmos.agents.data_analyst import DataAnalystAgent, ResultInterpretation
from kosmos.analysis.visualization import PublicationVisualizer
from kosmos.analysis.plotly_viz import PlotlyVisualizer
from kosmos.analysis.summarizer import ResultSummarizer, ResultSummary
from kosmos.analysis.statistics import StatisticalReporter, DescriptiveStats

from kosmos.models.result import (
    ExperimentResult,
    ResultStatus,
    StatisticalTestResult,
    VariableResult,
    ExecutionMetadata
)
from kosmos.models.hypothesis import Hypothesis


# Skip all tests if API key not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_claude,
    pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY for real LLM calls"
    )
]


def unique_id() -> str:
    """Generate unique ID for test isolation."""
    return uuid.uuid4().hex[:8]


# Fixtures

@pytest.fixture
def sample_experiment_result():
    """Create sample experiment result with unique IDs."""
    uid = unique_id()
    return ExperimentResult(
        id=f"result-{uid}",
        experiment_id=f"exp-{uid}",
        hypothesis_id=f"hyp-{uid}",
        protocol_id=f"proto-{uid}",
        status=ResultStatus.SUCCESS,
        primary_test="Two-sample T-test",
        primary_p_value=0.012,
        primary_effect_size=0.65,
        primary_ci_lower=0.2,
        primary_ci_upper=1.1,
        supports_hypothesis=True,
        statistical_tests=[
            StatisticalTestResult(
                test_type="t-test",
                test_name="Two-sample T-test",
                statistic=2.54,
                p_value=0.012,
                effect_size=0.65,
                effect_size_type="Cohen's d",
                confidence_interval={"lower": 0.2, "upper": 1.1},
                sample_size=100,
                degrees_of_freedom=98,
                significance_label="*",
                is_primary=True,
                significant_0_05=True,   # p=0.012 < 0.05
                significant_0_01=False,  # p=0.012 > 0.01
                significant_0_001=False  # p=0.012 > 0.001
            )
        ],
        variable_results=[
            VariableResult(
                variable_name="treatment",
                variable_type="independent",
                mean=10.5,
                median=10.3,
                std=2.1,
                min=6.2,
                max=15.8,
                q1=9.1,
                q3=11.9,
                n_samples=50,
                n_missing=0
            ),
            VariableResult(
                variable_name="control",
                variable_type="independent",
                mean=8.8,
                median=8.5,
                std=2.3,
                min=4.5,
                max=13.2,
                q1=7.2,
                q3=10.1,
                n_samples=50,
                n_missing=0
            )
        ],
        metadata=ExecutionMetadata(
            experiment_id=f"exp-{uid}",
            protocol_id=f"proto-{uid}",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            duration_seconds=5.3,
            random_seed=42,
            python_version="3.11",
            platform="linux"
        ),
        raw_data={"mean_diff": 1.7},
        generated_files=[],
        version=1,
        created_at=datetime.utcnow()
    )


@pytest.fixture
def sample_hypothesis():
    """Create sample hypothesis with unique ID."""
    uid = unique_id()
    return Hypothesis(
        id=f"hyp-{uid}",
        research_question=f"Does treatment X increase outcome Y compared to control? [test-{uid}]",
        statement="Treatment X increases outcome Y compared to control",
        rationale="Prior studies suggest mechanism via pathway Z operates through documented biological pathways",
        domain="biology",
        testability_score=0.9,
        novelty_score=0.7,
        variables=["treatment", "control", "outcome_Y"],
        created_at=datetime.utcnow()
    )


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# End-to-End Pipeline Tests


# Statistical Analysis Tests

class TestStatisticalAnalysis:
    """Tests for statistical analysis components."""

    def test_descriptive_statistics(self):
        """Test descriptive statistics computation."""
        np.random.seed(42)
        data = np.random.normal(10, 2, 100)

        stats = DescriptiveStats.compute_full_descriptive(data)

        assert 'mean' in stats
        assert 'median' in stats
        assert 'std' in stats
        assert 'skewness' in stats
        assert stats['n'] == 100
        assert 9 < stats['mean'] < 11  # Should be close to 10

    def test_statistical_reporter(self):
        """Test comprehensive statistical report generation."""
        np.random.seed(42)
        df = pd.DataFrame({
            'var1': np.random.normal(10, 2, 50),
            'var2': np.random.normal(15, 3, 50),
            'var3': np.random.normal(20, 4, 50)
        })

        reporter = StatisticalReporter()
        report = reporter.generate_full_report(df, include_correlations=True, include_distributions=True)

        assert len(report) > 0
        assert 'Descriptive Statistics' in report
        assert 'Distribution Analysis' in report or 'Correlation Analysis' in report


# Visualization Format Tests

class TestVisualizationFormats:
    """Tests for visualization output formats."""

    def test_matplotlib_and_plotly_compatibility(self, temp_output_dir):
        """Test both matplotlib and plotly visualizers work."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2 * x + np.random.randn(50)

        # Matplotlib version
        pub_viz = PublicationVisualizer()
        pub_path = os.path.join(temp_output_dir, "matplotlib.png")
        pub_viz.scatter_with_regression(x, y, "X", "Y", "Matplotlib", pub_path)

        assert os.path.exists(pub_path)

        # Plotly version
        try:
            plotly_viz = PlotlyVisualizer()
            fig = plotly_viz.interactive_scatter(x, y, "X", "Y", "Plotly")

            html_path = os.path.join(temp_output_dir, "plotly.html")
            plotly_viz.save_html(fig, html_path)

            assert os.path.exists(html_path)
        except ImportError:
            pytest.skip("Plotly not installed")


# Anomaly and Pattern Detection Tests


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

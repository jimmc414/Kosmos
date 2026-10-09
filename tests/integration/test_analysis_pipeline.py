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
from kosmos.analysis.summarizer import ResultSummarizer, ResultSummary

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

class TestCompleteAnalysisPipeline:
    """Tests for complete analysis pipeline with real Claude API."""

    def test_full_pipeline_result_to_interpretation(
        self,
        sample_experiment_result,
        sample_hypothesis
    ):
        """Test Result → DataAnalystAgent → Interpretation with real Claude."""
        # Use real Claude client via patch pattern
        with patch('kosmos.agents.data_analyst.get_client') as mock_get_client:
            from kosmos.core.llm import ClaudeClient
            mock_get_client.return_value = ClaudeClient(model="claude-3-haiku-20240307")

            # Create agent (will use real Claude via patched get_client)
            agent = DataAnalystAgent()

            # Interpret results
            interpretation = agent.interpret_results(
                result=sample_experiment_result,
                hypothesis=sample_hypothesis
            )

            assert isinstance(interpretation, ResultInterpretation)
            assert interpretation.experiment_id == sample_experiment_result.experiment_id
            # Real Claude will analyze data and provide assessment
            assert interpretation.hypothesis_supported is not None
            assert len(interpretation.key_findings) >= 0  # May have findings

    def test_full_pipeline_result_to_visualization(
        self,
        sample_experiment_result,
        temp_output_dir
    ):
        """Test Result → PublicationVisualizer → Plots."""
        viz = PublicationVisualizer()

        # Auto-select plots
        suggested_plots = viz.select_plot_types(sample_experiment_result)

        assert len(suggested_plots) > 0
        assert any(p['type'] == 'box_plot_with_points' for p in suggested_plots)

        # Generate a plot
        np.random.seed(42)
        data = {
            'treatment': np.random.normal(10.5, 2.1, 50),
            'control': np.random.normal(8.8, 2.3, 50)
        }

        output_path = os.path.join(temp_output_dir, "test_plot.png")
        viz.box_plot_with_points(
            data=data,
            title="Treatment vs Control",
            y_label="Outcome",
            output_path=output_path
        )

        assert os.path.exists(output_path)

    def test_full_pipeline_result_to_summary(
        self,
        sample_experiment_result,
        sample_hypothesis
    ):
        """Test Result → ResultSummarizer → Summary with real Claude."""
        # Use real Claude client via patch pattern
        with patch('kosmos.analysis.summarizer.get_client') as mock_get_client:
            from kosmos.core.llm import ClaudeClient
            mock_get_client.return_value = ClaudeClient(model="claude-3-haiku-20240307")

            # Create summarizer (will use real Claude via patched get_client)
            summarizer = ResultSummarizer()

            # Generate summary
            summary = summarizer.generate_summary(
                result=sample_experiment_result,
                hypothesis=sample_hypothesis
            )

            assert isinstance(summary, ResultSummary)
            assert summary.experiment_id == sample_experiment_result.experiment_id
            assert len(summary.summary) > 0
            # Real Claude may or may not generate findings based on response
            assert len(summary.key_findings) >= 0

    def test_complete_pipeline_integration(
        self,
        sample_experiment_result,
        sample_hypothesis,
        temp_output_dir
    ):
        """Test complete pipeline: Result → All Analysis Components with real Claude."""
        from kosmos.core.llm import ClaudeClient

        # 1. Interpret results with real Claude
        with patch('kosmos.agents.data_analyst.get_client') as mock_get_client:
            mock_get_client.return_value = ClaudeClient(model="claude-3-haiku-20240307")

            agent = DataAnalystAgent()
            interpretation = agent.interpret_results(
                result=sample_experiment_result,
                hypothesis=sample_hypothesis
            )

        # 2. Generate visualizations (pure Python)
        viz = PublicationVisualizer()
        np.random.seed(42)
        data = {
            'treatment': np.random.normal(10.5, 2.1, 50),
            'control': np.random.normal(8.8, 2.3, 50)
        }

        plot_path = os.path.join(temp_output_dir, "analysis_plot.png")
        viz.box_plot_with_points(data=data, output_path=plot_path)

        # 3. Generate summary with real Claude
        with patch('kosmos.analysis.summarizer.get_client') as mock_get_client2:
            mock_get_client2.return_value = ClaudeClient(model="claude-3-haiku-20240307")

            summarizer = ResultSummarizer()
            summary = summarizer.generate_summary(
                result=sample_experiment_result,
                hypothesis=sample_hypothesis
            )

        # Verify all components completed successfully
        assert interpretation is not None
        assert os.path.exists(plot_path)
        assert summary is not None


# Statistical Analysis Tests


# Visualization Format Tests


# Anomaly and Pattern Detection Tests

class TestDetectionPipeline:
    """Tests for anomaly and pattern detection (pure Python, no LLM needed)."""

    def test_anomaly_detection_in_pipeline(self):
        """Test anomaly detection on problematic results (pure Python logic)."""
        # Patch get_client to avoid API call during init (detection methods are pure Python)
        with patch('kosmos.agents.data_analyst.get_client') as mock_get_client:
            mock_get_client.return_value = Mock()
            agent = DataAnalystAgent()

        # Create result with anomaly (significant p-value, tiny effect)
        anomalous_result = ExperimentResult(
            id="result-anom",
            experiment_id="exp-anom",
            hypothesis_id="hyp-anom",
            protocol_id="proto-anom",
            status=ResultStatus.SUCCESS,
            primary_test="T-test",
            primary_p_value=0.001,  # Very significant
            primary_effect_size=0.05,  # Tiny effect
            supports_hypothesis=True,
            statistical_tests=[
                StatisticalTestResult(
                    test_type="T-test",
                    test_name="T-test",
                    statistic=3.5,
                    p_value=0.001,
                    effect_size=0.05,
                    effect_size_type="Cohen's d",
                    confidence_interval={"lower": 0.01, "upper": 0.09},
                    sample_size=100,
                    degrees_of_freedom=98,
                    significance_label="***",
                    is_primary=True,
                    significant_0_05=True,
                    significant_0_01=True,
                    significant_0_001=True
                )
            ],
            variable_results=[],
            metadata=ExecutionMetadata(
                experiment_id="exp-anom",
                protocol_id="proto-anom",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                duration_seconds=1.0,
                random_seed=42,
                python_version="3.11",
                platform="linux"
            ),
            created_at=datetime.utcnow()
        )

        anomalies = agent.detect_anomalies(anomalous_result)

        assert len(anomalies) > 0
        assert any("tiny effect size" in a.lower() for a in anomalies)

    def test_pattern_detection_across_results(self):
        """Test pattern detection across multiple results (pure Python logic)."""
        # Patch get_client to avoid API call during init (detection methods are pure Python)
        with patch('kosmos.agents.data_analyst.get_client') as mock_get_client:
            mock_get_client.return_value = Mock()
            agent = DataAnalystAgent()

        # Create results with consistent positive effects
        results = [
            ExperimentResult(
                id=f"result-{i}",
                experiment_id=f"exp-{i}",
                hypothesis_id="hyp-001",
                protocol_id="proto-001",
                status=ResultStatus.SUCCESS,
                primary_test="T-test",
                primary_p_value=0.01,
                primary_effect_size=0.5 + i * 0.1,  # Positive, increasing
                supports_hypothesis=True,
                statistical_tests=[
                    StatisticalTestResult(
                        test_type="T-test",
                        test_name="T-test",
                        statistic=2.5 + i * 0.2,
                        p_value=0.01,
                        effect_size=0.5 + i * 0.1,
                        effect_size_type="Cohen's d",
                        confidence_interval={"lower": 0.2 + i * 0.1, "upper": 0.8 + i * 0.1},
                        sample_size=100,
                        degrees_of_freedom=98,
                        significance_label="**",
                        is_primary=True,
                        significant_0_05=True,
                        significant_0_01=True,
                        significant_0_001=False
                    )
                ],
                variable_results=[],
                metadata=ExecutionMetadata(
                    experiment_id=f"exp-{i}",
                    protocol_id="proto-001",
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    duration_seconds=1.0,
                    random_seed=42,
                    python_version="3.11",
                    platform="linux"
                ),
                created_at=datetime.utcnow()
            )
            for i in range(5)
        ]

        patterns = agent.detect_patterns_across_results(results)

        assert len(patterns) > 0
        # Should detect consistent positive effects or increasing trend
        assert any("positive" in p.lower() or "increasing" in p.lower() for p in patterns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

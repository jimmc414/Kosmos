"""
Integration tests for complete analysis pipeline (Phase 6).

Tests end-to-end flow: ExperimentResult → Analysis → Visualization → Summary.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import os
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


# Fixtures

@pytest.fixture
def sample_experiment_result():
    """Create sample experiment result."""
    return ExperimentResult(
        id="result-001",
        experiment_id="exp-001",
        hypothesis_id="hyp-001",
        protocol_id="proto-001",
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
                n_samples=50,
                n_missing=0
            )
        ],
        metadata=ExecutionMetadata(
            experiment_id="exp-001",
            protocol_id="proto-001",
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
    """Create sample hypothesis."""
    return Hypothesis(
        id="hyp-001",
        research_question="Does treatment X increase outcome Y compared to control?",
        statement="Treatment X increases outcome Y compared to control",
        rationale="Prior studies suggest mechanism via pathway Z operates through documented biological pathways",
        domain="biology",
        testability_score=0.9,
        novelty_score=0.7,
        created_at=datetime.utcnow()
    )


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# End-to-End Pipeline Tests

class TestCompleteAnalysisPipeline:
    """Tests for complete analysis pipeline."""

    @patch('kosmos.agents.data_analyst.get_client')
    def test_full_pipeline_result_to_interpretation(
        self,
        mock_get_client,
        sample_experiment_result,
        sample_hypothesis
    ):
        """Test Result → DataAnalystAgent → Interpretation."""
        # Mock Claude client
        mock_client = Mock()
        mock_client.generate.return_value = '{"hypothesis_supported": true, "confidence": 0.85, "summary": "Test summary", "key_findings": ["Finding 1"], "significance_interpretation": "Significant", "biological_significance": "Meaningful", "comparison_to_prior_work": "Similar", "potential_confounds": ["Confound 1"], "follow_up_experiments": ["Experiment 1"], "overall_assessment": "Good"}'
        mock_get_client.return_value = mock_client

        # Create agent
        agent = DataAnalystAgent()
        agent.llm_client = mock_client

        # Interpret results
        interpretation = agent.interpret_results(
            result=sample_experiment_result,
            hypothesis=sample_hypothesis
        )

        assert isinstance(interpretation, ResultInterpretation)
        assert interpretation.experiment_id == "exp-001"
        assert interpretation.hypothesis_supported is True
        assert len(interpretation.key_findings) > 0

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

    @patch('kosmos.analysis.summarizer.get_client')
    def test_full_pipeline_result_to_summary(
        self,
        mock_get_client,
        sample_experiment_result,
        sample_hypothesis
    ):
        """Test Result → ResultSummarizer → Summary."""
        # Mock Claude client
        mock_client = Mock()
        mock_client.generate.return_value = """
SUMMARY: The experiment tested whether Treatment X increases outcome Y compared to control. Results show a statistically significant increase with medium-large effect size.

KEY FINDINGS:
1. Treatment group significantly higher than control (p=0.012)
2. Effect size Cohen's d=0.65 (medium-large)
3. Results support the hypothesis

HYPOTHESIS ASSESSMENT: Results support the hypothesis that Treatment X increases outcome Y.

LIMITATIONS:
- Sample size moderate (n=100)
- Single-center study

FUTURE WORK:
1. Replicate in larger cohort
2. Investigate mechanism
3. Test dose-response
"""
        mock_get_client.return_value = mock_client

        # Create summarizer
        summarizer = ResultSummarizer()
        summarizer.llm_client = mock_client

        # Generate summary
        summary = summarizer.generate_summary(
            result=sample_experiment_result,
            hypothesis=sample_hypothesis
        )

        assert isinstance(summary, ResultSummary)
        assert summary.experiment_id == "exp-001"
        assert len(summary.summary) > 0
        assert len(summary.key_findings) > 0
        assert len(summary.future_work) > 0

    def test_complete_pipeline_integration(
        self,
        sample_experiment_result,
        sample_hypothesis,
        temp_output_dir
    ):
        """Test complete pipeline: Result → All Analysis Components."""
        # 1. Interpret results
        with patch('kosmos.agents.data_analyst.get_client') as mock_client:
            mock_client.return_value = Mock()
            mock_client.return_value.generate.return_value = '{"hypothesis_supported": true, "confidence": 0.85, "summary": "Test", "key_findings": ["F1"], "significance_interpretation": "Sig", "biological_significance": "Bio", "comparison_to_prior_work": "Comp", "potential_confounds": ["C1"], "follow_up_experiments": ["E1"], "overall_assessment": "Good"}'

            agent = DataAnalystAgent()
            agent.llm_client = mock_client.return_value

            interpretation = agent.interpret_results(
                result=sample_experiment_result,
                hypothesis=sample_hypothesis
            )

        # 2. Generate visualizations
        viz = PublicationVisualizer()
        np.random.seed(42)
        data = {
            'treatment': np.random.normal(10.5, 2.1, 50),
            'control': np.random.normal(8.8, 2.3, 50)
        }

        plot_path = os.path.join(temp_output_dir, "analysis_plot.png")
        viz.box_plot_with_points(data=data, output_path=plot_path)

        # 3. Generate summary
        with patch('kosmos.analysis.summarizer.get_client') as mock_client2:
            mock_client2.return_value = Mock()
            mock_client2.return_value.generate.return_value = "SUMMARY: Test\nKEY FINDINGS:\n1. Finding 1\nHYPOTHESIS ASSESSMENT: Supported\nLIMITATIONS:\n- Limit 1\nFUTURE WORK:\n1. Work 1"

            summarizer = ResultSummarizer()
            summarizer.llm_client = mock_client2.return_value

            summary = summarizer.generate_summary(
                result=sample_experiment_result,
                hypothesis=sample_hypothesis
            )

        # Verify all components completed successfully
        assert interpretation is not None
        assert os.path.exists(plot_path)
        assert summary is not None


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

class TestDetectionPipeline:
    """Tests for anomaly and pattern detection."""

    @patch('kosmos.agents.data_analyst.get_client')
    def test_anomaly_detection_in_pipeline(self, mock_get_client):
        """Test anomaly detection on problematic results."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        agent = DataAnalystAgent()
        agent.llm_client = mock_client

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

    @patch('kosmos.agents.data_analyst.get_client')
    def test_pattern_detection_across_results(self, mock_get_client):
        """Test pattern detection across multiple results."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        agent = DataAnalystAgent()
        agent.llm_client = mock_client

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

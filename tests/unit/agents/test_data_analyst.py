"""
Tests for Data Analyst Agent.

Tests Claude-powered result interpretation, anomaly detection, pattern detection.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from kosmos.agents.data_analyst import DataAnalystAgent, ResultInterpretation
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
def data_analyst_agent():
    """Create DataAnalystAgent without real Claude client."""
    with patch('kosmos.agents.data_analyst.get_client'):
        agent = DataAnalystAgent(config={
            "use_literature_context": True,
            "detailed_interpretation": True,
            "anomaly_detection_enabled": True,
            "pattern_detection_enabled": True
        })
        # Mock LLM client
        agent.llm_client = Mock()
        return agent


@pytest.fixture
def sample_experiment_result():
    """Create sample successful experiment result."""
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
                is_primary=True
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
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            duration_seconds=5.3,
            random_seed=42
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
        rationale="Prior studies suggest mechanism via pathway Z",
        domain="biology",
        testability_score=0.9,
        novelty_score=0.7,
        created_at=datetime.utcnow()
    )


@pytest.fixture
def mock_claude_interpretation():
    """Mock Claude interpretation response."""
    return json.dumps({
        "hypothesis_supported": True,
        "confidence": 0.85,
        "summary": "The experiment provides strong evidence supporting the hypothesis. "
                   "Treatment X showed a statistically significant increase in outcome Y "
                   "with a medium-large effect size.",
        "key_findings": [
            "Statistically significant difference (p=0.012) between treatment and control",
            "Medium-large effect size (Cohen's d=0.65) suggests practical significance",
            "Treatment group mean (10.5) exceeded control (8.8) by 1.7 units"
        ],
        "significance_interpretation": "The p-value of 0.012 provides strong evidence against "
                                      "the null hypothesis. Combined with a Cohen's d of 0.65, "
                                      "this suggests both statistical and practical significance.",
        "biological_significance": "The effect size suggests the treatment has a meaningful "
                                  "biological impact, likely mediated through pathway Z as hypothesized.",
        "comparison_to_prior_work": "Results align with prior studies showing similar effect sizes "
                                   "for this intervention type.",
        "potential_confounds": [
            "Sample allocation method not specified - randomization should be verified",
            "Possible placebo effect if blinding was not used",
            "Baseline differences between groups should be checked"
        ],
        "follow_up_experiments": [
            "Test dose-response relationship to establish optimal treatment level",
            "Investigate mechanism via pathway Z using targeted assays",
            "Replicate in independent cohort to confirm findings",
            "Test long-term effects and sustainability of treatment benefit"
        ],
        "overall_assessment": "Quality: 4/5. Well-designed experiment with appropriate statistical "
                             "analysis and meaningful effect size. Some methodological details "
                             "(randomization, blinding) should be verified."
    })


# Result Interpretation Tests

class TestResultInterpretation:
    """Tests for result interpretation functionality."""

    def test_interpret_results_success(self, data_analyst_agent, sample_experiment_result,
                                       sample_hypothesis, mock_claude_interpretation):
        """Test successful result interpretation."""
        # Mock Claude response
        data_analyst_agent.llm_client.generate.return_value = mock_claude_interpretation

        interpretation = data_analyst_agent.interpret_results(
            result=sample_experiment_result,
            hypothesis=sample_hypothesis,
            literature_context="Prior work shows similar effects..."
        )

        assert isinstance(interpretation, ResultInterpretation)
        assert interpretation.experiment_id == "exp-001"
        assert interpretation.hypothesis_supported is True
        assert interpretation.confidence == 0.85
        assert len(interpretation.key_findings) == 3
        assert len(interpretation.potential_confounds) > 0
        assert len(interpretation.follow_up_experiments) > 0

    def test_interpret_results_without_hypothesis(self, data_analyst_agent, sample_experiment_result,
                                                  mock_claude_interpretation):
        """Test interpretation without hypothesis."""
        data_analyst_agent.llm_client.generate.return_value = mock_claude_interpretation

        interpretation = data_analyst_agent.interpret_results(
            result=sample_experiment_result,
            hypothesis=None,
            literature_context=None
        )

        assert isinstance(interpretation, ResultInterpretation)
        assert interpretation.experiment_id == "exp-001"

    def test_interpret_results_claude_failure(self, data_analyst_agent, sample_experiment_result):
        """Test fallback when Claude fails."""
        data_analyst_agent.llm_client.generate.side_effect = Exception("Claude API error")

        interpretation = data_analyst_agent.interpret_results(
            result=sample_experiment_result
        )

        # Should return fallback interpretation
        assert isinstance(interpretation, ResultInterpretation)
        assert "fallback" in interpretation.overall_assessment.lower() or \
               "automated" in interpretation.overall_assessment.lower()

    def test_interpret_results_invalid_json(self, data_analyst_agent, sample_experiment_result):
        """Test handling of invalid JSON from Claude."""
        data_analyst_agent.llm_client.generate.return_value = "This is not valid JSON"

        interpretation = data_analyst_agent.interpret_results(
            result=sample_experiment_result
        )

        # Should return fallback interpretation
        assert isinstance(interpretation, ResultInterpretation)

    def test_extract_result_summary(self, data_analyst_agent, sample_experiment_result):
        """Test extraction of result summary."""
        summary = data_analyst_agent._extract_result_summary(sample_experiment_result)

        assert summary["experiment_id"] == "exp-001"
        assert summary["status"] == "success"
        assert summary["primary_test"] == "Two-sample T-test"
        assert summary["primary_p_value"] == 0.012
        assert summary["primary_effect_size"] == 0.65
        assert len(summary["statistical_tests"]) == 1
        assert len(summary["variables"]) == 2

    def test_build_interpretation_prompt(self, data_analyst_agent, sample_experiment_result,
                                        sample_hypothesis):
        """Test interpretation prompt building."""
        summary = data_analyst_agent._extract_result_summary(sample_experiment_result)
        prompt = data_analyst_agent._build_interpretation_prompt(
            summary, sample_hypothesis, "Literature context..."
        )

        assert "HYPOTHESIS:" in prompt
        assert sample_hypothesis.statement in prompt
        assert "EXPERIMENTAL RESULTS:" in prompt
        assert "P-value: 0.012" in prompt
        assert "LITERATURE CONTEXT:" in prompt
        assert "Format your response as JSON" in prompt


# Anomaly Detection Tests

class TestAnomalyDetection:
    """Tests for anomaly detection functionality."""

    def test_detect_anomaly_tiny_effect_significant_p(self, data_analyst_agent):
        """Test detection of significant p-value with tiny effect size."""
        result = ExperimentResult(
            id="result-002",
            experiment_id="exp-002",
            hypothesis_id="hyp-002",
            protocol_id="proto-002",
            status=ResultStatus.SUCCESS,
            primary_test="T-test",
            primary_p_value=0.001,  # Very significant
            primary_effect_size=0.05,  # Tiny effect
            supports_hypothesis=True,
            statistical_tests=[],
            variable_results=[],
            metadata=ExecutionMetadata(
                experiment_id="exp-002",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                duration_seconds=1.0,
                random_seed=42
            ),
            created_at=datetime.utcnow()
        )

        anomalies = data_analyst_agent.detect_anomalies(result)

        assert len(anomalies) > 0
        assert any("tiny effect size" in a.lower() for a in anomalies)

    def test_detect_anomaly_large_effect_nonsignificant_p(self, data_analyst_agent):
        """Test detection of large effect size with non-significant p-value."""
        result = ExperimentResult(
            id="result-003",
            experiment_id="exp-003",
            hypothesis_id="hyp-003",
            protocol_id="proto-003",
            status=ResultStatus.SUCCESS,
            primary_test="T-test",
            primary_p_value=0.15,  # Non-significant
            primary_effect_size=0.8,  # Large effect
            supports_hypothesis=False,
            statistical_tests=[],
            variable_results=[],
            metadata=ExecutionMetadata(
                experiment_id="exp-003",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                duration_seconds=1.0,
                random_seed=42
            ),
            created_at=datetime.utcnow()
        )

        anomalies = data_analyst_agent.detect_anomalies(result)

        assert len(anomalies) > 0
        assert any("large effect size" in a.lower() for a in anomalies)

    def test_detect_anomaly_pvalue_zero(self, data_analyst_agent):
        """Test detection of p-value exactly 0."""
        result = ExperimentResult(
            id="result-004",
            experiment_id="exp-004",
            hypothesis_id="hyp-004",
            protocol_id="proto-004",
            status=ResultStatus.SUCCESS,
            primary_test="T-test",
            primary_p_value=0.0,  # Exactly 0 (unusual)
            primary_effect_size=2.5,
            supports_hypothesis=True,
            statistical_tests=[],
            variable_results=[],
            metadata=ExecutionMetadata(
                experiment_id="exp-004",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                duration_seconds=1.0,
                random_seed=42
            ),
            created_at=datetime.utcnow()
        )

        anomalies = data_analyst_agent.detect_anomalies(result)

        assert len(anomalies) > 0
        assert any("exactly 0.0" in a for a in anomalies)

    def test_detect_anomaly_high_variability(self, data_analyst_agent):
        """Test detection of high variability in variables."""
        result = ExperimentResult(
            id="result-005",
            experiment_id="exp-005",
            hypothesis_id="hyp-005",
            protocol_id="proto-005",
            status=ResultStatus.SUCCESS,
            primary_test="T-test",
            primary_p_value=0.05,
            primary_effect_size=0.5,
            supports_hypothesis=True,
            statistical_tests=[],
            variable_results=[
                VariableResult(
                    variable_name="highly_variable",
                    variable_type="dependent",
                    mean=10.0,
                    median=9.5,
                    std=15.0,  # Std > mean (CV > 1.0)
                    min=0.1,
                    max=50.0,
                    n_samples=30,
                    n_missing=0
                )
            ],
            metadata=ExecutionMetadata(
                experiment_id="exp-005",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                duration_seconds=1.0,
                random_seed=42
            ),
            created_at=datetime.utcnow()
        )

        anomalies = data_analyst_agent.detect_anomalies(result)

        assert len(anomalies) > 0
        assert any("high variability" in a.lower() for a in anomalies)

    def test_detect_no_anomalies(self, data_analyst_agent, sample_experiment_result):
        """Test normal result with no anomalies."""
        anomalies = data_analyst_agent.detect_anomalies(sample_experiment_result)

        # Should have no major anomalies (or only minor notes)
        assert len(anomalies) == 0 or all("NOTE:" in a for a in anomalies)


# Pattern Detection Tests

class TestPatternDetection:
    """Tests for pattern detection across multiple results."""

    def test_detect_pattern_consistent_positive_effects(self, data_analyst_agent):
        """Test detection of consistent positive effects."""
        results = [
            ExperimentResult(
                id=f"result-{i}",
                experiment_id=f"exp-{i}",
                hypothesis_id="hyp-001",
                protocol_id="proto-001",
                status=ResultStatus.SUCCESS,
                primary_test="T-test",
                primary_p_value=0.01,
                primary_effect_size=0.5 + i * 0.1,  # Positive effects
                supports_hypothesis=True,
                statistical_tests=[],
                variable_results=[],
                metadata=ExecutionMetadata(
                    experiment_id=f"exp-{i}",
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    duration_seconds=1.0,
                    random_seed=42
                ),
                created_at=datetime.utcnow()
            )
            for i in range(5)
        ]

        patterns = data_analyst_agent.detect_patterns_across_results(results)

        assert len(patterns) > 0
        assert any("positive effects" in p.lower() for p in patterns)

    def test_detect_pattern_increasing_trend(self, data_analyst_agent):
        """Test detection of increasing effect size trend."""
        results = [
            ExperimentResult(
                id=f"result-{i}",
                experiment_id=f"exp-{i}",
                hypothesis_id="hyp-001",
                protocol_id="proto-001",
                status=ResultStatus.SUCCESS,
                primary_test="T-test",
                primary_p_value=0.01,
                primary_effect_size=0.2 * (i + 1),  # Monotonically increasing
                supports_hypothesis=True,
                statistical_tests=[],
                variable_results=[],
                metadata=ExecutionMetadata(
                    experiment_id=f"exp-{i}",
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    duration_seconds=1.0,
                    random_seed=42
                ),
                created_at=datetime.utcnow()
            )
            for i in range(4)
        ]

        patterns = data_analyst_agent.detect_patterns_across_results(results)

        assert len(patterns) > 0
        assert any("increasing trend" in p.lower() for p in patterns)

    def test_detect_pattern_bimodal_pvalues(self, data_analyst_agent):
        """Test detection of bimodal p-value distribution."""
        p_values_list = [0.001, 0.005, 0.002, 0.15, 0.20, 0.18]  # Either very sig or very non-sig
        results = [
            ExperimentResult(
                id=f"result-{i}",
                experiment_id=f"exp-{i}",
                hypothesis_id="hyp-001",
                protocol_id="proto-001",
                status=ResultStatus.SUCCESS,
                primary_test="T-test",
                primary_p_value=p_values_list[i],
                primary_effect_size=0.5,
                supports_hypothesis=p_values_list[i] < 0.05,
                statistical_tests=[],
                variable_results=[],
                metadata=ExecutionMetadata(
                    experiment_id=f"exp-{i}",
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    duration_seconds=1.0,
                    random_seed=42
                ),
                created_at=datetime.utcnow()
            )
            for i in range(len(p_values_list))
        ]

        patterns = data_analyst_agent.detect_patterns_across_results(results)

        assert len(patterns) > 0
        # May detect bimodal distribution
        assert any("bimodal" in p.lower() or "pattern" in p.lower() for p in patterns)

    def test_detect_patterns_insufficient_data(self, data_analyst_agent):
        """Test pattern detection with insufficient data."""
        results = [
            ExperimentResult(
                id="result-001",
                experiment_id="exp-001",
                hypothesis_id="hyp-001",
                protocol_id="proto-001",
                status=ResultStatus.SUCCESS,
                primary_test="T-test",
                primary_p_value=0.01,
                primary_effect_size=0.5,
                supports_hypothesis=True,
                statistical_tests=[],
                variable_results=[],
                metadata=ExecutionMetadata(
                    experiment_id="exp-001",
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    duration_seconds=1.0,
                    random_seed=42
                ),
                created_at=datetime.utcnow()
            )
        ]

        patterns = data_analyst_agent.detect_patterns_across_results(results)

        # Should return empty list for single result
        assert len(patterns) == 0


# Significance Interpretation Tests

class TestSignificanceInterpretation:
    """Tests for statistical significance interpretation."""

    def test_interpret_very_significant(self, data_analyst_agent):
        """Test interpretation of very significant result."""
        interpretation = data_analyst_agent.interpret_significance(
            p_value=0.0001,
            effect_size=0.8,
            sample_size=100
        )

        assert "very strong evidence" in interpretation.lower()
        assert "p < 0.001" in interpretation.lower()
        assert "large" in interpretation.lower()

    def test_interpret_significant_small_effect(self, data_analyst_agent):
        """Test interpretation of significant but small effect."""
        interpretation = data_analyst_agent.interpret_significance(
            p_value=0.01,
            effect_size=0.15,
            sample_size=1000
        )

        assert "strong evidence" in interpretation.lower()
        assert "negligible" in interpretation.lower() or "small" in interpretation.lower()
        assert "large sample size" in interpretation.lower()

    def test_interpret_nonsignificant_large_effect(self, data_analyst_agent):
        """Test interpretation of non-significant but large effect."""
        interpretation = data_analyst_agent.interpret_significance(
            p_value=0.08,
            effect_size=0.75,
            sample_size=20
        )

        assert "suggestive" in interpretation.lower() or "not provide sufficient" in interpretation.lower()
        assert "large" in interpretation.lower()
        assert "small sample size" in interpretation.lower()

    def test_interpret_small_sample_warning(self, data_analyst_agent):
        """Test warning for small sample size."""
        interpretation = data_analyst_agent.interpret_significance(
            p_value=0.05,
            effect_size=0.5,
            sample_size=15
        )

        assert "small sample size" in interpretation.lower()
        assert "caution" in interpretation.lower()


# Agent Lifecycle Tests

class TestAgentLifecycle:
    """Tests for agent lifecycle and task execution."""

    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        with patch('kosmos.agents.data_analyst.get_client'):
            agent = DataAnalystAgent(config={
                "use_literature_context": False,
                "detailed_interpretation": True
            })

            assert agent.agent_type == "DataAnalystAgent"
            assert agent.use_literature_context is False
            assert agent.detailed_interpretation is True
            assert agent.interpretation_history == []

    def test_execute_interpret_results_task(self, data_analyst_agent, sample_experiment_result,
                                           mock_claude_interpretation):
        """Test execute() with interpret_results action."""
        data_analyst_agent.llm_client.generate.return_value = mock_claude_interpretation

        task = {
            "action": "interpret_results",
            "result": sample_experiment_result,
            "hypothesis": None,
            "literature_context": None
        }

        result = data_analyst_agent.execute(task)

        assert result["success"] is True
        assert "interpretation" in result
        assert result["interpretation"]["experiment_id"] == "exp-001"

    def test_execute_detect_anomalies_task(self, data_analyst_agent, sample_experiment_result):
        """Test execute() with detect_anomalies action."""
        task = {
            "action": "detect_anomalies",
            "result": sample_experiment_result
        }

        result = data_analyst_agent.execute(task)

        assert result["success"] is True
        assert "anomalies" in result
        assert isinstance(result["anomalies"], list)

    def test_execute_detect_patterns_task(self, data_analyst_agent):
        """Test execute() with detect_patterns action."""
        results = [
            ExperimentResult(
                id=f"result-{i}",
                experiment_id=f"exp-{i}",
                hypothesis_id="hyp-001",
                protocol_id="proto-001",
                status=ResultStatus.SUCCESS,
                primary_test="T-test",
                primary_p_value=0.01,
                primary_effect_size=0.5 + i * 0.1,
                supports_hypothesis=True,
                statistical_tests=[],
                variable_results=[],
                metadata=ExecutionMetadata(
                    experiment_id=f"exp-{i}",
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    duration_seconds=1.0,
                    random_seed=42
                ),
                created_at=datetime.utcnow()
            )
            for i in range(3)
        ]

        task = {
            "action": "detect_patterns",
            "results": results
        }

        result = data_analyst_agent.execute(task)

        assert result["success"] is True
        assert "patterns" in result
        assert isinstance(result["patterns"], list)

    def test_execute_unknown_action(self, data_analyst_agent):
        """Test execute() with unknown action."""
        task = {
            "action": "unknown_action"
        }

        result = data_analyst_agent.execute(task)

        assert result["success"] is False
        assert "error" in result


# ResultInterpretation Class Tests

class TestResultInterpretationClass:
    """Tests for ResultInterpretation data class."""

    def test_result_interpretation_to_dict(self):
        """Test ResultInterpretation to_dict() method."""
        interpretation = ResultInterpretation(
            experiment_id="exp-001",
            hypothesis_supported=True,
            confidence=0.85,
            summary="Test summary",
            key_findings=["Finding 1", "Finding 2"],
            significance_interpretation="Significant",
            biological_significance="Meaningful",
            comparison_to_prior_work="Similar to prior work",
            potential_confounds=["Confound 1"],
            follow_up_experiments=["Experiment 1"],
            anomalies_detected=[],
            patterns_detected=[],
            overall_assessment="Good quality"
        )

        result_dict = interpretation.to_dict()

        assert result_dict["experiment_id"] == "exp-001"
        assert result_dict["hypothesis_supported"] is True
        assert result_dict["confidence"] == 0.85
        assert len(result_dict["key_findings"]) == 2
        assert "created_at" in result_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

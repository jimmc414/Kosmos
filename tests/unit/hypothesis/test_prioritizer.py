"""Tests for hypothesis prioritizer."""

import pytest
from unittest.mock import Mock, patch
from kosmos.hypothesis.prioritizer import HypothesisPrioritizer
from kosmos.models.hypothesis import Hypothesis

@pytest.fixture
def prioritizer():
    return HypothesisPrioritizer(use_novelty_checker=False, use_testability_analyzer=False, use_impact_prediction=False)

@pytest.fixture
def sample_hypotheses():
    return [
        Hypothesis(
            research_question="Q1",
            statement="Hypothesis 1 with high scores",
            rationale="Well supported hypothesis with strong prior evidence",
            domain="test",
            novelty_score=0.9,
            testability_score=0.8,
            confidence_score=0.85
        ),
        Hypothesis(
            research_question="Q2",
            statement="Hypothesis 2 with medium scores",
            rationale="Moderate hypothesis with partial prior evidence",
            domain="test",
            novelty_score=0.6,
            testability_score=0.5,
            confidence_score=0.55
        ),
        Hypothesis(
            research_question="Q3",
            statement="Hypothesis 3 with low scores",
            rationale="Weak hypothesis with little prior evidence",
            domain="test",
            novelty_score=0.3,
            testability_score=0.4,
            confidence_score=0.35
        )
    ]

@pytest.mark.unit
class TestHypothesisPrioritizer:
    def test_init(self, prioritizer):
        assert sum(prioritizer.weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_prioritize_hypotheses(self, prioritizer, sample_hypotheses):
        ranked = prioritizer.prioritize(sample_hypotheses, run_analysis=False)

        assert len(ranked) == 3
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2
        assert ranked[2].rank == 3

        # Highest scores should be ranked first
        assert ranked[0].priority_score > ranked[1].priority_score
        assert ranked[1].priority_score > ranked[2].priority_score

    def test_feasibility_calculation(self, prioritizer):
        hyp = Hypothesis(
            research_question="Test",
            statement="Test hypothesis",
            rationale="Test rationale for the feasibility score",
            domain="test",
            estimated_resources={"cost_usd": 10, "duration_days": 3, "compute_hours": 5}
        )

        score = prioritizer._calculate_feasibility_score(hyp)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Low cost/duration should be feasible

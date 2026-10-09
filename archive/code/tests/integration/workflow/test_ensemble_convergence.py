"""
Integration tests for Multi-Run Convergence Framework (Issue #64).

Tests end-to-end convergence analysis with realistic data scenarios.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from kosmos.workflow.ensemble import (
    EnsembleRunner,
    ConvergenceAnalyzer,
    ConvergenceReporter,
    EnsembleResult,
    FindingMatch,
    ConvergenceMetrics,
    RunConfig,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def realistic_multi_run_findings():
    """
    Create realistic findings across 5 runs with known replication patterns.

    Pattern:
    - Finding A (KRAS): Appears in all 5 runs (strong convergence)
    - Finding B (EGFR): Appears in 3 runs (moderate convergence)
    - Finding C (TP53): Appears in 2 runs (weak convergence)
    - Finding D (BRCA): Appears in 1 run (no convergence)
    """
    # Finding A - KRAS (strong convergence: 5/5 runs)
    kras_base = {
        'finding_id': 'kras_{}',
        'cycle': 1,
        'task_id': 1,
        'summary': 'KRAS G12C mutations are strongly associated with resistance to EGFR inhibitors',
        'statistics': {'effect_size': 0.72, 'p_value': 0.001, 'n': 150},
        'scholar_eval': {'overall_score': 0.85, 'rigor': 0.88},
        'evidence_type': 'correlation'
    }

    # Finding B - EGFR (moderate convergence: 3/5 runs)
    egfr_base = {
        'finding_id': 'egfr_{}',
        'cycle': 2,
        'task_id': 1,
        'summary': 'EGFR amplification patterns correlate with tumor progression',
        'statistics': {'effect_size': 0.55, 'p_value': 0.008, 'n': 120},
        'scholar_eval': {'overall_score': 0.78, 'rigor': 0.80},
        'evidence_type': 'correlation'
    }

    # Finding C - TP53 (weak convergence: 2/5 runs)
    tp53_base = {
        'finding_id': 'tp53_{}',
        'cycle': 3,
        'task_id': 1,
        'summary': 'TP53 mutations show distinct patterns in late-stage tumors',
        'statistics': {'effect_size': 0.45, 'p_value': 0.02, 'n': 100},
        'scholar_eval': {'overall_score': 0.72, 'rigor': 0.75},
        'evidence_type': 'differential_analysis'
    }

    # Finding D - BRCA (no convergence: 1/5 runs)
    brca_finding = {
        'finding_id': 'brca_run4',
        'cycle': 1,
        'task_id': 2,
        'summary': 'BRCA1 expression levels predict response to platinum therapy',
        'statistics': {'effect_size': 0.38, 'p_value': 0.04, 'n': 80},
        'scholar_eval': {'overall_score': 0.68, 'rigor': 0.70},
        'evidence_type': 'predictive'
    }

    # Add small variation to effect sizes to simulate realistic variation
    def vary_stats(base, run_idx):
        result = base.copy()
        result['finding_id'] = base['finding_id'].format(run_idx)
        result['statistics'] = base['statistics'].copy()
        # Add ±10% variation to effect size
        variation = 1 + (np.random.RandomState(42 + run_idx).random() - 0.5) * 0.2
        result['statistics']['effect_size'] = base['statistics']['effect_size'] * variation
        return result

    return [
        # Run 0: KRAS + EGFR + TP53
        [vary_stats(kras_base, 0), vary_stats(egfr_base, 0), vary_stats(tp53_base, 0)],
        # Run 1: KRAS + EGFR
        [vary_stats(kras_base, 1), vary_stats(egfr_base, 1)],
        # Run 2: KRAS + EGFR + TP53
        [vary_stats(kras_base, 2), vary_stats(egfr_base, 2), vary_stats(tp53_base, 2)],
        # Run 3: KRAS only
        [vary_stats(kras_base, 3)],
        # Run 4: KRAS + BRCA (unique)
        [vary_stats(kras_base, 4), brca_finding],
    ]


@pytest.fixture
def analyzer_with_embeddings_disabled():
    """Create analyzer without embeddings for faster testing."""
    return ConvergenceAnalyzer(
        similarity_threshold=0.70,
        replication_threshold=0.6,
        use_embeddings=False
    )


@pytest.fixture
def run_configs_5():
    """Create 5 run configs."""
    return [
        RunConfig(run_id=f'run_{i}', seed=42 + i, temperature=0.7, run_index=i)
        for i in range(5)
    ]


# =============================================================================
# Full Pipeline Tests
# =============================================================================

class TestEnsembleFullPipeline:
    """End-to-end tests for ensemble convergence pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_mocked_workflows(self, realistic_multi_run_findings, run_configs_5):
        """Test complete pipeline from raw findings to convergence report."""
        # Create analyzer
        analyzer = ConvergenceAnalyzer(
            similarity_threshold=0.70,
            use_embeddings=False
        )

        # Analyze findings
        matched, metrics = analyzer.analyze(realistic_multi_run_findings, run_configs_5)

        # Verify metrics
        assert metrics.total_runs == 5
        assert metrics.total_unique_findings >= 2  # At least KRAS and something else
        assert metrics.total_raw_findings == 11  # 3+2+3+1+2

        # Create result
        result = EnsembleResult(
            research_objective='Investigate cancer mutations',
            n_runs=5,
            run_configs=run_configs_5,
            matched_findings=matched,
            convergence_metrics=metrics,
            convergent_findings=[m for m in matched if m.is_convergent],
            non_convergent_findings=[m for m in matched if not m.is_convergent]
        )

        # Generate report
        reporter = ConvergenceReporter()
        report = reporter.generate_markdown_report(result)

        # Verify report content
        assert '# Multi-Run Convergence Report' in report
        assert 'Investigate cancer mutations' in report

    def test_analyzer_identifies_convergent_findings(
        self, realistic_multi_run_findings, analyzer_with_embeddings_disabled
    ):
        """Test analyzer correctly identifies convergent findings."""
        matched, metrics = analyzer_with_embeddings_disabled.analyze(realistic_multi_run_findings)

        # Should have at least one strongly convergent finding (KRAS appears in all 5 runs)
        strong_findings = [m for m in matched if m.convergence_strength == 'strong']

        # KRAS should be strongly convergent (in 5 runs)
        # Check that we have at least one finding with high replication
        max_replication = max(m.replication_count for m in matched)
        assert max_replication >= 4, f"Expected at least 4 replications, got {max_replication}"

    def test_metrics_count_replication_levels(
        self, realistic_multi_run_findings, analyzer_with_embeddings_disabled
    ):
        """Test metrics correctly count replication levels."""
        matched, metrics = analyzer_with_embeddings_disabled.analyze(realistic_multi_run_findings)

        # Total should be distributed across levels
        total_counted = (
            metrics.findings_replicated_1 +
            metrics.findings_replicated_2_3 +
            metrics.findings_replicated_4_plus
        )

        assert total_counted == metrics.total_unique_findings

    def test_result_to_dict_roundtrip(
        self, realistic_multi_run_findings, run_configs_5, analyzer_with_embeddings_disabled
    ):
        """Test EnsembleResult serialization roundtrip."""
        matched, metrics = analyzer_with_embeddings_disabled.analyze(realistic_multi_run_findings)

        result = EnsembleResult(
            research_objective='Test',
            n_runs=5,
            run_configs=run_configs_5,
            matched_findings=matched,
            convergence_metrics=metrics,
            convergent_findings=[m for m in matched if m.is_convergent],
            non_convergent_findings=[m for m in matched if not m.is_convergent],
            total_time_seconds=100.0,
            start_timestamp='2025-01-01T00:00:00',
            end_timestamp='2025-01-01T01:00:00'
        )

        d = result.to_dict()

        # Verify all fields serialized
        assert d['research_objective'] == 'Test'
        assert d['n_runs'] == 5
        assert len(d['run_configs']) == 5
        assert len(d['matched_findings']) == len(matched)
        assert d['convergence_metrics'] is not None

    @pytest.mark.asyncio
    async def test_runner_integration_with_mocked_workflow(self):
        """Test EnsembleRunner integration with mocked ResearchWorkflow."""
        with patch('kosmos.workflow.research_loop.ResearchWorkflow') as mock_class, \
             patch('kosmos.safety.reproducibility.ReproducibilityManager'):

            # Create mock findings for each run
            run_findings = [
                [{'summary': f'Finding from run {i}', 'statistics': {'effect_size': 0.5}}]
                for i in range(3)
            ]

            call_count = [0]

            def create_mock_workflow(*args, **kwargs):
                mock = AsyncMock()
                mock.run = AsyncMock(return_value={'cycles_completed': 5})
                mock.state_manager = Mock()

                idx = call_count[0]
                call_count[0] += 1

                findings = run_findings[idx] if idx < len(run_findings) else []
                mock.state_manager.get_all_findings = Mock(return_value=[
                    Mock(to_dict=Mock(return_value=f)) for f in findings
                ])
                return mock

            mock_class.side_effect = create_mock_workflow

            runner = EnsembleRunner(n_runs=3)
            result = await runner.run('Test objective', num_cycles=1, tasks_per_cycle=5)

            assert result.n_runs == 3
            assert len(result.run_findings) == 3


# =============================================================================
# Finding Matching Tests
# =============================================================================

class TestFindingMatching:
    """Tests for finding matching across runs."""

    def test_similar_findings_cluster_together(self, analyzer_with_embeddings_disabled):
        """Test that semantically similar findings are clustered."""
        findings = [
            # Run 0
            [{'summary': 'KRAS mutations cause drug resistance', 'statistics': {'effect_size': 0.7}}],
            # Run 1
            [{'summary': 'KRAS mutations lead to drug resistance', 'statistics': {'effect_size': 0.75}}],
            # Run 2
            [{'summary': 'Mutations in KRAS gene cause resistance to drugs',
              'statistics': {'effect_size': 0.68}}],
        ]

        matched = analyzer_with_embeddings_disabled.match_findings(findings)

        # All 3 should cluster together (high similarity)
        max_cluster_size = max(m.replication_count for m in matched)
        assert max_cluster_size >= 2  # At least 2 findings clustered

    def test_different_findings_separate(self, analyzer_with_embeddings_disabled):
        """Test that different findings remain separate."""
        findings = [
            # Run 0
            [{'summary': 'KRAS mutations in cancer', 'statistics': {'effect_size': 0.7}}],
            # Run 1
            [{'summary': 'Weather patterns in Antarctica', 'statistics': {'effect_size': 0.5}}],
            # Run 2
            [{'summary': 'Economic indicators in Europe', 'statistics': {'effect_size': 0.3}}],
        ]

        matched = analyzer_with_embeddings_disabled.match_findings(findings)

        # Each should be in its own cluster (low similarity)
        assert len(matched) == 3
        for m in matched:
            assert m.replication_count == 1

    def test_effect_size_variation_tracked(self, analyzer_with_embeddings_disabled):
        """Test that effect size variation is calculated correctly."""
        findings = [
            [{'summary': 'Same finding A', 'statistics': {'effect_size': 0.50}}],
            [{'summary': 'Same finding A', 'statistics': {'effect_size': 0.55}}],
            [{'summary': 'Same finding A', 'statistics': {'effect_size': 0.52}}],
        ]

        matched = analyzer_with_embeddings_disabled.match_findings(findings)

        # Should cluster and calculate statistics
        main_match = max(matched, key=lambda m: m.replication_count)
        assert len(main_match.effect_sizes) == 3
        assert main_match.effect_size_std > 0

    def test_significance_agreement_calculated(self, analyzer_with_embeddings_disabled):
        """Test significance agreement calculation."""
        findings = [
            [{'summary': 'Finding X', 'statistics': {'p_value': 0.01}}],  # Significant
            [{'summary': 'Finding X', 'statistics': {'p_value': 0.02}}],  # Significant
            [{'summary': 'Finding X', 'statistics': {'p_value': 0.08}}],  # Not significant
        ]

        matched = analyzer_with_embeddings_disabled.match_findings(findings)

        # Find the main cluster
        main_match = max(matched, key=lambda m: m.replication_count)

        # 2/3 agree on significance
        if main_match.p_values:
            assert main_match.significance_agreement == pytest.approx(2/3, abs=0.01)

    def test_direction_agreement_calculated(self, analyzer_with_embeddings_disabled):
        """Test direction agreement calculation."""
        findings = [
            [{'summary': 'Finding Y', 'statistics': {'effect_size': 0.5}}],   # Positive
            [{'summary': 'Finding Y', 'statistics': {'effect_size': 0.6}}],   # Positive
            [{'summary': 'Finding Y', 'statistics': {'effect_size': -0.4}}],  # Negative
        ]

        matched = analyzer_with_embeddings_disabled.match_findings(findings)

        main_match = max(matched, key=lambda m: m.replication_count)

        # 2/3 have positive direction
        if main_match.effect_sizes:
            assert main_match.direction_agreement == pytest.approx(2/3, abs=0.01)


# =============================================================================
# Report Generation Tests
# =============================================================================

class TestConvergenceReport:
    """Tests for convergence report generation."""

    def test_report_includes_all_sections(self, run_configs_5):
        """Test report includes all required sections."""
        metrics = ConvergenceMetrics(
            total_runs=5,
            total_unique_findings=10,
            total_raw_findings=25,
            findings_replicated_1=3,
            findings_replicated_2_3=4,
            findings_replicated_4_plus=3,
            strong_convergence_count=3,
            moderate_convergence_count=2,
            weak_convergence_count=2
        )

        result = EnsembleResult(
            research_objective='Test objective',
            n_runs=5,
            run_configs=run_configs_5,
            convergence_metrics=metrics,
            end_timestamp='2025-01-01T12:00:00'
        )

        reporter = ConvergenceReporter()
        report = reporter.generate_markdown_report(result)

        # Check all sections
        assert '# Multi-Run Convergence Report' in report
        assert 'Executive Summary' in report
        assert 'Methodology' in report

    def test_report_formats_statistics_correctly(self, run_configs_5):
        """Test report formats statistics with appropriate precision."""
        match = FindingMatch(
            match_id='m1',
            canonical_summary='Test finding',
            matched_findings=[],
            run_indices=[0, 1, 2, 3],
            replication_count=4,
            replication_rate=0.8,
            effect_sizes=[0.5, 0.52, 0.48, 0.51],
            effect_size_mean=0.5025,
            effect_size_std=0.0158,
            effect_size_cv=0.031,
            p_values=[0.01, 0.02],
            significance_agreement=1.0,
            direction_agreement=1.0,
            scholar_scores=[0.85],
            scholar_score_mean=0.85,
            scholar_score_std=0.0,
            convergence_strength='strong',
            is_convergent=True
        )

        metrics = ConvergenceMetrics(
            total_runs=5,
            total_unique_findings=1,
            total_raw_findings=4,
            strong_convergence_count=1
        )

        result = EnsembleResult(
            research_objective='Test',
            n_runs=5,
            run_configs=run_configs_5,
            matched_findings=[match],
            convergence_metrics=metrics,
            convergent_findings=[match]
        )

        reporter = ConvergenceReporter()
        report = reporter.generate_markdown_report(result)

        # Check formatting
        assert 'Effect Size' in report
        assert '0.5' in report  # Effect size value
        assert '4/5' in report  # Replication count

    def test_json_report_serializable(self, run_configs_5):
        """Test JSON report can be serialized."""
        import json

        metrics = ConvergenceMetrics(
            total_runs=5,
            total_unique_findings=5,
            total_raw_findings=10
        )

        result = EnsembleResult(
            research_objective='Test',
            n_runs=5,
            run_configs=run_configs_5,
            convergence_metrics=metrics
        )

        reporter = ConvergenceReporter()
        json_data = reporter.generate_json_report(result)

        # Should be JSON serializable
        serialized = json.dumps(json_data, default=str)
        assert len(serialized) > 0

    def test_report_handles_empty_findings(self, run_configs_5):
        """Test report handles case with no findings gracefully."""
        metrics = ConvergenceMetrics(
            total_runs=5,
            total_unique_findings=0,
            total_raw_findings=0
        )

        result = EnsembleResult(
            research_objective='Test',
            n_runs=5,
            run_configs=run_configs_5,
            convergence_metrics=metrics,
            matched_findings=[]
        )

        reporter = ConvergenceReporter()
        report = reporter.generate_markdown_report(result)

        # Should still generate valid report
        assert '# Multi-Run Convergence Report' in report
        assert 'Total Unique Findings**: 0' in report

    def test_report_includes_seeds(self, run_configs_5):
        """Test report includes seed information."""
        metrics = ConvergenceMetrics(
            total_runs=5,
            total_unique_findings=1,
            total_raw_findings=1
        )

        result = EnsembleResult(
            research_objective='Test',
            n_runs=5,
            run_configs=run_configs_5,
            convergence_metrics=metrics
        )

        reporter = ConvergenceReporter()
        report = reporter.generate_markdown_report(result)

        # Should include seeds
        assert '42' in report  # First seed


# =============================================================================
# Statistical Validation Tests
# =============================================================================

class TestStatisticalValidation:
    """Tests validating statistical calculations."""

    def test_effect_size_cv_calculation(self, analyzer_with_embeddings_disabled):
        """Test coefficient of variation calculation is correct."""
        findings = [
            [{'summary': 'Test', 'statistics': {'effect_size': 0.50}}],
            [{'summary': 'Test', 'statistics': {'effect_size': 0.60}}],
            [{'summary': 'Test', 'statistics': {'effect_size': 0.40}}],
        ]

        matched = analyzer_with_embeddings_disabled.match_findings(findings)
        main_match = max(matched, key=lambda m: m.replication_count)

        # CV = std / |mean|
        # mean = 0.5, std = ~0.082
        # CV should be ~0.16
        if main_match.effect_sizes and len(main_match.effect_sizes) > 1:
            expected_mean = np.mean(main_match.effect_sizes)
            expected_std = np.std(main_match.effect_sizes)
            expected_cv = expected_std / abs(expected_mean) if expected_mean != 0 else 0

            assert main_match.effect_size_cv == pytest.approx(expected_cv, abs=0.01)

    def test_convergence_thresholds_applied(self, analyzer_with_embeddings_disabled):
        """Test convergence thresholds are correctly applied."""
        # Finding with high replication but high CV
        findings = [
            [{'summary': 'Unstable finding', 'statistics': {'effect_size': 0.3}}],
            [{'summary': 'Unstable finding', 'statistics': {'effect_size': 0.9}}],  # Very different
            [{'summary': 'Unstable finding', 'statistics': {'effect_size': 0.5}}],
            [{'summary': 'Unstable finding', 'statistics': {'effect_size': 0.7}}],
        ]

        matched = analyzer_with_embeddings_disabled.match_findings(findings)
        main_match = max(matched, key=lambda m: m.replication_count)

        # High replication but high variance - should not be convergent
        if main_match.effect_size_cv > 0.2:
            assert main_match.is_convergent is False

    def test_replication_rate_calculated(self, analyzer_with_embeddings_disabled):
        """Test replication rate is correctly calculated."""
        findings = [
            [{'summary': 'Finding in some runs', 'statistics': {}}],  # Run 0
            [{'summary': 'Finding in some runs', 'statistics': {}}],  # Run 1
            [{'summary': 'Finding in some runs', 'statistics': {}}],  # Run 2
            [],  # Run 3 - no finding
            [],  # Run 4 - no finding
        ]

        matched, metrics = analyzer_with_embeddings_disabled.analyze(findings)

        if matched:
            main_match = max(matched, key=lambda m: m.replication_count)
            # Should be 3/5 = 0.6
            assert main_match.replication_rate == pytest.approx(0.6, abs=0.01)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

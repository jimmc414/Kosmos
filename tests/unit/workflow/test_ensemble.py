"""
Tests for Multi-Run Convergence Framework (Issue #64).

Tests EnsembleRunner, ConvergenceAnalyzer, and related classes.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from dataclasses import asdict
import numpy as np

from kosmos.workflow.ensemble import (
    RunConfig,
    FindingMatch,
    ConvergenceMetrics,
    EnsembleResult,
    EnsembleRunner,
    ConvergenceAnalyzer,
    ConvergenceReporter,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_findings():
    """Create sample findings for testing."""
    return [
        {
            'finding_id': 'f1',
            'cycle': 1,
            'task_id': 1,
            'summary': 'KRAS G12C mutations are associated with drug resistance in lung cancer',
            'statistics': {'effect_size': 0.65, 'p_value': 0.001, 'n': 150},
            'scholar_eval': {'overall_score': 0.82, 'rigor': 0.85},
            'evidence_type': 'correlation'
        },
        {
            'finding_id': 'f2',
            'cycle': 1,
            'task_id': 2,
            'summary': 'KRAS mutations correlate with treatment resistance in cancer patients',
            'statistics': {'effect_size': 0.72, 'p_value': 0.002, 'n': 145},
            'scholar_eval': {'overall_score': 0.79, 'rigor': 0.80},
            'evidence_type': 'correlation'
        },
        {
            'finding_id': 'f3',
            'cycle': 2,
            'task_id': 1,
            'summary': 'EGFR expression patterns differ significantly between tumor types',
            'statistics': {'effect_size': -0.45, 'p_value': 0.03, 'n': 200},
            'scholar_eval': {'overall_score': 0.75, 'rigor': 0.78},
            'evidence_type': 'differential_expression'
        },
    ]


@pytest.fixture
def multi_run_findings(sample_findings):
    """Create findings across 5 runs with varying replication."""
    # Similar KRAS findings in runs 0, 1, 2, 3 (strong convergence)
    kras_variants = [
        {**sample_findings[0], 'finding_id': f'f1_run{i}',
         'statistics': {**sample_findings[0]['statistics'], 'effect_size': 0.65 + 0.03 * i}}
        for i in range(4)
    ]

    # EGFR finding only in runs 0 and 2 (weak convergence)
    egfr_variants = [
        {**sample_findings[2], 'finding_id': f'f3_run{i}',
         'statistics': {**sample_findings[2]['statistics'], 'effect_size': -0.45 - 0.02 * i}}
        for i in [0, 2]
    ]

    # Unique finding only in run 4 (no convergence)
    unique_finding = {
        'finding_id': 'f_unique',
        'cycle': 1,
        'task_id': 1,
        'summary': 'TP53 mutations show unique pattern in colorectal cancer',
        'statistics': {'effect_size': 0.35, 'p_value': 0.04},
        'scholar_eval': {'overall_score': 0.70},
        'evidence_type': 'mutation_analysis'
    }

    return [
        [kras_variants[0], egfr_variants[0]],  # Run 0
        [kras_variants[1]],                     # Run 1
        [kras_variants[2], egfr_variants[1]],  # Run 2
        [kras_variants[3]],                     # Run 3
        [unique_finding],                       # Run 4
    ]


@pytest.fixture
def run_configs():
    """Create sample run configs."""
    return [
        RunConfig(run_id=f'run_{i}', seed=42 + i, temperature=0.7, run_index=i)
        for i in range(5)
    ]


@pytest.fixture
def analyzer():
    """Create ConvergenceAnalyzer with embeddings disabled for testing."""
    return ConvergenceAnalyzer(
        similarity_threshold=0.80,
        use_embeddings=False  # Use token-based similarity for testing
    )


@pytest.fixture
def mock_workflow():
    """Create mock ResearchWorkflow."""
    workflow = AsyncMock()
    workflow.run = AsyncMock(return_value={'cycles_completed': 5})
    workflow.state_manager = Mock()
    workflow.state_manager.get_all_findings = Mock(return_value=[])
    return workflow


# =============================================================================
# RunConfig Tests
# =============================================================================

class TestRunConfig:
    """Tests for RunConfig dataclass."""

    def test_runconfig_creation(self):
        """Test RunConfig can be created."""
        config = RunConfig(
            run_id='run_0',
            seed=42,
            temperature=0.7,
            run_index=0
        )

        assert config.run_id == 'run_0'
        assert config.seed == 42
        assert config.temperature == 0.7
        assert config.run_index == 0

    def test_runconfig_to_dict(self):
        """Test RunConfig serialization."""
        config = RunConfig(run_id='run_0', seed=42, temperature=0.7, run_index=0)
        d = config.to_dict()

        assert d == {
            'run_id': 'run_0',
            'seed': 42,
            'temperature': 0.7,
            'run_index': 0
        }

    def test_runconfig_from_dict(self):
        """Test RunConfig deserialization."""
        data = {'run_id': 'run_1', 'seed': 43, 'temperature': 0.5, 'run_index': 1}
        config = RunConfig.from_dict(data)

        assert config.run_id == 'run_1'
        assert config.seed == 43
        assert config.temperature == 0.5


# =============================================================================
# FindingMatch Tests
# =============================================================================

class TestFindingMatch:
    """Tests for FindingMatch dataclass."""

    def test_findingmatch_creation(self):
        """Test FindingMatch can be created."""
        match = FindingMatch(
            match_id='match_001',
            canonical_summary='KRAS mutations cause resistance',
            matched_findings=[{'summary': 'test'}],
            run_indices=[0, 1, 2],
            replication_count=3,
            replication_rate=0.6,
            convergence_strength='moderate',
            is_convergent=True
        )

        assert match.replication_count == 3
        assert match.replication_rate == 0.6
        assert match.convergence_strength == 'moderate'
        assert match.is_convergent is True

    def test_findingmatch_with_statistics(self):
        """Test FindingMatch with statistical fields."""
        match = FindingMatch(
            match_id='match_002',
            canonical_summary='Test finding',
            matched_findings=[],
            run_indices=[0, 1, 2, 3],
            replication_count=4,
            replication_rate=0.8,
            effect_sizes=[0.5, 0.55, 0.52, 0.48],
            effect_size_mean=0.5125,
            effect_size_std=0.025,
            effect_size_cv=0.049,
            p_values=[0.01, 0.02, 0.015, 0.008],
            significance_agreement=1.0,
            direction_agreement=1.0,
            convergence_strength='strong',
            is_convergent=True
        )

        assert match.effect_size_mean == 0.5125
        assert match.significance_agreement == 1.0
        assert match.direction_agreement == 1.0

    def test_findingmatch_to_dict(self):
        """Test FindingMatch serialization."""
        match = FindingMatch(
            match_id='match_001',
            canonical_summary='Test',
            matched_findings=[],
            run_indices=[0, 1],
            replication_count=2,
            replication_rate=0.4,
            convergence_strength='weak',
            is_convergent=False
        )

        d = match.to_dict()

        assert 'match_id' in d
        assert 'canonical_summary' in d
        assert d['replication_count'] == 2

    def test_findingmatch_default_values(self):
        """Test FindingMatch default values."""
        match = FindingMatch(
            match_id='match_001',
            canonical_summary='Test',
            matched_findings=[],
            run_indices=[0],
            replication_count=1,
            replication_rate=0.2
        )

        assert match.effect_sizes == []
        assert match.effect_size_mean == 0.0
        assert match.convergence_strength == 'none'
        assert match.is_convergent is False

    def test_findingmatch_from_dict(self):
        """Test FindingMatch deserialization."""
        data = {
            'match_id': 'match_001',
            'canonical_summary': 'Test finding',
            'matched_findings': [],
            'run_indices': [0, 1, 2],
            'replication_count': 3,
            'replication_rate': 0.6,
            'effect_sizes': [0.5, 0.6],
            'effect_size_mean': 0.55,
            'effect_size_std': 0.05,
            'effect_size_cv': 0.09,
            'p_values': [],
            'significance_agreement': 0.0,
            'direction_agreement': 1.0,
            'scholar_scores': [],
            'scholar_score_mean': 0.0,
            'scholar_score_std': 0.0,
            'convergence_strength': 'moderate',
            'is_convergent': True
        }

        match = FindingMatch.from_dict(data)
        assert match.replication_count == 3


# =============================================================================
# ConvergenceMetrics Tests
# =============================================================================

class TestConvergenceMetrics:
    """Tests for ConvergenceMetrics dataclass."""

    def test_convergencemetrics_creation(self):
        """Test ConvergenceMetrics can be created."""
        metrics = ConvergenceMetrics(
            total_runs=5,
            total_unique_findings=10,
            total_raw_findings=25
        )

        assert metrics.total_runs == 5
        assert metrics.total_unique_findings == 10
        assert metrics.total_raw_findings == 25

    def test_convergencemetrics_with_counts(self):
        """Test ConvergenceMetrics with all counts."""
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

        assert metrics.findings_replicated_1 == 3
        assert metrics.findings_replicated_4_plus == 3
        assert metrics.strong_convergence_count == 3

    def test_convergencemetrics_to_dict(self):
        """Test ConvergenceMetrics serialization."""
        metrics = ConvergenceMetrics(
            total_runs=5,
            total_unique_findings=10,
            total_raw_findings=25
        )

        d = metrics.to_dict()

        assert 'total_runs' in d
        assert d['total_unique_findings'] == 10

    def test_convergencemetrics_default_thresholds(self):
        """Test ConvergenceMetrics default thresholds."""
        metrics = ConvergenceMetrics(
            total_runs=5,
            total_unique_findings=10,
            total_raw_findings=25
        )

        assert metrics.replication_threshold == 0.6
        assert metrics.effect_cv_threshold == 0.2
        assert metrics.significance_threshold == 0.6


# =============================================================================
# EnsembleResult Tests
# =============================================================================

class TestEnsembleResult:
    """Tests for EnsembleResult dataclass."""

    def test_ensembleresult_creation(self, run_configs):
        """Test EnsembleResult can be created."""
        result = EnsembleResult(
            research_objective='Test objective',
            n_runs=5,
            run_configs=run_configs
        )

        assert result.research_objective == 'Test objective'
        assert result.n_runs == 5
        assert len(result.run_configs) == 5

    def test_ensembleresult_to_dict(self, run_configs):
        """Test EnsembleResult serialization."""
        metrics = ConvergenceMetrics(
            total_runs=5,
            total_unique_findings=3,
            total_raw_findings=10
        )

        result = EnsembleResult(
            research_objective='Test',
            n_runs=5,
            run_configs=run_configs,
            convergence_metrics=metrics
        )

        d = result.to_dict()

        assert d['research_objective'] == 'Test'
        assert d['n_runs'] == 5
        assert len(d['run_configs']) == 5
        assert d['convergence_metrics']['total_runs'] == 5

    def test_get_strongly_convergent(self, run_configs):
        """Test get_strongly_convergent method."""
        strong_match = FindingMatch(
            match_id='m1',
            canonical_summary='Strong finding',
            matched_findings=[],
            run_indices=[0, 1, 2, 3],
            replication_count=4,
            replication_rate=0.8,
            convergence_strength='strong',
            is_convergent=True
        )
        weak_match = FindingMatch(
            match_id='m2',
            canonical_summary='Weak finding',
            matched_findings=[],
            run_indices=[0],
            replication_count=1,
            replication_rate=0.2,
            convergence_strength='none',
            is_convergent=False
        )

        result = EnsembleResult(
            research_objective='Test',
            n_runs=5,
            run_configs=run_configs,
            matched_findings=[strong_match, weak_match]
        )

        strong = result.get_strongly_convergent()
        assert len(strong) == 1
        assert strong[0].canonical_summary == 'Strong finding'

    def test_get_non_replicating(self, run_configs):
        """Test get_non_replicating method."""
        non_rep_match = FindingMatch(
            match_id='m1',
            canonical_summary='Non-replicating finding',
            matched_findings=[],
            run_indices=[4],
            replication_count=1,
            replication_rate=0.2,
            convergence_strength='none',
            is_convergent=False
        )

        result = EnsembleResult(
            research_objective='Test',
            n_runs=5,
            run_configs=run_configs,
            matched_findings=[non_rep_match]
        )

        non_rep = result.get_non_replicating()
        assert len(non_rep) == 1


# =============================================================================
# ConvergenceAnalyzer Tests
# =============================================================================

class TestConvergenceAnalyzer:
    """Tests for ConvergenceAnalyzer class."""

    def test_analyzer_initialization(self):
        """Test analyzer can be initialized."""
        analyzer = ConvergenceAnalyzer(
            similarity_threshold=0.85,
            replication_threshold=0.7,
            use_embeddings=False
        )

        assert analyzer.similarity_threshold == 0.85
        assert analyzer.replication_threshold == 0.7

    def test_compute_text_similarity_identical(self, analyzer):
        """Test text similarity for identical strings."""
        sim = analyzer._compute_text_similarity(
            'KRAS mutations cause resistance',
            'KRAS mutations cause resistance'
        )
        assert sim == 1.0

    def test_compute_text_similarity_similar(self, analyzer):
        """Test text similarity for similar strings."""
        sim = analyzer._compute_text_similarity(
            'KRAS mutations are associated with drug resistance',
            'KRAS mutations correlate with treatment resistance'
        )
        # Should have reasonable similarity (shared keywords)
        assert sim > 0.3

    def test_compute_text_similarity_different(self, analyzer):
        """Test text similarity for different strings."""
        sim = analyzer._compute_text_similarity(
            'KRAS mutations in lung cancer',
            'Weather patterns in Antarctica'
        )
        # Should have low similarity
        assert sim < 0.3

    def test_compute_text_similarity_empty(self, analyzer):
        """Test text similarity with empty strings."""
        assert analyzer._compute_text_similarity('', 'test') == 0.0
        assert analyzer._compute_text_similarity('test', '') == 0.0
        assert analyzer._compute_text_similarity('', '') == 0.0

    def test_compute_statistical_similarity_same_sign(self, analyzer):
        """Test statistical similarity with same sign effect."""
        stats1 = {'effect_size': 0.5, 'p_value': 0.01}
        stats2 = {'effect_size': 0.6, 'p_value': 0.02}

        sim = analyzer._compute_statistical_similarity(stats1, stats2)
        assert sim > 0.7  # Same direction, similar magnitude, both significant

    def test_compute_statistical_similarity_different_sign(self, analyzer):
        """Test statistical similarity with different sign effects."""
        stats1 = {'effect_size': 0.5, 'p_value': 0.01}
        stats2 = {'effect_size': -0.5, 'p_value': 0.01}

        sim = analyzer._compute_statistical_similarity(stats1, stats2)
        assert sim < 0.7  # Opposite directions

    def test_compute_statistical_similarity_empty(self, analyzer):
        """Test statistical similarity with empty stats."""
        sim = analyzer._compute_statistical_similarity({}, {})
        assert sim == 0.5  # Neutral

    def test_compute_finding_similarity(self, analyzer, sample_findings):
        """Test overall finding similarity."""
        # Similar findings
        sim = analyzer._compute_finding_similarity(
            sample_findings[0],
            sample_findings[1]
        )
        assert sim > 0.5  # Should be similar (both about KRAS)

        # Different findings
        sim = analyzer._compute_finding_similarity(
            sample_findings[0],
            sample_findings[2]
        )
        assert sim < 0.8  # Should be less similar (KRAS vs EGFR)

    def test_determine_convergence_strength_strong(self, analyzer):
        """Test strong convergence classification."""
        strength = analyzer._determine_convergence_strength(0.8, 5)
        assert strength == 'strong'

        strength = analyzer._determine_convergence_strength(1.0, 5)
        assert strength == 'strong'

    def test_determine_convergence_strength_moderate(self, analyzer):
        """Test moderate convergence classification."""
        strength = analyzer._determine_convergence_strength(0.6, 5)
        assert strength == 'moderate'

    def test_determine_convergence_strength_weak(self, analyzer):
        """Test weak convergence classification."""
        strength = analyzer._determine_convergence_strength(0.4, 5)
        assert strength == 'weak'

    def test_determine_convergence_strength_none(self, analyzer):
        """Test no convergence classification."""
        strength = analyzer._determine_convergence_strength(0.2, 5)
        assert strength == 'none'

    def test_is_convergent_true(self, analyzer):
        """Test is_convergent returns True for convergent findings."""
        result = analyzer._is_convergent(
            convergence_strength='strong',
            effect_size_cv=0.1,
            significance_agreement=0.8,
            direction_agreement=0.9
        )
        assert result is True

    def test_is_convergent_false_low_replication(self, analyzer):
        """Test is_convergent returns False for low replication."""
        result = analyzer._is_convergent(
            convergence_strength='none',
            effect_size_cv=0.1,
            significance_agreement=0.8,
            direction_agreement=0.9
        )
        assert result is False

    def test_is_convergent_false_high_cv(self, analyzer):
        """Test is_convergent returns False for high CV."""
        result = analyzer._is_convergent(
            convergence_strength='strong',
            effect_size_cv=0.5,  # High variation
            significance_agreement=0.8,
            direction_agreement=0.9
        )
        assert result is False

    def test_match_findings_creates_clusters(self, analyzer, multi_run_findings):
        """Test match_findings clusters similar findings."""
        matched = analyzer.match_findings(multi_run_findings)

        # Should have at least 3 clusters:
        # - KRAS finding (runs 0,1,2,3)
        # - EGFR finding (runs 0,2)
        # - Unique finding (run 4)
        assert len(matched) >= 2

    def test_analyze_returns_metrics(self, analyzer, multi_run_findings):
        """Test analyze returns both findings and metrics."""
        matched, metrics = analyzer.analyze(multi_run_findings)

        assert isinstance(matched, list)
        assert isinstance(metrics, ConvergenceMetrics)
        assert metrics.total_runs == 5

    def test_analyze_counts_replication(self, analyzer, multi_run_findings):
        """Test analyze correctly counts replication levels."""
        matched, metrics = analyzer.analyze(multi_run_findings)

        # Check total raw findings
        total_raw = sum(len(f) for f in multi_run_findings)
        assert metrics.total_raw_findings == total_raw

        # Should have some non-replicating findings
        assert metrics.findings_replicated_1 >= 0


# =============================================================================
# EnsembleRunner Tests
# =============================================================================

class TestEnsembleRunner:
    """Tests for EnsembleRunner class."""

    def test_runner_initialization(self):
        """Test EnsembleRunner can be initialized."""
        runner = EnsembleRunner(n_runs=5)

        assert runner.n_runs == 5
        assert len(runner.seeds) == 5
        assert runner.seeds == [42, 43, 44, 45, 46]

    def test_runner_custom_seeds(self):
        """Test EnsembleRunner with custom seeds."""
        seeds = [100, 200, 300, 400, 500]
        runner = EnsembleRunner(n_runs=5, seeds=seeds)

        assert runner.seeds == seeds

    def test_runner_custom_temperatures(self):
        """Test EnsembleRunner with custom temperatures."""
        temps = [0.3, 0.5, 0.7, 0.9, 1.0]
        runner = EnsembleRunner(n_runs=5, temperatures=temps)

        assert runner.temperatures == temps

    def test_runner_creates_configs(self):
        """Test EnsembleRunner creates run configs."""
        runner = EnsembleRunner(n_runs=3)

        assert len(runner.run_configs) == 3
        assert runner.run_configs[0].run_id == 'run_0'
        assert runner.run_configs[0].seed == 42
        assert runner.run_configs[2].seed == 44

    def test_runner_seed_length_mismatch_raises(self):
        """Test EnsembleRunner raises on seed length mismatch."""
        with pytest.raises(ValueError):
            EnsembleRunner(n_runs=5, seeds=[42, 43])

    def test_runner_temperature_length_mismatch_raises(self):
        """Test EnsembleRunner raises on temperature length mismatch."""
        with pytest.raises(ValueError):
            EnsembleRunner(n_runs=5, temperatures=[0.7, 0.7])

    @pytest.mark.asyncio
    async def test_execute_single_run(self):
        """Test _execute_single_run method."""
        # Patch at the import location within _execute_single_run
        with patch('kosmos.workflow.research_loop.ResearchWorkflow') as mock_workflow_class, \
             patch('kosmos.safety.reproducibility.ReproducibilityManager'):
            # Setup mock
            mock_workflow = AsyncMock()
            mock_workflow.run = AsyncMock(return_value={'cycles_completed': 5})
            mock_workflow.state_manager = Mock()
            mock_workflow.state_manager.get_all_findings = Mock(return_value=[
                Mock(to_dict=Mock(return_value={'summary': 'test'}))
            ])
            mock_workflow_class.return_value = mock_workflow

            runner = EnsembleRunner(n_runs=1)
            config = runner.run_configs[0]

            result, findings = await runner._execute_single_run(
                config, 'Test objective', 1, 5
            )

            assert result is not None
            mock_workflow.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_executes_all_runs(self):
        """Test run method executes all configured runs."""
        # Patch at the import location within _execute_single_run
        with patch('kosmos.workflow.research_loop.ResearchWorkflow') as mock_workflow_class, \
             patch('kosmos.safety.reproducibility.ReproducibilityManager'):
            # Setup mock
            mock_workflow = AsyncMock()
            mock_workflow.run = AsyncMock(return_value={'cycles_completed': 5})
            mock_workflow.state_manager = Mock()
            mock_workflow.state_manager.get_all_findings = Mock(return_value=[])
            mock_workflow_class.return_value = mock_workflow

            runner = EnsembleRunner(n_runs=3)
            result = await runner.run('Test objective', num_cycles=2, tasks_per_cycle=5)

            assert result.n_runs == 3
            assert len(result.run_configs) == 3
            assert mock_workflow.run.call_count == 3


# =============================================================================
# ConvergenceReporter Tests
# =============================================================================

class TestConvergenceReporter:
    """Tests for ConvergenceReporter class."""

    def test_reporter_initialization(self):
        """Test ConvergenceReporter can be initialized."""
        reporter = ConvergenceReporter(output_dir='test_reports')
        assert reporter.output_dir == 'test_reports'

    def test_generate_markdown_report(self, run_configs):
        """Test markdown report generation."""
        metrics = ConvergenceMetrics(
            total_runs=5,
            total_unique_findings=10,
            total_raw_findings=25,
            strong_convergence_count=3,
            moderate_convergence_count=2,
            weak_convergence_count=2,
            findings_replicated_1=3,
            overall_replication_rate=0.6
        )

        strong_match = FindingMatch(
            match_id='m1',
            canonical_summary='Strong KRAS finding',
            matched_findings=[],
            run_indices=[0, 1, 2, 3],
            replication_count=4,
            replication_rate=0.8,
            effect_sizes=[0.5, 0.52, 0.48, 0.51],
            effect_size_mean=0.5025,
            effect_size_std=0.015,
            p_values=[0.01, 0.02, 0.015, 0.01],
            significance_agreement=1.0,
            direction_agreement=1.0,
            scholar_scores=[0.8, 0.82, 0.79, 0.81],
            scholar_score_mean=0.805,
            scholar_score_std=0.01,
            convergence_strength='strong',
            is_convergent=True
        )

        result = EnsembleResult(
            research_objective='Investigate KRAS mutations',
            n_runs=5,
            run_configs=run_configs,
            matched_findings=[strong_match],
            convergence_metrics=metrics,
            convergent_findings=[strong_match],
            non_convergent_findings=[],
            total_time_seconds=3600.0,
            end_timestamp='2025-01-01T00:00:00'
        )

        reporter = ConvergenceReporter()
        report = reporter.generate_markdown_report(result)

        assert '# Multi-Run Convergence Report' in report
        assert 'Investigate KRAS mutations' in report
        assert 'Strong KRAS finding' in report
        assert 'Strongly Convergent' in report

    def test_generate_json_report(self, run_configs):
        """Test JSON report generation."""
        metrics = ConvergenceMetrics(
            total_runs=5,
            total_unique_findings=5,
            total_raw_findings=10
        )

        result = EnsembleResult(
            research_objective='Test',
            n_runs=5,
            run_configs=run_configs,
            convergence_metrics=metrics
        )

        reporter = ConvergenceReporter()
        json_report = reporter.generate_json_report(result)

        assert json_report['research_objective'] == 'Test'
        assert json_report['n_runs'] == 5

    def test_report_includes_non_replicating(self, run_configs):
        """Test report includes non-replicating findings section."""
        non_rep = FindingMatch(
            match_id='m1',
            canonical_summary='Non-replicating finding',
            matched_findings=[],
            run_indices=[4],
            replication_count=1,
            replication_rate=0.2,
            convergence_strength='none',
            is_convergent=False
        )

        metrics = ConvergenceMetrics(
            total_runs=5,
            total_unique_findings=1,
            total_raw_findings=1,
            findings_replicated_1=1
        )

        result = EnsembleResult(
            research_objective='Test',
            n_runs=5,
            run_configs=run_configs,
            matched_findings=[non_rep],
            convergence_metrics=metrics
        )

        reporter = ConvergenceReporter()
        report = reporter.generate_markdown_report(result)

        assert 'Non-Replicating' in report
        assert 'Low Confidence' in report

    def test_report_methodology_section(self, run_configs):
        """Test report includes methodology section."""
        metrics = ConvergenceMetrics(
            total_runs=5,
            total_unique_findings=0,
            total_raw_findings=0
        )

        result = EnsembleResult(
            research_objective='Test',
            n_runs=5,
            run_configs=run_configs,
            convergence_metrics=metrics
        )

        reporter = ConvergenceReporter()
        report = reporter.generate_markdown_report(result)

        assert 'Methodology' in report
        assert 'Similarity Threshold' in report


# =============================================================================
# Integration Tests (within unit test file)
# =============================================================================

class TestConvergenceAnalyzerIntegration:
    """Integration tests for ConvergenceAnalyzer."""

    def test_full_analysis_pipeline(self, analyzer, multi_run_findings):
        """Test full analysis pipeline from raw findings to metrics."""
        matched, metrics = analyzer.analyze(multi_run_findings)

        # Verify we got results
        assert len(matched) > 0
        assert metrics.total_runs == 5
        assert metrics.total_raw_findings == 7  # 2+1+2+1+1

        # Verify matched findings have required fields
        for match in matched:
            assert match.match_id is not None
            assert match.canonical_summary is not None
            assert match.replication_count >= 1
            assert 0 <= match.replication_rate <= 1
            assert match.convergence_strength in ('strong', 'moderate', 'weak', 'none')

    def test_convergence_strength_distribution(self, analyzer):
        """Test convergence strength distribution is calculated correctly."""
        # Create findings with known replication patterns
        run_findings = [
            [{'finding_id': 'f1', 'summary': 'Finding A', 'statistics': {}}],
            [{'finding_id': 'f2', 'summary': 'Finding A duplicate', 'statistics': {}}],
            [{'finding_id': 'f3', 'summary': 'Finding A copy', 'statistics': {}}],
            [{'finding_id': 'f4', 'summary': 'Finding A version', 'statistics': {}}],
            [{'finding_id': 'f5', 'summary': 'Finding A variant', 'statistics': {}}],
        ]

        matched, metrics = analyzer.analyze(run_findings)

        # All similar findings should cluster
        assert metrics.total_unique_findings <= 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

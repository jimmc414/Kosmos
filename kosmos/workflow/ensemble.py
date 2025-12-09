"""
Multi-Run Convergence Framework for Kosmos.

Implements the paper's requirement to run N independent research workflows
with different random seeds and analyze convergence of findings.

Paper Requirements (Section 5 & 6.3):
    "Each research question was run five times with different random seeds"
    "Kosmos is non-deterministic. If a finding is critical, run multiple times
     and look for convergent results."

Example:
    runner = EnsembleRunner(n_runs=5)
    result = await runner.run(
        research_objective="Investigate KRAS mutations in cancer",
        num_cycles=5,
        tasks_per_cycle=10
    )

    # Get strongly convergent findings (appeared in 4-5 runs)
    strong = result.get_strongly_convergent()

    # Generate report
    reporter = ConvergenceReporter()
    report = reporter.generate_markdown_report(result)
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RunConfig:
    """Configuration for a single ensemble run."""
    run_id: str                    # Unique identifier (e.g., "run_0", "run_1")
    seed: int                      # Random seed for this run
    temperature: float             # LLM temperature for this run
    run_index: int                 # 0-based index in ensemble

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunConfig':
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class FindingMatch:
    """
    A finding that was matched across multiple runs.

    Represents a cluster of similar findings from different runs,
    with statistics about their consistency.
    """
    match_id: str                             # Unique identifier for this match
    canonical_summary: str                    # Representative summary (from best finding)
    matched_findings: List[Dict]              # All matched Finding dicts
    run_indices: List[int]                    # Which runs contained this finding
    replication_count: int                    # How many runs found this
    replication_rate: float                   # replication_count / total_runs

    # Statistical consistency
    effect_sizes: List[float] = field(default_factory=list)
    effect_size_mean: float = 0.0
    effect_size_std: float = 0.0
    effect_size_cv: float = 0.0              # Coefficient of variation (std/mean)

    p_values: List[float] = field(default_factory=list)
    significance_agreement: float = 0.0       # Proportion with same significance conclusion
    direction_agreement: float = 0.0          # Proportion with same effect direction

    # ScholarEval consistency
    scholar_scores: List[float] = field(default_factory=list)
    scholar_score_mean: float = 0.0
    scholar_score_std: float = 0.0

    # Convergence verdict
    convergence_strength: str = "none"        # "strong", "moderate", "weak", "none"
    is_convergent: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FindingMatch':
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class ConvergenceMetrics:
    """Aggregate convergence statistics across all runs."""
    total_runs: int
    total_unique_findings: int                # After deduplication
    total_raw_findings: int                   # Before deduplication

    # Replication distribution
    findings_replicated_1: int = 0            # Found in 1 run only
    findings_replicated_2_3: int = 0          # Found in 2-3 runs
    findings_replicated_4_plus: int = 0       # Found in 4+ runs

    # Convergence rates
    overall_replication_rate: float = 0.0     # Avg replication rate across findings
    strong_convergence_count: int = 0         # Findings with >= 4/5 runs
    moderate_convergence_count: int = 0       # Findings with 3/5 runs
    weak_convergence_count: int = 0           # Findings with 2/5 runs

    # Statistical consistency
    avg_effect_size_cv: float = 0.0           # Avg coefficient of variation
    avg_significance_agreement: float = 0.0   # Avg proportion same conclusion
    avg_direction_agreement: float = 0.0      # Avg proportion same direction

    # Quality consistency
    avg_scholar_score_variance: float = 0.0   # Variance in ScholarEval scores

    # Thresholds used
    replication_threshold: float = 0.6        # Default 0.6 (3/5 runs)
    effect_cv_threshold: float = 0.2          # Default 0.2 (20% variation)
    significance_threshold: float = 0.6       # Default 0.6 (60% agreement)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConvergenceMetrics':
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class EnsembleResult:
    """Complete result from an ensemble run."""
    research_objective: str
    n_runs: int
    run_configs: List[RunConfig]

    # Per-run results
    run_results: List[Dict] = field(default_factory=list)      # From ResearchWorkflow.run()
    run_findings: List[List[Dict]] = field(default_factory=list)  # Findings per run

    # Convergence analysis
    matched_findings: List[FindingMatch] = field(default_factory=list)
    convergence_metrics: Optional[ConvergenceMetrics] = None

    # Summary
    convergent_findings: List[FindingMatch] = field(default_factory=list)
    non_convergent_findings: List[FindingMatch] = field(default_factory=list)

    # Metadata
    total_time_seconds: float = 0.0
    start_timestamp: str = ""
    end_timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            'research_objective': self.research_objective,
            'n_runs': self.n_runs,
            'run_configs': [rc.to_dict() for rc in self.run_configs],
            'run_results': self.run_results,
            'run_findings': self.run_findings,
            'matched_findings': [mf.to_dict() for mf in self.matched_findings],
            'convergence_metrics': self.convergence_metrics.to_dict() if self.convergence_metrics else None,
            'convergent_findings': [cf.to_dict() for cf in self.convergent_findings],
            'non_convergent_findings': [ncf.to_dict() for ncf in self.non_convergent_findings],
            'total_time_seconds': self.total_time_seconds,
            'start_timestamp': self.start_timestamp,
            'end_timestamp': self.end_timestamp,
        }
        return result

    def get_strongly_convergent(self) -> List[FindingMatch]:
        """Get findings with strong convergence (4+ runs)."""
        return [f for f in self.matched_findings if f.convergence_strength == "strong"]

    def get_moderately_convergent(self) -> List[FindingMatch]:
        """Get findings with moderate convergence (3 runs)."""
        return [f for f in self.matched_findings if f.convergence_strength == "moderate"]

    def get_non_replicating(self) -> List[FindingMatch]:
        """Get findings that appeared in only 1 run."""
        return [f for f in self.matched_findings if f.convergence_strength == "none"]


# =============================================================================
# Convergence Analyzer
# =============================================================================

class ConvergenceAnalyzer:
    """
    Analyzes findings across multiple runs for convergence.

    Handles:
    - Finding matching/deduplication using semantic similarity
    - Statistical consistency analysis
    - Convergence metric calculation

    Example:
        analyzer = ConvergenceAnalyzer(similarity_threshold=0.80)
        matched, metrics = analyzer.analyze(run_findings)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.80,
        replication_threshold: float = 0.6,
        effect_cv_threshold: float = 0.2,
        significance_threshold: float = 0.6,
        scholar_variance_threshold: float = 0.15,
        use_embeddings: bool = True
    ):
        """
        Initialize ConvergenceAnalyzer.

        Args:
            similarity_threshold: Semantic similarity for matching (0-1)
            replication_threshold: Minimum replication rate for convergence (default 0.6 = 3/5)
            effect_cv_threshold: Maximum coefficient of variation for effect sizes (default 0.2)
            significance_threshold: Minimum agreement on significance conclusion (default 0.6)
            scholar_variance_threshold: Maximum variance in ScholarEval scores (default 0.15)
            use_embeddings: Whether to use sentence-transformers for similarity
        """
        self.similarity_threshold = similarity_threshold
        self.replication_threshold = replication_threshold
        self.effect_cv_threshold = effect_cv_threshold
        self.significance_threshold = significance_threshold
        self.scholar_variance_threshold = scholar_variance_threshold
        self.use_embeddings = use_embeddings

        # Initialize embedding model
        self.model = None
        if use_embeddings:
            self._initialize_model()

    def _initialize_model(self):
        """Initialize sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded sentence transformer: all-MiniLM-L6-v2")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Using fallback token-based similarity. "
                "Install with: pip install sentence-transformers"
            )
            self.model = None

    def analyze(
        self,
        run_findings: List[List[Dict]],
        run_configs: Optional[List[RunConfig]] = None
    ) -> Tuple[List[FindingMatch], ConvergenceMetrics]:
        """
        Main analysis entry point.

        Args:
            run_findings: List of findings per run (each is a list of Finding dicts)
            run_configs: Configuration for each run (optional)

        Returns:
            Tuple of (matched findings, aggregate metrics)
        """
        n_runs = len(run_findings)
        total_raw = sum(len(findings) for findings in run_findings)

        logger.info(f"Analyzing {total_raw} raw findings across {n_runs} runs")

        # Match similar findings across runs
        matched_findings = self.match_findings(run_findings)

        # Calculate aggregate metrics
        metrics = self._calculate_convergence_metrics(matched_findings, n_runs, total_raw)

        logger.info(
            f"Found {len(matched_findings)} unique findings: "
            f"{metrics.strong_convergence_count} strong, "
            f"{metrics.moderate_convergence_count} moderate, "
            f"{metrics.weak_convergence_count} weak, "
            f"{metrics.findings_replicated_1} non-replicating"
        )

        return matched_findings, metrics

    def match_findings(self, run_findings: List[List[Dict]]) -> List[FindingMatch]:
        """
        Match similar findings across runs using semantic similarity.

        Algorithm:
        1. Pool all findings from all runs with run index
        2. Compute pairwise semantic similarity
        3. Cluster findings with similarity > threshold
        4. Create FindingMatch for each cluster

        Args:
            run_findings: List of findings per run

        Returns:
            List of FindingMatch objects
        """
        n_runs = len(run_findings)

        # Pool all findings with their run index
        all_findings = []
        for run_idx, findings in enumerate(run_findings):
            for finding in findings:
                all_findings.append((run_idx, finding))

        if not all_findings:
            return []

        # Compute pairwise similarity and cluster
        clusters = self._cluster_similar_findings(all_findings)

        # Create FindingMatch for each cluster
        matched_findings = []
        for cluster_idx, cluster in enumerate(clusters):
            match = self._create_finding_match(cluster, n_runs, cluster_idx)
            matched_findings.append(match)

        # Sort by replication rate (highest first)
        matched_findings.sort(key=lambda m: (-m.replication_rate, -m.scholar_score_mean))

        return matched_findings

    def _cluster_similar_findings(
        self,
        all_findings: List[Tuple[int, Dict]]
    ) -> List[List[Tuple[int, Dict]]]:
        """
        Cluster similar findings using union-find.

        Args:
            all_findings: List of (run_index, finding_dict) tuples

        Returns:
            List of clusters, each containing (run_idx, finding) tuples
        """
        n = len(all_findings)
        if n == 0:
            return []

        # Initialize union-find
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Compute similarities and merge similar findings
        for i in range(n):
            for j in range(i + 1, n):
                run_i, finding_i = all_findings[i]
                run_j, finding_j = all_findings[j]

                # Don't merge findings from the same run
                if run_i == run_j:
                    continue

                similarity = self._compute_finding_similarity(finding_i, finding_j)
                if similarity >= self.similarity_threshold:
                    union(i, j)

        # Group by cluster
        cluster_map = defaultdict(list)
        for i, (run_idx, finding) in enumerate(all_findings):
            cluster_id = find(i)
            cluster_map[cluster_id].append((run_idx, finding))

        return list(cluster_map.values())

    def _compute_finding_similarity(self, finding1: Dict, finding2: Dict) -> float:
        """
        Compute similarity between two findings.

        Combines:
        - Semantic similarity of summaries (weight: 0.6)
        - Statistical similarity (effect size direction + magnitude) (weight: 0.3)
        - Evidence type match (weight: 0.1)

        Args:
            finding1: First finding dict
            finding2: Second finding dict

        Returns:
            Similarity score (0-1)
        """
        # Text similarity
        text_sim = self._compute_text_similarity(
            finding1.get('summary', ''),
            finding2.get('summary', '')
        )

        # Statistical similarity
        stat_sim = self._compute_statistical_similarity(
            finding1.get('statistics', {}),
            finding2.get('statistics', {})
        )

        # Evidence type match
        type1 = finding1.get('evidence_type', 'unknown')
        type2 = finding2.get('evidence_type', 'unknown')
        type_sim = 1.0 if type1 == type2 else 0.5

        # Weighted combination
        similarity = 0.6 * text_sim + 0.3 * stat_sim + 0.1 * type_sim

        return similarity

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two text strings."""
        if not text1 or not text2:
            return 0.0

        if self.model is not None:
            # Use sentence embeddings
            try:
                emb1 = self.model.encode([text1])[0]
                emb2 = self.model.encode([text2])[0]
                similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
                return max(0.0, min(1.0, similarity))
            except Exception as e:
                logger.warning(f"Embedding error, falling back to token similarity: {e}")

        # Fallback: Jaccard similarity on tokens
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def _compute_statistical_similarity(self, stats1: Dict, stats2: Dict) -> float:
        """
        Compute statistical similarity between two sets of statistics.

        Considers:
        - Effect direction match (same sign)
        - Effect magnitude closeness
        - Significance agreement
        """
        if not stats1 or not stats2:
            return 0.5  # Neutral when no stats

        scores = []

        # Effect size comparison
        es1 = stats1.get('effect_size')
        es2 = stats2.get('effect_size')

        if es1 is not None and es2 is not None:
            # Direction match
            sign_match = 1.0 if (es1 >= 0) == (es2 >= 0) else 0.0

            # Magnitude similarity
            max_es = max(abs(es1), abs(es2))
            if max_es > 0:
                magnitude_sim = 1.0 - abs(es1 - es2) / (2 * max_es)
            else:
                magnitude_sim = 1.0

            scores.append(0.5 * sign_match + 0.5 * magnitude_sim)

        # P-value / significance comparison
        p1 = stats1.get('p_value')
        p2 = stats2.get('p_value')

        if p1 is not None and p2 is not None:
            sig1 = p1 < 0.05
            sig2 = p2 < 0.05
            sig_match = 1.0 if sig1 == sig2 else 0.5
            scores.append(sig_match)

        return np.mean(scores) if scores else 0.5

    def _create_finding_match(
        self,
        cluster: List[Tuple[int, Dict]],
        n_runs: int,
        cluster_idx: int
    ) -> FindingMatch:
        """
        Create a FindingMatch from a cluster of similar findings.

        Args:
            cluster: List of (run_index, finding_dict) tuples
            n_runs: Total number of runs
            cluster_idx: Index for generating match_id

        Returns:
            FindingMatch object with computed statistics
        """
        # Get unique run indices (a run might have multiple similar findings)
        run_indices = sorted(set(run_idx for run_idx, _ in cluster))
        replication_count = len(run_indices)
        replication_rate = replication_count / n_runs

        # Select canonical summary (from finding with highest ScholarEval score)
        findings = [finding for _, finding in cluster]
        best_finding = max(
            findings,
            key=lambda f: self._get_scholar_score(f)
        )
        canonical_summary = best_finding.get('summary', 'Unknown finding')

        # Extract statistics
        effect_sizes = []
        p_values = []
        scholar_scores = []

        for finding in findings:
            stats = finding.get('statistics', {})
            if 'effect_size' in stats and stats['effect_size'] is not None:
                effect_sizes.append(float(stats['effect_size']))
            if 'p_value' in stats and stats['p_value'] is not None:
                p_values.append(float(stats['p_value']))

            scholar_score = self._get_scholar_score(finding)
            if scholar_score > 0:
                scholar_scores.append(scholar_score)

        # Calculate effect size statistics
        effect_size_mean = float(np.mean(effect_sizes)) if effect_sizes else 0.0
        effect_size_std = float(np.std(effect_sizes)) if len(effect_sizes) > 1 else 0.0
        effect_size_cv = (
            abs(effect_size_std / effect_size_mean) if effect_size_mean != 0 else 0.0
        )

        # Calculate significance agreement
        if p_values:
            significance_conclusions = [p < 0.05 for p in p_values]
            most_common_count = max(
                sum(1 for c in significance_conclusions if c),
                sum(1 for c in significance_conclusions if not c)
            )
            significance_agreement = most_common_count / len(significance_conclusions)
        else:
            significance_agreement = 0.0

        # Calculate direction agreement
        if effect_sizes:
            positive_count = sum(1 for es in effect_sizes if es > 0)
            negative_count = sum(1 for es in effect_sizes if es < 0)
            direction_agreement = max(positive_count, negative_count) / len(effect_sizes)
        else:
            direction_agreement = 0.0

        # Calculate ScholarEval statistics
        scholar_score_mean = float(np.mean(scholar_scores)) if scholar_scores else 0.0
        scholar_score_std = float(np.std(scholar_scores)) if len(scholar_scores) > 1 else 0.0

        # Determine convergence strength
        convergence_strength = self._determine_convergence_strength(
            replication_rate, n_runs
        )

        # Determine if convergent
        is_convergent = self._is_convergent(
            convergence_strength,
            effect_size_cv,
            significance_agreement,
            direction_agreement
        )

        return FindingMatch(
            match_id=f"match_{cluster_idx:03d}",
            canonical_summary=canonical_summary,
            matched_findings=findings,
            run_indices=run_indices,
            replication_count=replication_count,
            replication_rate=replication_rate,
            effect_sizes=effect_sizes,
            effect_size_mean=effect_size_mean,
            effect_size_std=effect_size_std,
            effect_size_cv=effect_size_cv,
            p_values=p_values,
            significance_agreement=significance_agreement,
            direction_agreement=direction_agreement,
            scholar_scores=scholar_scores,
            scholar_score_mean=scholar_score_mean,
            scholar_score_std=scholar_score_std,
            convergence_strength=convergence_strength,
            is_convergent=is_convergent
        )

    def _get_scholar_score(self, finding: Dict) -> float:
        """Extract ScholarEval overall score from finding."""
        scholar_eval = finding.get('scholar_eval', {})
        if isinstance(scholar_eval, dict):
            return scholar_eval.get('overall_score', 0.0)
        return 0.0

    def _determine_convergence_strength(
        self,
        replication_rate: float,
        n_runs: int
    ) -> str:
        """
        Determine convergence strength for a finding.

        Args:
            replication_rate: Fraction of runs containing this finding
            n_runs: Total number of runs

        Returns:
            "strong" (4+/5), "moderate" (3/5), "weak" (2/5), "none" (1/5)
        """
        if replication_rate >= 0.8:
            return "strong"
        elif replication_rate >= 0.6:
            return "moderate"
        elif replication_rate >= 0.4:
            return "weak"
        else:
            return "none"

    def _is_convergent(
        self,
        convergence_strength: str,
        effect_size_cv: float,
        significance_agreement: float,
        direction_agreement: float
    ) -> bool:
        """
        Determine if a finding is convergent.

        A finding is convergent if it has moderate+ replication AND statistical consistency.
        """
        return (
            convergence_strength in ("strong", "moderate") and
            effect_size_cv <= self.effect_cv_threshold and
            significance_agreement >= self.significance_threshold and
            direction_agreement >= 0.8  # Require 80% direction agreement
        )

    def _calculate_convergence_metrics(
        self,
        matched_findings: List[FindingMatch],
        n_runs: int,
        total_raw: int
    ) -> ConvergenceMetrics:
        """Calculate aggregate convergence metrics."""
        total_unique = len(matched_findings)

        # Count by replication
        findings_replicated_1 = sum(1 for m in matched_findings if m.replication_count == 1)
        findings_replicated_2_3 = sum(1 for m in matched_findings if 2 <= m.replication_count <= 3)
        findings_replicated_4_plus = sum(1 for m in matched_findings if m.replication_count >= 4)

        # Count by strength
        strong_count = sum(1 for m in matched_findings if m.convergence_strength == "strong")
        moderate_count = sum(1 for m in matched_findings if m.convergence_strength == "moderate")
        weak_count = sum(1 for m in matched_findings if m.convergence_strength == "weak")

        # Calculate averages
        if matched_findings:
            overall_rep_rate = np.mean([m.replication_rate for m in matched_findings])

            effect_cvs = [m.effect_size_cv for m in matched_findings if m.effect_sizes]
            avg_effect_cv = float(np.mean(effect_cvs)) if effect_cvs else 0.0

            sig_agreements = [m.significance_agreement for m in matched_findings if m.p_values]
            avg_sig_agreement = float(np.mean(sig_agreements)) if sig_agreements else 0.0

            dir_agreements = [m.direction_agreement for m in matched_findings if m.effect_sizes]
            avg_dir_agreement = float(np.mean(dir_agreements)) if dir_agreements else 0.0

            scholar_vars = [m.scholar_score_std ** 2 for m in matched_findings if m.scholar_scores]
            avg_scholar_var = float(np.mean(scholar_vars)) if scholar_vars else 0.0
        else:
            overall_rep_rate = 0.0
            avg_effect_cv = 0.0
            avg_sig_agreement = 0.0
            avg_dir_agreement = 0.0
            avg_scholar_var = 0.0

        return ConvergenceMetrics(
            total_runs=n_runs,
            total_unique_findings=total_unique,
            total_raw_findings=total_raw,
            findings_replicated_1=findings_replicated_1,
            findings_replicated_2_3=findings_replicated_2_3,
            findings_replicated_4_plus=findings_replicated_4_plus,
            overall_replication_rate=float(overall_rep_rate),
            strong_convergence_count=strong_count,
            moderate_convergence_count=moderate_count,
            weak_convergence_count=weak_count,
            avg_effect_size_cv=avg_effect_cv,
            avg_significance_agreement=avg_sig_agreement,
            avg_direction_agreement=avg_dir_agreement,
            avg_scholar_score_variance=avg_scholar_var,
            replication_threshold=self.replication_threshold,
            effect_cv_threshold=self.effect_cv_threshold,
            significance_threshold=self.significance_threshold
        )


# =============================================================================
# Ensemble Runner
# =============================================================================

class EnsembleRunner:
    """
    Runs N independent research workflows and analyzes convergence.

    Implements the paper's requirement to run each research question
    multiple times with different random seeds and compare results.

    Example:
        runner = EnsembleRunner(n_runs=5, seeds=[42, 43, 44, 45, 46])

        result = await runner.run(
            research_objective="Investigate KRAS mutations",
            num_cycles=5,
            tasks_per_cycle=10
        )

        print(f"Convergent findings: {len(result.convergent_findings)}")
    """

    DEFAULT_SEEDS = [42, 43, 44, 45, 46]
    DEFAULT_TEMPERATURE = 0.7

    def __init__(
        self,
        n_runs: int = 5,
        seeds: Optional[List[int]] = None,
        temperatures: Optional[List[float]] = None,
        anthropic_client=None,
        artifacts_base_dir: str = "artifacts",
        convergence_config: Optional[Dict] = None
    ):
        """
        Initialize EnsembleRunner.

        Args:
            n_runs: Number of independent runs (default 5 per paper)
            seeds: Random seeds for each run (default [42, 43, 44, 45, 46])
            temperatures: LLM temperatures for each run (default all 0.7)
            anthropic_client: Anthropic client for LLM calls
            artifacts_base_dir: Base directory for artifacts
            convergence_config: Override thresholds for ConvergenceAnalyzer
        """
        self.n_runs = n_runs
        self.anthropic_client = anthropic_client
        self.artifacts_base_dir = artifacts_base_dir

        # Generate seeds
        if seeds is not None:
            if len(seeds) != n_runs:
                raise ValueError(f"seeds must have {n_runs} elements, got {len(seeds)}")
            self.seeds = seeds
        else:
            self.seeds = [42 + i for i in range(n_runs)]

        # Generate temperatures
        if temperatures is not None:
            if len(temperatures) != n_runs:
                raise ValueError(f"temperatures must have {n_runs} elements, got {len(temperatures)}")
            self.temperatures = temperatures
        else:
            self.temperatures = [self.DEFAULT_TEMPERATURE] * n_runs

        # Create run configs
        self.run_configs = [
            RunConfig(
                run_id=f"run_{i}",
                seed=self.seeds[i],
                temperature=self.temperatures[i],
                run_index=i
            )
            for i in range(n_runs)
        ]

        # Initialize analyzer with config overrides
        analyzer_kwargs = convergence_config or {}
        self.analyzer = ConvergenceAnalyzer(**analyzer_kwargs)

        logger.info(
            f"Initialized EnsembleRunner with {n_runs} runs, "
            f"seeds={self.seeds}, temperatures={self.temperatures}"
        )

    async def run(
        self,
        research_objective: str,
        num_cycles: int = 5,
        tasks_per_cycle: int = 10
    ) -> EnsembleResult:
        """
        Run N independent research workflows and analyze convergence.

        Args:
            research_objective: The research question to investigate
            num_cycles: Number of research cycles per run
            tasks_per_cycle: Tasks to generate per cycle

        Returns:
            EnsembleResult with all findings and convergence analysis
        """
        import time
        start_time = time.time()
        start_timestamp = datetime.utcnow().isoformat()

        logger.info(
            f"Starting ensemble run: '{research_objective}' "
            f"({self.n_runs} runs, {num_cycles} cycles each)"
        )

        # Execute runs sequentially
        run_results = []
        run_findings = []

        for config in self.run_configs:
            logger.info(f"Executing {config.run_id} with seed={config.seed}")

            result, findings = await self._execute_single_run(
                config, research_objective, num_cycles, tasks_per_cycle
            )

            run_results.append(result)
            run_findings.append(findings)

            logger.info(f"Completed {config.run_id}: {len(findings)} findings")

        # Analyze convergence
        matched_findings, metrics = self.analyzer.analyze(run_findings, self.run_configs)

        # Separate convergent and non-convergent
        convergent = [m for m in matched_findings if m.is_convergent]
        non_convergent = [m for m in matched_findings if not m.is_convergent]

        end_time = time.time()
        end_timestamp = datetime.utcnow().isoformat()

        result = EnsembleResult(
            research_objective=research_objective,
            n_runs=self.n_runs,
            run_configs=self.run_configs,
            run_results=run_results,
            run_findings=run_findings,
            matched_findings=matched_findings,
            convergence_metrics=metrics,
            convergent_findings=convergent,
            non_convergent_findings=non_convergent,
            total_time_seconds=end_time - start_time,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp
        )

        logger.info(
            f"Ensemble complete: {len(convergent)} convergent, "
            f"{len(non_convergent)} non-convergent findings "
            f"({end_time - start_time:.1f}s)"
        )

        return result

    async def _execute_single_run(
        self,
        run_config: RunConfig,
        research_objective: str,
        num_cycles: int,
        tasks_per_cycle: int
    ) -> Tuple[Dict, List[Dict]]:
        """
        Execute a single research workflow with given config.

        Args:
            run_config: Configuration for this run
            research_objective: Research question
            num_cycles: Number of cycles
            tasks_per_cycle: Tasks per cycle

        Returns:
            Tuple of (run result dict, list of finding dicts)
        """
        from kosmos.workflow.research_loop import ResearchWorkflow
        from kosmos.safety.reproducibility import ReproducibilityManager

        # Set random seed
        repro_manager = ReproducibilityManager(default_seed=run_config.seed)
        repro_manager.set_seed(run_config.seed)

        # Create workflow with run-specific artifacts directory
        artifacts_dir = f"{self.artifacts_base_dir}/{run_config.run_id}"

        workflow = ResearchWorkflow(
            research_objective=research_objective,
            anthropic_client=self.anthropic_client,
            artifacts_dir=artifacts_dir,
            seed=run_config.seed,
            temperature=run_config.temperature
        )

        # Execute workflow
        result = await workflow.run(
            num_cycles=num_cycles,
            tasks_per_cycle=tasks_per_cycle
        )

        # Extract findings from state manager
        findings = []
        if hasattr(workflow, 'state_manager'):
            all_findings = workflow.state_manager.get_all_findings()
            findings = [f.to_dict() if hasattr(f, 'to_dict') else f for f in all_findings]

        return result, findings


# =============================================================================
# Convergence Reporter
# =============================================================================

class ConvergenceReporter:
    """
    Generates convergence reports in various formats.

    Example:
        reporter = ConvergenceReporter()
        markdown = reporter.generate_markdown_report(ensemble_result)
        json_data = reporter.generate_json_report(ensemble_result)
    """

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize ConvergenceReporter.

        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = output_dir

    def generate_markdown_report(self, result: EnsembleResult) -> str:
        """
        Generate markdown convergence report.

        Args:
            result: EnsembleResult from EnsembleRunner

        Returns:
            Markdown-formatted report string
        """
        metrics = result.convergence_metrics
        lines = []

        # Header
        lines.append("# Multi-Run Convergence Report\n")
        lines.append(f"**Research Objective**: {result.research_objective}\n")
        lines.append(f"**Date**: {result.end_timestamp[:10] if result.end_timestamp else 'N/A'}\n")
        lines.append(f"**Total Runs**: {result.n_runs}\n")
        lines.append(f"**Seeds Used**: {[c.seed for c in result.run_configs]}\n")
        lines.append(f"**Total Time**: {result.total_time_seconds:.1f}s\n")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary\n")
        if metrics:
            lines.append(f"- **Total Unique Findings**: {metrics.total_unique_findings} (from {metrics.total_raw_findings} raw findings)\n")
            lines.append(f"- **Strongly Convergent**: {metrics.strong_convergence_count} findings (4-5 runs)\n")
            lines.append(f"- **Moderately Convergent**: {metrics.moderate_convergence_count} findings (3 runs)\n")
            lines.append(f"- **Weakly Convergent**: {metrics.weak_convergence_count} findings (2 runs)\n")
            lines.append(f"- **Non-Replicating**: {metrics.findings_replicated_1} findings (1 run only)\n")
            lines.append("")
            lines.append(f"**Overall Replication Rate**: {metrics.overall_replication_rate:.1%}\n")
        lines.append("")

        # Strongly Convergent Findings
        strong_findings = result.get_strongly_convergent()
        if strong_findings:
            lines.append("---\n")
            lines.append("## Strongly Convergent Findings (High Confidence)\n")
            lines.append("These findings appeared in 4-5 of 5 runs and show statistical consistency.\n")

            for i, match in enumerate(strong_findings, 1):
                lines.append(f"\n### {i}. {match.canonical_summary}\n")
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                lines.append(f"| Replication | {match.replication_count}/{result.n_runs} runs |")
                if match.effect_sizes:
                    lines.append(f"| Effect Size | {match.effect_size_mean:.3f} +/- {match.effect_size_std:.3f} |")
                if match.p_values:
                    lines.append(f"| Significance Agreement | {match.significance_agreement:.1%} |")
                if match.effect_sizes:
                    lines.append(f"| Direction Agreement | {match.direction_agreement:.1%} |")
                if match.scholar_scores:
                    lines.append(f"| ScholarEval Score | {match.scholar_score_mean:.2f} +/- {match.scholar_score_std:.2f} |")
                lines.append("")
                lines.append(f"**Evidence**: Runs {match.run_indices}\n")

        # Moderately Convergent Findings
        moderate_findings = result.get_moderately_convergent()
        if moderate_findings:
            lines.append("---\n")
            lines.append("## Moderately Convergent Findings (Likely Real)\n")
            lines.append("These findings appeared in 3 of 5 runs.\n")

            for i, match in enumerate(moderate_findings, 1):
                lines.append(f"\n### {i}. {match.canonical_summary}\n")
                lines.append(f"- **Replication**: {match.replication_count}/{result.n_runs} runs ({match.run_indices})\n")
                if match.effect_sizes:
                    lines.append(f"- **Effect Size**: {match.effect_size_mean:.3f} +/- {match.effect_size_std:.3f}\n")
                if match.p_values:
                    lines.append(f"- **Significance Agreement**: {match.significance_agreement:.1%}\n")

        # Non-Replicating Findings
        non_rep_findings = result.get_non_replicating()
        if non_rep_findings:
            lines.append("---\n")
            lines.append("## Non-Replicating Findings (Low Confidence)\n")
            lines.append("These findings appeared in only 1 run and should be treated with caution.\n")
            lines.append("")

            for match in non_rep_findings[:10]:  # Limit to 10
                lines.append(f"- {match.canonical_summary} (run {match.run_indices[0]})\n")

            if len(non_rep_findings) > 10:
                lines.append(f"\n*...and {len(non_rep_findings) - 10} more non-replicating findings*\n")

        # Statistical Consistency Summary
        if metrics:
            lines.append("---\n")
            lines.append("## Statistical Consistency Summary\n")
            lines.append("| Metric | Mean |\n")
            lines.append("|--------|------|\n")
            lines.append(f"| Effect Size CV | {metrics.avg_effect_size_cv:.3f} |\n")
            lines.append(f"| Significance Agreement | {metrics.avg_significance_agreement:.1%} |\n")
            lines.append(f"| Direction Agreement | {metrics.avg_direction_agreement:.1%} |\n")
            lines.append(f"| ScholarEval Variance | {metrics.avg_scholar_score_variance:.3f} |\n")

        # Methodology
        lines.append("---\n")
        lines.append("## Methodology\n")
        lines.append(f"- **N={result.n_runs}** independent runs with seeds {[c.seed for c in result.run_configs]}\n")
        if metrics:
            lines.append(f"- **Similarity Threshold**: {self._get_sim_threshold(result)}\n")
            lines.append(f"- **Replication Threshold**: {metrics.replication_threshold:.0%}\n")
        lines.append("- **Matching Algorithm**: Semantic clustering with sentence-transformers\n")
        lines.append("- **Paper Reference**: Section 5 & 6.3 of Kosmos paper\n")

        return "\n".join(lines)

    def _get_sim_threshold(self, result: EnsembleResult) -> str:
        """Get similarity threshold (not stored in result, use default)."""
        return "0.80"

    def generate_json_report(self, result: EnsembleResult) -> Dict:
        """Generate JSON-serializable report."""
        return result.to_dict()

    def save_report(
        self,
        result: EnsembleResult,
        output_path: Optional[str] = None,
        format: str = "markdown"
    ) -> str:
        """
        Save report to file.

        Args:
            result: EnsembleResult from EnsembleRunner
            output_path: Output file path (auto-generated if None)
            format: "markdown" or "json"

        Returns:
            Path to saved file
        """
        import os
        import json

        os.makedirs(self.output_dir, exist_ok=True)

        if output_path is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            ext = "md" if format == "markdown" else "json"
            output_path = os.path.join(self.output_dir, f"convergence_report_{timestamp}.{ext}")

        if format == "markdown":
            content = self.generate_markdown_report(result)
            with open(output_path, 'w') as f:
                f.write(content)
        else:
            content = self.generate_json_report(result)
            with open(output_path, 'w') as f:
                json.dump(content, f, indent=2, default=str)

        logger.info(f"Saved convergence report to {output_path}")
        return output_path


# =============================================================================
# Convenience Functions
# =============================================================================

async def run_ensemble(
    research_objective: str,
    n_runs: int = 5,
    num_cycles: int = 5,
    tasks_per_cycle: int = 10,
    **kwargs
) -> EnsembleResult:
    """
    Convenience function to run an ensemble analysis.

    Args:
        research_objective: Research question
        n_runs: Number of runs (default 5)
        num_cycles: Cycles per run (default 5)
        tasks_per_cycle: Tasks per cycle (default 10)
        **kwargs: Additional EnsembleRunner arguments

    Returns:
        EnsembleResult
    """
    runner = EnsembleRunner(n_runs=n_runs, **kwargs)
    return await runner.run(research_objective, num_cycles, tasks_per_cycle)

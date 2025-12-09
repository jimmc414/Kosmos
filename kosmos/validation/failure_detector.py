"""
Failure Mode Detection for Autonomous Research Findings.

Implements detection for three critical failure modes:
1. Over-interpretation - Claims exceed statistical support
2. Invented Metrics - Metrics don't exist in data or standards
3. Rabbit Hole - Research drifted from original question

Paper Reference (Section 6.2):
"Common failure modes: Over-interpretation, Invented Metrics, Pipeline Pivots, Rabbit Holes"

Issue: #63 (GAP-010)
"""

import re
import time
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Set

logger = logging.getLogger(__name__)


@dataclass
class FailureModeScore:
    """
    Score for a single failure mode detection.

    Attributes:
        score: 0-1 scale (0 = no failure detected, 1 = severe failure)
        detected: True if score exceeds threshold
        confidence: 0-1 confidence in the detection
        evidence: Specific evidence for the failure
        recommendations: Actionable recommendations to fix
    """
    score: float
    detected: bool
    confidence: float
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FailureModeScore':
        """Create FailureModeScore from dictionary."""
        return cls(**data)


@dataclass
class FailureDetectionResult:
    """
    Result of failure mode detection for a finding.

    Three failure modes are detected:
    1. Over-interpretation: Claims exceed statistical support
    2. Invented metrics: Referenced metrics don't exist in data
    3. Rabbit hole: Finding drifted from original research question

    Attributes:
        over_interpretation: Score for over-interpretation detection
        invented_metrics: Score for invented metrics detection
        rabbit_hole: Score for rabbit hole detection
        overall_score: Weighted average (0-1, lower is better)
        passes_validation: True if all failure scores below thresholds
        finding_id: ID of the finding being validated
        hypothesis_id: Optional hypothesis ID
        research_question: Optional research question for context
        warnings: List of warning messages
        recommendations: Aggregated recommendations
        computation_time_seconds: Time taken for detection
    """
    over_interpretation: FailureModeScore
    invented_metrics: FailureModeScore
    rabbit_hole: FailureModeScore
    overall_score: float
    passes_validation: bool
    finding_id: str
    hypothesis_id: Optional[str] = None
    research_question: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    computation_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'over_interpretation': self.over_interpretation.to_dict(),
            'invented_metrics': self.invented_metrics.to_dict(),
            'rabbit_hole': self.rabbit_hole.to_dict(),
            'overall_score': self.overall_score,
            'passes_validation': self.passes_validation,
            'finding_id': self.finding_id,
            'hypothesis_id': self.hypothesis_id,
            'research_question': self.research_question,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'computation_time_seconds': self.computation_time_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FailureDetectionResult':
        """Create FailureDetectionResult from dictionary."""
        return cls(
            over_interpretation=FailureModeScore.from_dict(data['over_interpretation']),
            invented_metrics=FailureModeScore.from_dict(data['invented_metrics']),
            rabbit_hole=FailureModeScore.from_dict(data['rabbit_hole']),
            overall_score=data['overall_score'],
            passes_validation=data['passes_validation'],
            finding_id=data['finding_id'],
            hypothesis_id=data.get('hypothesis_id'),
            research_question=data.get('research_question'),
            warnings=data.get('warnings', []),
            recommendations=data.get('recommendations', []),
            computation_time_seconds=data.get('computation_time_seconds', 0.0),
        )

    @property
    def has_failures(self) -> bool:
        """Check if any failure mode was detected."""
        return (
            self.over_interpretation.detected or
            self.invented_metrics.detected or
            self.rabbit_hole.detected
        )

    def get_summary(self) -> str:
        """Generate human-readable summary."""
        if not self.has_failures:
            return "No failure modes detected."

        failures = []
        if self.over_interpretation.detected:
            failures.append(f"Over-interpretation (score={self.over_interpretation.score:.2f})")
        if self.invented_metrics.detected:
            failures.append(f"Invented metrics (score={self.invented_metrics.score:.2f})")
        if self.rabbit_hole.detected:
            failures.append(f"Rabbit hole (score={self.rabbit_hole.score:.2f})")

        return f"Failure modes detected: {', '.join(failures)}"


class FailureDetector:
    """
    Detects common failure modes in autonomous research findings.

    Paper Reference (Section 6.2):
    "Common failure modes: Over-interpretation, Invented Metrics,
    Pipeline Pivots, Rabbit Holes"

    Usage:
        detector = FailureDetector()
        result = detector.detect_failures(
            finding,
            context={
                'research_question': 'Original RQ',
                'dataset_schema': ['col1', 'col2'],
                'prior_findings': [...]
            }
        )

        if result.has_failures:
            print(f"Failures detected: {result.warnings}")
    """

    # Standard statistical metrics that are valid
    STANDARD_METRICS: Set[str] = {
        # Core statistics
        'p_value', 'p', 'pvalue',
        'effect_size', 'cohens_d', 'd', 'r', 'r_squared', 'r2',
        'correlation', 'rho', 'tau',

        # Test statistics
        't_statistic', 't', 'f_statistic', 'f', 'chi2', 'chi_squared',
        'z', 'z_score', 'statistic',

        # Sample statistics
        'n', 'sample_size', 'df', 'degrees_of_freedom',
        'mean', 'median', 'std', 'sd', 'variance', 'var',
        'se', 'standard_error', 'ci', 'confidence_interval',
        'ci_lower', 'ci_upper', 'min', 'max', 'range', 'iqr',

        # Effect measures
        'odds_ratio', 'or', 'hazard_ratio', 'hr', 'relative_risk', 'rr',
        'auc', 'accuracy', 'precision', 'recall', 'f1', 'sensitivity',
        'specificity', 'ppv', 'npv',

        # Multiple testing corrections
        'fdr', 'q_value', 'bonferroni', 'holm', 'benjamini_hochberg',
        'adjusted_p', 'corrected_p',

        # Common analysis-specific
        'fold_change', 'log_fold_change', 'logfc', 'beta', 'coefficient',
        'slope', 'intercept', 'residual',
    }

    # Words indicating strong claims
    STRONG_CLAIM_WORDS: Set[str] = {
        'proves', 'prove', 'proven', 'proof',
        'demonstrates', 'demonstrate', 'demonstrated',
        'confirms', 'confirm', 'confirmed', 'confirmation',
        'establishes', 'establish', 'established',
        'definitively', 'definitive', 'definite',
        'conclusively', 'conclusive', 'conclusion',
        'clearly', 'clear', 'obvious', 'obviously',
        'significantly', 'significant',  # Can be appropriate with p<0.05
        'strongly', 'strong',
        'undoubtedly', 'undoubted', 'unquestionably',
        'certainly', 'certain', 'sure', 'surely',
        'shows', 'show', 'shown',  # Moderate strength
        'reveals', 'reveal', 'revealed',
        'identifies', 'identify', 'identified',
    }

    # Words indicating hedged/qualified claims
    HEDGED_CLAIM_WORDS: Set[str] = {
        'suggests', 'suggest', 'suggested', 'suggestion',
        'indicates', 'indicate', 'indicated', 'indication',
        'may', 'might', 'could', 'would',
        'potentially', 'potential',
        'possibly', 'possible', 'possibility',
        'appears', 'appear', 'appeared',
        'seems', 'seem', 'seemed',
        'likely', 'unlikely',
        'tends', 'tend', 'tendency',
        'hints', 'hint', 'hinted',
        'implies', 'imply', 'implied', 'implication',
        'preliminary', 'initial', 'tentative',
        'exploratory', 'hypothesis',
        'trend', 'trending',
        'consistent', 'compatible',  # Neutral
        'associated', 'association', 'correlates', 'correlated',
    }

    # Weights for overall score calculation
    FAILURE_WEIGHTS = {
        'over_interpretation': 0.4,
        'invented_metrics': 0.3,
        'rabbit_hole': 0.3,
    }

    def __init__(
        self,
        anthropic_client=None,
        over_interpretation_threshold: float = 0.6,
        invented_metrics_threshold: float = 0.5,
        rabbit_hole_threshold: float = 0.7,
        similarity_threshold: float = 0.3,
        model: str = None,
    ):
        """
        Initialize FailureDetector.

        Args:
            anthropic_client: Optional Anthropic client for sophisticated LLM-based detection
            over_interpretation_threshold: Score above which over-interpretation flagged (default 0.6)
            invented_metrics_threshold: Score above which invented metrics flagged (default 0.5)
            rabbit_hole_threshold: Score above which rabbit hole flagged (default 0.7)
            similarity_threshold: Minimum semantic similarity to RQ (default 0.3)
            model: Model for LLM-based detection (if client provided)
        """
        self.client = anthropic_client
        self.over_interpretation_threshold = over_interpretation_threshold
        self.invented_metrics_threshold = invented_metrics_threshold
        self.rabbit_hole_threshold = rabbit_hole_threshold
        self.similarity_threshold = similarity_threshold
        self.model = model

    def detect_failures(
        self,
        finding: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> FailureDetectionResult:
        """
        Detect all failure modes in a finding.

        Args:
            finding: Finding dictionary with:
                - summary: High-level claims
                - statistics: Dict with p_value, effect_size, etc.
                - interpretation: Detailed interpretation
                - finding_id: Optional ID
                - hypothesis_id: Optional link to hypothesis
            context: Optional context dictionary with:
                - research_question: Original research question
                - dataset_schema: List of column/metric names available in data
                - prior_findings: List of prior findings
                - hypothesis: Hypothesis object/dict with generation info

        Returns:
            FailureDetectionResult with scores for each failure mode
        """
        start_time = time.time()

        context = context or {}
        research_question = context.get('research_question', '')
        dataset_schema = context.get('dataset_schema')
        hypothesis = context.get('hypothesis')

        # Detect each failure mode
        over_interp = self.detect_over_interpretation(finding)
        invented = self.detect_invented_metrics(finding, dataset_schema)
        rabbit_hole = self.detect_rabbit_hole(finding, research_question, hypothesis)

        # Calculate overall score (weighted average)
        overall = (
            self.FAILURE_WEIGHTS['over_interpretation'] * over_interp.score +
            self.FAILURE_WEIGHTS['invented_metrics'] * invented.score +
            self.FAILURE_WEIGHTS['rabbit_hole'] * rabbit_hole.score
        )

        # Collect warnings
        warnings = []
        if over_interp.detected:
            warnings.append("Over-interpretation detected: claims exceed statistical support")
        if invented.detected:
            warnings.append("Invented metrics detected: undefined metrics referenced")
        if rabbit_hole.detected:
            warnings.append("Rabbit hole detected: finding drifted from research question")

        # Collect recommendations (deduplicated)
        recommendations = list(dict.fromkeys(
            over_interp.recommendations +
            invented.recommendations +
            rabbit_hole.recommendations
        ))

        # Determine if passes validation
        passes = not any([
            over_interp.detected,
            invented.detected,
            rabbit_hole.detected
        ])

        return FailureDetectionResult(
            over_interpretation=over_interp,
            invented_metrics=invented,
            rabbit_hole=rabbit_hole,
            overall_score=overall,
            passes_validation=passes,
            finding_id=finding.get('finding_id', 'unknown'),
            hypothesis_id=finding.get('hypothesis_id'),
            research_question=research_question if research_question else None,
            warnings=warnings,
            recommendations=recommendations,
            computation_time_seconds=time.time() - start_time,
        )

    def detect_over_interpretation(
        self,
        finding: Dict[str, Any]
    ) -> FailureModeScore:
        """
        Detect when claims exceed statistical support.

        Algorithm:
        1. Extract claim strength from interpretation text
        2. Extract statistical strength from statistics dict
        3. Compare: strong claims need strong statistics
        4. Flag when claim_strength >> statistical_strength

        Args:
            finding: Finding dictionary with interpretation and statistics

        Returns:
            FailureModeScore with over-interpretation assessment
        """
        interpretation = finding.get('interpretation', '') or ''
        summary = finding.get('summary', '') or ''
        statistics = finding.get('statistics', {}) or {}

        # Combine text for analysis
        text = f"{interpretation} {summary}"

        # Step 1: Measure claim strength (0-1)
        claim_strength = self._measure_claim_strength(text)

        # Step 2: Measure statistical strength (0-1)
        stat_strength = self._measure_statistical_strength(statistics)

        # Step 3: Calculate over-interpretation score
        # High claim + low stats = over-interpretation
        if stat_strength > 0:
            # Claims should not exceed evidence by too much
            over_interp_score = max(0.0, claim_strength - stat_strength)
        else:
            # No statistics = any non-hedged claim is potentially over-interpretation
            over_interp_score = claim_strength * 0.8  # Slightly reduce if no stats to compare

        # Step 4: Handle edge cases
        if not text.strip():
            # No interpretation = no over-interpretation
            over_interp_score = 0.0

        # Generate evidence
        evidence = self._generate_over_interp_evidence(
            claim_strength, stat_strength, text, statistics
        )

        # Generate recommendations
        recommendations = []
        if over_interp_score >= self.over_interpretation_threshold:
            recommendations.append("Tone down claims to match statistical evidence")
            recommendations.append("Use hedging language: 'suggests', 'indicates', 'may'")
            if stat_strength < 0.5:
                recommendations.append("Strengthen statistical support with additional tests")

        return FailureModeScore(
            score=min(1.0, over_interp_score),
            detected=over_interp_score >= self.over_interpretation_threshold,
            confidence=0.8 if statistics else 0.6,
            evidence=evidence,
            recommendations=recommendations,
        )

    def _measure_claim_strength(self, text: str) -> float:
        """
        Measure strength of claims in interpretation text.

        Returns 0-1 score (1 = very strong claims).
        """
        if not text:
            return 0.0

        text_lower = text.lower()

        # Count strong vs hedged indicators
        strong_count = sum(1 for w in self.STRONG_CLAIM_WORDS if w in text_lower)
        hedged_count = sum(1 for w in self.HEDGED_CLAIM_WORDS if w in text_lower)

        # Special handling for statistical significance language
        # "significantly" is appropriate if p < 0.05, but we don't know that here
        # So we count it as moderate strength

        # Calculate ratio
        total = strong_count + hedged_count
        if total == 0:
            # No claim words found - assume moderate/neutral
            return 0.4

        # Ratio of strong to total claim words
        strength_ratio = strong_count / total

        # Also consider absolute count of strong claims
        # Many strong claims without hedging = higher score
        if strong_count >= 3 and hedged_count == 0:
            strength_ratio = min(1.0, strength_ratio + 0.2)

        return min(1.0, strength_ratio)

    def _measure_statistical_strength(self, statistics: Dict[str, Any]) -> float:
        """
        Measure strength of statistical evidence.

        Returns 0-1 score (1 = very strong evidence).
        """
        if not statistics:
            return 0.0

        scores = []
        weights = []

        # P-value contribution (most important)
        p_value = statistics.get('p_value') or statistics.get('p')
        if p_value is not None:
            try:
                p = float(p_value)
                if p < 0.001:
                    scores.append(1.0)
                elif p < 0.01:
                    scores.append(0.85)
                elif p < 0.05:
                    scores.append(0.65)
                elif p < 0.1:
                    scores.append(0.4)
                else:
                    scores.append(0.2)
                weights.append(0.4)
            except (ValueError, TypeError):
                pass

        # Effect size contribution
        effect_size = (
            statistics.get('effect_size') or
            statistics.get('cohens_d') or
            statistics.get('d')
        )
        if effect_size is not None:
            try:
                es = abs(float(effect_size))
                if es >= 0.8:
                    scores.append(1.0)
                elif es >= 0.5:
                    scores.append(0.75)
                elif es >= 0.2:
                    scores.append(0.5)
                else:
                    scores.append(0.25)
                weights.append(0.35)
            except (ValueError, TypeError):
                pass

        # Sample size contribution
        n = statistics.get('sample_size') or statistics.get('n')
        if n is not None:
            try:
                sample = int(n)
                if sample >= 200:
                    scores.append(0.9)
                elif sample >= 100:
                    scores.append(0.75)
                elif sample >= 50:
                    scores.append(0.6)
                elif sample >= 30:
                    scores.append(0.45)
                else:
                    scores.append(0.3)
                weights.append(0.25)
            except (ValueError, TypeError):
                pass

        # Calculate weighted average
        if not scores:
            return 0.0

        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        return sum(s * w for s, w in zip(scores, weights)) / total_weight

    def _generate_over_interp_evidence(
        self,
        claim_strength: float,
        stat_strength: float,
        text: str,
        statistics: Dict[str, Any]
    ) -> List[str]:
        """Generate evidence for over-interpretation assessment."""
        evidence = []

        evidence.append(f"Claim strength: {claim_strength:.2f}")
        evidence.append(f"Statistical strength: {stat_strength:.2f}")

        if claim_strength > stat_strength:
            gap = claim_strength - stat_strength
            evidence.append(f"Claim-evidence gap: {gap:.2f}")

        # Identify specific strong words used
        text_lower = text.lower()
        strong_used = [w for w in self.STRONG_CLAIM_WORDS if w in text_lower]
        if strong_used:
            evidence.append(f"Strong claim words: {', '.join(strong_used[:5])}")

        # Check p-value
        p_value = statistics.get('p_value') or statistics.get('p')
        if p_value is not None:
            try:
                p = float(p_value)
                if p >= 0.05:
                    evidence.append(f"p-value ({p:.4f}) not significant at 0.05 level")
            except (ValueError, TypeError):
                pass

        return evidence

    def detect_invented_metrics(
        self,
        finding: Dict[str, Any],
        dataset_schema: Optional[List[str]] = None
    ) -> FailureModeScore:
        """
        Detect metrics that don't exist in data or standard statistics.

        Algorithm:
        1. Extract all metric names from interpretation and statistics
        2. Check against STANDARD_METRICS set
        3. Check against dataset_schema (column names)
        4. Flag metrics that appear nowhere

        Args:
            finding: Finding dictionary
            dataset_schema: Optional list of column/metric names available in the dataset

        Returns:
            FailureModeScore with invented metrics assessment
        """
        interpretation = finding.get('interpretation', '') or ''
        summary = finding.get('summary', '') or ''
        statistics = finding.get('statistics', {}) or {}

        # Combine text for analysis
        text = f"{interpretation} {summary}"

        # Step 1: Extract mentioned metrics from text
        mentioned_metrics = self._extract_metrics_from_text(text)

        # Step 2: Build valid metrics set
        valid_metrics = self.STANDARD_METRICS.copy()

        # Add dataset schema if provided
        if dataset_schema:
            valid_metrics.update(set(col.lower().replace(' ', '_') for col in dataset_schema))

        # Add metrics from statistics dict (they're at least defined somewhere)
        if statistics:
            valid_metrics.update(set(k.lower() for k in statistics.keys()))

        # Step 3: Find potentially invented metrics
        invented = []
        for metric in mentioned_metrics:
            metric_lower = metric.lower()
            # Check if metric or close variant exists in valid set
            is_valid = False
            for valid in valid_metrics:
                # Exact match
                if metric_lower == valid:
                    is_valid = True
                    break
                # Substring match - but only if the valid metric is substantial (>2 chars)
                # This prevents single-letter metrics like 'n', 't', 'r' from matching everything
                if len(valid) > 2 and (metric_lower in valid or valid in metric_lower):
                    is_valid = True
                    break
                # Variant match using custom logic
                if self._is_metric_variant(metric_lower, valid):
                    is_valid = True
                    break
            if not is_valid:
                invented.append(metric)

        # Step 4: Calculate score
        if not mentioned_metrics:
            score = 0.0
        else:
            score = len(invented) / len(mentioned_metrics)

        # Generate evidence
        evidence = []
        if invented:
            evidence.append(f"Potentially undefined metrics: {', '.join(invented)}")
        evidence.append(f"Total metrics mentioned: {len(mentioned_metrics)}")
        evidence.append(f"Unrecognized metrics: {len(invented)}")

        # Generate recommendations
        recommendations = []
        if score >= self.invented_metrics_threshold:
            recommendations.append(f"Define or remove these metrics: {', '.join(invented)}")
            recommendations.append("Use standard statistical terminology")
            recommendations.append("Verify metrics exist in the source data")

        return FailureModeScore(
            score=min(1.0, score),
            detected=score >= self.invented_metrics_threshold,
            confidence=0.7,
            evidence=evidence,
            recommendations=recommendations,
        )

    def _extract_metrics_from_text(self, text: str) -> List[str]:
        """
        Extract metric-like terms from text.

        Looks for patterns like:
        - "metric = 0.85"
        - "metric: 0.5"
        - "(metric = 0.95)"
        - "the metric was 0.7"
        - "metric_name = 0.9" (compound names with underscores)
        """
        if not text:
            return []

        metrics = set()

        # Pattern 1: metric = value or metric: value (supports compound names like metric_name)
        # Matches: synergy_index = 0.95, p_value = 0.01
        pattern1 = r'([a-zA-Z][a-zA-Z0-9_]*)\s*[=:]\s*[-+]?[0-9]*\.?[0-9]+'
        matches1 = re.findall(pattern1, text)
        metrics.update(matches1)

        # Pattern 2: metric in parentheses with value
        pattern2 = r'\(([a-zA-Z][a-zA-Z0-9_]*)\s*[=:]\s*[-+]?[0-9]*\.?[0-9]+\)'
        matches2 = re.findall(pattern2, text)
        metrics.update(matches2)

        # Pattern 3: "the <metric> was/is <value>"
        pattern3 = r'the\s+([a-zA-Z][a-zA-Z0-9_]*)\s+(?:was|is|were|are)\s+[-+]?[0-9]*\.?[0-9]+'
        matches3 = re.findall(pattern3, text.lower())
        metrics.update(matches3)

        # Filter out common non-metric words that might match patterns
        non_metrics = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall',
            'this', 'that', 'these', 'those', 'it', 'its',
            'we', 'our', 'they', 'their', 'i', 'my',
            'figure', 'table', 'equation', 'section', 'chapter',
            'group', 'sample', 'subject', 'patient', 'participant',
            'year', 'month', 'day', 'time', 'date',
            'gene', 'protein', 'cell', 'tissue', 'organ',
            # Common verbs and adjectives that might appear before numbers
            'value', 'score', 'rate', 'level', 'count', 'number',
            'result', 'finding', 'total', 'average', 'mean',
        }

        # Return filtered list
        return [m for m in metrics if m.lower() not in non_metrics]

    def _is_metric_variant(self, metric: str, standard: str) -> bool:
        """
        Check if metric is a variant of a standard metric name.

        E.g., 'p_val' is a variant of 'p_value'
        """
        # Remove common separators
        m_clean = metric.replace('_', '').replace('-', '').replace(' ', '').lower()
        s_clean = standard.replace('_', '').replace('-', '').replace(' ', '').lower()

        # Only match if the shorter one is a substantial portion of the longer
        # This prevents "novelty" from matching "n" or random substrings
        min_len = min(len(m_clean), len(s_clean))
        if min_len < 3:
            # Too short - require exact match
            return m_clean == s_clean

        # For longer strings, check if one is a prefix/suffix of the other
        # or if they share significant overlap (>70% of shorter string)
        if m_clean.startswith(s_clean) or m_clean.endswith(s_clean):
            return True
        if s_clean.startswith(m_clean) or s_clean.endswith(m_clean):
            return True

        # Check character overlap ratio
        common_chars = set(m_clean) & set(s_clean)
        overlap_ratio = len(common_chars) / min_len

        # Require very high overlap (>80%) for it to count as a variant
        # AND the lengths must be similar (within 2x)
        max_len = max(len(m_clean), len(s_clean))
        length_ratio = min_len / max_len if max_len > 0 else 0

        return overlap_ratio > 0.8 and length_ratio > 0.5

    def detect_rabbit_hole(
        self,
        finding: Dict[str, Any],
        research_question: str,
        hypothesis: Optional[Dict[str, Any]] = None
    ) -> FailureModeScore:
        """
        Detect when finding drifts far from original research question.

        Algorithm:
        1. Compute semantic similarity between finding and RQ
        2. Check hypothesis generation (> 3-4 refinements = concerning)
        3. Check if finding addresses RQ at all
        4. Flag if similarity < threshold or generation too high

        Args:
            finding: Finding dictionary
            research_question: Original research question
            hypothesis: Optional hypothesis with generation/evolution_history

        Returns:
            FailureModeScore with rabbit hole assessment
        """
        summary = finding.get('summary', '') or ''
        interpretation = finding.get('interpretation', '') or ''
        finding_text = f"{summary} {interpretation}"

        # Handle empty research question
        if not research_question or not research_question.strip():
            # Without RQ, we can't detect rabbit holes meaningfully
            return FailureModeScore(
                score=0.0,
                detected=False,
                confidence=0.3,
                evidence=["No research question provided for comparison"],
                recommendations=[],
            )

        # Step 1: Semantic similarity to RQ
        similarity = self._compute_relevance_similarity(finding_text, research_question)

        # Step 2: Check hypothesis generation depth
        generation_penalty = 0.0
        generation = 1
        if hypothesis:
            generation = hypothesis.get('generation', 1)
            refinement_count = hypothesis.get('refinement_count', 0)

            # Use max of generation and refinement_count
            depth = max(generation, refinement_count)

            if depth > 4:
                generation_penalty = 0.25  # High penalty for deep evolution
            elif depth > 3:
                generation_penalty = 0.15
            elif depth > 2:
                generation_penalty = 0.05

        # Step 3: Calculate rabbit hole score (inverse of relevance)
        base_score = 1.0 - similarity
        rabbit_hole_score = min(1.0, base_score + generation_penalty)

        # Generate evidence
        evidence = []
        evidence.append(f"Relevance to RQ: {similarity:.2f}")
        if generation_penalty > 0:
            evidence.append(f"Hypothesis generation: {generation} (penalty: {generation_penalty:.2f})")
        if similarity < self.similarity_threshold:
            evidence.append(f"Low relevance detected (threshold: {self.similarity_threshold})")

        # Generate recommendations
        recommendations = []
        if rabbit_hole_score >= self.rabbit_hole_threshold:
            recommendations.append("Refocus analysis on original research question")
            recommendations.append("Review hypothesis evolution for drift")
            if generation > 3:
                recommendations.append("Consider starting fresh from original hypothesis")

        return FailureModeScore(
            score=rabbit_hole_score,
            detected=rabbit_hole_score >= self.rabbit_hole_threshold,
            confidence=0.6,  # Keyword similarity is less reliable than LLM
            evidence=evidence,
            recommendations=recommendations,
        )

    def _compute_relevance_similarity(
        self,
        finding_text: str,
        research_question: str
    ) -> float:
        """
        Compute semantic similarity between finding and research question.

        Uses keyword-based Jaccard similarity as fallback.
        Could be extended to use embeddings if available.
        """
        # Try embeddings if available (future enhancement)
        # For now, use keyword similarity
        return self._keyword_similarity(finding_text, research_question)

    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """
        Compute keyword-based Jaccard similarity between two texts.

        Removes stopwords and computes intersection / union of keywords.
        """
        if not text1 or not text2:
            return 0.0

        # Stopwords to remove
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to',
            'this', 'that', 'these', 'those', 'it', 'its', 'itself',
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves',
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
            'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'whose',
            'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
            'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
            'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 's', 't', 'just', 'don', 'now',
        }

        # Tokenize and clean
        def tokenize(text: str) -> Set[str]:
            # Convert to lowercase and split on non-alphanumeric
            words = re.findall(r'\b[a-z]+\b', text.lower())
            # Remove stopwords and short words
            return {w for w in words if w not in stopwords and len(w) > 2}

        words1 = tokenize(text1)
        words2 = tokenize(text2)

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def batch_detect(
        self,
        findings: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> List[FailureDetectionResult]:
        """
        Detect failures in multiple findings.

        Args:
            findings: List of finding dictionaries
            context: Optional shared context for all findings

        Returns:
            List of FailureDetectionResult for each finding
        """
        return [self.detect_failures(f, context) for f in findings]

    def get_failure_statistics(
        self,
        results: List[FailureDetectionResult]
    ) -> Dict[str, Any]:
        """
        Compute aggregate statistics from batch detection results.

        Args:
            results: List of FailureDetectionResult from batch_detect

        Returns:
            Dictionary with aggregate statistics
        """
        if not results:
            return {
                'count': 0,
                'pass_rate': 0.0,
                'failure_rates': {},
            }

        n = len(results)
        passed = sum(1 for r in results if r.passes_validation)

        over_interp_failures = sum(1 for r in results if r.over_interpretation.detected)
        invented_failures = sum(1 for r in results if r.invented_metrics.detected)
        rabbit_hole_failures = sum(1 for r in results if r.rabbit_hole.detected)

        return {
            'count': n,
            'passed': passed,
            'failed': n - passed,
            'pass_rate': passed / n,
            'failure_rates': {
                'over_interpretation': over_interp_failures / n,
                'invented_metrics': invented_failures / n,
                'rabbit_hole': rabbit_hole_failures / n,
            },
            'avg_scores': {
                'overall': sum(r.overall_score for r in results) / n,
                'over_interpretation': sum(r.over_interpretation.score for r in results) / n,
                'invented_metrics': sum(r.invented_metrics.score for r in results) / n,
                'rabbit_hole': sum(r.rabbit_hole.score for r in results) / n,
            },
        }

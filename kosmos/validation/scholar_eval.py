"""
ScholarEval Validation Framework.

Implements peer-review style validation for scientific discoveries
using LLM-based 8-dimension scoring.

Design Philosophy:
- Multi-dimensional scoring catches different failure modes
- Weighted scoring prioritizes scientific rigor
- Minimum thresholds prevent catastrophic failures
- Actionable feedback explains rejections

Performance Target: ~75% validation rate
"""

import json
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict

from kosmos.config import _DEFAULT_CLAUDE_SONNET_MODEL
from kosmos.core.utils.json_parser import parse_json_response, JSONParseError
from kosmos.validation.null_model import NullModelValidator, NullModelResult

logger = logging.getLogger(__name__)


@dataclass
class ScholarEvalScore:
    """
    Container for ScholarEval scores across 8 dimensions.

    Dimensions (all 0.0-1.0):
    - novelty: Is this finding new?
    - rigor: Are methods scientifically sound?
    - clarity: Is finding clearly stated?
    - reproducibility: Can others reproduce this?
    - impact: How important is this?
    - coherence: Does it fit existing knowledge?
    - limitations: Are limitations acknowledged?
    - ethics: Are ethical concerns addressed?
    """
    novelty: float
    rigor: float
    clarity: float
    reproducibility: float
    impact: float
    coherence: float
    limitations: float
    ethics: float
    overall_score: float
    passes_threshold: bool
    feedback: str
    reasoning: Optional[str] = None
    # Issue #70: Null model validation results
    null_model_result: Optional[Dict] = None  # Null model validation: {p_value, persists_in_noise, etc.}
    statistical_validity: Optional[float] = None  # 0-1 score from null model (1.0 if passes)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ScholarEvalScore':
        """Create ScholarEvalScore from dictionary."""
        return cls(**data)


class ScholarEvalValidator:
    """
    ScholarEval validation for scientific discoveries.

    Uses LLM to score findings on 8 dimensions with weighted aggregation.

    Scoring Formula:
        overall_score = (
            0.25 * rigor +           # Heavily weight rigor
            0.20 * impact +          # Importance matters
            0.15 * novelty +         # Prefer new findings
            0.15 * reproducibility + # Must be reproducible
            0.10 * clarity +         # Clear communication
            0.10 * coherence +       # Fits existing knowledge
            0.03 * limitations +     # Acknowledge weaknesses
            0.02 * ethics            # Ethical considerations
        )

    Approval Criteria:
        - overall_score >= threshold (default: 0.75)
        - rigor >= min_rigor_score (default: 0.70)
    """

    # Dimension weights (must sum to 1.0)
    DIMENSION_WEIGHTS = {
        'rigor': 0.25,
        'impact': 0.20,
        'novelty': 0.15,
        'reproducibility': 0.15,
        'clarity': 0.10,
        'coherence': 0.10,
        'limitations': 0.03,
        'ethics': 0.02
    }

    def __init__(
        self,
        anthropic_client=None,
        llm_client=None,
        threshold: float = 0.75,
        min_rigor_score: float = 0.70,
        model: str = _DEFAULT_CLAUDE_SONNET_MODEL,
        temperature: float = 0.3,
        allow_mock: bool = False,
        run_null_model: bool = True
    ):
        """
        Initialize ScholarEval validator.

        Args:
            anthropic_client: Anthropic SDK client (messages.create) for LLM scoring
            llm_client: Any kosmos provider client (generate); takes precedence
                over anthropic_client
            threshold: Minimum overall score for approval (default: 0.75)
            min_rigor_score: Minimum rigor score required (default: 0.70)
            model: Model to use for scoring (anthropic_client only)
            temperature: LLM temperature for evaluation
            allow_mock: When True, a missing client or a failed evaluation returns
                the optimistic mock score; when False (default) it returns a
                failing score whose feedback starts with "evaluation_error"
            run_null_model: Run the parametric null model on the finding's
                statistics; callers that run a real permutation test pass False
        """
        self.client = anthropic_client
        self.llm_client = llm_client
        self.threshold = threshold
        self.min_rigor_score = min_rigor_score
        self.model = model
        self._temperature = temperature
        self.allow_mock = allow_mock
        self.run_null_model = run_null_model

    def _error_score(self, error: Any) -> ScholarEvalScore:
        """A failing score that records why the evaluation could not run."""
        return ScholarEvalScore(
            novelty=0.0, rigor=0.0, clarity=0.0, reproducibility=0.0,
            impact=0.0, coherence=0.0, limitations=0.0, ethics=0.0,
            overall_score=0.0,
            passes_threshold=False,
            feedback=f"evaluation_error: {error}",
        )

    def _query_llm(self, prompt: str) -> str:
        """Send the scoring prompt to whichever client is configured."""
        if self.llm_client is not None:
            response = self.llm_client.generate(
                prompt, max_tokens=1500, temperature=self._temperature
            )
            return response.content if hasattr(response, 'content') else str(response)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature
        )
        return response.content[0].text

    def evaluate_finding(self, finding: Dict) -> ScholarEvalScore:
        """
        Evaluate finding using 8-dimension framework.

        Args:
            finding: Dictionary with summary, statistics, methods, interpretation

        Returns:
            ScholarEvalScore with 8 dimension scores + overall + approval
        """
        # Without an LLM client only tests and smoke scripts may use mock scoring
        if self.client is None and self.llm_client is None:
            if self.allow_mock:
                return self._mock_evaluation(finding)
            return self._error_score("no LLM client configured")

        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(finding)

        try:
            # Query LLM for scores; strict parse so a garbled reply is an error, not 0.5s
            scores = self._parse_llm_response(self._query_llm(prompt), strict=not self.allow_mock)

            # Calculate weighted overall score
            overall = self._calculate_overall_score(scores)

            # Check approval thresholds
            passes = (
                overall >= self.threshold
                and scores.get('rigor', 0) >= self.min_rigor_score
            )

            # Generate feedback
            feedback = self._generate_feedback(scores, passes, finding)

            # Issue #70: Run null model validation for statistical grounding
            null_result = None
            statistical_validity = None
            if self.run_null_model and finding.get('statistics'):
                try:
                    null_validator = NullModelValidator()
                    null_result_obj = null_validator.validate_finding(finding)
                    null_result = null_result_obj.to_dict()
                    statistical_validity = 1.0 if null_result_obj.passes_null_test else 0.0

                    # Adjust overall score if finding fails null model test
                    if null_result_obj.persists_in_noise:
                        overall *= 0.5  # Penalize findings that persist in noise
                        feedback += "\n\nWARNING: Finding persists in randomized data (potential false positive)"
                        passes = False  # Fail validation if persists in noise
                except Exception as e:
                    logger.warning(f"Null model validation failed: {e}")
                    # Continue with LLM-only validation

            return ScholarEvalScore(
                novelty=scores.get('novelty', 0.5),
                rigor=scores.get('rigor', 0.5),
                clarity=scores.get('clarity', 0.5),
                reproducibility=scores.get('reproducibility', 0.5),
                impact=scores.get('impact', 0.5),
                coherence=scores.get('coherence', 0.5),
                limitations=scores.get('limitations', 0.5),
                ethics=scores.get('ethics', 0.5),
                overall_score=overall,
                passes_threshold=passes,
                feedback=feedback,
                reasoning=scores.get('reasoning', ''),
                null_model_result=null_result,
                statistical_validity=statistical_validity
            )

        except Exception as e:
            logger.error(f"ScholarEval evaluation failed: {e}")
            if self.allow_mock:
                return self._mock_evaluation(finding)
            return self._error_score(e)

    def _build_evaluation_prompt(self, finding: Dict) -> str:
        """
        Build prompt for ScholarEval scoring.

        Args:
            finding: Finding dictionary

        Returns:
            Formatted prompt string
        """
        summary = finding.get('summary', 'No summary provided')
        statistics = finding.get('statistics', {})
        methods = finding.get('methods', 'Not specified')
        interpretation = finding.get('interpretation', 'Not provided')

        # Format statistics nicely
        stats_str = json.dumps(statistics, indent=2) if statistics else "No statistics"

        return f"""You are a scientific peer reviewer evaluating a research finding.

**Finding**:
{summary}

**Statistics**:
{stats_str}

**Methods**:
{methods}

**Interpretation**:
{interpretation}

**Evaluate on 8 dimensions** (score 0.0-1.0 for each):

1. **Novelty**: Is this finding new? Does it advance knowledge?
   - 1.0: Highly novel, advances field significantly
   - 0.5: Some novelty, incremental advance
   - 0.0: Not novel, already known

2. **Rigor**: Are methods scientifically sound?
   - 1.0: Rigorous methods, appropriate statistics
   - 0.5: Adequate but some concerns
   - 0.0: Flawed methods, invalid conclusions

3. **Clarity**: Is finding clearly stated?
   - 1.0: Crystal clear, well-articulated
   - 0.5: Somewhat clear, room for improvement
   - 0.0: Unclear, confusing

4. **Reproducibility**: Can others reproduce this?
   - 1.0: Fully reproducible, complete methods
   - 0.5: Partially reproducible, some gaps
   - 0.0: Not reproducible, insufficient detail

5. **Impact**: How important is this finding?
   - 1.0: High impact, significant implications
   - 0.5: Moderate impact
   - 0.0: Low impact, limited implications

6. **Coherence**: Does it fit existing knowledge?
   - 1.0: Fully coherent, well-integrated
   - 0.5: Mostly coherent, some tensions
   - 0.0: Contradicts without explanation

7. **Limitations**: Are limitations acknowledged?
   - 1.0: Limitations clearly stated
   - 0.5: Some limitations noted
   - 0.0: No limitations mentioned

8. **Ethics**: Ethical considerations addressed?
   - 1.0: Fully addressed
   - 0.5: Partially addressed
   - 0.0: Not addressed (or not applicable: score 0.7)

**Output Format** (JSON):
{{
  "novelty": <0.0-1.0>,
  "rigor": <0.0-1.0>,
  "clarity": <0.0-1.0>,
  "reproducibility": <0.0-1.0>,
  "impact": <0.0-1.0>,
  "coherence": <0.0-1.0>,
  "limitations": <0.0-1.0>,
  "ethics": <0.0-1.0>,
  "reasoning": "Brief explanation of scores (2-3 sentences)"
}}

Provide scores as JSON object only, no additional text."""

    def _parse_llm_response(self, response_text: str, strict: bool = False) -> Dict:
        """
        Parse LLM response to extract scores.

        Args:
            response_text: Raw LLM response
            strict: Raise JSONParseError instead of returning neutral 0.5 scores
                when the response holds no JSON object

        Returns:
            Dictionary with dimension scores
        """
        # Try to extract JSON from response
        try:
            # Tolerant parse: fences, preamble, trailing commas
            scores = parse_json_response(response_text)

            # Validate all required dimensions present
            required_dims = [
                'novelty', 'rigor', 'clarity', 'reproducibility',
                'impact', 'coherence', 'limitations', 'ethics'
            ]

            for dim in required_dims:
                if dim not in scores:
                    logger.warning(f"Missing dimension {dim}, using default 0.5")
                    scores[dim] = 0.5

                # Clamp to [0, 1]
                scores[dim] = max(0.0, min(1.0, float(scores[dim])))

            return scores

        except JSONParseError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            if strict:
                raise

        # Fallback: return neutral scores
        return {dim: 0.5 for dim in ['novelty', 'rigor', 'clarity',
                                      'reproducibility', 'impact', 'coherence',
                                      'limitations', 'ethics']}

    def _calculate_overall_score(self, scores: Dict) -> float:
        """
        Calculate weighted average of 8 dimensions.

        Args:
            scores: Dictionary with dimension scores

        Returns:
            Weighted overall score (0.0-1.0)
        """
        overall = 0.0
        for dimension, weight in self.DIMENSION_WEIGHTS.items():
            overall += weight * scores.get(dimension, 0.5)

        return overall

    def _generate_feedback(
        self,
        scores: Dict,
        passes: bool,
        finding: Dict
    ) -> str:
        """
        Generate actionable feedback based on scores.

        Args:
            scores: Dimension scores
            passes: Whether finding passed thresholds
            finding: Original finding dictionary

        Returns:
            Feedback string
        """
        if passes:
            feedback = f"✅ Finding APPROVED (overall: {self._calculate_overall_score(scores):.2f})"

            # Highlight strengths
            strengths = []
            for dim, score in scores.items():
                if dim == 'reasoning':
                    continue
                if score >= 0.8:
                    strengths.append(f"{dim} ({score:.2f})")

            if strengths:
                feedback += f"\nStrengths: {', '.join(strengths)}"

        else:
            overall = self._calculate_overall_score(scores)
            feedback = f"❌ Finding REJECTED (overall: {overall:.2f}, threshold: {self.threshold:.2f})"

            # Identify weaknesses
            weaknesses = []
            for dim, score in scores.items():
                if dim == 'reasoning':
                    continue
                if score < 0.6:
                    weaknesses.append(f"{dim} ({score:.2f})")

            if weaknesses:
                feedback += f"\nWeaknesses: {', '.join(weaknesses)}"

            # Specific concerns
            if scores.get('rigor', 0) < self.min_rigor_score:
                feedback += f"\nCRITICAL: Rigor score ({scores['rigor']:.2f}) below minimum ({self.min_rigor_score:.2f})"
                feedback += "\nSuggestion: Review statistical methods and ensure they are appropriate for the data."

            if scores.get('reproducibility', 0) < 0.6:
                feedback += "\nConcern: Low reproducibility. Provide more methodological detail."

            if scores.get('clarity', 0) < 0.6:
                feedback += "\nConcern: Unclear findings. Restate conclusion more precisely."

        # Add reasoning if available
        if 'reasoning' in scores:
            feedback += f"\n\nReasoning: {scores['reasoning']}"

        return feedback

    def _mock_evaluation(self, finding: Dict) -> ScholarEvalScore:
        """
        Mock evaluation for testing (when no LLM client available).

        Provides optimistic but not perfect scores.

        Args:
            finding: Finding dictionary

        Returns:
            Mock ScholarEvalScore
        """
        # Base scores (slightly above threshold)
        base_score = 0.78

        # Check for statistical evidence
        has_stats = bool(finding.get('statistics'))
        has_methods = bool(finding.get('methods'))

        # Adjust scores based on content
        rigor_score = base_score + (0.05 if has_stats else -0.05)
        reproducibility_score = base_score + (0.05 if has_methods else -0.05)

        scores = {
            'novelty': base_score,
            'rigor': rigor_score,
            'clarity': base_score,
            'reproducibility': reproducibility_score,
            'impact': base_score - 0.03,
            'coherence': base_score,
            'limitations': base_score - 0.05,
            'ethics': 0.7  # Neutral for most findings
        }

        overall = self._calculate_overall_score(scores)
        passes = overall >= self.threshold and scores['rigor'] >= self.min_rigor_score

        # Issue #70: Run null model validation even in mock mode
        null_result = None
        statistical_validity = None
        feedback_extra = ""
        if self.run_null_model and finding.get('statistics'):
            try:
                null_validator = NullModelValidator()
                null_result_obj = null_validator.validate_finding(finding)
                null_result = null_result_obj.to_dict()
                statistical_validity = 1.0 if null_result_obj.passes_null_test else 0.0

                # Adjust scores if finding fails null model test
                if null_result_obj.persists_in_noise:
                    overall *= 0.5
                    feedback_extra = "\nWARNING: Finding persists in randomized data"
                    passes = False
            except Exception as e:
                logger.debug(f"Mock null model validation failed: {e}")

        return ScholarEvalScore(
            novelty=scores['novelty'],
            rigor=scores['rigor'],
            clarity=scores['clarity'],
            reproducibility=scores['reproducibility'],
            impact=scores['impact'],
            coherence=scores['coherence'],
            limitations=scores['limitations'],
            ethics=scores['ethics'],
            overall_score=overall,
            passes_threshold=passes,
            feedback=f"Mock evaluation: {'APPROVED' if passes else 'REJECTED'} (overall: {overall:.2f}){feedback_extra}",
            reasoning="Mock evaluation (no LLM client provided)",
            null_model_result=null_result,
            statistical_validity=statistical_validity
        )

    def batch_evaluate(self, findings: list[Dict]) -> list[ScholarEvalScore]:
        """
        Evaluate multiple findings.

        Args:
            findings: List of finding dictionaries

        Returns:
            List of ScholarEvalScore objects
        """
        return [self.evaluate_finding(finding) for finding in findings]

    def get_validation_statistics(self, scores: list[ScholarEvalScore]) -> Dict:
        """
        Compute statistics over batch of scores.

        Args:
            scores: List of ScholarEvalScore objects

        Returns:
            Dictionary with validation statistics
        """
        if not scores:
            return {}

        passed = sum(1 for s in scores if s.passes_threshold)
        total = len(scores)

        # Average scores per dimension
        avg_scores = {}
        for dim in ['novelty', 'rigor', 'clarity', 'reproducibility',
                    'impact', 'coherence', 'limitations', 'ethics']:
            avg_scores[f'avg_{dim}'] = sum(getattr(s, dim) for s in scores) / total

        return {
            'total_evaluated': total,
            'passed': passed,
            'rejected': total - passed,
            'validation_rate': passed / total,
            'avg_overall_score': sum(s.overall_score for s in scores) / total,
            **avg_scores
        }

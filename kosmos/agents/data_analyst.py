"""
Data Analyst Agent.

Claude-powered agent for interpreting experiment results, detecting patterns,
identifying anomalies, and generating scientific insights.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import numpy as np

from kosmos.agents.base import BaseAgent, AgentMessage, MessageType, AgentStatus
from kosmos.core.llm import get_client
from kosmos.core.utils.json_parser import parse_json_response, JSONParseError
from kosmos.models.result import ExperimentResult, ResultStatus, StatisticalTestResult
from kosmos.models.hypothesis import Hypothesis

logger = logging.getLogger(__name__)


class ResultInterpretation:
    """Structured interpretation of experiment results."""

    def __init__(
        self,
        experiment_id: str,
        hypothesis_supported: Optional[bool],
        confidence: float,
        summary: str,
        key_findings: List[str],
        significance_interpretation: str,
        biological_significance: Optional[str],
        comparison_to_prior_work: Optional[str],
        potential_confounds: List[str],
        follow_up_experiments: List[str],
        anomalies_detected: List[str],
        patterns_detected: List[str],
        overall_assessment: str,
        created_at: Optional[datetime] = None
    ):
        """
        Initialize result interpretation.

        Args:
            experiment_id: ID of experiment being interpreted
            hypothesis_supported: Whether hypothesis is supported (None if unclear)
            confidence: Confidence in interpretation (0.0-1.0)
            summary: High-level summary of results
            key_findings: List of 3-5 key findings
            significance_interpretation: Interpretation of statistical significance
            biological_significance: Scientific/practical meaning (if applicable)
            comparison_to_prior_work: How results compare to literature
            potential_confounds: List of potential confounding factors
            follow_up_experiments: Suggested follow-up experiments
            anomalies_detected: Any anomalies found in results
            patterns_detected: Patterns identified across results
            overall_assessment: Overall assessment of experiment quality
            created_at: Timestamp of interpretation creation
        """
        self.experiment_id = experiment_id
        self.hypothesis_supported = hypothesis_supported
        self.confidence = confidence
        self.summary = summary
        self.key_findings = key_findings
        self.significance_interpretation = significance_interpretation
        self.biological_significance = biological_significance
        self.comparison_to_prior_work = comparison_to_prior_work
        self.potential_confounds = potential_confounds
        self.follow_up_experiments = follow_up_experiments
        self.anomalies_detected = anomalies_detected
        self.patterns_detected = patterns_detected
        self.overall_assessment = overall_assessment
        self.created_at = created_at or datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "hypothesis_supported": self.hypothesis_supported,
            "confidence": self.confidence,
            "summary": self.summary,
            "key_findings": self.key_findings,
            "significance_interpretation": self.significance_interpretation,
            "biological_significance": self.biological_significance,
            "comparison_to_prior_work": self.comparison_to_prior_work,
            "potential_confounds": self.potential_confounds,
            "follow_up_experiments": self.follow_up_experiments,
            "anomalies_detected": self.anomalies_detected,
            "patterns_detected": self.patterns_detected,
            "overall_assessment": self.overall_assessment,
            "created_at": self.created_at.isoformat()
        }


class DataAnalystAgent(BaseAgent):
    """
    Agent for analyzing and interpreting experiment results using Claude.

    Capabilities:
    - Interpret statistical results in scientific context
    - Detect patterns across multiple experiments
    - Identify anomalies in results
    - Provide significance interpretation beyond p-values
    - Generate actionable insights for next steps

    Example:
        ```python
        agent = DataAnalystAgent(config={
            "use_literature_context": True,
            "detailed_interpretation": True
        })
        agent.start()

        # Interpret results
        interpretation = agent.interpret_results(
            result=experiment_result,
            hypothesis=original_hypothesis,
            literature_context="Recent papers found..."
        )

        print(f"Hypothesis supported: {interpretation.hypothesis_supported}")
        print(f"Key findings: {interpretation.key_findings}")
        ```
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Data Analyst Agent.

        Args:
            agent_id: Unique agent identifier
            agent_type: Agent type name
            config: Configuration dictionary
        """
        super().__init__(agent_id, agent_type or "DataAnalystAgent", config)

        # Configuration
        self.use_literature_context = self.config.get("use_literature_context", True)
        self.detailed_interpretation = self.config.get("detailed_interpretation", True)
        self.anomaly_detection_enabled = self.config.get("anomaly_detection_enabled", True)
        self.pattern_detection_enabled = self.config.get("pattern_detection_enabled", True)
        self.significance_threshold_strict = self.config.get("significance_threshold_strict", 0.01)
        self.significance_threshold_relaxed = self.config.get("significance_threshold_relaxed", 0.05)
        self.effect_size_threshold = self.config.get("effect_size_threshold", 0.3)

        # Components
        self.llm_client = get_client()

        # State: Store interpretations for pattern detection
        self.interpretation_history: List[ResultInterpretation] = []

        logger.info(f"Initialized DataAnalystAgent {self.agent_id}")

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent task.

        Args:
            task: Task specification with:
                - action: "interpret_results", "detect_patterns", "detect_anomalies"
                - result: ExperimentResult object
                - hypothesis: Optional Hypothesis object
                - literature_context: Optional literature context string

        Returns:
            dict: Task result with interpretation
        """
        self.status = AgentStatus.WORKING
        action = task.get("action", "interpret_results")

        try:
            if action == "interpret_results":
                result = task["result"]
                hypothesis = task.get("hypothesis")
                literature_context = task.get("literature_context")

                interpretation = self.interpret_results(result, hypothesis, literature_context)

                return {
                    "success": True,
                    "interpretation": interpretation.to_dict()
                }

            elif action == "detect_patterns":
                results = task["results"]
                patterns = self.detect_patterns_across_results(results)

                return {
                    "success": True,
                    "patterns": patterns
                }

            elif action == "detect_anomalies":
                result = task["result"]
                anomalies = self.detect_anomalies(result)

                return {
                    "success": True,
                    "anomalies": anomalies
                }

            else:
                raise ValueError(f"Unknown action: {action}")

        except Exception as e:
            logger.error(f"Error executing task in DataAnalystAgent: {e}")
            self.errors_encountered += 1
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            self.status = AgentStatus.IDLE
            self.tasks_completed += 1

    def analyze(
        self,
        results: List[ExperimentResult],
        hypothesis: Optional[Hypothesis] = None,
        literature_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze multiple experiment results and synthesize findings.

        This is a convenience method that combines result interpretation
        and pattern detection for batch analysis of results.

        Args:
            results: List of ExperimentResult objects to analyze
            hypothesis: Optional hypothesis being tested
            literature_context: Optional literature context string

        Returns:
            Dict with:
                - individual_analyses: List of interpretations per result
                - synthesis: Overall synthesis of findings
                - patterns: Detected patterns across results
                - anomalies: Detected anomalies

        Example:
            ```python
            analysis = agent.analyze(
                results=[result1, result2],
                hypothesis=hypothesis
            )
            print(f"Synthesis: {analysis['synthesis']}")
            ```
        """
        individual_analyses = []
        all_anomalies = []

        # Analyze each result individually
        for result in results:
            try:
                interpretation = self.interpret_results(
                    result=result,
                    hypothesis=hypothesis,
                    literature_context=literature_context
                )
                individual_analyses.append(interpretation)

                # Detect anomalies if enabled
                if self.anomaly_detection_enabled:
                    anomalies = self.detect_anomalies(result)
                    all_anomalies.extend(anomalies)

            except Exception as e:
                logger.warning(f"Failed to analyze result {result.id}: {e}")

        # Detect patterns across all results
        patterns = []
        if len(results) > 1 and self.pattern_detection_enabled:
            patterns = self.detect_patterns_across_results(results)

        # Generate synthesis
        synthesis = self._generate_synthesis(individual_analyses, patterns)

        return {
            "individual_analyses": individual_analyses,
            "synthesis": synthesis,
            "patterns": patterns,
            "anomalies": all_anomalies
        }

    def _generate_synthesis(
        self,
        analyses: List['ResultInterpretation'],
        patterns: List[str]
    ) -> str:
        """Generate a synthesis of multiple analyses."""
        if not analyses:
            return "No results to synthesize."

        supported_count = sum(1 for a in analyses if a.hypothesis_supported)
        total_count = len(analyses)

        synthesis_parts = [
            f"Analyzed {total_count} experiment results.",
            f"{supported_count}/{total_count} results support the hypothesis." if total_count > 0 else "",
        ]

        if patterns:
            synthesis_parts.append(f"Detected {len(patterns)} patterns across results.")

        # Aggregate key findings
        all_findings = []
        for a in analyses:
            if hasattr(a, 'key_findings'):
                all_findings.extend(a.key_findings[:2])  # Top 2 findings per analysis

        if all_findings:
            synthesis_parts.append(f"Key findings: {'; '.join(all_findings[:5])}")

        return " ".join(filter(None, synthesis_parts))

    # ========================================================================
    # RESULT INTERPRETATION
    # ========================================================================

    def interpret_results(
        self,
        result: ExperimentResult,
        hypothesis: Optional[Hypothesis] = None,
        literature_context: Optional[str] = None
    ) -> ResultInterpretation:
        """
        Interpret experiment results using Claude.

        Args:
            result: ExperimentResult object to interpret
            hypothesis: Optional original hypothesis
            literature_context: Optional context from literature

        Returns:
            ResultInterpretation: Structured interpretation
        """
        logger.info(f"Interpreting results for experiment {result.experiment_id}")

        # Extract key information from result
        result_summary = self._extract_result_summary(result)

        # Build interpretation prompt
        prompt = self._build_interpretation_prompt(
            result_summary, hypothesis, literature_context
        )

        # Get Claude interpretation
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system="You are an expert scientific data analyst. Provide nuanced, "
                       "evidence-based interpretations of experimental results. Focus on "
                       "scientific meaning, not just statistical significance.",
                max_tokens=2000,
                temperature=0.3  # Lower temperature for more focused analysis
            )

            # Parse Claude's response (extract content from LLMResponse)
            response_text = response.content if hasattr(response, 'content') else str(response)
            interpretation = self._parse_interpretation_response(
                response_text, result.experiment_id, result
            )

            # Store in history for pattern detection
            self.interpretation_history.append(interpretation)

            logger.info(f"Completed interpretation for {result.experiment_id}")
            return interpretation

        except Exception as e:
            logger.error(f"Error getting Claude interpretation: {e}")
            # Return fallback interpretation
            return self._create_fallback_interpretation(result)

    def _extract_result_summary(self, result: ExperimentResult) -> Dict[str, Any]:
        """Extract key information from result for prompt."""
        summary = {
            "experiment_id": result.experiment_id,
            "status": result.status.value,
            "primary_test": result.primary_test,
            "primary_p_value": result.primary_p_value,
            "primary_effect_size": result.primary_effect_size,
            "supports_hypothesis": result.supports_hypothesis,
            "statistical_tests": []
        }

        # Add statistical test details
        for test in result.statistical_tests:
            summary["statistical_tests"].append({
                "test_name": test.test_name,
                "statistic": test.statistic,
                "p_value": test.p_value,
                "effect_size": test.effect_size,
                "effect_size_type": test.effect_size_type,
                "significance_label": test.significance_label,
                "sample_size": test.sample_size
            })

        # Add variable summaries
        if result.variable_results:
            summary["variables"] = []
            for var in result.variable_results[:5]:  # Top 5 variables
                summary["variables"].append({
                    "name": var.variable_name,
                    "mean": var.mean,
                    "median": var.median,
                    "std": var.std,
                    "min": var.min,
                    "max": var.max,
                    "n_samples": var.n_samples
                })

        return summary

    def _build_interpretation_prompt(
        self,
        result_summary: Dict[str, Any],
        hypothesis: Optional[Hypothesis],
        literature_context: Optional[str]
    ) -> str:
        """Build prompt for Claude interpretation."""
        prompt_parts = []

        # Hypothesis context
        if hypothesis:
            prompt_parts.append(f"""
HYPOTHESIS:
{hypothesis.statement}

Domain: {hypothesis.domain}
Expected Outcome: {getattr(hypothesis, 'expected_outcome', 'Not specified')}
""")

        # Result summary
        prompt_parts.append(f"""
EXPERIMENTAL RESULTS:
Status: {result_summary['status']}
Primary Test: {result_summary['primary_test']}
Primary P-value: {result_summary['primary_p_value']}
Primary Effect Size: {result_summary['primary_effect_size']}
Hypothesis Supported: {result_summary['supports_hypothesis']}

Statistical Tests:
""")

        for i, test in enumerate(result_summary['statistical_tests'][:3], 1):
            prompt_parts.append(f"""
Test {i}: {test['test_name']}
  - Statistic: {test['statistic']:.4f}
  - P-value: {test['p_value']:.6f}
  - Effect Size: {test['effect_size']} ({test['effect_size_type']})
  - Significance: {test['significance_label']}
  - Sample Size: {test['sample_size']}
""")

        # Literature context
        if literature_context and self.use_literature_context:
            prompt_parts.append(f"""
LITERATURE CONTEXT:
{literature_context[:1000]}  # Limit to 1000 chars
""")

        # Instructions
        prompt_parts.append("""
Please provide a comprehensive scientific interpretation of these results:

1. HYPOTHESIS SUPPORT: Does the data support or reject the hypothesis? (Be nuanced - consider strength of evidence)

2. KEY FINDINGS: What are the 3-5 most important findings from these results?

3. STATISTICAL SIGNIFICANCE: Interpret the statistical significance beyond just "p < 0.05". What does it mean scientifically?

4. EFFECT SIZE: Is the effect size scientifically/practically meaningful, even if statistically significant?

5. BIOLOGICAL/PHYSICAL SIGNIFICANCE: What is the real-world meaning of these results? (If applicable)

6. COMPARISON TO PRIOR WORK: How do these results compare to existing literature? (If context provided)

7. POTENTIAL CONFOUNDS: What are 2-3 potential confounding factors or limitations?

8. FOLLOW-UP EXPERIMENTS: What 3-5 follow-up experiments would you recommend based on these results?

9. OVERALL ASSESSMENT: Rate the quality and reliability of this experiment (Low/Medium/High) and explain why.

Format your response as JSON with the following structure:
{
    "hypothesis_supported": true/false/null,
    "confidence": 0.0-1.0,
    "summary": "2-3 sentence overview",
    "key_findings": ["finding 1", "finding 2", ...],
    "significance_interpretation": "detailed statistical interpretation",
    "biological_significance": "real-world meaning or null",
    "comparison_to_prior_work": "comparison or null",
    "potential_confounds": ["confound 1", "confound 2", ...],
    "follow_up_experiments": ["experiment 1", "experiment 2", ...],
    "overall_assessment": "Quality: X/5. Explanation..."
}
""")

        return "\n".join(prompt_parts)

    def _parse_interpretation_response(
        self,
        response: str,
        experiment_id: str,
        result: ExperimentResult
    ) -> ResultInterpretation:
        """Parse Claude's JSON response into ResultInterpretation."""
        try:
            # Extract JSON from response (Claude sometimes adds text before/after)
            data = parse_json_response(str(response))

            # Detect anomalies and patterns
            anomalies = self.detect_anomalies(result) if self.anomaly_detection_enabled else []
            patterns = []  # Will be populated in pattern detection

            return ResultInterpretation(
                experiment_id=experiment_id,
                hypothesis_supported=data.get("hypothesis_supported"),
                confidence=data.get("confidence", 0.5),
                summary=data.get("summary", ""),
                key_findings=data.get("key_findings", []),
                significance_interpretation=data.get("significance_interpretation", ""),
                biological_significance=data.get("biological_significance"),
                comparison_to_prior_work=data.get("comparison_to_prior_work"),
                potential_confounds=data.get("potential_confounds", []),
                follow_up_experiments=data.get("follow_up_experiments", []),
                anomalies_detected=anomalies,
                patterns_detected=patterns,
                overall_assessment=data.get("overall_assessment", "")
            )

        except JSONParseError as e:
            logger.error(f"Failed to parse JSON from Claude response: {e}")
            logger.debug(f"Response was: {response}")
            return self._create_fallback_interpretation(result)

    def _create_fallback_interpretation(self, result: ExperimentResult) -> ResultInterpretation:
        """Create fallback interpretation if Claude fails."""
        _p = result.primary_p_value
        if _p is None:
            _significance = 'undetermined'
        else:
            _significance = 'significant' if _p < 0.05 else 'non-significant'
        return ResultInterpretation(
            experiment_id=result.experiment_id,
            hypothesis_supported=result.supports_hypothesis,
            confidence=0.5,
            summary=f"Experiment {result.status.value} with p-value {result.primary_p_value}",
            key_findings=[
                f"Primary test: {result.primary_test}",
                f"P-value: {result.primary_p_value}",
                f"Effect size: {result.primary_effect_size}"
            ],
            significance_interpretation=f"P-value of {_p} indicates {_significance} results",
            biological_significance=None,
            comparison_to_prior_work=None,
            potential_confounds=["Automated analysis - manual review recommended"],
            follow_up_experiments=["Manual review needed for recommendations"],
            anomalies_detected=[],
            patterns_detected=[],
            overall_assessment="Automated fallback interpretation - Claude unavailable"
        )

    # ========================================================================
    # ANOMALY DETECTION
    # ========================================================================

    def detect_anomalies(self, result: ExperimentResult) -> List[str]:
        """
        Detect anomalies in experimental results.

        Checks for:
        - Unusual p-value distributions
        - Effect size/significance mismatches
        - Outliers in statistical tests
        - Data quality issues

        Args:
            result: ExperimentResult to analyze

        Returns:
            list: List of anomaly descriptions
        """
        anomalies = []

        # Check for significant p-value with tiny effect size
        if result.primary_p_value is not None and result.primary_effect_size is not None:
            if result.primary_p_value < self.significance_threshold_strict:
                if abs(result.primary_effect_size) < self.effect_size_threshold:
                    anomalies.append(
                        f"ANOMALY: Statistically significant (p={result.primary_p_value:.4f}) "
                        f"but tiny effect size ({result.primary_effect_size:.4f}). "
                        f"May indicate large sample size masking practical insignificance."
                    )

        # Check for large effect size with non-significant p-value
        if result.primary_p_value is not None and result.primary_effect_size is not None:
            if result.primary_p_value > self.significance_threshold_relaxed:
                if abs(result.primary_effect_size) > 0.5:  # Cohen's d > 0.5 is medium/large
                    anomalies.append(
                        f"ANOMALY: Large effect size ({result.primary_effect_size:.4f}) "
                        f"but non-significant p-value (p={result.primary_p_value:.4f}). "
                        f"May indicate insufficient sample size or high variance."
                    )

        # Check for p-value exactly 0 or 1 (usually indicates error)
        if result.primary_p_value is not None:
            if result.primary_p_value == 0.0:
                anomalies.append(
                    "ANOMALY: P-value is exactly 0.0. This is unusual and may indicate "
                    "a computational error or extremely strong effect."
                )
            elif result.primary_p_value == 1.0:
                anomalies.append(
                    "ANOMALY: P-value is exactly 1.0. This may indicate a computational error."
                )

        # Check for inconsistent statistical tests
        if len(result.statistical_tests) >= 2:
            p_values = [t.p_value for t in result.statistical_tests if t.p_value is not None]
            if len(p_values) >= 2:
                # Check if some tests are significant and others not (may be expected, but worth noting)
                significant = [p < self.significance_threshold_relaxed for p in p_values]
                if any(significant) and not all(significant):
                    sig_count = sum(significant)
                    anomalies.append(
                        f"NOTE: {sig_count}/{len(significant)} statistical tests are significant. "
                        f"Mixed results may indicate variability in experimental conditions."
                    )

        # Check variable results for outliers
        if result.variable_results:
            for var in result.variable_results:
                if var.mean is not None and var.std is not None and var.std > 0:
                    cv = var.std / var.mean  # Coefficient of variation
                    if cv > 1.0:  # Very high variability
                        anomalies.append(
                            f"ANOMALY: Variable '{var.variable_name}' has very high variability "
                            f"(CV={cv:.2f}). This may affect result reliability."
                        )

        logger.debug(f"Detected {len(anomalies)} anomalies in {result.experiment_id}")
        return anomalies

    # ========================================================================
    # PATTERN DETECTION
    # ========================================================================

    def detect_patterns_across_results(
        self,
        results: List[ExperimentResult]
    ) -> List[str]:
        """
        Detect patterns across multiple experiment results.

        Looks for:
        - Consistent trends (e.g., always positive/negative effects)
        - Non-linear relationships
        - Unexpected similarities/differences

        Args:
            results: List of ExperimentResult objects

        Returns:
            list: List of pattern descriptions
        """
        patterns = []

        if len(results) < 2:
            return patterns

        # Extract p-values and effect sizes
        p_values = [r.primary_p_value for r in results if r.primary_p_value is not None]
        effect_sizes = [r.primary_effect_size for r in results if r.primary_effect_size is not None]

        # Pattern: Consistent effect direction
        if len(effect_sizes) >= 3:
            positive = [e > 0 for e in effect_sizes]
            if all(positive):
                patterns.append(
                    f"PATTERN: All {len(effect_sizes)} experiments show positive effects "
                    f"(mean effect size: {np.mean(effect_sizes):.3f}). "
                    f"This suggests a consistent underlying phenomenon."
                )
            elif not any(positive):
                patterns.append(
                    f"PATTERN: All {len(effect_sizes)} experiments show negative effects "
                    f"(mean effect size: {np.mean(effect_sizes):.3f}). "
                    f"This suggests a consistent inverse relationship."
                )

        # Pattern: Increasing/decreasing trend in effect sizes
        if len(effect_sizes) >= 4:
            # Simple monotonicity check
            increasing = all(effect_sizes[i] <= effect_sizes[i+1] for i in range(len(effect_sizes)-1))
            decreasing = all(effect_sizes[i] >= effect_sizes[i+1] for i in range(len(effect_sizes)-1))

            if increasing:
                patterns.append(
                    f"PATTERN: Effect sizes show increasing trend across {len(effect_sizes)} experiments "
                    f"({effect_sizes[0]:.3f} → {effect_sizes[-1]:.3f}). "
                    f"This may indicate a dose-response or temporal relationship."
                )
            elif decreasing:
                patterns.append(
                    f"PATTERN: Effect sizes show decreasing trend across {len(effect_sizes)} experiments "
                    f"({effect_sizes[0]:.3f} → {effect_sizes[-1]:.3f}). "
                    f"This may indicate diminishing returns or saturation effects."
                )

        # Pattern: Bimodal p-value distribution (either very significant or not)
        if len(p_values) >= 5:
            very_sig = sum(p < 0.01 for p in p_values)
            very_nonsig = sum(p > 0.1 for p in p_values)
            middle = len(p_values) - very_sig - very_nonsig

            if middle == 0 and very_sig > 0 and very_nonsig > 0:
                patterns.append(
                    f"PATTERN: Bimodal p-value distribution detected ({very_sig} highly significant, "
                    f"{very_nonsig} clearly non-significant, 0 borderline). "
                    f"This suggests different experimental conditions or subgroups."
                )

        logger.debug(f"Detected {len(patterns)} patterns across {len(results)} results")
        return patterns

    # ========================================================================
    # SIGNIFICANCE INTERPRETATION
    # ========================================================================

    def interpret_significance(
        self,
        p_value: float,
        effect_size: Optional[float],
        sample_size: Optional[int]
    ) -> str:
        """
        Provide nuanced interpretation of statistical significance.

        Goes beyond "p < 0.05 = significant" to explain:
        - Strength of evidence
        - Practical vs statistical significance
        - Role of sample size

        Args:
            p_value: P-value from statistical test
            effect_size: Effect size (e.g., Cohen's d)
            sample_size: Sample size

        Returns:
            str: Interpretation of significance
        """
        interpretation_parts = []

        # Statistical significance level
        if p_value < 0.001:
            interpretation_parts.append(
                f"The p-value (p={p_value:.6f}) provides very strong evidence against "
                f"the null hypothesis (p < 0.001)."
            )
        elif p_value < 0.01:
            interpretation_parts.append(
                f"The p-value (p={p_value:.4f}) provides strong evidence against "
                f"the null hypothesis (p < 0.01)."
            )
        elif p_value < 0.05:
            interpretation_parts.append(
                f"The p-value (p={p_value:.4f}) provides moderate evidence against "
                f"the null hypothesis (p < 0.05), meeting conventional significance threshold."
            )
        elif p_value < 0.1:
            interpretation_parts.append(
                f"The p-value (p={p_value:.4f}) provides suggestive but inconclusive evidence "
                f"(0.05 < p < 0.1). This may warrant further investigation."
            )
        else:
            interpretation_parts.append(
                f"The p-value (p={p_value:.4f}) does not provide sufficient evidence to reject "
                f"the null hypothesis (p > 0.1)."
            )

        # Effect size interpretation
        if effect_size is not None:
            if abs(effect_size) < 0.2:
                size_label = "negligible"
            elif abs(effect_size) < 0.5:
                size_label = "small"
            elif abs(effect_size) < 0.8:
                size_label = "medium"
            else:
                size_label = "large"

            interpretation_parts.append(
                f"The effect size ({effect_size:.3f}) is {size_label}, indicating "
                f"{'a practically meaningful difference' if abs(effect_size) >= 0.5 else 'limited practical significance'}."
            )

            # Check for mismatch
            is_stat_sig = p_value < 0.05
            is_pract_sig = abs(effect_size) >= 0.5

            if is_stat_sig and not is_pract_sig:
                interpretation_parts.append(
                    "⚠ Note: While statistically significant, the small effect size suggests "
                    "limited practical importance. This may be due to large sample size."
                )
            elif not is_stat_sig and is_pract_sig:
                interpretation_parts.append(
                    "⚠ Note: While not statistically significant, the large effect size suggests "
                    "potential practical importance. This may be due to small sample size or high variance."
                )

        # Sample size interpretation
        if sample_size is not None:
            if sample_size < 30:
                interpretation_parts.append(
                    f"The small sample size (n={sample_size}) limits statistical power. "
                    f"Results should be interpreted with caution."
                )
            elif sample_size > 1000:
                interpretation_parts.append(
                    f"The large sample size (n={sample_size}) provides high statistical power, "
                    f"making even small effects statistically significant."
                )

        return " ".join(interpretation_parts)

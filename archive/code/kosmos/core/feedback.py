"""
Feedback Loop System - Learns from experimental results to improve research (Phase 7).

Processes results and generates feedback signals to:
- Update hypothesis priorities
- Adapt experiment templates
- Adjust strategy weights
- Learn success/failure patterns
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from enum import Enum
import logging

from kosmos.models.result import ExperimentResult, ResultStatus
from kosmos.models.hypothesis import Hypothesis
from kosmos.utils.compat import model_to_dict

logger = logging.getLogger(__name__)


class FeedbackSignalType(str, Enum):
    """Types of feedback signals."""

    SUCCESS_PATTERN = "success_pattern"
    FAILURE_PATTERN = "failure_pattern"
    HYPOTHESIS_UPDATE = "hypothesis_update"
    STRATEGY_ADJUSTMENT = "strategy_adjustment"
    TEMPLATE_UPDATE = "template_update"
    PRIORITY_CHANGE = "priority_change"


class FeedbackSignal(BaseModel):
    """A feedback signal from experimental results."""

    signal_type: FeedbackSignalType
    source: str  # Result ID or analysis ID
    data: Dict[str, Any]
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    applied: bool = False


class SuccessPattern(BaseModel):
    """Pattern extracted from successful experiments."""

    pattern_id: str
    description: str
    hypothesis_characteristics: Dict[str, Any] = Field(default_factory=dict)
    experiment_design: Dict[str, Any] = Field(default_factory=dict)
    statistical_approach: Dict[str, Any] = Field(default_factory=dict)
    occurrences: int = 1
    success_rate: float = 1.0
    confidence: float = 0.5
    examples: List[str] = Field(default_factory=list)  # Result IDs
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FailurePattern(BaseModel):
    """Pattern extracted from failed experiments."""

    pattern_id: str
    description: str
    failure_type: str  # "statistical", "methodological", "conceptual"
    common_characteristics: Dict[str, Any] = Field(default_factory=dict)
    recommended_fixes: List[str] = Field(default_factory=list)
    occurrences: int = 1
    examples: List[str] = Field(default_factory=list)  # Result IDs
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FeedbackLoop:
    """
    Feedback loop system for autonomous learning.

    Processes experimental results and generates feedback signals
    to improve future research decisions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feedback loop.

        Args:
            config: Configuration dict
        """
        self.config = config or {}

        # Pattern storage
        self.success_patterns: Dict[str, SuccessPattern] = {}
        self.failure_patterns: Dict[str, FailurePattern] = {}

        # Feedback signal queue
        self.pending_signals: List[FeedbackSignal] = []
        self.applied_signals: List[FeedbackSignal] = []

        # Learning rates
        self.success_learning_rate = self.config.get("success_learning_rate", 0.3)
        self.failure_learning_rate = self.config.get("failure_learning_rate", 0.4)

        logger.info("FeedbackLoop initialized")

    # ========================================================================
    # PROCESS RESULTS
    # ========================================================================

    def process_result_feedback(
        self,
        result: ExperimentResult,
        hypothesis: Hypothesis
    ) -> List[FeedbackSignal]:
        """
        Process experimental result and generate feedback signals.

        Args:
            result: Experiment result
            hypothesis: Hypothesis that was tested

        Returns:
            List[FeedbackSignal]: Generated feedback signals
        """
        logger.info(f"Processing feedback from result {result.id}")

        signals = []

        # Analyze success or failure
        if result.supports_hypothesis is True and result.status == ResultStatus.SUCCESS:
            # Success - learn patterns
            signals.extend(self._analyze_success(result, hypothesis))

        elif result.supports_hypothesis is False or result.status == ResultStatus.FAILURE:
            # Failure - learn from mistakes
            signals.extend(self._analyze_failure(result, hypothesis))

        # Always generate hypothesis update signal
        signals.append(self._generate_hypothesis_update_signal(result, hypothesis))

        # Store signals
        self.pending_signals.extend(signals)

        logger.info(f"Generated {len(signals)} feedback signals")
        return signals

    def _analyze_success(
        self,
        result: ExperimentResult,
        hypothesis: Hypothesis
    ) -> List[FeedbackSignal]:
        """Analyze successful experiment to extract patterns."""
        logger.debug(f"Analyzing success patterns from {result.id}")

        signals = []

        # Extract success pattern
        pattern = self._extract_success_pattern(result, hypothesis)

        if pattern:
            # Check if similar pattern exists
            existing_pattern = self._find_similar_success_pattern(pattern)

            if existing_pattern:
                # Update existing pattern
                existing_pattern.occurrences += 1
                existing_pattern.examples.append(result.id)
                existing_pattern.success_rate = (
                    (existing_pattern.success_rate * (existing_pattern.occurrences - 1) + 1.0) /
                    existing_pattern.occurrences
                )
                existing_pattern.confidence = min(1.0, existing_pattern.confidence + 0.1)
                existing_pattern.updated_at = datetime.now(timezone.utc)

                pattern_id = existing_pattern.pattern_id
            else:
                # Store new pattern
                pattern_id = f"success_{len(self.success_patterns) + 1}"
                pattern.pattern_id = pattern_id
                self.success_patterns[pattern_id] = pattern

            # Generate signal to increase priority of similar hypotheses
            signals.append(FeedbackSignal(
                signal_type=FeedbackSignalType.SUCCESS_PATTERN,
                source=result.id,
                data={
                    "pattern_id": pattern_id,
                    "pattern": model_to_dict(pattern),
                    "action": "increase_priority",
                    "target": "similar_hypotheses"
                },
                confidence=pattern.confidence
            ))

        return signals

    def _analyze_failure(
        self,
        result: ExperimentResult,
        hypothesis: Hypothesis
    ) -> List[FeedbackSignal]:
        """Analyze failed experiment to identify issues."""
        logger.debug(f"Analyzing failure patterns from {result.id}")

        signals = []

        # Categorize failure
        failure_type = self._categorize_failure(result)

        # Extract failure pattern
        pattern = self._extract_failure_pattern(result, hypothesis, failure_type)

        if pattern:
            # Check if similar pattern exists
            existing_pattern = self._find_similar_failure_pattern(pattern)

            if existing_pattern:
                # Update existing pattern
                existing_pattern.occurrences += 1
                existing_pattern.examples.append(result.id)
                existing_pattern.updated_at = datetime.now(timezone.utc)

                pattern_id = existing_pattern.pattern_id
            else:
                # Store new pattern
                pattern_id = f"failure_{len(self.failure_patterns) + 1}"
                pattern.pattern_id = pattern_id
                self.failure_patterns[pattern_id] = pattern

            # Generate signal to avoid similar approaches
            signals.append(FeedbackSignal(
                signal_type=FeedbackSignalType.FAILURE_PATTERN,
                source=result.id,
                data={
                    "pattern_id": pattern_id,
                    "pattern": model_to_dict(pattern),
                    "action": "avoid_pattern",
                    "recommended_fixes": pattern.recommended_fixes
                },
                confidence=0.8
            ))

        return signals

    def _categorize_failure(self, result: ExperimentResult) -> str:
        """
        Categorize type of failure.

        Args:
            result: Experiment result

        Returns:
            str: Failure category
        """
        if result.status == ResultStatus.FAILURE:
            return "execution_error"

        # Check statistical issues
        if result.primary_p_value is not None:
            if result.primary_p_value > 0.05 and result.primary_effect_size is not None:
                if abs(result.primary_effect_size) < 0.2:
                    return "underpowered"  # Not significant, small effect
                else:
                    return "statistical"  # Large effect but not significant

        return "conceptual"  # Hypothesis itself may be flawed

    def _extract_success_pattern(
        self,
        result: ExperimentResult,
        hypothesis: Hypothesis
    ) -> Optional[SuccessPattern]:
        """Extract success pattern from result."""

        description = f"Successful {result.primary_test} with effect size {result.primary_effect_size:.2f}"

        pattern = SuccessPattern(
            pattern_id="",  # Will be assigned later
            description=description,
            hypothesis_characteristics={
                "domain": hypothesis.domain,
                "testability_score": hypothesis.testability_score,
                "novelty_score": hypothesis.novelty_score
            },
            experiment_design={
                "test_type": result.primary_test,
                "sample_size": model_to_dict(result.metadata) if hasattr(result, 'metadata') else {}
            },
            statistical_approach={
                "p_value": result.primary_p_value,
                "effect_size": result.primary_effect_size,
                "effect_size_type": "Cohen's d"  # Simplified
            },
            examples=[result.id]
        )

        return pattern

    def _extract_failure_pattern(
        self,
        result: ExperimentResult,
        hypothesis: Hypothesis,
        failure_type: str
    ) -> Optional[FailurePattern]:
        """Extract failure pattern from result."""

        # Generate recommended fixes based on failure type
        fixes = []
        if failure_type == "underpowered":
            fixes = [
                "Increase sample size",
                "Use more sensitive statistical test",
                "Reduce measurement error"
            ]
        elif failure_type == "statistical":
            fixes = [
                "Check for outliers",
                "Verify assumptions",
                "Consider non-parametric test"
            ]
        elif failure_type == "conceptual":
            fixes = [
                "Refine hypothesis",
                "Add moderating variables",
                "Explore boundary conditions"
            ]
        else:
            fixes = ["Review experimental design"]

        description = f"{failure_type.capitalize()} failure in {result.primary_test}"

        pattern = FailurePattern(
            pattern_id="",  # Will be assigned later
            description=description,
            failure_type=failure_type,
            common_characteristics={
                "domain": hypothesis.domain,
                "test_type": result.primary_test
            },
            recommended_fixes=fixes,
            examples=[result.id]
        )

        return pattern

    def _find_similar_success_pattern(self, pattern: SuccessPattern) -> Optional[SuccessPattern]:
        """Find similar existing success pattern."""
        # Simplified - check if same test type and similar characteristics
        for existing in self.success_patterns.values():
            if (existing.experiment_design.get("test_type") ==
                pattern.experiment_design.get("test_type")):
                return existing

        return None

    def _find_similar_failure_pattern(self, pattern: FailurePattern) -> Optional[FailurePattern]:
        """Find similar existing failure pattern."""
        # Simplified - check if same failure type
        for existing in self.failure_patterns.values():
            if existing.failure_type == pattern.failure_type:
                return existing

        return None

    def _generate_hypothesis_update_signal(
        self,
        result: ExperimentResult,
        hypothesis: Hypothesis
    ) -> FeedbackSignal:
        """Generate signal to update hypothesis based on result."""

        if result.supports_hypothesis is True:
            action = "increase_confidence"
            update_value = self.success_learning_rate
        elif result.supports_hypothesis is False:
            action = "decrease_confidence"
            update_value = self.failure_learning_rate
        else:
            action = "no_change"
            update_value = 0.0

        return FeedbackSignal(
            signal_type=FeedbackSignalType.HYPOTHESIS_UPDATE,
            source=result.id,
            data={
                "hypothesis_id": hypothesis.id,
                "action": action,
                "update_value": update_value,
                "result_summary": {
                    "supports": result.supports_hypothesis,
                    "p_value": result.primary_p_value,
                    "effect_size": result.primary_effect_size
                }
            },
            confidence=1.0
        )

    # ========================================================================
    # APPLY FEEDBACK
    # ========================================================================

    def apply_feedback(
        self,
        signal: FeedbackSignal,
        hypotheses: List[Hypothesis],
        strategy_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Apply feedback signal to update system state.

        Args:
            signal: Feedback signal to apply
            hypotheses: Current hypotheses
            strategy_weights: Current strategy weights

        Returns:
            dict: Changes made
        """
        logger.info(f"Applying feedback signal: {signal.signal_type}")

        changes = {
            "hypotheses_updated": [],
            "strategies_adjusted": [],
            "templates_modified": []
        }

        if signal.signal_type == FeedbackSignalType.HYPOTHESIS_UPDATE:
            # Update specific hypothesis
            hypothesis_id = signal.data.get("hypothesis_id")
            hypothesis = next((h for h in hypotheses if h.id == hypothesis_id), None)

            if hypothesis:
                action = signal.data.get("action")
                update_value = signal.data.get("update_value", 0.0)

                if action == "increase_confidence":
                    old_conf = hypothesis.confidence_score or 0.5
                    hypothesis.confidence_score = min(1.0, old_conf + update_value)
                    changes["hypotheses_updated"].append(hypothesis_id)

                elif action == "decrease_confidence":
                    old_conf = hypothesis.confidence_score or 0.5
                    hypothesis.confidence_score = max(0.0, old_conf - update_value)
                    changes["hypotheses_updated"].append(hypothesis_id)

        elif signal.signal_type == FeedbackSignalType.SUCCESS_PATTERN:
            # Apply success pattern learning
            pattern_data = signal.data.get("pattern", {})
            # Would increase priority of hypotheses matching pattern characteristics
            changes["strategies_adjusted"].append("success_pattern_applied")

        elif signal.signal_type == FeedbackSignalType.FAILURE_PATTERN:
            # Apply failure pattern learning
            # Would decrease priority of hypotheses matching failure characteristics
            changes["strategies_adjusted"].append("failure_pattern_avoided")

        # Mark signal as applied
        signal.applied = True
        self.applied_signals.append(signal)
        if signal in self.pending_signals:
            self.pending_signals.remove(signal)

        logger.info(f"Feedback applied: {changes}")
        return changes

    # ========================================================================
    # QUERIES & REPORTING
    # ========================================================================

    def get_success_patterns(self) -> List[SuccessPattern]:
        """Get all success patterns."""
        return list(self.success_patterns.values())

    def get_failure_patterns(self) -> List[FailurePattern]:
        """Get all failure patterns."""
        return list(self.failure_patterns.values())

    def get_pending_signals(self) -> List[FeedbackSignal]:
        """Get pending feedback signals."""
        return self.pending_signals.copy()

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning."""
        return {
            "success_patterns_learned": len(self.success_patterns),
            "failure_patterns_learned": len(self.failure_patterns),
            "pending_signals": len(self.pending_signals),
            "applied_signals": len(self.applied_signals),
            "most_common_success": self._get_most_common_success_pattern(),
            "most_common_failure": self._get_most_common_failure_pattern()
        }

    def _get_most_common_success_pattern(self) -> Optional[str]:
        """Get most frequently occurring success pattern."""
        if not self.success_patterns:
            return None

        most_common = max(self.success_patterns.values(), key=lambda p: p.occurrences)
        return most_common.description

    def _get_most_common_failure_pattern(self) -> Optional[str]:
        """Get most frequently occurring failure pattern."""
        if not self.failure_patterns:
            return None

        most_common = max(self.failure_patterns.values(), key=lambda p: p.occurrences)
        return most_common.description

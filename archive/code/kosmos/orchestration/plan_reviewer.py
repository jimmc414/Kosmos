"""
Plan Reviewer Agent for Kosmos.

Validates research plans on 5 dimensions before execution to ensure quality.

5 Review Dimensions (0-10 each):
1. Specificity: Are tasks concrete and executable?
2. Relevance: Do tasks address research objective?
3. Novelty: Do tasks avoid redundancy?
4. Coverage: Do tasks cover important aspects?
5. Feasibility: Are tasks achievable within constraints?

Approval Criteria:
- Average score ≥ 7.0/10
- Minimum score ≥ 5.0/10 (no catastrophic failures)
- At least 3 data_analysis tasks
- At least 2 different task types

Performance Target: ~80% approval rate on first submission
"""

import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

from kosmos.config import _DEFAULT_CLAUDE_SONNET_MODEL
logger = logging.getLogger(__name__)


@dataclass
class PlanReview:
    """Container for plan review results."""
    approved: bool
    scores: Dict[str, float]  # dimension → score (0-10)
    average_score: float
    min_score: float
    feedback: str
    required_changes: List[str]
    suggestions: List[str]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'approved': self.approved,
            'scores': self.scores,
            'average_score': self.average_score,
            'min_score': self.min_score,
            'feedback': self.feedback,
            'required_changes': self.required_changes,
            'suggestions': self.suggestions
        }


class PlanReviewerAgent:
    """
    Plan quality validation agent.

    Evaluates research plans on multiple dimensions before execution
    to prevent low-quality or unfocused research.

    Design Philosophy:
    - Multi-dimensional scoring catches different failure modes
    - Minimum thresholds prevent catastrophic failures
    - Actionable feedback enables plan revision
    """

    # Review dimension weights (not currently used, but available for future)
    DIMENSION_WEIGHTS = {
        'specificity': 0.25,
        'relevance': 0.25,
        'novelty': 0.20,
        'coverage': 0.15,
        'feasibility': 0.15
    }

    def __init__(
        self,
        anthropic_client=None,
        model: str = _DEFAULT_CLAUDE_SONNET_MODEL,
        min_average_score: float = 7.0,
        min_dimension_score: float = 5.0,
        temperature: float = 0.3
    ):
        """
        Initialize Plan Reviewer Agent.

        Args:
            anthropic_client: Anthropic client for LLM-based review
            model: Model to use for review
            min_average_score: Minimum average score for approval
            min_dimension_score: Minimum score on any single dimension
            temperature: LLM temperature for review evaluation
        """
        self.client = anthropic_client
        self.model = model
        self.min_average_score = min_average_score
        self.min_dimension_score = min_dimension_score
        self._temperature = temperature

    def review_plan(
        self,
        plan: Dict,
        context: Dict
    ) -> PlanReview:
        """
        Review research plan on 5 dimensions.

        Args:
            plan: ResearchPlan as dictionary
            context: Context from State Manager

        Returns:
            PlanReview with scores, approval status, and feedback
        """
        # If no LLM client, use mock review
        if self.client is None:
            return self._mock_review(plan, context)

        # Build review prompt
        prompt = self._build_review_prompt(plan, context)

        try:
            # Query LLM
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature
            )

            # Parse review
            review_data = self._parse_review_response(response.content[0].text)

            # Extract scores
            scores = review_data.get('scores', {})
            avg_score = sum(scores.values()) / len(scores) if scores else 0
            min_score = min(scores.values()) if scores else 0

            # Check structural requirements
            structural_ok = self._meets_structural_requirements(plan)

            # Determine approval
            approved = (
                avg_score >= self.min_average_score
                and min_score >= self.min_dimension_score
                and structural_ok
            )

            return PlanReview(
                approved=approved,
                scores=scores,
                average_score=avg_score,
                min_score=min_score,
                feedback=review_data.get('feedback', ''),
                required_changes=review_data.get('required_changes', []),
                suggestions=review_data.get('suggestions', [])
            )

        except Exception as e:
            logger.error(f"Plan review failed: {e}, using mock review")
            return self._mock_review(plan, context)

    def _build_review_prompt(self, plan: Dict, context: Dict) -> str:
        """Build prompt for plan review."""
        research_objective = context.get('research_objective', 'Not specified')
        plan_json = json.dumps(plan, indent=2)

        return f"""Review this research plan for quality.

**Research Objective**: {research_objective}

**Plan**:
{plan_json}

**Scoring Criteria** (0-10 each):

1. **Specificity**: Are tasks concrete and executable?
   - 10: Fully specified with datasets, methods, expected outputs
   - 7: Generally clear, minor gaps
   - 5: Somewhat vague, needs clarification
   - 0: Too abstract to execute

2. **Relevance**: Do tasks directly address the research objective?
   - 10: All tasks directly advance the main goal
   - 7: Most tasks relevant, some tangential
   - 5: Partially relevant
   - 0: Off-topic or unrelated

3. **Novelty**: Do tasks avoid redundancy with past work?
   - 10: All tasks explore new directions or deepen insights
   - 7: Mostly novel, some repetition
   - 5: Some redundancy with past work
   - 0: Highly redundant

4. **Coverage**: Do tasks comprehensively cover the research domain?
   - 10: Comprehensive coverage of key aspects
   - 7: Good coverage, minor gaps
   - 5: Partial coverage, significant gaps
   - 0: Narrow focus, missing key areas

5. **Feasibility**: Are tasks achievable within time/resource constraints?
   - 10: All tasks clearly executable with available resources
   - 7: Most tasks feasible, some may be challenging
   - 5: Some tasks may be too complex
   - 0: Unrealistic or impossible tasks

**Structural Requirements**:
- At least 3 data_analysis tasks
- At least 2 different task types
- Each task has description, expected_output, required_skills

**Output Format** (JSON):
{{
  "scores": {{
    "specificity": <0-10>,
    "relevance": <0-10>,
    "novelty": <0-10>,
    "coverage": <0-10>,
    "feasibility": <0-10>
  }},
  "feedback": "Detailed assessment (2-3 sentences)",
  "required_changes": ["Change 1", "Change 2"] or [],
  "suggestions": ["Suggestion 1", "Suggestion 2"] or []
}}

Provide review as JSON only, no additional text."""

    def _parse_review_response(self, response_text: str) -> Dict:
        """Parse LLM response to extract review."""
        try:
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                review_data = json.loads(json_str)

                # Validate scores present
                if 'scores' not in review_data:
                    review_data['scores'] = {
                        'specificity': 5.0,
                        'relevance': 5.0,
                        'novelty': 5.0,
                        'coverage': 5.0,
                        'feasibility': 5.0
                    }

                # Clamp scores to [0, 10]
                for dim, score in review_data['scores'].items():
                    review_data['scores'][dim] = max(0.0, min(10.0, float(score)))

                return review_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse review JSON: {e}")

        return {
            'scores': {
                'specificity': 5.0,
                'relevance': 5.0,
                'novelty': 5.0,
                'coverage': 5.0,
                'feasibility': 5.0
            },
            'feedback': 'Failed to parse review',
            'required_changes': [],
            'suggestions': []
        }

    def _meets_structural_requirements(self, plan: Dict) -> bool:
        """
        Check if plan meets basic structural requirements.

        Args:
            plan: Plan dictionary

        Returns:
            True if structural requirements met
        """
        tasks = plan.get('tasks', [])

        if not tasks:
            return False

        # Requirement 1: At least 3 data_analysis tasks
        data_analysis_count = sum(
            1 for t in tasks if t.get('type') == 'data_analysis'
        )
        if data_analysis_count < 3:
            logger.warning(
                f"Only {data_analysis_count} data_analysis tasks, need >= 3"
            )
            return False

        # Requirement 2: At least 2 different task types
        task_types = set(t.get('type') for t in tasks)
        if len(task_types) < 2:
            logger.warning(
                f"Only {len(task_types)} task types, need >= 2"
            )
            return False

        # Requirement 3: Each task has required fields
        for task in tasks:
            if not task.get('description'):
                logger.warning(f"Task {task.get('id')} missing description")
                return False
            if not task.get('expected_output'):
                logger.warning(f"Task {task.get('id')} missing expected_output")
                return False

        return True

    def _mock_review(self, plan: Dict, context: Dict) -> PlanReview:
        """
        Mock review for testing (when no LLM available).

        Provides optimistic scores that usually pass.
        """
        # Check structural requirements
        structural_ok = self._meets_structural_requirements(plan)

        # Base scores (slightly above minimum)
        base_score = 7.5 if structural_ok else 6.0

        scores = {
            'specificity': base_score,
            'relevance': base_score,
            'novelty': base_score - 0.5,
            'coverage': base_score,
            'feasibility': base_score + 0.5
        }

        avg_score = sum(scores.values()) / len(scores)
        min_score = min(scores.values())

        approved = (
            avg_score >= self.min_average_score
            and min_score >= self.min_dimension_score
            and structural_ok
        )

        required_changes = []
        if not structural_ok:
            required_changes.append("Fix structural requirements (3+ data_analysis, 2+ task types)")

        return PlanReview(
            approved=approved,
            scores=scores,
            average_score=avg_score,
            min_score=min_score,
            feedback=f"Mock review: {'APPROVED' if approved else 'NEEDS REVISION'} (avg: {avg_score:.1f})",
            required_changes=required_changes,
            suggestions=["This is a mock review (no LLM client provided)"]
        )

    def get_approval_statistics(self, reviews: List[PlanReview]) -> Dict:
        """
        Compute statistics over batch of reviews.

        Args:
            reviews: List of PlanReview objects

        Returns:
            Dictionary with approval statistics
        """
        if not reviews:
            return {}

        approved_count = sum(1 for r in reviews if r.approved)
        total = len(reviews)

        # Average scores per dimension
        avg_scores = {}
        for dim in ['specificity', 'relevance', 'novelty', 'coverage', 'feasibility']:
            avg_scores[f'avg_{dim}'] = sum(
                r.scores.get(dim, 0) for r in reviews
            ) / total

        return {
            'total_reviewed': total,
            'approved': approved_count,
            'rejected': total - approved_count,
            'approval_rate': approved_count / total,
            'avg_overall_score': sum(r.average_score for r in reviews) / total,
            **avg_scores
        }

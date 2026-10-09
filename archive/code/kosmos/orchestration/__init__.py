"""
Orchestration Module for Kosmos.

Implements strategic research planning and task execution orchestration.

Gap addressed: Gap 2 (Task Generation Strategy)
Pattern source: kosmos-karpathy (Plan Creator/Reviewer orchestration)

Key Components:
1. PlanCreatorAgent: Generates 10 strategic tasks per cycle
2. PlanReviewerAgent: Validates plans on 5 dimensions before execution
3. DelegationManager: Executes approved plans by routing to specialized agents
4. NoveltyDetector: Prevents redundant tasks using semantic similarity
5. Instructions: YAML configuration for agent prompts

Orchestration Flow:
    Context → Plan Creator → Novelty Check → Plan Reviewer →
    Delegation Manager → State Manager
    ↑                                                           ↓
    └─────────────── Update for next cycle ────────────────────┘

Performance Targets:
- Plan approval rate: ~80% on first submission
- Task success rate: ~90% completion
- Novelty threshold: 75% similarity = redundant
"""

from .plan_creator import PlanCreatorAgent
from .plan_reviewer import PlanReviewerAgent
from .delegation import DelegationManager
from .novelty_detector import NoveltyDetector

__all__ = [
    "PlanCreatorAgent",
    "PlanReviewerAgent",
    "DelegationManager",
    "NoveltyDetector",
]

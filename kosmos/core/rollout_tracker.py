"""
Rollout Tracker - Track agent rollouts matching paper metrics (Issue #58).

Paper Claim: "Agent Rollouts: ~200 total (~166 data analysis, ~36 literature)"

This module tracks per-agent-type rollout counts to match the paper's metrics.
"""

from dataclasses import dataclass, field
from typing import Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class RolloutTracker:
    """
    Track agent rollouts matching paper metrics.

    Paper states '166 data analysis agent rollouts and 36 literature review
    agent rollouts' (202 total) as average per research run.

    Attributes:
        data_analysis: Count of DataAnalystAgent rollouts
        literature: Count of literature review/synthesis rollouts
        hypothesis_generation: Count of HypothesisGeneratorAgent rollouts
        experiment_design: Count of ExperimentDesignerAgent rollouts
        code_execution: Count of code execution rollouts
    """
    data_analysis: int = 0
    literature: int = 0
    hypothesis_generation: int = 0
    experiment_design: int = 0
    code_execution: int = 0

    @property
    def total(self) -> int:
        """Get total rollout count across all agent types."""
        return sum([
            self.data_analysis,
            self.literature,
            self.hypothesis_generation,
            self.experiment_design,
            self.code_execution
        ])

    def increment(self, agent_type: str) -> None:
        """
        Increment rollout count for a specific agent type.

        Args:
            agent_type: One of 'data_analysis', 'literature', 'hypothesis_generation',
                       'experiment_design', 'code_execution'
        """
        if agent_type == "data_analysis":
            self.data_analysis += 1
        elif agent_type == "literature":
            self.literature += 1
        elif agent_type == "hypothesis_generation":
            self.hypothesis_generation += 1
        elif agent_type == "experiment_design":
            self.experiment_design += 1
        elif agent_type == "code_execution":
            self.code_execution += 1
        else:
            logger.warning(f"Unknown rollout type: {agent_type}")

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary for serialization."""
        return {
            "data_analysis": self.data_analysis,
            "literature": self.literature,
            "hypothesis_generation": self.hypothesis_generation,
            "experiment_design": self.experiment_design,
            "code_execution": self.code_execution,
            "total": self.total
        }

    def summary(self) -> str:
        """
        Get human-readable summary matching paper format.

        Returns:
            String like "166 data analysis + 36 literature = 202 total rollouts"
        """
        return (
            f"{self.data_analysis} data analysis + "
            f"{self.literature} literature = "
            f"{self.total} total rollouts"
        )

    def reset(self) -> None:
        """Reset all counters to zero."""
        self.data_analysis = 0
        self.literature = 0
        self.hypothesis_generation = 0
        self.experiment_design = 0
        self.code_execution = 0

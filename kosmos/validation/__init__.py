"""
Discovery Validation Module for Kosmos.

Implements ScholarEval 8-dimension validation framework to filter
low-quality discoveries before they enter the State Manager.

Gap addressed: Gap 5 (Discovery Evaluation & Filtering)
Pattern source: kosmos-claude-scientific-writer (ScholarEval)

Key Insight: Multi-dimensional quality control catches different failure modes.

8 Evaluation Dimensions:
1. Novelty (0-1): Is this discovery new?
2. Rigor (0-1): Are methods scientifically sound?
3. Clarity (0-1): Is the finding clearly stated?
4. Reproducibility (0-1): Can others reproduce this?
5. Impact (0-1): How important is this?
6. Coherence (0-1): Does it fit with existing knowledge?
7. Limitations (0-1): Are limitations acknowledged?
8. Ethics (0-1): Are ethical concerns addressed?

Approval Thresholds:
- Overall score ≥ 0.75 (75%)
- Rigor score ≥ 0.70 (minimum quality bar)

Target: ~75% validation rate (typical for good research)
"""

from .scholar_eval import ScholarEvalValidator, ScholarEvalScore
from .null_model import NullModelValidator, NullModelResult

__all__ = [
    "ScholarEvalValidator",
    "ScholarEvalScore",
    "NullModelValidator",
    "NullModelResult",
]

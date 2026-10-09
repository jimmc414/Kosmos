"""
Workflow Module for Kosmos.

Integrates all 6 gap implementations into a cohesive research workflow.

Components Integrated:
- Gap 0: Context Compression (manages context within LLM limits)
- Gap 1: State Manager (maintains research state across cycles)
- Gap 2: Task Generation (strategic planning and orchestration)
- Gap 3: Agent Integration (skill loading and domain expertise)
- Gap 4: Language/Tooling (Python-first approach - documented)
- Gap 5: Discovery Validation (ScholarEval filtering)

Research Loop Flow:
1. Context Retrieval → Get current state from State Manager
2. Plan Creation → Generate 10 strategic tasks
3. Novelty Check → Filter redundant tasks
4. Plan Review → Validate plan quality
5. Task Execution → Delegate to specialized agents
6. Validation → ScholarEval filtering
7. State Update → Save validated findings
8. Compression → Compress cycle results
9. Repeat for 20 cycles

Performance Targets:
- Complete 20 cycles autonomously
- 20:1 context compression
- 80% plan approval rate
- 90% task completion rate
- 75% finding validation rate
"""

from .research_loop import ResearchWorkflow

__all__ = [
    "ResearchWorkflow",
]

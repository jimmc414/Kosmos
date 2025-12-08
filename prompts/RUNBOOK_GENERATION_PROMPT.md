# Kosmos Operational Runbook Generation

You are tasked with creating a comprehensive operational runbook for the Kosmos autonomous AI scientist system. This runbook will be used for E2E testing and operational documentation.

## Your Context

### The Paper's Claims

Kosmos is an autonomous AI scientist that:
- Runs 20 research cycles over 10-15 hours
- Processes 1,500 papers and generates 42,000 lines of code per run
- Achieves 79.4% statement accuracy
- Produced 7 validated scientific discoveries
- Uses a "Structured World Model" (State Manager) as its core advancement

### The 6 Core Components

1. **Discovery Orchestrator**: Manages 20 cycles, dispatches 10 tasks/cycle, coordinates 200+ agent rollouts
2. **State Manager (World Model)**: Central knowledge repository enabling coherence across rollouts
3. **Task Generator**: Produces 10 prioritized tasks per cycle based on State Manager state
4. **Data Analysis Agent**: Writes/executes code in Jupyter notebooks (85.5% accuracy)
5. **Literature Search Agent**: Searches/synthesizes 1,500 papers (82.1% accuracy)
6. **Report Synthesizer**: Consolidates findings into publication-quality reports

### The 5 Implementation Gaps (and Solutions)

| Gap | Problem | Solution |
|-----|---------|----------|
| Gap 0 | Context compression (1500 papers + 42K lines won't fit in context) | ContextCompressor: 20x compression via hierarchical summarization |
| Gap 1 | State Manager schema undefined | Hybrid architecture: JSON artifacts + Neo4j knowledge graph |
| Gap 2 | Task generation strategy undefined | Plan Creator + Plan Reviewer + Novelty Detector orchestration |
| Gap 3 | Agent integration undefined | SkillLoader with 120+ domain-specific scientific skills |
| Gap 4 | R vs Python ambiguity | Python-only with Docker sandbox |
| Gap 5 | Discovery validation undefined | ScholarEval 8-dimension validation framework |

### The Codebase Structure

```
kosmos/
├── agents/
│   ├── research_director.py    # Main orchestrator (ResearchDirectorAgent)
│   ├── hypothesis_generator.py # Generates hypotheses from research question
│   ├── experiment_designer.py  # Designs experimental protocols
│   ├── data_analyst.py         # Interprets experiment results
│   ├── skill_loader.py         # Loads domain-specific skills (Gap 3)
│   └── base.py                 # Base agent class
├── orchestration/
│   ├── plan_creator.py         # Strategic task generation (Gap 2)
│   ├── plan_reviewer.py        # Plan quality validation
│   ├── delegation.py           # Task delegation
│   └── novelty_detector.py     # Prevents redundant work
├── compression/
│   └── compressor.py           # Context compression (Gap 0)
├── world_model/
│   ├── artifacts.py            # ArtifactStateManager (Gap 1)
│   └── simple.py               # SimpleWorldModel
├── knowledge/
│   ├── graph.py                # Neo4j knowledge graph (Gap 1)
│   └── graph_builder.py        # Graph construction
├── execution/
│   ├── executor.py             # Code execution
│   ├── code_generator.py       # Generates experiment code
│   └── sandbox.py              # Docker sandbox (Gap 4)
├── validation/
│   └── scholar_eval.py         # ScholarEval validation (Gap 5)
├── workflow/
│   └── research_loop.py        # ResearchWorkflow
├── literature/
│   ├── semantic_scholar.py     # Literature search
│   └── literature_analyzer.py  # Paper analysis
├── core/
│   ├── llm.py                  # LLM client abstraction
│   ├── workflow.py             # WorkflowState machine
│   └── metrics.py              # Budget tracking
└── cli/
    └── commands/run.py         # `kosmos run` CLI
```

### Current Known Issues

1. **CLI Hangs**: `kosmos run` hangs because ResearchDirectorAgent uses message-passing but no agent runtime processes messages
2. **SkillLoader Broken**: Returns None because COMMON_SKILLS references non-existent files (see `docs/ISSUE_SKILLLOADER_BROKEN.md`)
3. **Agent Timing**: Individual agents work but are slow (19-89 seconds per call due to LLM latency)

## Your Task

Generate a **detailed operational runbook** that covers:

### Section 1: System Architecture Overview
- How the 6 components interact
- Data flow between components
- State management patterns

### Section 2: Research Cycle Lifecycle
For each of the 20 cycles, document:
1. Cycle initialization
2. Task generation (10 tasks)
3. Task execution (hypothesis → experiment → analysis)
4. State update
5. Convergence check

### Section 3: Component Operations

For EACH component, provide:
- **Purpose**: What it does
- **Inputs**: What it receives
- **Outputs**: What it produces
- **Entry Point**: The function/method to call
- **Expected Timing**: How long it takes
- **Success Criteria**: How to verify it worked
- **Failure Modes**: What can go wrong

### Section 4: Step-by-Step E2E Flow

Provide a numbered sequence of operations for a complete research run:

1. User provides research question and domain
2. System initializes...
3. ...
N. System produces final report

For each step include:
- The exact function/method called
- Expected inputs and outputs
- Validation checkpoint
- Estimated time

### Section 5: Validation Checkpoints

Define checkpoints to verify the system is working:

| Checkpoint | Location | Success Criteria | Failure Action |
|------------|----------|------------------|----------------|
| CP1: Initialization | After ResearchDirectorAgent.__init__ | workflow.current_state == INITIALIZING | Check config |
| CP2: First Hypothesis | After generate_hypotheses() | len(hypotheses) > 0 | Check LLM connection |
| ... | ... | ... | ... |

### Section 6: Configuration Requirements

Document all required configuration:
- Environment variables
- API keys
- Optional services (Neo4j, Redis)
- Performance tuning

### Section 7: Troubleshooting Guide

For each known failure mode:
- Symptom
- Root cause
- Diagnostic steps
- Resolution

## Output Format

Produce a single Markdown document with:
- Clear section headers
- Tables for structured data
- Code blocks for function signatures
- Mermaid diagrams for flows (if helpful)

## Files to Read

To complete this task, read these files:
1. `archive/implementation/OPEN_QUESTIONS.md` - Original gap analysis
2. `archive/implementation/OPENQUESTIONS_SOLUTION.md` - Gap solutions
3. `docs/E2E_DIAGNOSTIC.md` - Current system state
4. `docs/ISSUE_SKILLLOADER_BROKEN.md` - Gap 3 status
5. `kosmos/agents/research_director.py` - Main orchestrator
6. `kosmos/workflow/research_loop.py` - Research workflow
7. `kosmos/orchestration/plan_creator.py` - Task generation
8. `kosmos/core/workflow.py` - State machine

## Critical Directive: Go Beyond This Prompt

This prompt provides a starting framework, but **you must analyze the original sources** to identify anything missing:

1. **Read the Kosmos paper** (if available) or `archive/implementation/OPEN_QUESTIONS.md` which summarizes it
2. **Compare paper claims vs. implementation** - Are there features described in the paper that aren't reflected in the codebase?
3. **Identify undocumented behaviors** - What does the code do that isn't captured in existing docs?
4. **Surface implicit assumptions** - What does the system assume about inputs, environment, or user behavior?
5. **Find the gaps** - What would a developer need to know that isn't written anywhere?

### Specific Questions to Answer

- Does the paper describe any agent behaviors not implemented?
- Are there evaluation metrics from the paper (79.4% accuracy, etc.) that should have validation code?
- What about the "7 validated discoveries" - is there a discovery validation workflow?
- The paper mentions "166 data analysis rollouts" and "200 agent rollouts" - is parallelism implemented?
- How does the system handle the "12-hour runtime" constraint mentioned in the paper?
- What termination criteria does the paper specify vs. what's implemented?

### Add a "Paper vs. Implementation" Section

Include a section in the runbook that documents:

| Paper Claim | Implementation Status | Gap Description |
|-------------|----------------------|-----------------|
| 20 research cycles | Implemented | `max_iterations` config |
| 1,500 papers processed | Partial | Literature search works but scale untested |
| ... | ... | ... |

## Success Criteria for the Runbook

The runbook is complete when:
1. A developer can follow it to run a complete research cycle
2. Each component has clear inputs/outputs/success criteria
3. All validation checkpoints are defined
4. Known failure modes have troubleshooting steps
5. The document can serve as an operational reference
6. **Paper vs. implementation gaps are documented**
7. **No implicit assumptions remain undocumented**

## Output Destination

Save the generated runbook to: `docs/OPERATIONAL_RUNBOOK.md`

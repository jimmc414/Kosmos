# Kosmos Operational Runbook

**Version**: 0.2.0-alpha
**Last Updated**: 2025-12-08
**Purpose**: Complete operational guide for running the Kosmos autonomous AI scientist system

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Research Cycle Lifecycle](#2-research-cycle-lifecycle)
3. [Component Operations](#3-component-operations)
4. [Step-by-Step E2E Flow](#4-step-by-step-e2e-flow)
5. [Validation Checkpoints](#5-validation-checkpoints)
6. [Configuration Requirements](#6-configuration-requirements)
7. [Troubleshooting Guide](#7-troubleshooting-guide)
8. [Paper vs. Implementation](#8-paper-vs-implementation)

---

## 1. System Architecture Overview

### 1.1 The 6 Core Components

Kosmos implements a multi-agent orchestration pattern where parallel agents coordinate through a central State Manager over iterative research cycles.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DISCOVERY ORCHESTRATOR                             │
│                   (ResearchDirectorAgent)                               │
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │  Hypothesis │  │  Experiment │  │    Data     │  │  Literature │   │
│  │  Generator  │  │  Designer   │  │   Analyst   │  │  Analyzer   │   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘   │
│         │                │                │                │           │
│         └────────────────┴────────────────┴────────────────┘           │
│                                   │                                     │
│                                   ▼                                     │
│                    ┌──────────────────────────────┐                    │
│                    │    STATE MANAGER (World Model)│                    │
│                    │  ┌─────────┐  ┌─────────────┐│                    │
│                    │  │ JSON    │  │ Knowledge   ││                    │
│                    │  │Artifacts│  │ Graph (Neo4j)│                    │
│                    │  └─────────┘  └─────────────┘│                    │
│                    └──────────────────────────────┘                    │
│                                   │                                     │
│                                   ▼                                     │
│                    ┌──────────────────────────────┐                    │
│                    │        TASK GENERATOR        │                    │
│                    │  ┌─────────┐  ┌───────────┐ │                    │
│                    │  │  Plan   │  │   Plan    │ │                    │
│                    │  │ Creator │  │ Reviewer  │ │                    │
│                    │  └─────────┘  └───────────┘ │                    │
│                    └──────────────────────────────┘                    │
│                                   │                                     │
│                                   ▼                                     │
│                    ┌──────────────────────────────┐                    │
│                    │     VALIDATION FRAMEWORK     │                    │
│                    │       (ScholarEval)          │                    │
│                    └──────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Interaction Summary

| Component | Receives From | Sends To | Primary Function |
|-----------|--------------|----------|------------------|
| **Discovery Orchestrator** | User input, all agents | All agents | Master coordination |
| **State Manager** | All agents | Task Generator, Report Synthesizer | Knowledge persistence |
| **Task Generator** | State Manager | Discovery Orchestrator | Strategic task generation |
| **Hypothesis Generator** | Orchestrator | State Manager | Generate testable hypotheses |
| **Experiment Designer** | Orchestrator | Executor | Design experimental protocols |
| **Data Analyst** | Executor results | State Manager | Interpret experimental results |
| **Literature Analyzer** | Orchestrator | State Manager | Search/synthesize papers |
| **ScholarEval** | State Manager findings | State Manager | Validate discoveries |

### 1.3 Data Flow

```
Research Question
       │
       ▼
┌──────────────────┐
│  Task Generation │ ← State Manager context
│ (Plan Creator)   │
└────────┬─────────┘
         │ 10 tasks/cycle
         ▼
┌──────────────────┐
│  Plan Review     │ ← 5-dimension scoring
│ (Plan Reviewer)  │
└────────┬─────────┘
         │ Approved plan
         ▼
┌──────────────────┐
│  Task Execution  │ ← Parallel execution
│ (Delegation Mgr) │
└────────┬─────────┘
         │ Task results
         ▼
┌──────────────────┐
│  Context         │ ← 20:1 compression
│  Compression     │
└────────┬─────────┘
         │ Compressed summaries
         ▼
┌──────────────────┐
│  ScholarEval     │ ← 8-dimension validation
│  Validation      │
└────────┬─────────┘
         │ Validated findings
         ▼
┌──────────────────┐
│  State Manager   │ → Next cycle context
│  Update          │
└──────────────────┘
```

### 1.4 State Management Pattern

Kosmos uses a **4-layer hybrid architecture**:

1. **Layer 1: JSON Artifacts** (Always Active)
   - Human-readable storage
   - Path: `artifacts/cycle_N/task_M_finding.json`
   - Complete traceability

2. **Layer 2: Knowledge Graph** (Optional, Neo4j)
   - Structural queries
   - Relationship traversal
   - Entity: Finding, Hypothesis, Evidence

3. **Layer 3: Vector Store** (Optional, ChromaDB)
   - Semantic similarity search
   - Novelty detection

4. **Layer 4: Citation Tracking** (Integrated)
   - Evidence chains
   - Claim → source links

---

## 2. Research Cycle Lifecycle

### 2.1 Workflow State Machine

```
INITIALIZING → GENERATING_HYPOTHESES → DESIGNING_EXPERIMENTS → EXECUTING
                                                                    │
                                                                    ▼
CONVERGED ← REFINING ← ANALYZING ← ─────────────────────────────────┘
```

**States**:
| State | Description | Exit Condition |
|-------|-------------|----------------|
| `INITIALIZING` | System startup, config validation | Config validated |
| `GENERATING_HYPOTHESES` | Create testable hypotheses | ≥1 hypothesis generated |
| `DESIGNING_EXPERIMENTS` | Create experimental protocols | Protocol designed |
| `EXECUTING` | Run experiments in sandbox | Execution complete |
| `ANALYZING` | Interpret results | Analysis complete |
| `REFINING` | Refine hypotheses based on results | Refinement complete |
| `CONVERGED` | Research complete | N/A (terminal) |
| `ERROR` | Error recovery mode | Recovered or aborted |
| `PAUSED` | User-initiated pause | Resume command |

### 2.2 Single Cycle Structure (1 of 20)

Each research cycle follows this sequence:

```python
async def _execute_cycle(self, cycle: int, num_tasks: int = 10):
    # Step 1: Get context from State Manager (lookback=3 cycles)
    context = self.state_manager.get_cycle_context(cycle, lookback=3)

    # Step 2: Plan Creator generates 10 tasks
    plan = self.plan_creator.create_plan(
        research_objective=self.research_objective,
        context=context,
        num_tasks=num_tasks
    )

    # Step 3: Novelty Detector checks for redundancy
    novelty = self.novelty_detector.check_plan_novelty(plan)

    # Step 4: Plan Reviewer validates quality (5 dimensions)
    review = self.plan_reviewer.review_plan(plan, context)

    # Step 5: If rejected, attempt revision (once)
    if not review.approved:
        plan = self.plan_creator.revise_plan(plan, review, context)
        review = self.plan_reviewer.review_plan(plan, context)

    # Step 6: Delegation Manager executes approved tasks
    results = await self.delegation_manager.execute_plan(plan, cycle)

    # Step 7: ScholarEval validates each finding
    for task_result in results.completed_tasks:
        eval_score = self.scholar_eval.evaluate_finding(task_result.finding)
        if eval_score.passes_threshold:
            await self.state_manager.save_finding_artifact(cycle, task_result)

    # Step 8: Compress cycle results for context
    compressed = self.context_compressor.compress_cycle_results(cycle, results)

    # Step 9: Generate cycle summary
    await self.state_manager.generate_cycle_summary(cycle)

    return cycle_result
```

### 2.3 Exploration vs. Exploitation Balance

Task generation adapts based on cycle number:

| Cycles | Exploration | Exploitation | Strategy |
|--------|-------------|--------------|----------|
| 1-7    | 70%         | 30%          | Find new directions |
| 8-14   | 50%         | 50%          | Balanced |
| 15-20  | 30%         | 70%          | Deepen findings |

### 2.4 Convergence Criteria

Research terminates when any of these conditions are met:

1. **Iteration limit**: `iteration_count >= max_iterations` (default: 20)
2. **No testable hypotheses**: Empty hypothesis pool after generation attempt
3. **Budget exceeded**: API costs exceed `RESEARCH_BUDGET_USD`
4. **No novel tasks**: Novelty score drops below threshold for 3 consecutive cycles
5. **User-initiated stop**: Manual convergence trigger

---

## 3. Component Operations

### 3.1 Research Director Agent

**File**: `kosmos/agents/research_director.py`

| Property | Value |
|----------|-------|
| **Purpose** | Master orchestrator coordinating all agents |
| **Entry Point** | `ResearchDirectorAgent.execute({"action": "start_research"})` |
| **Expected Timing** | 10-15 hours for 20 cycles |
| **Memory** | ~500MB base + ~50MB per cycle |

**Inputs**:
```python
director = ResearchDirectorAgent(
    research_question="Does X affect Y?",  # Required
    domain="biology",                       # Optional
    config={
        "max_iterations": 20,
        "enable_concurrent_operations": True,
        "max_concurrent_experiments": 4
    }
)
```

**Outputs**:
```python
{
    "status": "research_started" | "step_executed" | "converged",
    "research_plan": "LLM-generated strategy",
    "next_action": "generate_hypothesis" | "design_experiment" | ...
}
```

**Success Criteria**:
- `workflow.current_state != WorkflowState.ERROR`
- `research_plan.has_converged == True` at completion
- `len(research_plan.supported_hypotheses) > 0`

**Failure Modes**:
| Failure | Symptom | Recovery |
|---------|---------|----------|
| LLM timeout | No response after 120s | Retry with exponential backoff |
| No hypotheses generated | Empty hypothesis pool | Check LLM connection, domain config |
| Budget exceeded | `BudgetExceededError` | Graceful convergence |
| Infinite loop | Actions > 50 per iteration | Force convergence |

### 3.2 Hypothesis Generator Agent

**File**: `kosmos/agents/hypothesis_generator.py`

| Property | Value |
|----------|-------|
| **Purpose** | Generate testable hypotheses from research question |
| **Entry Point** | `agent.generate_hypotheses(research_question, domain)` |
| **Expected Timing** | ~19 seconds per call |
| **LLM Calls** | 1-2 per invocation |

**Inputs**:
```python
result = agent.generate_hypotheses(
    research_question="Does KRAS mutations drive metabolic reprogramming?",
    domain="biology",
    num_hypotheses=5,            # Default: 5
    literature_context=None      # Optional: prior papers
)
```

**Outputs**:
```python
HypothesisGenerationResult(
    hypotheses=[
        Hypothesis(
            statement="KRAS mutations...",
            rationale="Based on...",
            testability_score=0.85,
            novelty_score=0.72
        ),
        ...
    ],
    research_context="Literature summary",
    count=5
)
```

**Success Criteria**:
- `len(result.hypotheses) > 0`
- All hypotheses have `testability_score >= 0.5`
- Each hypothesis has a non-empty `statement` and `rationale`

**Failure Modes**:
| Failure | Symptom | Recovery |
|---------|---------|----------|
| LLM parse error | Invalid JSON response | Retry with simplified prompt |
| No novel hypotheses | All filtered by novelty | Lower novelty threshold |
| Domain mismatch | Hypotheses off-topic | Add domain context |

### 3.3 Experiment Designer Agent

**File**: `kosmos/agents/experiment_designer.py`

| Property | Value |
|----------|-------|
| **Purpose** | Design experimental protocols to test hypotheses |
| **Entry Point** | `agent.design_experiment(hypothesis)` |
| **Expected Timing** | ~89 seconds per call |
| **LLM Calls** | 3-5 per invocation (multi-pass) |

**Inputs**:
```python
protocol = agent.design_experiment(
    hypothesis=hypothesis_object,
    domain="biology",
    available_datasets=["GSE12345", "GSE67890"],
    constraints={"max_runtime": 300}
)
```

**Outputs**:
```python
ExperimentProtocol(
    protocol_id="proto_abc123",
    hypothesis_id="hyp_xyz789",
    description="Differential expression analysis...",
    methodology="Use PyDESeq2 to...",
    statistical_tests=["t-test", "FDR correction"],
    expected_results="Significant DEGs in...",
    code_template="import pydeseq2..."
)
```

**Success Criteria**:
- `protocol.methodology` is non-empty
- `protocol.statistical_tests` contains valid tests
- Protocol is executable in sandbox

### 3.4 Data Analyst Agent

**File**: `kosmos/agents/data_analyst.py`

| Property | Value |
|----------|-------|
| **Purpose** | Interpret experiment results and update hypotheses |
| **Entry Point** | `agent.interpret_results(result, hypothesis)` |
| **Expected Timing** | ~30-60 seconds per call |
| **LLM Calls** | 1-2 per invocation |

**Inputs**:
```python
interpretation = agent.interpret_results(
    result=ExperimentResult(...),
    hypothesis=Hypothesis(...)
)
```

**Outputs**:
```python
ResultInterpretation(
    result_id="res_123",
    hypothesis_supported=True | False | None,
    confidence=0.85,
    p_value=0.001,
    effect_size=2.3,
    key_findings=["Finding 1", "Finding 2"],
    statistical_summary="...",
    next_steps=["Validate with...", "Extend to..."]
)
```

**Success Criteria**:
- `interpretation.confidence >= 0.6`
- `interpretation.hypothesis_supported` is not None (unless truly inconclusive)
- Statistical measures are valid

### 3.5 Plan Creator Agent

**File**: `kosmos/orchestration/plan_creator.py`

| Property | Value |
|----------|-------|
| **Purpose** | Generate 10 strategic tasks per cycle |
| **Entry Point** | `agent.create_plan(research_objective, context, num_tasks)` |
| **Expected Timing** | ~10-15 seconds per call |
| **LLM Calls** | 1 per invocation |

**Inputs**:
```python
plan = agent.create_plan(
    research_objective="Investigate KRAS in cancer",
    context={
        "cycle": 5,
        "recent_findings": [...],
        "unsupported_hypotheses": [...]
    },
    num_tasks=10
)
```

**Outputs**:
```python
ResearchPlan(
    cycle=5,
    tasks=[
        Task(
            task_id=1,
            task_type="data_analysis",
            description="...",
            expected_output="...",
            required_skills=["scanpy", "pydeseq2"],
            exploration=True
        ),
        ...  # 10 tasks total
    ],
    rationale="Strategic reasoning...",
    exploration_ratio=0.50
)
```

**Success Criteria**:
- `len(plan.tasks) == 10`
- At least 3 `data_analysis` tasks
- At least 2 different task types
- Exploration/exploitation ratio matches cycle

### 3.6 Plan Reviewer Agent

**File**: `kosmos/orchestration/plan_reviewer.py`

| Property | Value |
|----------|-------|
| **Purpose** | Validate plan quality before execution |
| **Entry Point** | `agent.review_plan(plan, context)` |
| **Expected Timing** | ~8-12 seconds per call |
| **LLM Calls** | 1 per invocation |

**5-Dimension Scoring**:
| Dimension | Weight | Description |
|-----------|--------|-------------|
| Specificity | 20% | Are tasks concrete and executable? |
| Relevance | 20% | Do tasks address research objective? |
| Novelty | 20% | Do tasks avoid redundancy? |
| Coverage | 20% | Do tasks cover important aspects? |
| Feasibility | 20% | Are tasks achievable? |

**Approval Thresholds**:
- Average score ≥ 7.0/10
- Minimum dimension score ≥ 5.0/10
- Structural requirements met (≥3 data_analysis, ≥2 types)

### 3.7 ScholarEval Validator

**File**: `kosmos/validation/scholar_eval.py`

| Property | Value |
|----------|-------|
| **Purpose** | Peer-review style validation of discoveries |
| **Entry Point** | `validator.evaluate_finding(finding)` |
| **Expected Timing** | ~5-10 seconds per finding |
| **LLM Calls** | 1 per invocation |

**8-Dimension Scoring**:
| Dimension | Weight | Description |
|-----------|--------|-------------|
| Rigor | 25% | Scientific soundness of methods |
| Impact | 20% | Importance of finding |
| Novelty | 15% | Is this finding new? |
| Reproducibility | 15% | Can others reproduce this? |
| Clarity | 10% | Is finding clearly stated? |
| Coherence | 10% | Does it fit existing knowledge? |
| Limitations | 3% | Are limitations acknowledged? |
| Ethics | 2% | Ethical considerations addressed? |

**Approval Thresholds**:
- Overall score ≥ 0.75 (75%)
- Rigor score ≥ 0.70 (minimum quality bar)

### 3.8 Context Compressor

**File**: `kosmos/compression/compressor.py`

| Property | Value |
|----------|-------|
| **Purpose** | Compress research artifacts for context windows |
| **Entry Point** | `compressor.compress_notebook(notebook_path)` |
| **Expected Timing** | ~2-5 seconds per notebook |
| **Compression Ratio** | 20:1 average |

**Compression Tiers**:
1. **Task-Level**: 42K lines → 2-line summary + statistics
2. **Cycle-Level**: 10 task summaries → 1 cycle overview
3. **Final Synthesis**: 20 cycle overviews → Research narrative

### 3.9 Skill Loader

**File**: `kosmos/agents/skill_loader.py`

| Property | Value |
|----------|-------|
| **Purpose** | Load domain-specific scientific skills |
| **Entry Point** | `loader.load_skills_for_task(task_type, domain)` |
| **Available Skills** | 116 domain-specific skills |

**Skill Bundles**:
| Bundle | Skills Included |
|--------|-----------------|
| `single_cell_analysis` | scanpy, anndata, scvi-tools, cellxgene |
| `genomics_analysis` | biopython, pysam, pydeseq2, biomart |
| `drug_discovery` | rdkit, datamol, deepchem, chembl-database |
| `proteomics` | pyopenms, matchms |
| `clinical_research` | clinvar, clinicaltrials, omim-database |

**Known Issue**: SkillLoader returns None because `COMMON_SKILLS` references non-existent files. See `docs/ISSUE_SKILLLOADER_BROKEN.md`.

---

## 4. Step-by-Step E2E Flow

### 4.1 Complete Research Run Sequence

```
Step 1:  User provides research question
         ├── Input: "Does KRAS mutations drive metabolic reprogramming?"
         ├── Function: CLI or Python API
         └── Validation: Question is non-empty string

Step 2:  System initializes ResearchDirectorAgent
         ├── Function: ResearchDirectorAgent.__init__()
         ├── Actions:
         │   ├── Validate domain against enabled domains
         │   ├── Load domain-specific skills
         │   ├── Initialize workflow state machine
         │   ├── Initialize database connection
         │   └── Connect to knowledge graph (optional)
         ├── Time: ~2-5 seconds
         └── Checkpoint: workflow.current_state == INITIALIZING

Step 3:  Generate initial research plan
         ├── Function: director.generate_research_plan()
         ├── LLM Call: 1
         ├── Output: High-level research strategy
         ├── Time: ~10 seconds
         └── Checkpoint: research_plan.initial_strategy is non-empty

Step 4:  Transition to GENERATING_HYPOTHESES
         ├── Function: workflow.transition_to(GENERATING_HYPOTHESES)
         └── Checkpoint: workflow.current_state == GENERATING_HYPOTHESES

Step 5:  Generate hypotheses (via message-passing or direct call)
         ├── Function: HypothesisGeneratorAgent.generate_hypotheses()
         ├── LLM Calls: 1-2
         ├── Output: 3-5 testable hypotheses
         ├── Time: ~19 seconds
         └── Checkpoint: len(research_plan.hypothesis_pool) > 0

Step 6:  Persist hypotheses to knowledge graph
         ├── Function: director._persist_hypothesis_to_graph()
         ├── Creates: SPAWNED_BY relationship to research question
         └── Checkpoint: Hypotheses visible in Neo4j (if enabled)

Step 7:  Decide next action
         ├── Function: director.decide_next_action()
         ├── Decision tree based on workflow state
         └── Returns: NextAction.DESIGN_EXPERIMENT

Step 8:  Transition to DESIGNING_EXPERIMENTS
         ├── Function: workflow.transition_to(DESIGNING_EXPERIMENTS)
         └── Checkpoint: workflow.current_state == DESIGNING_EXPERIMENTS

Step 9:  Design experiment for first untested hypothesis
         ├── Function: ExperimentDesignerAgent.design_experiment()
         ├── LLM Calls: 3-5
         ├── Output: ExperimentProtocol with methodology
         ├── Time: ~89 seconds
         └── Checkpoint: len(research_plan.experiment_queue) > 0

Step 10: Persist protocol to knowledge graph
         ├── Function: director._persist_protocol_to_graph()
         ├── Creates: TESTS relationship to hypothesis
         └── Checkpoint: Protocol visible in Neo4j (if enabled)

Step 11: Transition to EXECUTING
         ├── Function: workflow.transition_to(EXECUTING)
         └── Checkpoint: workflow.current_state == EXECUTING

Step 12: Execute experiment in sandbox
         ├── Function: Executor.execute_experiment()
         ├── Environment: Docker sandbox (or direct exec if no Docker)
         ├── Time: Variable (up to MAX_EXPERIMENT_EXECUTION_TIME)
         └── Checkpoint: Execution completes without timeout

Step 13: Transition to ANALYZING
         ├── Function: workflow.transition_to(ANALYZING)
         └── Checkpoint: workflow.current_state == ANALYZING

Step 14: Analyze results
         ├── Function: DataAnalystAgent.interpret_results()
         ├── LLM Calls: 1-2
         ├── Output: ResultInterpretation with hypothesis_supported
         ├── Time: ~30-60 seconds
         └── Checkpoint: interpretation.hypothesis_supported is not None

Step 15: Update hypothesis status
         ├── Function: research_plan.mark_supported() or mark_rejected()
         └── Checkpoint: Hypothesis status updated in research_plan

Step 16: Transition to REFINING
         ├── Function: workflow.transition_to(REFINING)
         └── Checkpoint: workflow.current_state == REFINING

Step 17: Refine hypotheses based on results
         ├── Function: HypothesisRefiner.refine()
         ├── Output: Refined or spawned hypotheses
         └── Checkpoint: New hypotheses added to pool (if applicable)

Step 18: Increment iteration counter
         ├── Function: research_plan.increment_iteration()
         └── Checkpoint: iteration_count incremented

Step 19: Check convergence criteria
         ├── Function: director._should_check_convergence()
         ├── Criteria: iteration limit, no hypotheses, budget exceeded
         └── Decision: Continue (goto Step 4) or Converge (goto Step 20)

Step 20: Transition to CONVERGED
         ├── Function: workflow.transition_to(CONVERGED)
         ├── Triggers: director.stop()
         └── Checkpoint: workflow.current_state == CONVERGED

Step 21: Generate final report
         ├── Function: workflow.generate_report()
         ├── Output: Markdown report with validated findings
         └── Checkpoint: Report contains evidence-linked findings
```

### 4.2 Alternative Entry Points

**Using ResearchWorkflow (Recommended for E2E)**:
```python
import asyncio
from kosmos.workflow.research_loop import ResearchWorkflow

async def run():
    workflow = ResearchWorkflow(
        research_objective="Investigate KRAS mutations in cancer",
        artifacts_dir="./artifacts",
        max_cycles=5
    )

    result = await workflow.run(num_cycles=5, tasks_per_cycle=10)
    report = await workflow.generate_report()
    print(report)

asyncio.run(run())
```

**Using CLI**:
```bash
# Basic run
kosmos run --objective "Your research question" --max-iterations 5

# With trace logging
kosmos run --trace --objective "Your research question"

# With specific domain
kosmos run --objective "Your research question" --domain biology
```

**Using Direct Agent Invocation (Workaround for CLI hang)**:
```python
from kosmos.agents.hypothesis_generator import HypothesisGeneratorAgent

agent = HypothesisGeneratorAgent(config={})
result = agent.generate_hypotheses(
    research_question="Does X affect Y?",
    domain="biology"
)
print(f"Generated {len(result.hypotheses)} hypotheses")
```

---

## 5. Validation Checkpoints

### 5.1 Checkpoint Table

| Checkpoint | Location | Success Criteria | Failure Action |
|------------|----------|------------------|----------------|
| CP1: Init | After `ResearchDirectorAgent.__init__` | `workflow.current_state == INITIALIZING` | Check config, API keys |
| CP2: Skills | After `_load_skills()` | `self.skills is not None` or warning logged | Check skill files exist |
| CP3: Plan | After `generate_research_plan()` | `research_plan.initial_strategy != ""` | Check LLM connection |
| CP4: Hypotheses | After hypothesis generation | `len(hypothesis_pool) > 0` | Check domain, lower novelty |
| CP5: Protocol | After experiment design | `protocol.methodology != ""` | Check hypothesis clarity |
| CP6: Execution | After experiment run | `result.status == "completed"` | Check sandbox, timeout |
| CP7: Analysis | After result interpretation | `interpretation.confidence >= 0.5` | Check result data |
| CP8: Validation | After ScholarEval | `eval_score.passes_threshold == True` | Review finding quality |
| CP9: State | After cycle complete | `state_manager.get_cycle_findings(cycle) != []` | Check artifact storage |
| CP10: Convergence | At research end | `research_plan.has_converged == True` | Check iteration count |

### 5.2 Quick Health Check Commands

```bash
# System diagnostics
kosmos doctor

# Smoke test all components
python scripts/smoke_test.py

# Unit tests
pytest tests/unit/ -v --tb=short

# Integration tests (requires services)
pytest tests/integration/ -v --tb=short

# E2E tests
pytest tests/e2e/ -v --tb=short
```

### 5.3 Component Smoke Tests

```python
# Test individual components
from kosmos.compression import ContextCompressor
from kosmos.world_model.artifacts import ArtifactStateManager
from kosmos.orchestration import PlanCreatorAgent, PlanReviewerAgent, NoveltyDetector
from kosmos.validation import ScholarEvalValidator
from kosmos.agents import SkillLoader

# Context Compressor
compressor = ContextCompressor(None)
assert compressor is not None

# State Manager
state_manager = ArtifactStateManager(artifacts_dir="./test_artifacts")
assert state_manager is not None

# Orchestration
plan_creator = PlanCreatorAgent(None)
plan_reviewer = PlanReviewerAgent(None)
novelty_detector = NoveltyDetector()
assert plan_creator is not None

# Validation
validator = ScholarEvalValidator(None)
assert validator is not None

# Skill Loader
skill_loader = SkillLoader()
assert skill_loader is not None

print("All component smoke tests passed!")
```

---

## 6. Configuration Requirements

### 6.1 Required Environment Variables

```bash
# REQUIRED: At least one LLM provider
LLM_PROVIDER=anthropic           # Options: anthropic, openai, litellm
ANTHROPIC_API_KEY=sk-ant-...     # If using Anthropic
# OR
OPENAI_API_KEY=sk-...            # If using OpenAI
# OR
LITELLM_MODEL=ollama/llama3.1:8b # If using LiteLLM
```

### 6.2 Optional Environment Variables

```bash
# Database (defaults to SQLite)
DATABASE_URL=sqlite:///kosmos.db
# OR for production:
DATABASE_URL=postgresql://kosmos:password@localhost:5432/kosmos

# Neo4j Knowledge Graph (optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=kosmos-password

# Redis Cache (optional)
REDIS_ENABLED=true
REDIS_URL=redis://localhost:6379/0

# Budget Control
BUDGET_ENABLED=true
BUDGET_LIMIT_USD=10.00

# Research Configuration
MAX_RESEARCH_ITERATIONS=10
ENABLED_DOMAINS=biology,physics,chemistry,neuroscience

# Logging
LOG_LEVEL=INFO
DEBUG_MODE=false
DEBUG_LEVEL=0

# Literature APIs (optional, increases rate limits)
SEMANTIC_SCHOLAR_API_KEY=your-key-here
PUBMED_API_KEY=your-key-here

# Safety
ENABLE_SANDBOXING=true
MAX_EXPERIMENT_EXECUTION_TIME=300
```

### 6.3 Service Configuration

**Starting Optional Services with Docker**:
```bash
# Start all dev services (Neo4j, Redis, PostgreSQL)
docker compose --profile dev up -d

# Or individual services
docker compose up -d neo4j
docker compose up -d redis
docker compose up -d postgres

# Stop services
docker compose --profile dev down
```

**Service URLs**:
| Service | URL | Credentials |
|---------|-----|-------------|
| Neo4j Browser | http://localhost:7474 | neo4j / kosmos-password |
| PostgreSQL | localhost:5432 | kosmos / kosmos-dev-password |
| Redis | localhost:6379 | N/A |
| pgAdmin | http://localhost:5050 | admin@kosmos.local / admin |

### 6.4 Minimum System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 4GB | 8GB+ |
| CPU | 2 cores | 4+ cores |
| Disk | 10GB | 50GB+ |
| Python | 3.11+ | 3.11 |
| Docker | Optional | Recommended for sandbox |

### 6.5 Performance Tuning

```bash
# Enable concurrent operations for faster research
ENABLE_CONCURRENT_OPERATIONS=true
MAX_CONCURRENT_EXPERIMENTS=4
MAX_PARALLEL_HYPOTHESES=3
MAX_CONCURRENT_LLM_CALLS=5

# Adjust based on API tier
LLM_RATE_LIMIT_PER_MINUTE=50  # Tier 1: 50, Tier 2: 100, Tier 3: 200

# Memory optimization
PARALLEL_EXPERIMENTS=0  # Sequential for low memory
CACHE_TTL=3600          # 1 hour cache
```

---

## 7. Troubleshooting Guide

### 7.1 CLI Hangs After Banner

**Symptom**: `kosmos run` starts, shows banner, then hangs indefinitely.

**Root Cause**: `ResearchDirectorAgent` uses message-passing architecture but no agent runtime processes the messages. Messages are sent but never received.

**Diagnostic**:
```bash
timeout 30 kosmos run "Test question" --domain biology --max-iterations 1
# If times out, confirms the hang issue
```

**Solution**: Use direct agent invocation instead of CLI:
```python
from kosmos.agents.hypothesis_generator import HypothesisGeneratorAgent

agent = HypothesisGeneratorAgent(config={})
result = agent.generate_hypotheses(
    research_question="Your question",
    domain="biology"
)
```

Or use `ResearchWorkflow` directly:
```python
from kosmos.workflow.research_loop import ResearchWorkflow
import asyncio

workflow = ResearchWorkflow(research_objective="Your question")
result = asyncio.run(workflow.run(num_cycles=1))
```

### 7.2 SkillLoader Returns None

**Symptom**: Skills not being loaded, agents use generic prompts.

**Root Cause**: `COMMON_SKILLS` in `skill_loader.py` references non-existent files (pandas, numpy, matplotlib).

**Diagnostic**:
```python
from kosmos.agents.skill_loader import SkillLoader
loader = SkillLoader()
skills = loader.load_skills_for_task(task_type='research', domain='biology')
print(skills)  # None
```

**Solution**: Use domain-specific bundles instead:
```python
skills = loader.load_skills_for_task(
    task_type='single_cell_analysis'  # Use bundle name, not domain
)
```

See `docs/ISSUE_SKILLLOADER_BROKEN.md` for detailed fix instructions.

### 7.3 LLM Connection Errors

**Symptom**: `APIConnectionError`, `AuthenticationError`, or timeout errors.

**Diagnostic**:
```bash
# Check API key is set
echo $ANTHROPIC_API_KEY

# Test connection
python -c "from kosmos.core.llm import get_client; c = get_client(); print(c)"
```

**Solutions**:
| Error | Solution |
|-------|----------|
| `AuthenticationError` | Check API key is correct and has credits |
| `RateLimitError` | Reduce `MAX_CONCURRENT_LLM_CALLS` or add delays |
| `APIConnectionError` | Check internet connection, firewall |
| `Timeout` | Increase timeout or check API status |

### 7.4 Database Errors

**Symptom**: `OperationalError`, `IntegrityError`, or connection failures.

**Diagnostic**:
```bash
# Check database URL
echo $DATABASE_URL

# Test SQLite connection
python -c "from kosmos.db import init_from_config; init_from_config()"
```

**Solutions**:
| Error | Solution |
|-------|----------|
| `database is locked` | SQLite concurrent access issue; use PostgreSQL |
| `no such table` | Run migrations: `alembic upgrade head` |
| `connection refused` | Check PostgreSQL is running |

### 7.5 Neo4j Connection Issues

**Symptom**: Knowledge graph features don't work.

**Diagnostic**:
```bash
# Check Neo4j is running
curl http://localhost:7474

# Test connection
python -c "from kosmos.world_model import get_world_model; wm = get_world_model()"
```

**Solutions**:
| Error | Solution |
|-------|----------|
| `ServiceUnavailable` | Start Neo4j: `docker compose up -d neo4j` |
| `AuthenticationError` | Check `NEO4J_PASSWORD` matches container |
| `Timeout` | Increase `NEO4J_MAX_CONNECTION_LIFETIME` |

### 7.6 Docker Sandbox Issues

**Symptom**: Code execution fails or times out.

**Diagnostic**:
```bash
# Check Docker is running
docker info

# Test sandbox
python -c "from kosmos.execution.sandbox import Sandbox; s = Sandbox(); print(s)"
```

**Solutions**:
| Error | Solution |
|-------|----------|
| `Cannot connect to Docker daemon` | Start Docker service |
| `Permission denied` | Add user to docker group |
| `Timeout` | Increase `MAX_EXPERIMENT_EXECUTION_TIME` |
| `OOMKilled` | Increase `MAX_MEMORY_MB` or container limits |

### 7.7 Budget Exceeded

**Symptom**: Research stops with `BudgetExceededError`.

**Diagnostic**:
```python
from kosmos.core.metrics import get_metrics
metrics = get_metrics()
print(f"Current cost: ${metrics.total_cost:.2f}")
print(f"Budget: ${metrics.budget_limit:.2f}")
```

**Solutions**:
- Increase `RESEARCH_BUDGET_USD`
- Use cheaper models (e.g., Haiku instead of Sonnet)
- Reduce `MAX_RESEARCH_ITERATIONS`
- Enable caching: `ENABLE_RESULT_CACHING=true`

### 7.8 Slow Performance

**Symptom**: Each cycle takes much longer than expected.

**Diagnostic**:
```bash
# Enable profiling
ENABLE_PROFILING=true
PROFILING_MODE=standard
kosmos run --trace --objective "Test"
```

**Solutions**:
| Issue | Solution |
|-------|----------|
| LLM latency | Enable concurrent operations |
| No caching | Enable `ENABLE_RESULT_CACHING=true` |
| Sequential experiments | Set `PARALLEL_EXPERIMENTS=3` |
| Large context | Check compression is working |

---

## 8. Paper vs. Implementation

### 8.1 Paper Claims vs. Implementation Status

| Paper Claim | Implementation Status | Gap Description |
|-------------|----------------------|-----------------|
| **20 research cycles** | ✅ Implemented | `max_iterations` config, default 10 |
| **1,500 papers per run** | ⚠️ Partial | Literature search works, scale untested |
| **42,000 lines of code** | ⚠️ Partial | Code generation works, not full sandbox |
| **79.4% statement accuracy** | ❌ Not validated | Architecture implemented, needs testing |
| **7 validated discoveries** | ❌ Not reproduced | System can discover, none validated |
| **200 agent rollouts** | ⚠️ Partial | Parallelism configurable but limited |
| **12-hour runtime constraint** | ⚠️ Partial | No explicit time limit enforcement |
| **85.5% data analysis accuracy** | ❌ Not validated | Agent works, accuracy unmeasured |
| **82.1% literature accuracy** | ❌ Not validated | Agent works, accuracy unmeasured |
| **57.9% interpretation accuracy** | ❌ Not validated | Known weak point from paper |

### 8.2 Implementation Gap Status

| Gap | Problem | Solution Status |
|-----|---------|-----------------|
| **Gap 0** | Context compression (1500 papers + 42K lines won't fit) | ✅ Implemented: 20:1 hierarchical compression |
| **Gap 1** | State Manager schema undefined | ✅ Implemented: 4-layer hybrid architecture |
| **Gap 2** | Task generation strategy undefined | ✅ Implemented: Plan Creator + Reviewer |
| **Gap 3** | Agent integration undefined | ⚠️ Partial: SkillLoader broken, 116 skills available |
| **Gap 4** | R vs Python ambiguity | ✅ Resolved: Python-only with Docker sandbox |
| **Gap 5** | Discovery validation undefined | ✅ Implemented: ScholarEval 8-dimension framework |

### 8.3 Features Described in Paper Not Implemented

1. **166 data analysis rollouts**: Parallelism exists but not at this scale
2. **R package support**: Python-only; use rpy2 for R packages
3. **Multi-language kernel**: Single Python kernel only
4. **Automatic package installation**: Not implemented
5. **Discovery validation workflow**: Architecture exists, no validation study

### 8.4 Undocumented Behaviors

1. **Error recovery**: Exponential backoff with circuit breaker (max 3 consecutive errors)
2. **Infinite loop prevention**: Max 50 actions per iteration before forced convergence
3. **Thread safety**: Locks on research_plan, strategy_stats, workflow, agent_registry
4. **Async LLM support**: Optional AsyncClaudeClient for concurrent operations
5. **Budget enforcement**: Graceful convergence when budget exceeded

### 8.5 Implicit Assumptions

1. **LLM availability**: System assumes API is available; no offline mode
2. **Network connectivity**: Literature search requires internet
3. **Sufficient API credits**: No local model fallback
4. **Docker for safety**: Without Docker, code execution is unsafe
5. **Neo4j optional**: Knowledge graph features gracefully degrade
6. **Single user**: No multi-tenancy or isolation
7. **Research quality**: Results not validated against ground truth

### 8.6 Recommended Validation Studies

Before production use, conduct these validations:

1. **Accuracy study**: Compare generated statements against expert annotations
2. **Reproducibility study**: Run same question multiple times, measure consistency
3. **Discovery validation**: Have domain experts validate any "discoveries"
4. **Scale test**: Test with 1,500+ papers, measure performance
5. **Long-run test**: Complete 20-cycle run, verify no degradation

---

## Appendix A: Quick Reference

### A.1 Key Files

| Component | File Path |
|-----------|-----------|
| Research Director | `kosmos/agents/research_director.py` |
| Hypothesis Generator | `kosmos/agents/hypothesis_generator.py` |
| Experiment Designer | `kosmos/agents/experiment_designer.py` |
| Data Analyst | `kosmos/agents/data_analyst.py` |
| Research Workflow | `kosmos/workflow/research_loop.py` |
| Plan Creator | `kosmos/orchestration/plan_creator.py` |
| Plan Reviewer | `kosmos/orchestration/plan_reviewer.py` |
| State Manager | `kosmos/world_model/artifacts.py` |
| ScholarEval | `kosmos/validation/scholar_eval.py` |
| Context Compressor | `kosmos/compression/compressor.py` |
| Skill Loader | `kosmos/agents/skill_loader.py` |
| Workflow State | `kosmos/core/workflow.py` |
| LLM Client | `kosmos/core/llm.py` |

### A.2 Common Commands

```bash
# Installation
pip install -e .
cp .env.example .env

# Testing
python scripts/smoke_test.py
pytest tests/unit/ -v
kosmos doctor

# Running
kosmos run --objective "Your question" --domain biology
kosmos run --trace --objective "Your question"  # Verbose
kosmos info                                       # System info

# Docker services
docker compose --profile dev up -d    # Start all
docker compose --profile dev down     # Stop all
```

### A.3 Workflow State Transitions

```
INITIALIZING
    ↓
GENERATING_HYPOTHESES ←──────────────┐
    ↓                                │
DESIGNING_EXPERIMENTS ←──────────────┤
    ↓                                │
EXECUTING                            │
    ↓                                │
ANALYZING                            │
    ↓                                │
REFINING ────────────────────────────┘
    ↓
CONVERGED
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Cycle** | One complete iteration: plan → execute → analyze → refine |
| **Rollout** | Single agent execution (166 data analysis rollouts = 166 executions) |
| **Finding** | Validated research result stored in State Manager |
| **ScholarEval** | 8-dimension quality validation framework |
| **Exploration** | Tasks exploring new research directions |
| **Exploitation** | Tasks deepening existing findings |
| **Convergence** | Research completion based on criteria |
| **Artifact** | JSON file storing cycle/task results |

---

**End of Runbook**

*Generated for Kosmos v0.2.0-alpha | For questions, see [docs/E2E_DIAGNOSTIC.md](docs/E2E_DIAGNOSTIC.md)*

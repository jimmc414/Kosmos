# Kosmos Project Runbook

**Version**: 0.2.0-alpha
**Last Updated**: 2025-12-09
**Purpose**: Complete guide for running the Kosmos autonomous AI scientist system

---

## Table of Contents

1. [Quick Start Guide](#1-quick-start-guide)
2. [System Requirements](#2-system-requirements)
3. [Installation](#3-installation)
4. [Configuration](#4-configuration)
5. [Running Kosmos](#5-running-kosmos)
6. [System Architecture](#6-system-architecture)
7. [Research Cycle Lifecycle](#7-research-cycle-lifecycle)
8. [Component Reference](#8-component-reference)
9. [Validation Checkpoints](#9-validation-checkpoints)
10. [Troubleshooting](#10-troubleshooting)
11. [Paper vs. Implementation](#11-paper-vs-implementation)
12. [Advanced Configuration](#12-advanced-configuration)

---

## 1. Quick Start Guide

### 1.1 Minimum Setup (5 minutes)

```bash
# 1. Clone and install
git clone https://github.com/jimmc414/Kosmos.git
cd Kosmos
pip install -e .

# 2. Configure API key
cp .env.example .env
# Edit .env and set: ANTHROPIC_API_KEY=sk-ant-your-key-here

# 3. Verify installation
python scripts/smoke_test.py
```

### 1.2 Run Your First Research

**Option A: Python API (Recommended)**
```python
import asyncio
from kosmos.workflow.research_loop import ResearchWorkflow

async def run():
    workflow = ResearchWorkflow(
        research_objective="How do KRAS mutations affect cancer metabolism?",
        artifacts_dir="./artifacts",
        max_cycles=5
    )
    result = await workflow.run(num_cycles=5, tasks_per_cycle=10)
    report = await workflow.generate_report()
    print(report)

asyncio.run(run())
```

**Option B: CLI**
```bash
kosmos run "How do KRAS mutations affect cancer metabolism?" --domain biology -i 5
```

### 1.3 Expected Output

After a successful run, you'll see:
- `artifacts/` directory with cycle findings
- Console output showing cycle progress
- Final research report with validated discoveries

---

## 2. System Requirements

### 2.1 Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 4GB | 8GB+ |
| CPU | 2 cores | 4+ cores |
| Disk | 10GB | 50GB+ |
| Network | Required | Required |

### 2.2 Software Requirements

| Software | Version | Required |
|----------|---------|----------|
| Python | 3.11+ | Yes |
| pip | Latest | Yes |
| Docker | 20.10+ | Recommended (for sandboxed execution) |
| Git | 2.0+ | Yes |

### 2.3 API Requirements

At least ONE of these LLM providers:

| Provider | API Key Variable | Notes |
|----------|-----------------|-------|
| Anthropic | `ANTHROPIC_API_KEY` | Default, recommended |
| OpenAI | `OPENAI_API_KEY` | Alternative |
| LiteLLM | `LITELLM_API_KEY` | 100+ providers including Ollama |

### 2.4 Optional Services

| Service | Purpose | Default |
|---------|---------|---------|
| Neo4j | Knowledge graph | Disabled |
| Redis | Distributed caching | Disabled |
| PostgreSQL | Production database | SQLite |
| Docker | Code sandbox | exec() fallback |

---

## 3. Installation

### 3.1 Standard Installation

```bash
# Clone repository
git clone https://github.com/jimmc414/Kosmos.git
cd Kosmos

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OR: .venv\Scripts\activate  # Windows

# Install core package
pip install -e .
```

### 3.2 Installation with Optional Dependencies

```bash
# With development tools
pip install -e ".[dev]"

# With scientific computing extras (scanpy, anndata)
pip install -e ".[science]"

# With Docker execution support
pip install -e ".[execution]"

# Full installation
pip install -e ".[all]"
```

### 3.3 Verify Installation

```bash
# Run smoke tests (no API key needed)
python scripts/smoke_test.py

# Run unit tests
pytest tests/unit/ -v --tb=short

# Check system status
kosmos doctor

# View system info
kosmos info
```

**Expected smoke test output:**
```
Testing component imports...
  ContextCompressor: OK
  ArtifactStateManager: OK
  PlanCreatorAgent: OK
  PlanReviewerAgent: OK
  NoveltyDetector: OK
  ScholarEvalValidator: OK
  SkillLoader: OK
  ResearchWorkflow: OK
All smoke tests passed!
```

---

## 4. Configuration

### 4.1 Environment Setup

```bash
# Copy template
cp .env.example .env

# Edit with your preferred editor
nano .env  # or vim, code, etc.
```

### 4.2 Essential Configuration

```bash
# REQUIRED: LLM Provider (choose one)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# OPTIONAL: Research limits
MAX_RESEARCH_ITERATIONS=10
RESEARCH_BUDGET_USD=10.0

# OPTIONAL: Domains
ENABLED_DOMAINS=biology,physics,chemistry,neuroscience
```

### 4.3 LLM Provider Options

**Anthropic (Default, Recommended):**
```bash
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-sonnet-4-5
```

**OpenAI:**
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4-turbo
```

**LiteLLM (Local/Multi-Provider):**
```bash
LLM_PROVIDER=litellm
LITELLM_MODEL=ollama/llama3.1:8b
LITELLM_API_BASE=http://localhost:11434
```

### 4.4 Budget Configuration

```bash
# Enable budget tracking (recommended)
BUDGET_ENABLED=true
BUDGET_LIMIT_USD=10.00

# Disable budget limit (not recommended)
# BUDGET_ENABLED=false
```

When budget is exceeded, research gracefully converges with current findings.

### 4.5 Database Configuration

```bash
# SQLite (default, easy setup)
DATABASE_URL=sqlite:///kosmos.db

# PostgreSQL (production)
DATABASE_URL=postgresql://kosmos:password@localhost:5432/kosmos
```

---

## 5. Running Kosmos

### 5.1 Using the Python API (Recommended)

```python
import asyncio
from kosmos.workflow.research_loop import ResearchWorkflow

async def run_research():
    # Initialize workflow
    workflow = ResearchWorkflow(
        research_objective="Your research question here",
        artifacts_dir="./artifacts",
        max_cycles=20  # Paper default: 20 cycles
    )

    # Run research cycles
    result = await workflow.run(
        num_cycles=5,          # Number of cycles to run
        tasks_per_cycle=10     # Tasks per cycle (paper default)
    )

    # Generate report
    report = await workflow.generate_report()

    # Save report
    with open("RESEARCH_REPORT.md", "w") as f:
        f.write(report)

    # Print statistics
    print(f"Cycles completed: {result['cycles_completed']}")
    print(f"Findings: {result['total_findings']}")
    print(f"Validated: {result['validated_findings']}")
    print(f"Validation rate: {result['validation_rate']*100:.1f}%")

    return result

# Run
asyncio.run(run_research())
```

### 5.2 Using the CLI

```bash
# Basic research run
kosmos run "Your research question" --domain biology

# With iteration limit
kosmos run "Your question" -d biology -i 5

# With budget limit
kosmos run "Your question" --domain materials --budget 50

# Maximum verbosity (debug mode)
kosmos run "Your question" --domain biology --trace

# Interactive mode (guided setup)
kosmos run --interactive

# Save results to file
kosmos run "Your question" -o results.json
```

### 5.3 Using Individual Agents Directly

If the CLI hangs (known issue), use agents directly:

```python
from kosmos.agents.hypothesis_generator import HypothesisGeneratorAgent

# Initialize agent
agent = HypothesisGeneratorAgent(config={})

# Generate hypotheses
result = agent.generate_hypotheses(
    research_question="Does X affect Y?",
    domain="biology",
    num_hypotheses=5
)

print(f"Generated {len(result.hypotheses)} hypotheses")
for h in result.hypotheses:
    print(f"  - {h.statement}")
```

### 5.4 Running with Optional Services

```bash
# Start all optional services (Neo4j, Redis, PostgreSQL)
docker compose --profile dev up -d

# Verify services are running
docker compose ps

# Run Kosmos with full stack
kosmos run "Your question" --domain biology

# Stop services when done
docker compose --profile dev down
```

---

## 6. System Architecture

### 6.1 Component Overview

```
+-----------------------------------------------------------------------+
|                      DISCOVERY ORCHESTRATOR                            |
|                   (ResearchDirectorAgent)                              |
|                                                                        |
|  +-------------+  +-------------+  +-------------+  +-------------+    |
|  | Hypothesis  |  | Experiment  |  |    Data     |  | Literature  |    |
|  | Generator   |  | Designer    |  |   Analyst   |  |  Analyzer   |    |
|  +------+------+  +------+------+  +------+------+  +------+------+    |
|         |                |                |                |           |
|         +----------------+----------------+----------------+           |
|                                |                                       |
|                                v                                       |
|                 +-----------------------------+                        |
|                 |   STATE MANAGER (World Model)|                       |
|                 | +--------+  +-------------+ |                        |
|                 | |  JSON  |  |  Knowledge  | |                        |
|                 | |Artifacts|  | Graph(Neo4j)| |                        |
|                 | +--------+  +-------------+ |                        |
|                 +-----------------------------+                        |
|                                |                                       |
|                                v                                       |
|                 +-----------------------------+                        |
|                 |       TASK GENERATOR        |                        |
|                 | +--------+  +-----------+   |                        |
|                 | |  Plan  |  |   Plan    |   |                        |
|                 | | Creator|  |  Reviewer |   |                        |
|                 | +--------+  +-----------+   |                        |
|                 +-----------------------------+                        |
|                                |                                       |
|                                v                                       |
|                 +-----------------------------+                        |
|                 |    VALIDATION FRAMEWORK     |                        |
|                 |       (ScholarEval)         |                        |
|                 +-----------------------------+                        |
+-----------------------------------------------------------------------+
```

### 6.2 Data Flow

1. **User Input** -> Research question enters ResearchWorkflow
2. **Plan Creator** -> Generates 10 tasks per cycle
3. **Novelty Detector** -> Filters redundant tasks
4. **Plan Reviewer** -> Validates plan quality (5 dimensions)
5. **Delegation Manager** -> Dispatches tasks to agents
6. **Agents Execute** -> Data analysis, literature search, etc.
7. **ScholarEval** -> Validates findings (8 dimensions)
8. **State Manager** -> Persists validated findings
9. **Context Compressor** -> Compresses results (20:1 ratio)
10. **Report Synthesizer** -> Generates final report

### 6.3 State Management Layers

| Layer | Technology | Purpose |
|-------|------------|---------|
| 1. JSON Artifacts | Files | Human-readable, traceable |
| 2. Knowledge Graph | Neo4j (optional) | Relationship queries |
| 3. Vector Store | ChromaDB | Semantic similarity |
| 4. Citations | Integrated | Evidence chains |

---

## 7. Research Cycle Lifecycle

### 7.1 Workflow State Machine

```
INITIALIZING -> GENERATING_HYPOTHESES -> DESIGNING_EXPERIMENTS -> EXECUTING
                                                                       |
                                                                       v
CONVERGED <- REFINING <- ANALYZING <-----------------------------------+
```

### 7.2 Single Cycle Steps

```python
# Pseudocode for one research cycle
async def execute_cycle(cycle: int, num_tasks: int = 10):
    # Step 1: Get context from State Manager (lookback=3 cycles)
    context = state_manager.get_cycle_context(cycle, lookback=3)

    # Step 2: Plan Creator generates 10 tasks
    plan = plan_creator.create_plan(research_objective, context, num_tasks)

    # Step 3: Novelty Detector checks for redundancy
    novelty = novelty_detector.check_plan_novelty(plan)

    # Step 4: Plan Reviewer validates quality (5 dimensions)
    review = plan_reviewer.review_plan(plan, context)

    # Step 5: If rejected, revise once
    if not review.approved:
        plan = plan_creator.revise_plan(plan, review, context)
        review = plan_reviewer.review_plan(plan, context)

    # Step 6: Execute approved tasks
    results = await delegation_manager.execute_plan(plan, cycle)

    # Step 7: ScholarEval validates each finding
    for result in results.completed_tasks:
        eval_score = scholar_eval.evaluate_finding(result.finding)
        if eval_score.passes_threshold:
            await state_manager.save_finding_artifact(cycle, result)

    # Step 8: Compress cycle results
    compressed = context_compressor.compress_cycle_results(cycle, results)

    # Step 9: Generate cycle summary
    await state_manager.generate_cycle_summary(cycle)
```

### 7.3 Exploration vs. Exploitation

| Cycles | Exploration | Exploitation | Strategy |
|--------|-------------|--------------|----------|
| 1-7 | 70% | 30% | Explore new directions |
| 8-14 | 50% | 50% | Balanced |
| 15-20 | 30% | 70% | Deepen findings |

### 7.4 Convergence Criteria

Research terminates when ANY of these conditions are met:

1. **Iteration limit**: `iteration_count >= max_iterations` (default: 10)
2. **No hypotheses**: Empty hypothesis pool after generation
3. **Budget exceeded**: API costs exceed `RESEARCH_BUDGET_USD`
4. **No novel tasks**: Novelty score below threshold for 3 consecutive cycles
5. **Runtime exceeded**: Elapsed time exceeds `max_runtime_hours` (default: 12h)
6. **User stop**: Manual convergence trigger

---

## 8. Component Reference

### 8.1 ResearchWorkflow

**File**: `kosmos/workflow/research_loop.py`

```python
from kosmos.workflow.research_loop import ResearchWorkflow

workflow = ResearchWorkflow(
    research_objective="Your question",
    artifacts_dir="./artifacts",
    max_cycles=20,
    seed=42  # Optional: for reproducibility
)

result = await workflow.run(num_cycles=5, tasks_per_cycle=10)
report = await workflow.generate_report()
stats = workflow.get_statistics()
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `research_objective` | str | Required | Main research goal |
| `artifacts_dir` | str | "artifacts" | Output directory |
| `max_cycles` | int | 20 | Maximum cycles |
| `seed` | int | None | Random seed for reproducibility |

### 8.2 HypothesisGeneratorAgent

**File**: `kosmos/agents/hypothesis_generator.py`

```python
from kosmos.agents.hypothesis_generator import HypothesisGeneratorAgent

agent = HypothesisGeneratorAgent(config={})
result = agent.generate_hypotheses(
    research_question="Does X affect Y?",
    domain="biology",
    num_hypotheses=5,
    literature_context=None  # Optional prior papers
)
```

**Expected Timing**: ~19 seconds per call

**Output**:
```python
HypothesisGenerationResult(
    hypotheses=[
        Hypothesis(
            statement="...",
            rationale="...",
            testability_score=0.85,
            novelty_score=0.72
        )
    ],
    count=5
)
```

### 8.3 ExperimentDesignerAgent

**File**: `kosmos/agents/experiment_designer.py`

```python
from kosmos.agents.experiment_designer import ExperimentDesignerAgent

agent = ExperimentDesignerAgent(config={})
protocol = agent.design_experiment(
    hypothesis=hypothesis_object,
    domain="biology",
    available_datasets=["GSE12345"],
    constraints={"max_runtime": 300}
)
```

**Expected Timing**: ~89 seconds per call

### 8.4 PlanCreatorAgent

**File**: `kosmos/orchestration/plan_creator.py`

```python
from kosmos.orchestration import PlanCreatorAgent

agent = PlanCreatorAgent(anthropic_client)
plan = agent.create_plan(
    research_objective="Your objective",
    context={"cycle": 5, "recent_findings": [...]},
    num_tasks=10
)
```

**Output**: `ResearchPlan` with 10 tasks and rationale

### 8.5 ScholarEvalValidator

**File**: `kosmos/validation/scholar_eval.py`

```python
from kosmos.validation import ScholarEvalValidator

validator = ScholarEvalValidator(anthropic_client)
score = validator.evaluate_finding(finding)

if score.passes_threshold:
    print(f"Validated: {score.overall_score:.2f}")
```

**8-Dimension Scoring**:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Rigor | 25% | Scientific soundness |
| Impact | 20% | Importance of finding |
| Novelty | 15% | Is this new? |
| Reproducibility | 15% | Can others reproduce? |
| Clarity | 10% | Clearly stated? |
| Coherence | 10% | Fits existing knowledge? |
| Limitations | 3% | Limitations acknowledged? |
| Ethics | 2% | Ethical considerations? |

**Threshold**: Overall score >= 0.75 (75%)

### 8.6 ContextCompressor

**File**: `kosmos/compression/compressor.py`

```python
from kosmos.compression import ContextCompressor

compressor = ContextCompressor(anthropic_client)
compressed = compressor.compress_notebook("path/to/notebook.ipynb")
```

**Compression Ratio**: ~20:1

**Tiers**:
1. Task: 42K lines -> 2-line summary
2. Cycle: 10 task summaries -> 1 overview
3. Final: 20 cycles -> Research narrative

---

## 9. Validation Checkpoints

### 9.1 Checkpoint Table

| ID | Location | Success Criteria | Failure Action |
|----|----------|------------------|----------------|
| CP1 | After `ResearchWorkflow.__init__` | All components initialized | Check config |
| CP2 | After `_load_skills()` | `skills is not None` or warning | Check skill files |
| CP3 | After `create_plan()` | `len(plan.tasks) == 10` | Check LLM connection |
| CP4 | After hypothesis generation | `len(hypotheses) > 0` | Check domain config |
| CP5 | After experiment design | `protocol.methodology != ""` | Check hypothesis |
| CP6 | After execution | `result.status == "completed"` | Check sandbox |
| CP7 | After analysis | `confidence >= 0.5` | Check result data |
| CP8 | After ScholarEval | `passes_threshold == True` | Review finding |
| CP9 | After cycle complete | Findings saved to artifacts | Check storage |
| CP10 | At research end | `has_converged == True` | Check criteria |

### 9.2 Quick Health Checks

```bash
# System diagnostics
kosmos doctor

# Smoke test all components
python scripts/smoke_test.py

# Unit tests
pytest tests/unit/ -v --tb=short

# Integration tests (requires services)
pytest tests/integration/ -v --tb=short
```

### 9.3 Manual Component Verification

```python
# Test each component individually
from kosmos.compression import ContextCompressor
from kosmos.world_model.artifacts import ArtifactStateManager
from kosmos.orchestration import PlanCreatorAgent, PlanReviewerAgent, NoveltyDetector
from kosmos.validation import ScholarEvalValidator
from kosmos.agents import SkillLoader

# All should initialize without errors
compressor = ContextCompressor(None)
state_manager = ArtifactStateManager(artifacts_dir="./test_artifacts")
plan_creator = PlanCreatorAgent(None)
plan_reviewer = PlanReviewerAgent(None)
novelty_detector = NoveltyDetector()
validator = ScholarEvalValidator(None)
skill_loader = SkillLoader()

print("All components initialized successfully!")
```

---

## 10. Troubleshooting

### 10.1 CLI Hangs After Starting

**Symptom**: `kosmos run` shows banner then hangs indefinitely.

**Root Cause**: `ResearchDirectorAgent` uses message-passing architecture without a runtime to process messages.

**Solution**: Use `ResearchWorkflow` directly instead of CLI:

```python
import asyncio
from kosmos.workflow.research_loop import ResearchWorkflow

async def run():
    workflow = ResearchWorkflow(
        research_objective="Your question",
        artifacts_dir="./artifacts"
    )
    result = await workflow.run(num_cycles=5)
    report = await workflow.generate_report()
    print(report)

asyncio.run(run())
```

### 10.2 API Connection Errors

**Symptom**: `AuthenticationError`, `APIConnectionError`, or timeout.

**Diagnostic**:
```bash
# Check API key
echo $ANTHROPIC_API_KEY

# Test connection
python -c "from kosmos.core.llm import get_client; c = get_client(); print(c)"
```

**Solutions**:

| Error | Solution |
|-------|----------|
| `AuthenticationError` | Check API key format, verify credits |
| `RateLimitError` | Reduce `MAX_CONCURRENT_LLM_CALLS` |
| `APIConnectionError` | Check internet, firewall |
| `Timeout` | Increase timeout, check API status |

### 10.3 SkillLoader Returns None

**Symptom**: Skills not loaded, generic prompts used.

**Root Cause**: `COMMON_SKILLS` references non-existent files.

**Solution**: Use domain-specific bundles:

```python
from kosmos.agents.skill_loader import SkillLoader

loader = SkillLoader()
# Instead of:
# skills = loader.load_skills_for_task(task_type='research', domain='biology')

# Use bundle names directly:
skills = loader.load_skills_for_task(task_type='single_cell_analysis')
```

See `docs/ISSUE_SKILLLOADER_BROKEN.md` for details.

### 10.4 Database Errors

**Symptom**: `OperationalError`, `IntegrityError`.

**Solutions**:

| Error | Solution |
|-------|----------|
| `database is locked` | Use PostgreSQL instead of SQLite |
| `no such table` | Run `alembic upgrade head` |
| `connection refused` | Check database is running |

### 10.5 Neo4j Not Connecting

**Symptom**: Knowledge graph features disabled.

**Solution**:
```bash
# Start Neo4j with Docker
docker compose up -d neo4j

# Verify connection
curl http://localhost:7474

# Check environment variables
echo $NEO4J_URI
echo $NEO4J_PASSWORD
```

### 10.6 Budget Exceeded

**Symptom**: `BudgetExceededError`, research stops early.

**Solutions**:
- Increase `RESEARCH_BUDGET_USD` in `.env`
- Use cheaper models (Haiku vs Sonnet)
- Reduce `MAX_RESEARCH_ITERATIONS`
- Enable caching: `ENABLE_RESULT_CACHING=true`

### 10.7 Slow Performance

**Symptom**: Each cycle takes much longer than expected.

**Solutions**:

| Issue | Solution |
|-------|----------|
| LLM latency | Enable `ENABLE_CONCURRENT_OPERATIONS=true` |
| No caching | Enable `ENABLE_RESULT_CACHING=true` |
| Sequential execution | Set `PARALLEL_EXPERIMENTS=3` |
| Large context | Verify compression is working |

---

## 11. Paper vs. Implementation

### 11.1 Paper Claims vs. Current Status

| Paper Claim | Implementation Status | Notes |
|-------------|----------------------|-------|
| **20 research cycles** | Implemented | `max_iterations` configurable |
| **1,500 papers per run** | Partial | Literature search works, scale untested |
| **42,000 lines of code** | Partial | Code generation works, sandbox ready |
| **79.4% accuracy** | Not validated | Architecture implemented, needs testing |
| **7 validated discoveries** | Not reproduced | System can discover, none validated |
| **200 agent rollouts** | Partial | Parallelism configurable |
| **12-hour runtime** | Implemented | `max_runtime_hours` config |
| **85.5% data analysis accuracy** | Not validated | Agent works, accuracy unmeasured |
| **82.1% literature accuracy** | Not validated | Agent works, accuracy unmeasured |

### 11.2 Implementation Gap Solutions

| Gap | Problem | Solution Status |
|-----|---------|-----------------|
| **Gap 0** | Context compression for 1,500 papers | Implemented: 20:1 hierarchical compression |
| **Gap 1** | State Manager schema undefined | Implemented: 4-layer hybrid architecture |
| **Gap 2** | Task generation strategy undefined | Implemented: Plan Creator + Plan Reviewer |
| **Gap 3** | Agent integration undefined | Partial: SkillLoader has issues, 116 skills exist |
| **Gap 4** | R vs Python ambiguity | Resolved: Python-only with Docker sandbox |
| **Gap 5** | Discovery validation undefined | Implemented: ScholarEval 8-dimension framework |

### 11.3 Features Not Yet Implemented

1. **166 data analysis rollouts at scale**: Parallelism exists but not at paper scale
2. **R package support**: Python-only; use rpy2 for R packages
3. **Multi-language kernel**: Single Python kernel only
4. **Automatic package installation**: Not implemented
5. **Discovery reproduction**: Paper discoveries not reproduced

### 11.4 Undocumented Behaviors

1. **Error recovery**: Exponential backoff (2s, 4s, 8s) with circuit breaker (max 3 errors)
2. **Infinite loop prevention**: Max 50 actions per iteration
3. **Thread safety**: Locks on research_plan, strategy_stats, workflow
4. **Async LLM support**: Optional AsyncClaudeClient for concurrent ops
5. **Budget enforcement**: Graceful convergence when budget exceeded

---

## 12. Advanced Configuration

### 12.1 Concurrent Operations

```bash
# Enable parallel operations (2-4x speedup)
ENABLE_CONCURRENT_OPERATIONS=true
MAX_CONCURRENT_EXPERIMENTS=4
MAX_PARALLEL_HYPOTHESES=3
MAX_CONCURRENT_LLM_CALLS=5
```

### 12.2 Rate Limiting

```bash
# Adjust based on API tier
LLM_RATE_LIMIT_PER_MINUTE=50   # Tier 1
# LLM_RATE_LIMIT_PER_MINUTE=100 # Tier 2
# LLM_RATE_LIMIT_PER_MINUTE=200 # Tier 3
```

### 12.3 Debug Mode

```bash
# Enable debug output
DEBUG_MODE=true
DEBUG_LEVEL=2  # 0=off, 1=critical, 2=full, 3=data dumps

# Or use CLI flag
kosmos run "Your question" --trace
```

### 12.4 Profiling

```bash
# Enable performance profiling
ENABLE_PROFILING=true
PROFILING_MODE=standard  # light, standard, full
```

### 12.5 Reproducibility

```python
# Set seed for reproducible runs
workflow = ResearchWorkflow(
    research_objective="Your question",
    seed=42,  # Reproducibility seed
    temperature=0.7  # LLM temperature
)
```

---

## Appendix A: Key Files Reference

| Component | File Path |
|-----------|-----------|
| Research Workflow | `kosmos/workflow/research_loop.py` |
| Research Director | `kosmos/agents/research_director.py` |
| Hypothesis Generator | `kosmos/agents/hypothesis_generator.py` |
| Experiment Designer | `kosmos/agents/experiment_designer.py` |
| Data Analyst | `kosmos/agents/data_analyst.py` |
| Literature Analyzer | `kosmos/agents/literature_analyzer.py` |
| Plan Creator | `kosmos/orchestration/plan_creator.py` |
| Plan Reviewer | `kosmos/orchestration/plan_reviewer.py` |
| State Manager | `kosmos/world_model/artifacts.py` |
| ScholarEval | `kosmos/validation/scholar_eval.py` |
| Context Compressor | `kosmos/compression/compressor.py` |
| Skill Loader | `kosmos/agents/skill_loader.py` |
| Workflow State | `kosmos/core/workflow.py` |
| LLM Client | `kosmos/core/llm.py` |
| CLI Entry | `kosmos/cli/main.py` |
| Configuration | `kosmos/config.py` |

---

## Appendix B: Common Commands

```bash
# Installation
pip install -e .
cp .env.example .env

# Testing
python scripts/smoke_test.py
pytest tests/unit/ -v
kosmos doctor

# Running
kosmos run "Your question" --domain biology
kosmos run "Your question" -i 5              # Limit iterations
kosmos run --interactive                      # Guided mode
kosmos info                                   # System info

# Docker services
docker compose --profile dev up -d    # Start all
docker compose --profile dev down     # Stop all
docker compose logs neo4j             # View logs

# Debugging
kosmos run "Your question" --trace    # Max verbosity
```

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **Cycle** | One complete research iteration: plan -> execute -> analyze -> refine |
| **Rollout** | Single agent execution instance |
| **Finding** | Validated research result stored in State Manager |
| **ScholarEval** | 8-dimension quality validation framework |
| **Exploration** | Tasks exploring new research directions |
| **Exploitation** | Tasks deepening existing findings |
| **Convergence** | Research completion based on termination criteria |
| **Artifact** | JSON file storing cycle/task results |
| **State Manager** | Central knowledge repository (Gap 1 solution) |
| **World Model** | Alternative name for State Manager |

---

**End of Runbook**

*For questions or issues, see [GitHub Issues](https://github.com/jimmc414/Kosmos/issues)*

*Based on Kosmos v0.2.0-alpha | Last Updated: 2025-12-09*

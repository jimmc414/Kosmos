# Kosmos

Open-source implementation of the autonomous AI scientist described in [Lu et al. (2024)](https://arxiv.org/abs/2511.02824). The original paper reported 79.4% accuracy on scientific statements and 7 validated discoveries, but omitted implementation details for 6 critical components. This repository provides those implementations using patterns from the K-Dense ecosystem.

[![Version](https://img.shields.io/badge/version-0.2.0--alpha-blue.svg)](https://github.com/jimmc414/Kosmos)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/jimmc414/Kosmos)
[![Implementation](https://img.shields.io/badge/core-75%25%20production--ready-green.svg)](120525_implementation_gaps_v2.md)
[![Implementation](https://img.shields.io/badge/deferred-20%25%20Phase%202%2F4-yellow.svg)](120525_implementation_gaps_v2.md)
[![Tests](https://img.shields.io/badge/unit-339%20passing-green.svg)](TESTS_STATUS.md)

**Current state**: Core research loop operational. All 6 original paper gaps implemented. Neo4j knowledge graph fully implemented (1,025 lines) but requires integration testing. Debug mode with `--trace` flag provides full observability. See [Project Status](#project-status) for honest assessment and [Implementation Gaps Analysis](120525_implementation_gaps_v2.md) for detailed breakdown.

## Paper Gap Analysis

The original Kosmos paper demonstrated results but left critical implementation details unspecified. Analysis in [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) identified 6 gaps blocking reproduction:

| Gap | Problem | Severity |
|-----|---------|----------|
| 0 | Paper processes 1,500 papers and 42,000 lines of code per run, exceeding any LLM context window | Foundational |
| 1 | State Manager described as "core advancement" but no schema, storage strategy, or update mechanisms provided | Critical |
| 2 | Strategic reasoning algorithm for generating research tasks completely unstated | Critical |
| 3 | System prompts, output formats, and domain expertise injection mechanisms not specified | Critical |
| 4 | Paper contradicts itself on R vs Python usage; code execution environment not described | High |
| 5 | Paper reports 57.9% interpretation accuracy but quality metrics and filtering criteria not specified | Moderate |

## Gap Solutions

Each gap was addressed using patterns from the K-Dense ecosystem. Detailed analysis in [OPENQUESTIONS_SOLUTION.md](OPENQUESTIONS_SOLUTION.md).

### Gap 0: Context Compression (Complete)

**Problem**: 1,500 papers + 42,000 lines of code cannot fit in any LLM context window.

**Solution**: Hierarchical 3-tier compression achieving 20:1 ratio.
- Tier 1: Task-level compression (42K lines -> 2-line summary + statistics)
- Tier 2: Cycle-level compression (10 task summaries -> 1 cycle overview)
- Tier 3: Final synthesis with lazy loading for full content retrieval

**Pattern source**: kosmos-claude-skills-mcp (progressive disclosure)

**Implementation**: [`kosmos/compression/`](kosmos/compression/)

### Gap 1: State Manager (Complete)

**Problem**: Paper's "core advancement" has no schema specification.

**Solution**: Hybrid 4-layer architecture.
- Layer 1: JSON artifacts (human-readable, version-controllable)
- Layer 2: Knowledge graph (structural queries via Neo4j, optional)
- Layer 3: Vector store (semantic search, optional)
- Layer 4: Citation tracking (evidence chains)

**Implementation**: [`kosmos/world_model/artifacts.py`](kosmos/world_model/artifacts.py)

### Gap 2: Task Generation (Complete)

**Problem**: How does the system generate 10 strategic research tasks per cycle?

**Solution**: Plan Creator + Plan Reviewer orchestration pattern.
- Plan Creator: Generates tasks with exploration/exploitation ratio (70% early cycles, 30% late cycles)
- Plan Reviewer: 5-dimension scoring (specificity, relevance, novelty, coverage, feasibility)
- Novelty Detector: Prevents redundant analyses across 200 rollouts
- Delegation Manager: Routes tasks to appropriate agents

**Pattern source**: kosmos-karpathy (orchestration patterns)

**Implementation**: [`kosmos/orchestration/`](kosmos/orchestration/) (1,949 lines across 6 files)

### Gap 3: Agent Integration (Complete)

**Problem**: How are domain-specific capabilities injected into agents?

**Solution**: Skill loader with 566 domain-specific scientific prompts auto-loaded by domain matching.

**Pattern source**: kosmos-claude-scientific-skills (566 skills)

**Implementation**: [`kosmos/agents/skill_loader.py`](kosmos/agents/skill_loader.py)

**Skills submodule**: [`kosmos-claude-scientific-skills/`](kosmos-claude-scientific-skills/)

### Gap 4: Execution Environment (Complete)

**Problem**: Paper contradicts itself on R vs Python. No execution environment described.

**Solution**: Docker-based Jupyter sandbox with:
- Container pooling for performance (pre-warmed containers)
- Automatic package resolution and installation
- Resource limits (memory, CPU, timeout)
- Security constraints (network isolation, read-only rootfs, dropped capabilities)

This was the final gap implemented. The execution environment is now production-ready pending Docker availability.

**Implementation**: [`kosmos/execution/`](kosmos/execution/)

Key files:
- `docker_manager.py` - Container lifecycle management with pooling
- `jupyter_client.py` - Kernel gateway integration for code execution
- `package_resolver.py` - Automatic dependency detection and installation
- `production_executor.py` - Unified execution interface

### Gap 5: Discovery Validation (Complete)

**Problem**: How are discoveries evaluated before inclusion in reports?

**Solution**: ScholarEval 8-dimension quality framework with weighted scoring.

Dimensions evaluated:
1. Statistical validity
2. Reproducibility
3. Novelty
4. Significance
5. Methodological soundness
6. Evidence quality
7. Claim calibration
8. Citation support

**Pattern source**: kosmos-claude-scientific-writer (validation patterns)

**Implementation**: [`kosmos/validation/`](kosmos/validation/)

## K-Dense Pattern Sources

This implementation draws from the K-Dense ecosystem:

| Repository | Contribution | Gap |
|------------|--------------|-----|
| kosmos-claude-skills-mcp | Context compression, progressive disclosure | 0 |
| kosmos-karpathy | Orchestration, plan creator/reviewer pattern | 2 |
| kosmos-claude-scientific-skills | 566 domain-specific scientific prompts | 3 |
| kosmos-claude-scientific-writer | ScholarEval validation framework | 5 |

Reference repositories in [`kosmos-reference/`](kosmos-reference/). Skills integrated as git subtree at project root.

## Project Status

### Implementation Completeness (as of 2025-12-05)

| Category | Percentage | Description |
|----------|------------|-------------|
| Production-ready | 75% | Core research loop, agents, LLM providers |
| Deferred to future phases | 20% | Phase 2 annotations, Phase 4 production mode |
| Known issues | 5% | ArXiv Python 3.11+ incompatibility |

For detailed breakdown, see [Implementation Gaps Analysis](120525_implementation_gaps_v2.md).

### Test Results

| Category | Total | Pass | Fail | Skip | Notes |
|----------|-------|------|------|------|-------|
| Unit tests | 339 | 339 | 0 | 0 | Core gap implementations |
| Integration | 43 | 43 | 0 | 0 | Pipeline tests |
| LiteLLM provider | 22 | 22 | 0 | 0 | Multi-provider support |
| E2E tests | 39 | 32 | 0 | 7 | Skipped pending Neo4j/Docker |

E2E tests skipped based on environment:
- Neo4j not configured (`@pytest.mark.requires_neo4j`)
- Docker not running (sandbox execution tests)
- API keys not set (tests requiring live LLM calls)

### What Works

- Research workflow initialization and hypothesis generation
- Experiment design from hypotheses via LLM
- Result analysis and interpretation
- Multi-provider LLM support (Anthropic, OpenAI, LiteLLM/Ollama)
- Basic research cycle progression
- Docker-based sandboxed code execution (requires Docker to be running)
- Debug mode with configurable verbosity (levels 0-3)
- Real-time stage tracking with JSON output
- LLM call instrumentation across all providers
- Provider timeout configuration
- Cost calculation and usage tracking per provider
- Model comparison infrastructure (see [MODEL_COMPARISON_REPORT.md](MODEL_COMPARISON_REPORT.md))

### What Needs Work

1. **Neo4j Integration**: The knowledge graph is fully implemented (1,025 lines in `kosmos/knowledge/graph.py`) but:
   - E2E tests are skipped (`@pytest.mark.skip(reason="Neo4j authentication not configured")`)
   - Not integrated into the main research loop
   - Requires Neo4j instance + environment variables to activate

2. **Budget Enforcement**: Cost tracking works, but execution continues when budget exceeded. The system:
   - Tracks spending accurately via `MetricsCollector`
   - Emits alerts at 50%, 75%, 90%, 100% thresholds
   - Returns `budget_exceeded: True` but **does not halt**

3. **Error Recovery**: Error handlers log errors but lack recovery strategy:
   - No retry logic with backoff
   - No circuit breaker pattern
   - Workflow continues in degraded state after failures

4. **Phase 2 Features (Deferred)**: Annotation storage is stubbed:
   - `add_annotation()` only logs, doesn't persist
   - `get_annotations()` returns empty list
   - Designed for future curation workflows

5. **ArXiv Compatibility**: The `arxiv` package has Python 3.11+ issues:
   - Depends on `sgmllib3k` which may fail to build
   - Semantic Scholar works as fallback
   - Not blocking for core functionality

### Honest Assessment

This implementation provides the architectural skeleton described in the Lu et al. paper. The 6 gaps identified in the paper have been filled with working code. However:

- We have not reproduced the paper's claimed 79.4% accuracy or 7 validated discoveries
- The system has been tested primarily with small local models (Qwen 4B via Ollama), not production-scale LLMs
- Multi-cycle autonomous research runs have not been validated end-to-end
- The codebase has accumulated technical debt from rapid development

The project is suitable for experimentation and further development, not production research use.

### Next Steps

Prioritized implementation plan available in [Implementation Plan](120525_implementation_plan_v2.md).

**Phase A: Critical Safety** (estimated 6 hours)
1. Add budget enforcement - halt execution when limit exceeded
2. Implement error recovery with exponential backoff

**Phase B: Data Integrity** (estimated 5 hours)
3. Implement annotation storage for Phase 2 curation
4. Load actual hypothesis/result data for LLM prompts

**Phase C: Performance** (estimated 4 hours)
5. Implement true async in OpenAI/Anthropic providers

**Phase D: Future-Proofing** (estimated 4 hours)
6. Verify Neo4j integration with E2E tests

## Implementation Status

### Phase Architecture

The codebase follows a phased implementation approach:

| Phase | Scope | Status |
|-------|-------|--------|
| Phase 1 | Simple Mode - JSON artifacts, entity storage (up to 10K entities) | Complete |
| Phase 2 | Curation - Annotation storage, metadata management | Stubbed |
| Phase 4 | Production Mode - Polyglot persistence (PostgreSQL + Neo4j + Elasticsearch) | Not Implemented |

### December 2025 Updates

| Category | Status | Details |
|----------|--------|---------|
| Code Quality | Complete | All silent exception handlers now log appropriately |
| Debug Mode | Complete | All config flags implemented with config-gating |
| Gaps Analysis | Complete | See [120525_implementation_gaps_v2.md](120525_implementation_gaps_v2.md) |
| Implementation Plan | Complete | See [120525_implementation_plan_v2.md](120525_implementation_plan_v2.md) |

### NotImplementedError Summary

10 occurrences across 6 files (verified):
- 3 are Phase 2/4 deferred features (expected)
- 4 are abstract base class methods (expected)
- 2 are provider streaming (not all providers support)
- 1 is domain-specific (neuroscience stage limit)

### Debug Features
- `--trace` flag for maximum verbosity
- `log_llm_calls` - Token counts and latency for all providers
- `log_agent_messages` - Inter-agent message routing with correlation tracking
- `log_workflow_transitions` - State machine transitions with timing
- Stage tracking with JSONL output

## Limitations

1. **Docker required**: Gap 4 execution environment requires Docker. Without it, code execution falls back to direct `exec()` which is unsafe.

2. **Neo4j optional**: Knowledge graph features require Neo4j. The implementation is complete but tests skip without it. Set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` to enable.

3. **ArXiv compatibility**: The `arxiv` package may fail on Python 3.11+ due to `sgmllib3k`. Semantic Scholar is used as fallback.

4. **Python only**: The paper references R packages (MendelianRandomization, susieR). This implementation is Python-only.

5. **No budget enforcement**: Cost tracking works but execution continues when budget exceeded. See [Implementation Plan](120525_implementation_plan_v2.md) for fix.

6. **Async is sync**: LLM provider `generate_async()` methods currently delegate to sync. True async planned for Phase C.

7. **Single-user**: No multi-tenancy or user isolation.

8. **Not a reproduction study**: We have not reproduced the paper's 7 validated discoveries. This is an implementation of the architecture, not a validation of the results.

## Getting Started

### Requirements

- Python 3.11+
- Anthropic API key or OpenAI API key
- Docker (for sandboxed code execution)

### Installation

```bash
git clone https://github.com/jimmc414/Kosmos.git
cd Kosmos
pip install -e .
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY or OPENAI_API_KEY
```

### Verify Installation

```bash
# Run smoke tests
python scripts/smoke_test.py

# Run unit tests for gap modules
pytest tests/unit/compression/ tests/unit/orchestration/ \
       tests/unit/validation/ tests/unit/workflow/ \
       tests/unit/agents/test_skill_loader.py \
       tests/unit/world_model/test_artifacts.py -v
```

### Run Research Workflow

```python
import asyncio
from kosmos.workflow.research_loop import ResearchWorkflow

async def run():
    workflow = ResearchWorkflow(
        research_objective="Your research question here",
        artifacts_dir="./artifacts"
    )
    result = await workflow.run(num_cycles=5, tasks_per_cycle=10)
    report = await workflow.generate_report()
    print(report)

asyncio.run(run())
```

See [GETTING_STARTED.md](GETTING_STARTED.md) for detailed examples.

## Configuration

All configuration via environment variables. See `.env.example` for full list.

### LLM Provider

```bash
# Anthropic (default)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5-nano

# LiteLLM (supports 100+ providers including local models)
LLM_PROVIDER=litellm
LITELLM_MODEL=ollama/llama3.1:8b
LITELLM_API_BASE=http://localhost:11434
LITELLM_TIMEOUT=300

# DeepSeek via LiteLLM
LLM_PROVIDER=litellm
LITELLM_MODEL=deepseek/deepseek-chat
LITELLM_API_KEY=sk-...
```

### Debug Mode (Basic)

```bash
# Enable debug mode
DEBUG_MODE=true
DEBUG_LEVEL=1
```

See [Debug Mode Guide](#debug-mode-guide) below for comprehensive documentation.

### Optional Services

```bash
# Neo4j (optional, for knowledge graph features)
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=your-password

# Redis (optional, for distributed caching)
REDIS_URL=redis://localhost:6379
```

## Architecture

```
kosmos/
├── compression/      # Gap 0: Context compression (20:1 ratio)
├── world_model/      # Gap 1: State manager (JSON artifacts + optional graph)
├── orchestration/    # Gap 2: Task generation (plan creator/reviewer)
├── agents/           # Gap 3: Agent integration (skill loader)
├── execution/        # Gap 4: Sandboxed execution (Docker + Jupyter)
├── validation/       # Gap 5: Discovery validation (ScholarEval)
├── workflow/         # Integration layer combining all components
├── core/             # LLM clients, configuration, stage_tracker
│   ├── providers/    # Anthropic, OpenAI, LiteLLM providers with instrumentation
│   └── stage_tracker.py  # Real-time observability for multi-step processes
├── literature/       # Literature search (arXiv, PubMed, Semantic Scholar)
├── knowledge/        # Vector store, embeddings
├── monitoring/       # Metrics, alerts, cost tracking
└── cli/              # Command-line interface with debug options
```

### CLI Usage

```bash
# Run research with default settings
kosmos run --objective "Your research question"

# Enable trace logging (maximum verbosity)
kosmos run --trace --objective "Your research question"

# Set specific debug level (0-3)
kosmos run --debug-level 2 --objective "Your research question"

# Debug specific modules only
kosmos run --debug --debug-modules "research_director,workflow" --objective "Your research question"

# Show system information
kosmos info

# Run diagnostics
kosmos doctor

# Show version
kosmos version
```

## Debug Mode Guide

Kosmos includes comprehensive debug instrumentation for diagnosing issues, understanding execution flow, and troubleshooting research runs. This section covers all debug features and how to use them effectively.

### Quick Start

For most debugging scenarios, use the `--trace` flag:

```bash
kosmos run --trace --objective "Your research question" --max-iterations 2
```

This enables maximum verbosity including:
- All debug log messages
- LLM call logging (requests/responses)
- Agent message routing
- Workflow state transitions
- Real-time stage tracking

### Debug Levels

Debug verbosity is controlled by levels 0-3:

| Level | Name | What's Logged | Use Case |
|-------|------|---------------|----------|
| 0 | Off | Standard INFO/WARNING/ERROR only | Production runs |
| 1 | Critical Path | Decision points, action execution, phase transitions | Basic debugging |
| 2 | Full Trace | All of level 1 + LLM calls, message routing, timing | Deep debugging |
| 3 | Data Dumps | All of level 2 + full payloads, state snapshots | Issue reproduction |

### Environment Variables

Configure debug mode via environment variables in your `.env` file:

```bash
# Core debug settings
DEBUG_MODE=true                    # Master debug switch
DEBUG_LEVEL=2                      # Verbosity level (0-3)
DEBUG_MODULES=research_director,workflow  # Comma-separated module filter (optional)

# Specific logging toggles
LOG_LLM_CALLS=true                 # Log LLM request/response summaries
LOG_AGENT_MESSAGES=true            # Log inter-agent message routing
LOG_WORKFLOW_TRANSITIONS=true      # Log state machine transitions with timing

# Stage tracking (real-time observability)
STAGE_TRACKING_ENABLED=true        # Enable stage tracking output
STAGE_TRACKING_FILE=logs/stages.jsonl  # Output file path
```

### CLI Flags

All debug settings can be overridden via CLI flags:

```bash
# Enable trace mode (maximum verbosity)
kosmos run --trace --objective "..."

# Set specific debug level
kosmos run --debug-level 2 --objective "..."

# Debug specific modules only
kosmos run --debug --debug-modules "research_director,workflow" --objective "..."

# Combine with quiet mode (suppress Rich formatting, keep debug logs)
kosmos run --trace --quiet --objective "..."
```

| Flag | Short | Description |
|------|-------|-------------|
| `--debug` | | Enable debug mode (level 1) |
| `--trace` | | Enable trace mode (level 3, all toggles on) |
| `--debug-level N` | `-dl N` | Set specific debug level (0-3) |
| `--debug-modules M` | | Comma-separated list of modules to debug |
| `--verbose` | `-v` | Enable verbose output (INFO level) |
| `--quiet` | `-q` | Suppress non-essential console output |

### Understanding Debug Output

#### Decision Logging (`[DECISION]`)

Shows research director decision-making:

```
[DECISION] decide_next_action: state=ANALYZING, iteration=2/10, hypotheses=3, untested=1, experiments_queued=0
```

Fields:
- `state`: Current workflow state
- `iteration`: Current/max iteration count
- `hypotheses`: Total hypotheses generated
- `untested`: Hypotheses not yet tested
- `experiments_queued`: Pending experiments

#### Action Logging (`[ACTION]`)

Shows action execution:

```
[ACTION] Executing: GENERATE_HYPOTHESIS
```

#### Agent Message Logging (`[MSG]`)

Shows inter-agent communication (enable with `LOG_AGENT_MESSAGES=true` or `--trace`):

```
[MSG] research_director -> hypothesis_generator: type=REQUEST, correlation_id=abc123, content_preview={"task": "generate"...
[MSG] hypothesis_generator <- research_director: type=REQUEST, msg_id=abc123
```

Fields:
- `->`: Outgoing message (sender -> recipient)
- `<-`: Incoming message (recipient <- sender)
- `type`: Message type (REQUEST, RESPONSE, NOTIFICATION, ERROR)
- `correlation_id`: Links request/response pairs
- `msg_id`: Unique message identifier
- `content_preview`: First 100 chars of message content (outgoing only)

#### Workflow Transitions (`[WORKFLOW]`)

Shows state machine transitions with timing:

```
[WORKFLOW] Transition: HYPOTHESIZING -> DESIGNING (was in HYPOTHESIZING for 12.34s) action='hypothesis_complete'
```

#### LLM Call Logging (`[LLM]`)

Shows LLM API interactions:

```
[LLM] Request: model=claude-sonnet-4-5-20241022, prompt_len=2456, system_len=890, max_tokens=4096, temp=0.70
[LLM] Response: model=claude-sonnet-4-5-20241022, in_tokens=3346, out_tokens=1205, latency=4521ms, finish=end_turn
```

#### Iteration Summary (`[ITER]`)

Shows per-iteration progress:

```
[ITER 2/10] state=ANALYZING, hyps=3, exps=2, duration=45.67s
```

### Stage Tracking

Stage tracking provides structured JSON output for programmatic analysis and real-time monitoring.

#### Enabling Stage Tracking

```bash
# Via environment
STAGE_TRACKING_ENABLED=true
STAGE_TRACKING_FILE=logs/stages.jsonl

# Via CLI (--trace enables automatically)
kosmos run --trace --objective "..."
```

#### Stage Output Format

Each stage event is written as a JSON line to `logs/stages.jsonl`:

```json
{
  "timestamp": "2025-11-29T14:23:45.123Z",
  "process_id": "research_1732889025",
  "stage": "GENERATE_HYPOTHESIS",
  "status": "completed",
  "duration_ms": 3456,
  "iteration": 2,
  "parent_stage": "RESEARCH_ITERATION",
  "substage": null,
  "output_summary": null,
  "error": null,
  "metadata": {"hypothesis_count": 3}
}
```

#### Viewing Stage Events

```bash
# Watch stages in real-time
tail -f logs/stages.jsonl | jq .

# Filter for failed stages
cat logs/stages.jsonl | jq 'select(.status == "failed")'

# Get timing summary
cat logs/stages.jsonl | jq 'select(.status == "completed") | {stage, duration_ms}'

# Count stages by type
cat logs/stages.jsonl | jq -s 'group_by(.stage) | map({stage: .[0].stage, count: length})'
```

#### Programmatic Access

```python
from kosmos.core.stage_tracker import get_stage_tracker

# Get tracker instance
tracker = get_stage_tracker()

# Get all recorded events
events = tracker.get_events()

# Get summary statistics
summary = tracker.get_summary()
print(f"Total stages: {summary['total_stages']}")
print(f"Completed: {summary['completed']}")
print(f"Failed: {summary['failed']}")
print(f"Total duration: {summary['total_duration_ms']}ms")
```

### Troubleshooting Common Issues

#### Research Loop Stalls

Enable full trace to identify where execution stops:

```bash
kosmos run --trace --objective "..." --max-iterations 3 2>&1 | tee debug.log
```

Look for:
- Last `[DECISION]` entry to see decision state
- Missing `[ACTION]` after `[DECISION]` indicates decision logic issue
- Long gaps in `[LLM]` responses may indicate API timeouts

#### LLM Errors

Enable LLM call logging:

```bash
LOG_LLM_CALLS=true kosmos run --debug --objective "..."
```

Check for:
- Token count approaching limits
- High latency responses
- Unexpected finish reasons

#### Workflow Stuck in State

Enable workflow transition logging:

```bash
LOG_WORKFLOW_TRANSITIONS=true kosmos run --debug-level 2 --objective "..."
```

Look for:
- Repeated transitions to same state
- Long duration in single state
- Missing expected transitions

### Log File Locations

| File | Content |
|------|---------|
| `logs/kosmos.log` | Main application log |
| `logs/stages.jsonl` | Stage tracking events (JSON lines) |
| `~/.kosmos/logs/` | CLI-specific logs |

### Performance Considerations

Debug logging adds overhead. For production runs:

```bash
# Minimal logging (recommended for long runs)
DEBUG_LEVEL=0 kosmos run --objective "..." --max-iterations 20

# Monitor progress without full debug
STAGE_TRACKING_ENABLED=true DEBUG_LEVEL=0 kosmos run --objective "..."
```

Stage tracking alone (without full debug logging) adds minimal overhead and is suitable for production monitoring.

## Documentation

### Current Status
- [120525_implementation_gaps_v2.md](120525_implementation_gaps_v2.md) - **Detailed gaps analysis** (December 2025)
- [120525_implementation_plan_v2.md](120525_implementation_plan_v2.md) - **Implementation plan with code** (December 2025)
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines

### Architecture
- [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) - Original paper gap analysis
- [OPENQUESTIONS_SOLUTION.md](OPENQUESTIONS_SOLUTION.md) - How gaps were addressed
- [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) - Architecture decisions

### Operations
- [GETTING_STARTED.md](GETTING_STARTED.md) - Usage examples
- [TESTS_STATUS.md](TESTS_STATUS.md) - Test coverage
- [MODEL_COMPARISON_REPORT.md](MODEL_COMPARISON_REPORT.md) - Multi-model performance

## Based On

- **Paper**: [Kosmos: An AI Scientist for Autonomous Discovery](https://arxiv.org/abs/2511.02824) (Lu et al., 2024)
- **K-Dense ecosystem**: Pattern repositories for AI agent systems
- **kosmos-figures**: [Analysis patterns](https://github.com/EdisonScientific/kosmos-figures)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

Areas where contributions would be useful:
- Docker sandbox testing and hardening
- Integration test updates
- R language support via rpy2
- Additional scientific domain skills
- Performance benchmarking

## License

MIT License - see [LICENSE](LICENSE).

---

**Version**: 0.2.0-alpha
**Core Implementation**: 75% production-ready | 20% Phase 2/4 deferred | 5% known issues
**Tests**: 339 unit + 43 integration passing | 7 E2E skipped (environment-dependent)
**Features**: Debug mode, stage tracking, multi-provider LLM support, Neo4j knowledge graph
**Last Updated**: 2025-12-05

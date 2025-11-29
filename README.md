# Kosmos

Open-source implementation of the autonomous AI scientist described in [Lu et al. (2024)](https://arxiv.org/abs/2511.02824). The original paper reported 79.4% accuracy on scientific statements and 7 validated discoveries, but omitted implementation details for 6 critical components. This repository provides those implementations using patterns from the K-Dense ecosystem.

[![Version](https://img.shields.io/badge/version-0.2.0--alpha-blue.svg)](https://github.com/jimmc414/Kosmos)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/jimmc414/Kosmos)
[![Implementation](https://img.shields.io/badge/gaps-6%2F6%20complete-green.svg)](IMPLEMENTATION_REPORT.md)
[![Tests](https://img.shields.io/badge/unit-339%20passing-green.svg)](TESTS_STATUS.md)
[![Tests](https://img.shields.io/badge/E2E-32%2F39%20passing-yellow.svg)](TESTS_STATUS.md)

**Current state**: All 6 gaps implemented. Debug mode with enhanced observability now available. Model comparison infrastructure tested across DeepSeek, GPT-4o-mini, and Claude Haiku 4.5. E2E testing validated with local LLMs via Ollama/LiteLLM. See [Project Status](#project-status) for honest assessment of remaining work.

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

### Test Results (as of 2025-11-29)

| Category | Total | Pass | Fail | Skip | Notes |
|----------|-------|------|------|------|-------|
| Unit tests | 339 | 339 | 0 | 0 | Core gap implementations |
| Integration | 43 | 43 | 0 | 0 | Pipeline tests |
| LiteLLM provider | 22 | 22 | 0 | 0 | Multi-provider support |
| E2E tests | 39 | 32 | 0 | 7 | Tested with Ollama |

Some E2E tests are skipped based on environment:
- Docker not running (sandbox execution tests)
- Neo4j not configured (knowledge graph tests)
- API keys not set (tests requiring LLM calls)

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

### What Does Not Work Yet

1. **Knowledge graph**: Neo4j integration requires external database setup. The code exists but is untested in E2E flows.

2. **Full autonomous loop**: While individual components work, running 20 cycles with 10 tasks each (as described in the paper) has not been validated. The workflow tends to converge early or requires manual intervention.

3. **Budget enforcement**: Cost calculation is implemented in all providers, but there is no mechanism to halt execution when a budget limit is reached.

4. **Literature search**: The `arxiv` package has Python 3.11+ compatibility issues. Literature features are limited.

### Honest Assessment

This implementation provides the architectural skeleton described in the Lu et al. paper. The 6 gaps identified in the paper have been filled with working code. However:

- We have not reproduced the paper's claimed 79.4% accuracy or 7 validated discoveries
- The system has been tested primarily with small local models (Qwen 4B via Ollama), not production-scale LLMs
- Multi-cycle autonomous research runs have not been validated end-to-end
- The codebase has accumulated technical debt from rapid development

The project is suitable for experimentation and further development, not production research use.

### Next Steps

1. Validate full research cycles with production LLMs
2. Integrate Docker sandbox into E2E test suite
3. Fix remaining test setup issues
4. Add cost tracking and budget limits
5. Document actual performance vs paper claims

## Limitations

1. **Docker required**: Gap 4 execution environment requires Docker. Without it, code execution uses mock implementations.

2. **Dependency compatibility**: The `arxiv` package fails to build on Python 3.11+ due to `sgmllib3k` incompatibility. Literature search features are limited without this package.

3. **Python only**: The paper references R packages (MendelianRandomization, susieR). This implementation is Python-only.

4. **LLM costs**: Running 20 research cycles with 10 tasks each requires significant API usage. Cost tracking is implemented but budget enforcement during execution is not.

5. **Single-user**: No multi-tenancy or user isolation.

6. **Not a reproduction study**: We have not reproduced the paper's 7 validated discoveries. This is an implementation of the architecture, not a validation of the results.

7. **Integration test maintenance**: Some integration tests have API mismatches with current implementation.

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

### Debug Mode

```bash
# Enable debug mode
DEBUG_MODE=true

# Debug verbosity level (0-3)
# 0=off, 1=critical path, 2=full trace, 3=data dumps
DEBUG_LEVEL=2

# Log LLM request/response summaries
LOG_LLM_CALLS=true

# Log inter-agent message routing
LOG_AGENT_MESSAGES=true

# Log workflow state transitions with timing
LOG_WORKFLOW_TRANSITIONS=true

# Enable real-time stage tracking (outputs to logs/stages.jsonl)
STAGE_TRACKING_ENABLED=true
STAGE_TRACKING_FILE=logs/stages.jsonl
```

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

## Documentation

- [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) - Original gap analysis
- [OPENQUESTIONS_SOLUTION.md](OPENQUESTIONS_SOLUTION.md) - How gaps were addressed
- [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) - Architecture decisions
- [PRODUCTION_READINESS_REPORT.md](PRODUCTION_READINESS_REPORT.md) - Current status
- [TESTS_STATUS.md](TESTS_STATUS.md) - Test coverage
- [MODEL_COMPARISON_REPORT.md](MODEL_COMPARISON_REPORT.md) - Multi-model performance comparison
- [GETTING_STARTED.md](GETTING_STARTED.md) - Usage examples

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

Version: 0.2.0-alpha
Gap Implementation: 6/6 complete
Test Coverage: 339 unit tests + 43 integration tests passing
Features: Debug mode, stage tracking, multi-provider support, model comparison
Last Updated: 2025-11-29

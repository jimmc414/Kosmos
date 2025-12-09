# Kosmos

An autonomous AI scientist for scientific discovery, implementing the architecture described in [Lu et al. (2024)](https://arxiv.org/abs/2511.02824).

[![Version](https://img.shields.io/badge/version-0.2.0--alpha-blue.svg)](https://github.com/jimmc414/Kosmos)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/jimmc414/Kosmos)
[![Implementation](https://img.shields.io/badge/core-90%25%20complete-green.svg)](120525_implementation_gaps_v2.md)
[![Tests](https://img.shields.io/badge/tests-3621%20passing-green.svg)](120625_code_review.md)

## What is Kosmos?

Kosmos is an open-source implementation of an autonomous AI scientist that can:

- **Generate hypotheses** from literature and data analysis
- **Design experiments** to test those hypotheses
- **Execute code** in sandboxed Docker containers
- **Validate discoveries** using an 8-dimension quality framework
- **Build knowledge graphs** to track relationships between concepts

The system runs autonomous research cycles, generating tasks, executing analyses, and synthesizing findings into validated discoveries.

## Quick Start

### Requirements

- Python 3.11+
- Anthropic API key or OpenAI API key
- Docker (recommended for code execution)

Without Docker, code runs via `exec()` with static validation. See "Code Execution Security" below.

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

# Run unit tests
pytest tests/unit/ -v --tb=short
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

### CLI Usage

```bash
# Run research with default settings
kosmos run "What metabolic pathways differ between cancer and normal cells?" --domain biology

# With budget limit
kosmos run "How do perovskites optimize efficiency?" --domain materials --budget 50

# Interactive mode (recommended for first time)
kosmos run --interactive

# Maximum verbosity
kosmos run "Your question" --domain biology --trace

# Show system information
kosmos info

# Run diagnostics
kosmos doctor
```

## Features

### Core Capabilities

| Feature | Description | Status |
|---------|-------------|--------|
| Research Loop | Multi-cycle autonomous research with hypothesis generation | Complete |
| Literature Search | ArXiv, PubMed, Semantic Scholar integration | Complete |
| Code Execution | Docker-sandboxed Jupyter notebooks | Complete |
| Knowledge Graph | Neo4j-based relationship storage (optional) | Complete |
| Context Compression | Query-based hierarchical compression (20:1 ratio) | Complete |
| Discovery Validation | 8-dimension ScholarEval quality framework | Complete |
| Multi-Provider LLM | Anthropic, OpenAI, LiteLLM (100+ providers) | Complete |
| Budget Enforcement | Cost tracking with configurable limits and enforcement | Complete |
| Error Recovery | Exponential backoff with circuit breaker | Complete |
| Debug Mode | 4-level verbosity with stage tracking | Complete |

### Code Execution Security

AI-generated code runs in isolated Docker containers:

| Layer | Implementation |
|-------|---------------|
| Container Isolation | `--cap-drop=ALL`, no privileged access |
| Network | Disabled (`--network=none`) |
| Filesystem | Read-only root, tmpfs for scratch |
| Resources | CPU: 2 cores, Memory: 2GB, Timeout: 300s |
| Pooling | Pre-warmed containers reduce cold start |

See: `kosmos/execution/sandbox.py`, `docker_manager.py`

Without Docker, falls back to `CodeValidator` static analysis + `exec()`. Not recommended for untrusted inputs.

### Agent Architecture

| Agent | Role |
|-------|------|
| Research Director | Master orchestrator coordinating all agents |
| Hypothesis Generator | Generates testable hypotheses from literature |
| Experiment Designer | Creates experimental protocols |
| Data Analyst | Analyzes results and interprets findings |
| Literature Analyzer | Searches and synthesizes papers |
| Plan Creator/Reviewer | Strategic task generation with 70/30 exploration/exploitation |

### How Context Compression Works

The system processes literature in batches, not bulk:

1. **Relevance Sorting**: Papers ranked by query relevance before processing
2. **Batch Size**: Top 10 papers per batch
3. **Statistics Extraction**: Regex-based extraction of p-values, sample sizes, effect sizes
4. **Tiered Summarization**:
   - Task: 42K lines code to 2-line summary + extracted stats
   - Cycle: 10 task summaries to cycle overview
   - Synthesis: 20 cycles to final narrative
   - Detail: Full content lazy-loaded when needed

Effective ratio: ~20:1. See `kosmos/compression/compressor.py`.

## Configuration

All configuration via environment variables. See `.env.example` for the full list.

### LLM Provider

```bash
# Anthropic (default)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# LiteLLM (supports 100+ providers including local models)
LLM_PROVIDER=litellm
LITELLM_MODEL=ollama/llama3.1:8b
LITELLM_API_BASE=http://localhost:11434
```

### Budget Control

```bash
BUDGET_ENABLED=true
BUDGET_LIMIT_USD=10.00
```

Budget enforcement raises `BudgetExceededError` when the limit is reached, gracefully transitioning the research to completion.

### Concurrency

Three independent limits in `kosmos/config.py`:

| Setting | Default | Range |
|---------|---------|-------|
| `max_parallel_hypotheses` | 3 | 1-10 |
| `max_concurrent_experiments` | 10 | 1-16 |
| `max_concurrent_llm_calls` | 5 | 1-20 |

The paper describes 10 parallel tasks. Default now matches paper specification.

### Optional Services

```bash
# Neo4j (optional, for knowledge graph features)
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=your-password

# Redis (optional, for distributed caching)
REDIS_URL=redis://localhost:6379
```

#### Docker Setup for Optional Services

Start Neo4j, Redis, and PostgreSQL with Docker Compose:

```bash
# Start all optional services (Neo4j, Redis, PostgreSQL)
docker compose --profile dev up -d

# Or start individual services
docker compose up -d neo4j
docker compose up -d redis
docker compose up -d postgres

# Stop services
docker compose --profile dev down
```

Service URLs when running via Docker:
- Neo4j Browser: http://localhost:7474 (user: neo4j, password: kosmos-password)
- PostgreSQL: localhost:5432 (user: kosmos, password: kosmos-dev-password)
- Redis: localhost:6379

#### Semantic Scholar API

Literature search via Semantic Scholar works without authentication. An API key is optional but increases rate limits:

```bash
# Optional: Get API key from https://www.semanticscholar.org/product/api
SEMANTIC_SCHOLAR_API_KEY=your-key-here
```

### Debug Mode

```bash
# Enable debug mode with level 1-3
DEBUG_MODE=true
DEBUG_LEVEL=2

# Or use CLI flag for maximum verbosity
kosmos run "Your research question" --trace
```

See [docs/DEBUG_MODE.md](docs/DEBUG_MODE.md) for comprehensive debug documentation.

## Architecture

```
kosmos/
├── agents/           # Research agents (director, hypothesis, experiment, etc.)
├── compression/      # Context compression (20:1 ratio)
├── core/             # LLM providers, metrics, configuration
│   └── providers/    # Anthropic, OpenAI, LiteLLM with async support
├── execution/        # Docker-based sandboxed code execution
├── knowledge/        # Neo4j knowledge graph (1,025 lines)
├── literature/       # ArXiv, PubMed, Semantic Scholar clients
├── orchestration/    # Plan creation/review, task delegation
├── validation/       # ScholarEval 8-dimension quality framework
├── workflow/         # Main research loop integration
└── world_model/      # State management, JSON artifacts
```

## Project Status

### Implementation Completeness

| Category | Percentage | Description |
|----------|------------|-------------|
| Production-ready | 90% | Core research loop, agents, LLM providers, validation |
| Deferred | 5% | Phase 4 production mode (polyglot persistence) |
| Known issues | 5% | Some environment-dependent tests skip |

### Fixed Issues (Recent)

| Issue | Description | Status |
|-------|-------------|--------|
| [#66](https://github.com/jimmc414/Kosmos/issues/66) | CLI deadlock - async refactor | ✅ Fixed |
| [#67](https://github.com/jimmc414/Kosmos/issues/67) | SkillLoader domain mapping | ✅ Fixed |
| [#68](https://github.com/jimmc414/Kosmos/issues/68) | Pydantic V2 migration | ✅ Fixed |
| [#54-#58](https://github.com/jimmc414/Kosmos/issues/54) | Critical paper gaps | ✅ Fixed |
| [#59](https://github.com/jimmc414/Kosmos/issues/59) | h5ad/Parquet data formats | ✅ Fixed |
| [#69](https://github.com/jimmc414/Kosmos/issues/69) | R language execution | ✅ Fixed |
| [#60](https://github.com/jimmc414/Kosmos/issues/60) | Figure generation | ✅ Fixed |
| [#61](https://github.com/jimmc414/Kosmos/issues/61) | Jupyter notebook generation | ✅ Fixed |
| [#70](https://github.com/jimmc414/Kosmos/issues/70) | Null model statistical validation | ✅ Fixed |
| [#63](https://github.com/jimmc414/Kosmos/issues/63) | Failure mode detection | ✅ Fixed |
| [#62](https://github.com/jimmc414/Kosmos/issues/62) | Code line provenance | ✅ Fixed |
| [#64](https://github.com/jimmc414/Kosmos/issues/64) | Multi-run convergence framework | ✅ Fixed |
| [#65](https://github.com/jimmc414/Kosmos/issues/65) | Paper accuracy validation | ✅ Fixed |

### Implementation Complete

All 17 paper implementation gaps have been addressed. Full tracking: [PAPER_IMPLEMENTATION_GAPS.md](docs/PAPER_IMPLEMENTATION_GAPS.md)

### Test Coverage

| Category | Count | Status |
|----------|-------|--------|
| Unit tests | 2168 | Passing |
| Integration tests | 401 | Passing |
| E2E tests | 121 | Most pass, some skip (environment-dependent) |
| Requirements tests | 815 | Passing |

E2E tests skip based on environment:
- Neo4j not configured (`@pytest.mark.requires_neo4j`)
- Docker not running (sandbox execution tests)
- API keys not set (tests requiring live LLM calls)

### Paper Implementation

This project implements the architecture from the Kosmos paper but **has not yet reproduced** the paper's claimed results:

| Paper Claim | Implementation Status |
|-------------|----------------------|
| 79.4% accuracy on scientific statements | Architecture implemented, not validated |
| 7 validated discoveries | Not reproduced |
| 1,500 papers per run | Architecture supports this |
| 42,000 lines of code per run | Architecture supports this |
| 200 agent rollouts | Configurable via `max_iterations` |

The system is suitable for experimentation and further development. Before production research use, validation studies should be conducted.

## Limitations

1. **Docker recommended**: Without Docker, code execution falls back to direct `exec()` which is unsafe for untrusted code.

2. **Neo4j optional**: Knowledge graph features require Neo4j. Set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` to enable.

3. **R support via Docker**: R language execution requires the R-enabled Docker image (`docker/sandbox/Dockerfile.r`) with TwoSampleMR, susieR, and MendelianRandomization packages.

4. **Single-user**: No multi-tenancy or user isolation.

5. **Not a reproduction study**: We have not yet reproduced the paper's 79.4% accuracy or 7 validated discoveries.

## Documentation

### Current Status
- [120525_implementation_gaps_v2.md](120525_implementation_gaps_v2.md) - Implementation gaps analysis
- [120625_code_review.md](120625_code_review.md) - Comprehensive code review
- [docs/DEBUG_MODE.md](docs/DEBUG_MODE.md) - Debug mode guide

### Operations
- [GETTING_STARTED.md](GETTING_STARTED.md) - Detailed usage examples
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines
- [CHANGELOG.md](CHANGELOG.md) - Version history

## Paper Gap Solutions

The original paper omitted implementation details for 6 critical components. This repository provides those implementations:

| Gap | Problem | Solution |
|-----|---------|----------|
| 0 | Context compression for 1,500 papers | Hierarchical 3-tier compression (20:1 ratio) |
| 1 | State Manager schema unspecified | 4-layer hybrid architecture (JSON + Neo4j + Vector + Citations) |
| 2 | Task generation algorithm unstated | Plan Creator + Plan Reviewer pattern |
| 3 | Agent integration mechanism unclear | Skill loader with 116 domain-specific skills (see [#67](https://github.com/jimmc414/Kosmos/issues/67)) |
| 4 | Execution environment not described | Docker-based Jupyter sandbox with pooling |
| 5 | Discovery validation criteria missing | ScholarEval 8-dimension quality framework |

For detailed analysis, see [120525_implementation_gaps_v2.md](120525_implementation_gaps_v2.md).

## Based On

- **Paper**: [Kosmos: An AI Scientist for Autonomous Discovery](https://arxiv.org/abs/2511.02824) (Lu et al., 2024)
- **K-Dense ecosystem**: Pattern repositories for AI agent systems
- **kosmos-figures**: [Analysis patterns](https://github.com/EdisonScientific/kosmos-figures)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

Areas where contributions would be useful:
- Docker sandbox testing and hardening
- R language support via rpy2
- Additional scientific domain skills
- Performance benchmarking with production LLMs
- Validation studies to measure actual accuracy

## License

MIT License

---

**Version**: 0.2.0-alpha | **Tests**: 3621 passing | **Last Updated**: 2025-12-09

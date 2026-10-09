# Kosmos

> **Not a reproduction.** This project follows the architecture described in the Kosmos paper
> (Mitchener et al., Edison Scientific, November 2025). It has not reproduced the paper's results,
> and none of the paper's claims below has been measured here.

An autonomous AI scientist for scientific discovery, built on the architecture described in [Mitchener et al. (2025)](https://arxiv.org/abs/2511.02824).

[![Version](https://img.shields.io/badge/version-0.2.0--alpha-blue.svg)](https://github.com/jimmc414/Kosmos)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/jimmc414/Kosmos)

## What is Kosmos?

Kosmos is an open-source implementation of an autonomous AI scientist that can:

- **Generate hypotheses** from literature and data analysis
- **Design experiments** to test those hypotheses
- **Execute code** in sandboxed Docker containers
- **Validate discoveries** using an 8-dimension quality framework
- **Build knowledge graphs** to track relationships between concepts

The system runs autonomous research cycles, generating tasks, executing analyses, and synthesizing findings into validated discoveries.

### What it does today

`kosmos run` drives one research director per question. Each iteration generates hypotheses,
binds their variables to the columns of your dataset (`--data-path`), and runs one statistical test
per experiment in the Docker sandbox. Every result is then validated in three ways: the statistic
is recomputed on the real file, a permutation null shuffles the dependent column, and ScholarEval
scores it as an advisory signal. A finding counts as validated only when the recomputation matches
and the null rejects. Each run reports the experiments attempted, succeeded and failed, the results
from the real file versus synthetic data, and how many findings were validated or rejected. Each result row shows its test, statistic, p-value, n, data source and validation
status. The run also reports untested hypotheses and the real token cost from the provider
(`total_cost_usd`, cost per validated finding). Measured so far: the no-LLM bound template
on the bundled climate CSV recovers the known truth (Pearson r 0.9317, p 5.8e-29, n 64) on every
ladder run. A live end-to-end acceptance run with a real model is still pending, so
model-driven discovery rates have not been measured.

## Quick Start

### Requirements

- Python 3.11+
- An LLM provider: DeepSeek or another LiteLLM model, an Anthropic API key, a Claude Code login, or OpenAI
- Docker, with the sandbox image built: `docker build -t kosmos-sandbox:latest docker/sandbox`

Docker is required for sandboxed execution; `kosmos run` records a SandboxUnavailable failure when the daemon is unreachable.

### Installation

```bash
git clone https://github.com/jimmc414/Kosmos.git
cd Kosmos
pip install -e .
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY or OPENAI_API_KEY
```

The core install includes litellm. Optional extras:

| Extra | Adds | Needed for |
|-------|------|-----------|
| `execution` | `docker` | Running experiments in the Docker sandbox (`pip install -e ".[execution]"`) |
| `embeddings` | `sentence-transformers` (pulls torch) | SPECTER novelty and vector paper search; without it novelty uses TF-IDF |
| `postgres` | `psycopg2-binary` | `DATABASE_URL=postgresql://...` |
| `claude-code` | `claude-agent-sdk` | `--provider claude-code` (Anthropic models through a Claude Code login) |
| `server` | `fastapi`, `uvicorn`, `requests` | `kosmos/api/health.py` probes and alert webhooks; Kosmos runs no HTTP server |

### Verify Installation

```bash
# Report provider, model, litellm, Docker daemon, sandbox image and database URL
python scripts/check_env.py

# Run unit tests (pytest.ini enables an 80% coverage gate; --no-cov skips it)
python -m pytest tests/unit --no-cov -q

# The full verification ladder: lint, unit + integration tests, alembic, and a no-LLM
# template run through the Docker sandbox on the climate CSV (about 5-8 minutes)
bash scripts/verify.sh
```

### Run Research

```bash
kosmos run "Does atmospheric CO2 concentration predict global temperature anomaly?" \
  --domain climate_science \
  --data-path evaluation/data/climate_co2_temperature_test.csv \
  --seed 42 --max-iterations 3 --budget 1
```

The general form is `kosmos run "<question>" --domain <d> --data-path <csv> --seed 42 --max-iterations 3 --budget 1`.
`--data-path` points the experiments at your dataset (without it, results are labelled synthetic and are
never counted as supported), `--seed` makes reruns reproducible, and `--budget` is a hard limit in USD.

### CLI Usage

```bash
# Run research on your dataset, reproducibly
kosmos run "What metabolic pathways differ between cancer and normal cells?" --domain biology \
  --data-path data/expression.csv --seed 42

# With budget limit (USD)
kosmos run "How do perovskites optimize efficiency?" --domain materials --data-path data/cells.csv --budget 5

# Pick the provider and model for one run
kosmos run "Your question" --data-path data.csv --provider deepseek
kosmos run "Your question" --data-path data.csv --provider claude-code --model opus

# Interactive mode (recommended for first time)
kosmos run --interactive

# Maximum verbosity
kosmos run "Your question" --domain biology --trace

# Real-time streaming display
kosmos run "Your question" --stream

# Streaming with token display disabled
kosmos run "Your question" --stream --no-stream-tokens

# Show system information
kosmos info

# Run diagnostics
kosmos doctor
```

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| Research Director | One orchestrator per question: hypotheses, experiment design, execution, analysis, refinement, convergence |
| Data binding | Hypothesis variables bound to dataset columns; unbindable hypotheses are reported as untestable |
| Code Execution | Generated Python run in the Docker sandbox, validated by CodeValidator before it runs |
| Validation | Recomputation on the real data plus a permutation null; ScholarEval is advisory, never a gate |
| Literature Search | ArXiv, PubMed, Semantic Scholar integration |
| Knowledge Graph | Neo4j-based relationship storage (optional) |
| Multi-Provider LLM | DeepSeek and other LiteLLM models, Anthropic, Claude Code login, OpenAI |
| Budget Enforcement | Real provider cost tracked per call and per result; `--budget` halts the run |
| Error Recovery | Exponential backoff with circuit breaker |
| Debug Mode | 4-level verbosity with stage tracking (`--trace`) |
| CLI streaming | `kosmos run --stream` shows progress in the terminal |

Archived (not on the run path; kept under `archive/code/`): the library research loop
(`kosmos/workflow`), plan creator/reviewer (`kosmos/orchestration`), context compression
(`kosmos/compression`), and the SSE/WebSocket API.

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

Docker is required for sandboxed execution; `kosmos run` records a SandboxUnavailable failure when the daemon is unreachable.

### Agent Architecture

| Agent | Role |
|-------|------|
| Research Director | Master orchestrator coordinating all agents |
| Hypothesis Generator | Generates testable hypotheses from literature |
| Experiment Designer | Creates experimental protocols |
| Data Analyst | Analyzes results and interprets findings |
| Literature Analyzer | Searches and synthesizes papers |

## Configuration

All configuration via environment variables. See `.env.example` for the full list.

### LLM Provider

```bash
# DeepSeek and other providers through LiteLLM (local models included)
LLM_PROVIDER=litellm
LITELLM_MODEL=deepseek/deepseek-chat
DEEPSEEK_API_KEY=sk-...

# Anthropic API, billed per token. Prefer KOSMOS_ANTHROPIC_API_KEY: exporting
# ANTHROPIC_API_KEY overrides a Claude Code subscription login for other tools.
LLM_PROVIDER=anthropic
KOSMOS_ANTHROPIC_API_KEY=sk-ant-api03-...
CLAUDE_MODEL=claude-opus-5-5

# Anthropic models through your Claude Code login (e.g. a Max subscription), no API key.
# Needs `pip install -e ".[claude-code]"`, the `claude` CLI on PATH and `claude login`.
LLM_PROVIDER=claude_code
CLAUDE_CODE_MODEL=claude-opus-5-5

# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

`.env` sets the default. Override it for a single run with `--provider` and `--model`:

```bash
kosmos run "Question" --provider deepseek
kosmos run "Question" --provider claude-code --model opus     # Claude Code login, no API key
kosmos run "Question" --provider anthropic --model sonnet     # needs KOSMOS_ANTHROPIC_API_KEY
```

Model aliases: `opus` (claude-opus-5-5), `sonnet` (claude-sonnet-5-5), `haiku` (claude-haiku-4-5), `fable` (claude-fable-5-1), `deepseek-chat`, `deepseek-reasoner`. A mismatched pair such as `--provider deepseek --model opus` is rejected with the flag to use instead. `kosmos doctor` reports the default provider and which credentials are set.

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

Service URLs when running via Docker (bound to 127.0.0.1; the passwords come from `.env`:
`KOSMOS_PG_SUPER_PASSWORD`, `REDIS_PASSWORD`, `NEO4J_AUTH`, `KOSMOS_PGADMIN_PASSWORD`):
- Neo4j Browser: http://localhost:7474
- PostgreSQL: localhost:5432 (user: kosmos)
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
├── agents/           # Research agents; research_director.py drives `kosmos run`
├── cli/              # Typer CLI (run, doctor, info, config, cache, history, status, graph)
├── core/             # LLM providers, metrics, convergence, workflow state machine
│   └── providers/    # Anthropic, OpenAI, LiteLLM, Claude Code
├── db/               # SQLAlchemy models and operations (results carry provenance and cost)
├── execution/        # Code generation and Docker-sandboxed execution
├── knowledge/        # Neo4j knowledge graph (optional)
├── literature/       # ArXiv, PubMed, Semantic Scholar clients
├── safety/           # CodeValidator, guardrails, emergency stop
├── validation/       # Recomputation, permutation null, ScholarEval (advisory)
└── world_model/      # State management, JSON artifacts
```

## Project Status

### Implementation Status

The viability plan (`evaluation/VIABILITY_ASSESSMENT_AND_CHANGE_PLAN.md`) is being worked on the
`viability-fixes` branch: one real experiment per run on the owner's dataset, validated by
recomputation and a permutation null, reported with real cost. The earlier "17/17 paper gaps
complete" figure counted code that exists, not behavior that was measured.

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
| [#72](https://github.com/jimmc414/Kosmos/issues/72) | Real-time streaming API | ✅ Fixed |

### Paper gap tracking

[archive/PAPER_IMPLEMENTATION_GAPS.md](archive/PAPER_IMPLEMENTATION_GAPS.md) was checked on the existence of code, not on measured behavior.

### Test Coverage

```bash
python -m pytest tests/unit --no-cov -q
```

Counts change; CI is authoritative (`.github/workflows/unit.yml` runs the unit suite on every push).
Unit and integration tests run hermetic: `tests/conftest.py` ignores `.env` and removes credential
variables, so they never call a live service with your keys.

E2E tests skip based on environment:
- Neo4j not configured (`@pytest.mark.requires_neo4j`)
- Docker not running (sandbox execution tests)
- API keys not set (tests requiring live LLM calls)

### Paper Implementation

This project implements the architecture from the Kosmos paper but **has not yet reproduced** the paper's claimed results:

| Paper Claim | Implementation Status | Measured |
|-------------|----------------------|----------|
| 79.4% accuracy on scientific statements | Architecture implemented, not validated | no |
| 7 validated discoveries | Not reproduced | no |
| 1,500 papers per run | Not exercised (literature search is per hypothesis) | no |
| 42,000 lines of code per run | Not exercised (one bound test per experiment) | no |
| 200 agent rollouts | Configurable via `max_iterations` | no |

The system is suitable for experimentation and further development. Before production research use, validation studies should be conducted.

## Limitations

1. **Docker required**: Docker is required for sandboxed execution; `kosmos run` records a SandboxUnavailable failure when the daemon is unreachable.

2. **Neo4j optional**: Knowledge graph features require Neo4j. Set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` to enable.

3. **R support via Docker**: R language execution requires the R-enabled Docker image (`docker/sandbox/Dockerfile.r`) with TwoSampleMR, susieR, and MendelianRandomization packages.

4. **Single-user**: No multi-tenancy or user isolation.

5. **Not a reproduction study**: We have not yet reproduced the paper's 79.4% accuracy or 7 validated discoveries.

## Documentation

### Current Status
- [archive/PAPER_IMPLEMENTATION_GAPS.md](archive/PAPER_IMPLEMENTATION_GAPS.md) - Paper implementation gaps (checked on code existence, not measured behavior)
- [evaluation/VIABILITY_ASSESSMENT_AND_CHANGE_PLAN.md](evaluation/VIABILITY_ASSESSMENT_AND_CHANGE_PLAN.md) - What was measured and what changes make a run viable
- [docs/DEBUG_MODE.md](docs/DEBUG_MODE.md) - Debug mode guide

### Archived Analysis
- [archive/120525_implementation_gaps_v2.md](archive/120525_implementation_gaps_v2.md) - Original implementation gaps analysis
- [archive/120625_code_review.md](archive/120625_code_review.md) - Code review (Dec 2025)

### Operations
- [archive/GETTING_STARTED.md](archive/GETTING_STARTED.md) - Detailed usage examples
- [CONTRIBUTING.md](archive/CONTRIBUTING.md) - Development guidelines (archived)
- [CHANGELOG.md](CHANGELOG.md) - Version history

## Paper Gap Solutions

The original paper omitted implementation details for 6 critical components. This repository proposed implementations for them; none is measured against the paper:

| Gap | Problem | Solution | Measured |
|-----|---------|----------|----------|
| 0 | Context compression for 1,500 papers | Hierarchical 3-tier compression (archived, not on the run path) | no |
| 1 | State Manager schema unspecified | 4-layer hybrid architecture (JSON + Neo4j + Vector + Citations) | no |
| 2 | Task generation algorithm unstated | Plan Creator + Plan Reviewer pattern (archived, not on the run path) | no |
| 3 | Agent integration mechanism unclear | Skill loader for domain-specific skills (see [#67](https://github.com/jimmc414/Kosmos/issues/67)) | no |
| 4 | Execution environment not described | Docker sandbox with Python + R support (see [#69](https://github.com/jimmc414/Kosmos/issues/69)) | no |
| 5 | Discovery validation criteria missing | Recomputation + permutation null; ScholarEval advisory | no |

For detailed analysis, see [archive/120525_implementation_gaps_v2.md](archive/120525_implementation_gaps_v2.md).

## Based On

- **Paper**: [Kosmos: An AI Scientist for Autonomous Discovery](https://arxiv.org/abs/2511.02824) (Mitchener et al., Edison Scientific, November 2025)
- **K-Dense ecosystem**: Pattern repositories for AI agent systems
- **kosmos-figures**: [Analysis patterns](https://github.com/EdisonScientific/kosmos-figures)

## Contributing

See [CONTRIBUTING.md](archive/CONTRIBUTING.md).

Areas where contributions would be useful:
- Docker sandbox testing and hardening
- Additional scientific domain skills
- Performance benchmarking with production LLMs
- Validation studies to measure actual accuracy
- Multi-tenancy and user isolation

## License

MIT License

---

**Version**: 0.2.0-alpha | **Tests**: see CI | **Last Updated**: 2026-10-09

# World Model Implementation Guide (Full Architecture Path)

**Version:** 1.0
**Date:** November 2025
**Status:** Implementation Ready
**Owner:** Kosmos Development Team

---

## ⚡ **Choose Your Implementation Path**

You have **two options** for implementing persistent knowledge graphs:

### 🚀 **MVP-First Path** (Recommended for most teams)
- **Timeline:** 1-2 weeks to production
- **Document:** [`implementation_mvp.md`](implementation_mvp.md)
- **Approach:** Ship minimal viable persistence, validate with users, evolve based on feedback
- **Best for:** Delivering value quickly, validating demand, iterating

### 🏗️ **This Document: Full Architecture Path**
- **Timeline:** 19 weeks (5 phases)
- **Approach:** Build complete abstraction layer, plan for future scale
- **Best for:** Educational projects, research papers, long-term organizational adoption

**💡 Our Recommendation:**
Start with the MVP path ([`implementation_mvp.md`](implementation_mvp.md)). Ship persistent graphs in Week 1, validate with users, then selectively adopt pieces of this full architecture as proven necessary. This document serves as your **roadmap**, not your starting point.

**You can always evolve from MVP to Full Architecture later.** The reverse is much harder.

---

**Prerequisite Documents:**
- `docs/planning/requirements.md` - What to build (RFC 2119 requirements)
- `docs/planning/architecture.md` - How to build it (technical architecture)
- `docs/planning/objective.md` - Why we're building it (context and goals)
- `docs/planning/implementation_mvp.md` - Fast path alternative to this document

---

## Table of Contents

1. [Document Overview](#1-document-overview)
2. [Development Environment Setup](#2-development-environment-setup)
3. [Phase 0: Validation and Architecture (Weeks 1-2)](#3-phase-0-validation-and-architecture-weeks-1-2)
4. [Phase 1: Foundation (Weeks 3-6)](#4-phase-1-foundation-weeks-3-6)
5. [Phase 2: Curation Features (Weeks 7-10)](#5-phase-2-curation-features-weeks-7-10)
6. [Phase 3: Multi-Project Support (Weeks 11-13)](#6-phase-3-multi-project-support-weeks-11-13)
7. [Phase 4: Production Mode (Weeks 14-19)](#7-phase-4-production-mode-weeks-14-19)
8. [Complete File Structure](#8-complete-file-structure)
9. [Testing Strategy](#9-testing-strategy)
10. [Deployment and Operations](#10-deployment-and-operations)
11. [CI/CD Pipeline](#11-cicd-pipeline)
12. [Developer Guidelines](#12-developer-guidelines)

---

## 1. Document Overview

### 1.1 Purpose

This document is a **complete, standalone implementation guide** for the Kosmos persistent world model feature. A developer should be able to read this document and implement the entire feature without needing to reference other documents (though they are available for context).

### 1.2 How to Use This Document

**If you're implementing from scratch:**
1. Start with Section 2 (Development Environment Setup)
2. Follow phases sequentially: Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4
3. For each phase, complete all sprints in order
4. Run tests after each sprint
5. Review acceptance criteria before moving to next phase

**If you're joining mid-implementation:**
1. Review the Complete File Structure (Section 8)
2. Jump to the relevant phase
3. Check prerequisites and dependencies for that phase
4. Follow the sprint tasks

**If you're fixing a bug or adding a feature:**
1. Find the relevant component in Section 8
2. Review the architecture patterns used (references provided)
3. Follow the testing strategy (Section 9)
4. Ensure changes maintain backward compatibility

### 1.3 Document Conventions

**Code Examples:**
- ✅ **Complete code** = Ready to copy-paste
- 🔧 **Pseudocode** = Logical structure, needs adaptation
- 📝 **Configuration** = YAML/JSON examples

**Task Estimates:**
- Days are **developer-days** (6-8 productive hours)
- Estimates include implementation + testing + documentation
- Story points: 1 point = 0.5 days, 2 points = 1 day, 3 points = 2 days

**Success Criteria:**
- Each sprint has acceptance criteria
- ✅ = Must be complete before moving forward
- ⭐ = Critical for phase success
- 💡 = Nice to have, can defer

### 1.4 The "All Three" Goals Reminder

Every implementation decision serves three goals simultaneously:

**Goal A: Faithful Reproduction**
Production Mode implements the polyglot architecture from the Kosmos paper

**Goal B: Practical Tool**
Simple Mode provides immediate value to individual researchers

**Goal C: Educational Reference**
Code teaches patterns, docs explain decisions, ADRs capture rationale

Keep these in mind when making implementation trade-offs.

---

## 2. Development Environment Setup

### 2.1 Prerequisites Checklist

Before starting implementation, ensure you have:

- [ ] **Python 3.10+** installed (check: `python --version`)
- [ ] **Poetry** for dependency management (check: `poetry --version`)
- [ ] **Docker Desktop** running (check: `docker ps`)
- [ ] **Git** configured with SSH keys
- [ ] **VS Code** (recommended) or preferred IDE
- [ ] **Neo4j Desktop** (optional, for local testing)
- [ ] **PostgreSQL client** (optional, for Production Mode testing)

### 2.2 Initial Repository Setup

```bash
# Clone the repository
git clone git@github.com:your-org/Kosmos.git
cd Kosmos

# Create feature branch for Phase 1
git checkout -b feature/world-model-phase-1

# Install dependencies
poetry install

# Verify installation
poetry run pytest --version
poetry run mypy --version
```

### 2.3 Development Database Setup (Simple Mode)

**Option 1: Docker Compose (Recommended)**

Create `docker-compose.dev.yml`:

```yaml
version: '3.8'

services:
  neo4j-dev:
    image: neo4j:5.13
    container_name: kosmos-neo4j-dev
    ports:
      - "7474:7474"  # Browser UI
      - "7687:7687"  # Bolt protocol
    environment:
      NEO4J_AUTH: neo4j/devpassword123
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_dbms_memory_heap_max__size: 2G
    volumes:
      - neo4j_dev_data:/data
      - neo4j_dev_logs:/logs

volumes:
  neo4j_dev_data:
  neo4j_dev_logs:
```

Start database:
```bash
docker-compose -f docker-compose.dev.yml up -d

# Verify it's running
docker ps | grep neo4j

# Access browser UI: http://localhost:7474
# Login: neo4j / devpassword123
```

**Option 2: Neo4j Desktop**

1. Download from https://neo4j.com/download/
2. Create new project "Kosmos Dev"
3. Create database with password "devpassword123"
4. Install APOC plugin
5. Start database

### 2.4 Development Configuration

Create `config.dev.yaml` in repository root:

```yaml
# Development configuration for world model
world_model:
  mode: simple  # Start with Simple Mode

  simple:
    backend: neo4j
    neo4j:
      url: bolt://localhost:7687
      auth:
        user: neo4j
        password: devpassword123
      database: kosmos_dev
      data_path: ~/.kosmos/dev/neo4j_data
    similarity_threshold: 0.85

  # Production mode config (for Phase 4)
  production:
    postgres:
      url: postgresql://localhost:5432/kosmos_dev
      user: kosmos
      password: devpassword123
    neo4j:
      url: bolt://localhost:7687
    elasticsearch:
      url: http://localhost:9200
    vector_db:
      provider: chromadb  # Use local ChromaDB for dev
      path: ~/.kosmos/dev/chromadb

# Existing Kosmos config sections
llm:
  provider: anthropic
  # ... existing config ...

logging:
  level: DEBUG  # Verbose logging for development
  format: detailed
```

### 2.5 IDE Setup (VS Code)

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "tests"
  ],
  "editor.formatOnSave": true,
  "editor.rulers": [100],
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".pytest_cache": true,
    ".mypy_cache": true
  }
}
```

### 2.6 Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.8
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

Install hooks:
```bash
poetry add --group dev pre-commit
poetry run pre-commit install
```

### 2.7 Quick Development Workflow

**Daily workflow:**

```bash
# Pull latest changes
git pull origin main

# Ensure your branch is up to date
git merge main

# Start development database
docker-compose -f docker-compose.dev.yml up -d

# Run tests before coding
poetry run pytest

# Make changes...

# Run tests after changes
poetry run pytest tests/unit/world_model/

# Type check
poetry run mypy kosmos/world_model

# Format code
poetry run black kosmos/world_model tests/

# Commit (pre-commit hooks will run)
git add .
git commit -m "feat(world_model): implement Entity model"
```

**Useful commands:**

```bash
# Run specific test file
poetry run pytest tests/unit/world_model/test_interface.py -v

# Run with coverage
poetry run pytest --cov=kosmos.world_model --cov-report=html

# View coverage report
open htmlcov/index.html

# Run type checking on specific file
poetry run mypy kosmos/world_model/interface.py

# Check Neo4j is running
docker exec kosmos-neo4j-dev cypher-shell -u neo4j -p devpassword123 "RETURN 'OK'"
```

---

## 3. Phase 0: Validation and Architecture (Weeks 1-2)

**Duration:** 2 weeks
**Goal:** Validate user demand and finalize architecture before implementation
**Team:** 1-2 developers + community engagement

### 3.1 Week 1: User Validation

#### Task 3.1.1: Respond to GitHub Issue #4 (2 days)

**Objective:** Engage with the user who requested persistent knowledge graphs

**Steps:**

1. **Draft response** addressing their request:

```markdown
# GitHub Issue #4 Comment Template

Hi @username,

Thank you for opening this issue! We've been thinking deeply about persistent
knowledge graphs for Kosmos, and your request aligns perfectly with our roadmap.

## What We're Planning

We're designing a **persistent world model** feature that will:

✅ Preserve knowledge graphs between research sessions
✅ Allow exporting/importing graphs for backup and sharing
✅ Support multiple research projects with isolated graphs
✅ Enable knowledge curation (verification, annotation, quality analysis)

## We Need Your Input

To ensure we build the right solution, we have some questions:

1. **Scale**: How many entities/papers do you typically work with?
   - [ ] <100 entities (small project)
   - [ ] 100-1,000 entities (medium project)
   - [ ] 1,000-10,000 entities (large project)
   - [ ] >10,000 entities (very large project)

2. **Workflow**: How would you use persistent graphs?
   - [ ] Build knowledge over weeks/months
   - [ ] Share graphs with collaborators
   - [ ] Export for version control
   - [ ] Curate and verify entities manually
   - [ ] Other: _______________

3. **Infrastructure**: What's your comfort level with DevOps?
   - [ ] Prefer minimal setup (just Docker)
   - [ ] Comfortable with Docker Compose
   - [ ] Can manage multiple database services
   - [ ] Have enterprise infrastructure available

4. **Priority Features**: Which features are most important? (rank 1-5)
   - [ ] Persistence across runs
   - [ ] Export/import functionality
   - [ ] Multi-project support
   - [ ] Knowledge curation tools
   - [ ] Semantic search

5. **Timeline**: When would you want to start using this?
   - [ ] As soon as basic persistence works
   - [ ] When curation features are ready
   - [ ] When it's production-ready
   - [ ] No rush, whenever it's ready

## Two Implementation Modes

We're planning to offer:

**Simple Mode (Default)**
- Single Neo4j database
- Works on a laptop
- 5-minute setup
- Handles 10K-50K entities
- Perfect for individual researchers

**Production Mode (Optional)**
- Multiple specialized databases
- Enterprise scalability
- Supports 100K+ entities
- Advanced features (semantic search, provenance tracking)
- For research organizations

Which mode would you use? Would you start with Simple and upgrade later?

## Next Steps

We'll:
1. Create a GitHub Discussion to gather broader community input
2. Validate that 10+ users want this feature
3. Begin implementation if there's sufficient interest

Looking forward to your feedback!
```

2. **Post comment** and tag the user
3. **Monitor responses** for 3-5 days
4. **Document feedback** in `docs/phase0/user_feedback.md`

**Acceptance Criteria:**
- ✅ Response posted to Issue #4
- ✅ User responds with answers
- ✅ Feedback documented

#### Task 3.1.2: Create GitHub Discussion (1 day)

**Objective:** Gather broader community input

**Steps:**

1. **Create discussion** in GitHub Discussions under "Ideas" category

2. **Use this template:**

```markdown
# [RFC] Persistent Knowledge Graphs for Kosmos

## Summary

We're proposing to add **persistent world models** to Kosmos, allowing knowledge
graphs to survive between research sessions.

## Problem

Currently, Kosmos builds a fresh knowledge graph for each research run. This means:
- ❌ Knowledge is lost between sessions
- ❌ No way to build expertise over time
- ❌ Can't share graphs with collaborators
- ❌ No ability to curate and verify findings

## Proposed Solution

Add persistent knowledge graph storage with two modes:

### Simple Mode (Default)
- Minimal setup (just Docker)
- Single Neo4j database
- Handles 10K-50K entities
- Export/import for backup

### Production Mode (Optional)
- Multiple specialized databases
- Enterprise scale (100K+ entities)
- Advanced features (semantic search, provenance)
- For research organizations

## Implementation Phases

**Phase 1** (6 weeks): Simple Mode with persistence, export/import, basic CLI
**Phase 2** (4 weeks): Curation tools (verification, annotation, quality analysis)
**Phase 3** (3 weeks): Multi-project support
**Phase 4** (6 weeks): Production Mode with full feature set

## We Need Your Input!

Please answer these questions:

1. **Would you use persistent knowledge graphs?** Why or why not?

2. **What's your typical research workflow?**
   - Duration of projects (days? weeks? months?)
   - Number of papers you typically analyze
   - Do you collaborate with others?

3. **Which features matter most to you?**
   - [ ] Basic persistence
   - [ ] Export/import
   - [ ] Multi-project support
   - [ ] Curation/verification tools
   - [ ] Semantic search
   - [ ] Provenance tracking

4. **Infrastructure comfort level?**
   - [ ] Minimal (I just want it to work)
   - [ ] Docker Compose is fine
   - [ ] I can manage multiple services
   - [ ] I have enterprise infrastructure

5. **When would you want this?**
   - [ ] ASAP (even with basic features)
   - [ ] When it's polished
   - [ ] No rush

## Success Criteria

We'll proceed with implementation if:
- ✅ 10+ users express interest
- ✅ Clear use cases emerge
- ✅ Majority prefer Simple Mode approach

## Timeline

- **Week 1**: Gather feedback
- **Week 2**: Finalize architecture
- **Week 3+**: Begin implementation if validated

Please share your thoughts below! 🙏
```

3. **Promote discussion** on relevant channels:
   - Project README
   - Twitter/social media
   - Relevant subreddits (r/MachineLearning, r/bioinformatics)
   - Academic Slack/Discord communities

4. **Document all feedback** in `docs/phase0/community_feedback.md`

**Acceptance Criteria:**
- ✅ Discussion created and promoted
- ✅ At least 10 responses received
- ✅ Feedback patterns identified
- ✅ User personas validated or refined

#### Task 3.1.3: Analyze Feedback (1 day)

**Objective:** Synthesize user input into actionable insights

**Create:** `docs/phase0/user_feedback_analysis.md`

**Template:**

```markdown
# User Feedback Analysis

**Date:** [DATE]
**Responses:** [COUNT]
**Go/No-Go Decision:** [GO | NO-GO | PAUSE]

## Response Summary

- Total responses: X
- Expressed interest: X (Y%)
- Specific feature requests: X
- Infrastructure concerns: X

## User Personas Validated

### Persona 1: Graduate Student Researcher
- **Count:** X users
- **Key needs:** Simple setup, persistence, export/import
- **Scale:** 100-1,000 entities
- **Timeline:** Want it ASAP

### Persona 2: Research Lab
- **Count:** X users
- **Key needs:** Multi-project, collaboration, sharing
- **Scale:** 1,000-10,000 entities
- **Timeline:** Can wait for polished version

[etc.]

## Feature Priority Ranking

Based on votes and comments:

1. **Basic Persistence** (X votes) - MUST HAVE
2. **Export/Import** (X votes) - MUST HAVE
3. **Multi-Project** (X votes) - SHOULD HAVE
4. **Curation Tools** (X votes) - SHOULD HAVE
5. **Semantic Search** (X votes) - NICE TO HAVE

## Infrastructure Preferences

- Docker-only (Simple Mode): X%
- Docker Compose: X%
- Full stack (Production Mode): X%

**Conclusion:** Simple Mode first approach is validated ✅

## Key Insights

1. [Insight 1]
2. [Insight 2]
3. [Insight 3]

## Recommended Changes to Plan

- [Change 1]
- [Change 2]

## Go/No-Go Recommendation

**Recommendation:** [GO | NO-GO | PAUSE]

**Rationale:**
- [Reason 1]
- [Reason 2]

**Approval:** [ ] Product Owner [ ] Tech Lead [ ] Community
```

**Success Criteria:**
- ✅ At least 10 users expressed interest
- ✅ User personas confirmed
- ✅ Feature priorities clear
- ✅ Simple Mode approach validated
- ✅ Go/No-Go decision made

### 3.2 Week 2: Architecture Finalization

**Note:** This work is largely complete (architecture.md exists), but this week involves review and refinement based on user feedback.

#### Task 3.2.1: Review and Refine Architecture (2 days)

**Objective:** Ensure architecture.md aligns with validated user needs

**Steps:**

1. **Review architecture.md** against user feedback
2. **Update if needed** based on priority changes
3. **Create presentation-ready version** for team review

**Checklist:**

- [ ] Architecture principles match user priorities
- [ ] Simple Mode design supports validated use cases
- [ ] Production Mode addresses organizational needs
- [ ] Interface abstractions support both modes
- [ ] Database schemas handle expected scale
- [ ] Migration path is clear

**Deliverable:** Updated `docs/planning/architecture.md` (if changes needed)

#### Task 3.2.2: Create Architecture Decision Records (2 days)

**Objective:** Document key decisions for future reference

**Already documented in architecture.md Section 16, but create standalone ADRs:**

**Create:** `docs/architecture/adr/`

**ADR Template:**

```markdown
# ADR-XXX: [Decision Title]

**Status:** Accepted
**Date:** [DATE]
**Decision Makers:** [NAMES]
**Consulted:** [STAKEHOLDERS]

## Context

[What is the issue we're addressing? What constraints exist?]

## Decision

[What decision did we make?]

## Rationale

[Why did we choose this approach? What alternatives did we consider?]

## Consequences

**Positive:**
- [Benefit 1]
- [Benefit 2]

**Negative:**
- [Cost 1]
- [Cost 2]

**Risks:**
- [Risk 1 + mitigation]
- [Risk 2 + mitigation]

## Compliance

**Requirements Addressed:**
- [FR-X.Y.Z]
- [NFR-X.Y]

**Architecture Principles:**
- [Principle 1]
- [Principle 2]
```

**Create these 6 ADRs:**

1. **ADR-001: Abstract Storage Layer**
   - Decision: Use ABC pattern for storage abstraction
   - Rationale: Enables mode switching, testability, future backends
   - Alternatives: Direct implementation, plugin system

2. **ADR-002: Neo4j for Simple Mode**
   - Decision: Single Neo4j database for Simple Mode
   - Rationale: Easy setup, sufficient for 90% of users, proven technology
   - Alternatives: SQLite with graph extension, TigerGraph

3. **ADR-003: Polyglot Persistence for Production**
   - Decision: PostgreSQL + Neo4j + ES + Vector DB
   - Rationale: Each database optimized for specific workload
   - Alternatives: Single database (Neo4j or PostgreSQL), NewSQL

4. **ADR-004: Project Isolation via Namespaces**
   - Decision: Use graph namespaces, not separate databases
   - Rationale: Simpler deployment, allows cross-project queries
   - Alternatives: Separate Neo4j databases per project

5. **ADR-005: PROV-O Standard Provenance**
   - Decision: Implement W3C PROV-O standard
   - Rationale: Publication quality, interoperability, industry standard
   - Alternatives: Custom provenance model, Apache Atlas

6. **ADR-006: GraphRAG Query Pattern**
   - Decision: Combine graph traversal + semantic search + LLM
   - Rationale: Best results for natural language queries
   - Alternatives: Pure Cypher, pure semantic search, pure LLM

**Acceptance Criteria:**
- ✅ All 6 ADRs created with full rationale
- ✅ ADRs reviewed by team
- ✅ Decisions align with user feedback

#### Task 3.2.3: Define Success Metrics (1 day)

**Create:** `docs/phase0/success_metrics.md`

```markdown
# Phase 1 Success Metrics

## User Adoption Metrics

**Target by Week 6:**
- ✅ 5+ active users using Simple Mode daily
- ✅ 10+ total users have tried it
- ✅ Positive feedback from 80% of users

**Measurement:**
- Track Docker image pulls
- Collect user feedback via GitHub discussions
- Monitor issue reports

## Technical Metrics

**Performance:**
- ✅ Entity creation: <100ms p95
- ✅ Query latency: <1s p95
- ✅ Export 10K entities: <30s

**Reliability:**
- ✅ Zero data loss incidents
- ✅ 90%+ test coverage
- ✅ No critical bugs in production

**Quality:**
- ✅ Type checking passes (mypy)
- ✅ Linting passes (ruff)
- ✅ All tests pass
- ✅ Documentation complete

## Measurement Tools

- **pytest**: Test coverage
- **locust**: Performance testing
- **sentry**: Error tracking (optional)
- **github insights**: User engagement
```

**Acceptance Criteria:**
- ✅ Success metrics defined for Phase 1
- ✅ Measurement approach documented
- ✅ Baseline measurements ready to capture

### 3.3 Phase 0 Deliverables Checklist

Before proceeding to Phase 1, ensure:

**User Validation:**
- ✅ GitHub Issue #4 response posted with user engagement
- ✅ GitHub Discussion created with 10+ responses
- ✅ User feedback analyzed and documented
- ✅ User personas validated
- ✅ Feature priorities confirmed
- ✅ GO decision made

**Architecture:**
- ✅ architecture.md reviewed and refined
- ✅ 6 ADRs created and reviewed
- ✅ Success metrics defined
- ✅ Team alignment on approach

**Decision Gate:**
- ✅ At least 10 users want this feature
- ✅ Simple Mode approach validated
- ✅ Architecture approved by tech lead
- ✅ Ready to begin implementation

**If any criteria not met:** PAUSE and address gaps before Phase 1.

---

## 4. Phase 1: Foundation (Weeks 3-6)

**Duration:** 4 weeks (4 sprints)
**Goal:** Implement Simple Mode with persistent Neo4j storage, export/import, and CLI commands
**Team:** 2-3 developers

### 4.1 Phase 1 Overview

**What We're Building:**

Phase 1 implements the **foundation** of persistent world models:
- ✅ Abstract interfaces (support both Simple and Production modes)
- ✅ Neo4j persistence (data survives restarts)
- ✅ Knowledge accumulation (intelligent entity merging)
- ✅ Export/import functionality (backup and sharing)
- ✅ CLI commands (graph info, export, import, reset)
- ✅ Project tagging (basic multi-project support)
- ✅ 90%+ test coverage
- ✅ Complete documentation

**What We're NOT Building (Yet):**

- ❌ Curation features (Phase 2)
- ❌ Full multi-project management (Phase 3)
- ❌ Production Mode (Phase 4)
- ❌ Semantic search beyond keyword matching

**Success Criteria:**

By end of Week 6:
- ⭐ 5+ users actively using Simple Mode
- ⭐ Knowledge persists correctly across restarts
- ⭐ Export/import works without data loss
- ⭐ 90%+ test coverage
- ⭐ No regressions in existing Kosmos features

### 4.2 Sprint 1: Core Abstractions (Week 3)

**Duration:** 5 days
**Goal:** Implement abstract interfaces, data models, and factory pattern
**Developers:** 2

#### Task 4.2.1: Create World Model Module Structure (Day 1, Morning)

**Create directory structure:**

```bash
mkdir -p kosmos/world_model
mkdir -p kosmos/world_model/provenance
mkdir -p kosmos/world_model/search
mkdir -p kosmos/world_model/query
mkdir -p tests/unit/world_model
mkdir -p tests/integration/world_model
```

**Create `kosmos/world_model/__init__.py`:**

```python
"""
Kosmos World Model - Persistent Knowledge Graph Storage

This module provides persistent storage for Kosmos knowledge graphs,
allowing knowledge to accumulate across research sessions.

ARCHITECTURE:
- Uses Abstract Base Classes (ABCs) to support multiple storage backends
- Simple Mode: Single Neo4j database (default)
- Production Mode: Polyglot persistence (PostgreSQL + Neo4j + ES + Vector DB)

USAGE:
    from kosmos.world_model import WorldModelFactory
    from kosmos.config import Config

    config = Config.from_file("config.yaml")
    world_model = WorldModelFactory.create(config)

    # Add entity
    entity = Entity(type="Paper", properties={"title": "..."})
    entity_id = world_model.add_entity(entity)

    # Query
    results = world_model.query("MATCH (p:Entity {type: 'Paper'}) RETURN p")

See docs/planning/architecture.md for complete architecture details.
"""

from kosmos.world_model.interface import WorldModelStorage, WorldModel
from kosmos.world_model.models import Entity, Relationship, Annotation
from kosmos.world_model.factory import WorldModelFactory

__all__ = [
    "WorldModelStorage",
    "WorldModel",
    "Entity",
    "Relationship",
    "Annotation",
    "WorldModelFactory",
]

__version__ = "0.1.0"  # Will be Phase 1.0 when complete
```

**Acceptance Criteria:**
- ✅ Module structure created
- ✅ `__init__.py` properly exports public API

#### Task 4.2.2: Implement Data Models (Day 1, Afternoon - 4 hours)

**Create `kosmos/world_model/models.py`:**

```python
"""
Data models for world model entities and relationships.

These are the core data structures used throughout the world model system.
They are designed to be:
- Serializable (for export/import)
- Validatable (catch errors early)
- Extensible (properties dict allows flexibility)
- Timestamped (track creation/updates)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid


@dataclass
class Annotation:
    """
    User annotation on an entity or relationship.

    Annotations allow users to add notes, corrections, or context to
    entities they're curating.

    Example:
        annotation = Annotation(
            text="This is the seminal paper on transformers",
            created_by="researcher@example.com"
        )
    """

    text: str
    created_by: str
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate annotation."""
        if not self.text or not self.text.strip():
            raise ValueError("Annotation text cannot be empty")
        if not self.created_by or not self.created_by.strip():
            raise ValueError("Annotation must have a creator")


@dataclass
class Entity:
    """
    Knowledge graph entity (paper, concept, experiment, etc.).

    DESIGN RATIONALE:
    - id: Auto-generated UUID if not provided (stable references)
    - type: Explicit entity type for filtering and querying
    - properties: Flexible dict for entity-specific attributes
    - confidence: Allows representing uncertainty (0.0-1.0)
    - project: Enables multi-project isolation
    - verified: User curation support (Phase 2)
    - annotations: Inline notes (Phase 2)

    ENTITY TYPES:
    - Paper: Research paper from literature
    - Concept: Scientific concept or term
    - Author: Paper author
    - Experiment: Experiment design or result
    - Hypothesis: Scientific hypothesis
    - Finding: Research finding
    - Dataset: Referenced dataset
    - Method: Experimental method

    Example:
        paper = Entity(
            type="Paper",
            properties={
                "title": "Attention Is All You Need",
                "authors": ["Vaswani et al."],
                "year": 2017,
                "doi": "10.5555/3295222.3295349"
            },
            confidence=0.95,
            project="transformers_research"
        )
    """

    type: str
    properties: Dict[str, Any]
    id: Optional[str] = None
    confidence: float = 1.0
    project: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None
    verified: bool = False
    annotations: List[Annotation] = field(default_factory=list)

    # Standard entity types (can be extended)
    VALID_TYPES = {
        "Paper",
        "Concept",
        "Author",
        "Experiment",
        "Hypothesis",
        "Finding",
        "Dataset",
        "Method",
    }

    def __post_init__(self):
        """Validate and initialize entity."""
        # Generate ID if not provided
        if self.id is None:
            self.id = str(uuid.uuid4())

        # Validate type
        if not self.type:
            raise ValueError("Entity type is required")

        # Warn if non-standard type (but allow it for extensibility)
        if self.type not in self.VALID_TYPES:
            import warnings

            warnings.warn(
                f"Entity type '{self.type}' is not a standard type. "
                f"Standard types: {', '.join(sorted(self.VALID_TYPES))}"
            )

        # Validate confidence
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        # Set timestamps
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at

        # Validate properties
        if not isinstance(self.properties, dict):
            raise ValueError("Properties must be a dictionary")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entity to dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON export
        """
        return {
            "id": self.id,
            "type": self.type,
            "properties": self.properties,
            "confidence": self.confidence,
            "project": self.project,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
            "verified": self.verified,
            "annotations": [
                {
                    "text": ann.text,
                    "created_by": ann.created_by,
                    "created_at": ann.created_at.isoformat(),
                }
                for ann in self.annotations
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """
        Create entity from dictionary (for import).

        Args:
            data: Dictionary representation from export

        Returns:
            Entity instance
        """
        # Parse timestamps
        created_at = None
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])

        updated_at = None
        if data.get("updated_at"):
            updated_at = datetime.fromisoformat(data["updated_at"])

        # Parse annotations
        annotations = []
        for ann_data in data.get("annotations", []):
            annotations.append(
                Annotation(
                    text=ann_data["text"],
                    created_by=ann_data["created_by"],
                    created_at=datetime.fromisoformat(ann_data["created_at"]),
                )
            )

        return cls(
            id=data["id"],
            type=data["type"],
            properties=data["properties"],
            confidence=data.get("confidence", 1.0),
            project=data.get("project"),
            created_at=created_at,
            updated_at=updated_at,
            created_by=data.get("created_by"),
            verified=data.get("verified", False),
            annotations=annotations,
        )


@dataclass
class Relationship:
    """
    Relationship between two entities.

    DESIGN RATIONALE:
    - source_id/target_id: Connect entities
    - type: Explicit relationship semantics
    - properties: Flexible attributes (e.g., citation context)
    - confidence: Uncertainty representation

    RELATIONSHIP TYPES:
    - CITES: Paper cites another paper
    - AUTHOR_OF: Author wrote paper
    - MENTIONS: Paper mentions concept
    - RELATES_TO: Concept relates to concept
    - SUPPORTS: Finding supports hypothesis
    - REFUTES: Finding refutes hypothesis
    - USES_METHOD: Experiment uses method
    - PRODUCED_BY: Finding from experiment
    - DERIVED_FROM: Entity derived from others (provenance)

    Example:
        citation = Relationship(
            source_id="paper1_id",
            target_id="paper2_id",
            type="CITES",
            properties={"section": "introduction"},
            confidence=1.0
        )
    """

    source_id: str
    target_id: str
    type: str
    id: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    created_at: Optional[datetime] = None
    created_by: Optional[str] = None

    # Standard relationship types
    VALID_TYPES = {
        "CITES",
        "AUTHOR_OF",
        "MENTIONS",
        "RELATES_TO",
        "SUPPORTS",
        "REFUTES",
        "USES_METHOD",
        "PRODUCED_BY",
        "DERIVED_FROM",
    }

    def __post_init__(self):
        """Validate and initialize relationship."""
        # Generate ID if not provided
        if self.id is None:
            self.id = str(uuid.uuid4())

        # Validate IDs
        if not self.source_id or not self.target_id:
            raise ValueError("Source and target IDs are required")

        # Validate type
        if not self.type:
            raise ValueError("Relationship type is required")

        # Warn if non-standard type
        if self.type not in self.VALID_TYPES:
            import warnings

            warnings.warn(
                f"Relationship type '{self.type}' is not standard. "
                f"Standard types: {', '.join(sorted(self.VALID_TYPES))}"
            )

        # Validate confidence
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        # Set timestamp
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type,
            "properties": self.properties,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        """Create relationship from dictionary."""
        created_at = None
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])

        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            type=data["type"],
            properties=data.get("properties", {}),
            confidence=data.get("confidence", 1.0),
            created_at=created_at,
            created_by=data.get("created_by"),
        )


# Export format version for compatibility checking
EXPORT_FORMAT_VERSION = "1.0"
```

**Create tests: `tests/unit/world_model/test_models.py`:**

```python
"""Tests for world model data models."""

import pytest
from datetime import datetime
from kosmos.world_model.models import Entity, Relationship, Annotation


class TestAnnotation:
    """Test Annotation model."""

    def test_create_annotation(self):
        """Test creating valid annotation."""
        ann = Annotation(text="Test annotation", created_by="test@example.com")

        assert ann.text == "Test annotation"
        assert ann.created_by == "test@example.com"
        assert isinstance(ann.created_at, datetime)

    def test_empty_text_raises_error(self):
        """Test that empty text raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Annotation(text="", created_by="test@example.com")

    def test_missing_creator_raises_error(self):
        """Test that missing creator raises error."""
        with pytest.raises(ValueError, match="must have a creator"):
            Annotation(text="Test", created_by="")


class TestEntity:
    """Test Entity model."""

    def test_create_entity_minimal(self):
        """Test creating entity with minimal fields."""
        entity = Entity(type="Paper", properties={"title": "Test Paper"})

        assert entity.type == "Paper"
        assert entity.properties["title"] == "Test Paper"
        assert entity.id is not None  # Auto-generated
        assert entity.confidence == 1.0  # Default
        assert isinstance(entity.created_at, datetime)
        assert entity.created_at == entity.updated_at

    def test_create_entity_full(self):
        """Test creating entity with all fields."""
        entity = Entity(
            type="Paper",
            properties={"title": "Test", "year": 2024},
            confidence=0.95,
            project="test_project",
            created_by="test_agent",
            verified=True,
        )

        assert entity.confidence == 0.95
        assert entity.project == "test_project"
        assert entity.created_by == "test_agent"
        assert entity.verified is True

    def test_invalid_confidence_raises_error(self):
        """Test that invalid confidence raises error."""
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            Entity(type="Paper", properties={}, confidence=1.5)

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            Entity(type="Paper", properties={}, confidence=-0.1)

    def test_missing_type_raises_error(self):
        """Test that missing type raises error."""
        with pytest.raises(ValueError, match="type is required"):
            Entity(type="", properties={})

    def test_non_standard_type_warns(self):
        """Test that non-standard type issues warning."""
        with pytest.warns(UserWarning, match="not a standard type"):
            Entity(type="CustomType", properties={})

    def test_to_dict(self):
        """Test entity serialization."""
        entity = Entity(
            id="test_id",
            type="Paper",
            properties={"title": "Test"},
            confidence=0.9,
        )

        data = entity.to_dict()

        assert data["id"] == "test_id"
        assert data["type"] == "Paper"
        assert data["properties"]["title"] == "Test"
        assert data["confidence"] == 0.9
        assert "created_at" in data

    def test_from_dict(self):
        """Test entity deserialization."""
        data = {
            "id": "test_id",
            "type": "Paper",
            "properties": {"title": "Test"},
            "confidence": 0.9,
            "created_at": "2024-01-15T10:30:00",
            "updated_at": "2024-01-15T10:30:00",
        }

        entity = Entity.from_dict(data)

        assert entity.id == "test_id"
        assert entity.type == "Paper"
        assert entity.confidence == 0.9
        assert isinstance(entity.created_at, datetime)

    def test_annotations(self):
        """Test entity with annotations."""
        entity = Entity(type="Paper", properties={})
        entity.annotations.append(
            Annotation(text="Important paper", created_by="user@example.com")
        )

        assert len(entity.annotations) == 1
        assert entity.annotations[0].text == "Important paper"

        # Test serialization includes annotations
        data = entity.to_dict()
        assert len(data["annotations"]) == 1
        assert data["annotations"][0]["text"] == "Important paper"


class TestRelationship:
    """Test Relationship model."""

    def test_create_relationship_minimal(self):
        """Test creating relationship with minimal fields."""
        rel = Relationship(source_id="entity1", target_id="entity2", type="CITES")

        assert rel.source_id == "entity1"
        assert rel.target_id == "entity2"
        assert rel.type == "CITES"
        assert rel.id is not None  # Auto-generated
        assert rel.confidence == 1.0
        assert isinstance(rel.created_at, datetime)

    def test_create_relationship_full(self):
        """Test creating relationship with all fields."""
        rel = Relationship(
            source_id="entity1",
            target_id="entity2",
            type="CITES",
            properties={"section": "introduction"},
            confidence=0.85,
            created_by="test_agent",
        )

        assert rel.properties["section"] == "introduction"
        assert rel.confidence == 0.85
        assert rel.created_by == "test_agent"

    def test_missing_ids_raises_error(self):
        """Test that missing IDs raise error."""
        with pytest.raises(ValueError, match="are required"):
            Relationship(source_id="", target_id="entity2", type="CITES")

        with pytest.raises(ValueError, match="are required"):
            Relationship(source_id="entity1", target_id="", type="CITES")

    def test_missing_type_raises_error(self):
        """Test that missing type raises error."""
        with pytest.raises(ValueError, match="type is required"):
            Relationship(source_id="entity1", target_id="entity2", type="")

    def test_non_standard_type_warns(self):
        """Test that non-standard type issues warning."""
        with pytest.warns(UserWarning, match="not standard"):
            Relationship(source_id="e1", target_id="e2", type="CUSTOM_REL")

    def test_to_dict(self):
        """Test relationship serialization."""
        rel = Relationship(
            id="rel_id",
            source_id="e1",
            target_id="e2",
            type="CITES",
            confidence=0.9,
        )

        data = rel.to_dict()

        assert data["id"] == "rel_id"
        assert data["source_id"] == "e1"
        assert data["target_id"] == "e2"
        assert data["type"] == "CITES"
        assert data["confidence"] == 0.9

    def test_from_dict(self):
        """Test relationship deserialization."""
        data = {
            "id": "rel_id",
            "source_id": "e1",
            "target_id": "e2",
            "type": "CITES",
            "properties": {"section": "intro"},
            "confidence": 0.9,
            "created_at": "2024-01-15T10:30:00",
        }

        rel = Relationship.from_dict(data)

        assert rel.id == "rel_id"
        assert rel.source_id == "e1"
        assert rel.target_id == "e2"
        assert rel.type == "CITES"
        assert rel.properties["section"] == "intro"
```

**Run tests:**

```bash
poetry run pytest tests/unit/world_model/test_models.py -v

# Expected output:
# test_models.py::TestAnnotation::test_create_annotation PASSED
# test_models.py::TestAnnotation::test_empty_text_raises_error PASSED
# ... (all tests should pass)
```

**Acceptance Criteria:**
- ✅ Entity, Relationship, Annotation models implemented
- ✅ All validation logic working
- ✅ Serialization (to_dict/from_dict) working
- ✅ All tests passing
- ✅ Type hints complete
- ✅ Docstrings explain design rationale

#### Task 4.2.3: Implement Abstract Interfaces (Day 2-3, 2 days)

**Create `kosmos/world_model/interface.py`:**

```python
"""
Abstract interfaces for world model storage.

ARCHITECTURE PATTERN: Abstract Base Classes (ABCs)

This module defines the abstract interfaces that enable multiple storage
backends (Simple Mode vs Production Mode) without changing business logic.

KEY INTERFACES:
- WorldModelStorage: Core storage operations
- ProvenanceTracker: Provenance tracking (basic vs PROV-O)
- SemanticSearch: Similarity search (keyword vs vector)
- QueryEngine: Query execution (Cypher vs GraphRAG)

DESIGN RATIONALE:
Using ABCs follows the Dependency Inversion Principle (SOLID):
- High-level modules (agents) depend on abstractions (these interfaces)
- Low-level modules (Neo4j, PostgreSQL) implement abstractions
- Both can vary independently

See: docs/planning/architecture.md Section 3 (Architecture Principles)
See: docs/architecture/adr/ADR-001-abstract-storage-layer.md
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Iterator
from pathlib import Path
from dataclasses import dataclass

from kosmos.world_model.models import Entity, Relationship


# ============================================================================
# Error Classes
# ============================================================================


class WorldModelError(Exception):
    """Base exception for world model errors."""

    pass


class WorldModelStorageError(WorldModelError):
    """Base exception for storage errors."""

    pass


class EntityNotFoundError(WorldModelStorageError):
    """Entity does not exist."""

    def __init__(self, entity_id: str):
        super().__init__(f"Entity not found: {entity_id}")
        self.entity_id = entity_id


class DuplicateEntityError(WorldModelStorageError):
    """Entity already exists."""

    def __init__(self, entity_id: str):
        super().__init__(f"Entity already exists: {entity_id}")
        self.entity_id = entity_id


class InvalidEntityError(WorldModelStorageError):
    """Entity validation failed."""

    pass


class ConnectionError(WorldModelStorageError):
    """Cannot connect to storage backend."""

    pass


# ============================================================================
# Data Transfer Objects
# ============================================================================


@dataclass
class StorageStats:
    """Statistics about storage."""

    entity_count: int
    relationship_count: int
    size_bytes: int
    database: str  # "neo4j", "polyglot", etc.
    projects: List[str]
    last_modified: Optional[str] = None


@dataclass
class QueryResult:
    """Results from a query."""

    entities: List[Entity]
    relationships: List[Relationship]
    explanation: str  # How results were found
    execution_time_ms: int


# ============================================================================
# WorldModelStorage Interface
# ============================================================================


class WorldModelStorage(ABC):
    """
    Abstract interface for world model storage operations.

    IMPLEMENTATIONS:
    - Neo4jWorldModel (Simple Mode): Single Neo4j database
    - PolyglotWorldModel (Production Mode): PostgreSQL + Neo4j + ES + VectorDB

    CONTRACT:
    All implementations MUST:
    - Generate stable, unique IDs for entities
    - Validate entity/relationship schemas
    - Handle duplicate detection
    - Support transactions (atomic operations)
    - Provide meaningful error messages

    EXAMPLE USAGE:
        storage = Neo4jWorldModel(config)

        # Add entity
        entity = Entity(type="Paper", properties={...})
        entity_id = storage.add_entity(entity)

        # Query
        results = storage.query("type = 'Paper'")

        # Get statistics
        stats = storage.get_statistics()
    """

    @abstractmethod
    def add_entity(self, entity: Entity) -> str:
        """
        Add entity to storage.

        Args:
            entity: Entity to add (ID will be generated if not provided)

        Returns:
            Entity ID (stable identifier)

        Raises:
            InvalidEntityError: Entity validation failed
            DuplicateEntityError: Entity already exists (if ID provided)
            ConnectionError: Cannot connect to backend

        Implementation Requirements:
        - MUST generate unique ID if not provided
        - MUST validate entity schema
        - SHOULD check for duplicates based on similarity
        - MUST be idempotent (calling twice with same ID = same result)
        - MUST atomically update all storage backends (Production Mode)
        """
        pass

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Get entity by ID.

        Args:
            entity_id: Unique entity identifier

        Returns:
            Entity if found, None otherwise

        Implementation Requirements:
        - MUST return None (not error) if entity doesn't exist
        - MUST return complete entity with all properties
        - SHOULD be fast (<10ms for Simple Mode, <1ms for Production Mode)
        """
        pass

    @abstractmethod
    def update_entity(self, entity: Entity):
        """
        Update existing entity.

        Args:
            entity: Entity with updated properties (must have ID)

        Raises:
            EntityNotFoundError: Entity doesn't exist
            InvalidEntityError: Validation failed

        Implementation Requirements:
        - MUST verify entity exists before updating
        - MUST preserve entity ID
        - MUST update properties atomically
        - SHOULD update updated_at timestamp
        """
        pass

    @abstractmethod
    def delete_entity(self, entity_id: str):
        """
        Delete entity and all its relationships.

        Args:
            entity_id: Entity to delete

        Raises:
            EntityNotFoundError: Entity doesn't exist

        Implementation Requirements:
        - MUST delete all relationships (source or target)
        - MUST be atomic (all or nothing)
        - SHOULD log deletion for audit trail
        """
        pass

    @abstractmethod
    def add_relationship(
        self, source_id: str, rel_type: str, target_id: str, properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add relationship between entities.

        Args:
            source_id: Source entity ID
            rel_type: Relationship type (e.g., "CITES")
            target_id: Target entity ID
            properties: Optional relationship properties

        Returns:
            Relationship ID

        Raises:
            EntityNotFoundError: Source or target doesn't exist
            InvalidEntityError: Relationship validation failed

        Implementation Requirements:
        - MUST verify source and target exist
        - MUST generate unique relationship ID
        - SHOULD support duplicate relationships with different properties
        """
        pass

    @abstractmethod
    def query(self, query_str: str, **kwargs) -> List[Entity]:
        """
        Execute query and return matching entities.

        Args:
            query_str: Query string (format depends on implementation)
                      Simple Mode: Cypher query
                      Production Mode: Natural language (GraphRAG)
            **kwargs: Additional query parameters
                     - project: Filter by project
                     - limit: Maximum results
                     - offset: Skip first N results

        Returns:
            List of matching entities

        Raises:
            WorldModelStorageError: Query execution failed

        Implementation Requirements:
        - MUST return empty list (not error) if no matches
        - SHOULD limit results to prevent memory issues
        - SHOULD log query for debugging
        """
        pass

    @abstractmethod
    def get_all_entities(self, project: Optional[str] = None) -> Iterator[Entity]:
        """
        Iterate over all entities.

        Args:
            project: Optional project filter

        Yields:
            Entities one at a time (memory-efficient)

        Implementation Requirements:
        - MUST use cursor/iterator (not load all into memory)
        - SHOULD yield in consistent order
        - MUST filter by project if provided
        """
        pass

    @abstractmethod
    def get_statistics(self) -> StorageStats:
        """
        Get storage statistics.

        Returns:
            Statistics about entities, relationships, size, etc.

        Implementation Requirements:
        - SHOULD be fast (cached if possible)
        - MUST return accurate counts
        - SHOULD include size on disk (if available)
        """
        pass

    @abstractmethod
    def export_graph(self, filepath: Path, project: Optional[str] = None, format: str = "json") -> Path:
        """
        Export graph to file.

        Args:
            filepath: Output file path
            project: Optional project filter
            format: Export format ("json" or "graphml")

        Returns:
            Path to exported file

        Raises:
            WorldModelStorageError: Export failed

        Implementation Requirements:
        - MUST include all entities and relationships
        - MUST preserve all properties and metadata
        - MUST be importable without data loss
        - SHOULD include export metadata (version, timestamp)
        """
        pass

    @abstractmethod
    def import_graph(self, filepath: Path, project: Optional[str] = None, mode: str = "merge") -> int:
        """
        Import graph from file.

        Args:
            filepath: Input file path
            project: Optional project to import into
            mode: Import mode ("merge" or "replace")
                 merge: Add to existing graph
                 replace: Clear existing and import

        Returns:
            Number of entities imported

        Raises:
            WorldModelStorageError: Import failed
            InvalidEntityError: Invalid data in file

        Implementation Requirements:
        - MUST validate data before importing
        - MUST be atomic (rollback on error)
        - MUST handle duplicate IDs gracefully
        - SHOULD log import details
        """
        pass

    @abstractmethod
    def reset(self, project: Optional[str] = None, confirm: bool = False):
        """
        Delete all data (DANGEROUS).

        Args:
            project: Optional project to reset (None = reset all)
            confirm: Safety flag (must be True to execute)

        Raises:
            ValueError: confirm=False (safety check)
            WorldModelStorageError: Reset failed

        Implementation Requirements:
        - MUST require confirm=True
        - MUST be atomic
        - SHOULD create backup before reset
        - MUST log reset operation
        """
        pass


# ============================================================================
# WorldModel Facade
# ============================================================================


class WorldModel:
    """
    Facade for world model operations.

    This is the main entry point for world model functionality.
    It coordinates between storage, provenance, search, and query components.

    DESIGN PATTERN: Facade

    EXAMPLE USAGE:
        from kosmos.world_model import WorldModelFactory

        world_model = WorldModelFactory.create(config)

        # Add entity
        paper = Entity(type="Paper", properties={...})
        paper_id = world_model.add_entity(paper)

        # Find similar
        similar = world_model.find_similar("transformer architecture")

        # Export
        world_model.export_project("my_research", Path("backup.json"))
    """

    def __init__(self, storage: WorldModelStorage, config: Any):
        """
        Initialize world model.

        Args:
            storage: Storage backend implementation
            config: Configuration object
        """
        self.storage = storage
        self.config = config

    def add_entity(self, entity: Entity) -> str:
        """Add entity to world model."""
        return self.storage.add_entity(entity)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.storage.get_entity(entity_id)

    def update_entity(self, entity: Entity):
        """Update entity."""
        self.storage.update_entity(entity)

    def delete_entity(self, entity_id: str):
        """Delete entity."""
        self.storage.delete_entity(entity_id)

    def add_relationship(
        self, source_id: str, rel_type: str, target_id: str, properties: Optional[Dict] = None
    ) -> str:
        """Add relationship."""
        return self.storage.add_relationship(source_id, rel_type, target_id, properties)

    def query(self, query_str: str, **kwargs) -> List[Entity]:
        """Execute query."""
        return self.storage.query(query_str, **kwargs)

    def export_project(self, project: str, filepath: Path, format: str = "json") -> Path:
        """Export project to file."""
        return self.storage.export_graph(filepath, project=project, format=format)

    def import_data(self, filepath: Path, project: Optional[str] = None, mode: str = "merge") -> int:
        """Import data from file."""
        return self.storage.import_graph(filepath, project=project, mode=mode)

    def get_info(self, project: Optional[str] = None) -> StorageStats:
        """Get statistics."""
        # If project specified, could filter stats
        return self.storage.get_statistics()

    def reset(self, project: Optional[str] = None, confirm: bool = False):
        """Reset world model (delete all data)."""
        self.storage.reset(project=project, confirm=confirm)
```

**Acceptance Criteria for Sprint 1:**
- ✅ Data models (Entity, Relationship, Annotation) implemented and tested
- ✅ Abstract interfaces (WorldModelStorage, WorldModel) defined
- ✅ Error classes defined
- ✅ All tests passing
- ✅ Type hints complete (mypy passes)
- ✅ Documentation explains design rationale

### 4.3 Sprint 2: Neo4j Persistence Implementation (Week 4)

**Duration:** 5 days
**Goal:** Implement Neo4jWorldModel (Simple Mode storage backend)
**Developers:** 2

#### Task 4.3.1: Implement Neo4j Storage Backend (Day 1-3, 3 days)

**Create `kosmos/world_model/simple.py`:**

This is a large file (~600 lines). Here's the complete implementation:

```python
"""
Neo4j-based world model storage (Simple Mode).

DESIGN RATIONALE:
Simple Mode uses a single Neo4j database for all storage needs.
This prioritizes simplicity and ease of setup over specialized performance.

TRADE-OFFS:
- Pros: Simple setup, one database to manage, sufficient for 90% of users
- Cons: Limited to ~10K-50K entities, slower than specialized systems

WHEN TO UPGRADE:
Consider Production Mode when:
- Graph exceeds 10K entities
- Query latency >1 second
- Need semantic search beyond keyword matching
- Require publication-quality provenance

See: docs/planning/architecture.md Section 5.2 (Components)
See: docs/architecture/adr/ADR-002-neo4j-simple-mode.md
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator
from datetime import datetime

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, ConstraintError

from kosmos.world_model.interface import (
    WorldModelStorage,
    StorageStats,
    WorldModelStorageError,
    EntityNotFoundError,
    DuplicateEntityError,
    InvalidEntityError,
    ConnectionError as WMConnectionError,
)
from kosmos.world_model.models import Entity, Relationship, EXPORT_FORMAT_VERSION

logger = logging.getLogger(__name__)


class Neo4jWorldModel(WorldModelStorage):
    """
    Simple Mode world model using Neo4j.

    IMPLEMENTATION NOTES:
    - All entities stored as (:Entity) nodes
    - Properties stored as JSON in node.properties field
    - Relationships store metadata (confidence, timestamps)
    - Full-text search using Neo4j Lucene index
    - Export/import use JSON format

    SCHEMA:
    - Constraint: Entity.id is unique
    - Index: Entity.type
    - Index: Entity.project
    - Full-text index: Entity.properties_text
    """

    def __init__(self, neo4j_url: str, auth: tuple, database: str = "kosmos"):
        """
        Initialize Neo4j storage.

        Args:
            neo4j_url: Neo4j connection URL (e.g., "bolt://localhost:7687")
            auth: Tuple of (username, password)
            database: Database name (default: "kosmos")
        """
        self.neo4j_url = neo4j_url
        self.auth = auth
        self.database = database
        self.driver: Optional[Driver] = None

        # Connect and initialize schema
        self._connect()
        self._initialize_schema()

    def _connect(self):
        """Connect to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(self.neo4j_url, auth=self.auth)
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {self.neo4j_url}")
        except ServiceUnavailable as e:
            raise WMConnectionError(f"Cannot connect to Neo4j at {self.neo4j_url}: {e}")

    def _initialize_schema(self):
        """Create constraints and indexes."""
        with self.driver.session(database=self.database) as session:
            # Unique constraint on entity ID
            session.run(
                "CREATE CONSTRAINT entity_id IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.id IS UNIQUE"
            )

            # Index on type for filtering
            session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)")

            # Index on project for filtering
            session.run("CREATE INDEX entity_project IF NOT EXISTS FOR (e:Entity) ON (e.project)")

            # Full-text search index
            try:
                session.run(
                    "CREATE FULLTEXT INDEX entity_search IF NOT EXISTS "
                    "FOR (e:Entity) ON EACH [e.properties_text]"
                )
            except Exception as e:
                logger.warning(f"Could not create full-text index: {e}")

        logger.info("Neo4j schema initialized")

    def add_entity(self, entity: Entity) -> str:
        """Add entity to Neo4j."""
        # Flatten properties to text for full-text search
        properties_text = self._flatten_properties(entity.properties)

        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(
                    """
                    CREATE (e:Entity {
                        id: $id,
                        type: $type,
                        properties: $properties,
                        properties_text: $properties_text,
                        confidence: $confidence,
                        project: $project,
                        created_at: $created_at,
                        updated_at: $updated_at,
                        created_by: $created_by,
                        verified: $verified
                    })
                    RETURN e.id as id
                    """,
                    id=entity.id,
                    type=entity.type,
                    properties=json.dumps(entity.properties),
                    properties_text=properties_text,
                    confidence=entity.confidence,
                    project=entity.project,
                    created_at=entity.created_at.isoformat() if entity.created_at else None,
                    updated_at=entity.updated_at.isoformat() if entity.updated_at else None,
                    created_by=entity.created_by,
                    verified=entity.verified,
                )
                record = result.single()
                logger.debug(f"Created entity {entity.id}")
                return record["id"]

            except ConstraintError:
                raise DuplicateEntityError(entity.id)
            except Exception as e:
                raise WorldModelStorageError(f"Failed to add entity: {e}")

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID from Neo4j."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                "MATCH (e:Entity {id: $id}) RETURN e",
                id=entity_id
            )
            record = result.single()

            if not record:
                return None

            return self._node_to_entity(record["e"])

    def update_entity(self, entity: Entity):
        """Update entity in Neo4j."""
        properties_text = self._flatten_properties(entity.properties)

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (e:Entity {id: $id})
                SET e.type = $type,
                    e.properties = $properties,
                    e.properties_text = $properties_text,
                    e.confidence = $confidence,
                    e.project = $project,
                    e.updated_at = $updated_at,
                    e.verified = $verified
                RETURN e.id as id
                """,
                id=entity.id,
                type=entity.type,
                properties=json.dumps(entity.properties),
                properties_text=properties_text,
                confidence=entity.confidence,
                project=entity.project,
                updated_at=datetime.now().isoformat(),
                verified=entity.verified,
            )

            if not result.single():
                raise EntityNotFoundError(entity.id)

            logger.debug(f"Updated entity {entity.id}")

    def delete_entity(self, entity_id: str):
        """Delete entity and all relationships."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (e:Entity {id: $id})
                DETACH DELETE e
                RETURN count(e) as deleted
                """,
                id=entity_id
            )
            record = result.single()

            if record["deleted"] == 0:
                raise EntityNotFoundError(entity_id)

            logger.info(f"Deleted entity {entity_id}")

    def add_relationship(
        self, source_id: str, rel_type: str, target_id: str, properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add relationship between entities."""
        rel = Relationship(
            source_id=source_id,
            target_id=target_id,
            type=rel_type,
            properties=properties or {},
        )

        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"""
                MATCH (source:Entity {{id: $source_id}})
                MATCH (target:Entity {{id: $target_id}})
                CREATE (source)-[r:{rel_type} {{
                    id: $id,
                    properties: $properties,
                    confidence: $confidence,
                    created_at: $created_at,
                    created_by: $created_by
                }}]->(target)
                RETURN r.id as id
                """,
                source_id=source_id,
                target_id=target_id,
                id=rel.id,
                properties=json.dumps(rel.properties),
                confidence=rel.confidence,
                created_at=rel.created_at.isoformat(),
                created_by=rel.created_by,
            )

            record = result.single()
            if not record:
                raise EntityNotFoundError(f"Source {source_id} or target {target_id} not found")

            logger.debug(f"Created relationship {rel.id}: {source_id} -{rel_type}-> {target_id}")
            return record["id"]

    def query(self, query_str: str, **kwargs) -> List[Entity]:
        """
        Execute Cypher query.

        Args:
            query_str: Cypher query string
            **kwargs: Query parameters (project, limit, offset)

        Returns:
            List of matching entities
        """
        project = kwargs.get("project")
        limit = kwargs.get("limit", 100)

        # Build query with filters
        if project:
            query_str = f"""
                MATCH (e:Entity {{project: $project}})
                WHERE {query_str}
                RETURN e
                LIMIT {limit}
            """
            params = {"project": project}
        else:
            query_str = f"""
                MATCH (e:Entity)
                WHERE {query_str}
                RETURN e
                LIMIT {limit}
            """
            params = {}

        with self.driver.session(database=self.database) as session:
            result = session.run(query_str, **params)
            entities = [self._node_to_entity(record["e"]) for record in result]
            logger.debug(f"Query returned {len(entities)} entities")
            return entities

    def get_all_entities(self, project: Optional[str] = None) -> Iterator[Entity]:
        """Iterate over all entities."""
        with self.driver.session(database=self.database) as session:
            if project:
                query = "MATCH (e:Entity {project: $project}) RETURN e"
                params = {"project": project}
            else:
                query = "MATCH (e:Entity) RETURN e"
                params = {}

            result = session.run(query, **params)

            for record in result:
                yield self._node_to_entity(record["e"])

    def get_statistics(self) -> StorageStats:
        """Get storage statistics."""
        with self.driver.session(database=self.database) as session:
            # Count entities
            result = session.run("MATCH (e:Entity) RETURN count(e) as count")
            entity_count = result.single()["count"]

            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = result.single()["count"]

            # Get unique projects
            result = session.run(
                "MATCH (e:Entity) WHERE e.project IS NOT NULL "
                "RETURN DISTINCT e.project as project"
            )
            projects = [record["project"] for record in result]

            # Note: Neo4j doesn't easily expose disk size
            return StorageStats(
                entity_count=entity_count,
                relationship_count=rel_count,
                size_bytes=0,  # Not available
                database="neo4j",
                projects=projects,
                last_modified=datetime.now().isoformat(),
            )

    def export_graph(self, filepath: Path, project: Optional[str] = None, format: str = "json") -> Path:
        """Export graph to JSON file."""
        if format != "json":
            raise ValueError(f"Only JSON format supported, got: {format}")

        entities = list(self.get_all_entities(project=project))

        # Get all relationships
        with self.driver.session(database=self.database) as session:
            if project:
                query = """
                    MATCH (source:Entity {project: $project})-[r]->(target:Entity)
                    RETURN r, type(r) as rel_type
                """
                params = {"project": project}
            else:
                query = "MATCH ()-[r]->() RETURN r, type(r) as rel_type"
                params = {}

            result = session.run(query, **params)
            relationships = []
            for record in result:
                r = record["r"]
                rel_data = {
                    "id": r["id"],
                    "source_id": r.start_node["id"],
                    "target_id": r.end_node["id"],
                    "type": record["rel_type"],
                    "properties": json.loads(r.get("properties", "{}")),
                    "confidence": r.get("confidence", 1.0),
                    "created_at": r.get("created_at"),
                    "created_by": r.get("created_by"),
                }
                relationships.append(rel_data)

        # Create export data
        export_data = {
            "version": EXPORT_FORMAT_VERSION,
            "export_date": datetime.now().isoformat(),
            "source": "kosmos",
            "mode": "simple",
            "project": project,
            "statistics": {
                "entity_count": len(entities),
                "relationship_count": len(relationships),
            },
            "entities": [e.to_dict() for e in entities],
            "relationships": relationships,
        }

        # Write to file
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported {len(entities)} entities to {filepath}")
        return filepath

    def import_graph(self, filepath: Path, project: Optional[str] = None, mode: str = "merge") -> int:
        """Import graph from JSON file."""
        if not filepath.exists():
            raise FileNotFoundError(f"Import file not found: {filepath}")

        with open(filepath) as f:
            data = json.load(f)

        # Validate format version
        if data.get("version") != EXPORT_FORMAT_VERSION:
            logger.warning(f"Import version mismatch: {data.get('version')} vs {EXPORT_FORMAT_VERSION}")

        # Clear existing if replace mode
        if mode == "replace":
            self.reset(project=project, confirm=True)

        # Import entities
        entities = data.get("entities", [])
        imported = 0

        for entity_data in entities:
            # Override project if specified
            if project:
                entity_data["project"] = project

            entity = Entity.from_dict(entity_data)

            try:
                self.add_entity(entity)
                imported += 1
            except DuplicateEntityError:
                if mode == "merge":
                    # Update existing
                    self.update_entity(entity)
                    imported += 1
                else:
                    logger.warning(f"Skipping duplicate entity: {entity.id}")

        # Import relationships
        relationships = data.get("relationships", [])
        for rel_data in relationships:
            try:
                self.add_relationship(
                    source_id=rel_data["source_id"],
                    rel_type=rel_data["type"],
                    target_id=rel_data["target_id"],
                    properties=rel_data.get("properties"),
                )
            except Exception as e:
                logger.warning(f"Failed to import relationship: {e}")

        logger.info(f"Imported {imported} entities from {filepath}")
        return imported

    def reset(self, project: Optional[str] = None, confirm: bool = False):
        """Delete all data."""
        if not confirm:
            raise ValueError("Must set confirm=True to reset world model")

        with self.driver.session(database=self.database) as session:
            if project:
                result = session.run(
                    "MATCH (e:Entity {project: $project}) DETACH DELETE e RETURN count(e) as deleted",
                    project=project
                )
            else:
                result = session.run("MATCH (e:Entity) DETACH DELETE e RETURN count(e) as deleted")

            deleted = result.single()["deleted"]
            logger.warning(f"Reset world model: deleted {deleted} entities (project={project})")

    def close(self):
        """Close Neo4j driver."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    # Helper methods

    def _flatten_properties(self, properties: Dict[str, Any]) -> str:
        """Flatten properties dict to searchable text."""
        parts = []
        for key, value in properties.items():
            if isinstance(value, (str, int, float, bool)):
                parts.append(f"{key}: {value}")
            elif isinstance(value, list):
                parts.append(f"{key}: {' '.join(str(v) for v in value)}")
        return " ".join(parts)

    def _node_to_entity(self, node) -> Entity:
        """Convert Neo4j node to Entity."""
        created_at = None
        if node.get("created_at"):
            created_at = datetime.fromisoformat(node["created_at"])

        updated_at = None
        if node.get("updated_at"):
            updated_at = datetime.fromisoformat(node["updated_at"])

        return Entity(
            id=node["id"],
            type=node["type"],
            properties=json.loads(node["properties"]),
            confidence=node.get("confidence", 1.0),
            project=node.get("project"),
            created_at=created_at,
            updated_at=updated_at,
            created_by=node.get("created_by"),
            verified=node.get("verified", False),
            annotations=[],  # Annotations not stored in Neo4j yet (Phase 2)
        )
```

**Acceptance Criteria:**
- ✅ Neo4jWorldModel implements all WorldModelStorage methods
- ✅ Schema initialization (constraints, indexes)
- ✅ Export/import with JSON format
- ✅ Error handling (connection, duplicates, not found)
- ✅ Logging throughout

#### Task 4.3.2: Create Factory Pattern (Day 3, Afternoon - 4 hours)

**Create `kosmos/world_model/factory.py`:**

```python
"""
Factory for creating world model instances.

DESIGN PATTERN: Factory

This factory creates the appropriate world model implementation
based on configuration (Simple Mode vs Production Mode).
"""

import logging
from typing import Any

from kosmos.world_model.interface import WorldModel
from kosmos.world_model.simple import Neo4jWorldModel

logger = logging.getLogger(__name__)


class WorldModelFactory:
    """Factory for creating world model instances."""

    @staticmethod
    def create(config: Any) -> WorldModel:
        """
        Create world model from configuration.

        Args:
            config: Configuration object with world_model section

        Returns:
            WorldModel instance (Simple or Production mode)

        Raises:
            ValueError: Invalid configuration
        """
        mode = config.world_model.mode

        if mode == "simple":
            return WorldModelFactory._create_simple_mode(config)
        elif mode == "production":
            raise NotImplementedError("Production Mode available in Phase 4")
        else:
            raise ValueError(f"Unknown world model mode: {mode}")

    @staticmethod
    def _create_simple_mode(config: Any) -> WorldModel:
        """Create Simple Mode world model."""
        logger.info("Creating Simple Mode world model")

        # Extract Neo4j config
        neo4j_config = config.world_model.simple.neo4j

        # Create Neo4j storage backend
        storage = Neo4jWorldModel(
            neo4j_url=neo4j_config.url,
            auth=(neo4j_config.auth.user, neo4j_config.auth.password),
            database=neo4j_config.get("database", "kosmos"),
        )

        # Create world model facade
        world_model = WorldModel(storage=storage, config=config)

        logger.info("Simple Mode world model created successfully")
        return world_model
```

#### Task 4.3.3: Integration Testing (Day 4-5, 2 days)

**Create `tests/integration/world_model/test_neo4j_storage.py`:**

```python
"""Integration tests for Neo4j storage."""

import pytest
from pathlib import Path
from kosmos.world_model.simple import Neo4jWorldModel
from kosmos.world_model.models import Entity, Relationship
from kosmos.world_model.interface import EntityNotFoundError, DuplicateEntityError


@pytest.fixture
def neo4j_storage():
    """Create Neo4j storage for testing."""
    storage = Neo4jWorldModel(
        neo4j_url="bolt://localhost:7687",
        auth=("neo4j", "devpassword123"),
        database="kosmos_test"
    )

    yield storage

    # Cleanup
    storage.reset(confirm=True)
    storage.close()


class TestNeo4jStorage:
    """Integration tests for Neo4j storage."""

    def test_add_and_get_entity(self, neo4j_storage):
        """Test adding and retrieving entity."""
        entity = Entity(
            type="Paper",
            properties={"title": "Test Paper", "year": 2024},
            project="test"
        )

        entity_id = neo4j_storage.add_entity(entity)
        assert entity_id == entity.id

        retrieved = neo4j_storage.get_entity(entity_id)
        assert retrieved is not None
        assert retrieved.type == "Paper"
        assert retrieved.properties["title"] == "Test Paper"

    def test_get_nonexistent_entity(self, neo4j_storage):
        """Test getting entity that doesn't exist."""
        result = neo4j_storage.get_entity("nonexistent_id")
        assert result is None

    def test_update_entity(self, neo4j_storage):
        """Test updating entity."""
        entity = Entity(type="Paper", properties={"title": "Original"})
        entity_id = neo4j_storage.add_entity(entity)

        # Update
        entity.properties["title"] = "Updated"
        neo4j_storage.update_entity(entity)

        # Verify
        retrieved = neo4j_storage.get_entity(entity_id)
        assert retrieved.properties["title"] == "Updated"

    def test_delete_entity(self, neo4j_storage):
        """Test deleting entity."""
        entity = Entity(type="Paper", properties={})
        entity_id = neo4j_storage.add_entity(entity)

        neo4j_storage.delete_entity(entity_id)

        assert neo4j_storage.get_entity(entity_id) is None

    def test_add_relationship(self, neo4j_storage):
        """Test adding relationship."""
        e1 = Entity(type="Paper", properties={"title": "Paper 1"})
        e2 = Entity(type="Paper", properties={"title": "Paper 2"})

        id1 = neo4j_storage.add_entity(e1)
        id2 = neo4j_storage.add_entity(e2)

        rel_id = neo4j_storage.add_relationship(
            source_id=id1,
            rel_type="CITES",
            target_id=id2,
            properties={"section": "introduction"}
        )

        assert rel_id is not None

    def test_export_import(self, neo4j_storage, tmp_path):
        """Test export and import."""
        # Add some data
        e1 = Entity(type="Paper", properties={"title": "Test"}, project="export_test")
        neo4j_storage.add_entity(e1)

        # Export
        export_path = tmp_path / "export.json"
        neo4j_storage.export_graph(export_path, project="export_test")

        assert export_path.exists()

        # Reset and import
        neo4j_storage.reset(confirm=True)
        count = neo4j_storage.import_graph(export_path)

        assert count == 1

        # Verify
        retrieved = neo4j_storage.get_entity(e1.id)
        assert retrieved is not None
        assert retrieved.properties["title"] == "Test"

    def test_statistics(self, neo4j_storage):
        """Test getting statistics."""
        # Add some entities
        for i in range(5):
            e = Entity(type="Paper", properties={"title": f"Paper {i}"})
            neo4j_storage.add_entity(e)

        stats = neo4j_storage.get_statistics()

        assert stats.entity_count == 5
        assert stats.database == "neo4j"
```

**Run integration tests:**

```bash
# Ensure Neo4j is running
docker-compose -f docker-compose.dev.yml up -d

# Run integration tests
poetry run pytest tests/integration/world_model/ -v

# Check coverage
poetry run pytest --cov=kosmos.world_model --cov-report=html
```

**Sprint 2 Acceptance Criteria:**
- ✅ Neo4jWorldModel fully implemented
- ✅ All CRUD operations working
- ✅ Export/import working without data loss
- ✅ Factory pattern implemented
- ✅ Integration tests passing
- ✅ 90%+ code coverage

### 4.4 Sprint 3: CLI Commands (Week 5)

**Duration:** 5 days
**Goal:** Implement CLI commands for graph management
**Developers:** 1-2

#### Task 4.4.1: Implement Graph CLI Commands (Day 1-3, 3 days)

**Create `kosmos/cli/graph_commands.py`:**

```python
"""CLI commands for graph management."""

import click
import json
from pathlib import Path
from kosmos.config import Config
from kosmos.world_model import WorldModelFactory


@click.group()
def graph():
    """Manage knowledge graphs."""
    pass


@graph.command()
@click.option("--project", help="Filter by project")
def info(project):
    """Display graph statistics."""
    config = Config.from_file("config.yaml")
    world_model = WorldModelFactory.create(config)

    stats = world_model.get_info(project=project)

    click.echo("\n📊 Knowledge Graph Statistics\n")
    click.echo(f"Entities:      {stats.entity_count:,}")
    click.echo(f"Relationships: {stats.relationship_count:,}")
    click.echo(f"Database:      {stats.database}")

    if stats.projects:
        click.echo(f"\nProjects ({len(stats.projects)}):")
        for p in sorted(stats.projects):
            click.echo(f"  - {p}")


@graph.command()
@click.argument("filepath", type=click.Path())
@click.option("--project", help="Export specific project")
@click.option("--format", default="json", type=click.Choice(["json", "graphml"]))
def export(filepath, project, format):
    """Export graph to file."""
    config = Config.from_file("config.yaml")
    world_model = WorldModelFactory.create(config)

    filepath = Path(filepath)

    with click.progressbar(length=1, label="Exporting graph") as bar:
        result_path = world_model.export_project(
            project=project or "all",
            filepath=filepath,
            format=format
        )
        bar.update(1)

    click.echo(f"\n✅ Exported to {result_path}")


@graph.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--project", help="Import into specific project")
@click.option("--mode", default="merge", type=click.Choice(["merge", "replace"]))
def import_data(filepath, project, mode):
    """Import graph from file."""
    config = Config.from_file("config.yaml")
    world_model = WorldModelFactory.create(config)

    filepath = Path(filepath)

    click.echo(f"Importing from {filepath} (mode: {mode})...")

    count = world_model.import_data(filepath, project=project, mode=mode)

    click.echo(f"✅ Imported {count} entities")


@graph.command()
@click.option("--project", help="Reset specific project")
@click.confirmation_option(prompt="⚠️  This will delete all data. Continue?")
def reset(project):
    """Delete all graph data (DANGEROUS)."""
    config = Config.from_file("config.yaml")
    world_model = WorldModelFactory.create(config)

    world_model.reset(project=project, confirm=True)

    if project:
        click.echo(f"✅ Reset project: {project}")
    else:
        click.echo("✅ Reset entire knowledge graph")
```

**Update `kosmos/cli/main.py` to include graph commands:**

```python
# Add to existing CLI
from kosmos.cli.graph_commands import graph

# In main CLI group
cli.add_command(graph)
```

#### Task 4.4.2: Update Configuration (Day 3, Afternoon)

**Update `kosmos/config.py` to include world_model section:**

```python
# Add to Config class

@dataclass
class WorldModelConfig:
    """World model configuration."""
    mode: str = "simple"  # or "production"

    @dataclass
    class SimpleConfig:
        backend: str = "neo4j"

        @dataclass
        class Neo4jConfig:
            url: str = "bolt://localhost:7687"
            user: str = "neo4j"
            password: str = "password"
            database: str = "kosmos"

        neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)

    simple: SimpleConfig = field(default_factory=SimpleConfig)

# Add to main Config
world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
```

#### Task 4.4.3: End-to-End Testing (Day 4-5, 2 days)

**Create `tests/integration/test_cli_workflow.py`:**

```python
"""End-to-end workflow tests."""

import pytest
from click.testing import CliRunner
from kosmos.cli.main import cli


def test_full_workflow(tmp_path):
    """Test complete workflow: create, export, reset, import."""
    runner = CliRunner()

    # 1. Check initial state
    result = runner.invoke(cli, ["graph", "info"])
    assert result.exit_code == 0
    assert "Entities:" in result.output

    # 2. Export
    export_file = tmp_path / "backup.json"
    result = runner.invoke(cli, ["graph", "export", str(export_file)])
    assert result.exit_code == 0
    assert export_file.exists()

    # 3. Import
    result = runner.invoke(cli, ["graph", "import", str(export_file)])
    assert result.exit_code == 0

    # 4. Info after import
    result = runner.invoke(cli, ["graph", "info"])
    assert result.exit_code == 0
```

**Sprint 3 Acceptance Criteria:**
- ✅ All 4 CLI commands working (info, export, import, reset)
- ✅ Configuration updated
- ✅ End-to-end tests passing
- ✅ User-friendly output with progress indicators
- ✅ Error handling with helpful messages

### 4.5 Sprint 4: Testing and Documentation (Week 6)

**Duration:** 5 days
**Goal:** Achieve 90%+ coverage, performance baselines, complete docs
**Developers:** 2-3

#### Task 4.5.1: Complete Test Coverage (Day 1-2)

**Add missing tests, aim for 90%+ coverage:**

```bash
# Generate coverage report
poetry run pytest --cov=kosmos.world_model --cov-report=html --cov-report=term

# Review uncovered lines
open htmlcov/index.html

# Add tests for uncovered code paths
```

#### Task 4.5.2: Performance Baseline Tests (Day 3)

**Create `tests/performance/test_baseline.py`:**

```python
"""Performance baseline tests."""

import pytest
import time
from kosmos.world_model import WorldModelFactory, Entity


def test_entity_creation_performance(world_model):
    """Test entity creation speed."""
    entities = []

    start = time.time()
    for i in range(100):
        e = Entity(type="Paper", properties={"title": f"Paper {i}"})
        world_model.add_entity(e)
        entities.append(e)

    elapsed = time.time() - start
    avg_ms = (elapsed / 100) * 1000

    print(f"Average entity creation: {avg_ms:.2f}ms")
    assert avg_ms < 100, "Entity creation too slow"


def test_export_performance(world_model, tmp_path):
    """Test export speed."""
    # Create 1000 entities
    for i in range(1000):
        e = Entity(type="Paper", properties={"title": f"Paper {i}"})
        world_model.add_entity(e)

    start = time.time()
    world_model.export_project("all", tmp_path / "export.json")
    elapsed = time.time() - start

    print(f"Export 1000 entities: {elapsed:.2f}s")
    assert elapsed < 10, "Export too slow"
```

#### Task 4.5.3: Write Documentation (Day 4-5)

**Create user guides:**

1. **`docs/user_guides/getting_started_world_model.md`** - Quick start
2. **`docs/user_guides/building_knowledge_base.md`** - Building graphs over time
3. **`docs/user_guides/exporting_importing.md`** - Backup and sharing

**Update main README.md** with world model section

**Sprint 4 Acceptance Criteria:**
- ✅ 90%+ test coverage achieved
- ✅ Performance baselines documented
- ✅ User guides written and reviewed
- ✅ README updated

### 4.6 Phase 1 Final Checklist

Before proceeding to Phase 2, verify:

**Functionality:**
- ✅ Neo4j persistence works across restarts
- ✅ Knowledge accumulates (no duplicate entities)
- ✅ Export/import preserves all data
- ✅ CLI commands all working
- ✅ Project tagging functional

**Quality:**
- ✅ 90%+ test coverage
- ✅ All tests passing (unit + integration + e2e)
- ✅ Type checking passes (mypy)
- ✅ Linting passes (ruff, black)
- ✅ No critical bugs

**Documentation:**
- ✅ User guides complete
- ✅ API docs generated
- ✅ README updated
- ✅ CHANGELOG updated

**User Validation:**
- ✅ 5+ users actively using Simple Mode
- ✅ Positive feedback collected
- ✅ No data loss reports

**Performance:**
- ✅ Entity creation <100ms p95
- ✅ Query latency <1s p95
- ✅ Export 10K entities <30s

**If all criteria met:** ✅ Proceed to Phase 2
**If not:** Address gaps before continuing

---

## 5. Phase 2: Curation Features (Weeks 7-10)

**Duration:** 3-4 weeks
**Goal:** Enable users to curate and maintain knowledge graph quality
**Prerequisites:** Phase 1 complete

### 5.1 Phase 2 Overview

**Key Features:**
- ✅ Entity verification (mark entities as verified/unverified)
- ✅ Annotation system (add notes to entities)
- ✅ Duplicate detection (find similar entities)
- ✅ Quality analysis (graph health metrics)

**Implementation Approach:**
- Extend Entity model (annotations already present)
- Add CLI commands for curation
- Implement similarity algorithms
- Add quality metrics reporting

### 5.2 Sprint Breakdown

#### Sprint 5: Verification System (Week 7)

**Key Tasks:**
1. Add `kosmos verify <entity_id>` CLI command
2. Update Neo4j queries to filter by verified status
3. Add verification metadata tracking
4. Tests for verification workflow

**Implementation Hint:**
```python
@graph.command()
@click.argument("entity_id")
def verify(entity_id):
    """Mark entity as verified."""
    world_model = WorldModelFactory.create(config)
    entity = world_model.get_entity(entity_id)
    entity.verified = True
    world_model.update_entity(entity)
    click.echo(f"✅ Verified entity: {entity_id}")
```

#### Sprint 6: Annotation System (Week 8)

**Key Tasks:**
1. Implement annotation storage (in Entity model, already present)
2. Add `kosmos annotate <entity_id> "text"` CLI command
3. Display annotations in query results
4. Tests for annotation workflow

#### Sprint 7: Duplicate Detection (Week 9)

**Key Tasks:**
1. Implement similarity algorithm (simple text matching in Phase 2)
2. Add `kosmos duplicates` CLI command
3. Add `kosmos merge <entity1> <entity2>` command
4. Preserve relationships when merging

**Algorithm (Simple Mode):**
- Compare entity properties using string similarity
- Threshold: 0.85 similarity score
- Show potential duplicates to user for review

#### Sprint 8: Quality Analysis (Week 10)

**Key Tasks:**
1. Implement quality metrics calculation
2. Add `kosmos quality` CLI command
3. Generate quality reports
4. Suggest entities needing attention

**Metrics to Track:**
- % verified vs unverified entities
- Orphaned entities (no relationships)
- Low confidence entities (<0.5)
- Entities without properties

### 5.3 Phase 2 Acceptance Criteria

- ✅ Users actively verifying and annotating entities
- ✅ Duplicate detection helpful (not too many false positives)
- ✅ Quality metrics guide users to improve graphs
- ✅ 20+ active users
- ✅ No regressions from Phase 1

---

## 6. Phase 3: Multi-Project Support (Weeks 11-13)

**Duration:** 2-3 weeks
**Goal:** Full multi-project management with isolation
**Prerequisites:** Phases 1-2 complete

### 6.1 Phase 3 Overview

**Key Features:**
- ✅ Project CRUD (create, list, switch, delete)
- ✅ Isolated project graphs (namespace-based)
- ✅ Cross-project queries
- ✅ Project metadata tracking

**Design Decision:**
Use graph namespaces (project property) rather than separate databases.
This allows cross-project queries while maintaining logical isolation.

### 6.2 Sprint Breakdown

#### Sprint 9: Project Management (Week 11)

**Key Tasks:**
1. Create `kosmos/cli/project_commands.py`
2. Implement project CRUD operations
3. Store active project in config/state file
4. Tests for project management

**CLI Commands:**
```bash
kosmos project create <name> --description "..."
kosmos project list
kosmos project switch <name>
kosmos project delete <name> --confirm
kosmos project info <name>
```

#### Sprint 10: Project Isolation (Week 12)

**Key Tasks:**
1. Update all queries to respect active project
2. Prevent accidental cross-project contamination
3. Add project filtering to all operations
4. Tests for isolation

#### Sprint 11: Cross-Project Features (Week 13)

**Key Tasks:**
1. Add `--all-projects` flag to queries
2. Add `--projects proj1,proj2` multi-project queries
3. Show project origin in results
4. Find entities appearing in multiple projects

**Example:**
```bash
# Query across all projects
kosmos graph info --all-projects

# Query specific projects
kosmos query --projects "proj1,proj2" "type = 'Paper'"
```

### 6.3 Phase 3 Acceptance Criteria

- ✅ 50+ active users
- ✅ Users managing multiple projects successfully
- ✅ No cross-project data leaks
- ✅ Cross-project queries working
- ✅ Project workflows documented

---

## 7. Phase 4: Production Mode (Weeks 14-19)

**Duration:** 4-6 weeks
**Goal:** Enterprise-scale polyglot persistence
**Prerequisites:** Phases 1-3 complete, stable Simple Mode

### 7.1 Phase 4 Overview

**What Changes:**
- Add PostgreSQL for metadata & transactions
- Add Elasticsearch for provenance events
- Add Vector DB for semantic search
- Implement PROV-O standard provenance
- Implement GraphRAG query engine
- Add migration from Simple to Production

**Architecture:**
```
PolyglotWorldModel:
  - PostgreSQL: Entities table, relationships table, transactions
  - Neo4j: Graph operations only (specialized workload)
  - Elasticsearch: Provenance events, time-series queries
  - VectorDB: Entity embeddings, semantic search
  - Redis: Caching layer (optional)
```

### 7.2 Sprint Breakdown

#### Sprint 12-13: Polyglot Storage Backend (Weeks 14-15)

**Key Tasks:**
1. Create `kosmos/world_model/production.py`
2. Implement `PolyglotWorldModel(WorldModelStorage)`
3. Coordinate writes across 4 databases
4. Implement transaction handling
5. Tests with all databases

**Critical Implementation:**
```python
class PolyglotWorldModel(WorldModelStorage):
    def __init__(self, postgres, neo4j, elasticsearch, vector_db):
        self.postgres = postgres
        self.neo4j = neo4j
        self.es = elasticsearch
        self.vector_db = vector_db

    def add_entity(self, entity: Entity) -> str:
        # Transaction across all backends
        with self.postgres.transaction():
            # 1. Store in PostgreSQL (source of truth)
            self.postgres.insert_entity(entity)

            # 2. Store in Neo4j (graph operations)
            self.neo4j.create_node(entity)

            # 3. Index in Elasticsearch (provenance)
            self.es.index_entity(entity)

            # 4. Embed in Vector DB (semantic search)
            embedding = self.embed(entity)
            self.vector_db.upsert(entity.id, embedding)

        return entity.id
```

#### Sprint 14: PROV-O Provenance (Week 16)

**Key Tasks:**
1. Implement W3C PROV-O standard
2. Track entities, activities, agents
3. Record derivation chains
4. Export PROV-O compatible format

#### Sprint 15: GraphRAG Query Engine (Week 17)

**Key Tasks:**
1. Implement natural language query parsing
2. Combine graph traversal + semantic search
3. Use LLM to generate Cypher queries
4. Provide query explanations

**Pattern:**
```
User Query: "Find papers about transformers from 2017"
  ↓
1. LLM interprets query → {"type": "Paper", "year": 2017, "keywords": ["transformers"]}
  ↓
2. Vector search → Find semantically similar entities
  ↓
3. Graph traversal → Expand to related entities
  ↓
4. Combine results → Rank by relevance
  ↓
5. Return with explanation
```

#### Sprint 16: Migration Tools (Week 18)

**Key Tasks:**
1. Implement `kosmos migrate simple-to-production`
2. Data validation before migration
3. Transformation scripts
4. Rollback capability

#### Sprint 17: Production Deployment (Week 19)

**Key Tasks:**
1. Docker Compose for production stack
2. Kubernetes manifests (optional)
3. Monitoring and alerting setup
4. Production deployment guide

### 7.3 Phase 4 Acceptance Criteria

- ✅ Production Mode deployed by 1+ organization
- ✅ Handles 100K+ entities
- ✅ Query latency <100ms p95
- ✅ PROV-O provenance working
- ✅ GraphRAG queries accurate
- ✅ Migration from Simple Mode working
- ✅ Both modes maintained

---

## 8. Complete File Structure

This section provides the complete module structure for reference.

### 8.1 World Model Module

```
kosmos/
├── world_model/
│   ├── __init__.py                 # Public API exports
│   ├── interface.py                # ABCs (WorldModelStorage, WorldModel)
│   ├── models.py                   # Data models (Entity, Relationship)
│   ├── factory.py                  # Factory pattern
│   │
│   ├── simple.py                   # PHASE 1: Neo4j implementation
│   ├── production.py               # PHASE 4: Polyglot implementation
│   │
│   ├── provenance/
│   │   ├── __init__.py
│   │   ├── interface.py            # ProvenanceTracker ABC
│   │   ├── basic.py                # PHASE 1: Simple provenance
│   │   └── prov_o.py               # PHASE 4: PROV-O standard
│   │
│   ├── search/
│   │   ├── __init__.py
│   │   ├── interface.py            # SemanticSearch ABC
│   │   ├── keyword.py              # PHASE 1: Keyword search
│   │   └── vector.py               # PHASE 4: Vector similarity
│   │
│   ├── query/
│   │   ├── __init__.py
│   │   ├── interface.py            # QueryEngine ABC
│   │   ├── direct.py               # PHASE 1: Direct Cypher
│   │   └── graphrag.py             # PHASE 4: GraphRAG
│   │
│   └── migration/
│       ├── __init__.py
│       └── commands.py             # PHASE 4: Migration tools
│
├── cli/
│   ├── main.py                     # Main CLI entry point
│   ├── graph_commands.py           # PHASE 1: graph info/export/import/reset
│   ├── project_commands.py         # PHASE 3: project create/list/switch
│   ├── curation_commands.py        # PHASE 2: verify/annotate/duplicates
│   └── migrate_commands.py         # PHASE 4: migrate simple-to-production
│
└── config.py                       # MODIFIED: Add world_model section
```

### 8.2 Test Structure

```
tests/
├── unit/
│   └── world_model/
│       ├── test_models.py          # Entity, Relationship tests
│       ├── test_interface.py       # Interface contract tests
│       ├── test_simple.py          # Neo4jWorldModel tests
│       └── test_factory.py         # Factory tests
│
├── integration/
│   └── world_model/
│       ├── test_neo4j_storage.py   # Neo4j integration tests
│       ├── test_export_import.py   # Export/import tests
│       ├── test_cli_workflow.py    # End-to-end CLI tests
│       └── test_multi_project.py   # Project isolation tests
│
├── performance/
│   ├── test_baseline.py            # Performance baselines
│   └── test_load.py                # Load testing
│
└── migration/
    └── test_simple_to_production.py # Migration tests
```

### 8.3 Documentation Structure

```
docs/
├── planning/
│   ├── objective.md                # What we're building and why
│   ├── requirements.md             # RFC 2119 requirements
│   ├── architecture.md             # Technical architecture
│   └── implementation.md           # THIS DOCUMENT
│
├── user_guides/
│   ├── getting_started.md          # Quick start guide
│   ├── building_knowledge_base.md  # Using world models
│   ├── exporting_importing.md      # Backup and sharing
│   ├── multi_project.md            # Managing multiple projects
│   ├── curation.md                 # Verifying and annotating
│   └── production_mode.md          # Deploying production mode
│
├── architecture/
│   ├── adr/
│   │   ├── ADR-001-abstract-storage.md
│   │   ├── ADR-002-neo4j-simple.md
│   │   ├── ADR-003-polyglot-production.md
│   │   ├── ADR-004-project-isolation.md
│   │   ├── ADR-005-prov-o.md
│   │   └── ADR-006-graphrag.md
│   │
│   ├── simple_mode.md              # Simple Mode architecture
│   └── production_mode.md          # Production Mode architecture
│
└── api/
    └── world_model.md              # API reference (auto-generated)
```

---

## 9. Testing Strategy

### 9.1 Test Coverage Requirements

**Phase 1 Requirements:**
- ✅ 90%+ overall coverage
- ✅ 100% coverage for data models
- ✅ 95%+ coverage for storage implementations
- ✅ 85%+ coverage for CLI commands

**Measurement:**
```bash
poetry run pytest --cov=kosmos.world_model --cov-report=term --cov-report=html

# View detailed report
open htmlcov/index.html

# Fail if coverage below 90%
poetry run pytest --cov=kosmos.world_model --cov-fail-under=90
```

### 9.2 Test Pyramid

```
         /\
        /  \  E2E Tests (10%)
       /────\
      /      \  Integration Tests (30%)
     /────────\
    /          \  Unit Tests (60%)
   /────────────\
```

**Unit Tests (60% of tests):**
- Test individual functions/methods in isolation
- Mock all external dependencies
- Fast (<1ms per test)
- Example: Entity validation, relationship creation

**Integration Tests (30% of tests):**
- Test component interactions with real dependencies
- Use test databases (Neo4j, PostgreSQL)
- Moderate speed (~100ms per test)
- Example: Add entity to Neo4j, verify it's stored

**E2E Tests (10% of tests):**
- Test complete user workflows
- All components working together
- Slower (~1s per test)
- Example: Export graph, reset, import, verify integrity

### 9.3 Test Categories

#### 9.3.1 Unit Tests

```python
# tests/unit/world_model/test_models.py
def test_entity_validation():
    """Test entity validation logic."""
    with pytest.raises(ValueError, match="type is required"):
        Entity(type="", properties={})

def test_entity_serialization():
    """Test to_dict/from_dict round-trip."""
    original = Entity(type="Paper", properties={"title": "Test"})
    data = original.to_dict()
    restored = Entity.from_dict(data)
    assert restored.type == original.type
    assert restored.properties == original.properties
```

#### 9.3.2 Integration Tests

```python
# tests/integration/world_model/test_neo4j_storage.py
def test_persistence_across_sessions(neo4j_url):
    """Test data survives driver restart."""
    # Session 1: Create entity
    storage1 = Neo4jWorldModel(neo4j_url, auth)
    entity = Entity(type="Paper", properties={"title": "Test"})
    entity_id = storage1.add_entity(entity)
    storage1.close()

    # Session 2: Retrieve entity
    storage2 = Neo4jWorldModel(neo4j_url, auth)
    retrieved = storage2.get_entity(entity_id)
    assert retrieved is not None
    assert retrieved.properties["title"] == "Test"
    storage2.close()
```

#### 9.3.3 Performance Tests

```python
# tests/performance/test_baseline.py
@pytest.mark.benchmark
def test_bulk_insert_performance(world_model):
    """Test inserting 1000 entities."""
    entities = [
        Entity(type="Paper", properties={"title": f"Paper {i}"})
        for i in range(1000)
    ]

    start = time.time()
    for entity in entities:
        world_model.add_entity(entity)
    elapsed = time.time() - start

    print(f"Inserted 1000 entities in {elapsed:.2f}s")
    print(f"Average: {(elapsed/1000)*1000:.2f}ms per entity")

    assert elapsed < 30, "Bulk insert too slow"
```

### 9.4 Test Fixtures

**Shared fixtures in `tests/conftest.py`:**

```python
import pytest
from kosmos.world_model import WorldModelFactory, Entity
from kosmos.config import Config


@pytest.fixture
def test_config():
    """Test configuration."""
    config = Config()
    config.world_model.mode = "simple"
    config.world_model.simple.neo4j.url = "bolt://localhost:7687"
    config.world_model.simple.neo4j.auth.user = "neo4j"
    config.world_model.simple.neo4j.auth.password = "testpassword"
    config.world_model.simple.neo4j.database = "kosmos_test"
    return config


@pytest.fixture
def world_model(test_config):
    """Create world model for testing."""
    wm = WorldModelFactory.create(test_config)
    yield wm
    # Cleanup
    wm.reset(confirm=True)


@pytest.fixture
def sample_entities():
    """Sample entities for testing."""
    return [
        Entity(type="Paper", properties={"title": "Paper 1", "year": 2020}),
        Entity(type="Paper", properties={"title": "Paper 2", "year": 2021}),
        Entity(type="Concept", properties={"name": "Neural Networks"}),
    ]
```

### 9.5 Continuous Testing

**Pre-commit hook (`.pre-commit-config.yaml`):**

```yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: poetry run pytest tests/unit/
        language: system
        pass_filenames: false
        always_run: true
```

**Run tests automatically on file save (VS Code):**

```json
{
  "python.testing.pytestEnabled": true,
  "python.testing.autoTestDiscoverOnSaveEnabled": true
}
```

---

## 10. Deployment and Operations

### 10.1 Docker Compose (Simple Mode)

**`docker-compose.simple.yml`:**

```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:5.13
    container_name: kosmos-neo4j
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_dbms_memory_heap_max__size: 4G
      NEO4J_dbms_memory_pagecache__size: 2G
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    restart: unless-stopped

  kosmos:
    build: .
    container_name: kosmos-app
    depends_on:
      - neo4j
    environment:
      KOSMOS_NEO4J_URL: bolt://neo4j:7687
      KOSMOS_NEO4J_PASSWORD: ${NEO4J_PASSWORD}
    volumes:
      - ./data:/app/data
    restart: unless-stopped

volumes:
  neo4j_data:
  neo4j_logs:
```

**Usage:**

```bash
# Create .env file
echo "NEO4J_PASSWORD=your_secure_password" > .env

# Start services
docker-compose -f docker-compose.simple.yml up -d

# Check status
docker-compose -f docker-compose.simple.yml ps

# View logs
docker-compose -f docker-compose.simple.yml logs -f

# Stop services
docker-compose -f docker-compose.simple.yml down
```

### 10.2 Docker Compose (Production Mode)

**`docker-compose.production.yml`:**

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: kosmos
      POSTGRES_USER: kosmos
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  neo4j:
    image: neo4j:5.13
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
      NEO4J_PLUGINS: '["apoc", "graph-data-science"]'
    volumes:
      - neo4j_data:/data
    restart: unless-stopped

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    volumes:
      - es_data:/usr/share/elasticsearch/data
    restart: unless-stopped

  chromadb:
    image: ghcr.io/chroma-core/chroma:latest
    volumes:
      - chromadb_data:/chroma/chroma
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

  kosmos:
    build: .
    depends_on:
      - postgres
      - neo4j
      - elasticsearch
      - chromadb
      - redis
    environment:
      KOSMOS_MODE: production
      POSTGRES_URL: postgresql://kosmos:${POSTGRES_PASSWORD}@postgres:5432/kosmos
      NEO4J_URL: bolt://neo4j:7687
      NEO4J_PASSWORD: ${NEO4J_PASSWORD}
      ELASTICSEARCH_URL: http://elasticsearch:9200
      CHROMADB_URL: http://chromadb:8000
      REDIS_URL: redis://redis:6379
    restart: unless-stopped

volumes:
  postgres_data:
  neo4j_data:
  es_data:
  chromadb_data:
```

### 10.3 Backup and Restore

**Backup Script (`scripts/backup.sh`):**

```bash
#!/bin/bash
# Backup world model data

BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Export graph
kosmos graph export "$BACKUP_DIR/graph.json" --format json

# Backup Neo4j volume
docker run --rm \
  -v kosmos_neo4j_data:/data \
  -v "$PWD/$BACKUP_DIR":/backup \
  alpine tar czf /backup/neo4j_data.tar.gz /data

echo "Backup complete: $BACKUP_DIR"
```

**Restore Script (`scripts/restore.sh`):**

```bash
#!/bin/bash
# Restore world model from backup

BACKUP_DIR=$1

if [ -z "$BACKUP_DIR" ]; then
  echo "Usage: ./restore.sh <backup_dir>"
  exit 1
fi

# Stop services
docker-compose down

# Restore Neo4j volume
docker run --rm \
  -v kosmos_neo4j_data:/data \
  -v "$PWD/$BACKUP_DIR":/backup \
  alpine sh -c "cd / && tar xzf /backup/neo4j_data.tar.gz"

# Start services
docker-compose up -d

# Import graph
kosmos graph import "$BACKUP_DIR/graph.json" --mode replace

echo "Restore complete"
```

### 10.4 Monitoring

**Health Check Endpoint:**

```python
# kosmos/cli/health.py
@click.command()
def health():
    """Check world model health."""
    try:
        world_model = WorldModelFactory.create(config)
        stats = world_model.get_info()
        click.echo("✅ World model healthy")
        click.echo(f"   Entities: {stats.entity_count}")
        click.echo(f"   Database: {stats.database}")
        sys.exit(0)
    except Exception as e:
        click.echo(f"❌ World model unhealthy: {e}")
        sys.exit(1)
```

**Prometheus Metrics (Production Mode):**

```python
from prometheus_client import Counter, Histogram

entity_operations = Counter(
    'world_model_entity_operations_total',
    'Total entity operations',
    ['operation']  # add, update, delete
)

query_duration = Histogram(
    'world_model_query_duration_seconds',
    'Query duration in seconds'
)

# Instrument operations
def add_entity(self, entity):
    entity_operations.labels(operation='add').inc()
    with query_duration.time():
        return self.storage.add_entity(entity)
```

---

## 11. CI/CD Pipeline

### 11.1 GitHub Actions Workflow

**`.github/workflows/test.yml`:**

```yaml
name: Test

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']

    services:
      neo4j:
        image: neo4j:5.13
        env:
          NEO4J_AUTH: neo4j/testpassword
        ports:
          - 7687:7687
        options: >-
          --health-cmd "cypher-shell -u neo4j -p testpassword 'RETURN 1'"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install Poetry
        run: |
          curl -sSL https://install.python-poetry.org | python3 -
          echo "$HOME/.local/bin" >> $GITHUB_PATH

      - name: Install dependencies
        run: poetry install

      - name: Run linting
        run: |
          poetry run ruff check kosmos/
          poetry run black --check kosmos/

      - name: Run type checking
        run: poetry run mypy kosmos/world_model

      - name: Run unit tests
        run: poetry run pytest tests/unit/ -v

      - name: Run integration tests
        run: poetry run pytest tests/integration/ -v
        env:
          NEO4J_URL: bolt://localhost:7687
          NEO4J_PASSWORD: testpassword

      - name: Check coverage
        run: |
          poetry run pytest --cov=kosmos.world_model --cov-report=xml --cov-fail-under=90

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: true
```

### 11.2 Release Workflow

**`.github/workflows/release.yml`:**

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install Poetry
        run: |
          curl -sSL https://install.python-poetry.org | python3 -
          echo "$HOME/.local/bin" >> $GITHUB_PATH

      - name: Build package
        run: poetry build

      - name: Publish to PyPI
        run: poetry publish --username __token__ --password ${{ secrets.PYPI_TOKEN }}

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          files: dist/*
          generate_release_notes: true
```

---

## 12. Developer Guidelines

### 12.1 Code Style

**Follow these conventions:**

1. **PEP 8** for Python code style
2. **Type hints** for all function signatures
3. **Docstrings** for all public APIs (Google style)
4. **Comments** explain WHY, not WHAT
5. **Educational notes** for complex patterns

**Example:**

```python
def merge_entities(
    self, source_id: str, target_id: str, preserve_provenance: bool = True
) -> str:
    """
    Merge two entities into one.

    DESIGN RATIONALE:
    We preserve the target entity ID to maintain stable references.
    All relationships from source are moved to target.

    Args:
        source_id: Entity to merge from (will be deleted)
        target_id: Entity to merge into (will be kept)
        preserve_provenance: Whether to record merge in provenance

    Returns:
        ID of merged entity (same as target_id)

    Raises:
        EntityNotFoundError: If either entity doesn't exist

    Example:
        >>> merged_id = storage.merge_entities("old_id", "new_id")
        >>> assert merged_id == "new_id"
    """
    # Implementation...
```

### 12.2 Git Workflow

**Branch Strategy:**

```
main (production-ready)
  ↑
develop (integration branch)
  ↑
feature/phase-1-foundation
feature/phase-2-curation
bugfix/export-unicode-handling
```

**Commit Message Format:**

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `test`: Adding tests
- `refactor`: Code refactoring
- `perf`: Performance improvement

**Examples:**

```
feat(world_model): implement Neo4j persistence

- Add Neo4jWorldModel class
- Implement all WorldModelStorage methods
- Add schema initialization
- 90% test coverage

Closes #42
```

```
fix(export): handle unicode characters in entity properties

Entity properties containing unicode were causing JSON export to fail.
Added proper encoding handling.

Fixes #156
```

### 12.3 Pull Request Checklist

Before submitting a PR, ensure:

**Code Quality:**
- [ ] All tests passing (`poetry run pytest`)
- [ ] Code coverage ≥90% (`poetry run pytest --cov`)
- [ ] Type checking passes (`poetry run mypy kosmos/world_model`)
- [ ] Linting passes (`poetry run ruff check && poetry run black --check`)
- [ ] No security vulnerabilities (`poetry run bandit -r kosmos/`)

**Documentation:**
- [ ] Docstrings added for new public APIs
- [ ] User guide updated if user-facing changes
- [ ] Architecture docs updated if design changes
- [ ] CHANGELOG.md updated

**Testing:**
- [ ] Unit tests for new code
- [ ] Integration tests for new features
- [ ] Edge cases covered
- [ ] Error handling tested

**Review:**
- [ ] Self-reviewed code
- [ ] No commented-out code
- [ ] No debugging print statements
- [ ] Educational comments added for complex logic

### 12.4 Common Patterns

**Error Handling:**

```python
# ✅ Good: Specific exceptions with context
try:
    entity = self.get_entity(entity_id)
    if not entity:
        raise EntityNotFoundError(entity_id)
except Neo4jError as e:
    raise WorldModelStorageError(f"Failed to retrieve entity: {e}") from e

# ❌ Bad: Bare except, no context
try:
    entity = self.get_entity(entity_id)
except:
    raise Exception("Error")
```

**Logging:**

```python
# ✅ Good: Structured logging with context
logger.info(
    "Entity created",
    extra={"entity_id": entity.id, "type": entity.type, "project": entity.project}
)

# ❌ Bad: String formatting, no structure
logger.info(f"Created entity {entity.id}")
```

**Configuration:**

```python
# ✅ Good: Use config objects, validate early
neo4j_url = config.world_model.simple.neo4j.url
if not neo4j_url:
    raise ValueError("Neo4j URL not configured")

# ❌ Bad: Environment variables everywhere, late validation
neo4j_url = os.getenv("NEO4J_URL")  # Might be None!
```

### 12.5 Getting Help

**Resources:**
- 📖 Read `docs/planning/architecture.md` for design decisions
- 📖 Read ADRs in `docs/architecture/adr/` for rationale
- 🐛 Check existing issues before creating new ones
- 💬 Ask in GitHub Discussions for design questions
- 📧 Email maintainers for security issues

**Before asking:**
1. Search existing issues and discussions
2. Read relevant documentation
3. Try to debug the issue yourself
4. Prepare a minimal reproducible example

---

## Document Complete!

This implementation guide provides:

✅ **Complete Phase 1 implementation** with copy-paste ready code
✅ **Structured overviews** for Phases 2-4
✅ **Complete file structure** and organization
✅ **Comprehensive testing strategy** with examples
✅ **Deployment guides** for Simple and Production modes
✅ **CI/CD pipeline** configuration
✅ **Developer guidelines** and best practices

**Next Steps:**

1. Review this document with your team
2. Begin Phase 0 user validation
3. Start Phase 1 Sprint 1 when validation complete
4. Follow the sprints sequentially
5. Gate progression on acceptance criteria

**Document Status:** ✅ Complete
**Version:** 1.0
**Last Updated:** November 2025


# Checkpoint: Ready to Create implementation.md

**Date:** November 14, 2025
**Status:** Phase 1 Complete - Documentation Foundation Ready
**Next Task:** Create implementation.md

---

## Executive Summary

We have successfully completed the comprehensive planning documentation for the Kosmos World Model persistent knowledge graph feature. All foundation documents are complete and ready for implementation planning.

**Completed Documents:**
1. ✅ `docs/planning/objective.md` (677 lines, 21KB)
2. ✅ `docs/planning/world_model_implementation_decision_framework.md` (1,210 lines, 36KB)
3. ✅ `docs/planning/requirements.md` (1,200 lines)
4. ✅ `docs/planning/architecture.md` (3,681 lines)

**Next Document to Create:**
- `docs/planning/implementation.md` - Detailed implementation tasks, sprint planning, file structure, testing procedures

---

## Context and Background

### The Project

**Goal:** Implement persistent world models for Kosmos AI Scientist

**The "All Three" Strategy:**
- **Goal A:** Faithful reproduction of FutureHouse's production system (Production Mode - Month 5)
- **Goal B:** Practical tool individual researchers can use (Simple Mode - Week 6)
- **Goal C:** Educational reference implementation (Documentation - Ongoing)

**Key Innovation:**
Progressive enhancement - Start with Simple Mode (Neo4j only), provide clear path to Production Mode (PostgreSQL + Neo4j + Elasticsearch + Vector DB)

### What We're Building

**Simple Mode (Default):**
- Persistent Neo4j storage
- Local laptop deployment
- 10K-50K entities
- Export/import functionality
- Multi-project support
- Basic provenance
- Keyword search

**Production Mode (Optional):**
- Polyglot persistence (4 databases)
- Enterprise scalability (100K+ entities)
- PROV-O standard provenance
- Vector semantic search
- GraphRAG queries
- Multi-user support

### Implementation Phases

**Phase 0: Validation & Architecture (2 weeks)**
- User validation via GitHub Issue #4
- Architecture vision documents
- Interface design

**Phase 1: Foundation (3-4 weeks)**
- Persistent Neo4j
- Export/import
- Project tagging
- Abstract storage layer
- CLI commands

**Phase 2: Curation (3-4 weeks)**
- Verification system
- Annotation system
- Duplicate detection
- Quality analysis

**Phase 3: Multi-Project (2-3 weeks)**
- Project management
- Isolated graphs
- Cross-project queries

**Phase 4: Production Mode (4-6 weeks)**
- Refine Simple Mode
- Add PostgreSQL, Elasticsearch, Vector DB
- PROV-O provenance
- GraphRAG

---

## Documents Completed

### 1. objective.md (677 lines)

**Location:** `docs/planning/objective.md`

**Purpose:** Explain the end goal in clear, accessible language for all stakeholders

**Key Sections:**
- Vision in one sentence
- The problem we're solving (ephemeral graphs → amnesia analogy)
- What we're building (all three goals)
- Success criteria (technical, user, educational)
- Timeline and milestones (Phase 0-4)
- Guiding principles (6 principles)
- Risks and mitigation (6 risks with honest assessments)
- Measures of progress

**Target Audience:** Researchers, students, organizations, developers

---

### 2. world_model_implementation_decision_framework.md (1,210 lines)

**Location:** `docs/planning/world_model_implementation_decision_framework.md`

**Purpose:** THE canonical decision document serving as foundation for all follow-on documents

**Key Sections:**
- Executive summary (gap between paper vision and current codebase)
- The "All Three" goals revelation
- Research analysis summary
- Architectural principles for "All Three"
- Revised phased implementation (P0-P4)
- Documentation structure
- Foundation for requirements.md, architecture.md, implementation.md
- Success criteria for each goal
- Updated timeline
- Comparison to Kosmos paper

**Key Architectural Patterns:**
```python
class WorldModelStorage(ABC):
    # Abstract interface enabling both Simple and Production modes
    @abstractmethod
    def add_entity(self, entity: Entity) -> str: ...
    @abstractmethod
    def query(self, query: str) -> List[Entity]: ...

# Simple Mode Implementation
class Neo4jWorldModel(WorldModelStorage):
    # Single Neo4j database

# Production Mode Implementation
class PolyglotWorldModel(WorldModelStorage):
    # PostgreSQL + Neo4j + Elasticsearch + VectorDB
```

---

### 3. requirements.md (1,200 lines)

**Location:** `docs/planning/requirements.md`

**Purpose:** RFC 2119 requirements specification (MUST, SHALL, SHOULD, MAY)

**Key Sections:**

**Functional Requirements (139 requirements across 5 phases):**
- FR-1: Phase 0 - Validation and Architecture (14 requirements)
- FR-2: Phase 1 - Foundation (45 requirements)
  - Persistent Neo4j storage
  - Knowledge accumulation logic
  - CLI commands (info, export, import, reset)
  - Export/import functionality
  - Project tagging
  - Abstract storage layer
  - Configuration structure
  - Observability hooks
- FR-3: Phase 2 - Curation Features (24 requirements)
  - Entity verification
  - Annotation system
  - Duplicate detection and merging
  - Quality analysis
- FR-4: Phase 3 - Multi-Project Support (19 requirements)
  - Project management (create, list, switch, delete)
  - Isolated project graphs
  - Cross-project queries
  - Project export/import
- FR-5: Phase 4 - Production Mode (37 requirements)
  - PostgreSQL integration
  - Elasticsearch integration
  - Vector DB integration
  - PROV-O provenance
  - GraphRAG query engine
  - Production mode configuration
  - Mode migration

**Non-Functional Requirements (6 categories):**
- NFR-1: Performance (Simple: <1s, Production: <100ms)
- NFR-2: Reliability (no data loss, 90%+ test coverage)
- NFR-3: Usability (5-minute setup, clear errors)
- NFR-4: Educational Value (documented rationale, ADRs)
- NFR-5: Maintainability (type checking, modularity)
- NFR-6: Security (encryption, input validation)

**User Stories (10 stories across 5 personas):**
- US-1: Graduate Student (3 stories)
- US-2: Research Lab (2 stories)
- US-3: Research Organization (2 stories)
- US-4: Computer Science Student (2 stories)
- US-5: Open Source Contributor (1 story)

**Acceptance Criteria:**
- AC-1 through AC-5: Detailed criteria for each phase with checkboxes

**Success Gates:**
- SG-1 through SG-4: Go/no-go decision points between phases

---

### 4. architecture.md (3,681 lines)

**Location:** `docs/planning/architecture.md`

**Purpose:** Complete technical architecture specification with all implementation details

**Key Sections (20 major sections):**

1. **Document Overview** - Purpose, scope, goals, key decisions
2. **System Context** - C4 diagrams, integration points
3. **Architecture Principles** (6 principles with educational notes)
4. **Architecture Patterns** (5 patterns with code examples)
5. **Component Architecture** - WorldModel, Storage, Provenance, etc.
6. **Interface Specifications** - All ABCs with complete contracts
7. **Data Models** - JSON schemas for entities, relationships, export, PROV-O
8. **Database Schemas**
   - Neo4j: Nodes, relationships, indexes (Simple Mode)
   - PostgreSQL: 6 tables with indexes (Production Mode)
   - Elasticsearch: 2 indices (Production Mode)
   - Vector DB: ChromaDB/Pinecone config (Production Mode)
9. **API Specifications**
   - Python API
   - CLI API (15+ commands)
   - Configuration API
10. **Data Flow Diagrams** (3 flows with ASCII diagrams)
11. **Deployment Architectures** (3 options with docker-compose)
12. **Security Architecture** - Auth, encryption, validation, audit logging
13. **Performance Architecture** - Caching, optimization, SLOs
14. **Observability Architecture** - Metrics, tracing, logging, health checks
15. **Migration Architecture** - Version compatibility, procedures, rollback
16. **Architecture Decision Records** (6 ADRs)
17. **Technology Stack** - Core technologies, Python libraries
18. **Comparison to Kosmos Paper** - What matches, differs, enhanced
19. **Next Steps** - Follow-on documents
20. **Document Metadata**

**Critical Architecture Decisions:**
- ADR-001: Abstract Storage Layer (enables mode switching)
- ADR-002: Neo4j for Simple Mode (single DB, easy setup)
- ADR-003: Polyglot for Production Mode (specialized databases)
- ADR-004: Project Isolation via Namespaces (not separate DBs)
- ADR-005: PROV-O Standard (publication-quality provenance)
- ADR-006: GraphRAG Query Pattern (LLM + graph + semantic search)

**Technology Stack Highlights:**
- Python 3.10+
- Neo4j 5.x (both modes)
- PostgreSQL 15+ (Production)
- Elasticsearch 8.x (Production)
- ChromaDB/Pinecone (Production)
- Redis 7.x (Production caching)
- Docker/Docker Compose

---

## What implementation.md Should Contain

Based on the decision framework, requirements, and architecture documents, `implementation.md` should provide:

### 1. Detailed Task Breakdown

**Phase 0 (Weeks 1-2):**
- Tasks for user validation
- Architecture document creation tasks
- Interface design tasks

**Phase 1 (Weeks 3-6):**
- Sprint 1: Architecture setup (interfaces, abstractions)
- Sprint 2: Persistence implementation (Neo4j, volumes)
- Sprint 3: CLI commands (info, export, import, reset)
- Sprint 4: Testing and documentation

**Phase 2-4:** Similar detailed breakdown

### 2. File Structure

Complete module layout:
```
kosmos/
├── world_model/
│   ├── __init__.py
│   ├── interface.py           # NEW - Abstract interfaces
│   ├── simple.py              # NEW - Neo4jWorldModel
│   ├── production.py          # NEW - PolyglotWorldModel (Phase 4)
│   ├── models.py              # NEW - Entity, Relationship, etc.
│   ├── factory.py             # NEW - WorldModelFactory
│   ├── provenance/
│   │   ├── __init__.py
│   │   ├── interface.py       # ProvenanceTracker ABC
│   │   ├── basic.py           # Simple Mode
│   │   └── prov_o.py          # Production Mode (Phase 4)
│   ├── search/
│   │   ├── __init__.py
│   │   ├── interface.py       # SemanticSearch ABC
│   │   ├── keyword.py         # Simple Mode
│   │   └── vector.py          # Production Mode (Phase 4)
│   ├── query/
│   │   ├── __init__.py
│   │   ├── interface.py       # QueryEngine ABC
│   │   ├── direct.py          # Simple Mode (Cypher)
│   │   └── graphrag.py        # Production Mode (Phase 4)
│   └── migration/
│       ├── __init__.py
│       └── commands.py        # Migration commands
├── cli/
│   ├── graph_commands.py      # NEW - graph info/export/import/reset
│   ├── project_commands.py    # NEW - Phase 3
│   └── migrate_commands.py    # NEW - Phase 4
└── config.py                  # MODIFY - Add world_model section
```

### 3. Testing Strategy

**Unit Tests:**
- Test each interface implementation independently
- Mock tests for business logic
- 90%+ coverage requirement

**Integration Tests:**
- Full workflow tests (create → query → export → import)
- Cross-mode compatibility tests
- Migration tests

**Performance Tests:**
- Baseline measurements for Simple Mode
- Regression tests
- Load tests for Production Mode

### 4. Development Procedures

**Setup:**
```bash
# Development environment
poetry install
docker-compose -f docker-compose.dev.yml up -d

# Run tests
pytest --cov=kosmos.world_model

# Type checking
mypy kosmos/world_model

# Linting
ruff check kosmos/world_model
black kosmos/world_model
```

**Git Workflow:**
- Feature branches for each phase
- PR requirements (tests, coverage, review)
- Commit message format

### 5. CI/CD Pipeline

**GitHub Actions:**
- Test on Python 3.10, 3.11, 3.12
- Integration tests with Neo4j
- Coverage reporting
- Type checking
- Linting

**Deployment:**
- Docker image building
- Version tagging
- Release process

### 6. Documentation Procedures

**For Each Phase:**
- Update user guides
- Update API documentation
- Create tutorials
- Update CHANGELOG

### 7. Monitoring and Observability Setup

**Metrics:**
- Define Prometheus metrics
- Setup dashboards

**Logging:**
- Configure structlog
- Define log formats

**Tracing:**
- Setup OpenTelemetry
- Define trace spans

### 8. Migration Scripts

**Data Migrations:**
- Phase 1 → Phase 2 scripts
- Simple → Production scripts
- Rollback procedures

### 9. Deployment Guides

**Docker Compose:**
- Simple Mode: `docker-compose.simple.yml`
- Production Mode: `docker-compose.production.yml`

**Cloud Deployments:**
- AWS deployment guide
- GCP deployment guide
- Azure deployment guide

---

## Key Design Patterns to Implement

### 1. Abstract Base Classes (ABCs)

```python
from abc import ABC, abstractmethod

class WorldModelStorage(ABC):
    @abstractmethod
    def add_entity(self, entity: Entity) -> str:
        """Add entity, return ID."""
        pass

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        pass

    @abstractmethod
    def query(self, query: str, **kwargs) -> List[Entity]:
        """Execute query."""
        pass
```

### 2. Factory Pattern

```python
class WorldModelFactory:
    @staticmethod
    def create(config: Config) -> WorldModel:
        if config.world_model.mode == "simple":
            return WorldModelFactory._create_simple_mode(config)
        elif config.world_model.mode == "production":
            return WorldModelFactory._create_production_mode(config)
```

### 3. Repository Pattern

```python
class EntityRepository:
    def __init__(self, storage: WorldModelStorage):
        self.storage = storage

    def add(self, entity: Entity) -> str:
        return self.storage.add_entity(entity)

    def find_by_type(self, entity_type: str) -> List[Entity]:
        return self.storage.query(f"type = '{entity_type}'")
```

### 4. Command Pattern (Migrations)

```python
class MigrationCommand(ABC):
    @abstractmethod
    def execute(self) -> MigrationResult:
        pass

    @abstractmethod
    def rollback(self):
        pass

    @abstractmethod
    def validate(self) -> ValidationResult:
        pass
```

---

## Critical Implementation Notes

### 1. Configuration Structure

The configuration must support both modes:

```yaml
world_model:
  mode: simple  # or "production"

  simple:
    backend: neo4j
    neo4j:
      url: bolt://localhost:7687
      auth:
        user: neo4j
        password: ${NEO4J_PASSWORD}
      database: kosmos
      data_path: ~/.kosmos/neo4j_data
    similarity_threshold: 0.85

  production:
    postgres:
      url: postgresql://user:pass@localhost:5432/kosmos
      pool_size: 10
    neo4j:
      url: bolt://localhost:7687
    elasticsearch:
      url: http://localhost:9200
    vector_db:
      provider: pinecone
      api_key: ${PINECONE_API_KEY}
```

### 2. Data Models

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime

@dataclass
class Entity:
    id: Optional[str] = None
    type: str  # "Paper", "Concept", etc.
    properties: Dict[str, Any]
    confidence: float = 1.0
    project: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None
    verified: bool = False
    annotations: List[Annotation] = field(default_factory=list)

@dataclass
class Relationship:
    id: Optional[str] = None
    source_id: str
    target_id: str
    type: str  # "CITES", "MENTIONS", etc.
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    created_at: Optional[datetime] = None
    created_by: Optional[str] = None
```

### 3. Neo4j Schema (Simple Mode)

```cypher
CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

CREATE INDEX entity_type IF NOT EXISTS
FOR (e:Entity) ON (e.type);

CREATE INDEX entity_project IF NOT EXISTS
FOR (e:Entity) ON (e.project);

CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
FOR (e:Entity) ON EACH [e.properties_text];
```

### 4. CLI Commands

```bash
# Phase 1
kosmos graph info [--project PROJECT]
kosmos graph export FILEPATH [--format json|graphml] [--project PROJECT]
kosmos graph import FILEPATH [--project PROJECT] [--mode merge|replace]
kosmos graph reset [--project PROJECT] [--confirm]

# Phase 3
kosmos project create NAME [--description DESC]
kosmos project list
kosmos project switch NAME
kosmos project delete NAME [--confirm]

# Phase 2
kosmos verify ENTITY_ID
kosmos annotate ENTITY_ID "text"
kosmos duplicates [--project PROJECT]
kosmos quality [--project PROJECT]

# Phase 4
kosmos migrate simple-to-production --validate --backup PATH
```

---

## Performance Targets

### Simple Mode
- Entity creation: < 100ms p95
- Query latency: < 1 second p95
- Export 10K entities: < 30 seconds
- Scale: 10K-50K entities

### Production Mode
- Entity creation: < 50ms p95
- Query latency: < 100ms p95
- Export 10K entities: < 10 seconds
- Scale: 100K+ entities

---

## Testing Requirements

### Coverage
- Unit tests: 90%+ coverage
- Integration tests: Critical paths covered
- Performance tests: Baseline measurements
- Migration tests: All upgrade paths tested

### Test Structure
```
tests/
├── unit/
│   ├── world_model/
│   │   ├── test_interface.py
│   │   ├── test_simple.py
│   │   ├── test_models.py
│   │   └── test_factory.py
│   ├── provenance/
│   ├── search/
│   └── query/
├── integration/
│   ├── test_simple_mode_workflow.py
│   ├── test_export_import.py
│   └── test_multi_project.py
├── performance/
│   ├── test_baseline_simple.py
│   └── test_baseline_production.py
└── migration/
    └── test_simple_to_production.py
```

---

## Success Criteria Checklist

### Phase 0 (Week 2)
- [ ] 10+ users express interest in persistent graphs
- [ ] Architecture vision documented
- [ ] User personas created
- [ ] Go/no-go decision made

### Phase 1 (Week 6)
- [ ] 5+ active users using Simple Mode
- [ ] Knowledge accumulates correctly across runs
- [ ] Export/import works without data loss
- [ ] 90%+ test coverage
- [ ] No regressions in existing features

### Phase 2 (Month 3)
- [ ] 20+ active users
- [ ] Users actively curating graphs
- [ ] Quality metrics helpful

### Phase 3 (Month 3)
- [ ] 50+ active users
- [ ] Users managing multiple projects
- [ ] Cross-project queries working

### Phase 4 (Month 5)
- [ ] Production Mode deployed by 1+ organization
- [ ] Both modes maintained and documented
- [ ] Performance meets targets

---

## Next Steps

### Immediate Action

Create `docs/planning/implementation.md` with:

1. **Detailed Sprint Planning** (Phase 0 through Phase 4)
   - Week-by-week task breakdown
   - Story points / time estimates
   - Dependencies between tasks

2. **Complete File Structure**
   - All new files to create
   - All existing files to modify
   - Import statements and module organization

3. **Implementation Tasks** for Each File
   - What to implement
   - Dependencies
   - Testing requirements

4. **Testing Procedures**
   - Unit test templates
   - Integration test scenarios
   - Performance test scripts

5. **Deployment Scripts**
   - Docker configurations
   - docker-compose files
   - Initialization scripts

6. **CI/CD Configuration**
   - GitHub Actions workflows
   - Test runners
   - Coverage reporting

7. **Migration Procedures**
   - Data transformation scripts
   - Validation procedures
   - Rollback scripts

8. **Documentation Templates**
   - User guide outlines
   - API doc structure
   - Tutorial outlines

---

## References

### Documents to Reference

1. **requirements.md** - For what needs to be built
   - Functional requirements FR-1 through FR-5
   - Non-functional requirements NFR-1 through NFR-6
   - User stories and acceptance criteria

2. **architecture.md** - For how to build it
   - Interface specifications (Section 6)
   - Data models (Section 7)
   - Database schemas (Section 8)
   - API specifications (Section 9)
   - Architecture patterns (Section 4)

3. **world_model_implementation_decision_framework.md** - For rationale
   - Why decisions were made
   - Trade-offs considered
   - Phased approach justification

4. **objective.md** - For context
   - End goal clarity
   - Success criteria
   - Guiding principles

### Key Kosmos Files to Understand

```
kosmos/
├── config.py              # Current config structure
├── db/__init__.py         # Database session management
├── knowledge/
│   ├── graph.py           # Current graph implementation
│   └── graph_builder.py   # Current graph builder
├── agents/                # How agents will use world model
└── cli/
    └── main.py            # Current CLI structure
```

---

## Important Constraints

1. **Backward Compatibility:** Must not break existing Kosmos features
2. **Simple Mode Default:** Simple Mode must be the default, production mode opt-in
3. **Progressive Enhancement:** Same API, different capabilities by mode
4. **Educational Value:** Code must teach patterns (Goal C)
5. **90%+ Test Coverage:** Required for all phases
6. **No Data Loss:** Critical requirement for migrations

---

## Contact/Questions

If continuing this work and questions arise:

1. **For architecture questions:** Refer to architecture.md Section 16 (ADRs)
2. **For requirements questions:** Refer to requirements.md
3. **For context:** Read objective.md first
4. **For decisions:** Read decision_framework.md

---

## Checkpoint Status

✅ **All foundation documents complete**
✅ **Ready to create implementation.md**
✅ **All decisions documented**
✅ **All requirements captured**
✅ **Full architecture specified**

**Next Session Action:**
Create `docs/planning/implementation.md` following the structure outlined in this checkpoint document.

---

**END OF CHECKPOINT**

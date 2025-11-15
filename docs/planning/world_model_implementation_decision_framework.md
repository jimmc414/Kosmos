# World Model Implementation Decision Framework

**Status:** Decision Made - "All Three" Goals
**Date:** November 2025
**Context:** GitHub Issue #4 + World Model Architecture Research Analysis
**Decision:** Path 3 with Architectural Discipline

---

## Executive Summary

### The Fundamental Disconnect

A critical gap exists between the Kosmos paper's vision and the current codebase implementation:

**What Andrew White Designed:**
- Structured world models as **queryable databases** (not expanded context windows)
- Information persists across millions of tokens and 200+ agent rollouts
- Enables "coherent pursuit" over extended research cycles
- **This is the core innovation** that made Kosmos work

**What the Current Codebase Has:**
- Ephemeral knowledge graphs that reset each run
- No persistence between research sessions
- Knowledge graph feature **disabled by default** (`use_knowledge_graph = False`)
- Infrastructure code exists but no accumulated knowledge

**Analogy:** We've built a brain that forgets everything each night. The coherence White describes is impossible without persistence.

### The Revelation: Pursuing All Three Goals

**The Core Question:** *"What is jimmc414/Kosmos trying to be?"*

**The Answer:** **ALL THREE:**

- **Goal A:** Faithful reproduction of FutureHouse's production system architecture
- **Goal B:** Practical tool individual researchers can actually use today
- **Goal C:** Educational reference implementation showing concepts at multiple scales

**This is not a compromise - it's a strength.** By pursuing all three goals simultaneously through phased implementation, we create:
- A usable tool NOW (Goal B)
- That teaches architectural patterns (Goal C)
- While building toward production-grade system (Goal A)

### The Implementation Strategy

**Path 3: Hybrid Incremental with Architectural Discipline**

**Simple Mode (Default):** Works great for 90% of users
- Persistent Neo4j
- Local deployment
- Low operational complexity
- Practical and usable immediately

**Production Mode (Optional):** For organizations and power users
- Full polyglot architecture (PostgreSQL + Neo4j + Elasticsearch + ChromaDB)
- Enterprise features and scale
- Opt-in via configuration
- Faithful to paper's vision

**Educational Value:** Throughout
- Architecture documentation explaining the why
- Code shows patterns at multiple scales
- Examples demonstrate progression
- Clear upgrade paths teach scaling decisions

**Timeline:**
- **6 weeks:** Simple mode working (Phase 1)
- **3 months:** Full simple mode features (Phases 1-3)
- **5 months:** Production mode available (Phase 4)
- **Ongoing:** Both modes maintained and improved

---

## How "All Three" Changes The Approach

### What Stays The Same

- **Path 3 (Hybrid Incremental) is still the foundation**
- **Deliver practical value first** (Phase 1 in 2-3 weeks)
- **Validate with users** before heavy investment
- **Incremental phases** justified by demand

### What Changes Significantly

#### 1. Architecture Must Be "Upgrade-Friendly"

Design with abstraction layers from Day 1:

```python
# Abstract interface all phases use
class WorldModelStorage(ABC):
    """
    World model storage abstraction.

    Current: Neo4j simple implementation (Phase 1-3)
    Future: Polyglot persistence (Phase 4+ production mode)
    Paper: White et al. "structured world models" principle
    """
    @abstractmethod
    def add_entity(self, entity: Entity) -> str:
        """Add entity to world model."""
        pass

    @abstractmethod
    def query(self, query: str, **kwargs) -> Results:
        """Query world model."""
        pass

# Phase 1-3: Simple implementation
class Neo4jWorldModel(WorldModelStorage):
    """Simple Neo4j-based world model for individual researchers."""
    def add_entity(self, entity: Entity) -> str:
        # Direct Neo4j operations
        return self.neo4j.create_node(entity)

    def query(self, query: str, **kwargs) -> Results:
        # Cypher queries
        return self.neo4j.execute(query)

# Phase 4+: Production implementation
class PolyglotWorldModel(WorldModelStorage):
    """Production polyglot world model."""
    def __init__(self):
        self.postgres = PostgresClient()    # Transactions
        self.neo4j = Neo4jClient()          # Relationships
        self.elasticsearch = ESClient()     # Provenance
        self.vector_db = ChromaDBClient()   # Semantic search

    def add_entity(self, entity: Entity) -> str:
        # Route to appropriate backend
        entity_id = self.postgres.insert(entity)
        self.neo4j.create_node(entity, entity_id)
        self.elasticsearch.index_event(entity)
        self.vector_db.embed(entity)
        return entity_id
```

**Key Principle:** Write Phase 1 so Phase 4 can swap implementations without breaking user code.

#### 2. Documentation Becomes First-Class

Since we're aiming for educational value:

```
docs/
├── architecture/
│   ├── world_model_vision.md          # Full vision (Goal A)
│   ├── design_principles.md           # Why decisions made (Goal C)
│   ├── simple_mode_architecture.md    # Phase 1-3 design
│   ├── production_mode_architecture.md # Phase 4 design
│   ├── migration_paths.md             # Phase transitions
│   └── decision_records/              # ADRs for key choices
├── guides/
│   ├── getting_started.md             # Quick start (Goal B)
│   ├── building_knowledge_base.md     # Using simple mode
│   ├── production_deployment.md       # Using production mode (Goal A)
│   └── troubleshooting.md
└── tutorials/
    ├── understanding_world_models.md  # Concepts (Goal C)
    ├── from_ephemeral_to_persistent.md # Evolution story
    └── scaling_architecture.md        # When and how (Goal A+C)
```

Each phase document explains:
- What we're building
- Why this architecture choice
- How it maps to paper's vision
- What we're deferring and why
- How to migrate to next phase

#### 3. Each Phase Must Be Production-Quality

Not prototypes - real implementations:
- **Phase 1:** "Persistent world models at small scale" (not toy)
- **Phase 2:** "User control and quality patterns" (not hacks)
- **Phase 3:** "Multi-project architecture" (not workaround)

Each phase:
- Thoroughly tested
- Well documented
- Usable in production at that scale
- A learning resource showing patterns

#### 4. Progressive Enhancement Pattern

Same interfaces, different capabilities:

```python
class WorldModel:
    def __init__(self, mode: str = "simple"):
        if mode == "simple":
            self.storage = Neo4jWorldModel()
        else:  # production
            self.storage = PolyglotWorldModel()

    def query(self, question: str) -> Results:
        """Query world model with appropriate backend."""
        if self.config.mode == "simple":
            # Direct Cypher query
            return self.storage.simple_query(question)
        else:
            # Full GraphRAG with semantic search
            return self.storage.advanced_query(question)
```

#### 5. Clear Migration Paths

Tools for transitions:

```bash
# Migrate between phases
kosmos migrate phase1-to-phase2
kosmos migrate simple-to-production

# Export for different targets
kosmos export --format simple           # Phase 1-3 format
kosmos export --format production       # Phase 4 format
kosmos export --format paper-supplement # Research publication

# Validate compatibility
kosmos validate-migration --target production
```

---

## Research Analysis Summary

[Content remains the same as original document through "Critical Gaps Identified" section - lines 46-117]

### What the Polyglot Architecture Research Reveals

**Source:** `docs/planning/optimal_world_model_architecture_research.md`

**Key Findings:**
- Kosmos paper describes a production system coordinating 200 parallel agents
- Processes 42,000 lines of code per run across 20+ cycles
- Costs $200 per investigation with $5,700-6,500/month infrastructure
- Requires polyglot persistence: PostgreSQL (transactions), Neo4j (relationships), Elasticsearch (provenance), ChromaDB/Pinecone (semantic search)

**Critical Insight:**
This architecture solves for **production-scale multi-agent coordination** at FutureHouse, not individual researcher workflows.

**What It Doesn't Address:**
- Individual researcher workflows (weeks/months, not 12-hour runs)
- Multiple concurrent projects per user
- Knowledge sharing between researchers
- Local laptop deployment
- Minimal operational complexity for open-source users

### What Andrew White's Blog Insights Reveal

**Source:** `docs/planning/optimal_world_model_architecture_research_from_blog.md`

**Key Principles Extracted:**
1. **Structured world models enable persistence beyond context limits**
2. **Simplicity beats complexity**
3. **Small specialized models exceed large generalists**
4. **Complete provenance is mandatory**
5. **Demonstration trajectories are more valuable than training data**
6. **Parallel execution with shared memory beats sequential chaining**
7. **Iterative cycles with world model updates enable progressive refinement**

**Critical Insight:**
White describes **design principles** and **why** they matter, not **how** to implement them for end users.

### Critical Gaps Identified

**1. Scale Mismatch**
- Research describes 200-agent production system
- Open-source users need single-researcher local tool

**2. User Workflow Mismatch**
- Research assumes one 12-hour continuous run
- Real users work over weeks/months with interruptions

**3. Mental Model Mismatch**
- Research focuses on system architecture
- Users think in terms of projects and expertise

**4. Operational Complexity Mismatch**
- Research recommends 4 database systems
- Open-source users want simple setup

---

## Architectural Principles for "All Three" Goals

### Principle 1: Interface-Based Design

Every major component has a clean interface allowing implementation swaps:

```python
from abc import ABC, abstractmethod

class WorldModelStorage(ABC):
    """Storage abstraction - simple or polyglot."""
    @abstractmethod
    def add_entity(self, entity: Entity) -> str: ...
    @abstractmethod
    def get_entity(self, entity_id: str) -> Entity: ...
    @abstractmethod
    def query(self, query: str) -> List[Entity]: ...

class ProvenanceTracker(ABC):
    """Provenance tracking - basic or PROV-O standard."""
    @abstractmethod
    def record_derivation(self, output: str, inputs: List[str]): ...
    @abstractmethod
    def trace_lineage(self, entity_id: str) -> ProvenanceChain: ...

class SemanticSearch(ABC):
    """Semantic search - keywords or vector embeddings."""
    @abstractmethod
    def find_similar(self, query: str, top_k: int) -> List[Entity]: ...

class QueryEngine(ABC):
    """Query engine - Cypher or GraphRAG."""
    @abstractmethod
    def execute_query(self, natural_language: str) -> Results: ...
```

### Principle 2: Configuration-Driven Mode Selection

One codebase, multiple deployment modes:

```yaml
# config.yaml
world_model:
  mode: simple  # or "production"

  simple:
    backend: neo4j
    path: ~/.kosmos/neo4j_data
    provenance: basic
    semantic_search: keyword

  production:
    postgres:
      url: postgresql://localhost/kosmos
    neo4j:
      url: bolt://localhost:7687
    elasticsearch:
      url: http://localhost:9200
    vector_db:
      provider: pinecone  # or chromadb
      api_key: ${PINECONE_API_KEY}
    provenance: prov-o
    semantic_search: vector
```

### Principle 3: Progressive Enhancement

Features work at basic level, enhance in production mode:

**Simple Mode:**
- Basic keyword search
- Simple lineage tracking (parent/child)
- Direct Cypher queries
- Manual curation

**Production Mode:**
- Vector semantic search
- Complete PROV-O provenance
- GraphRAG with LLM
- Automated quality metrics

**Same User Experience:** Just more powerful in production mode.

### Principle 4: Observability from Day 1

Even simple mode includes hooks for metrics:

```python
class WorldModel:
    def add_entity(self, entity: Entity) -> str:
        # Instrument even in simple mode
        with self.metrics.timer('add_entity'):
            with self.tracer.span('add_entity'):
                entity_id = self.storage.add_entity(entity)
                self.metrics.increment('entities_created')
                self.log.info(f"Created entity {entity_id}")
                return entity_id
```

Metrics collected but not exported in simple mode. In production, sends to monitoring.

### Principle 5: Clear Upgrade Boundaries

Document what changes between modes:

| Feature | Simple Mode | Production Mode |
|---------|-------------|-----------------|
| Storage | Neo4j only | PostgreSQL + Neo4j + ES |
| Provenance | Basic (parent/child) | PROV-O standard |
| Search | Keyword | Vector semantic |
| Query | Cypher | GraphRAG |
| Caching | None | Redis |
| Monitoring | Logs | Prometheus + Grafana |
| Backup | Manual export | Automated with retention |
| Multi-user | No | Yes |

### Principle 6: Education Through Code

Code as teaching tool:

```python
class Neo4jWorldModel(WorldModelStorage):
    """
    Simple world model implementation using Neo4j.

    EDUCATIONAL NOTE:
    This implementation prioritizes simplicity and understandability.
    In production (PolyglotWorldModel), we:
    - Use PostgreSQL for metadata (ACID guarantees)
    - Use Neo4j only for graph operations (specialized)
    - Use Elasticsearch for time-series events
    - Use vector DB for semantic search

    Each system optimized for its workload.

    See: docs/architecture/production_mode_architecture.md
    Paper: White et al. Section 3.2 "Polyglot Persistence"
    """
```

---

## Revised Phased Implementation

### Phase 0: Validate Assumptions + Architecture Planning (2 weeks)

**Goals:**
1. Validate user interest in persistent graphs
2. Create architectural foundation documents
3. Design abstraction layers for future compatibility

**Actions:**

**Week 1: User Validation**
1. Respond to GitHub Issue #4 (updated response below)
2. Survey existing users about workflows
3. Create user personas document
4. Identify priority features

**Week 2: Architecture Planning**
5. Create `docs/architecture/world_model_vision.md` - full Path 1 end state
6. Create `docs/architecture/design_principles.md` - explain key decisions
7. Design abstract interfaces (WorldModelStorage, etc.)
8. Document Simple Mode vs Production Mode split
9. Create migration path documentation

**Success Criteria:**
- 10+ users express interest
- Clear understanding of usage patterns
- Complete architecture vision documented
- Abstract interfaces designed

**Deliverables:**
- User requirements document
- Architecture vision document (Goal A)
- Design principles document (Goal C)
- Interface specifications

**If Validation Fails:**
Stop and focus on single-run research quality instead.

---

### Phase 1: Foundation with Future in Mind (3-4 weeks)

**Goals:**
1. Implement Simple Mode (persistent Neo4j)
2. Create abstraction layer for future production mode
3. Deliver immediate user value
4. Enable architectural learning

**Implementation:**

**Core Features** (2-3 weeks):
1. **Persistent Neo4j:**
   ```yaml
   volumes:
     - ~/.kosmos/neo4j_data:/data
   ```

2. **Accumulation Logic:**
   - Check for existing entities before creating
   - Merge duplicates intelligently
   - Update confidence scores with new evidence

3. **CLI Commands:**
   ```bash
   kosmos graph info          # Stats and health
   kosmos graph export FILE   # JSON/GraphML
   kosmos graph import FILE   # Load from file
   kosmos graph reset         # Clear and restart
   ```

4. **Project Tagging:**
   ```bash
   kosmos research "question" --project alzheimers
   ```

**Architectural Work** (1 week):
5. **Abstract Storage Layer:**
   ```python
   from kosmos.world_model.interface import WorldModelStorage
   # Not directly importing Neo4j classes
   ```

6. **Configuration Structure:**
   ```yaml
   world_model:
     mode: simple  # Future: production
     backend: neo4j
     path: ~/.kosmos/neo4j_data
   ```

7. **Observability Hooks:**
   - Metrics collection points (even if not exported)
   - Logging for provenance (even if basic)
   - Query timing instrumentation

8. **Separation of Concerns:**
   - Graph operations independent of business logic
   - Backend swappable without touching agents
   - Clear module boundaries

**Code Changes:**
- `kosmos/world_model/interface.py`: **NEW** - Abstract interfaces
- `kosmos/world_model/simple.py`: **NEW** - Neo4j implementation
- `kosmos/knowledge/graph.py`: **MODIFY** - Use new interfaces
- `kosmos/knowledge/graph_builder.py`: **MODIFY** - Add merge logic
- `kosmos/cli/graph_commands.py`: **NEW** - Management commands
- `kosmos/config.py`: **MODIFY** - Add world_model section
- `docker-compose.yml`: **MODIFY** - Named volumes

**Testing:**
- Unit tests for interface implementations
- Integration tests for full workflow
- Migration tests (export → import → verify)
- Performance baseline measurements

**Documentation:**
- `docs/guides/getting_started.md` (Goal B)
- `docs/guides/building_knowledge_base.md` (Goal B)
- `docs/architecture/simple_mode_architecture.md` (Goal C)
- `docs/tutorials/understanding_world_models.md` (Goal C)

**Success Criteria:**
- Knowledge accumulates correctly
- Export/import works flawlessly
- No regressions in existing features
- Architecture supports future production mode
- Documentation explains both what and why

**Deliverables:**
- Working Simple Mode
- Abstract interfaces
- Comprehensive documentation
- Test suite

---

### Phase 2: Curation Features (3-4 weeks)

**Goals:**
1. Give users control over accumulated knowledge
2. Implement quality management patterns
3. Demonstrate user-in-the-loop workflow

**Implementation:**

[Same as original document for Phase 2 content - verification, annotation, duplicates, quality analysis]

**Additional:**
- Document why curation matters (Goal C)
- Show how this scales to production (Goal A)
- Provide practical examples (Goal B)

---

### Phase 3: Multi-Project Support (2-3 weeks)

**Goals:**
1. Enable multiple research contexts
2. Demonstrate architectural flexibility
3. Provide real-world workflow support

**Implementation:**

[Same as original document for Phase 3 content - project management, isolated graphs, cross-project queries]

**Additional:**
- Explain project isolation patterns (Goal C)
- Show how production mode handles this differently (Goal A)
- Practical project management guide (Goal B)

---

### Phase 4: The Fork - Simple Refinement + Production Option (4-6 weeks)

**Goals:**
1. Refine Simple Mode based on feedback
2. Add Production Mode as opt-in advanced option
3. Maintain both paths going forward

**Phase 4A: Simple Mode Refinement (2 weeks)**

Based on user feedback from Phases 1-3:
- Performance optimizations
- UX improvements
- Bug fixes
- Documentation gaps

**Phase 4B: Production Mode Implementation (4 weeks)**

Add full polyglot architecture as optional advanced mode:

**Week 1: PostgreSQL Integration**
- Transaction management
- Metadata storage
- Schema design

**Week 2: Elasticsearch Integration**
- Provenance event tracking
- Time-series queries
- Full-text search

**Week 3: Vector DB Integration**
- ChromaDB (dev) / Pinecone (prod)
- Semantic search
- Similarity queries

**Week 4: Integration & Testing**
- GraphRAG query engine
- PROV-O provenance
- Production deployment guide

**Configuration:**
```yaml
world_model:
  mode: production  # Enable advanced features

  postgres:
    url: postgresql://localhost/kosmos
    pool_size: 10

  neo4j:
    url: bolt://localhost:7687

  elasticsearch:
    url: http://localhost:9200
    index: kosmos-events

  vector_db:
    provider: pinecone
    api_key: ${PINECONE_API_KEY}
    index: kosmos-entities
```

**Deployment Options:**

**Option 1: Docker Compose**
```bash
cd deployment/production
docker-compose up -d  # All services
kosmos init --mode production
```

**Option 2: Managed Services**
```bash
# AWS RDS PostgreSQL
# Neo4j AuraDB
# AWS Elasticsearch
# Pinecone Cloud
kosmos init --mode production --managed
```

**Success Criteria:**
- Simple mode remains default and works great
- Production mode available for power users
- Clear migration path documented
- Both modes tested and maintained

---

## Documentation Structure for "All Three" Goals

### Architecture Documentation (Goal A + C)

```
docs/architecture/
├── world_model_vision.md              # Full polyglot vision from research
├── design_principles.md               # Why decisions were made
├── simple_mode_architecture.md        # Phase 1-3 design and rationale
├── production_mode_architecture.md    # Phase 4 design and rationale
├── migration_paths.md                 # Moving between phases and modes
├── comparison_to_paper.md             # How we match/differ from Kosmos paper
├── decision_records/
│   ├── 001-abstract-storage-layer.md
│   ├── 002-neo4j-for-simple-mode.md
│   ├── 003-polyglot-for-production.md
│   └── 004-project-isolation-strategy.md
└── diagrams/
    ├── simple_mode_architecture.png
    ├── production_mode_architecture.png
    └── migration_flow.png
```

### User Guides (Goal B)

```
docs/guides/
├── getting_started.md                 # Quick start with simple mode
├── building_knowledge_base.md         # Using Phase 1 features
├── curating_your_graph.md             # Using Phase 2 features
├── managing_multiple_projects.md      # Using Phase 3 features
├── production_deployment.md           # Deploying production mode
├── migration_guide.md                 # Moving from simple to production
└── troubleshooting.md                 # Common issues and solutions
```

### Tutorials (Goal C)

```
docs/tutorials/
├── understanding_world_models.md      # Concepts and principles
├── from_ephemeral_to_persistent.md    # Evolution story
├── simple_vs_production_modes.md      # When to use which
├── scaling_architecture.md            # When and how to scale
├── custom_storage_backend.md          # Implementing interfaces
└── case_studies/
    ├── individual_researcher.md       # Simple mode success story
    ├── small_lab.md                   # Simple mode with team
    └── research_organization.md       # Production mode deployment
```

### API Documentation (Goal A + C)

```
docs/api/
├── world_model_interface.md           # Abstract interfaces
├── simple_mode_implementation.md      # Neo4j backend
├── production_mode_implementation.md  # Polyglot backend
├── configuration_reference.md         # All config options
└── cli_commands.md                    # Command reference
```

---

## Foundation for Follow-On Documents

This decision framework serves as the canonical foundation for three follow-on documents:

### requirements.md - What We Need to Build

**Derives from this document:**
- Phase 0-4 implementation descriptions → Functional requirements
- Success criteria → Acceptance criteria
- User validation questions → User stories
- "All Three" goals → Non-functional requirements

**Structure:**
```markdown
# World Model Feature Requirements

## Goals (from decision framework)
- Goal A: Faithful reproduction...
- Goal B: Practical tool...
- Goal C: Educational reference...

## Functional Requirements
### FR-1: Persistent Storage (Phase 1)
- FR-1.1: Neo4j data persists across runs
- FR-1.2: Export to JSON/GraphML
[...]

### FR-2: Curation (Phase 2)
[...]

## Non-Functional Requirements
### NFR-1: Performance
- Simple mode: < 1s query latency
- Production mode: < 100ms query latency

### NFR-2: Educational Value
- All major components documented with rationale
[...]

## User Stories
From validation (Phase 0):
- As a graduate student, I want...
[...]
```

**Decisions Made Here:**
- What features to build (Phases 1-4)
- Success metrics for each phase
- User workflows to support

**Decisions Deferred to requirements.md:**
- Detailed acceptance criteria for each feature
- Edge cases and error handling
- Performance targets for specific operations
- Accessibility and i18n requirements

---

### architecture.md - How We'll Structure the System

**Derives from this document:**
- Architectural principles → Design patterns
- Interface definitions → Component specifications
- Simple vs Production mode → Deployment architectures
- Phase progression → Evolution strategy

**Structure:**
```markdown
# World Model Architecture Specification

## System Context
[C4 context diagram]

## Architecture Patterns
### Pattern 1: Abstract Storage Layer
From decision framework "Architectural Principles #1"
[Detailed design]

### Pattern 2: Progressive Enhancement
[...]

## Component Design

### Simple Mode Architecture
#### Storage Component
- Interface: WorldModelStorage
- Implementation: Neo4jWorldModel
- Responsibilities:
  - Entity CRUD operations
  - Relationship management
  - Query execution
- Dependencies:
  - Neo4j Python driver
  - Pydantic for validation

[Detailed component specs for each interface]

### Production Mode Architecture
[Full polyglot design]

## Data Models
### Entity Schema
### Relationship Schema
### Provenance Schema

## API Specifications
### Python API
### CLI API
### REST API (if needed)

## Deployment Architectures
### Local Development
### Simple Mode Production
### Production Mode Production

## Security Considerations
## Performance Considerations
## Monitoring & Observability
```

**Decisions Made Here:**
- High-level architectural approach (interfaces, progressive enhancement)
- Simple vs Production mode split
- Major component boundaries

**Decisions Deferred to architecture.md:**
- Detailed component specifications
- Data model schemas
- API contracts
- Deployment topologies
- Security mechanisms
- Performance optimization strategies

---

### implementation.md - How We'll Actually Build It

**Derives from this document:**
- Phase timelines → Sprint planning
- Code change lists → Implementation tasks
- Success criteria → Testing requirements
- Migration paths → Upgrade procedures

**Structure:**
```markdown
# World Model Implementation Guide

## Phase 1: Foundation (Weeks 1-4)

### Sprint 1: Architecture Setup (Week 1)
**Tasks:**
1. Create kosmos/world_model/interface.py
   - Define WorldModelStorage ABC
   - Define ProvenanceTracker ABC
   - Define SemanticSearch ABC
   - Unit tests for interface contracts

2. Create kosmos/world_model/simple.py
   - Implement Neo4jWorldModel
   - Connection management
   - Error handling
   [...]

**Testing:**
- Unit tests: 90%+ coverage
- Integration tests: Happy path scenarios
- Performance: Baseline measurements

**Documentation:**
- Architecture doc updated
- API docs generated
- Tutorial: "Understanding Interfaces"

### Sprint 2: Persistence Implementation (Week 2)
[...]

## Testing Strategy
### Unit Testing
### Integration Testing
### Performance Testing
### Migration Testing

## Deployment Procedures
### Development Environment Setup
### CI/CD Pipeline
### Production Deployment (Simple Mode)
### Production Deployment (Production Mode)

## Migration Procedures
### Phase 1 to Phase 2
### Simple Mode to Production Mode

## Rollback Procedures

## Monitoring Setup

## Troubleshooting Guide
```

**Decisions Made Here:**
- Phase boundaries and timelines
- Major code modules to create/modify
- Testing approaches

**Decisions Deferred to implementation.md:**
- Specific implementation tasks
- Testing procedures
- Deployment scripts
- CI/CD configuration
- Monitoring setup details
- Operational procedures

---

## Success Criteria for "All Three" Goals

### Goal A: Faithful Reproduction of Production System

**Technical Fidelity:**
- ✅ Production mode implements polyglot persistence (PostgreSQL + Neo4j + ES + VectorDB)
- ✅ PROV-O standard provenance tracking
- ✅ GraphRAG query patterns
- ✅ Event-driven architecture patterns documented
- ✅ Handles same entity types and relationships as paper

**Scalability:**
- ✅ Simple mode proven to 10K entities, 50K relationships
- ✅ Production mode supports 100K+ entities, 500K+ relationships
- ✅ Query performance meets paper's requirements (< 100ms)

**Documentation:**
- ✅ Architecture documentation explains paper's approach
- ✅ Clear mapping between paper and implementation
- ✅ Production deployment guide for enterprise use

**Validation:**
- ✅ At least one organization deploys production mode
- ✅ Production mode matches paper's capabilities
- ✅ Academic citations reference the implementation

### Goal B: Practical Tool for Researchers

**Usability:**
- ✅ Simple mode works out-of-box (`pip install && kosmos init`)
- ✅ Clear documentation for common workflows
- ✅ Intuitive CLI commands
- ✅ Helpful error messages and guidance

**Adoption:**
- ✅ 50+ active users (regular commits to their graphs)
- ✅ 10+ users export and share knowledge graphs
- ✅ Positive feedback on usability (survey)
- ✅ Community contributions (issues, PRs, discussions)

**Reliability:**
- ✅ No data loss incidents
- ✅ Export/import works reliably
- ✅ Graceful handling of failures
- ✅ Good test coverage (90%+)

**Value Delivered:**
- ✅ Users report knowledge accumulation helps their research
- ✅ Users share graphs that others find useful
- ✅ Saves researcher time vs manual organization

### Goal C: Educational Reference Implementation

**Learning Resource:**
- ✅ Architecture documentation explains the why
- ✅ Code includes educational comments
- ✅ Tutorials show progression from simple to sophisticated
- ✅ Examples demonstrate patterns at multiple scales

**Academic Impact:**
- ✅ Students use for learning graph databases
- ✅ Researchers cite as reference implementation
- ✅ Teaching materials reference the project
- ✅ Conference talks/papers about the approach

**Knowledge Transfer:**
- ✅ Clear progression: ephemeral → persistent → polyglot
- ✅ Decision records explain architectural choices
- ✅ Comparison with paper explains differences
- ✅ Migration guides teach scaling decisions

**Community:**
- ✅ Contributors understand the architecture
- ✅ Good onboarding experience
- ✅ Architecture discussions in issues/discussions
- ✅ Knowledge base of why decisions were made

---

## Updated Timeline and Milestones

| Phase | Duration | Simple Mode | Production Mode | All Goals |
|-------|----------|-------------|-----------------|-----------|
| **Phase 0** | 2 weeks | Validation + Architecture docs | Same | Foundation (A+C) |
| **Phase 1** | 3-4 weeks | Working | - | B: Usable, C: Learning |
| **Phase 2** | 3-4 weeks | Enhanced | - | B: Control, C: Patterns |
| **Phase 3** | 2-3 weeks | Complete | - | B: Multi-project |
| **Phase 4** | 4-6 weeks | Refined | **Available** | A: Production option |
| **Ongoing** | - | Maintained | Maintained | All three goals |

**Key Milestones:**

- **Week 2:** Architecture vision documented (Goals A+C)
- **Week 6:** Simple mode usable (Goal B) ✓
- **Month 3:** Full simple mode features (Goal B) ✓
- **Month 5:** Production mode available (Goal A) ✓
- **Month 6+:** Both modes actively used and maintained (All goals) ✓

---

## Updated GitHub Issue #4 Response

```markdown
Hi @WangRentu,

Thank you for the kind words and excellent question!

**Current State:**
Kosmos doesn't include a pre-built world-model knowledge graph. Instead, it builds
knowledge graphs dynamically during research runs using Neo4j. Currently these graphs
are ephemeral (reset each run), though all infrastructure exists to make them persistent.

**What We're Building:**
We're implementing persistent world models through a phased approach that achieves
three simultaneous goals:

1. **Practical tool you can use TODAY** - Simple persistent graphs that accumulate knowledge
2. **Educational reference** - Clear documentation showing how to build and scale world models
3. **Production-grade system** - Path to full architecture matching the Kosmos paper

**Timeline:**
- **6 weeks:** Simple mode working (persistent Neo4j, export/import, project management)
- **3 months:** Full curation features (verification, quality analysis, duplicate management)
- **5 months:** Production mode available (optional polyglot architecture for enterprise scale)

**How This Helps You:**
- Build domain expertise over weeks/months
- Export and share knowledge graphs
- Choose simple local deployment OR enterprise production deployment
- Learn architectural patterns through documented progression

**Your Input Matters:**
Before finalizing the design, we'd love your feedback:

1. Would you use this for ongoing research, or one-time investigations?
2. One unified graph, or separate graphs per project?
3. Individual use, or collaboration with colleagues?
4. What would make a sample/snapshot most useful to you?

**Next Steps:**
We're in Phase 0 (architecture planning and validation). Your feedback directly shapes
the implementation. Want to influence the design? We'd love your thoughts in the
discussion thread we're creating.

**See the Code:**
- `kosmos/knowledge/graph.py` - Current Neo4j implementation
- `docs/planning/world_model_implementation_decision_framework.md` - Full design

Thanks for your interest! This is exactly the kind of feature that makes Kosmos more powerful.
```

---

## Conclusion: The "All Three" Strategy

### Why This Works

**Goal A (Faithful Reproduction):**
- Production mode implements full paper architecture
- Clear documentation of polyglot persistence
- Proven path from simple to sophisticated
- **Achieved:** Month 5, maintained ongoing

**Goal B (Practical Tool):**
- Simple mode works immediately
- Low barrier to entry
- 90% of users never need complexity
- **Achieved:** Week 6, enhanced months 2-3

**Goal C (Educational Reference):**
- Architecture docs explain the journey
- Code shows patterns at multiple scales
- Tutorials teach scaling decisions
- **Achieved:** Ongoing throughout

### The Core Insight

**You don't have to choose.** With proper architectural discipline:
- Build the simple path first (Goal B)
- Document the sophisticated destination (Goal A)
- Explain the journey between them (Goal C)

Each phase delivers standalone value while building toward the complete vision.

### Updated Recommendation

**Implement Path 3 with Architectural Discipline:**

**Weeks 1-2:**
- Validate user interest
- Create architecture vision documents
- Design abstract interfaces

**Weeks 3-6:**
- Implement Simple Mode with abstraction layer
- Deliver immediate user value
- Document both what and why

**Months 2-3:**
- Add curation and multi-project features
- Refine based on user feedback
- Enhance educational materials

**Months 4-5:**
- Refine simple mode
- Add production mode as opt-in
- Complete the full vision

**Ongoing:**
- Maintain both simple and production modes
- Grow user community
- Enhance based on real usage

### The Final Answer

**"What is jimmc414/Kosmos trying to be?"**

**ALL THREE:**
- A faithful reproduction of FutureHouse's vision (eventually)
- A practical tool researchers use daily (immediately)
- An educational reference showing the way (continuously)

**Different users get different value:**
- Students learn from the architecture
- Researchers use the simple mode
- Organizations deploy the production mode

**Same codebase, same vision, multiple audiences.**

### Next Steps

1. **This week:** Respond to Issue #4, validate interest
2. **Weeks 1-2:** Architecture planning and documentation
3. **Weeks 3-6:** Implement Phase 1 (Simple Mode + abstractions)
4. **Month 2+:** Iterate based on user feedback
5. **Month 5:** Production mode available for those who need it

**Start with abstraction. Build for today. Plan for tomorrow.**

---

**Document Status:** Ready for requirements.md, architecture.md, implementation.md derivation
**Recommended Review:** Architecture team, UX team, Community feedback
**Next Document:** requirements.md (derive user stories and acceptance criteria)

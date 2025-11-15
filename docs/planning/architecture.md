# World Model Architecture Specification

**Version:** 1.0
**Date:** November 2025
**Status:** Draft
**Owner:** Kosmos Development Team
**Derives From:**
- `docs/planning/requirements.md`
- `docs/planning/world_model_implementation_decision_framework.md`
- `docs/planning/objective.md`

---

## Table of Contents

1. [Document Overview](#1-document-overview)
2. [System Context](#2-system-context)
3. [Architecture Principles](#3-architecture-principles)
4. [Architecture Patterns](#4-architecture-patterns)
5. [Component Architecture](#5-component-architecture)
6. [Interface Specifications](#6-interface-specifications)
7. [Data Models](#7-data-models)
8. [Database Schemas](#8-database-schemas)
9. [API Specifications](#9-api-specifications)
10. [Data Flow Diagrams](#10-data-flow-diagrams)
11. [Deployment Architectures](#11-deployment-architectures)
12. [Security Architecture](#12-security-architecture)
13. [Performance Architecture](#13-performance-architecture)
14. [Observability Architecture](#14-observability-architecture)
15. [Migration Architecture](#15-migration-architecture)
16. [Architecture Decision Records](#16-architecture-decision-records)
17. [Technology Stack](#17-technology-stack)
18. [Comparison to Kosmos Paper](#18-comparison-to-kosmos-paper)

---

## 1. Document Overview

### 1.1 Purpose

This document defines the complete technical architecture for persistent world models in Kosmos. It specifies:

- **All components** and their interactions
- **All interfaces** and their contracts
- **All data models** and schemas
- **All deployment options** and configurations
- **All technical decisions** and their rationale

This architecture serves three audiences simultaneously:

- **Goal A (Faithful Reproduction):** Production Mode implements the polyglot architecture from the Kosmos paper
- **Goal B (Practical Tool):** Simple Mode provides an immediately usable implementation
- **Goal C (Educational Reference):** Documentation explains patterns and teaches scaling decisions

### 1.2 Scope

**In Scope:**
- Persistent knowledge graph storage (Simple and Production modes)
- Export/import functionality
- Multi-project support
- Curation features (verification, annotation, quality analysis)
- Provenance tracking (basic and PROV-O standard)
- Query interfaces (direct and GraphRAG)

**Out of Scope:**
- Real-time collaborative editing
- Web-based GUI
- Mobile applications
- Automated hypothesis generation from graphs
- LLM fine-tuning on accumulated knowledge

### 1.3 Architecture Goals

1. **Abstraction:** Clean interfaces allow swapping implementations without breaking user code
2. **Progressive Enhancement:** Same interfaces, different capabilities based on mode
3. **Observability:** Instrumentation built in from Day 1
4. **Educational Value:** Code and documentation teach architectural patterns
5. **Migration Path:** Clear upgrade path from Simple to Production mode
6. **No Vendor Lock-in:** Support both self-hosted and managed services

### 1.4 Key Architecture Decisions

| Decision | Rationale | ADR |
|----------|-----------|-----|
| Abstract storage interfaces | Enable mode switching without code changes | ADR-001 |
| Neo4j for Simple Mode | Single database, easy setup, sufficient for 90% of users | ADR-002 |
| Polyglot for Production Mode | Each database optimized for its workload, matches paper | ADR-003 |
| Project-based isolation | Graph namespaces prevent cross-contamination | ADR-004 |
| PROV-O standard provenance | Publication-quality, interoperable, industry standard | ADR-005 |
| GraphRAG query pattern | Combines graph + semantic search + LLM for best results | ADR-006 |

---

## 2. System Context

### 2.1 C4 Context Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Researcher                              │
│                    (Graduate Student,                           │
│                     Lab Member, etc.)                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ Runs research queries
                         │ Curates knowledge
                         │ Manages projects
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                                                                  │
│                      Kosmos AI Scientist                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          World Model (this architecture)                 │  │
│  │                                                           │  │
│  │  ┌─────────────┐              ┌─────────────┐           │  │
│  │  │ Simple Mode │              │Production   │           │  │
│  │  │  (Neo4j)    │     OR       │Mode         │           │  │
│  │  │             │              │(Polyglot)   │           │  │
│  │  └─────────────┘              └─────────────┘           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            Existing Kosmos Components                    │  │
│  │  - LLM Providers (Anthropic, OpenAI)                     │  │
│  │  - Agents (Literature, Experiment, Analysis)             │  │
│  │  - Experiment Templates                                  │  │
│  │  - CLI Interface                                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ Uses for research
                         │ Stores findings in
                         │
         ┌───────────────┴───────────────┬───────────────────────┐
         │                               │                       │
         ▼                               ▼                       ▼
┌─────────────────┐          ┌─────────────────┐    ┌──────────────────┐
│  Literature     │          │  Lab Notebooks  │    │ Experiment Data  │
│  Databases      │          │                 │    │                  │
│  (PubMed, etc.) │          │                 │    │                  │
└─────────────────┘          └─────────────────┘    └──────────────────┘
```

### 2.2 World Model Position in Kosmos

The World Model sits at the **core** of Kosmos, serving as:

1. **Memory System:** Persistent storage of accumulated knowledge
2. **Knowledge Base:** Queryable repository of entities and relationships
3. **Context Provider:** Supplies relevant information to agents
4. **Provenance Tracker:** Records how knowledge was derived
5. **Project Organizer:** Separates knowledge by research topic

### 2.3 Integration Points

**Inbound (What Uses World Model):**
- Literature agents store paper metadata, concepts, citations
- Experiment agents store hypotheses, results, analyses
- Analysis agents query existing knowledge to inform new research
- CLI commands allow user curation and exploration

**Outbound (What World Model Uses):**
- Neo4j database for graph storage (Simple Mode)
- PostgreSQL for metadata (Production Mode)
- Elasticsearch for provenance events (Production Mode)
- Vector database for semantic search (Production Mode)
- LLM providers for GraphRAG queries (Production Mode)

### 2.4 External Systems

| System | Purpose | Mode | Protocol |
|--------|---------|------|----------|
| Neo4j | Graph storage and queries | Both | Bolt (port 7687) |
| PostgreSQL | Metadata and transactions | Production | PostgreSQL wire (port 5432) |
| Elasticsearch | Provenance event indexing | Production | HTTP REST (port 9200) |
| ChromaDB | Vector embeddings (local) | Production | Python API |
| Pinecone | Vector embeddings (cloud) | Production | HTTPS API |
| Docker | Container orchestration | Both | Docker socket |

---

## 3. Architecture Principles

### Principle 1: Interface-Based Design

**Statement:** Every major component has a clean abstract interface that allows implementation swapping.

**Rationale:**
- Enables Simple Mode and Production Mode to coexist
- Allows future storage backends without rewriting business logic
- Makes testing easier (mock implementations)
- Teaches good software engineering practices (Goal C)

**Implementation:**
- All storage operations go through `WorldModelStorage` ABC
- All provenance operations go through `ProvenanceTracker` ABC
- All semantic search through `SemanticSearch` ABC
- All queries through `QueryEngine` ABC

**Example:**
```python
# Business logic depends on interface, not implementation
def add_paper(paper: Paper, storage: WorldModelStorage):
    entity = storage.add_entity(paper)  # Works with ANY implementation
    return entity
```

**Educational Note:**
This is the **Dependency Inversion Principle** from SOLID. High-level modules (agents) don't depend on low-level modules (Neo4j). Both depend on abstractions (interfaces).

---

### Principle 2: Progressive Enhancement

**Statement:** Features work at a basic level in Simple Mode and enhance in Production Mode, using the same API.

**Rationale:**
- Users get immediate value without complexity
- Upgrade path is natural and clear
- Same user experience, better performance/features
- No rewrite needed when scaling up

**Implementation:**

| Feature | Simple Mode | Production Mode |
|---------|-------------|-----------------|
| Storage | Neo4j | PostgreSQL + Neo4j + ES |
| Search | Keyword | Vector semantic |
| Provenance | Parent/child | PROV-O standard |
| Query | Direct Cypher | GraphRAG with LLM |
| Caching | None | Redis |
| Monitoring | Logs | Prometheus + Grafana |

**Example:**
```python
# Same API, different capabilities
def find_similar(query: str, model: WorldModel):
    # Simple Mode: keyword matching in Neo4j
    # Production Mode: vector similarity in Pinecone
    return model.semantic_search.find_similar(query, top_k=10)
```

**Educational Note:**
This is **Progressive Enhancement** from web development applied to backend architecture. Start with working basics, layer on sophistication.

---

### Principle 3: Configuration-Driven Mode Selection

**Statement:** Single codebase supports multiple deployment modes via configuration, not code branches.

**Rationale:**
- Eliminates code duplication
- Reduces maintenance burden
- Makes mode selection explicit and documented
- Allows hybrid deployments (some users simple, some production)

**Implementation:**
```yaml
# config.yaml
world_model:
  mode: simple  # or "production"

  simple:
    backend: neo4j
    path: ~/.kosmos/neo4j_data

  production:
    postgres:
      url: postgresql://localhost/kosmos
    neo4j:
      url: bolt://localhost:7687
    elasticsearch:
      url: http://localhost:9200
    vector_db:
      provider: pinecone
      api_key: ${PINECONE_API_KEY}
```

**Factory Pattern:**
```python
def create_world_model(config: Config) -> WorldModel:
    if config.world_model.mode == "simple":
        storage = Neo4jWorldModel(config.simple)
    else:
        storage = PolyglotWorldModel(config.production)
    return WorldModel(storage)
```

**Educational Note:**
This is the **Strategy Pattern**. Algorithm (storage implementation) is selected at runtime based on configuration.

---

### Principle 4: Observability from Day 1

**Statement:** All operations are instrumented with metrics, logging, and tracing hooks from the first implementation.

**Rationale:**
- Debugging is easier with good observability
- Performance issues visible early
- Production Mode can enable full monitoring
- Teaches importance of observability (Goal C)

**Implementation:**

**Metrics Collection (all modes):**
```python
class WorldModel:
    def add_entity(self, entity: Entity) -> str:
        with self.metrics.timer('world_model.add_entity'):
            entity_id = self.storage.add_entity(entity)
            self.metrics.increment('world_model.entities_created')
            return entity_id
```

**Logging (all modes):**
```python
logger.info(f"Created entity {entity_id}", extra={
    "entity_type": entity.type,
    "project": entity.project,
    "confidence": entity.confidence
})
```

**Tracing (Production Mode):**
```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

def query(self, q: str) -> Results:
    with tracer.start_as_current_span("world_model.query") as span:
        span.set_attribute("query", q)
        results = self._execute_query(q)
        span.set_attribute("result_count", len(results))
        return results
```

**Educational Note:**
The **three pillars of observability** are metrics (what's happening), logs (detailed events), and traces (request flow). All three are essential for production systems.

---

### Principle 5: Clear Upgrade Boundaries

**Statement:** Document exactly what changes when moving between modes, with explicit migration paths.

**Rationale:**
- Users know what they're getting into
- No surprises during upgrade
- Can make informed cost/benefit decisions
- Teaches scaling decision-making (Goal C)

**Implementation:**

**Feature Matrix:**
| Capability | Simple Mode | Production Mode | Migration Impact |
|------------|-------------|-----------------|------------------|
| Max Entities | 10K | 100K+ | Export/Import |
| Query Latency | <1s | <100ms | None (API same) |
| Semantic Search | Keyword | Vector | Config change |
| Provenance | Basic | PROV-O | Data transformation |
| Multi-user | No | Yes | Add auth config |
| Infrastructure | 1 container | 4+ services | Deploy new services |

**Migration Command:**
```bash
kosmos migrate simple-to-production \
  --validate          # Check compatibility
  --transform         # Transform data
  --verify            # Verify result
```

**Educational Note:**
**Non-functional requirements** (performance, scalability) often drive architecture changes more than functional requirements. Document trade-offs explicitly.

---

### Principle 6: Education Through Code

**Statement:** Code includes explanatory comments, docstrings reference the paper, and implementations teach patterns.

**Rationale:**
- Serves Goal C (Educational Reference)
- Helps contributors understand the system
- Makes the "why" as clear as the "what"
- Provides learning resource for graph databases

**Implementation:**

**Docstrings with Context:**
```python
class Neo4jWorldModel(WorldModelStorage):
    """
    Simple world model implementation using Neo4j.

    DESIGN RATIONALE:
    Uses a single Neo4j database for all storage needs. This prioritizes
    simplicity and ease of setup over specialized performance.

    In Production Mode (PolyglotWorldModel), we:
    - Use PostgreSQL for metadata (ACID guarantees)
    - Use Neo4j only for graph operations (specialized workload)
    - Use Elasticsearch for time-series provenance events
    - Use vector DB for semantic search

    Each database is optimized for its specific workload, following the
    Polyglot Persistence pattern described in the Kosmos paper.

    PAPER REFERENCE:
    White et al. Section 3.2 "Structured World Models"

    TRADE-OFFS:
    - Pros: Simple setup, one database to manage, sufficient for 90% of users
    - Cons: Limited to ~10K entities, slower than specialized systems

    WHEN TO UPGRADE:
    Consider Production Mode when:
    - Graph exceeds 10K entities
    - Query latency >1 second
    - Need semantic search beyond keyword matching
    - Require publication-quality provenance

    See: docs/architecture/production_mode_architecture.md
    """
```

**Inline Comments Explaining Rationale:**
```python
def merge_entities(self, entity1_id: str, entity2_id: str) -> str:
    # IMPORTANT: Preserve ALL relationships when merging
    # This maintains graph connectivity and prevents orphaned nodes

    # Strategy: Move all relationships to entity1, then delete entity2
    # Alternative considered: Create new merged entity
    # Decision: Preserve entity1 ID for stable references

    relationships = self.get_relationships(entity2_id)
    for rel in relationships:
        self.move_relationship(rel, from_entity=entity2_id, to_entity=entity1_id)

    # Merge properties using confidence-weighted averaging
    # Higher confidence values have more weight in final result
    merged_props = self._confidence_weighted_merge(entity1_id, entity2_id)

    # Record provenance: entity1 now derived from entity1 + entity2
    self.provenance.record_merge(entity1_id, [entity1_id, entity2_id])

    return entity1_id
```

**Educational Note:**
Good code should be **self-documenting**, but complex decisions need explicit explanation. Comment the **why**, not the **what**.

---

## 4. Architecture Patterns

### Pattern 1: Abstract Storage Layer

**Intent:** Decouple business logic from storage implementation details.

**Structure:**
```
┌─────────────────────────────────────────────────────┐
│           Kosmos Agents (Business Logic)            │
│   (Literature Agent, Experiment Agent, etc.)        │
└───────────────────────┬─────────────────────────────┘
                        │ depends on
                        ▼
        ┌───────────────────────────────┐
        │   WorldModelStorage (ABC)     │ ◄──── Interface
        │   + add_entity()              │
        │   + get_entity()              │
        │   + query()                   │
        │   + add_relationship()        │
        └───────────────────────────────┘
                        △
            ┌───────────┴───────────┐
            │                       │
┌───────────▼────────────┐  ┌──────▼──────────────────┐
│   Neo4jWorldModel      │  │  PolyglotWorldModel     │
│   (Simple Mode)        │  │  (Production Mode)      │
│                        │  │                         │
│   - Uses Neo4j only    │  │  - PostgreSQL: metadata │
│   - Direct Cypher      │  │  - Neo4j: graph ops     │
│   - Keyword search     │  │  - ES: provenance       │
│                        │  │  - VectorDB: semantic   │
└────────────────────────┘  └─────────────────────────┘
```

**Participants:**
- **WorldModelStorage (ABC):** Abstract interface defining all storage operations
- **Neo4jWorldModel:** Simple implementation using only Neo4j
- **PolyglotWorldModel:** Production implementation using 4 specialized databases
- **Agents:** Business logic that uses storage without knowing implementation

**Collaborations:**
1. Agent calls `storage.add_entity(entity)`
2. Storage implementation handles details (which DB, how to store, etc.)
3. Agent receives entity_id, doesn't know/care about underlying storage

**Consequences:**
- ✅ Can swap implementations without changing agent code
- ✅ Easy to test with mock implementations
- ✅ Clear separation of concerns
- ❌ Slight performance overhead from abstraction
- ❌ Must design interface carefully to support all implementations

**Implementation Considerations:**
- Interface must be rich enough for Production Mode features
- Interface must not leak implementation details
- All methods must work reasonably in both modes
- Error handling consistent across implementations

---

### Pattern 2: Progressive Enhancement

**Intent:** Provide basic functionality immediately, with optional sophisticated features.

**Structure:**
```
User calls same API
       │
       ▼
┌──────────────────────────┐
│   WorldModel             │
│   + semantic_search()    │◄────── Same method signature
└──────────────────────────┘
       │
       │ delegates based on config.mode
       │
       ├────────────────────┬────────────────────┐
       │                    │                    │
Simple Mode          Production Mode      (config driven)
       │                    │
       ▼                    ▼
Keyword Search      Vector Similarity
in Neo4j            in Pinecone
       │                    │
       ▼                    ▼
Results             Results
(good enough)       (better quality)
```

**Key Examples:**

**Semantic Search:**
```python
class WorldModel:
    def find_similar(self, query: str, top_k: int = 10) -> List[Entity]:
        """Find entities similar to query.

        Simple Mode: Keyword matching in Neo4j (Lucene index)
        Production Mode: Vector similarity in Pinecone (embeddings)

        Same API, progressively better results.
        """
        return self.semantic_search.find_similar(query, top_k)

class KeywordSearch(SemanticSearch):
    """Simple Mode: Basic keyword matching."""
    def find_similar(self, query: str, top_k: int) -> List[Entity]:
        # Use Neo4j full-text index
        return self.neo4j.query(f"MATCH (n) WHERE n.text CONTAINS '{query}' RETURN n LIMIT {top_k}")

class VectorSearch(SemanticSearch):
    """Production Mode: Semantic vector search."""
    def find_similar(self, query: str, top_k: int) -> List[Entity]:
        # Embed query and find nearest neighbors in vector space
        embedding = self.embedder.embed(query)
        return self.vector_db.query(embedding, top_k=top_k)
```

**Provenance Tracking:**
```python
class BasicProvenance(ProvenanceTracker):
    """Simple Mode: Parent/child relationships."""
    def record_derivation(self, output: str, inputs: List[str]):
        for input_id in inputs:
            self.neo4j.add_relationship(input_id, "DERIVED", output)

class PROVOProvenance(ProvenanceTracker):
    """Production Mode: PROV-O standard provenance."""
    def record_derivation(self, output: str, inputs: List[str]):
        # Create PROV-O entity, activity, and derivation records
        activity_id = self.create_activity()
        for input_id in inputs:
            self.record_usage(activity_id, input_id)
        self.record_generation(output, activity_id)
        # Index in Elasticsearch for time-series queries
        self.elasticsearch.index_event({
            "type": "derivation",
            "output": output,
            "inputs": inputs,
            "activity": activity_id,
            "timestamp": datetime.now()
        })
```

**Consequences:**
- ✅ Users get value immediately (Simple Mode)
- ✅ Clear upgrade path when needed
- ✅ No code changes when upgrading
- ✅ Pay complexity cost only when you need the features
- ❌ Must maintain multiple implementations
- ❌ Must ensure parity in fundamental features

---

### Pattern 3: Factory Pattern for Mode Selection

**Intent:** Create appropriate implementations based on configuration.

**Structure:**
```python
class WorldModelFactory:
    """Factory for creating world model instances based on configuration."""

    @staticmethod
    def create(config: Config) -> WorldModel:
        """Create world model instance.

        Selects Simple or Production mode based on config.world_model.mode.
        """
        if config.world_model.mode == "simple":
            return WorldModelFactory._create_simple_mode(config)
        elif config.world_model.mode == "production":
            return WorldModelFactory._create_production_mode(config)
        else:
            raise ValueError(f"Unknown mode: {config.world_model.mode}")

    @staticmethod
    def _create_simple_mode(config: Config) -> WorldModel:
        """Create Simple Mode world model."""
        # Single Neo4j backend
        neo4j = Neo4jClient(config.simple.backend)

        # Simple implementations
        storage = Neo4jWorldModel(neo4j)
        provenance = BasicProvenance(neo4j)
        semantic = KeywordSearch(neo4j)
        query_engine = DirectQueryEngine(neo4j)

        return WorldModel(
            storage=storage,
            provenance=provenance,
            semantic_search=semantic,
            query_engine=query_engine,
            config=config
        )

    @staticmethod
    def _create_production_mode(config: Config) -> WorldModel:
        """Create Production Mode world model."""
        # Multiple specialized backends
        postgres = PostgresClient(config.production.postgres.url)
        neo4j = Neo4jClient(config.production.neo4j.url)
        elasticsearch = ElasticsearchClient(config.production.elasticsearch.url)
        vector_db = VectorDBClient(config.production.vector_db)

        # Production implementations
        storage = PolyglotWorldModel(postgres, neo4j, elasticsearch, vector_db)
        provenance = PROVOProvenance(elasticsearch, postgres)
        semantic = VectorSearch(vector_db)
        query_engine = GraphRAGQueryEngine(neo4j, vector_db, llm_client)

        return WorldModel(
            storage=storage,
            provenance=provenance,
            semantic_search=semantic,
            query_engine=query_engine,
            config=config
        )
```

**Consequences:**
- ✅ Single point of creation logic
- ✅ Easy to add new modes
- ✅ Configuration-driven (no code changes)
- ❌ Factory can become complex with many modes
- ❌ Must validate configuration before creating instances

---

### Pattern 4: Repository Pattern for Data Access

**Intent:** Encapsulate data access logic, providing collection-like interface.

**Structure:**
```python
class EntityRepository:
    """Repository for Entity objects.

    Provides collection-like interface abstracting storage details.
    """

    def __init__(self, storage: WorldModelStorage):
        self.storage = storage

    def add(self, entity: Entity) -> str:
        """Add entity to repository."""
        return self.storage.add_entity(entity)

    def get(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.storage.get_entity(entity_id)

    def find_by_type(self, entity_type: str) -> List[Entity]:
        """Find all entities of given type."""
        return self.storage.query(f"type = '{entity_type}'")

    def find_by_project(self, project: str) -> List[Entity]:
        """Find all entities in project."""
        return self.storage.query(f"project = '{project}'")

    def update(self, entity: Entity):
        """Update existing entity."""
        self.storage.update_entity(entity)

    def delete(self, entity_id: str):
        """Delete entity."""
        self.storage.delete_entity(entity_id)

class RelationshipRepository:
    """Repository for Relationship objects."""
    # Similar pattern for relationships
```

**Usage:**
```python
# Agent code works with repositories, not raw storage
entity_repo = EntityRepository(storage)

# Add paper to knowledge graph
paper_entity = Entity(type="Paper", properties={...})
paper_id = entity_repo.add(paper_entity)

# Find all papers in this project
papers = entity_repo.find_by_type("Paper")
for paper in papers:
    if paper.project == current_project:
        # process paper
```

**Consequences:**
- ✅ Higher-level abstraction than raw storage
- ✅ Can add business logic (validation, caching)
- ✅ Testable with mock repositories
- ❌ Another layer of abstraction
- ❌ Can lead to N+1 query problems if not careful

---

### Pattern 5: Command Pattern for Migrations

**Intent:** Encapsulate migration operations as objects for undo/redo support.

**Structure:**
```python
class MigrationCommand(ABC):
    """Abstract migration command."""

    @abstractmethod
    def execute(self) -> MigrationResult:
        """Execute migration."""
        pass

    @abstractmethod
    def rollback(self):
        """Rollback migration if it fails."""
        pass

    @abstractmethod
    def validate(self) -> ValidationResult:
        """Validate preconditions before execution."""
        pass

class SimpleToProductionMigration(MigrationCommand):
    """Migrate from Simple Mode to Production Mode."""

    def __init__(self, source: Neo4jWorldModel, target: PolyglotWorldModel):
        self.source = source
        self.target = target
        self.backup_path = None

    def validate(self) -> ValidationResult:
        """Check if migration can proceed."""
        # Check source has compatible schema
        # Check target is empty or can merge
        # Check sufficient disk space
        return ValidationResult(success=True)

    def execute(self) -> MigrationResult:
        """Execute migration."""
        # 1. Create backup
        self.backup_path = self.source.export("/tmp/migration_backup.json")

        # 2. Read all entities from Simple Mode
        entities = self.source.get_all_entities()

        # 3. Transform to Production Mode format
        for entity in entities:
            # Add to PostgreSQL
            self.target.postgres.insert_entity(entity)
            # Add to Neo4j
            self.target.neo4j.create_node(entity)
            # Add to Elasticsearch (provenance)
            self.target.elasticsearch.index_entity(entity)
            # Add to Vector DB (embedding)
            self.target.vector_db.embed(entity)

        return MigrationResult(success=True, entities_migrated=len(entities))

    def rollback(self):
        """Rollback if migration fails."""
        if self.backup_path:
            # Restore from backup
            self.source.import_data(self.backup_path)
```

**Usage:**
```python
migration = SimpleToProductionMigration(source=simple_model, target=production_model)

# Validate first
validation = migration.validate()
if not validation.success:
    print(f"Cannot migrate: {validation.errors}")
    return

# Execute with rollback on failure
try:
    result = migration.execute()
    print(f"Migrated {result.entities_migrated} entities")
except Exception as e:
    print(f"Migration failed: {e}")
    migration.rollback()
```

**Consequences:**
- ✅ Migrations are first-class objects
- ✅ Can validate before executing
- ✅ Built-in rollback support
- ✅ Can compose multiple migrations
- ❌ More complex than simple functions
- ❌ Rollback may not always be possible

---

## 5. Component Architecture

### 5.1 C4 Container Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        Kosmos Application                        │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    CLI Layer                               │ │
│  │  - kosmos graph info/export/import/reset                   │ │
│  │  - kosmos research (triggers agents)                       │ │
│  │  - kosmos migrate                                          │ │
│  └────────────────┬───────────────────────────────────────────┘ │
│                   │                                              │
│  ┌────────────────▼───────────────────────────────────────────┐ │
│  │                 Agent Layer                                │ │
│  │  - LiteratureAgent    - ExperimentAgent                    │ │
│  │  - AnalysisAgent      - HypothesisAgent                    │ │
│  └────────────────┬───────────────────────────────────────────┘ │
│                   │ uses                                         │
│  ┌────────────────▼───────────────────────────────────────────┐ │
│  │              WorldModel (Facade)                           │ │
│  │  + add_entity()      + query()                             │ │
│  │  + find_similar()    + get_provenance()                    │ │
│  └─┬────────────────────────────────────────────────────────┬─┘ │
│    │                                                          │   │
│    │ delegates to                                             │   │
│    │                                                          │   │
│  ┌─▼──────────────────┐   ┌──────────────────┐  ┌───────────▼─┐ │
│  │ WorldModelStorage  │   │ ProvenanceTracker│  │SemanticSearch│ │
│  │   (interface)      │   │   (interface)    │  │  (interface) │ │
│  └─┬──────────────────┘   └─┬────────────────┘  └─┬───────────┘ │
│    │                        │                     │             │
│    │ implemented by         │                     │             │
│    │                        │                     │             │
│  ┌─▼──────────────────┐   ┌─▼────────────────┐  ┌─▼───────────┐ │
│  │ Neo4jWorldModel    │   │BasicProvenance   │  │KeywordSearch │ │
│  │    (Simple)        │   │   (Simple)       │  │  (Simple)    │ │
│  └────────────────────┘   └──────────────────┘  └──────────────┘ │
│         OR                        OR                    OR        │
│  ┌────────────────────┐   ┌──────────────────┐  ┌──────────────┐ │
│  │ PolyglotWorldModel │   │PROVOProvenance   │  │VectorSearch  │ │
│  │   (Production)     │   │  (Production)    │  │ (Production) │ │
│  └────────────────────┘   └──────────────────┘  └──────────────┘ │
└──────────────────────────────────────────────────────────────────┘
         │                        │                      │
         │ uses                   │ uses                 │ uses
         ▼                        ▼                      ▼
┌─────────────────┐      ┌────────────────┐    ┌──────────────────┐
│     Neo4j       │      │ Elasticsearch  │    │   Vector DB      │
│   (Container)   │      │  (Container)   │    │  (Pinecone/      │
│                 │      │                │    │   ChromaDB)      │
└─────────────────┘      └────────────────┘    └──────────────────┘
```

### 5.2 Component Descriptions

#### 5.2.1 WorldModel (Facade)

**Purpose:** Main entry point for all world model operations.

**Responsibilities:**
- Coordinate between storage, provenance, search, and query components
- Enforce business rules (validation, authorization)
- Manage transactions across multiple backends (Production Mode)
- Provide simplified API to agents

**Dependencies:**
- WorldModelStorage (required)
- ProvenanceTracker (required)
- SemanticSearch (required)
- QueryEngine (required)
- Config (required)
- MetricsCollector (optional)
- Logger (required)

**Public API:**
```python
class WorldModel:
    def add_entity(self, entity: Entity) -> str
    def get_entity(self, entity_id: str) -> Optional[Entity]
    def update_entity(self, entity: Entity)
    def delete_entity(self, entity_id: str)
    def add_relationship(self, source_id: str, rel_type: str, target_id: str) -> str
    def query(self, query_str: str, **kwargs) -> QueryResult
    def find_similar(self, text: str, top_k: int) -> List[Entity]
    def get_provenance(self, entity_id: str) -> ProvenanceChain
    def export_project(self, project: str, format: str) -> Path
    def import_data(self, filepath: Path, project: str)
```

**Configuration:**
```yaml
world_model:
  mode: simple  # or production
  validation:
    enabled: true
    strict: false  # Strict mode rejects invalid data
  caching:
    enabled: false  # Enable in production
    ttl: 3600
```

---

#### 5.2.2 WorldModelStorage (Interface)

**Purpose:** Abstract interface for all storage operations.

**Implementations:**
- `Neo4jWorldModel` (Simple Mode)
- `PolyglotWorldModel` (Production Mode)

**Contract:**
```python
class WorldModelStorage(ABC):
    """Abstract storage interface."""

    @abstractmethod
    def add_entity(self, entity: Entity) -> str:
        """Add entity, return ID.

        Implementations MUST:
        - Generate unique ID if not provided
        - Validate entity schema
        - Check for duplicates
        - Store all properties
        - Return stable ID
        """
        pass

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID, return None if not found."""
        pass

    @abstractmethod
    def update_entity(self, entity: Entity):
        """Update existing entity.

        Implementations MUST:
        - Verify entity exists
        - Preserve entity ID
        - Update properties atomically
        - Raise EntityNotFoundError if missing
        """
        pass

    @abstractmethod
    def delete_entity(self, entity_id: str):
        """Delete entity and all relationships."""
        pass

    @abstractmethod
    def add_relationship(self, source: str, rel_type: str, target: str, properties: dict) -> str:
        """Add relationship between entities."""
        pass

    @abstractmethod
    def query(self, query: str, **kwargs) -> List[Entity]:
        """Execute query, return matching entities.

        Query language is implementation-specific:
        - Neo4jWorldModel: Cypher
        - PolyglotWorldModel: Natural language (GraphRAG)
        """
        pass

    @abstractmethod
    def get_all_entities(self, project: Optional[str] = None) -> Iterator[Entity]:
        """Iterate over all entities, optionally filtered by project."""
        pass

    @abstractmethod
    def get_statistics(self) -> StorageStats:
        """Get storage statistics (entity count, size, etc.)."""
        pass
```

**Error Handling:**
```python
class WorldModelStorageError(Exception):
    """Base exception for storage errors."""
    pass

class EntityNotFoundError(WorldModelStorageError):
    """Entity does not exist."""
    pass

class DuplicateEntityError(WorldModelStorageError):
    """Entity already exists."""
    pass

class InvalidEntityError(WorldModelStorageError):
    """Entity validation failed."""
    pass
```

---

## 6. Interface Specifications

### 6.1 Entity Data Class

**Purpose:** Represents a knowledge graph entity (paper, concept, experiment, etc.).

**Specification:**
```python
@dataclass
class Entity:
    """Knowledge graph entity."""

    id: Optional[str] = None  # Auto-generated if not provided
    type: str  # "Paper", "Concept", "Experiment", etc.
    properties: Dict[str, Any]  # Flexible property bag
    confidence: float = 1.0  # Confidence score (0.0-1.0)
    project: Optional[str] = None  # Project tag
    created_at: Optional[datetime] = None  # Creation timestamp
    updated_at: Optional[datetime] = None  # Last update timestamp
    created_by: Optional[str] = None  # Agent or user who created
    verified: bool = False  # User verification status
    annotations: List[Annotation] = field(default_factory=list)

    def __post_init__(self):
        """Validate entity after creation."""
        if not self.type:
            raise ValueError("Entity type is required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at
```

**Standard Entity Types:**
- `Paper`: Research paper from literature
- `Concept`: Scientific concept or term
- `Author`: Paper author
- `Experiment`: Experiment design or result
- `Hypothesis`: Scientific hypothesis
- `Finding`: Research finding or conclusion
- `Dataset`: Referenced dataset
- `Method`: Experimental or analytical method

**Common Properties by Type:**

```python
# Paper entity
{
    "title": str,
    "abstract": str,
    "authors": List[str],
    "year": int,
    "doi": str,
    "pmid": str,
    "citations": int,
    "journal": str
}

# Concept entity
{
    "name": str,
    "definition": str,
    "domain": str,  # "biology", "chemistry", etc.
    "synonyms": List[str],
    "related_concepts": List[str]
}

# Experiment entity
{
    "title": str,
    "description": str,
    "method": str,
    "results": str,
    "conclusion": str,
    "reproducibility_score": float
}
```

---

### 6.2 Relationship Data Class

**Purpose:** Represents a relationship between entities.

**Specification:**
```python
@dataclass
class Relationship:
    """Relationship between entities."""

    id: Optional[str] = None
    source_id: str  # Source entity ID
    target_id: str  # Target entity ID
    type: str  # Relationship type
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    created_at: Optional[datetime] = None
    created_by: Optional[str] = None

    def __post_init__(self):
        if not self.source_id or not self.target_id:
            raise ValueError("Source and target IDs are required")
        if not self.type:
            raise ValueError("Relationship type is required")
        if self.created_at is None:
            self.created_at = datetime.now()
```

**Standard Relationship Types:**
- `CITES`: Paper cites another paper
- `AUTHOR_OF`: Author wrote paper
- `MENTIONS`: Paper mentions concept
- `RELATES_TO`: Concept relates to another concept
- `SUPPORTS`: Finding supports hypothesis
- `REFUTES`: Finding refutes hypothesis
- `USES_METHOD`: Experiment uses method
- `PRODUCED_BY`: Finding produced by experiment
- `DERIVED_FROM`: Entity derived from other entities (provenance)

---

### 6.3 ProvenanceTracker Interface

**Purpose:** Track how knowledge was derived.

**Contract:**
```python
class ProvenanceTracker(ABC):
    """Abstract provenance tracking interface."""

    @abstractmethod
    def record_derivation(self, output_id: str, input_ids: List[str], activity: str):
        """Record that output was derived from inputs via activity.

        Args:
            output_id: ID of derived entity
            input_ids: IDs of source entities
            activity: Description of derivation process
        """
        pass

    @abstractmethod
    def record_attribution(self, entity_id: str, agent: str):
        """Record which agent/user created entity.

        Args:
            entity_id: Entity that was created
            agent: Agent or user identifier
        """
        pass

    @abstractmethod
    def get_provenance_chain(self, entity_id: str) -> ProvenanceChain:
        """Get complete derivation chain for entity.

        Returns tree showing how entity was derived from original sources.
        """
        pass

    @abstractmethod
    def export_provenance(self, format: str) -> str:
        """Export provenance in standard format.

        Args:
            format: "prov-o", "prov-json", "prov-xml"

        Returns:
            Serialized provenance in requested format
        """
        pass
```

**ProvenanceChain Data Structure:**
```python
@dataclass
class ProvenanceNode:
    """Node in provenance chain."""
    entity_id: str
    entity_type: str
    created_at: datetime
    created_by: str
    activity: Optional[str]  # How it was created
    sources: List['ProvenanceNode']  # What it was derived from

@dataclass
class ProvenanceChain:
    """Complete provenance chain."""
    root: ProvenanceNode
    depth: int  # How many derivation levels
    total_sources: int  # Total source entities
```

---

### 6.4 SemanticSearch Interface

**Purpose:** Find entities by semantic similarity.

**Contract:**
```python
class SemanticSearch(ABC):
    """Abstract semantic search interface."""

    @abstractmethod
    def find_similar(self, query: str, top_k: int = 10, filters: Optional[Dict] = None) -> List[ScoredEntity]:
        """Find entities semantically similar to query.

        Args:
            query: Text query
            top_k: Number of results to return
            filters: Optional filters (type, project, etc.)

        Returns:
            List of entities with similarity scores
        """
        pass

    @abstractmethod
    def find_similar_to_entity(self, entity_id: str, top_k: int = 10) -> List[ScoredEntity]:
        """Find entities similar to given entity.

        Args:
            entity_id: ID of reference entity
            top_k: Number of results

        Returns:
            List of similar entities with scores
        """
        pass

    @abstractmethod
    def cluster_entities(self, entity_ids: List[str], n_clusters: int) -> Dict[int, List[str]]:
        """Cluster entities by similarity.

        Args:
            entity_ids: Entities to cluster
            n_clusters: Number of clusters

        Returns:
            Dict mapping cluster ID to entity IDs
        """
        pass
```

**ScoredEntity Data Structure:**
```python
@dataclass
class ScoredEntity:
    """Entity with similarity score."""
    entity: Entity
    score: float  # Similarity score (0.0-1.0)
    explanation: Optional[str] = None  # Why it matched (Production Mode)
```

---

### 6.5 QueryEngine Interface

**Purpose:** Execute natural language queries against knowledge graph.

**Contract:**
```python
class QueryEngine(ABC):
    """Abstract query engine interface."""

    @abstractmethod
    def query(self, natural_language: str, **kwargs) -> QueryResult:
        """Execute natural language query.

        Simple Mode: Converts to Cypher, executes against Neo4j
        Production Mode: Uses GraphRAG (graph + semantic + LLM)

        Args:
            natural_language: Query in natural language
            **kwargs: Additional parameters (project filter, etc.)

        Returns:
            Query results with explanation
        """
        pass

    @abstractmethod
    def explain_query(self, natural_language: str) -> QueryExplanation:
        """Explain how query will be executed (without executing).

        Args:
            natural_language: Query to explain

        Returns:
            Explanation of query execution plan
        """
        pass
```

**QueryResult Data Structure:**
```python
@dataclass
class QueryResult:
    """Results of a query."""
    entities: List[Entity]
    relationships: List[Relationship]
    explanation: str  # How results were found
    confidence: float  # Overall confidence in results
    execution_time_ms: int
    used_semantic_search: bool  # Production Mode feature
    used_llm: bool  # Production Mode feature
```

**QueryExplanation Data Structure:**
```python
@dataclass
class QueryExplanation:
    """Explanation of query execution."""
    original_query: str
    interpreted_intent: str  # LLM interpretation (Production)
    cypher_query: Optional[str]  # Generated Cypher
    search_keywords: List[str]  # Keywords extracted
    filters_applied: Dict[str, Any]
    estimated_complexity: str  # "low", "medium", "high"
```

---

### 6.6 ProjectManager Interface

**Purpose:** Manage multiple research projects.

**Contract:**
```python
class ProjectManager(ABC):
    """Abstract project management interface."""

    @abstractmethod
    def create_project(self, name: str, description: str = "") -> Project:
        """Create new project."""
        pass

    @abstractmethod
    def get_project(self, name: str) -> Optional[Project]:
        """Get project by name."""
        pass

    @abstractmethod
    def list_projects(self) -> List[Project]:
        """List all projects."""
        pass

    @abstractmethod
    def delete_project(self, name: str, confirm: bool = False):
        """Delete project and all its entities."""
        pass

    @abstractmethod
    def get_active_project(self) -> Optional[Project]:
        """Get currently active project."""
        pass

    @abstractmethod
    def set_active_project(self, name: str):
        """Set active project."""
        pass
```

**Project Data Structure:**
```python
@dataclass
class Project:
    """Research project."""
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    entity_count: int
    relationship_count: int
    is_active: bool
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## 7. Data Models

### 7.1 Entity Schema (JSON)

**Purpose:** Standard JSON representation for export/import.

**Schema:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Entity",
  "type": "object",
  "required": ["id", "type", "properties"],
  "properties": {
    "id": {
      "type": "string",
      "description": "Unique entity identifier"
    },
    "type": {
      "type": "string",
      "enum": ["Paper", "Concept", "Author", "Experiment", "Hypothesis", "Finding", "Dataset", "Method"],
      "description": "Entity type"
    },
    "properties": {
      "type": "object",
      "description": "Entity-specific properties"
    },
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "default": 1.0
    },
    "project": {
      "type": "string",
      "description": "Project this entity belongs to"
    },
    "created_at": {
      "type": "string",
      "format": "date-time"
    },
    "updated_at": {
      "type": "string",
      "format": "date-time"
    },
    "created_by": {
      "type": "string",
      "description": "Agent or user who created entity"
    },
    "verified": {
      "type": "boolean",
      "default": false
    },
    "annotations": {
      "type": "array",
      "items": {
        "$ref": "#/definitions/annotation"
      }
    }
  },
  "definitions": {
    "annotation": {
      "type": "object",
      "required": ["text", "created_by", "created_at"],
      "properties": {
        "text": {"type": "string"},
        "created_by": {"type": "string"},
        "created_at": {"type": "string", "format": "date-time"}
      }
    }
  }
}
```

**Example Entity (JSON):**
```json
{
  "id": "paper_123",
  "type": "Paper",
  "properties": {
    "title": "Persistent World Models for AI-Driven Research",
    "abstract": "We present...",
    "authors": ["White, A.", "Smith, B."],
    "year": 2024,
    "doi": "10.1234/example",
    "journal": "Nature AI"
  },
  "confidence": 0.95,
  "project": "world_models",
  "created_at": "2025-01-15T10:30:00Z",
  "updated_at": "2025-01-15T10:30:00Z",
  "created_by": "literature_agent",
  "verified": true,
  "annotations": [
    {
      "text": "Key paper for our research direction",
      "created_by": "researcher@example.com",
      "created_at": "2025-01-16T14:20:00Z"
    }
  ]
}
```

---

### 7.2 Relationship Schema (JSON)

**Schema:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Relationship",
  "type": "object",
  "required": ["id", "source_id", "target_id", "type"],
  "properties": {
    "id": {"type": "string"},
    "source_id": {"type": "string"},
    "target_id": {"type": "string"},
    "type": {
      "type": "string",
      "enum": ["CITES", "AUTHOR_OF", "MENTIONS", "RELATES_TO", "SUPPORTS", "REFUTES", "USES_METHOD", "PRODUCED_BY", "DERIVED_FROM"]
    },
    "properties": {
      "type": "object",
      "description": "Relationship-specific properties"
    },
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0
    },
    "created_at": {
      "type": "string",
      "format": "date-time"
    },
    "created_by": {"type": "string"}
  }
}
```

**Example Relationship (JSON):**
```json
{
  "id": "rel_456",
  "source_id": "paper_123",
  "target_id": "concept_789",
  "type": "MENTIONS",
  "properties": {
    "section": "introduction",
    "relevance": "high"
  },
  "confidence": 0.88,
  "created_at": "2025-01-15T10:35:00Z",
  "created_by": "literature_agent"
}
```

---

### 7.3 Export Format (JSON)

**Purpose:** Complete graph export for backup/sharing.

**Structure:**
```json
{
  "version": "1.0",
  "export_date": "2025-01-20T15:00:00Z",
  "source": "kosmos",
  "mode": "simple",
  "project": "world_models",
  "statistics": {
    "entity_count": 1523,
    "relationship_count": 4891,
    "entity_types": {
      "Paper": 450,
      "Concept": 850,
      "Author": 223
    }
  },
  "entities": [
    {/* Entity objects */}
  ],
  "relationships": [
    {/* Relationship objects */}
  ],
  "projects": [
    {/* Project metadata */}
  ],
  "provenance": {
    /* Provenance records (if available) */
  }
}
```

---

### 7.4 PROV-O Provenance Model (Production Mode)

**Purpose:** W3C standard provenance for publication-quality tracking.

**Key Classes:**
- **prov:Entity**: A physical, digital, conceptual, or other kind of thing
- **prov:Activity**: Something that occurs over a period of time
- **prov:Agent**: Something that bears some form of responsibility for an activity

**Relationships:**
- **prov:wasDerivedFrom**: Entity was derived from another entity
- **prov:wasGeneratedBy**: Entity was generated by an activity
- **prov:used**: Activity used an entity
- **prov:wasAttributedTo**: Entity was attributed to an agent

**Example PROV-O Record:**
```json
{
  "@context": "https://www.w3.org/ns/prov",
  "@graph": [
    {
      "@id": "entity:finding_123",
      "@type": "prov:Entity",
      "label": "Key finding about protein folding",
      "prov:wasGeneratedBy": "activity:analysis_456",
      "prov:wasDerivedFrom": ["entity:paper_789", "entity:experiment_012"],
      "prov:wasAttributedTo": "agent:analysis_agent",
      "generatedAtTime": "2025-01-20T14:30:00Z"
    },
    {
      "@id": "activity:analysis_456",
      "@type": "prov:Activity",
      "label": "Analyze experiment results",
      "prov:used": ["entity:paper_789", "entity:experiment_012"],
      "prov:wasAssociatedWith": "agent:analysis_agent",
      "startedAtTime": "2025-01-20T14:25:00Z",
      "endedAtTime": "2025-01-20T14:30:00Z"
    },
    {
      "@id": "agent:analysis_agent",
      "@type": "prov:SoftwareAgent",
      "label": "Kosmos Analysis Agent",
      "version": "0.2.0"
    }
  ]
}
```

---

## 8. Database Schemas

### 8.1 Neo4j Schema (Simple Mode)

**Node Labels:**

**Entity Node:**
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

**Node Properties:**
```cypher
(:Entity {
  id: STRING,              // Unique identifier
  type: STRING,            // "Paper", "Concept", etc.
  properties: MAP,         // JSON properties
  properties_text: STRING, // Flattened text for search
  confidence: FLOAT,       // 0.0-1.0
  project: STRING,         // Project name
  created_at: DATETIME,
  updated_at: DATETIME,
  created_by: STRING,
  verified: BOOLEAN
})
```

**Relationship Types:**
```cypher
(:Entity)-[:CITES {
  confidence: FLOAT,
  created_at: DATETIME
}]->(:Entity)

(:Entity)-[:MENTIONS {
  confidence: FLOAT,
  section: STRING
}]->(:Entity)

(:Entity)-[:DERIVED_FROM {
  activity: STRING,
  created_at: DATETIME
}]->(:Entity)
```

**Indexes for Performance:**
```cypher
// Relationship type index for faster traversals
CREATE INDEX rel_type IF NOT EXISTS
FOR ()-[r:CITES]-() ON (r.created_at);

// Composite index for common queries
CREATE INDEX entity_type_project IF NOT EXISTS
FOR (e:Entity) ON (e.type, e.project);
```

---

### 8.2 PostgreSQL Schema (Production Mode)

**Entities Table:**
```sql
CREATE TABLE entities (
    id VARCHAR(255) PRIMARY KEY,
    type VARCHAR(50) NOT NULL,
    properties JSONB NOT NULL,
    confidence FLOAT CHECK (confidence >= 0.0 AND confidence <= 1.0),
    project VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255),
    verified BOOLEAN DEFAULT FALSE,

    -- Indexes
    CONSTRAINT check_type CHECK (type IN ('Paper', 'Concept', 'Author', 'Experiment', 'Hypothesis', 'Finding', 'Dataset', 'Method'))
);

CREATE INDEX idx_entities_type ON entities(type);
CREATE INDEX idx_entities_project ON entities(project);
CREATE INDEX idx_entities_created_at ON entities(created_at);
CREATE INDEX idx_entities_properties ON entities USING GIN (properties);
```

**Relationships Table:**
```sql
CREATE TABLE relationships (
    id VARCHAR(255) PRIMARY KEY,
    source_id VARCHAR(255) NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_id VARCHAR(255) NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    type VARCHAR(50) NOT NULL,
    properties JSONB,
    confidence FLOAT CHECK (confidence >= 0.0 AND confidence <= 1.0),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255),

    CONSTRAINT check_rel_type CHECK (type IN ('CITES', 'AUTHOR_OF', 'MENTIONS', 'RELATES_TO', 'SUPPORTS', 'REFUTES', 'USES_METHOD', 'PRODUCED_BY', 'DERIVED_FROM'))
);

CREATE INDEX idx_relationships_source ON relationships(source_id);
CREATE INDEX idx_relationships_target ON relationships(target_id);
CREATE INDEX idx_relationships_type ON relationships(type);
```

**Projects Table:**
```sql
CREATE TABLE projects (
    name VARCHAR(100) PRIMARY KEY,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT FALSE,
    tags TEXT[],
    metadata JSONB
);

CREATE INDEX idx_projects_active ON projects(is_active);
```

**Annotations Table:**
```sql
CREATE TABLE annotations (
    id SERIAL PRIMARY KEY,
    entity_id VARCHAR(255) NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_annotations_entity ON annotations(entity_id);
```

**Provenance Events Table (PROV-O):**
```sql
CREATE TABLE provenance_activities (
    id VARCHAR(255) PRIMARY KEY,
    label TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    ended_at TIMESTAMP WITH TIME ZONE,
    associated_agent VARCHAR(255)
);

CREATE TABLE provenance_derivations (
    id SERIAL PRIMARY KEY,
    generated_entity_id VARCHAR(255) NOT NULL REFERENCES entities(id),
    used_entity_id VARCHAR(255) NOT NULL REFERENCES entities(id),
    activity_id VARCHAR(255) NOT NULL REFERENCES provenance_activities(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_prov_generated ON provenance_derivations(generated_entity_id);
CREATE INDEX idx_prov_used ON provenance_derivations(used_entity_id);
```

---

### 8.3 Elasticsearch Schema (Production Mode)

**Entity Index Mapping:**
```json
{
  "mappings": {
    "properties": {
      "id": {"type": "keyword"},
      "type": {"type": "keyword"},
      "properties": {"type": "object", "enabled": true},
      "properties_text": {
        "type": "text",
        "analyzer": "standard",
        "fields": {
          "keyword": {"type": "keyword"}
        }
      },
      "confidence": {"type": "float"},
      "project": {"type": "keyword"},
      "created_at": {"type": "date"},
      "updated_at": {"type": "date"},
      "created_by": {"type": "keyword"},
      "verified": {"type": "boolean"}
    }
  },
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1,
    "index": {
      "max_result_window": 10000
    }
  }
}
```

**Provenance Events Index:**
```json
{
  "mappings": {
    "properties": {
      "event_id": {"type": "keyword"},
      "event_type": {"type": "keyword"},
      "entity_id": {"type": "keyword"},
      "activity_id": {"type": "keyword"},
      "agent": {"type": "keyword"},
      "timestamp": {"type": "date"},
      "payload": {"type": "object", "enabled": true}
    }
  },
  "settings": {
    "number_of_shards": 3,
    "index": {
      "lifecycle": {
        "name": "provenance_retention_policy",
        "rollover_alias": "provenance"
      }
    }
  }
}
```

---

### 8.4 Vector Database Schema (Production Mode)

**ChromaDB Collection (Local):**
```python
{
    "name": "kosmos_entities",
    "metadata": {
        "description": "Entity embeddings for semantic search",
        "embedding_model": "text-embedding-ada-002",
        "dimension": 1536
    },
    "distance_function": "cosine"
}
```

**Document Structure:**
```python
{
    "id": "entity_123",  // Entity ID
    "embedding": [0.023, -0.412, ...],  // 1536-dimensional vector
    "metadata": {
        "type": "Paper",
        "project": "world_models",
        "confidence": 0.95,
        "created_at": "2025-01-15T10:30:00Z"
    },
    "document": "Full text representation of entity for retrieval"
}
```

**Pinecone Index (Cloud):**
```python
{
    "name": "kosmos-entities",
    "dimension": 1536,
    "metric": "cosine",
    "pods": 1,
    "replicas": 1,
    "pod_type": "p1.x1",
    "metadata_config": {
        "indexed": ["type", "project", "confidence"]
    }
}
```

---

## 9. API Specifications

### 9.1 Python API

**World Model Creation:**
```python
from kosmos.world_model import WorldModelFactory
from kosmos.config import Config

# Load configuration
config = Config.from_file("config.yaml")

# Create world model (mode selected from config)
world_model = WorldModelFactory.create(config)
```

**Entity Operations:**
```python
from kosmos.world_model.models import Entity

# Create entity
paper = Entity(
    type="Paper",
    properties={
        "title": "Example Paper",
        "authors": ["Smith, J."],
        "year": 2024
    },
    project="my_research",
    confidence=0.95
)

# Add to world model
entity_id = world_model.add_entity(paper)

# Retrieve entity
retrieved = world_model.get_entity(entity_id)

# Update entity
paper.confidence = 0.98
world_model.update_entity(paper)

# Delete entity
world_model.delete_entity(entity_id)
```

**Relationship Operations:**
```python
# Add relationship
rel_id = world_model.add_relationship(
    source_id="paper_123",
    rel_type="CITES",
    target_id="paper_456",
    properties={"context": "in introduction"}
)
```

**Query Operations:**
```python
# Natural language query
results = world_model.query(
    "Find all papers about protein folding published after 2020",
    project="biology_research"
)

# Semantic search
similar = world_model.find_similar(
    "machine learning for drug discovery",
    top_k=10
)

# Get provenance
provenance = world_model.get_provenance("finding_123")
```

**Export/Import:**
```python
# Export project
export_path = world_model.export_project(
    project="my_research",
    format="json"
)

# Import data
world_model.import_data(
    filepath=export_path,
    project="imported_research"
)
```

---

### 9.2 CLI API

**Graph Management Commands:**
```bash
# Get graph information
kosmos graph info [--project PROJECT]

# Export graph
kosmos graph export FILEPATH [--format json|graphml] [--project PROJECT]

# Import graph
kosmos graph import FILEPATH [--project PROJECT] [--mode merge|replace]

# Reset graph (with confirmation)
kosmos graph reset [--project PROJECT] [--confirm]
```

**Project Management Commands:**
```bash
# Create project
kosmos project create NAME [--description DESC]

# List projects
kosmos project list

# Switch active project
kosmos project switch NAME

# Delete project
kosmos project delete NAME [--confirm]

# Project info
kosmos project info NAME
```

**Curation Commands:**
```bash
# Verify entity
kosmos verify ENTITY_ID

# Add annotation
kosmos annotate ENTITY_ID "annotation text"

# Find duplicates
kosmos duplicates [--project PROJECT] [--auto-merge]

# Quality report
kosmos quality [--project PROJECT]
```

**Migration Commands:**
```bash
# Migrate to production mode
kosmos migrate simple-to-production \
    --validate \
    --backup BACKUP_PATH \
    --target-config production.yaml

# Validate migration possibility
kosmos migrate validate --target production
```

**Query Commands:**
```bash
# Natural language query
kosmos query "find papers about CRISPR" [--project PROJECT]

# Semantic search
kosmos search "gene editing techniques" [--top-k 10]

# Get provenance
kosmos provenance ENTITY_ID [--format text|json|prov-o]
```

---

### 9.3 Configuration API

**Simple Mode Configuration:**
```yaml
world_model:
  mode: simple

  simple:
    backend: neo4j
    neo4j:
      url: bolt://localhost:7687
      auth:
        user: neo4j
        password: ${NEO4J_PASSWORD}
      database: kosmos
      data_path: ~/.kosmos/neo4j_data

    # Duplication detection
    similarity_threshold: 0.85

    # Caching
    cache_enabled: false

    # Observability
    metrics:
      collect: true
      export: false
    logging:
      level: INFO
      file: ~/.kosmos/logs/world_model.log
```

**Production Mode Configuration:**
```yaml
world_model:
  mode: production

  production:
    # PostgreSQL for metadata
    postgres:
      url: postgresql://user:pass@localhost:5432/kosmos
      pool_size: 10
      max_overflow: 20

    # Neo4j for graph operations
    neo4j:
      url: bolt://localhost:7687
      auth:
        user: neo4j
        password: ${NEO4J_PASSWORD}
      database: kosmos

    # Elasticsearch for provenance
    elasticsearch:
      url: http://localhost:9200
      index_prefix: kosmos
      retention_days: 365

    # Vector database
    vector_db:
      provider: pinecone  # or chromadb
      api_key: ${PINECONE_API_KEY}
      environment: us-west1-gcp
      index_name: kosmos-entities
      dimension: 1536

    # GraphRAG settings
    graphrag:
      enabled: true
      llm_provider: anthropic
      temperature: 0.0
      max_hops: 3

    # Caching (Redis)
    cache:
      enabled: true
      redis_url: redis://localhost:6379/0
      ttl: 3600

    # Observability
    metrics:
      collect: true
      export: true
      prometheus_port: 9090

    tracing:
      enabled: true
      provider: opentelemetry
      endpoint: http://localhost:4318

    logging:
      level: INFO
      structured: true
      file: /var/log/kosmos/world_model.log
```

---

## 10. Data Flow Diagrams

### 10.1 Entity Creation Flow (Simple Mode)

```
User/Agent
    │
    ▼
┌───────────────────────┐
│ WorldModel.add_entity()│
└───────────┬───────────┘
            │
            ▼
    ┌───────────────┐
    │ Validate      │
    │ entity schema │
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │ Check for     │
    │ duplicates    │
    └───────┬───────┘
            │
    ┌───────┴────────┐
    │ Exists?        │
    │                │
    ▼ No             ▼ Yes
┌──────────┐    ┌──────────┐
│ Generate │    │ Merge    │
│ new ID   │    │ with     │
│          │    │ existing │
└────┬─────┘    └────┬─────┘
     │               │
     └───────┬───────┘
             │
             ▼
    ┌────────────────┐
    │ Neo4jWorldModel│
    │ .add_entity()  │
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │ CREATE node    │
    │ in Neo4j       │
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │ Record metrics │
    │ & logs         │
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │ Return         │
    │ entity_id      │
    └────────────────┘
```

### 10.2 Entity Creation Flow (Production Mode)

```
User/Agent
    │
    ▼
┌───────────────────────┐
│ WorldModel.add_entity()│
└───────────┬───────────┘
            │
            ▼
    ┌───────────────┐
    │ Validate      │
    └───────┬───────┘
            │
            ▼
    ┌───────────────────┐
    │ Begin transaction │
    │ (PostgreSQL)      │
    └───────┬───────────┘
            │
            ▼
    ┌──────────────────┐
    │ Check duplicates │
    │ (Vector search)  │
    └───────┬──────────┘
            │
    ┌───────┴────────┐
    │ Exists?        │
    ▼ No             ▼ Yes
┌──────────┐    ┌──────────┐
│ Generate │    │ Merge    │
│ new ID   │    │ logic    │
└────┬─────┘    └────┬─────┘
     │               │
     └───────┬───────┘
             │
             ▼
┌──────────────────────────┐
│ Polyglot Storage:        │
│                          │
│ 1. PostgreSQL.insert()   │◄── Metadata
│ 2. Neo4j.create_node()   │◄── Graph
│ 3. ES.index_event()      │◄── Provenance
│ 4. VectorDB.embed()      │◄── Semantic
└──────────┬───────────────┘
           │
           ▼
   ┌───────────────┐
   │ Commit        │
   │ transaction   │
   └───────┬───────┘
           │
           ▼
   ┌───────────────┐
   │ Record        │
   │ metrics/trace │
   └───────┬───────┘
           │
           ▼
   ┌───────────────┐
   │ Return ID     │
   └───────────────┘
```

### 10.3 Query Flow (GraphRAG - Production Mode)

```
Natural Language Query
    │
    ▼
┌────────────────────┐
│ GraphRAGQueryEngine│
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ 1. LLM interprets  │
│    user intent     │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ 2. Extract         │
│    keywords &      │
│    filters         │
└────────┬───────────┘
         │
         ├──────────────┬──────────────┐
         │              │              │
         ▼              ▼              ▼
┌────────────┐  ┌─────────────┐  ┌──────────┐
│ Vector     │  │ Graph       │  │ Keyword  │
│ similarity │  │ traversal   │  │ search   │
│ (VectorDB) │  │ (Neo4j)     │  │ (ES)     │
└────┬───────┘  └─────┬───────┘  └────┬─────┘
     │                │                │
     └────────────────┼────────────────┘
                      │
                      ▼
           ┌──────────────────┐
           │ 3. Merge results │
           │    & rank        │
           └──────────┬───────┘
                      │
                      ▼
           ┌──────────────────┐
           │ 4. LLM generates │
           │    explanation   │
           └──────────┬───────┘
                      │
                      ▼
           ┌──────────────────┐
           │ Return           │
           │ QueryResult      │
           └──────────────────┘
```

---

## 11. Deployment Architectures

### 11.1 Local Development (Simple Mode)

```
┌─────────────────────────────────────┐
│    Developer Laptop                 │
│                                     │
│  ┌───────────────────────────────┐ │
│  │  Kosmos Application           │ │
│  │  (Python)                     │ │
│  │                               │ │
│  │  ┌──────────────────────────┐ │ │
│  │  │  WorldModel (Simple)     │ │ │
│  │  └───────────┬──────────────┘ │ │
│  └──────────────┼────────────────┘ │
│                 │                  │
│                 ▼                  │
│  ┌──────────────────────────────┐ │
│  │  Docker Container:           │ │
│  │  Neo4j 5.x                   │ │
│  │                              │ │
│  │  Volume: ~/.kosmos/neo4j_data│ │
│  │  Port: 7687 (bolt)           │ │
│  │  Port: 7474 (browser)        │ │
│  └──────────────────────────────┘ │
└─────────────────────────────────────┘
```

**Setup:**
```bash
# 1. Install Kosmos
pip install kosmos

# 2. Initialize (starts Neo4j container)
kosmos init --mode simple

# 3. Verify
kosmos graph info
```

---

### 11.2 Production Mode - Docker Compose

```
┌──────────────────────────────────────────────────────┐
│  Docker Compose Stack                                │
│                                                      │
│  ┌────────────────────────────────────────────────┐ │
│  │  Kosmos Application Container                  │ │
│  │  - Python application                          │ │
│  │  - WorldModel (Production)                     │ │
│  └────┬──────┬──────┬──────┬─────────────────────┘ │
│       │      │      │      │                        │
│       │      │      │      │                        │
│  ┌────▼───┐ ┌▼─────▼┐ ┌───▼────┐ ┌───────────────┐ │
│  │Postgres│ │Neo4j  │ │Elastic │ │ChromaDB       │ │
│  │        │ │       │ │search  │ │(Vector)       │ │
│  │Port    │ │Port   │ │Port    │ │Port 8000      │ │
│  │5432    │ │7687   │ │9200    │ │               │ │
│  └────────┘ └───────┘ └────────┘ └───────────────┘ │
│                                                      │
│  ┌──────────────┐                                   │
│  │  Redis       │                                   │
│  │  (Cache)     │                                   │
│  │  Port 6379   │                                   │
│  └──────────────┘                                   │
│                                                      │
│  Volumes:                                           │
│  - postgres_data                                    │
│  - neo4j_data                                       │
│  - es_data                                          │
│  - chromadb_data                                    │
└──────────────────────────────────────────────────────┘
```

**docker-compose.yml:**
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
    ports:
      - "5432:5432"

  neo4j:
    image: neo4j:5.13
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
      NEO4J_PLUGINS: '["apoc", "graph-data-science"]'
    volumes:
      - neo4j_data:/data
    ports:
      - "7687:7687"
      - "7474:7474"

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
      - xpack.security.enabled=false
    volumes:
      - es_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"

  chromadb:
    image: ghcr.io/chroma-core/chroma:latest
    volumes:
      - chromadb_data:/chroma/chroma
    ports:
      - "8000:8000"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  kosmos:
    build: .
    depends_on:
      - postgres
      - neo4j
      - elasticsearch
      - chromadb
      - redis
    environment:
      WORLD_MODEL_MODE: production
      POSTGRES_URL: postgresql://kosmos:${POSTGRES_PASSWORD}@postgres:5432/kosmos
      NEO4J_URL: bolt://neo4j:7687
      NEO4J_PASSWORD: ${NEO4J_PASSWORD}
      ES_URL: http://elasticsearch:9200
      CHROMADB_URL: http://chromadb:8000
      REDIS_URL: redis://redis:6379/0
    volumes:
      - ./config:/app/config
    command: kosmos server

volumes:
  postgres_data:
  neo4j_data:
  es_data:
  chromadb_data:
```

---

### 11.3 Production Mode - Managed Services (AWS)

```
┌────────────────────────────────────────────────────────┐
│  AWS Cloud                                             │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  ECS Fargate / EKS                               │ │
│  │  ┌────────────────────────────────────────────┐  │ │
│  │  │  Kosmos Application Container              │  │ │
│  │  │  (Auto-scaling)                            │  │ │
│  │  └────┬──────┬──────┬──────┬─────────────────┘  │ │
│  └───────┼──────┼──────┼──────┼────────────────────┘ │
│          │      │      │      │                       │
│  ┌───────▼──┐ ┌▼──────▼┐ ┌───▼──────┐ ┌───────────┐ │
│  │ RDS      │ │Neo4j   │ │Amazon   │ │Pinecone   │ │
│  │Postgres  │ │AuraDB  │ │OpenSearch│ │(Cloud)    │ │
│  │          │ │(Cloud) │ │         │ │           │ │
│  │Multi-AZ  │ │        │ │Multi-AZ │ │           │ │
│  └──────────┘ └────────┘ └─────────┘ └───────────┘ │
│                                                       │
│  ┌──────────────┐                                    │
│  │ ElastiCache  │                                    │
│  │ Redis        │                                    │
│  │ (Multi-AZ)   │                                    │
│  └──────────────┘                                    │
│                                                       │
│  ┌──────────────┐  ┌──────────────┐                 │
│  │ CloudWatch   │  │ X-Ray        │                 │
│  │ (Metrics)    │  │ (Tracing)    │                 │
│  └──────────────┘  └──────────────┘                 │
└────────────────────────────────────────────────────────┘
```

**Configuration:**
```yaml
world_model:
  mode: production

  production:
    postgres:
      url: postgresql://user:pass@kosmos.abc123.us-west-2.rds.amazonaws.com:5432/kosmos
      pool_size: 20

    neo4j:
      url: bolt+s://abc123.databases.neo4j.io:7687
      auth:
        user: neo4j
        password: ${NEO4J_AURADB_PASSWORD}

    elasticsearch:
      url: https://search-kosmos-abc123.us-west-2.es.amazonaws.com:443
      auth:
        api_key: ${AWS_OPENSEARCH_API_KEY}

    vector_db:
      provider: pinecone
      api_key: ${PINECONE_API_KEY}
      environment: us-west1-gcp
      index_name: kosmos-prod

    cache:
      enabled: true
      redis_url: redis://kosmos.abc123.cache.amazonaws.com:6379/0

    observability:
      cloudwatch:
        enabled: true
        region: us-west-2
      xray:
        enabled: true
```

**Cost Estimate (Monthly):**
- RDS PostgreSQL (db.t3.medium): $70
- Neo4j AuraDB (2GB RAM): $65
- Amazon OpenSearch (t3.small): $40
- ElastiCache Redis (cache.t3.micro): $15
- Pinecone (100K vectors): $70
- ECS Fargate (2 vCPU, 4GB): $60
- **Total: ~$320/month**

---

## 12. Security Architecture

### 12.1 Authentication and Authorization (Production Mode)

**Multi-tenancy Model:**
```python
class User:
    id: str
    email: str
    role: str  # "admin", "researcher", "viewer"
    projects: List[str]  # Projects user can access

class AccessControl:
    def can_read_entity(self, user: User, entity: Entity) -> bool:
        """Check if user can read entity."""
        if user.role == "admin":
            return True
        return entity.project in user.projects

    def can_write_entity(self, user: User, entity: Entity) -> bool:
        """Check if user can modify entity."""
        if user.role == "viewer":
            return False
        if user.role == "admin":
            return True
        return entity.project in user.projects
```

**JWT-Based Authentication:**
```python
# Configuration
security:
  authentication:
    enabled: true  # Production Mode only
    jwt_secret: ${JWT_SECRET}
    token_ttl: 3600  # 1 hour
    refresh_token_ttl: 604800  # 7 days

# Usage
headers = {
    "Authorization": f"Bearer {jwt_token}"
}
```

---

### 12.2 Data Security

**Encryption at Rest:**
```yaml
# Production Mode
world_model:
  production:
    postgres:
      ssl_mode: require
      ssl_cert: /path/to/cert.pem

    neo4j:
      encryption: true
      trust_strategy: TRUST_SYSTEM_CA_SIGNED_CERTIFICATES

    elasticsearch:
      use_ssl: true
      verify_certs: true
```

**Encryption in Transit:**
- All database connections use TLS/SSL
- API endpoints use HTTPS only
- Internal service communication encrypted

**Export Encryption:**
```python
# Encrypt exported graphs
world_model.export_project(
    project="sensitive_research",
    format="json",
    encrypt=True,
    encryption_key="${EXPORT_ENCRYPTION_KEY}"
)
```

---

### 12.3 Input Validation

**Entity Validation:**
```python
from pydantic import BaseModel, validator

class Entity(BaseModel):
    id: Optional[str]
    type: str
    properties: Dict[str, Any]

    @validator('type')
    def validate_type(cls, v):
        allowed_types = ['Paper', 'Concept', 'Author', 'Experiment', 'Hypothesis', 'Finding', 'Dataset', 'Method']
        if v not in allowed_types:
            raise ValueError(f'Invalid entity type: {v}')
        return v

    @validator('properties')
    def validate_properties(cls, v):
        # Prevent injection attacks via properties
        for key, value in v.items():
            if isinstance(value, str):
                if len(value) > 10000:  # Max length
                    raise ValueError(f'Property {key} too long')
                # Basic XSS prevention
                if '<script' in value.lower():
                    raise ValueError(f'Invalid characters in property {key}')
        return v
```

**Query Sanitization:**
```python
def sanitize_query(query: str) -> str:
    """Sanitize user input for Cypher queries."""
    # Remove Cypher keywords that could modify data
    dangerous = ['DELETE', 'REMOVE', 'SET', 'CREATE', 'MERGE']
    for keyword in dangerous:
        if keyword in query.upper():
            raise ValueError(f'Query contains dangerous keyword: {keyword}')

    # Parameterize queries
    return query

# Use parameterized queries
def get_entities_by_type(entity_type: str):
    # Safe: uses parameters
    return session.run(
        "MATCH (e:Entity {type: $type}) RETURN e",
        type=entity_type
    )
```

---

### 12.4 Audit Logging

**Security Event Logging:**
```python
class AuditLogger:
    def log_access(self, user: str, entity_id: str, action: str):
        """Log entity access."""
        self.logger.info("entity_access", extra={
            "user": user,
            "entity_id": entity_id,
            "action": action,
            "timestamp": datetime.now(),
            "ip_address": get_client_ip()
        })

    def log_modification(self, user: str, entity_id: str, changes: Dict):
        """Log entity modifications."""
        self.logger.warning("entity_modified", extra={
            "user": user,
            "entity_id": entity_id,
            "changes": changes,
            "timestamp": datetime.now()
        })
```

**Audit Events:**
- Entity access (read)
- Entity modification (create, update, delete)
- Failed authentication attempts
- Permission denials
- Export operations
- Migration operations

---

## 13. Performance Architecture

### 13.1 Caching Strategy

**Multi-Level Caching:**
```python
class CachedWorldModel:
    def __init__(self, storage: WorldModelStorage, cache: CacheClient):
        self.storage = storage
        self.cache = cache

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        # Level 1: In-memory cache (local)
        if entity_id in self.memory_cache:
            return self.memory_cache[entity_id]

        # Level 2: Redis cache (shared)
        cached = self.cache.get(f"entity:{entity_id}")
        if cached:
            entity = Entity.parse_raw(cached)
            self.memory_cache[entity_id] = entity  # Promote to L1
            return entity

        # Level 3: Database
        entity = self.storage.get_entity(entity_id)
        if entity:
            # Cache for future requests
            self.cache.set(f"entity:{entity_id}", entity.json(), ttl=3600)
            self.memory_cache[entity_id] = entity
        return entity
```

**Cache Invalidation:**
```python
def update_entity(self, entity: Entity):
    # Update database
    self.storage.update_entity(entity)

    # Invalidate caches
    self.cache.delete(f"entity:{entity.id}")
    if entity.id in self.memory_cache:
        del self.memory_cache[entity.id]
```

---

### 13.2 Query Optimization

**Neo4j Query Optimization:**
```cypher
// Bad: Full scan
MATCH (e:Entity)
WHERE e.properties.title CONTAINS "protein"
RETURN e;

// Good: Index usage
MATCH (e:Entity)
WHERE e.type = "Paper"  // Uses index
AND e.properties_text CONTAINS "protein"  // Uses fulltext index
RETURN e
LIMIT 100;

// Best: Parameterized with limits
MATCH (e:Entity {type: $type})
WHERE e.properties_text CONTAINS $search_term
RETURN e
LIMIT $limit;
```

**Connection Pooling:**
```python
# PostgreSQL
from sqlalchemy.pool import QueuePool

engine = create_engine(
    postgres_url,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True  # Verify connections before use
)

# Neo4j
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    neo4j_url,
    auth=auth,
    max_connection_pool_size=50,
    connection_acquisition_timeout=60
)
```

---

### 13.3 Batch Operations

**Bulk Entity Creation:**
```python
def add_entities_batch(self, entities: List[Entity]) -> List[str]:
    """Add multiple entities in a single transaction."""
    with self.storage.transaction() as tx:
        entity_ids = []
        for entity in entities:
            entity_id = tx.add_entity(entity)
            entity_ids.append(entity_id)
        tx.commit()
    return entity_ids
```

**Pagination:**
```python
def get_entities_paginated(
    self,
    page: int = 1,
    page_size: int = 100,
    **filters
) -> PaginatedResult:
    """Get entities with pagination."""
    offset = (page - 1) * page_size

    entities = self.storage.query(
        filters=filters,
        limit=page_size,
        offset=offset
    )

    total = self.storage.count(filters=filters)

    return PaginatedResult(
        entities=entities,
        page=page,
        page_size=page_size,
        total=total,
        total_pages=(total + page_size - 1) // page_size
    )
```

---

### 13.4 Performance Monitoring

**Key Metrics:**
```python
# Entity operations
metrics.histogram("world_model.add_entity.duration_ms")
metrics.histogram("world_model.query.duration_ms")
metrics.counter("world_model.entities_created")
metrics.counter("world_model.queries_executed")

# Database operations
metrics.histogram("neo4j.query.duration_ms")
metrics.histogram("postgres.query.duration_ms")
metrics.gauge("neo4j.connection_pool.active")
metrics.gauge("postgres.connection_pool.active")

# Cache performance
metrics.counter("cache.hits")
metrics.counter("cache.misses")
metrics.gauge("cache.hit_rate", lambda: hits / (hits + misses))
```

**Performance SLOs:**
- Simple Mode query p95: < 1 second
- Production Mode query p95: < 100ms
- Entity creation p95: < 500ms
- Export 10K entities: < 30 seconds

---

## 14. Observability Architecture

### 14.1 Metrics Collection

**Prometheus Metrics (Production Mode):**
```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
entity_operations = Counter(
    'world_model_entity_operations_total',
    'Total entity operations',
    ['operation', 'entity_type', 'project']
)

query_duration = Histogram(
    'world_model_query_duration_seconds',
    'Query duration in seconds',
    ['query_type', 'mode']
)

graph_size = Gauge(
    'world_model_graph_size',
    'Current graph size',
    ['project', 'metric']  # metric: entity_count, relationship_count
)

# Use metrics
with query_duration.labels(query_type='semantic', mode='production').time():
    results = self.semantic_search.find_similar(query)

entity_operations.labels(
    operation='create',
    entity_type='Paper',
    project='biology'
).inc()
```

**Metrics Endpoint:**
```python
# Expose metrics for Prometheus scraping
from prometheus_client import generate_latest

@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': 'text/plain'}
```

---

### 14.2 Distributed Tracing

**OpenTelemetry Integration:**
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Setup tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Export to collector
otlp_exporter = OTLPSpanExporter(endpoint="localhost:4317")
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrument operations
def query(self, query_str: str) -> QueryResult:
    with tracer.start_as_current_span("world_model.query") as span:
        span.set_attribute("query", query_str)
        span.set_attribute("mode", self.config.mode)

        # Sub-spans for each step
        with tracer.start_as_current_span("parse_query"):
            parsed = self.parse_query(query_str)

        with tracer.start_as_current_span("execute_query"):
            results = self.execute(parsed)

        span.set_attribute("result_count", len(results))
        return results
```

---

### 14.3 Structured Logging

**Log Format:**
```python
import structlog

logger = structlog.get_logger()

# Structured logs
logger.info(
    "entity_created",
    entity_id="paper_123",
    entity_type="Paper",
    project="biology",
    user="researcher@example.com",
    confidence=0.95,
    duration_ms=45
)

# Contextual logging
with structlog.contextvars.bound_contextvars(
    request_id="req_456",
    user="researcher@example.com"
):
    logger.info("processing_query")
    # ... operations ...
    logger.info("query_completed", result_count=10)
```

**Log Levels:**
- DEBUG: Detailed information for debugging
- INFO: Normal operations (entity creation, queries)
- WARNING: Unexpected but handled situations (duplicates, low confidence)
- ERROR: Errors that need attention (failed queries, data inconsistencies)
- CRITICAL: System failures (database connection lost)

---

### 14.4 Health Checks

**Health Check Endpoint:**
```python
@app.route('/health')
def health_check():
    """Comprehensive health check."""
    health = {
        "status": "healthy",
        "mode": config.world_model.mode,
        "checks": {}
    }

    # Check Neo4j
    try:
        with neo4j_driver.session() as session:
            session.run("RETURN 1")
        health["checks"]["neo4j"] = "healthy"
    except Exception as e:
        health["checks"]["neo4j"] = f"unhealthy: {e}"
        health["status"] = "degraded"

    # Production Mode: Check additional services
    if config.world_model.mode == "production":
        # PostgreSQL
        try:
            with postgres_engine.connect() as conn:
                conn.execute("SELECT 1")
            health["checks"]["postgres"] = "healthy"
        except Exception as e:
            health["checks"]["postgres"] = f"unhealthy: {e}"
            health["status"] = "degraded"

        # Elasticsearch
        # Vector DB
        # Redis

    return health, 200 if health["status"] == "healthy" else 503
```

---

## 15. Migration Architecture

### 15.1 Version Compatibility Matrix

| From Version | To Version | Migration Required | Breaking Changes |
|--------------|------------|-------------------|------------------|
| 1.0 Simple   | 1.1 Simple | No | No |
| 1.0 Simple   | 2.0 Production | Yes | Schema changes |
| 1.0 Production | 2.0 Production | Yes | API changes |

---

### 15.2 Migration Procedures

**Phase 1 to Phase 2 Migration:**
```bash
# 1. Backup current graph
kosmos graph export ~/backups/phase1_backup.json

# 2. Run migration
kosmos migrate phase1-to-phase2 \
    --validate \
    --dry-run  # Test first

# 3. If dry-run succeeds, execute
kosmos migrate phase1-to-phase2 \
    --execute

# 4. Verify
kosmos graph info
kosmos quality --check-annotations
```

**Simple to Production Migration:**
```bash
# 1. Validate readiness
kosmos migrate validate --target production

# 2. Create backup
kosmos graph export ~/backups/simple_mode_backup.json --format json

# 3. Prepare production infrastructure
docker-compose -f docker-compose.production.yml up -d

# 4. Run migration
kosmos migrate simple-to-production \
    --source-config simple.yaml \
    --target-config production.yaml \
    --validate \
    --backup ~/backups/migration_backup.json

# 5. Verify data integrity
kosmos migrate verify \
    --source ~/backups/simple_mode_backup.json \
    --target production
```

---

### 15.3 Rollback Strategy

**Rollback Procedure:**
```python
class MigrationManager:
    def rollback_migration(self, backup_path: Path):
        """Rollback to previous state."""
        logger.warning("Starting migration rollback")

        # 1. Stop current system
        self.world_model.shutdown()

        # 2. Clear target database
        if self.config.mode == "production":
            self.clear_production_databases()
        else:
            self.clear_neo4j()

        # 3. Restore from backup
        self.world_model.import_data(backup_path)

        # 4. Verify restoration
        stats = self.world_model.get_statistics()
        logger.info("Rollback complete", **stats)
```

---

## 16. Architecture Decision Records

### ADR-001: Abstract Storage Layer

**Status:** Accepted

**Context:**
Need to support both Simple Mode (Neo4j only) and Production Mode (polyglot) while avoiding code duplication.

**Decision:**
Implement abstract `WorldModelStorage` interface with two implementations: `Neo4jWorldModel` and `PolyglotWorldModel`.

**Consequences:**
- ✅ Clean separation between business logic and storage
- ✅ Easy to swap implementations via configuration
- ✅ Testable with mock implementations
- ❌ Slight performance overhead from abstraction
- ❌ Must design interface to support both modes

**Alternatives Considered:**
1. Separate codebases for Simple and Production modes
   - Rejected: Too much duplication
2. Feature flags within single implementation
   - Rejected: Too complex and error-prone
3. Inheritance-based approach
   - Rejected: Violates composition over inheritance

---

### ADR-002: Neo4j for Simple Mode

**Status:** Accepted

**Context:**
Need simple, immediately usable persistent storage for 90% of users.

**Decision:**
Use Neo4j as the sole database for Simple Mode, storing all data (metadata, graph, provenance) in Neo4j.

**Consequences:**
- ✅ Single database to manage
- ✅ Easy Docker setup
- ✅ Sufficient performance for 10K entities
- ✅ Native graph operations
- ❌ Not optimized for specialized workloads
- ❌ Limited to ~10K-50K entities practically

**Performance Characteristics:**
- Entity creation: ~50ms
- Simple query: ~100-500ms
- Complex traversal: ~1-2s
- Export 10K entities: ~20s

---

### ADR-003: Polyglot Persistence for Production Mode

**Status:** Accepted

**Context:**
Organizations need enterprise-scale performance, specialized features, and publication-quality provenance.

**Decision:**
Use four specialized databases in Production Mode:
- PostgreSQL: Metadata and transactions (ACID guarantees)
- Neo4j: Graph operations (relationship traversal)
- Elasticsearch: Provenance events (time-series queries)
- Vector DB: Semantic search (embeddings)

**Consequences:**
- ✅ Each database optimized for its workload
- ✅ Scalable to 100K+ entities
- ✅ Production-grade performance (<100ms queries)
- ❌ Complex operational overhead
- ❌ Higher infrastructure cost
- ❌ Requires expertise to operate

**Why Each Database:**
- **PostgreSQL:** Best-in-class ACID transactions, mature, reliable
- **Neo4j:** Specialized graph traversal, Cypher query language
- **Elasticsearch:** Time-series provenance events, full-text search
- **Vector DB:** Semantic similarity, embeddings, nearest-neighbor

---

### ADR-004: Project Isolation via Namespaces

**Status:** Accepted

**Context:**
Users need to maintain multiple research projects without cross-contamination.

**Decision:**
Implement project isolation using graph namespaces (project property on all entities) rather than separate databases.

**Consequences:**
- ✅ Share infrastructure across projects
- ✅ Enable cross-project queries when needed
- ✅ Simpler backup/restore (single graph)
- ✅ Lower storage overhead
- ❌ Queries must filter by project (performance impact)
- ❌ Accidental cross-project leakage possible

**Implementation:**
```cypher
// All entities tagged with project
CREATE (e:Entity {
    id: "entity_123",
    project: "biology_research",
    ...
})

// Queries filtered by project
MATCH (e:Entity {project: $project})
WHERE ...
RETURN e
```

**Alternative Considered:**
Separate databases per project - Rejected due to operational complexity.

---

### ADR-005: PROV-O Standard for Provenance

**Status:** Accepted

**Context:**
Production Mode needs publication-quality provenance that can be cited in research papers.

**Decision:**
Implement W3C PROV-O standard for provenance in Production Mode, while Simple Mode uses basic parent/child relationships.

**Consequences:**
- ✅ Interoperable with other provenance systems
- ✅ Industry standard, well-documented
- ✅ Publication-quality
- ✅ Tool support (visualization, validation)
- ❌ More complex than simple derivation
- ❌ Requires Elasticsearch for time-series storage

**PROV-O Model:**
- Entity: Things (papers, concepts, findings)
- Activity: Processes (analysis, synthesis)
- Agent: Actors (LLMs, users, agents)
- Derivation: How entities relate

---

### ADR-006: GraphRAG Query Pattern

**Status:** Accepted

**Context:**
Production Mode should support sophisticated natural language queries combining graph traversal, semantic search, and LLM reasoning.

**Decision:**
Implement GraphRAG pattern: use LLM to interpret query → execute graph traversal + semantic search in parallel → merge results → LLM generates explanation.

**Consequences:**
- ✅ Natural language queries feel intuitive
- ✅ Combines strengths of each system
- ✅ Explainable results
- ❌ Higher cost (LLM calls)
- ❌ More complex error handling
- ❌ Latency from LLM calls

**Query Flow:**
1. LLM interprets user intent
2. Extract keywords and filters
3. Parallel execution:
   - Vector similarity search
   - Graph traversal
   - Keyword search (Elasticsearch)
4. Merge and rank results
5. LLM generates explanation

---

## 17. Technology Stack

### 17.1 Core Technologies

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| Language | Python | 3.10+ | Kosmos ecosystem, ML libraries |
| Graph DB | Neo4j | 5.x | Industry standard, mature, Cypher |
| RDBMS | PostgreSQL | 15+ | ACID, JSONB, mature, reliable |
| Search | Elasticsearch | 8.x | Full-text, time-series, scalable |
| Vector DB | ChromaDB / Pinecone | Latest | Semantic search, embeddings |
| Cache | Redis | 7.x | Fast, reliable, proven |
| Container | Docker | Latest | Portability, consistency |

---

### 17.2 Python Libraries

**Core Dependencies:**
```toml
[tool.poetry.dependencies]
python = "^3.10"

# Graph database
neo4j = "^5.13.0"

# Relational database (Production Mode)
sqlalchemy = "^2.0.0"
psycopg2-binary = "^2.9.9"
alembic = "^1.12.0"  # Migrations

# Search (Production Mode)
elasticsearch = "^8.11.0"

# Vector DB (Production Mode)
chromadb = "^0.4.18"  # Local
pinecone-client = "^2.2.4"  # Cloud

# Caching (Production Mode)
redis = "^5.0.1"

# Data validation
pydantic = "^2.5.0"

# Configuration
pydantic-settings = "^2.1.0"
python-dotenv = "^1.0.0"

# Observability
prometheus-client = "^0.19.0"
opentelemetry-api = "^1.21.0"
opentelemetry-sdk = "^1.21.0"
opentelemetry-exporter-otlp = "^1.21.0"
structlog = "^23.2.0"

# CLI
click = "^8.1.7"
rich = "^13.7.0"  # Beautiful CLI output

# Export formats
networkx = "^3.2.1"  # GraphML export
```

**Development Dependencies:**
```toml
[tool.poetry.group.dev.dependencies]
pytest = "^7.4.3"
pytest-cov = "^4.1.0"
pytest-asyncio = "^0.21.1"
mypy = "^1.7.1"
black = "^23.12.0"
ruff = "^0.1.8"
```

---

### 17.3 Infrastructure Technologies

**Simple Mode:**
- Docker Compose: Single Neo4j container
- Volume: Local filesystem

**Production Mode (Self-Hosted):**
- Docker Compose: 5 containers (Postgres, Neo4j, ES, ChromaDB, Redis)
- Volumes: Named volumes with backup strategy

**Production Mode (Cloud - AWS Example):**
- Compute: ECS Fargate or EKS
- Database: RDS PostgreSQL Multi-AZ
- Graph: Neo4j AuraDB
- Search: Amazon OpenSearch
- Vector: Pinecone Cloud
- Cache: ElastiCache Redis
- Monitoring: CloudWatch + X-Ray
- Networking: VPC, Security Groups, ALB

---

## 18. Comparison to Kosmos Paper

### 18.1 What Matches the Paper

**Architecture:**
- ✅ Polyglot persistence (PostgreSQL + Neo4j + ES + Vector DB)
- ✅ Structured world models as queryable databases
- ✅ PROV-O standard provenance
- ✅ Graph-based knowledge representation
- ✅ Multi-cycle research with persistent memory

**Capabilities:**
- ✅ Entity and relationship storage
- ✅ Provenance tracking
- ✅ Semantic search
- ✅ Natural language queries
- ✅ Export/import functionality

---

### 18.2 What Differs from the Paper

**Differences (By Design):**

1. **Simple Mode:**
   - Paper: Production polyglot from day 1
   - Implementation: Progressive path (Simple → Production)
   - Rationale: 90% of users don't need enterprise scale

2. **Project Management:**
   - Paper: Single continuous research session
   - Implementation: Multiple concurrent projects
   - Rationale: Real users work on multiple topics

3. **User Curation:**
   - Paper: Fully autonomous agents
   - Implementation: User-in-the-loop verification
   - Rationale: Users want control over knowledge quality

4. **Deployment:**
   - Paper: Production infrastructure assumed
   - Implementation: Local laptop → Cloud deployment path
   - Rationale: Lower barrier to entry

---

### 18.3 What's Enhanced Beyond the Paper

**Additional Features:**

1. **Multi-Project Support:**
   - Not described in paper
   - Essential for real-world usage

2. **Export/Import:**
   - Share knowledge graphs
   - Version control integration
   - Collaboration enablement

3. **Quality Management:**
   - Verification system
   - Annotation system
   - Duplicate detection
   - Quality metrics

4. **Educational Documentation:**
   - Architecture explanations
   - Design rationale
   - Scaling guidelines
   - Migration paths

---

### 18.4 Implementation Timeline Mapping

| Paper Capability | Simple Mode (Week 6) | Production Mode (Month 5) |
|------------------|----------------------|---------------------------|
| Persistent storage | ✅ Neo4j | ✅ Polyglot |
| Entity/Relationship | ✅ Full | ✅ Full |
| Provenance | ⚠️ Basic | ✅ PROV-O |
| Semantic search | ⚠️ Keyword | ✅ Vector |
| Query | ⚠️ Cypher | ✅ GraphRAG |
| Scale | ✅ 10K entities | ✅ 100K+ entities |
| Multi-user | ❌ No | ✅ Yes |

---

## 19. Next Steps and Follow-On Documents

### 19.1 This Architecture Document

**Status:** Complete and ready for review

**Provides:**
- Complete technical specifications
- All interfaces and contracts
- Database schemas
- Deployment options
- Security and performance architectures
- Architecture decision rationale

---

### 19.2 Follow-On Documents

**From this architecture.md, create:**

1. **implementation.md**
   - Derives: Detailed implementation tasks from architecture specs
   - Content: Sprint planning, file structure, testing procedures, deployment scripts
   - Timeline: Detailed week-by-week implementation plan

2. **User guides** (derived from API specifications):
   - getting_started.md
   - building_knowledge_base.md
   - production_deployment.md

3. **Tutorials** (derived from architecture patterns):
   - understanding_world_models.md
   - scaling_architecture.md

---

### 19.3 Review Checklist

Before implementation:
- [ ] Architecture team review
- [ ] Security review (especially Production Mode auth)
- [ ] Performance targets validated
- [ ] Cost estimates reviewed
- [ ] Documentation completeness check
- [ ] Alignment with requirements.md verified
- [ ] Comparison to paper validated

---

## 20. Document Metadata

**Version:** 1.0
**Last Updated:** November 2025
**Status:** Draft - Ready for Review
**Authors:** Kosmos Development Team

**Review History:**
- [ ] Technical Review
- [ ] Architecture Review
- [ ] Security Review
- [ ] Final Approval

**Dependencies:**
- requirements.md (complete)
- world_model_implementation_decision_framework.md (complete)
- objective.md (complete)

**Next Document:** implementation.md

---

**END OF ARCHITECTURE SPECIFICATION**


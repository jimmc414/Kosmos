"""
World Model: Persistent knowledge graph for accumulating research knowledge.

The World Model maintains a persistent knowledge graph that accumulates across
research sessions, enabling:
- Knowledge accumulation over weeks/months (not just per-query)
- Backup and restore of knowledge graphs
- Collaboration (share graphs with colleagues)
- Version control (track knowledge evolution)

TWO MODES:
-----------

**Simple Mode** (Default):
- Single Neo4j database
- Easy setup (Docker Compose)
- Suitable for <10K entities
- Used by 90% of researchers
- Week 1-2 implementation

**Production Mode** (Future - Phase 4):
- Polyglot architecture (PostgreSQL + Neo4j + Elasticsearch + Vector DB)
- Advanced features (semantic search, PROV-O provenance, GraphRAG)
- Enterprise scale (100K+ entities)
- Research organizations

ARCHITECTURE:
------------

The world model uses the Strategy pattern with abstract interfaces:

    WorldModelStorage (ABC)
         ↑
         ├─ Neo4jWorldModel (Simple Mode)
         └─ PolyglotWorldModel (Production Mode - Phase 4)

    Factory: get_world_model() → Returns appropriate implementation

This enables:
- Swapping backends without changing client code
- Testing with mock implementations
- Progressive enhancement (same API, different capabilities)

QUICK START:
-----------

    from kosmos.world_model import get_world_model, Entity, Relationship

    # Get world model (mode from config)
    wm = get_world_model()

    # Add entity
    paper = Entity(
        type="Paper",
        properties={
            "title": "Attention Is All You Need",
            "authors": ["Vaswani et al."],
            "year": 2017
        }
    )
    entity_id = wm.add_entity(paper)

    # Query entity
    retrieved = wm.get_entity(entity_id)

    # Export graph
    wm.export_graph("backup.json")

    # Import graph
    wm.import_graph("backup.json")

    # Statistics
    stats = wm.get_statistics()
    print(f"Entities: {stats['entity_count']}")

CLI COMMANDS:
------------

    kosmos graph info                   # Show statistics
    kosmos graph export backup.json     # Export graph
    kosmos graph import backup.json     # Import graph
    kosmos graph reset                  # Clear graph (DANGEROUS)

CONFIGURATION:
-------------

    # config.yaml or environment variables
    world_model:
      enabled: true              # Enable world model
      mode: simple               # "simple" or "production"
      project: my_research       # Default project namespace
      auto_save_interval: 300    # Auto-export every 5 min (0 = disabled)

EDUCATIONAL NOTES:
-----------------

This implementation demonstrates several design patterns:

1. **Abstract Base Class (ABC)**: Interface definitions in interface.py
2. **Strategy Pattern**: Different storage implementations, same interface
3. **Factory Pattern**: get_world_model() selects implementation
4. **Singleton Pattern**: Single world model instance per process
5. **Data Transfer Object**: Entity/Relationship models
6. **Builder Pattern**: Entity.from_dict() for deserialization

These patterns enable:
- Testability (mock implementations)
- Extensibility (add new backends)
- Maintainability (clear separation of concerns)

SEE ALSO:
--------

- docs/guides/persistent_knowledge_graphs.md - User guide
- docs/architecture/world_model_simple_mode.md - Architecture docs
- docs/planning/implementation.md - Full implementation plan
- docs/planning/implementation_mvp.md - MVP implementation (Week 1-2)

HISTORY:
-------

- Phase 1 (Week 1-3): Simple Mode with Neo4j
- Phase 2 (Week 7-10): Curation features (verification, annotations)
- Phase 3 (Week 11-13): Multi-project support
- Phase 4 (Week 14-19): Production Mode (polyglot architecture)
"""

from kosmos.world_model.interface import (
    EntityManager,
    ProvenanceTracker,
    WorldModelStorage,
)
from kosmos.world_model.models import (
    EXPORT_FORMAT_VERSION,
    Annotation,
    Entity,
    Relationship,
)

# Factory (Day 5 - COMPLETE)
from kosmos.world_model.factory import get_world_model, reset_world_model

__all__ = [
    # Interfaces
    "WorldModelStorage",
    "EntityManager",
    "ProvenanceTracker",
    # Models
    "Entity",
    "Relationship",
    "Annotation",
    "EXPORT_FORMAT_VERSION",
    # Factory
    "get_world_model",
    "reset_world_model",
]

__version__ = "0.1.0"  # World model module version

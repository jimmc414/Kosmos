"""Factory for creating world model instances.

This module provides a singleton factory pattern for creating and managing
world model instances. The factory supports two modes:

- **Simple Mode** (default): Uses Neo4j as the primary storage backend
- **Production Mode** (Phase 4): Uses polyglot persistence (PostgreSQL + Neo4j +
  Elasticsearch + Vector DB)

The factory reads configuration from kosmos.config.KosmosConfig and provides
a singleton instance to ensure consistent state across the application.

Design Pattern:
    Singleton Factory - Ensures only one world model instance exists per process.
    This is critical for maintaining consistency when multiple components access
    the knowledge graph.

Usage:
    >>> from kosmos.world_model import get_world_model, Entity
    >>>
    >>> # Get singleton instance (reads config)
    >>> wm = get_world_model()
    >>>
    >>> # Or override mode
    >>> wm = get_world_model(mode="simple")
    >>>
    >>> # Add entity
    >>> entity = Entity(type="Paper", properties={"title": "Test"})
    >>> entity_id = wm.add_entity(entity)
    >>>
    >>> # For testing: reset singleton
    >>> from kosmos.world_model import reset_world_model
    >>> reset_world_model()

Thread Safety:
    The current implementation is NOT thread-safe. If you need concurrent access,
    consider adding locks or using thread-local storage in Phase 4.
"""

import logging
from typing import Literal, Optional

from kosmos.world_model.interface import WorldModelStorage

logger = logging.getLogger(__name__)

# Global singleton instance
# This follows the same pattern as:
# - kosmos.knowledge.graph.get_knowledge_graph()
# - kosmos.knowledge.vector_db.get_vector_db()
# - kosmos.knowledge.embeddings.get_embedder()
_world_model: Optional[WorldModelStorage] = None


def get_world_model(
    mode: Optional[Literal["simple", "production"]] = None,
    reset: bool = False,
) -> WorldModelStorage:
    """Get or create the singleton world model instance.

    This factory function returns the singleton world model instance. On first call,
    it reads configuration from KosmosConfig and instantiates the appropriate
    implementation based on the mode setting.

    The singleton pattern ensures that all components in the application share the
    same world model instance, maintaining consistency in the knowledge graph state.

    Args:
        mode: Override the mode from config. Options:
            - "simple": Use Neo4jWorldModel (wraps existing KnowledgeGraph)
            - "production": Use PolyglotWorldModel (Phase 4 - not yet implemented)
            If None, reads from config.world_model.mode
        reset: Force recreation of the singleton instance. Useful when you need
            to change modes or reinitialize the connection.

    Returns:
        WorldModelStorage: The singleton world model instance implementing the
            WorldModelStorage interface.

    Raises:
        ValueError: If mode is not "simple" or "production"
        NotImplementedError: If mode="production" (Phase 4 not yet implemented)
        ImportError: If required dependencies are not installed

    Examples:
        >>> # Get instance with default config
        >>> wm = get_world_model()
        >>>
        >>> # Override to simple mode
        >>> wm = get_world_model(mode="simple")
        >>>
        >>> # Force reset (creates new instance)
        >>> wm = get_world_model(reset=True)
        >>>
        >>> # All calls return same instance (singleton)
        >>> wm1 = get_world_model()
        >>> wm2 = get_world_model()
        >>> assert wm1 is wm2  # True

    Note:
        The factory reads config on first instantiation. If you change config
        after getting the instance, you must call get_world_model(reset=True)
        to pick up the new settings.
    """
    global _world_model

    # Create instance if doesn't exist or reset requested
    if _world_model is None or reset:
        # Import config (lazy import to avoid circular dependencies)
        from kosmos.config import get_config

        config = get_config()

        # Determine mode (parameter overrides config)
        mode = mode or config.world_model.mode

        logger.info(f"Creating world model instance (mode={mode})")

        if mode == "simple":
            # Simple Mode: Use Neo4j via existing KnowledgeGraph
            from kosmos.world_model.simple import Neo4jWorldModel

            _world_model = Neo4jWorldModel()
            logger.info("✓ Neo4jWorldModel initialized (Simple Mode)")

        elif mode == "production":
            # Production Mode: Polyglot persistence (Phase 4)
            raise NotImplementedError(
                "Production Mode is not yet implemented (planned for Phase 4). "
                "This mode will provide:\n"
                "  - Polyglot persistence (PostgreSQL + Neo4j + Elasticsearch)\n"
                "  - Vector database integration for semantic search\n"
                "  - PROV-O provenance tracking\n"
                "  - GraphRAG query capabilities\n"
                "  - Enterprise scale (100K+ entities)\n\n"
                "For now, please use mode='simple' which supports up to 10K entities."
            )

        else:
            raise ValueError(
                f"Unknown world model mode: '{mode}'. "
                f"Valid options: 'simple', 'production'"
            )

    return _world_model


def reset_world_model() -> None:
    """Reset the singleton world model instance.

    This function clears the singleton instance, forcing the next call to
    get_world_model() to create a new instance. This is primarily useful for:

    - **Testing**: Ensure clean state between tests
    - **Mode switching**: Change from simple to production mode (Phase 4)
    - **Connection reset**: Reinitialize database connections

    The function safely calls close() on the existing instance before clearing
    the singleton, ensuring proper cleanup of resources like database connections.

    Examples:
        >>> wm1 = get_world_model()
        >>> reset_world_model()
        >>> wm2 = get_world_model()
        >>> assert wm1 is not wm2  # Different instances

    Note:
        This is automatically called by the test suite's autouse fixture
        (see tests/conftest.py) to ensure test isolation.
    """
    global _world_model

    if _world_model is not None:
        # Safely close existing instance
        try:
            _world_model.close()
            logger.debug("Closed existing world model instance")
        except Exception as e:
            # Don't fail reset if close() fails
            logger.warning(f"Error closing world model during reset: {e}")

    # Clear singleton
    _world_model = None
    logger.debug("World model singleton reset")

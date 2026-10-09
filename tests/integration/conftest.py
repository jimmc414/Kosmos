"""
Integration test configuration and fixtures.

Ensures all integration tests use correct environment settings.
"""

import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_integration_env():
    """Set up environment variables for integration tests."""
    # Force bolt:// protocol for Neo4j (py2neo requirement)
    os.environ['NEO4J_URI'] = 'bolt://localhost:7687'
    os.environ['NEO4J_USER'] = 'neo4j'
    os.environ['NEO4J_PASSWORD'] = 'kosmos-password'
    os.environ['NEO4J_DATABASE'] = 'neo4j'

    # Provider settings come from tests/conftest.py (litellm with a placeholder key);
    # ANTHROPIC_API_KEY is never set (MAP D-19).

    # Force config reload to pick up new environment variables
    from kosmos.config import get_config
    try:
        get_config(reload=True)
    except Exception:
        pass  # Ignore errors during config reload

    yield

    # No cleanup needed - environment persists for session


@pytest.fixture(autouse=True)
def reset_env_for_each_test():
    """
    Reset environment for each test to ensure Neo4j URI is bolt://.

    This prevents tests from inheriting stale config objects.
    """
    os.environ['NEO4J_URI'] = 'bolt://localhost:7687'
    yield

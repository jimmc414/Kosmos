"""
E2E Smoke Tests.

Fast sanity checks that verify core system functionality.
Each test should run in < 2 seconds.
"""

import pytest

from tests.e2e.conftest import requires_neo4j


pytestmark = [pytest.mark.e2e, pytest.mark.smoke]


class TestSmoke:
    """Quick sanity checks for core system functionality."""

    def test_config_loads(self):
        """Config loads without errors."""
        from kosmos.config import get_config

        config = get_config()
        assert config is not None
        assert config.llm_provider is not None

    def test_database_connection(self):
        """SQLite DB initializes."""
        from kosmos.config import get_config
        from kosmos.db import init_database, get_session

        config = get_config()
        init_database(config.database.normalized_url)
        with get_session() as session:
            assert session is not None

    @requires_neo4j
    def test_neo4j_connection(self):
        """Neo4j connects (skip if unavailable)."""
        from kosmos.world_model.factory import get_world_model, reset_world_model

        try:
            wm = get_world_model()
            stats = wm.get_statistics()
            assert "entity_count" in stats
        finally:
            reset_world_model()

    def test_metrics_collector_initializes(self):
        """MetricsCollector works."""
        from kosmos.core.metrics import MetricsCollector

        collector = MetricsCollector()
        assert collector is not None
        collector.reset()

    def test_world_model_factory(self):
        """InMemoryWorldModel returns valid instance."""
        from kosmos.world_model.in_memory import InMemoryWorldModel

        wm = InMemoryWorldModel()
        assert wm is not None
        assert hasattr(wm, 'add_entity')
        assert hasattr(wm, 'get_statistics')
        wm.reset()

    def test_cli_help(self):
        """CLI --help works."""
        from typer.testing import CliRunner
        from kosmos.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        # Check that help output contains expected content
        assert "kosmos" in result.stdout.lower() or "usage" in result.stdout.lower()

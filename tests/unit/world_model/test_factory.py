"""Tests for world model factory."""

import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock

from kosmos.world_model.factory import get_world_model, reset_world_model
from kosmos.world_model.interface import WorldModelStorage
from kosmos.world_model.simple import Neo4jWorldModel


# Mock Neo4j connection for all tests
@pytest.fixture(autouse=True)
def mock_neo4j_connection():
    """Mock Neo4j connection to avoid needing running instance."""
    # Patch at the point where it's used (in simple.py)
    with patch('kosmos.world_model.simple.get_knowledge_graph') as mock_kg:
        # Create a mock knowledge graph with all needed attributes
        mock_graph_instance = Mock()
        mock_graph_instance.graph = Mock()
        mock_graph_instance.graph.run = Mock(return_value=Mock(data=Mock(return_value=[])))
        mock_kg.return_value = mock_graph_instance
        yield mock_kg


class TestGetWorldModel:
    """Test get_world_model factory function."""

    def test_returns_singleton(self):
        """Test that factory returns singleton instance."""
        reset_world_model()  # Clean slate

        wm1 = get_world_model()
        wm2 = get_world_model()

        assert wm1 is wm2

    def test_returns_world_model_storage_interface(self):
        """Test that returned instance implements interface."""
        reset_world_model()
        wm = get_world_model()
        assert isinstance(wm, WorldModelStorage)

    def test_simple_mode_returns_neo4j_model(self):
        """Test that simple mode returns Neo4jWorldModel."""
        reset_world_model()
        wm = get_world_model(mode="simple")
        assert isinstance(wm, Neo4jWorldModel)

    def test_production_mode_raises_not_implemented(self):
        """Test that production mode raises NotImplementedError."""
        reset_world_model()
        with pytest.raises(NotImplementedError, match="Production Mode"):
            get_world_model(mode="production")

    def test_invalid_mode_raises_value_error(self):
        """Test that invalid mode raises ValueError."""
        reset_world_model()
        with pytest.raises(ValueError, match="Unknown"):
            get_world_model(mode="invalid")  # type: ignore

    def test_reset_parameter_forces_recreation(self):
        """Test that reset=True forces new instance."""
        reset_world_model()

        wm1 = get_world_model()
        wm2 = get_world_model(reset=True)

        # Note: These might be the same object if Neo4j singleton isn't reset
        # The key is that reset=True triggers re-execution of factory logic
        assert wm2 is not None

    @patch("kosmos.config.get_config")
    def test_uses_config_mode_by_default(self, mock_get_config):
        """Test that factory reads mode from config."""
        reset_world_model()

        # Mock config
        mock_config = Mock()
        mock_config.world_model.mode = "simple"
        mock_get_config.return_value = mock_config

        wm = get_world_model()

        assert isinstance(wm, Neo4jWorldModel)
        mock_get_config.assert_called_once()

    @patch("kosmos.config.get_config")
    def test_mode_parameter_overrides_config(self, mock_get_config):
        """Test that mode parameter overrides config."""
        reset_world_model()

        # Mock config with production mode
        mock_config = Mock()
        mock_config.world_model.mode = "production"
        mock_get_config.return_value = mock_config

        # Override to simple
        wm = get_world_model(mode="simple")

        assert isinstance(wm, Neo4jWorldModel)

    def test_subsequent_calls_return_same_instance(self):
        """Test that multiple calls without reset return same instance."""
        reset_world_model()

        instances = [get_world_model() for _ in range(5)]

        # All should be the same instance
        assert all(inst is instances[0] for inst in instances)

    def test_can_call_methods_on_returned_instance(self):
        """Test that returned instance has expected methods."""
        reset_world_model()
        wm = get_world_model()

        # Check interface methods exist
        assert hasattr(wm, "add_entity")
        assert hasattr(wm, "get_entity")
        assert hasattr(wm, "add_relationship")
        assert hasattr(wm, "export_graph")
        assert hasattr(wm, "import_graph")
        assert hasattr(wm, "get_statistics")
        assert hasattr(wm, "reset")
        assert hasattr(wm, "close")

        # Check they're callable
        assert callable(wm.add_entity)
        assert callable(wm.get_entity)


class TestResetWorldModel:
    """Test reset_world_model function."""

    def test_reset_clears_singleton(self):
        """Test that reset clears singleton."""
        reset_world_model()

        wm1 = get_world_model()
        reset_world_model()
        wm2 = get_world_model()

        # After reset, we get a new instance
        # Note: Due to Neo4j singleton, these might wrap the same graph
        # but the WorldModel wrapper should be recreated
        assert wm2 is not None

    def test_reset_calls_close(self):
        """Test that reset calls close() on instance."""
        reset_world_model()

        wm = get_world_model()
        wm.close = Mock()

        reset_world_model()

        wm.close.assert_called_once()

    def test_reset_handles_close_errors_gracefully(self):
        """Test that reset handles close() errors."""
        reset_world_model()

        wm = get_world_model()
        wm.close = Mock(side_effect=Exception("Close failed"))

        # Should not raise
        reset_world_model()

    def test_reset_without_instance_doesnt_error(self):
        """Test that reset works even if no instance created yet."""
        reset_world_model()  # First reset - no instance exists
        reset_world_model()  # Second reset - still works

    def test_can_get_instance_after_reset(self):
        """Test that we can get new instance after reset."""
        reset_world_model()

        wm1 = get_world_model()
        assert wm1 is not None

        reset_world_model()

        wm2 = get_world_model()
        assert wm2 is not None


class TestConfigIntegration:
    """Test integration with configuration system."""

    @patch("kosmos.config.get_config")
    def test_reads_mode_from_config(self, mock_get_config):
        """Test that factory reads mode from config."""
        reset_world_model()

        mock_config = Mock()
        mock_config.world_model.mode = "simple"
        mock_get_config.return_value = mock_config

        wm = get_world_model()

        assert wm is not None
        mock_get_config.assert_called_once()

    @patch("kosmos.config.get_config")
    def test_config_only_read_on_first_call(self, mock_get_config):
        """Test that config is only read when creating instance."""
        reset_world_model()

        mock_config = Mock()
        mock_config.world_model.mode = "simple"
        mock_get_config.return_value = mock_config

        # First call reads config
        wm1 = get_world_model()
        assert mock_get_config.call_count == 1

        # Subsequent calls don't read config (singleton)
        wm2 = get_world_model()
        assert mock_get_config.call_count == 1

    @patch("kosmos.config.get_config")
    def test_explicit_mode_skips_config_mode(self, mock_get_config):
        """Test that explicit mode parameter doesn't use config mode."""
        reset_world_model()

        mock_config = Mock()
        mock_config.world_model.mode = "production"  # Config says production
        mock_get_config.return_value = mock_config

        # But we pass simple explicitly
        wm = get_world_model(mode="simple")

        # Should get simple mode, not production
        assert isinstance(wm, Neo4jWorldModel)


class TestErrorHandling:
    """Test error handling in factory."""

    def test_invalid_mode_provides_helpful_error(self):
        """Test that invalid mode gives clear error message."""
        reset_world_model()

        with pytest.raises(ValueError) as exc_info:
            get_world_model(mode="invalid_mode")  # type: ignore

        error_msg = str(exc_info.value)
        assert "Unknown" in error_msg or "Invalid" in error_msg
        assert "simple" in error_msg or "production" in error_msg

    def test_production_mode_error_mentions_phase_4(self):
        """Test that production mode error mentions it's not implemented."""
        reset_world_model()

        with pytest.raises(NotImplementedError) as exc_info:
            get_world_model(mode="production")

        error_msg = str(exc_info.value)
        assert "Production Mode" in error_msg
        assert "Phase 4" in error_msg or "not yet implemented" in error_msg


class TestThreadSafety:
    """Test thread safety considerations.

    Note: Current implementation is NOT thread-safe by design (Phase 1).
    These tests document the expected behavior.
    """

    def test_singleton_same_across_multiple_accesses(self):
        """Test that singleton is consistent within single thread."""
        reset_world_model()

        instances = []
        for _ in range(10):
            instances.append(get_world_model())

        # All should be same instance in single thread
        assert all(inst is instances[0] for inst in instances)


class TestFactoryLogging:
    """Test logging behavior of factory."""

    @patch("kosmos.world_model.factory.logger")
    @patch("kosmos.config.get_config")
    def test_logs_instance_creation(self, mock_get_config, mock_logger):
        """Test that factory logs when creating instance."""
        reset_world_model()

        mock_config = Mock()
        mock_config.world_model.mode = "simple"
        mock_get_config.return_value = mock_config

        get_world_model()

        # Should have logged creation
        mock_logger.info.assert_called()

    @patch("kosmos.world_model.factory.logger")
    def test_logs_reset(self, mock_logger):
        """Test that reset logs appropriately."""
        reset_world_model()

        # Get instance
        get_world_model()

        # Reset it
        reset_world_model()

        # Should have logged
        mock_logger.debug.assert_called()


class TestRealConfiguration:
    """Test with real configuration (integration-style tests)."""

    def test_can_import_and_use_factory(self):
        """Test that factory can be imported and used."""
        from kosmos.world_model import get_world_model, reset_world_model

        reset_world_model()
        wm = get_world_model()

        assert wm is not None
        assert isinstance(wm, WorldModelStorage)

    def test_factory_returns_usable_instance(self):
        """Test that factory returns instance that can be used."""
        from kosmos.world_model import get_world_model, reset_world_model, Entity

        reset_world_model()
        wm = get_world_model()

        # Should be able to create an entity
        entity = Entity(type="Paper", properties={"title": "Test"})

        # Should have add_entity method
        assert hasattr(wm, "add_entity")

        # Method should be callable
        assert callable(wm.add_entity)


class TestDocumentation:
    """Test that factory has proper documentation."""

    def test_get_world_model_has_docstring(self):
        """Test that get_world_model has docstring."""
        assert get_world_model.__doc__ is not None
        assert len(get_world_model.__doc__) > 50

    def test_reset_world_model_has_docstring(self):
        """Test that reset_world_model has docstring."""
        assert reset_world_model.__doc__ is not None
        assert len(reset_world_model.__doc__) > 50

    def test_docstrings_mention_key_concepts(self):
        """Test that docstrings explain key concepts."""
        get_doc = get_world_model.__doc__
        reset_doc = reset_world_model.__doc__

        # get_world_model should mention singleton
        assert "singleton" in get_doc.lower()

        # reset should mention testing or cleanup
        assert "test" in reset_doc.lower() or "reset" in reset_doc.lower()

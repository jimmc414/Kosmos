"""
Unit tests for graph CLI commands.

Tests the knowledge graph management commands: info, export, import, reset.
"""

import pytest
import json
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import MagicMock, patch, mock_open

from kosmos.cli.main import app


class TestGraphInfoCommand:
    """Test graph info/stats command."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_world_model(self):
        """Create mock world model."""
        mock_wm = MagicMock()
        mock_wm.get_statistics.return_value = {
            'entity_count': 150,
            'relationship_count': 320,
            'annotation_count': 45,
            'entity_types': {
                'Paper': 50,
                'Concept': 40,
                'Method': 30,
                'Author': 30
            },
            'relationship_types': {
                'CITES': 100,
                'DESCRIBES': 80,
                'AUTHORED_BY': 50,
                'USES_METHOD': 90
            }
        }
        return mock_wm

    @patch('kosmos.world_model.get_world_model')
    def test_info_default_shows_stats(self, mock_get_wm, mock_world_model, runner):
        """Test that default command shows statistics."""
        mock_get_wm.return_value = mock_world_model

        result = runner.invoke(app, ["graph"])

        assert result.exit_code == 0
        assert "150" in result.stdout  # Entity count
        assert "320" in result.stdout  # Relationship count

    @patch('kosmos.world_model.get_world_model')
    def test_info_with_stats_flag(self, mock_get_wm, mock_world_model, runner):
        """Test --stats flag shows statistics."""
        mock_get_wm.return_value = mock_world_model

        result = runner.invoke(app, ["graph", "--stats"])

        assert result.exit_code == 0
        assert "150" in result.stdout

    @patch('kosmos.world_model.get_world_model')
    def test_info_with_empty_graph(self, mock_get_wm, runner):
        """Test info command with empty graph."""
        mock_wm = MagicMock()
        mock_wm.get_statistics.return_value = {
            'entity_count': 0,
            'relationship_count': 0,
            'annotation_count': 0,
            'entity_types': {},
            'relationship_types': {}
        }
        mock_get_wm.return_value = mock_wm

        result = runner.invoke(app, ["graph"])

        assert result.exit_code == 0
        assert "0" in result.stdout

    @patch('kosmos.world_model.get_world_model')
    def test_info_shows_entity_types(self, mock_get_wm, mock_world_model, runner):
        """Test that entity types are displayed."""
        mock_get_wm.return_value = mock_world_model

        result = runner.invoke(app, ["graph", "--info"])

        assert result.exit_code == 0
        assert "Paper" in result.stdout or "Concept" in result.stdout


class TestGraphExportCommand:
    """Test graph export command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def mock_world_model(self):
        """Create mock world model for export."""
        mock_wm = MagicMock()
        mock_wm.get_statistics.return_value = {
            'entity_count': 100,
            'relationship_count': 250,
            'annotation_count': 20
        }
        mock_wm.export_graph = MagicMock()
        return mock_wm

    @patch('kosmos.world_model.get_world_model')
    def test_export_success(self, mock_get_wm, mock_world_model, runner, tmp_path):
        """Test successful export."""
        mock_get_wm.return_value = mock_world_model
        export_file = tmp_path / "test_export.json"

        # Create file to simulate export
        def side_effect(filepath):
            Path(filepath).write_text('{"version": "1.0", "nodes": []}')

        mock_world_model.export_graph.side_effect = side_effect

        result = runner.invoke(app, ["graph", "--export", str(export_file)])

        assert result.exit_code == 0
        assert "Export" in result.stdout or "export" in result.stdout
        mock_world_model.export_graph.assert_called_once_with(str(export_file))

    @patch('kosmos.world_model.get_world_model')
    def test_export_creates_parent_directory(self, mock_get_wm, mock_world_model, runner, tmp_path):
        """Test export creates parent directory if needed."""
        mock_get_wm.return_value = mock_world_model
        export_file = tmp_path / "subdir" / "nested" / "test.json"

        def side_effect(filepath):
            Path(filepath).write_text('{}')

        mock_world_model.export_graph.side_effect = side_effect

        result = runner.invoke(app, ["graph", "--export", str(export_file)])

        assert result.exit_code == 0
        assert export_file.parent.exists()

    @patch('kosmos.world_model.get_world_model')
    def test_export_shows_statistics(self, mock_get_wm, mock_world_model, runner, tmp_path):
        """Test export displays entity/relationship counts."""
        mock_get_wm.return_value = mock_world_model
        export_file = tmp_path / "test.json"

        def side_effect(filepath):
            Path(filepath).write_text('{}')

        mock_world_model.export_graph.side_effect = side_effect

        result = runner.invoke(app, ["graph", "--export", str(export_file)])

        assert result.exit_code == 0
        assert "100" in result.stdout  # Entity count
        assert "250" in result.stdout  # Relationship count

    @patch('kosmos.world_model.get_world_model')
    def test_export_error_handling(self, mock_get_wm, runner, tmp_path):
        """Test export error handling."""
        mock_wm = MagicMock()
        mock_wm.get_statistics.return_value = {'entity_count': 10, 'relationship_count': 20}
        mock_wm.export_graph.side_effect = Exception("Export failed")
        mock_get_wm.return_value = mock_wm

        export_file = tmp_path / "test.json"
        result = runner.invoke(app, ["graph", "--export", str(export_file)])

        assert result.exit_code == 1
        assert "failed" in result.stdout.lower() or "error" in result.stdout.lower()


class TestGraphImportCommand:
    """Test graph import command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def mock_world_model(self):
        """Create mock world model for import."""
        mock_wm = MagicMock()
        # Before import stats
        mock_wm.get_statistics.side_effect = [
            {'entity_count': 50, 'relationship_count': 100, 'annotation_count': 10},  # Before
            {'entity_count': 150, 'relationship_count': 300, 'annotation_count': 30}  # After
        ]
        mock_wm.import_graph = MagicMock()
        return mock_wm

    @pytest.fixture
    def sample_export_file(self, tmp_path):
        """Create sample export file."""
        export_data = {
            "version": "1.0",
            "exported_at": "2024-01-01T00:00:00",
            "source": "kosmos",
            "statistics": {
                "nodes": 100,
                "relationships": 200
            },
            "nodes": [],
            "relationships": []
        }
        export_file = tmp_path / "sample.json"
        export_file.write_text(json.dumps(export_data))
        return export_file

    @patch('kosmos.world_model.get_world_model')
    def test_import_success(self, mock_get_wm, mock_world_model, runner, sample_export_file):
        """Test successful import."""
        mock_get_wm.return_value = mock_world_model

        result = runner.invoke(app, ["graph", "--import", str(sample_export_file)])

        assert result.exit_code == 0
        assert "Import" in result.stdout or "import" in result.stdout
        mock_world_model.import_graph.assert_called_once_with(str(sample_export_file), clear=False)

    @patch('kosmos.world_model.get_world_model')
    @patch('kosmos.cli.commands.graph.confirm_action')
    def test_import_with_clear_flag(self, mock_confirm, mock_get_wm, mock_world_model, runner, sample_export_file):
        """Test import with --clear flag."""
        mock_get_wm.return_value = mock_world_model
        mock_confirm.return_value = True  # User confirms

        result = runner.invoke(app, ["graph", "--import", str(sample_export_file), "--clear"])

        assert result.exit_code == 0
        mock_world_model.import_graph.assert_called_once_with(str(sample_export_file), clear=True)

    @patch('kosmos.world_model.get_world_model')
    @patch('kosmos.cli.commands.graph.confirm_action')
    def test_import_clear_cancellation(self, mock_confirm, mock_get_wm, mock_world_model, runner, sample_export_file):
        """Test import with --clear can be cancelled."""
        mock_get_wm.return_value = mock_world_model
        mock_confirm.return_value = False  # User cancels

        result = runner.invoke(app, ["graph", "--import", str(sample_export_file), "--clear"])

        # Should exit cleanly without calling import
        assert result.exit_code == 0
        mock_world_model.import_graph.assert_not_called()

    @patch('kosmos.world_model.get_world_model')
    def test_import_nonexistent_file(self, mock_get_wm, runner):
        """Test import with non-existent file."""
        mock_wm = MagicMock()
        mock_get_wm.return_value = mock_wm

        result = runner.invoke(app, ["graph", "--import", "nonexistent.json"])

        assert result.exit_code == 1
        assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()

    @patch('kosmos.world_model.get_world_model')
    def test_import_shows_statistics(self, mock_get_wm, mock_world_model, runner, sample_export_file):
        """Test import shows before/after statistics."""
        mock_get_wm.return_value = mock_world_model

        result = runner.invoke(app, ["graph", "--import", str(sample_export_file)])

        assert result.exit_code == 0
        # Should show entity counts (before: 50, after: 150)
        assert "150" in result.stdout or "300" in result.stdout


class TestGraphResetCommand:
    """Test graph reset command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def mock_world_model(self):
        """Create mock world model for reset."""
        mock_wm = MagicMock()
        mock_wm.get_statistics.return_value = {
            'entity_count': 500,
            'relationship_count': 1200,
            'annotation_count': 150
        }
        mock_wm.reset = MagicMock()
        return mock_wm

    @patch('kosmos.world_model.get_world_model')
    @patch('kosmos.cli.commands.graph.confirm_action')
    def test_reset_with_confirmation(self, mock_confirm, mock_get_wm, mock_world_model, runner):
        """Test reset with confirmation."""
        mock_get_wm.return_value = mock_world_model
        # First and second confirmations
        mock_confirm.side_effect = [True, True]

        result = runner.invoke(app, ["graph", "--reset"])

        assert result.exit_code == 0
        assert mock_confirm.call_count == 2  # Double confirmation
        mock_world_model.reset.assert_called_once()

    @patch('kosmos.world_model.get_world_model')
    @patch('kosmos.cli.commands.graph.confirm_action')
    def test_reset_cancellation_first(self, mock_confirm, mock_get_wm, mock_world_model, runner):
        """Test reset cancelled at first confirmation."""
        mock_get_wm.return_value = mock_world_model
        mock_confirm.return_value = False  # User cancels

        result = runner.invoke(app, ["graph", "--reset"])

        assert result.exit_code == 0
        mock_world_model.reset.assert_not_called()

    @patch('kosmos.world_model.get_world_model')
    @patch('kosmos.cli.commands.graph.confirm_action')
    def test_reset_cancellation_second(self, mock_confirm, mock_get_wm, mock_world_model, runner):
        """Test reset cancelled at second confirmation."""
        mock_get_wm.return_value = mock_world_model
        # First confirms, second cancels
        mock_confirm.side_effect = [True, False]

        result = runner.invoke(app, ["graph", "--reset"])

        assert result.exit_code == 0
        mock_world_model.reset.assert_not_called()

    @patch('kosmos.world_model.get_world_model')
    def test_reset_empty_graph(self, mock_get_wm, runner):
        """Test reset with empty graph."""
        mock_wm = MagicMock()
        mock_wm.get_statistics.return_value = {
            'entity_count': 0,
            'relationship_count': 0,
            'annotation_count': 0
        }
        mock_wm.reset = MagicMock()
        mock_get_wm.return_value = mock_wm

        result = runner.invoke(app, ["graph", "--reset"])

        assert result.exit_code == 0
        # Should not attempt reset if already empty
        mock_wm.reset.assert_not_called()


class TestGraphCommandIntegration:
    """Integration tests for graph commands."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def mock_world_model(self):
        """Create comprehensive mock world model."""
        mock_wm = MagicMock()
        mock_wm.get_statistics.return_value = {
            'entity_count': 100,
            'relationship_count': 250,
            'annotation_count': 30,
            'entity_types': {'Paper': 50, 'Concept': 50},
            'relationship_types': {'CITES': 125, 'DESCRIBES': 125}
        }
        mock_wm.export_graph = MagicMock()
        mock_wm.import_graph = MagicMock()
        mock_wm.reset = MagicMock()
        return mock_wm

    @patch('kosmos.world_model.get_world_model')
    def test_export_import_workflow(self, mock_get_wm, mock_world_model, runner, tmp_path):
        """Test complete export/import workflow."""
        mock_get_wm.return_value = mock_world_model
        export_file = tmp_path / "workflow_test.json"

        # Create file for export
        def export_side_effect(filepath):
            Path(filepath).write_text('{"version": "1.0"}')

        mock_world_model.export_graph.side_effect = export_side_effect

        # Export
        result = runner.invoke(app, ["graph", "--export", str(export_file)])
        assert result.exit_code == 0
        assert export_file.exists()

        # Import
        result = runner.invoke(app, ["graph", "--import", str(export_file)])
        assert result.exit_code == 0

    @patch('kosmos.world_model.get_world_model')
    def test_multiple_operations_in_sequence(self, mock_get_wm, mock_world_model, runner, tmp_path):
        """Test multiple graph operations."""
        mock_get_wm.return_value = mock_world_model
        export_file = tmp_path / "multi_op.json"

        def export_side_effect(filepath):
            Path(filepath).write_text('{}')

        mock_world_model.export_graph.side_effect = export_side_effect

        # Info
        result = runner.invoke(app, ["graph"])
        assert result.exit_code == 0

        # Export
        result = runner.invoke(app, ["graph", "--export", str(export_file)])
        assert result.exit_code == 0

        # Import
        result = runner.invoke(app, ["graph", "--import", str(export_file)])
        assert result.exit_code == 0

    @patch('kosmos.world_model.get_world_model')
    def test_command_error_handling(self, mock_get_wm, runner):
        """Test graceful error handling."""
        mock_get_wm.side_effect = Exception("World model initialization failed")

        result = runner.invoke(app, ["graph"])

        assert result.exit_code == 1
        assert "failed" in result.stdout.lower() or "error" in result.stdout.lower()


class TestGraphCommandHelp:
    """Test graph command help and documentation."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_graph_command_registered(self, runner):
        """Test graph command is registered."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        # Graph command should be listed
        assert "graph" in result.stdout.lower()

    def test_graph_help_shows_options(self, runner):
        """Test graph help shows all options."""
        result = runner.invoke(app, ["graph", "--help"])

        # Should show at least some option flags
        assert result.exit_code == 0 or "--help" in result.stdout.lower()

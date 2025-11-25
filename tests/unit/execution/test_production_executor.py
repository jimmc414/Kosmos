"""
Tests for production executor.

Tests the main executor interface combining container management,
Jupyter execution, and package resolution.
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from kosmos.execution.production_executor import (
    ProductionExecutor,
    ProductionConfig,
    execute_code_safely,
)
from kosmos.execution.jupyter_client import ExecutionResult, ExecutionStatus


class TestProductionConfig:
    """Tests for ProductionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ProductionConfig()

        assert config.memory_limit == "4g"
        assert config.cpu_limit == 2.0
        assert config.timeout_seconds == 600
        assert config.pool_size == 3
        assert config.auto_install_packages is True
        assert config.network_enabled is False
        assert config.readonly_filesystem is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ProductionConfig(
            memory_limit="8g",
            cpu_limit=4.0,
            timeout_seconds=1200,
            pool_size=5,
            network_enabled=True
        )

        assert config.memory_limit == "8g"
        assert config.cpu_limit == 4.0
        assert config.timeout_seconds == 1200
        assert config.pool_size == 5
        assert config.network_enabled is True


class TestProductionExecutorInit:
    """Tests for ProductionExecutor initialization."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        executor = ProductionExecutor()

        assert executor.config is not None
        assert executor._initialized is False
        assert executor._execution_count == 0

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = ProductionConfig(memory_limit="16g")
        executor = ProductionExecutor(config=config)

        assert executor.config.memory_limit == "16g"


class TestProductionExecutorHealthCheck:
    """Tests for executor health check."""

    def test_health_check_not_initialized(self):
        """Test health check when not initialized."""
        executor = ProductionExecutor()

        # Run synchronously for test
        async def check():
            return await executor.check_health()

        health = asyncio.get_event_loop().run_until_complete(check())

        assert health["status"] == "not_initialized"
        assert health["pool"] is None

    @patch('kosmos.execution.production_executor.DockerManager')
    def test_health_check_initialized(self, mock_manager_class):
        """Test health check when initialized."""
        mock_manager = Mock()
        mock_manager.initialize_pool = AsyncMock()
        mock_manager.get_pool_stats.return_value = {
            "total": 3,
            "ready": 2,
            "in_use": 1,
            "unhealthy": 0,
            "target_size": 3
        }
        mock_manager_class.return_value = mock_manager

        executor = ProductionExecutor()
        executor._initialized = True
        executor._docker_manager = mock_manager

        async def check():
            return await executor.check_health()

        health = asyncio.get_event_loop().run_until_complete(check())

        assert health["status"] == "healthy"
        assert health["initialized"] is True
        assert health["pool"]["ready"] == 2


class TestExecutionResult:
    """Tests for ExecutionResult from jupyter_client."""

    def test_successful_result(self):
        """Test successful execution result."""
        result = ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            stdout="Hello, World!",
            stderr="",
            execution_time=0.5,
            return_value={"answer": 42}
        )

        assert result.success is True
        assert result.stdout == "Hello, World!"
        assert result.return_value == {"answer": 42}
        assert result.error_message is None

    def test_failed_result(self):
        """Test failed execution result."""
        result = ExecutionResult(
            status=ExecutionStatus.FAILED,
            error_message="NameError: name 'x' is not defined",
            error_traceback="Traceback...",
            execution_time=0.1
        )

        assert result.success is False
        assert "NameError" in result.error_message

    def test_timeout_result(self):
        """Test timeout execution result."""
        result = ExecutionResult(
            status=ExecutionStatus.TIMEOUT,
            error_message="Execution timed out after 300s"
        )

        assert result.success is False
        assert result.status == ExecutionStatus.TIMEOUT

    def test_result_to_dict(self):
        """Test conversion to dictionary."""
        result = ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            stdout="test output",
            execution_time=1.5
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["status"] == "completed"
        assert result_dict["stdout"] == "test output"
        assert result_dict["execution_time"] == 1.5


class TestProductionExecutorMocked:
    """Tests for ProductionExecutor with mocked dependencies."""

    @patch('kosmos.execution.production_executor.DockerManager')
    @patch('kosmos.execution.production_executor.JupyterClient')
    @patch('kosmos.execution.production_executor.PackageResolver')
    @pytest.mark.asyncio
    async def test_execute_code_success(self, mock_resolver_class, mock_jupyter_class, mock_manager_class):
        """Test successful code execution."""
        # Setup mocks
        mock_manager = Mock()
        mock_manager.initialize_pool = AsyncMock()
        mock_manager.get_container = AsyncMock(return_value=Mock(container_id="test-123"))
        mock_manager.release_container = AsyncMock()
        mock_manager.client = Mock()
        mock_manager_class.return_value = mock_manager

        mock_jupyter = Mock()
        mock_jupyter.execute_code = AsyncMock(return_value=ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            stdout="2",
            return_value={"result": 2}
        ))
        mock_jupyter_class.return_value = mock_jupyter

        mock_resolver = Mock()
        mock_resolver.ensure_dependencies = AsyncMock(return_value=(True, []))
        mock_resolver_class.return_value = mock_resolver

        # Execute
        executor = ProductionExecutor()
        await executor.initialize()

        result = await executor.execute_code("x = 1 + 1\nresults = {'result': x}")

        assert result.success is True
        assert result.return_value == {"result": 2}

        # Verify container was released
        mock_manager.release_container.assert_called_once()

    @patch('kosmos.execution.production_executor.DockerManager')
    @patch('kosmos.execution.production_executor.JupyterClient')
    @pytest.mark.asyncio
    async def test_execute_code_failure(self, mock_jupyter_class, mock_manager_class):
        """Test failed code execution."""
        # Setup mocks
        mock_manager = Mock()
        mock_manager.initialize_pool = AsyncMock()
        mock_manager.get_container = AsyncMock(return_value=Mock(container_id="test-123"))
        mock_manager.release_container = AsyncMock()
        mock_manager.client = Mock()
        mock_manager_class.return_value = mock_manager

        mock_jupyter = Mock()
        mock_jupyter.execute_code = AsyncMock(return_value=ExecutionResult(
            status=ExecutionStatus.FAILED,
            error_message="ZeroDivisionError: division by zero"
        ))
        mock_jupyter_class.return_value = mock_jupyter

        # Execute
        config = ProductionConfig(auto_install_packages=False)
        executor = ProductionExecutor(config)
        await executor.initialize()

        result = await executor.execute_code("x = 1 / 0")

        assert result.success is False
        assert "ZeroDivisionError" in result.error_message

    @patch('kosmos.execution.production_executor.DockerManager')
    @pytest.mark.asyncio
    async def test_cleanup(self, mock_manager_class):
        """Test executor cleanup."""
        mock_manager = Mock()
        mock_manager.initialize_pool = AsyncMock()
        mock_manager.cleanup = AsyncMock()
        mock_manager_class.return_value = mock_manager

        executor = ProductionExecutor()
        await executor.initialize()

        assert executor._initialized is True

        await executor.cleanup()

        assert executor._initialized is False
        mock_manager.cleanup.assert_called_once()

    @patch('kosmos.execution.production_executor.DockerManager')
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_manager_class):
        """Test async context manager."""
        mock_manager = Mock()
        mock_manager.initialize_pool = AsyncMock()
        mock_manager.cleanup = AsyncMock()
        mock_manager_class.return_value = mock_manager

        async with ProductionExecutor() as executor:
            assert executor._initialized is True

        # Cleanup should be called after context exit
        mock_manager.cleanup.assert_called_once()

    @patch('kosmos.execution.production_executor.DockerManager')
    @pytest.mark.asyncio
    async def test_execution_count_tracking(self, mock_manager_class):
        """Test execution count is tracked."""
        mock_manager = Mock()
        mock_manager.initialize_pool = AsyncMock()
        mock_manager.get_container = AsyncMock(return_value=Mock(container_id="test"))
        mock_manager.release_container = AsyncMock()
        mock_manager.client = Mock()
        mock_manager_class.return_value = mock_manager

        config = ProductionConfig(auto_install_packages=False)
        executor = ProductionExecutor(config)
        await executor.initialize()

        with patch('kosmos.execution.production_executor.JupyterClient') as mock_jupyter_class:
            mock_jupyter = Mock()
            mock_jupyter.execute_code = AsyncMock(return_value=ExecutionResult(
                status=ExecutionStatus.COMPLETED
            ))
            mock_jupyter_class.return_value = mock_jupyter

            await executor.execute_code("x = 1")
            await executor.execute_code("y = 2")
            await executor.execute_code("z = 3")

        assert executor._execution_count == 3


class TestExecutionStatus:
    """Tests for ExecutionStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.RUNNING.value == "running"
        assert ExecutionStatus.COMPLETED.value == "completed"
        assert ExecutionStatus.FAILED.value == "failed"
        assert ExecutionStatus.TIMEOUT.value == "timeout"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

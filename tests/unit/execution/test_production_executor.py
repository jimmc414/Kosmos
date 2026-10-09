"""
Tests for the jupyter_client result types.

The ProductionExecutor tests moved to archive/code/tests with the module (VIAB#P3-4).
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from kosmos.execution.jupyter_client import ExecutionResult, ExecutionStatus


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

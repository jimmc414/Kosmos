"""
Tests for code execution system.

Tests code execution, retry logic, error handling, output capture, and sandbox integration.
"""

import contextlib
import io
import json
import math

import pytest
from unittest.mock import Mock, patch, MagicMock
from kosmos.execution.sandbox import SandboxExecutionResult
from kosmos.execution.executor import (
    CodeExecutor,
    ExecutionResult,
    CodeValidator,
    RetryStrategy,
    execute_protocol_code
)


# Fixtures

@pytest.fixture
def executor():
    """Create basic code executor (no sandbox for unit tests)."""
    return CodeExecutor(max_retries=3, retry_delay=0.1, use_sandbox=False)


@pytest.fixture
def simple_code():
    """Simple valid code."""
    return "x = 1 + 1\nresults = {'value': x}"


@pytest.fixture
def code_with_output():
    """Code that produces stdout output."""
    return "print('Hello, World!')\nresults = {'status': 'success'}"


@pytest.fixture
def code_with_error():
    """Code that raises an error."""
    return "x = 1 / 0\nresults = {}"


@pytest.fixture
def code_with_import():
    """Code with imports."""
    return "import numpy as np\nx = np.array([1, 2, 3])\nresults = {'mean': float(np.mean(x))}"


# Basic Execution Tests

class TestBasicExecution:
    """Tests for basic code execution."""

    def test_execute_simple_code(self, executor, simple_code):
        """Test execution of simple code."""
        result = executor.execute(simple_code)

        assert result.success is True
        assert result.return_value == {'value': 2}
        assert result.error is None
        assert result.execution_time > 0

    def test_execute_code_with_output(self, executor, code_with_output):
        """Test execution captures stdout."""
        result = executor.execute(code_with_output)

        assert result.success is True
        assert "Hello, World!" in result.stdout
        assert result.return_value == {'status': 'success'}

    def test_execute_code_with_numpy(self, executor, code_with_import):
        """Test execution with imports."""
        result = executor.execute(code_with_import)

        assert result.success is True
        assert result.return_value == {'mean': 2.0}

    def test_execute_code_with_local_vars(self, executor):
        """Test execution with local variables."""
        code = "y = x * 2\nresults = {'y': y}"
        local_vars = {'x': 5}

        result = executor.execute(code, local_vars=local_vars)

        assert result.success is True
        assert result.return_value == {'y': 10}

    def test_execution_result_to_dict(self, executor, simple_code):
        """Test converting ExecutionResult to dict."""
        result = executor.execute(simple_code)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert 'success' in result_dict
        assert 'return_value' in result_dict
        assert 'execution_time' in result_dict
        assert result_dict['success'] is True


# Error Handling Tests

class TestErrorHandling:
    """Tests for error handling during execution."""

    def test_execute_code_with_division_by_zero(self, executor, code_with_error):
        """Test execution handles division by zero."""
        result = executor.execute(code_with_error)

        assert result.success is False
        assert result.error is not None
        assert "division by zero" in result.error.lower() or "ZeroDivisionError" in result.error_type
        assert result.error_type == "ZeroDivisionError"

    def test_execute_code_with_syntax_error(self, executor):
        """Test execution handles syntax errors."""
        invalid_code = "x = [1, 2, 3\nresults = {}"

        result = executor.execute(invalid_code)

        assert result.success is False
        assert result.error is not None
        assert result.error_type == "SyntaxError"

    def test_execute_code_with_name_error(self, executor):
        """Test execution handles undefined variables."""
        code = "results = {'value': undefined_variable}"

        result = executor.execute(code)

        assert result.success is False
        assert result.error_type == "NameError"

    def test_error_includes_traceback(self, executor, code_with_error):
        """Test error result includes traceback info."""
        result = executor.execute(code_with_error)

        assert result.success is False
        assert result.stderr is not None
        assert len(result.stderr) > 0


# Retry Logic Tests

class TestRetryLogic:
    """Tests for execution retry logic."""

    def test_retry_on_error_when_enabled(self, executor):
        """Test retry logic executes on error."""
        code = "import random\nif random.random() > 0.5:\n  raise ValueError('Random error')\nresults = {}"

        # With retries, should eventually succeed or fail after max attempts
        result = executor.execute(code, retry_on_error=True)

        # Either succeeded or failed after retries
        assert isinstance(result, ExecutionResult)

    def test_no_retry_when_disabled(self, executor, code_with_error):
        """Test no retry when retry_on_error=False."""
        result = executor.execute(code_with_error, retry_on_error=False)

        # Should fail immediately
        assert result.success is False
        assert result.error_type == "ZeroDivisionError"

    def test_retry_strategy_should_retry(self):
        """Test RetryStrategy.should_retry logic."""
        strategy = RetryStrategy(max_retries=3)

        # Should retry on runtime errors
        assert strategy.should_retry(1, "ValueError") is True

        # Should not retry on syntax errors
        assert strategy.should_retry(1, "SyntaxError") is False

        # Should not retry after max attempts
        assert strategy.should_retry(3, "ValueError") is False

    def test_retry_strategy_exponential_backoff(self):
        """Test exponential backoff delay."""
        strategy = RetryStrategy(base_delay=1.0)

        assert strategy.get_delay(1) == 1.0
        assert strategy.get_delay(2) == 2.0
        assert strategy.get_delay(3) == 4.0


# Output Capture Tests

class TestOutputCapture:
    """Tests for stdout/stderr capture."""

    def test_capture_stdout(self, executor):
        """Test stdout capture."""
        code = "print('Line 1')\nprint('Line 2')\nresults = {}"

        result = executor.execute(code)

        assert "Line 1" in result.stdout
        assert "Line 2" in result.stdout

    def test_capture_stderr(self, executor):
        """Test stderr capture via error."""
        code = "raise ValueError('Error message')"

        result = executor.execute(code)

        assert result.success is False
        assert "Error message" in result.stderr or "Error message" in result.error

    def test_separate_stdout_and_stderr(self, executor):
        """Test stdout and stderr are separate."""
        code = "print('stdout message')\nraise ValueError('stderr message')"

        result = executor.execute(code)

        assert "stdout message" in result.stdout
        assert "stderr" in result.stderr.lower() or "stderr message" in result.error


# Return Value Extraction Tests

class TestReturnValueExtraction:
    """Tests for extracting return values."""

    def test_extract_results_variable(self, executor):
        """Test extraction of 'results' variable."""
        code = "results = {'a': 1, 'b': 2}"

        result = executor.execute(code)

        assert result.return_value == {'a': 1, 'b': 2}

    def test_extract_result_variable(self, executor):
        """Test extraction of 'result' variable (singular)."""
        code = "result = {'value': 42}"

        result = executor.execute(code)

        assert result.return_value == {'value': 42}

    def test_results_preferred_over_result(self, executor):
        """Test 'results' is preferred over 'result'."""
        code = "result = {'wrong': 1}\nresults = {'correct': 2}"

        result = executor.execute(code)

        assert result.return_value == {'correct': 2}

    def test_no_return_value_if_not_set(self, executor):
        """Test return value is None if not set."""
        code = "x = 1 + 1"

        result = executor.execute(code)

        assert result.return_value is None


# Code Validation Tests

class TestCodeValidation:
    """Tests for code validation (instance-based CodeValidator from safety module)."""

    def test_validate_safe_code(self):
        """Test validation accepts safe code."""
        safe_code = "import numpy as np\nx = np.array([1, 2, 3])\nresults = {'mean': np.mean(x)}"

        validator = CodeValidator(allow_file_read=True)
        report = validator.validate(safe_code)

        assert report.passed is True
        assert len(report.violations) == 0

    def test_validate_dangerous_imports(self):
        """Test validation rejects dangerous imports."""
        dangerous_code = "import os\nos.system('rm -rf /')"

        validator = CodeValidator()
        report = validator.validate(dangerous_code)

        assert report.passed is False
        assert len(report.violations) > 0
        assert any('os' in v.message.lower() for v in report.violations)

    def test_validate_dangerous_operations(self):
        """Test validation rejects dangerous operations."""
        dangerous_code = "eval('malicious_code()')"

        validator = CodeValidator()
        report = validator.validate(dangerous_code)

        assert report.passed is False
        assert len(report.violations) > 0

    def test_validate_syntax_errors(self):
        """Test validation catches syntax errors."""
        invalid_code = "x = [1, 2, 3"

        validator = CodeValidator()
        report = validator.validate(invalid_code)

        assert report.passed is False
        assert len(report.violations) > 0

    def test_validate_file_read_allowed(self):
        """Test file read is allowed with flag."""
        code = "with open('data.csv', 'r') as f:\n  data = f.read()"

        validator = CodeValidator(allow_file_read=True)
        report = validator.validate(code)

        # Should be valid or have warnings
        assert report.passed is True or len(report.warnings) > 0

    def test_validate_file_write_rejected(self):
        """Test file write is rejected."""
        code = "with open('output.txt', 'w') as f:\n  f.write('data')"

        validator = CodeValidator(allow_file_read=True, allow_file_write=False)
        report = validator.validate(code)

        assert report.passed is False


# Execute with Data Tests

class TestExecuteWithData:
    """Tests for execute_with_data method."""

    def test_execute_with_data_path(self, executor, tmp_path):
        """Test execute_with_data provides data_path."""
        # Create temporary data file
        data_file = tmp_path / "test_data.csv"
        data_file.write_text("a,b,c\n1,2,3\n4,5,6")

        code = "results = {'data_path': data_path}"

        result = executor.execute_with_data(code, str(data_file))

        assert result.success is True
        assert result.return_value['data_path'] == str(data_file)


# Convenience Function Tests

class TestExecuteProtocolCode:
    """Tests for execute_protocol_code convenience function."""

    def test_execute_protocol_code_basic(self):
        """Test basic protocol code execution."""
        code = "import numpy as np\nresults = {'value': 42}"

        result = execute_protocol_code(code, use_sandbox=False)

        assert result['success'] is True
        assert result['return_value'] == {'value': 42}

    def test_execute_protocol_code_with_validation(self):
        """Test protocol code execution with validation (always on)."""
        safe_code = "import numpy as np\nresults = {}"

        result = execute_protocol_code(safe_code, use_sandbox=False)

        assert result['success'] is True

    def test_execute_protocol_code_rejects_unsafe(self):
        """Test protocol code execution rejects unsafe code."""
        unsafe_code = "import os\nos.system('bad_command')"

        result = execute_protocol_code(unsafe_code, use_sandbox=False)

        assert result['success'] is False
        assert 'validation_errors' in result

    def test_execute_protocol_code_with_data(self, tmp_path):
        """Test protocol code execution with data file."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("x,y\n1,2\n3,4")

        code = "import pandas as pd\ndf = pd.read_csv(data_path)\nresults = {'rows': len(df)}"

        result = execute_protocol_code(code, data_path=str(data_file), use_sandbox=False)

        assert result['success'] is True
        assert result['return_value']['rows'] == 2


# Sandbox Integration Tests (Mocked)

class TestSandboxIntegration:
    """Tests for sandbox integration (mocked)."""

    @patch('kosmos.execution.executor.SANDBOX_AVAILABLE', True)
    @patch('kosmos.execution.executor.DockerSandbox')
    def test_executor_uses_sandbox_when_enabled(self, mock_sandbox_class):
        """Test executor uses sandbox when use_sandbox=True."""
        mock_sandbox_instance = Mock()
        mock_sandbox_class.return_value = mock_sandbox_instance

        executor = CodeExecutor(use_sandbox=True)

        assert executor.sandbox is not None
        assert executor.use_sandbox is True

    @patch('kosmos.execution.executor.SANDBOX_AVAILABLE', True)
    @patch('kosmos.execution.executor.DockerSandbox')
    def test_sandboxed_code_gets_result_footer(self, mock_sandbox_class):
        """The footer prints results as one JSON RESULT: line, numpy types included (P0-2)."""
        import numpy as np  # noqa: F401  (used by the executed code)

        mock_sandbox = mock_sandbox_class.return_value
        mock_sandbox.execute.return_value = SandboxExecutionResult(success=True)
        executor = CodeExecutor(use_sandbox=True)

        executor.execute(
            "import numpy as np\n"
            "results={'p': np.float64(0.01), 'n': np.int64(3), 'ok': np.bool_(True), "
            "'arr': np.array([1,2]), 'x': float('nan')}"
        )

        code = mock_sandbox.execute.call_args.args[0]
        assert code.count("RESULT:") == 1
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(code, {})
        lines = [l for l in out.getvalue().splitlines() if l.startswith("RESULT:")]
        assert len(lines) == 1
        payload = json.loads(lines[0][len("RESULT:"):])
        assert {k: v for k, v in payload.items() if k != 'x'} == {'p': 0.01, 'n': 3, 'ok': True, 'arr': [1, 2]}
        assert math.isnan(payload['x'])

    @patch('kosmos.execution.executor.SANDBOX_AVAILABLE', True)
    @patch('kosmos.execution.executor.DockerSandbox')
    def test_result_footer_does_not_accumulate_on_retry(self, mock_sandbox_class):
        """Every retry sends exactly one footer (P0-2)."""
        mock_sandbox = mock_sandbox_class.return_value
        mock_sandbox.execute.return_value = SandboxExecutionResult(
            success=False, error="x", error_type="ExecutionError"
        )
        executor = CodeExecutor(use_sandbox=True, max_retries=2, retry_delay=0.01)

        executor.execute("results={}", retry_on_error=True)

        assert mock_sandbox.execute.call_count == 2
        for call in mock_sandbox.execute.call_args_list:
            assert call.args[0].count("RESULT:") == 1

    @patch('kosmos.execution.executor.SANDBOX_AVAILABLE', False)
    def test_executor_graceful_fallback_when_sandbox_unavailable(self):
        """Test executor gracefully falls back when sandbox unavailable (F-17)."""
        executor = CodeExecutor(use_sandbox=True)

        # Should fall back to non-sandbox with restricted builtins
        assert executor.sandbox is None
        assert executor.use_sandbox is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

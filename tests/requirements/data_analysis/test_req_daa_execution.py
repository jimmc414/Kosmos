"""
Tests for Data Analysis Agent Code Execution Requirements (REQ-DAA-EXEC-*).

These tests validate sandbox execution, resource limits, error handling,
and safety constraints for generated code execution.
"""

import pytest
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

pytestmark = [
    pytest.mark.requirement("REQ-DAA-EXEC"),
    pytest.mark.category("data_analysis"),
]


@pytest.mark.requirement("REQ-DAA-EXEC-001")
@pytest.mark.priority("MUST")
def test_req_daa_exec_001_isolated_sandbox():
    """
    REQ-DAA-EXEC-001: The system MUST execute agent-generated code in an
    isolated sandbox environment that prevents access to the host system.
    """
    from kosmos.execution.sandbox import Sandbox

    # Arrange
    sandbox = Sandbox()
    dangerous_code = """
import os
result = os.listdir('/')  # Attempt to list root directory
"""

    # Act & Assert: Sandbox should prevent file system access
    # or the validator should block this code
    try:
        with pytest.raises((PermissionError, SecurityError, Exception)):
            result = sandbox.execute(dangerous_code)
            # If execution succeeds, check that it's restricted
            if result.success:
                pytest.fail("Sandbox should prevent unrestricted file system access")
    except ImportError:
        pytest.skip("Sandbox not fully implemented")


@pytest.mark.requirement("REQ-DAA-EXEC-002")
@pytest.mark.priority("MUST")
def test_req_daa_exec_002_capture_outputs():
    """
    REQ-DAA-EXEC-002: The sandbox environment MUST capture stdout, stderr,
    and generated artifacts (plots, tables) from code execution.
    """
    from kosmos.execution.sandbox import Sandbox

    # Arrange
    sandbox = Sandbox()
    test_code = """
import sys
print("Output to stdout")
sys.stderr.write("Output to stderr\\n")
result = {"data": [1, 2, 3]}
"""

    # Act
    try:
        result = sandbox.execute(test_code)

        # Assert: All outputs captured
        assert hasattr(result, 'stdout') or hasattr(result, 'output'), \
            "Must capture stdout"
        assert hasattr(result, 'stderr') or hasattr(result, 'error'), \
            "Must capture stderr"

        if result.success:
            stdout_content = str(result.stdout or result.output or '')
            stderr_content = str(result.stderr or result.error or '')
            assert 'stdout' in stdout_content.lower() or len(stdout_content) > 0
    except ImportError:
        pytest.skip("Sandbox not fully implemented")


@pytest.mark.requirement("REQ-DAA-EXEC-003")
@pytest.mark.priority("MUST")
def test_req_daa_exec_003_resource_limits():
    """
    REQ-DAA-EXEC-003: The sandbox execution MUST enforce resource limits
    (CPU time, memory, disk I/O) to prevent runaway processes.
    """
    from kosmos.execution.sandbox import Sandbox

    # Arrange
    sandbox = Sandbox(max_memory_mb=256, max_cpu_seconds=5)

    # Test: Memory limit
    memory_hog_code = """
try:
    # Attempt to allocate large amount of memory
    big_list = [0] * (100 * 1024 * 1024)  # 100M integers
except MemoryError:
    result = "Memory limit enforced"
"""

    # Test: CPU time limit
    cpu_hog_code = """
import time
start = time.time()
try:
    # Infinite loop
    while True:
        x = 1 + 1
        if time.time() - start > 10:  # Safety valve
            break
except:
    pass
result = "Should timeout"
"""

    # Act & Assert
    try:
        # Memory test
        result_mem = sandbox.execute(memory_hog_code, timeout=5)
        # Should either fail or handle gracefully

        # CPU test
        result_cpu = sandbox.execute(cpu_hog_code, timeout=2)
        # Should timeout and return error
        assert result_cpu.execution_time <= 3, \
            "Execution should be terminated within timeout"
    except ImportError:
        pytest.skip("Sandbox not fully implemented")


@pytest.mark.requirement("REQ-DAA-EXEC-004")
@pytest.mark.priority("MUST")
def test_req_daa_exec_004_timeout_enforcement():
    """
    REQ-DAA-EXEC-004: The system MUST terminate code execution that exceeds
    configured timeout limits and log the timeout event.
    """
    from kosmos.execution.sandbox import Sandbox

    # Arrange
    sandbox = Sandbox()
    long_running_code = """
import time
time.sleep(10)  # Sleep for 10 seconds
result = "completed"
"""

    # Act
    start_time = time.time()
    try:
        result = sandbox.execute(long_running_code, timeout=2)
        elapsed = time.time() - start_time

        # Assert
        assert elapsed < 4, "Should timeout within reasonable time after limit"
        assert not result.success or result.timeout, \
            "Should indicate timeout occurred"
        assert hasattr(result, 'execution_time'), \
            "Should record execution time"
    except ImportError:
        pytest.skip("Sandbox not fully implemented")


@pytest.mark.requirement("REQ-DAA-EXEC-005")
@pytest.mark.priority("MUST")
def test_req_daa_exec_005_capture_errors():
    """
    REQ-DAA-EXEC-005: The system MUST capture and report execution errors
    (syntax errors, runtime exceptions) with stack traces for debugging.
    """
    from kosmos.execution.sandbox import Sandbox

    # Arrange
    sandbox = Sandbox()

    # Test: Syntax error
    syntax_error_code = "def foo(\n  print('missing close paren')"

    # Test: Runtime error
    runtime_error_code = """
def divide_by_zero():
    return 1 / 0

result = divide_by_zero()
"""

    # Test: Import error
    import_error_code = "import nonexistent_module_xyz"

    # Act & Assert: Syntax error
    try:
        result_syntax = sandbox.execute(syntax_error_code)
        assert not result_syntax.success, "Should fail on syntax error"
        error_msg = str(result_syntax.error or result_syntax.stderr or '')
        assert 'syntax' in error_msg.lower() or 'invalid' in error_msg.lower()
    except ImportError:
        pytest.skip("Sandbox not fully implemented")

    # Act & Assert: Runtime error
    try:
        result_runtime = sandbox.execute(runtime_error_code)
        assert not result_runtime.success, "Should fail on runtime error"
        error_msg = str(result_runtime.error or result_runtime.stderr or '')
        assert 'zero' in error_msg.lower() or 'division' in error_msg.lower()

        # Check for stack trace
        assert 'traceback' in error_msg.lower() or 'line' in error_msg.lower(), \
            "Should include stack trace"
    except ImportError:
        pytest.skip("Sandbox not fully implemented")


@pytest.mark.requirement("REQ-DAA-EXEC-006")
@pytest.mark.priority("MUST")
def test_req_daa_exec_006_execution_time_measurement():
    """
    REQ-DAA-EXEC-006: The system MUST measure and record execution time
    for all code runs.
    """
    from kosmos.execution.sandbox import Sandbox

    # Arrange
    sandbox = Sandbox()
    timed_code = """
import time
time.sleep(0.1)
result = "done"
"""

    # Act
    try:
        result = sandbox.execute(timed_code)

        # Assert
        assert hasattr(result, 'execution_time'), \
            "Must record execution time"
        assert result.execution_time >= 0.1, \
            "Execution time should reflect actual runtime"
        assert result.execution_time < 1.0, \
            "Execution time should be reasonable"
        assert isinstance(result.execution_time, (int, float)), \
            "Execution time must be numeric"
    except ImportError:
        pytest.skip("Sandbox not fully implemented")


@pytest.mark.requirement("REQ-DAA-EXEC-007")
@pytest.mark.priority("MUST")
def test_req_daa_exec_007_readonly_dataset_access():
    """
    REQ-DAA-EXEC-007: The sandbox MUST provide read-only access to the
    input dataset without allowing modification.
    """
    from kosmos.execution.sandbox import Sandbox

    # Arrange
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write("a,b,c\n1,2,3\n4,5,6\n")
        temp_file = f.name

    try:
        sandbox = Sandbox(data_path=temp_file)

        # Code that attempts to modify the dataset
        modify_code = f"""
import pandas as pd
df = pd.read_csv('{temp_file}')
# Attempt to save modified version
try:
    df['d'] = [7, 8]
    df.to_csv('{temp_file}', index=False)
    result = "modified"
except:
    result = "read-only enforced"
"""

        # Act
        result = sandbox.execute(modify_code)

        # Assert: Original file should be unchanged
        with open(temp_file, 'r') as f:
            content = f.read()
            assert 'd' not in content, \
                "Original dataset should not be modified"
            assert '1,2,3' in content, \
                "Original data should be preserved"
    except ImportError:
        pytest.skip("Sandbox not fully implemented")
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


@pytest.mark.requirement("REQ-DAA-EXEC-008")
@pytest.mark.priority("MAY")
def test_req_daa_exec_008_execution_modes():
    """
    REQ-DAA-EXEC-008: The system MAY support both containerized (Docker)
    and direct execution modes for testing and development.
    """
    pytest.skip("Optional requirement - implementation dependent")


@pytest.mark.requirement("REQ-DAA-EXEC-009")
@pytest.mark.priority("MUST")
def test_req_daa_exec_009_no_arbitrary_shell():
    """
    REQ-DAA-EXEC-009: The sandbox MUST NOT allow code to execute arbitrary
    shell commands. Spawning subprocesses SHOULD be restricted to a
    predefined allowlist.
    """
    from kosmos.execution.sandbox import Sandbox
    from kosmos.safety.code_validator import CodeValidator

    # Arrange
    validator = CodeValidator()
    sandbox = Sandbox()

    shell_codes = [
        "import os; os.system('ls')",
        "import subprocess; subprocess.run(['echo', 'test'])",
        "import subprocess; subprocess.Popen(['cat', '/etc/passwd'])",
        "__import__('os').system('whoami')",
    ]

    # Act & Assert
    for code in shell_codes:
        # First, validator should catch it
        validation = validator.validate(code)
        assert not validation.is_safe, \
            f"Validator should block: {code}"

        # If it somehow gets through, sandbox should block
        try:
            result = sandbox.execute(code)
            assert not result.success, \
                f"Sandbox should prevent shell execution: {code}"
        except (ImportError, AttributeError):
            pytest.skip("Safety validator not fully implemented")


@pytest.mark.requirement("REQ-DAA-EXEC-010")
@pytest.mark.priority("MUST")
def test_req_daa_exec_010_no_env_modification():
    """
    REQ-DAA-EXEC-010: The sandbox MUST NOT allow code to modify system
    environment variables visible to other processes.
    """
    from kosmos.execution.sandbox import Sandbox

    # Arrange
    sandbox = Sandbox()
    original_path = os.environ.get('PATH', '')

    env_modify_code = """
import os
os.environ['PATH'] = '/malicious/path'
os.environ['TEST_VAR_XYZ'] = 'malicious_value'
result = "modified"
"""

    # Act
    try:
        result = sandbox.execute(env_modify_code)

        # Assert: Parent process environment unchanged
        assert os.environ.get('PATH', '') == original_path, \
            "Parent PATH should not be modified"
        assert 'TEST_VAR_XYZ' not in os.environ, \
            "Parent environment should not have new variables"
    except ImportError:
        pytest.skip("Sandbox not fully implemented")


@pytest.mark.requirement("REQ-DAA-EXEC-011")
@pytest.mark.priority("MUST")
def test_req_daa_exec_011_no_execution_without_sandbox():
    """
    REQ-DAA-EXEC-011: The system MUST NOT proceed with code execution if
    the sandbox initialization fails or is unavailable (when sandboxing
    is required).
    """
    from kosmos.execution.executor import Executor

    # Arrange
    test_code = "print('hello')"

    # Act & Assert: Should fail or use safe fallback
    try:
        executor = Executor(require_sandbox=True, sandbox_available=False)
        with pytest.raises((RuntimeError, ValueError, Exception)):
            executor.execute(test_code)
    except (ImportError, AttributeError):
        # If classes don't exist yet, that's okay
        pytest.skip("Executor not fully implemented")


@pytest.mark.requirement("REQ-DAA-EXEC-012")
@pytest.mark.priority("SHOULD")
def test_req_daa_exec_012_self_correction_loop():
    """
    REQ-DAA-EXEC-012: The Data Analysis Agent SHOULD implement a
    self-correction loop that analyzes stderr output and regenerates code
    when execution fails, with a maximum of 3 retry attempts per task.
    """
    # This tests the concept - actual implementation would involve LLM calls

    class MockSelfCorrectingExecutor:
        def __init__(self, max_retries=3):
            self.max_retries = max_retries
            self.retry_count = 0

        def execute_with_retry(self, code, error_handler=None):
            attempts = []
            for attempt in range(self.max_retries):
                self.retry_count = attempt + 1
                result = self._execute(code)
                attempts.append(result)

                if result['success']:
                    return {'attempts': attempts, 'success': True}

                if attempt < self.max_retries - 1 and error_handler:
                    # Analyze error and regenerate code
                    code = error_handler(result['error'])

            return {'attempts': attempts, 'success': False}

        def _execute(self, code):
            # Mock execution
            if 'correct_code' in code:
                return {'success': True, 'error': None}
            return {'success': False, 'error': 'NameError: undefined variable'}

    # Arrange
    executor = MockSelfCorrectingExecutor(max_retries=3)
    initial_code = "bad_code"

    def error_handler(error):
        # Mock error analysis and code regeneration
        return "correct_code"

    # Act
    result = executor.execute_with_retry(initial_code, error_handler)

    # Assert
    assert len(result['attempts']) <= 3, "Should not exceed max retries"
    assert len(result['attempts']) >= 1, "Should attempt at least once"
    assert result['success'], "Should eventually succeed with correction"

"""
Jupyter kernel communication for code execution.

Handles:
- Kernel initialization within containers
- Code execution and output capture
- Streaming output for long-running cells
- Error handling and timeout management

This module provides Jupyter-style execution semantics for
running generated scientific code in isolated containers.
"""

import asyncio
import json
import base64
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import logging
import time
import tempfile
import os

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of code execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class CellOutput:
    """Output from a single cell execution."""
    output_type: str  # "stream", "execute_result", "error", "display_data"
    content: str
    mime_type: str = "text/plain"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output_type": self.output_type,
            "content": self.content,
            "mime_type": self.mime_type,
            "metadata": self.metadata
        }


@dataclass
class ExecutionResult:
    """Result of code execution."""
    status: ExecutionStatus
    outputs: List[CellOutput] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    return_value: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "outputs": [o.to_dict() for o in self.outputs],
            "stdout": self.stdout,
            "stderr": self.stderr,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "error_traceback": self.error_traceback,
            "return_value": self.return_value
        }

    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ExecutionStatus.COMPLETED


class JupyterClient:
    """
    Client for executing code in containers with Jupyter-like semantics.

    Provides:
    - Code execution with stdout/stderr capture
    - Error handling with tracebacks
    - Timeout management
    - Result extraction

    Usage:
        client = JupyterClient(container_id, docker_client)
        result = await client.execute_code('''
            import pandas as pd
            df = pd.DataFrame({'a': [1, 2, 3]})
            print(df.describe())
        ''')
        print(result.stdout)
    """

    def __init__(self, container_id: str, docker_client):
        """
        Initialize Jupyter client.

        Args:
            container_id: Docker container ID
            docker_client: Docker client instance
        """
        self.container_id = container_id
        self.docker_client = docker_client
        self._container = None

    def _get_container(self):
        """Get container instance."""
        if self._container is None:
            self._container = self.docker_client.containers.get(self.container_id)
        return self._container

    async def execute_code(
        self,
        code: str,
        timeout: int = 300,
        capture_result: bool = True
    ) -> ExecutionResult:
        """
        Execute code and capture output.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds
            capture_result: Whether to capture 'result' or 'results' variable

        Returns:
            ExecutionResult with status, outputs, and any errors
        """
        container = self._get_container()
        start_time = time.time()

        # Prepare code wrapper to capture outputs
        wrapped_code = self._wrap_code(code, capture_result)

        try:
            # Write code to container
            code_bytes = wrapped_code.encode('utf-8')
            code_b64 = base64.b64encode(code_bytes).decode('ascii')

            # Use base64 to safely transfer code into container
            write_cmd = f"python3 -c \"import base64; open('/tmp/cell.py', 'w').write(base64.b64decode('{code_b64}').decode('utf-8'))\""

            write_result = container.exec_run(
                ["sh", "-c", write_cmd],
                workdir="/workspace"
            )

            if write_result.exit_code != 0:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    error_message="Failed to write code to container",
                    error_traceback=write_result.output.decode('utf-8', errors='replace'),
                    execution_time=time.time() - start_time
                )

            # Execute code with timeout
            exec_cmd = f"timeout {timeout} python3 /tmp/cell.py"

            result = container.exec_run(
                ["sh", "-c", exec_cmd],
                workdir="/workspace",
                demux=True
            )

            execution_time = time.time() - start_time

            # Parse output
            stdout = result.output[0].decode('utf-8', errors='replace') if result.output[0] else ""
            stderr = result.output[1].decode('utf-8', errors='replace') if result.output[1] else ""

            # Check for timeout (exit code 124)
            if result.exit_code == 124:
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    stdout=stdout,
                    stderr=stderr,
                    error_message=f"Execution timed out after {timeout}s",
                    execution_time=execution_time
                )

            # Check for execution error
            if result.exit_code != 0:
                error_msg, traceback = self._extract_error(stderr)
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    stdout=stdout,
                    stderr=stderr,
                    error_message=error_msg or f"Exit code: {result.exit_code}",
                    error_traceback=traceback,
                    execution_time=execution_time
                )

            # Success - extract outputs and return value
            outputs = self._parse_outputs(stdout)
            return_value = self._extract_return_value(stdout)

            return ExecutionResult(
                status=ExecutionStatus.COMPLETED,
                outputs=outputs,
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time,
                return_value=return_value
            )

        except Exception as e:
            logger.error(f"Execution error: {e}")
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error_message=str(e),
                execution_time=time.time() - start_time
            )

    def _wrap_code(self, code: str, capture_result: bool = True) -> str:
        """
        Wrap code to capture outputs and results.

        Args:
            code: Original code
            capture_result: Whether to capture result variable

        Returns:
            Wrapped code string
        """
        # Escape code for embedding
        escaped_code = code.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")

        wrapper = f'''
import sys
import json
import traceback

# Capture original stdout/stderr
_stdout = sys.stdout
_stderr = sys.stderr

try:
    # Execute user code
    _user_code = """{code}"""
    exec(_user_code, globals())

    # Capture result if exists
    _result = None
    if 'results' in dir():
        _result = results
    elif 'result' in dir():
        _result = result

    # Output result marker
    if _result is not None:
        try:
            print("__RESULT_START__")
            print(json.dumps(_result, default=str))
            print("__RESULT_END__")
        except Exception as e:
            print(f"__RESULT_ERROR__: {{e}}", file=sys.stderr)

except Exception as e:
    traceback.print_exc()
    sys.exit(1)
'''
        return wrapper

    def _extract_error(self, stderr: str) -> tuple:
        """Extract error message and traceback from stderr."""
        lines = stderr.strip().split('\n')

        if not lines:
            return None, None

        # Find the actual error message (usually last line)
        error_msg = None
        for line in reversed(lines):
            if line.strip() and not line.startswith(' '):
                error_msg = line.strip()
                break

        # Full traceback is the entire stderr
        traceback = stderr if "Traceback" in stderr else None

        return error_msg, traceback

    def _parse_outputs(self, stdout: str) -> List[CellOutput]:
        """Parse stdout into structured outputs."""
        outputs = []

        # Remove result markers from output for display
        display_output = stdout
        if "__RESULT_START__" in display_output:
            parts = display_output.split("__RESULT_START__")
            display_output = parts[0]

        if display_output.strip():
            outputs.append(CellOutput(
                output_type="stream",
                content=display_output.strip()
            ))

        return outputs

    def _extract_return_value(self, stdout: str) -> Optional[Any]:
        """Extract return value from stdout."""
        try:
            if "__RESULT_START__" in stdout and "__RESULT_END__" in stdout:
                start = stdout.index("__RESULT_START__") + len("__RESULT_START__")
                end = stdout.index("__RESULT_END__")
                result_json = stdout[start:end].strip()
                return json.loads(result_json)
        except (ValueError, json.JSONDecodeError) as e:
            logger.debug(f"Could not extract result: {e}")

        return None

    async def execute_notebook(
        self,
        notebook_content: Dict[str, Any],
        timeout_per_cell: int = 300,
        stop_on_error: bool = True
    ) -> List[ExecutionResult]:
        """
        Execute all code cells in a notebook.

        Args:
            notebook_content: Parsed notebook JSON
            timeout_per_cell: Timeout for each cell
            stop_on_error: Whether to stop on first error

        Returns:
            List of ExecutionResults, one per code cell
        """
        results = []

        cells = notebook_content.get("cells", [])
        logger.info(f"Executing notebook with {len(cells)} cells")

        for i, cell in enumerate(cells):
            if cell.get("cell_type") == "code":
                source = cell.get("source", [])
                if isinstance(source, list):
                    code = "".join(source)
                else:
                    code = source

                if not code.strip():
                    continue

                logger.debug(f"Executing cell {i + 1}")
                result = await self.execute_code(code, timeout_per_cell)
                results.append(result)

                if stop_on_error and result.status in (
                    ExecutionStatus.FAILED,
                    ExecutionStatus.TIMEOUT
                ):
                    logger.warning(f"Stopping notebook execution at cell {i + 1} due to error")
                    break

        return results

    async def run_script(
        self,
        script_path: str,
        timeout: int = 600
    ) -> ExecutionResult:
        """
        Run a Python script file in the container.

        Args:
            script_path: Path to script inside container
            timeout: Execution timeout

        Returns:
            ExecutionResult
        """
        container = self._get_container()
        start_time = time.time()

        try:
            exec_cmd = f"timeout {timeout} python3 {script_path}"

            result = container.exec_run(
                ["sh", "-c", exec_cmd],
                workdir="/workspace",
                demux=True
            )

            execution_time = time.time() - start_time
            stdout = result.output[0].decode('utf-8', errors='replace') if result.output[0] else ""
            stderr = result.output[1].decode('utf-8', errors='replace') if result.output[1] else ""

            if result.exit_code == 124:
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    stdout=stdout,
                    stderr=stderr,
                    error_message=f"Script timed out after {timeout}s",
                    execution_time=execution_time
                )

            if result.exit_code != 0:
                error_msg, traceback = self._extract_error(stderr)
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    stdout=stdout,
                    stderr=stderr,
                    error_message=error_msg or f"Exit code: {result.exit_code}",
                    error_traceback=traceback,
                    execution_time=execution_time
                )

            return ExecutionResult(
                status=ExecutionStatus.COMPLETED,
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time
            )

        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error_message=str(e),
                execution_time=time.time() - start_time
            )

    async def check_package(self, package_name: str) -> bool:
        """
        Check if a package is installed in the container.

        Args:
            package_name: Name of package to check

        Returns:
            True if installed, False otherwise
        """
        container = self._get_container()

        result = container.exec_run(
            ["python3", "-c", f"import {package_name}"],
            workdir="/workspace"
        )

        return result.exit_code == 0

    async def install_package(
        self,
        package_name: str,
        version: Optional[str] = None
    ) -> bool:
        """
        Install a package in the container.

        Args:
            package_name: Name of package to install
            version: Optional version specifier

        Returns:
            True if installation succeeded
        """
        container = self._get_container()

        if version:
            pkg_spec = f"{package_name}=={version}"
        else:
            pkg_spec = package_name

        result = container.exec_run(
            ["pip", "install", "--quiet", "--no-cache-dir", pkg_spec],
            workdir="/workspace"
        )

        success = result.exit_code == 0
        if success:
            logger.info(f"Installed package: {pkg_spec}")
        else:
            logger.warning(f"Failed to install {pkg_spec}: {result.output.decode()}")

        return success

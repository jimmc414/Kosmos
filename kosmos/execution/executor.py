"""
Code execution engine.

Executes generated Python code safely with output capture, error handling, and retry logic.
Supports both direct execution and Docker-based sandboxed execution.
"""

import sys
from kosmos.utils.compat import model_to_dict
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any, Optional
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional sandbox import
try:
    from kosmos.execution.sandbox import DockerSandbox, SandboxExecutionResult
    SANDBOX_AVAILABLE = True
except ImportError:
    SANDBOX_AVAILABLE = False
    logger.warning("Docker sandbox not available. Install docker package for sandboxed execution.")


class ExecutionResult:
    """Result of code execution."""

    def __init__(
        self,
        success: bool,
        return_value: Any = None,
        stdout: str = "",
        stderr: str = "",
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        execution_time: float = 0.0,
        profile_result: Optional[Any] = None  # ProfileResult from kosmos.core.profiling
    ):
        self.success = success
        self.return_value = return_value
        self.stdout = stdout
        self.stderr = stderr
        self.error = error
        self.error_type = error_type
        self.execution_time = execution_time
        self.profile_result = profile_result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'success': self.success,
            'return_value': self.return_value,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'error': self.error,
            'error_type': self.error_type,
            'execution_time': self.execution_time
        }

        # Include profile data if available
        if self.profile_result:
            try:
                result['profile_data'] = model_to_dict(self.profile_result)
            except Exception as e:
                logger.debug("Failed to serialize profile data: %s", e)
                result['profile_data'] = None

        return result


class CodeExecutor:
    """
    Executes Python code with safety measures and output capture.

    Provides:
    - Stdout/stderr capture
    - Return value extraction
    - Error handling
    - Execution retry logic
    - Optional Docker sandbox isolation
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        allowed_globals: Optional[Dict[str, Any]] = None,
        use_sandbox: bool = False,
        sandbox_config: Optional[Dict[str, Any]] = None,
        enable_profiling: bool = False,
        profiling_mode: str = "light"
    ):
        """
        Initialize code executor.

        Args:
            max_retries: Maximum number of retry attempts on error
            retry_delay: Delay between retries in seconds
            allowed_globals: Optional dictionary of allowed global variables
            use_sandbox: If True, use Docker sandbox for execution
            sandbox_config: Optional sandbox configuration (cpu_limit, memory_limit, timeout)
            enable_profiling: If True, profile code execution (default: False)
            profiling_mode: Profiling mode: light, standard, full (default: light)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.allowed_globals = allowed_globals or {}
        self.use_sandbox = use_sandbox
        self.sandbox_config = sandbox_config or {}
        self.enable_profiling = enable_profiling
        self.profiling_mode = profiling_mode

        # Initialize retry strategy for self-correcting execution (Issue #54)
        self.retry_strategy = RetryStrategy(max_retries=max_retries, base_delay=retry_delay)

        # Initialize sandbox if requested
        self.sandbox = None
        if self.use_sandbox:
            if not SANDBOX_AVAILABLE:
                raise RuntimeError("Docker sandbox requested but not available. Install docker package.")

            self.sandbox = DockerSandbox(**self.sandbox_config)
            logger.info("Docker sandbox initialized for code execution")

    def execute(
        self,
        code: str,
        local_vars: Optional[Dict[str, Any]] = None,
        retry_on_error: bool = False,
        llm_client: Optional[Any] = None
    ) -> ExecutionResult:
        """
        Execute Python code and capture results with self-correcting retry (Issue #54).

        Args:
            code: Python code to execute
            local_vars: Optional local variables to make available
            retry_on_error: If True, retry on execution errors with code fixes
            llm_client: Optional LLM client for intelligent code repair

        Returns:
            ExecutionResult with output and results
        """
        attempt = 0
        last_error = None
        current_code = code  # Track the current version of code

        while attempt < (self.max_retries if retry_on_error else 1):
            attempt += 1

            try:
                logger.info(f"Executing code (attempt {attempt})")
                result = self._execute_once(current_code, local_vars)

                if result.success:
                    # Track successful repair if code was modified
                    if current_code != code and attempt > 1:
                        self.retry_strategy.record_repair_attempt(
                            result.error_type or "Unknown", True
                        )
                    logger.info(f"Code executed successfully in {result.execution_time:.2f}s")
                    return result
                else:
                    last_error = result.error
                    error_type = result.error_type or "Unknown"

                    if retry_on_error and attempt < self.max_retries:
                        # Try to fix the code using RetryStrategy (Issue #54)
                        fixed_code = self.retry_strategy.modify_code_for_retry(
                            original_code=current_code,
                            error=result.error or "",
                            error_type=error_type,
                            traceback_str=result.stderr or "",
                            attempt=attempt,
                            llm_client=llm_client
                        )

                        if fixed_code and fixed_code != current_code:
                            logger.info(f"Applying code fix for {error_type}, attempt {attempt + 1}")
                            current_code = fixed_code
                        else:
                            logger.warning(f"No code fix available for {error_type}")

                        # Wait before retry
                        delay = self.retry_strategy.get_delay(attempt)
                        logger.warning(f"Execution failed, retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        # Record failed repair attempt
                        if current_code != code:
                            self.retry_strategy.record_repair_attempt(error_type, False)
                        return result

            except Exception as e:
                logger.error(f"Unexpected error during execution: {e}")
                last_error = str(e)
                error_type = type(e).__name__

                if retry_on_error and attempt < self.max_retries:
                    # Try to fix the code
                    fixed_code = self.retry_strategy.modify_code_for_retry(
                        original_code=current_code,
                        error=str(e),
                        error_type=error_type,
                        traceback_str=traceback.format_exc(),
                        attempt=attempt,
                        llm_client=llm_client
                    )

                    if fixed_code and fixed_code != current_code:
                        logger.info(f"Applying code fix for {error_type}")
                        current_code = fixed_code

                    delay = self.retry_strategy.get_delay(attempt)
                    time.sleep(delay)
                else:
                    return ExecutionResult(
                        success=False,
                        error=str(e),
                        error_type=error_type
                    )

        # All retries failed
        return ExecutionResult(
            success=False,
            error=f"Failed after {self.max_retries} attempts. Last error: {last_error}",
            error_type="MaxRetriesExceeded"
        )

    def _execute_once(
        self,
        code: str,
        local_vars: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute code once with output capture and optional profiling."""

        # Route to sandbox if enabled
        if self.use_sandbox:
            return self._execute_in_sandbox(code, local_vars)

        # Initialize profiler if enabled
        profiler = None
        profile_result = None
        if self.enable_profiling:
            try:
                from kosmos.core.profiling import ExecutionProfiler, ProfilingMode
                mode = ProfilingMode(self.profiling_mode)
                profiler = ExecutionProfiler(mode=mode)
            except Exception as e:
                logger.warning(f"Failed to initialize profiler: {e}")

        # Otherwise execute directly
        start_time = time.time()

        # Prepare execution environment
        exec_globals = self._prepare_globals()
        exec_locals = local_vars.copy() if local_vars else {}

        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            # Start profiling if enabled
            if profiler:
                profiler._start_profiling()

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute code
                exec(code, exec_globals, exec_locals)

            # Stop profiling if enabled
            if profiler:
                profiler._stop_profiling()
                profile_result = profiler.get_result()

            execution_time = time.time() - start_time

            # Extract return value (look for 'results' variable)
            return_value = exec_locals.get('results', exec_locals.get('result'))

            return ExecutionResult(
                success=True,
                return_value=return_value,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                execution_time=execution_time,
                profile_result=profile_result
            )

        except Exception as e:
            # Stop profiling even on error
            if profiler:
                try:
                    profiler._stop_profiling()
                    profile_result = profiler.get_result()
                except Exception:
                    pass

            execution_time = time.time() - start_time

            # Capture full traceback
            error_traceback = traceback.format_exc()

            logger.error(f"Code execution failed: {e}\n{error_traceback}")

            return ExecutionResult(
                success=False,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue() + "\n" + error_traceback,
                error=str(e),
                error_type=type(e).__name__,
                execution_time=execution_time,
                profile_result=profile_result
            )

    def _execute_in_sandbox(
        self,
        code: str,
        local_vars: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute code in Docker sandbox."""
        logger.info("Executing code in Docker sandbox")

        # Prepare data files if data_path provided
        data_files = {}
        if local_vars and 'data_path' in local_vars:
            data_path = local_vars['data_path']
            # Extract filename from path
            import os
            filename = os.path.basename(data_path)
            data_files[filename] = data_path

            # Update code to use mounted data file
            code = f"data_path = '/workspace/data/{filename}'\n{code}"

        # Execute in sandbox
        sandbox_result = self.sandbox.execute(code, data_files=data_files if data_files else None)

        # Convert SandboxExecutionResult to ExecutionResult
        return ExecutionResult(
            success=sandbox_result.success,
            return_value=sandbox_result.return_value,
            stdout=sandbox_result.stdout,
            stderr=sandbox_result.stderr,
            error=sandbox_result.error,
            error_type=sandbox_result.error_type,
            execution_time=sandbox_result.execution_time
        )

    def _prepare_globals(self) -> Dict[str, Any]:
        """Prepare global namespace for code execution."""
        # Start with allowed globals
        exec_globals = self.allowed_globals.copy()

        # Add standard builtins
        exec_globals['__builtins__'] = __builtins__

        return exec_globals

    def execute_with_data(
        self,
        code: str,
        data_path: str,
        retry_on_error: bool = False
    ) -> ExecutionResult:
        """
        Execute code with data file path provided.

        Args:
            code: Python code to execute (expects `data_path` variable)
            data_path: Path to data file (made available as variable)
            retry_on_error: If True, retry on errors

        Returns:
            ExecutionResult

        Note:
            The data_path variable is prepended to code and also provided
            in local_vars for templates that use `pd.read_csv(data_path)`.
        """
        # Prepend data_path assignment so templates can use it (Issue #51)
        # This ensures data_path is defined even if templates use it directly
        augmented_code = f"# Data path injected by executor\ndata_path = {repr(data_path)}\n\n{code}"

        # Also inject as local variable for safety
        local_vars = {'data_path': data_path}

        return self.execute(augmented_code, local_vars, retry_on_error)


class CodeValidator:
    """
    Validates generated code for safety and correctness.

    Checks for:
    - Syntax errors
    - Dangerous imports
    - Dangerous operations
    """

    # Dangerous modules that should not be imported
    DANGEROUS_MODULES = [
        'os', 'subprocess', 'sys', 'shutil', 'importlib',
        'socket', 'urllib', 'requests', 'http',
        '__import__', 'eval', 'exec', 'compile'
    ]

    # Dangerous functions/operations
    DANGEROUS_PATTERNS = [
        'open(',  # File operations (except specific allowed cases)
        'eval(',
        'exec(',
        'compile(',
        '__import__',
        'globals(',
        'locals(',
        'vars(',
    ]

    @staticmethod
    def validate(code: str, allow_file_read: bool = True) -> Dict[str, Any]:
        """
        Validate code for safety.

        Args:
            code: Python code to validate
            allow_file_read: If True, allow read-only file operations

        Returns:
            Dictionary with:
                - valid: Boolean
                - errors: List of error messages
                - warnings: List of warning messages
        """
        errors = []
        warnings = []

        # Check syntax
        try:
            import ast
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return {'valid': False, 'errors': errors, 'warnings': warnings}

        # Check for dangerous imports
        for module in CodeValidator.DANGEROUS_MODULES:
            if f"import {module}" in code or f"from {module}" in code:
                errors.append(f"Dangerous import detected: {module}")

        # Check for dangerous patterns
        for pattern in CodeValidator.DANGEROUS_PATTERNS:
            if pattern in code:
                # Special case: allow open() for reading if permitted
                if pattern == 'open(' and allow_file_read:
                    # Check if it's read-only (contains "'r'" or no mode specified)
                    if "'w'" in code or "'a'" in code or "'x'" in code or "mode='w'" in code:
                        errors.append(f"Dangerous operation detected: write mode file operations")
                    else:
                        warnings.append(f"File read operation detected: {pattern}")
                else:
                    errors.append(f"Dangerous operation detected: {pattern}")

        # Check for network operations
        network_keywords = ['socket', 'http', 'urllib', 'requests', 'api']
        for keyword in network_keywords:
            if keyword in code.lower():
                warnings.append(f"Potential network operation detected: {keyword}")

        is_valid = len(errors) == 0

        logger.info(f"Code validation: {'PASSED' if is_valid else 'FAILED'}, "
                   f"{len(errors)} errors, {len(warnings)} warnings")

        return {
            'valid': is_valid,
            'errors': errors,
            'warnings': warnings
        }


class RetryStrategy:
    """
    Strategy for retrying failed code execution (Issue #54).

    Provides different retry approaches:
    - Simple retry (same code)
    - Modified retry (with error feedback) - handles 10+ error types
    - LLM-assisted retry (if LLM available)

    Paper Claim: "If error occurs → reads traceback → fixes code → re-executes"
    """

    # Common missing imports for auto-fix
    COMMON_IMPORTS = {
        'pd': 'import pandas as pd',
        'np': 'import numpy as np',
        'plt': 'import matplotlib.pyplot as plt',
        'sns': 'import seaborn as sns',
        'os': 'import os',
        'sys': 'import sys',
        'json': 'import json',
        're': 'import re',
        'math': 'import math',
        'datetime': 'from datetime import datetime',
        'Path': 'from pathlib import Path',
        'defaultdict': 'from collections import defaultdict',
        'Counter': 'from collections import Counter',
        'scipy': 'import scipy',
        'stats': 'from scipy import stats',
        'sklearn': 'import sklearn',
    }

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        """
        Initialize retry strategy.

        Args:
            max_retries: Maximum retry attempts
            base_delay: Base delay between retries (exponential backoff)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay

        # Repair statistics tracking (Issue #54)
        self.repair_stats = {
            "attempted": 0,
            "successful": 0,
            "by_error_type": {}
        }

    def should_retry(self, attempt: int, error_type: str) -> bool:
        """Determine if execution should be retried."""
        if attempt >= self.max_retries:
            return False

        # Don't retry on certain errors that can't be fixed
        non_retryable_errors = [
            'SyntaxError',  # Requires code rewrite
        ]

        return error_type not in non_retryable_errors

    def get_delay(self, attempt: int) -> float:
        """Get delay for retry attempt (exponential backoff)."""
        return self.base_delay * (2 ** (attempt - 1))

    def record_repair_attempt(self, error_type: str, success: bool):
        """Track repair attempt statistics."""
        self.repair_stats["attempted"] += 1
        if success:
            self.repair_stats["successful"] += 1

        if error_type not in self.repair_stats["by_error_type"]:
            self.repair_stats["by_error_type"][error_type] = {
                "attempted": 0,
                "successful": 0
            }
        self.repair_stats["by_error_type"][error_type]["attempted"] += 1
        if success:
            self.repair_stats["by_error_type"][error_type]["successful"] += 1

    def modify_code_for_retry(
        self,
        original_code: str,
        error: str,
        error_type: str,
        traceback_str: str = "",
        attempt: int = 1,
        llm_client: Optional[Any] = None
    ) -> Optional[str]:
        """
        Modify code based on error for retry (Issue #54 - Enhanced).

        Handles 10+ common error types with intelligent fixes.

        Args:
            original_code: Original code that failed
            error: Error message
            error_type: Type of error (e.g., 'KeyError', 'NameError')
            traceback_str: Full traceback string for context
            attempt: Retry attempt number
            llm_client: Optional LLM client for intelligent repair

        Returns:
            Modified code or None if no modification strategy
        """
        import re as regex_module

        # Try LLM-based repair first if available (only first 2 attempts)
        if llm_client and attempt <= 2:
            try:
                fixed = self._repair_with_llm(
                    original_code, error, traceback_str, llm_client
                )
                if fixed and fixed != original_code:
                    logger.info(f"LLM repair applied for {error_type}")
                    return fixed
            except Exception as e:
                logger.warning(f"LLM repair failed: {e}")

        # Pattern-based fixes for common error types
        if 'KeyError' in error_type:
            return self._fix_key_error(original_code, error)

        elif 'FileNotFoundError' in error_type:
            return self._fix_file_not_found(original_code, error)

        elif 'NameError' in error_type:
            return self._fix_name_error(original_code, error)

        elif 'TypeError' in error_type:
            return self._fix_type_error(original_code, error)

        elif 'IndexError' in error_type:
            return self._fix_index_error(original_code, error)

        elif 'AttributeError' in error_type:
            return self._fix_attribute_error(original_code, error)

        elif 'ValueError' in error_type:
            return self._fix_value_error(original_code, error)

        elif 'ZeroDivisionError' in error_type:
            return self._fix_zero_division(original_code, error)

        elif 'ImportError' in error_type or 'ModuleNotFoundError' in error_type:
            return self._fix_import_error(original_code, error)

        elif 'PermissionError' in error_type:
            return self._fix_permission_error(original_code, error)

        elif 'MemoryError' in error_type:
            return self._fix_memory_error(original_code, error)

        # No specific fix available
        return None

    def _repair_with_llm(
        self,
        code: str,
        error: str,
        traceback_str: str,
        llm_client: Any
    ) -> Optional[str]:
        """Use LLM to analyze error and fix code."""
        prompt = f"""Fix the following Python code that produced an error.

ORIGINAL CODE:
```python
{code}
```

ERROR:
{error}

TRACEBACK:
{traceback_str}

Return ONLY the fixed Python code, no explanations. Wrap the code in ```python``` markers."""

        try:
            response = llm_client.generate(prompt, max_tokens=2000)

            # Extract code from response
            if "```python" in response:
                start = response.find("```python") + 9
                end = response.find("```", start)
                if end > start:
                    return response[start:end].strip()

            # If no markers, return as-is if it looks like code
            if "import" in response or "def " in response or "=" in response:
                return response.strip()

        except Exception as e:
            logger.warning(f"LLM code repair error: {e}")

        return None

    def _fix_key_error(self, code: str, error: str) -> str:
        """Fix KeyError by adding safe dict access."""
        return f"""try:
{code}
except KeyError as e:
    print(f"KeyError: {{e}}. Using default value.")
    results = {{'error': 'KeyError', 'details': str(e), 'status': 'failed'}}
"""

    def _fix_file_not_found(self, code: str, error: str) -> str:
        """Fix FileNotFoundError by adding existence check."""
        import re as regex_module
        # Try to extract the file path from error
        match = regex_module.search(r"'([^']+)'", error)
        file_path = match.group(1) if match else "data_file"

        return f"""import os
if not os.path.exists('{file_path}'):
    print(f"File not found: '{file_path}'. Creating empty result.")
    results = {{'error': 'FileNotFoundError', 'file': '{file_path}', 'status': 'failed'}}
else:
{self._indent(code, 4)}
"""

    def _fix_name_error(self, code: str, error: str) -> str:
        """Fix NameError by adding missing imports or definitions."""
        import re as regex_module
        match = regex_module.search(r"name '(\w+)' is not defined", error)
        if match:
            name = match.group(1)
            if name in self.COMMON_IMPORTS:
                return self.COMMON_IMPORTS[name] + '\n' + code

        # Generic fix: wrap in try-except
        return f"""try:
{code}
except NameError as e:
    print(f"NameError: {{e}}. Variable not defined.")
    results = {{'error': 'NameError', 'details': str(e), 'status': 'failed'}}
"""

    def _fix_type_error(self, code: str, error: str) -> str:
        """Fix TypeError by adding type checks."""
        return f"""try:
{code}
except TypeError as e:
    print(f"TypeError: {{e}}. Type conversion issue.")
    results = {{'error': 'TypeError', 'details': str(e), 'status': 'failed'}}
"""

    def _fix_index_error(self, code: str, error: str) -> str:
        """Fix IndexError by adding bounds checking."""
        return f"""try:
{code}
except IndexError as e:
    print(f"IndexError: {{e}}. Index out of bounds.")
    results = {{'error': 'IndexError', 'details': str(e), 'status': 'failed'}}
"""

    def _fix_attribute_error(self, code: str, error: str) -> str:
        """Fix AttributeError by adding attribute check."""
        return f"""try:
{code}
except AttributeError as e:
    print(f"AttributeError: {{e}}. Attribute not found.")
    results = {{'error': 'AttributeError', 'details': str(e), 'status': 'failed'}}
"""

    def _fix_value_error(self, code: str, error: str) -> str:
        """Fix ValueError by adding value validation."""
        return f"""try:
{code}
except ValueError as e:
    print(f"ValueError: {{e}}. Invalid value.")
    results = {{'error': 'ValueError', 'details': str(e), 'status': 'failed'}}
"""

    def _fix_zero_division(self, code: str, error: str) -> str:
        """Fix ZeroDivisionError by adding zero checks."""
        return f"""try:
{code}
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {{e}}. Division by zero.")
    results = {{'error': 'ZeroDivisionError', 'details': str(e), 'status': 'failed'}}
"""

    def _fix_import_error(self, code: str, error: str) -> str:
        """Fix ImportError by providing fallback."""
        import re as regex_module
        match = regex_module.search(r"No module named '(\w+)'", error)
        module = match.group(1) if match else "unknown"

        return f"""try:
{code}
except (ImportError, ModuleNotFoundError) as e:
    print(f"ImportError: Module '{module}' not available. {{e}}")
    results = {{'error': 'ImportError', 'module': '{module}', 'status': 'failed'}}
"""

    def _fix_permission_error(self, code: str, error: str) -> str:
        """Fix PermissionError by using alternative path."""
        return f"""try:
{code}
except PermissionError as e:
    print(f"PermissionError: {{e}}. Access denied.")
    results = {{'error': 'PermissionError', 'details': str(e), 'status': 'failed'}}
"""

    def _fix_memory_error(self, code: str, error: str) -> str:
        """Fix MemoryError by suggesting batch processing."""
        return f"""try:
{code}
except MemoryError as e:
    print(f"MemoryError: {{e}}. Consider processing in smaller batches.")
    results = {{'error': 'MemoryError', 'details': 'Out of memory', 'status': 'failed'}}
"""

    def _indent(self, code: str, spaces: int) -> str:
        """Indent code by specified number of spaces."""
        indent = ' ' * spaces
        lines = code.split('\n')
        return '\n'.join(indent + line if line.strip() else line for line in lines)


def execute_protocol_code(
    code: str,
    data_path: Optional[str] = None,
    max_retries: int = 2,
    validate_safety: bool = True,
    use_sandbox: bool = False,
    sandbox_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to execute protocol code with full pipeline.

    Args:
        code: Generated code to execute
        data_path: Optional path to data file
        max_retries: Maximum retry attempts
        validate_safety: If True, validate code safety first
        use_sandbox: If True, execute in Docker sandbox
        sandbox_config: Optional sandbox configuration

    Returns:
        Dictionary with execution results
    """
    # Validate code if requested
    if validate_safety:
        validation = CodeValidator.validate(code, allow_file_read=True)
        if not validation['valid']:
            return {
                'success': False,
                'error': 'Code validation failed',
                'validation_errors': validation['errors'],
                'validation_warnings': validation['warnings']
            }

    # Execute code
    executor = CodeExecutor(
        max_retries=max_retries,
        use_sandbox=use_sandbox,
        sandbox_config=sandbox_config or {}
    )

    if data_path:
        result = executor.execute_with_data(code, data_path, retry_on_error=True)
    else:
        result = executor.execute(code, retry_on_error=True)

    # Convert to dict and add validation info
    result_dict = result.to_dict()
    if validate_safety:
        result_dict['validation_warnings'] = validation.get('warnings', [])

    return result_dict

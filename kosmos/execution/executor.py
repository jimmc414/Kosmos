"""
Code execution engine.

Executes generated Python code safely with output capture, error handling, and retry logic.
Supports both direct execution and Docker-based sandboxed execution.
"""

import sys
from kosmos.utils.compat import model_to_dict
import io
import traceback
import signal
import platform
import concurrent.futures
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

# Optional R executor import
try:
    from kosmos.execution.r_executor import RExecutor, RExecutionResult, is_r_code
    R_EXECUTOR_AVAILABLE = True
except ImportError:
    R_EXECUTOR_AVAILABLE = False
    logger.debug("R executor not available")

# Execution timeout (seconds) for unsandboxed exec (F-19)
DEFAULT_EXECUTION_TIMEOUT = 300

# Appended to sandboxed code so the container's results dict reaches the host:
# DockerSandbox._extract_return_value parses a single stdout line starting with RESULT:
SANDBOX_RESULT_FOOTER = '''
# --- kosmos sandbox result footer (appended by CodeExecutor._execute_in_sandbox) ---
import json as _kj
def _kdef(o):
    try:
        import numpy as _n
        if isinstance(o, _n.integer): return int(o)
        if isinstance(o, _n.floating): return float(o)
        if isinstance(o, _n.bool_): return bool(o)
        if isinstance(o, _n.ndarray): return o.tolist()
    except Exception:
        pass
    return str(o)
def _kfix(o):
    if isinstance(o, dict): return {str(k): _kfix(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_kfix(v) for v in o]
    return o
_kr = globals().get('results', globals().get('result'))
if _kr is not None:
    print("RESULT:" + _kj.dumps(_kfix(_kr), default=_kdef))
'''

# Restricted builtins for unsandboxed execution (F-16)
SAFE_BUILTINS = {
    # Core types
    'None': None, 'True': True, 'False': False,
    'int': int, 'float': float, 'str': str, 'bool': bool,
    'bytes': bytes, 'bytearray': bytearray, 'complex': complex,
    # Collections
    'list': list, 'dict': dict, 'tuple': tuple, 'set': set, 'frozenset': frozenset,
    # Iteration
    'range': range, 'enumerate': enumerate, 'zip': zip,
    'map': map, 'filter': filter, 'reversed': reversed, 'sorted': sorted,
    'iter': iter, 'next': next,
    # Math
    'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum,
    'pow': pow, 'divmod': divmod,
    # String/repr
    'repr': repr, 'format': format, 'chr': chr, 'ord': ord,
    'ascii': ascii, 'bin': bin, 'hex': hex, 'oct': oct, 'hash': hash,
    # Type introspection (safe subset)
    'isinstance': isinstance, 'issubclass': issubclass,
    'type': type, 'id': id, 'callable': callable,
    'hasattr': hasattr, 'len': len,
    # IO
    'print': print,
    # Object creation helpers
    'property': property, 'staticmethod': staticmethod, 'classmethod': classmethod,
    'super': super, 'object': object,
    # Exceptions (needed for try/except)
    'Exception': Exception, 'BaseException': BaseException,
    'ValueError': ValueError, 'TypeError': TypeError, 'KeyError': KeyError,
    'IndexError': IndexError, 'AttributeError': AttributeError,
    'RuntimeError': RuntimeError, 'StopIteration': StopIteration,
    'ZeroDivisionError': ZeroDivisionError, 'NameError': NameError,
    'ImportError': ImportError, 'FileNotFoundError': FileNotFoundError,
    'OSError': OSError, 'IOError': IOError, 'NotImplementedError': NotImplementedError,
    'OverflowError': OverflowError, 'ArithmeticError': ArithmeticError,
    'MemoryError': MemoryError, 'RecursionError': RecursionError,
    'UnicodeError': UnicodeError, 'UnicodeDecodeError': UnicodeDecodeError,
    'UnicodeEncodeError': UnicodeEncodeError,
    # Misc safe builtins
    'any': any, 'all': all, 'slice': slice, 'memoryview': memoryview,
}

# Modules allowed for restricted import (F-16)
_ALLOWED_MODULES = {
    'numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'seaborn',
    'statsmodels', 'math', 'statistics', 'collections', 'itertools',
    'functools', 'operator', 'decimal', 'fractions', 'random',
    'datetime', 'time', 'json', 'csv', 'io', 'pathlib', 'copy',
    're', 'string', 'textwrap', 'hashlib', 'warnings', 'logging',
    'typing', 'dataclasses', 'enum', 'abc',
    'xarray', 'netCDF4', 'h5py', 'zarr', 'dask',
}


def _make_restricted_import(allowed: set = _ALLOWED_MODULES):
    """Create a restricted __import__ that only allows scientific modules."""
    _real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        top_level = name.split('.')[0]
        if top_level not in allowed:
            raise ImportError(
                f"Import of '{name}' is not allowed in sandboxed execution. "
                f"Allowed top-level modules: {sorted(allowed)}"
            )
        return _real_import(name, globals, locals, fromlist, level)

    return _restricted_import


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
        profile_result: Optional[Any] = None,  # ProfileResult from kosmos.core.profiling
        data_source: Optional[str] = None  # 'file' or 'synthetic'
    ):
        self.success = success
        self.return_value = return_value
        self.stdout = stdout
        self.stderr = stderr
        self.error = error
        self.error_type = error_type
        self.execution_time = execution_time
        self.profile_result = profile_result
        self.data_source = data_source

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'success': self.success,
            'return_value': self.return_value,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'error': self.error,
            'error_type': self.error_type,
            'execution_time': self.execution_time,
            'data_source': self.data_source,
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
        use_sandbox: bool = True,
        sandbox_config: Optional[Dict[str, Any]] = None,
        enable_profiling: bool = False,
        profiling_mode: str = "light",
        test_determinism: bool = False,
        execution_timeout: int = DEFAULT_EXECUTION_TIMEOUT
    ):
        """
        Initialize code executor.

        Args:
            max_retries: Maximum number of retry attempts on error
            retry_delay: Delay between retries in seconds
            allowed_globals: Optional dictionary of allowed global variables
            use_sandbox: If True, use Docker sandbox for execution (default: True for F-17)
            sandbox_config: Optional sandbox configuration (cpu_limit, memory_limit, timeout)
            enable_profiling: If True, profile code execution (default: False)
            profiling_mode: Profiling mode: light, standard, full (default: light)
            test_determinism: If True, run determinism check after successful execution (default: False)
            execution_timeout: Timeout in seconds for unsandboxed execution (default: 300, F-19)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.allowed_globals = allowed_globals or {}
        self.use_sandbox = use_sandbox
        self.sandbox_config = sandbox_config or {}
        self.enable_profiling = enable_profiling
        self.profiling_mode = profiling_mode
        self.test_determinism = test_determinism
        self.execution_timeout = execution_timeout

        # Initialize retry strategy for self-correcting execution (Issue #54)
        self.retry_strategy = RetryStrategy(max_retries=max_retries, base_delay=retry_delay)

        # Initialize sandbox if requested (F-17: graceful fallback)
        self.sandbox = None
        if self.use_sandbox:
            if not SANDBOX_AVAILABLE:
                logger.warning(
                    "Docker sandbox requested but not available. "
                    "Falling back to restricted builtins execution."
                )
                self.use_sandbox = False
            else:
                self.sandbox = DockerSandbox(**self.sandbox_config)
                logger.info("Docker sandbox initialized for code execution")

        # Initialize R executor for R language support (Issue #69)
        self.r_executor = None
        if R_EXECUTOR_AVAILABLE:
            r_timeout = self.sandbox_config.get('timeout', 300) if self.sandbox_config else 300
            self.r_executor = RExecutor(
                timeout=r_timeout,
                use_docker=self.use_sandbox,
                docker_image="kosmos-sandbox-r:latest"
            )
            logger.info("R executor initialized for R language support")

    def execute(
        self,
        code: str,
        local_vars: Optional[Dict[str, Any]] = None,
        retry_on_error: bool = False,
        llm_client: Optional[Any] = None,
        language: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute code and capture results with self-correcting retry (Issue #54).

        Supports both Python and R code. Language is auto-detected if not specified.

        Args:
            code: Code to execute (Python or R)
            local_vars: Optional local variables to make available (Python only)
            retry_on_error: If True, retry on execution errors with code fixes
            llm_client: Optional LLM client for intelligent code repair
            language: Language to use ('python', 'r'). Auto-detected if None.

        Returns:
            ExecutionResult with output and results
        """
        # Auto-detect language if not specified (Issue #69)
        if language is None:
            if R_EXECUTOR_AVAILABLE and self.r_executor:
                detected_lang = self.r_executor.detect_language(code)
                language = detected_lang
            else:
                language = 'python'

        # Route to R executor for R code
        if language == 'r':
            return self._execute_r(code)

        # Python execution continues below
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

                    # Optional determinism check
                    if self.test_determinism:
                        try:
                            from kosmos.safety.reproducibility import ReproducibilityManager
                            mgr = ReproducibilityManager()
                            is_deterministic = mgr.test_determinism(
                                experiment_function=lambda: self._execute_once(current_code, local_vars),
                                seed=42, n_runs=2
                            )
                            if not is_deterministic:
                                logger.warning("Non-deterministic results detected")
                                if result.return_value and isinstance(result.return_value, dict):
                                    result.return_value.setdefault('warnings', []).append(
                                        "Non-deterministic results detected"
                                    )
                        except Exception as det_err:
                            logger.debug(f"Determinism check failed (non-fatal): {det_err}")

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

    def _execute_r(self, code: str) -> ExecutionResult:
        """
        Execute R code using the R executor (Issue #69).

        Args:
            code: R code to execute

        Returns:
            ExecutionResult converted from RExecutionResult
        """
        if not R_EXECUTOR_AVAILABLE or self.r_executor is None:
            return ExecutionResult(
                success=False,
                error="R execution not available. Install R and the r_executor module.",
                error_type="RNotAvailable"
            )

        logger.info("Executing R code")
        r_result = self.r_executor.execute(code, capture_results=True)

        # Convert RExecutionResult to ExecutionResult
        return ExecutionResult(
            success=r_result.success,
            return_value=r_result.parsed_results or r_result.return_value,
            stdout=r_result.stdout,
            stderr=r_result.stderr,
            error=r_result.error,
            error_type=r_result.error_type,
            execution_time=r_result.execution_time
        )

    def execute_r(
        self,
        code: str,
        capture_results: bool = True,
        output_dir: Optional[str] = None
    ) -> ExecutionResult:
        """
        Explicitly execute R code (Issue #69).

        Use this method when you know the code is R and want to skip
        language detection.

        Args:
            code: R code to execute
            capture_results: If True, capture results as JSON
            output_dir: Directory for output files

        Returns:
            ExecutionResult with R execution results
        """
        if not R_EXECUTOR_AVAILABLE or self.r_executor is None:
            return ExecutionResult(
                success=False,
                error="R execution not available. Install R and the r_executor module.",
                error_type="RNotAvailable"
            )

        logger.info("Explicitly executing R code")
        r_result = self.r_executor.execute(
            code,
            capture_results=capture_results,
            output_dir=output_dir
        )

        # Convert RExecutionResult to ExecutionResult
        return ExecutionResult(
            success=r_result.success,
            return_value=r_result.parsed_results or r_result.return_value,
            stdout=r_result.stdout,
            stderr=r_result.stderr,
            error=r_result.error,
            error_type=r_result.error_type,
            execution_time=r_result.execution_time
        )

    def is_r_available(self) -> bool:
        """Check if R execution is available."""
        if not R_EXECUTOR_AVAILABLE or self.r_executor is None:
            return False
        return self.r_executor.is_r_available()

    def get_r_version(self) -> Optional[str]:
        """Get R version if available."""
        if not R_EXECUTOR_AVAILABLE or self.r_executor is None:
            return None
        return self.r_executor.get_r_version()

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
                # Execute code with timeout (F-19)
                self._exec_with_timeout(code, exec_globals, exec_locals)

            # Stop profiling if enabled
            if profiler:
                profiler._stop_profiling()
                profile_result = profiler.get_result()

            execution_time = time.time() - start_time

            # Extract return value (look for 'results' variable)
            return_value = exec_locals.get('results', exec_locals.get('result'))
            data_source = exec_locals.get('_data_source')

            return ExecutionResult(
                success=True,
                return_value=return_value,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                execution_time=execution_time,
                profile_result=profile_result,
                data_source=data_source,
            )

        except Exception as e:
            # Stop profiling even on error
            if profiler:
                try:
                    profiler._stop_profiling()
                    profile_result = profiler.get_result()
                except Exception as e:
                    logger.warning(f"Profiler stop/result retrieval failed: {e}")

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

        # Each attempt receives a fresh copy of the code, so footers never accumulate
        code = code.rstrip("\n") + "\n" + SANDBOX_RESULT_FOOTER

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
        """Prepare global namespace with restricted builtins (F-16)."""
        exec_globals = self.allowed_globals.copy()

        # Use restricted builtins instead of full Python builtins
        restricted = dict(SAFE_BUILTINS)
        restricted['__import__'] = _make_restricted_import()
        exec_globals['__builtins__'] = restricted

        return exec_globals

    def _exec_with_timeout(
        self,
        code: str,
        exec_globals: Dict[str, Any],
        exec_locals: Dict[str, Any]
    ) -> None:
        """Execute code with timeout protection (F-19)."""
        if platform.system() != 'Windows' and hasattr(signal, 'SIGALRM'):
            # Unix: use signal.alarm for same-thread timeout
            old_handler = signal.getsignal(signal.SIGALRM)
            def _timeout_handler(signum, frame):
                raise TimeoutError(
                    f"Code execution timed out after {self.execution_timeout}s"
                )
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(self.execution_timeout)
            try:
                exec(code, exec_globals, exec_locals)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # Windows/other: use ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(exec, code, exec_globals, exec_locals)
                try:
                    future.result(timeout=self.execution_timeout)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(
                        f"Code execution timed out after {self.execution_timeout}s"
                    )

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
            On the host path the data_path assignment is prepended to the code.
            In the sandbox, _execute_in_sandbox mounts the file and assigns the
            container path instead.
        """
        local_vars = {'data_path': data_path}
        if self.use_sandbox:
            # _execute_in_sandbox assigns data_path to the container mount; a host path
            # prepended here would override it and point at a file the container cannot see
            return self.execute(code, local_vars, retry_on_error)

        # Prepend data_path assignment so templates can use it (Issue #51)
        augmented_code = f"# Data path injected by executor\ndata_path = {repr(data_path)}\n\n{code}"
        return self.execute(augmented_code, local_vars, retry_on_error)


# Re-export CodeValidator from canonical safety module (F-22: removed duplicate)
from kosmos.safety.code_validator import CodeValidator  # noqa: F811,F401


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
        # Issue #51 fix: FileNotFoundError is terminal - use synthetic data instead
        non_retryable_errors = [
            'SyntaxError',           # Requires code rewrite
            'FileNotFoundError',     # Missing data is terminal - use synthetic data
            'DataUnavailableError',  # Custom error for missing data
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
        indented = self._indent(code, 4)
        return f"""try:
{indented}
except KeyError as e:
    print(f"KeyError: {{e}}. Using default value.")
    results = {{'error': 'KeyError', 'details': str(e), 'status': 'failed'}}
"""

    def _fix_file_not_found(self, code: str, error: str) -> Optional[str]:
        """
        Handle FileNotFoundError - return None to indicate terminal failure.

        Issue #51 fix: Instead of wrapping in a try-except that produces a failed
        result (which causes infinite loops), we return None to signal that this
        error cannot be fixed by retry. The caller should use synthetic data or
        fail gracefully.

        Args:
            code: Original code
            error: Error message

        Returns:
            None to indicate no fix possible - caller should handle differently
        """
        import re as regex_module
        # Try to extract the file path from error
        match = regex_module.search(r"'([^']+)'", error)
        file_path = match.group(1) if match else "unknown"

        logger.error(
            f"FileNotFoundError is terminal - data file missing: {file_path}. "
            "Code templates should use synthetic data generation (Issue #51)."
        )

        # Return None to indicate no fix possible - this is a terminal error
        return None

    def _fix_name_error(self, code: str, error: str) -> str:
        """Fix NameError by adding missing imports or definitions."""
        import re as regex_module
        match = regex_module.search(r"name '(\w+)' is not defined", error)
        if match:
            name = match.group(1)
            if name in self.COMMON_IMPORTS:
                return self.COMMON_IMPORTS[name] + '\n' + code

        # Generic fix: wrap in try-except
        indented = self._indent(code, 4)
        return f"""try:
{indented}
except NameError as e:
    print(f"NameError: {{e}}. Variable not defined.")
    results = {{'error': 'NameError', 'details': str(e), 'status': 'failed'}}
"""

    def _fix_type_error(self, code: str, error: str) -> str:
        """Fix TypeError by adding type checks."""
        indented = self._indent(code, 4)
        return f"""try:
{indented}
except TypeError as e:
    print(f"TypeError: {{e}}. Type conversion issue.")
    results = {{'error': 'TypeError', 'details': str(e), 'status': 'failed'}}
"""

    def _fix_index_error(self, code: str, error: str) -> str:
        """Fix IndexError by adding bounds checking."""
        indented = self._indent(code, 4)
        return f"""try:
{indented}
except IndexError as e:
    print(f"IndexError: {{e}}. Index out of bounds.")
    results = {{'error': 'IndexError', 'details': str(e), 'status': 'failed'}}
"""

    def _fix_attribute_error(self, code: str, error: str) -> str:
        """Fix AttributeError by adding attribute check."""
        indented = self._indent(code, 4)
        return f"""try:
{indented}
except AttributeError as e:
    print(f"AttributeError: {{e}}. Attribute not found.")
    results = {{'error': 'AttributeError', 'details': str(e), 'status': 'failed'}}
"""

    def _fix_value_error(self, code: str, error: str) -> str:
        """Fix ValueError by adding value validation."""
        indented = self._indent(code, 4)
        return f"""try:
{indented}
except ValueError as e:
    print(f"ValueError: {{e}}. Invalid value.")
    results = {{'error': 'ValueError', 'details': str(e), 'status': 'failed'}}
"""

    def _fix_zero_division(self, code: str, error: str) -> str:
        """Fix ZeroDivisionError by adding zero checks."""
        indented = self._indent(code, 4)
        return f"""try:
{indented}
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {{e}}. Division by zero.")
    results = {{'error': 'ZeroDivisionError', 'details': str(e), 'status': 'failed'}}
"""

    def _fix_import_error(self, code: str, error: str) -> str:
        """Fix ImportError by providing fallback."""
        import re as regex_module
        match = regex_module.search(r"No module named '(\w+)'", error)
        module = match.group(1) if match else "unknown"

        indented = self._indent(code, 4)
        return f"""try:
{indented}
except (ImportError, ModuleNotFoundError) as e:
    print(f"ImportError: Module '{module}' not available. {{e}}")
    results = {{'error': 'ImportError', 'module': '{module}', 'status': 'failed'}}
"""

    def _fix_permission_error(self, code: str, error: str) -> str:
        """Fix PermissionError by using alternative path."""
        indented = self._indent(code, 4)
        return f"""try:
{indented}
except PermissionError as e:
    print(f"PermissionError: {{e}}. Access denied.")
    results = {{'error': 'PermissionError', 'details': str(e), 'status': 'failed'}}
"""

    def _fix_memory_error(self, code: str, error: str) -> str:
        """Fix MemoryError by suggesting batch processing."""
        indented = self._indent(code, 4)
        return f"""try:
{indented}
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
    use_sandbox: bool = True,
    sandbox_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to execute protocol code with full pipeline.

    Always validates code safety before execution (F-21: removed validate_safety bypass).

    Args:
        code: Generated code to execute
        data_path: Optional path to data file
        max_retries: Maximum retry attempts
        use_sandbox: If True, execute in Docker sandbox (default: True)
        sandbox_config: Optional sandbox configuration

    Returns:
        Dictionary with execution results
    """
    # Check SafetyGuardrails emergency stop before execution
    try:
        from kosmos.safety.guardrails import SafetyGuardrails
        guardrails = SafetyGuardrails()
        guardrails.sync_from_flag_file()
        if guardrails.is_emergency_stop_active():
            return {
                'success': False,
                'error': 'Emergency stop is active',
                'validation_errors': [f"Emergency stop: {guardrails.emergency_stop.reason}"],
                'validation_warnings': []
            }
        # Use guardrails-enforced resource limits for sandbox config
        if use_sandbox and sandbox_config is None:
            enforced_limits = guardrails.enforce_resource_limits()
            sandbox_config = {
                'memory_limit_mb': enforced_limits.max_memory_mb,
                'timeout_seconds': enforced_limits.max_execution_time_seconds,
            }
    except ImportError:
        logger.debug("SafetyGuardrails not available, proceeding without guardrails")

    # Always validate code safety (F-21)
    validator = CodeValidator(allow_file_read=True)
    report = validator.validate(code)
    if not report.passed:
        return {
            'success': False,
            'error': 'Code validation failed',
            'validation_errors': [v.message for v in report.violations],
            'validation_warnings': [str(w) for w in report.warnings]
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
    result_dict['validation_warnings'] = [str(w) for w in report.warnings]

    return result_dict

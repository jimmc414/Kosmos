"""
Kosmos Execution Module.

Provides sandboxed code execution capabilities for running generated
scientific code safely with resource limits and security isolation.

Components:
- DockerManager: Container lifecycle management with pooling
- JupyterClient: Code execution with output capture
- PackageResolver: Automatic dependency detection and installation
- DockerSandbox, CodeExecutor: the execution path the research director uses

ProductionExecutor was archived to archive/code/ (VIAB#P3-4): nothing ran it.

Usage:
    from kosmos.execution import DockerSandbox, CodeExecutor

    # Direct sandbox usage
    sandbox = DockerSandbox()
    result = sandbox.execute("print('Hello!')")

    # Code executor with optional sandbox
    executor = CodeExecutor(use_sandbox=True)
    result = executor.execute("x = 1 + 1")
"""

# Container, kernel and package helpers
from .docker_manager import (
    DockerManager,
    ContainerConfig,
    ContainerInstance,
    ContainerStatus,
)

from .jupyter_client import (
    JupyterClient,
    ExecutionResult,
    ExecutionStatus,
    CellOutput,
)

from .package_resolver import (
    PackageResolver,
    PackageRequirement,
    extract_imports_from_code,
    resolve_package_name,
    is_stdlib_module,
    IMPORT_TO_PIP,
    STDLIB_MODULES,
)

# Legacy components (existing)
from .sandbox import (
    DockerSandbox,
    SandboxExecutionResult,
    execute_in_sandbox,
)

from .executor import (
    CodeExecutor,
    ExecutionResult as LegacyExecutionResult,
    CodeValidator,
    RetryStrategy,
    execute_protocol_code,
)

# Issue #62: Code line provenance
from .provenance import (
    CodeProvenance,
    CellLineMapping,
    create_provenance_from_notebook,
    build_cell_line_mappings,
    get_cell_for_line,
)

# Re-export commonly used items at package level
__all__ = [
    # Docker management
    "DockerManager",
    "ContainerConfig",
    "ContainerInstance",
    "ContainerStatus",

    # Jupyter client
    "JupyterClient",
    "ExecutionResult",
    "ExecutionStatus",
    "CellOutput",

    # Package resolution
    "PackageResolver",
    "PackageRequirement",
    "extract_imports_from_code",
    "resolve_package_name",
    "is_stdlib_module",
    "IMPORT_TO_PIP",
    "STDLIB_MODULES",

    # Legacy (existing)
    "DockerSandbox",
    "SandboxExecutionResult",
    "execute_in_sandbox",
    "CodeExecutor",
    "CodeValidator",
    "RetryStrategy",
    "execute_protocol_code",

    # Issue #62: Code line provenance
    "CodeProvenance",
    "CellLineMapping",
    "create_provenance_from_notebook",
    "build_cell_line_mappings",
    "get_cell_for_line",
]

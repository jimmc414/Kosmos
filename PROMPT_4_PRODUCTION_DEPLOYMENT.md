# PROMPT 4: Production Deployment - Gap 4 Completion & Final Steps

## Context

This prompt follows the completion of PROMPTs 1-3:
- **PROMPT 1**: Repository setup (scientific skills, reference repos)
- **PROMPT 2**: Gap implementation (5/6 gaps complete, ~4,152 lines)
- **PROMPT 3**: Production overhaul (339 tests, documentation cleanup)

**Current State**:
- 339 tests passing
- 5/6 gaps fully implemented
- 10/10 paper requirements verified
- Mock execution environment functional

**Remaining Blocker**: Gap 4 - Sandboxed Code Execution

---

## Objective

Complete the final steps to production readiness:

1. **Phase A**: Implement Gap 4 (Docker-based sandboxed execution)
2. **Phase B**: Fix legacy test dependencies
3. **Phase C**: Deploy CI/CD pipeline
4. **Phase D**: Add monitoring and observability
5. **Phase E**: Final production checklist

---

## Phase A: Gap 4 - Sandboxed Code Execution

### A.1 Problem Statement

The Kosmos paper describes generating **42,000 lines of code** per run across **166 agent rollouts**. The current implementation uses mock executors that simulate code execution but don't actually run generated code.

**Required for Production**:
- Real Jupyter notebook execution
- Isolated Docker containers
- Package management
- Security sandboxing
- Resource limits

### A.2 Implementation Plan

#### Directory Structure
```
kosmos/execution/
├── __init__.py              # Exports
├── sandbox.py               # Main SandboxedExecutor class
├── docker_manager.py        # Container lifecycle management
├── jupyter_client.py        # Jupyter kernel communication
├── package_resolver.py      # Dependency detection and installation
├── resource_limiter.py      # CPU/memory/time limits
└── security.py              # Network isolation, filesystem restrictions
```

#### A.2.1 Create Docker Infrastructure

**File: `kosmos/execution/docker_manager.py`**

```python
"""
Docker container management for sandboxed code execution.

Manages container lifecycle:
- Pool of pre-warmed containers for performance
- Container creation with resource limits
- Cleanup after execution
- Health monitoring
"""

import asyncio
import docker
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class ContainerConfig:
    """Configuration for execution containers."""
    image: str = "kosmos/executor:latest"
    memory_limit: str = "4g"
    cpu_limit: float = 2.0
    timeout_seconds: int = 600
    network_mode: str = "none"  # Network isolation
    readonly_rootfs: bool = True
    working_dir: str = "/home/jovyan/work"

@dataclass
class ContainerInstance:
    """Represents a running container."""
    container_id: str
    status: str
    created_at: float
    config: ContainerConfig

class DockerManager:
    """Manages Docker containers for code execution."""

    def __init__(self, config: Optional[ContainerConfig] = None):
        self.config = config or ContainerConfig()
        self.client = docker.from_env()
        self._container_pool: Dict[str, ContainerInstance] = {}
        self._pool_size = 3

    async def initialize_pool(self):
        """Pre-warm container pool for faster execution."""
        logger.info(f"Initializing container pool (size={self._pool_size})")
        for _ in range(self._pool_size):
            await self._create_container()

    async def _create_container(self) -> ContainerInstance:
        """Create a new container with security constraints."""
        container = self.client.containers.run(
            self.config.image,
            detach=True,
            mem_limit=self.config.memory_limit,
            nano_cpus=int(self.config.cpu_limit * 1e9),
            network_mode=self.config.network_mode,
            read_only=self.config.readonly_rootfs,
            working_dir=self.config.working_dir,
            # Security options
            security_opt=["no-new-privileges:true"],
            cap_drop=["ALL"],
            # Temporary writable directories
            tmpfs={
                "/tmp": "size=512m,mode=1777",
                "/home/jovyan/.local": "size=1g,mode=755"
            },
            # Environment
            environment={
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1"
            },
            # Keep alive for reuse
            command="tail -f /dev/null"
        )

        instance = ContainerInstance(
            container_id=container.id,
            status="ready",
            created_at=asyncio.get_event_loop().time(),
            config=self.config
        )
        self._container_pool[container.id] = instance
        return instance

    async def get_container(self) -> ContainerInstance:
        """Get an available container from pool or create new."""
        # Find available container
        for cid, instance in self._container_pool.items():
            if instance.status == "ready":
                instance.status = "in_use"
                return instance

        # Create new if none available
        return await self._create_container()

    async def release_container(self, container_id: str):
        """Release container back to pool or destroy if unhealthy."""
        if container_id in self._container_pool:
            instance = self._container_pool[container_id]
            # Check health before reuse
            try:
                container = self.client.containers.get(container_id)
                if container.status == "running":
                    instance.status = "ready"
                    return
            except docker.errors.NotFound:
                pass

            # Remove unhealthy container
            del self._container_pool[container_id]

    async def cleanup(self):
        """Cleanup all containers."""
        for cid in list(self._container_pool.keys()):
            try:
                container = self.client.containers.get(cid)
                container.stop(timeout=5)
                container.remove(force=True)
            except Exception as e:
                logger.warning(f"Error cleaning container {cid}: {e}")
            del self._container_pool[cid]
```

#### A.2.2 Create Jupyter Client

**File: `kosmos/execution/jupyter_client.py`**

```python
"""
Jupyter kernel communication for code execution.

Handles:
- Kernel initialization within containers
- Code execution and output capture
- Streaming output for long-running cells
- Error handling and timeout management
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ExecutionStatus(Enum):
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

class JupyterClient:
    """Client for executing code in Jupyter kernels."""

    def __init__(self, container_id: str, docker_client):
        self.container_id = container_id
        self.docker_client = docker_client
        self._kernel_id: Optional[str] = None

    async def initialize_kernel(self) -> bool:
        """Start a Python kernel in the container."""
        container = self.docker_client.containers.get(self.container_id)

        # Start Jupyter kernel
        result = container.exec_run(
            "python -c 'import jupyter_client; km = jupyter_client.KernelManager(); km.start_kernel(); print(km.connection_file)'",
            demux=True
        )

        if result.exit_code == 0:
            self._kernel_id = result.output[0].decode().strip()
            logger.info(f"Kernel started: {self._kernel_id}")
            return True
        return False

    async def execute_code(
        self,
        code: str,
        timeout: int = 300
    ) -> ExecutionResult:
        """Execute code and capture output."""
        if not self._kernel_id:
            await self.initialize_kernel()

        container = self.docker_client.containers.get(self.container_id)

        # Write code to temp file
        container.exec_run(f"echo '{self._escape_code(code)}' > /tmp/cell.py")

        # Execute with timeout
        try:
            result = container.exec_run(
                f"timeout {timeout} python /tmp/cell.py",
                demux=True,
                workdir="/home/jovyan/work"
            )

            stdout = result.output[0].decode() if result.output[0] else ""
            stderr = result.output[1].decode() if result.output[1] else ""

            if result.exit_code == 124:  # timeout exit code
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    stdout=stdout,
                    stderr=stderr,
                    error_message=f"Execution timed out after {timeout}s"
                )
            elif result.exit_code != 0:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    stdout=stdout,
                    stderr=stderr,
                    error_message=f"Exit code: {result.exit_code}",
                    error_traceback=stderr
                )
            else:
                return ExecutionResult(
                    status=ExecutionStatus.COMPLETED,
                    stdout=stdout,
                    stderr=stderr
                )

        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error_message=str(e)
            )

    def _escape_code(self, code: str) -> str:
        """Escape code for shell execution."""
        return code.replace("'", "'\\''")

    async def execute_notebook(
        self,
        notebook_path: str,
        timeout_per_cell: int = 300
    ) -> List[ExecutionResult]:
        """Execute all cells in a notebook."""
        container = self.docker_client.containers.get(self.container_id)

        # Read notebook
        result = container.exec_run(f"cat {notebook_path}")
        notebook = json.loads(result.output.decode())

        results = []
        for cell in notebook.get("cells", []):
            if cell["cell_type"] == "code":
                code = "".join(cell["source"])
                result = await self.execute_code(code, timeout_per_cell)
                results.append(result)

                # Stop on error
                if result.status == ExecutionStatus.FAILED:
                    break

        return results
```

#### A.2.3 Create Package Resolver

**File: `kosmos/execution/package_resolver.py`**

```python
"""
Automatic package dependency resolution and installation.

Features:
- Parse imports from generated code
- Resolve package names (import name -> pip name)
- Install missing packages in sandbox
- Handle version conflicts
"""

import re
import ast
from dataclasses import dataclass
from typing import List, Set, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Mapping of import names to pip package names
IMPORT_TO_PIP = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "yaml": "pyyaml",
    "Bio": "biopython",
    "scanpy": "scanpy",
    "anndata": "anndata",
    "rdkit": "rdkit-pypi",
    "torch": "torch",
    "tensorflow": "tensorflow",
    "keras": "keras",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "statsmodels": "statsmodels",
    "scipy": "scipy",
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "plotly": "plotly",
}

# Pre-installed packages (don't try to install)
PREINSTALLED = {
    "os", "sys", "re", "json", "csv", "math", "random",
    "datetime", "time", "pathlib", "typing", "collections",
    "itertools", "functools", "operator", "copy", "io",
    "logging", "warnings", "traceback", "subprocess",
    "multiprocessing", "threading", "asyncio", "concurrent",
}

@dataclass
class PackageRequirement:
    """A required package with optional version."""
    name: str
    version: Optional[str] = None
    import_name: str = ""

class PackageResolver:
    """Resolves and installs package dependencies."""

    def __init__(self, docker_client, container_id: str):
        self.docker_client = docker_client
        self.container_id = container_id
        self._installed_cache: Set[str] = set()

    def extract_imports(self, code: str) -> Set[str]:
        """Extract import statements from code."""
        imports = set()

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
        except SyntaxError:
            # Fallback to regex for invalid Python
            import_pattern = r'^(?:from|import)\s+([\w\.]+)'
            for match in re.finditer(import_pattern, code, re.MULTILINE):
                imports.add(match.group(1).split('.')[0])

        return imports - PREINSTALLED

    def resolve_packages(self, imports: Set[str]) -> List[PackageRequirement]:
        """Resolve import names to pip packages."""
        packages = []
        for imp in imports:
            pip_name = IMPORT_TO_PIP.get(imp, imp)
            packages.append(PackageRequirement(
                name=pip_name,
                import_name=imp
            ))
        return packages

    async def install_packages(
        self,
        packages: List[PackageRequirement]
    ) -> Dict[str, bool]:
        """Install packages in container."""
        container = self.docker_client.containers.get(self.container_id)
        results = {}

        for pkg in packages:
            if pkg.name in self._installed_cache:
                results[pkg.name] = True
                continue

            # Try to install
            cmd = f"pip install --quiet --no-cache-dir {pkg.name}"
            if pkg.version:
                cmd = f"pip install --quiet --no-cache-dir {pkg.name}=={pkg.version}"

            result = container.exec_run(cmd)
            success = result.exit_code == 0
            results[pkg.name] = success

            if success:
                self._installed_cache.add(pkg.name)
                logger.info(f"Installed {pkg.name}")
            else:
                logger.warning(f"Failed to install {pkg.name}: {result.output.decode()}")

        return results

    async def ensure_dependencies(self, code: str) -> bool:
        """Extract, resolve, and install all dependencies for code."""
        imports = self.extract_imports(code)
        packages = self.resolve_packages(imports)
        results = await self.install_packages(packages)
        return all(results.values())
```

#### A.2.4 Create Main Sandbox Executor

**File: `kosmos/execution/sandbox.py`**

```python
"""
Main sandboxed executor combining all components.

This is the primary interface for executing code safely:
- Automatic container management
- Dependency resolution
- Resource limits
- Output capture
"""

import asyncio
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import logging

from .docker_manager import DockerManager, ContainerConfig
from .jupyter_client import JupyterClient, ExecutionResult, ExecutionStatus
from .package_resolver import PackageResolver

logger = logging.getLogger(__name__)

@dataclass
class SandboxConfig:
    """Configuration for sandboxed execution."""
    memory_limit: str = "4g"
    cpu_limit: float = 2.0
    timeout_seconds: int = 600
    auto_install_packages: bool = True
    network_enabled: bool = False

class SandboxedExecutor:
    """
    Production-ready sandboxed code executor.

    Usage:
        executor = SandboxedExecutor()
        await executor.initialize()

        result = await executor.execute_code('''
            import pandas as pd
            df = pd.DataFrame({'a': [1, 2, 3]})
            print(df.describe())
        ''')

        print(result.stdout)
        await executor.cleanup()
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self._docker_manager: Optional[DockerManager] = None
        self._initialized = False

    async def initialize(self):
        """Initialize the executor and warm up container pool."""
        if self._initialized:
            return

        container_config = ContainerConfig(
            memory_limit=self.config.memory_limit,
            cpu_limit=self.config.cpu_limit,
            timeout_seconds=self.config.timeout_seconds,
            network_mode="bridge" if self.config.network_enabled else "none"
        )

        self._docker_manager = DockerManager(container_config)
        await self._docker_manager.initialize_pool()
        self._initialized = True
        logger.info("SandboxedExecutor initialized")

    async def execute_code(
        self,
        code: str,
        timeout: Optional[int] = None
    ) -> ExecutionResult:
        """
        Execute code in sandboxed environment.

        Args:
            code: Python code to execute
            timeout: Optional timeout override

        Returns:
            ExecutionResult with status, outputs, and any errors
        """
        if not self._initialized:
            await self.initialize()

        timeout = timeout or self.config.timeout_seconds
        container = await self._docker_manager.get_container()

        try:
            # Create Jupyter client for this container
            jupyter = JupyterClient(
                container.container_id,
                self._docker_manager.client
            )

            # Auto-install dependencies if enabled
            if self.config.auto_install_packages:
                resolver = PackageResolver(
                    self._docker_manager.client,
                    container.container_id
                )
                await resolver.ensure_dependencies(code)

            # Execute code
            result = await jupyter.execute_code(code, timeout)
            return result

        finally:
            # Release container back to pool
            await self._docker_manager.release_container(container.container_id)

    async def execute_notebook(
        self,
        notebook_content: Dict[str, Any],
        timeout_per_cell: int = 300
    ) -> List[ExecutionResult]:
        """
        Execute a Jupyter notebook.

        Args:
            notebook_content: Parsed notebook JSON
            timeout_per_cell: Timeout for each cell

        Returns:
            List of ExecutionResults, one per code cell
        """
        if not self._initialized:
            await self.initialize()

        container = await self._docker_manager.get_container()

        try:
            jupyter = JupyterClient(
                container.container_id,
                self._docker_manager.client
            )

            results = []
            for cell in notebook_content.get("cells", []):
                if cell["cell_type"] == "code":
                    code = "".join(cell["source"])

                    # Install dependencies for this cell
                    if self.config.auto_install_packages:
                        resolver = PackageResolver(
                            self._docker_manager.client,
                            container.container_id
                        )
                        await resolver.ensure_dependencies(code)

                    result = await jupyter.execute_code(code, timeout_per_cell)
                    results.append(result)

                    # Stop on failure
                    if result.status in (ExecutionStatus.FAILED, ExecutionStatus.TIMEOUT):
                        break

            return results

        finally:
            await self._docker_manager.release_container(container.container_id)

    async def cleanup(self):
        """Cleanup all resources."""
        if self._docker_manager:
            await self._docker_manager.cleanup()
        self._initialized = False
        logger.info("SandboxedExecutor cleaned up")
```

#### A.2.5 Create Dockerfile

**File: `docker/Dockerfile.executor`**

```dockerfile
# Kosmos Code Executor
# Pre-built scientific Python environment

FROM jupyter/scipy-notebook:python-3.11

LABEL maintainer="Kosmos Team"
LABEL description="Sandboxed execution environment for Kosmos AI Scientist"

USER root

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

# Core scientific packages
RUN pip install --no-cache-dir \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    scikit-learn>=1.2.0 \
    statsmodels>=0.14.0

# Bioinformatics packages
RUN pip install --no-cache-dir \
    biopython>=1.81 \
    scanpy>=1.9.0 \
    anndata>=0.9.0

# Chemistry packages
RUN pip install --no-cache-dir \
    rdkit-pypi>=2023.3.0 || true

# Data handling
RUN pip install --no-cache-dir \
    openpyxl>=3.1.0 \
    xlrd>=2.0.0 \
    pyarrow>=12.0.0 \
    h5py>=3.8.0

# Jupyter integration
RUN pip install --no-cache-dir \
    jupyter_client>=8.0.0 \
    nbformat>=5.8.0

# Set working directory
WORKDIR /home/jovyan/work

# Security: Run as non-root
USER ${NB_UID}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import pandas; import numpy; print('OK')"
```

**File: `docker/docker-compose.yml`**

```yaml
version: '3.8'

services:
  kosmos-executor:
    build:
      context: .
      dockerfile: Dockerfile.executor
    image: kosmos/executor:latest
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '0.5'
          memory: 1G
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    read_only: true
    tmpfs:
      - /tmp:size=512m,mode=1777
      - /home/jovyan/.local:size=1g,mode=755
    networks:
      - kosmos-isolated

networks:
  kosmos-isolated:
    driver: bridge
    internal: true  # No external access
```

#### A.2.6 Create Tests for Execution Module

**File: `tests/unit/execution/__init__.py`**
```python
"""Tests for execution module."""
```

**File: `tests/unit/execution/test_sandbox.py`**

```python
"""
Tests for sandboxed code execution.

These tests verify:
- Container management
- Code execution
- Package installation
- Security constraints
- Resource limits
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Test the package resolver without Docker
class TestPackageResolver:
    """Test package dependency resolution."""

    def test_extract_simple_imports(self):
        from kosmos.execution.package_resolver import PackageResolver

        resolver = PackageResolver(Mock(), "test-container")
        code = """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
"""
        imports = resolver.extract_imports(code)
        assert "pandas" in imports
        assert "numpy" in imports
        assert "sklearn" in imports

    def test_extract_from_imports(self):
        from kosmos.execution.package_resolver import PackageResolver

        resolver = PackageResolver(Mock(), "test-container")
        code = "from scipy.stats import ttest_ind"
        imports = resolver.extract_imports(code)
        assert "scipy" in imports

    def test_excludes_stdlib(self):
        from kosmos.execution.package_resolver import PackageResolver

        resolver = PackageResolver(Mock(), "test-container")
        code = """
import os
import sys
import json
import pandas
"""
        imports = resolver.extract_imports(code)
        assert "os" not in imports
        assert "sys" not in imports
        assert "json" not in imports
        assert "pandas" in imports

    def test_resolve_package_names(self):
        from kosmos.execution.package_resolver import PackageResolver, PackageRequirement

        resolver = PackageResolver(Mock(), "test-container")
        imports = {"sklearn", "PIL", "cv2"}
        packages = resolver.resolve_packages(imports)

        names = {p.name for p in packages}
        assert "scikit-learn" in names
        assert "Pillow" in names
        assert "opencv-python" in names


class TestExecutionResult:
    """Test execution result data structures."""

    def test_successful_result(self):
        from kosmos.execution.jupyter_client import ExecutionResult, ExecutionStatus

        result = ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            stdout="Hello, World!",
            stderr="",
            execution_time=0.5
        )

        assert result.status == ExecutionStatus.COMPLETED
        assert result.stdout == "Hello, World!"
        assert result.error_message is None

    def test_failed_result(self):
        from kosmos.execution.jupyter_client import ExecutionResult, ExecutionStatus

        result = ExecutionResult(
            status=ExecutionStatus.FAILED,
            error_message="NameError: name 'x' is not defined",
            error_traceback="Traceback..."
        )

        assert result.status == ExecutionStatus.FAILED
        assert "NameError" in result.error_message

    def test_timeout_result(self):
        from kosmos.execution.jupyter_client import ExecutionResult, ExecutionStatus

        result = ExecutionResult(
            status=ExecutionStatus.TIMEOUT,
            error_message="Execution timed out after 300s"
        )

        assert result.status == ExecutionStatus.TIMEOUT


class TestContainerConfig:
    """Test container configuration."""

    def test_default_config(self):
        from kosmos.execution.docker_manager import ContainerConfig

        config = ContainerConfig()
        assert config.memory_limit == "4g"
        assert config.cpu_limit == 2.0
        assert config.network_mode == "none"
        assert config.readonly_rootfs == True

    def test_custom_config(self):
        from kosmos.execution.docker_manager import ContainerConfig

        config = ContainerConfig(
            memory_limit="8g",
            cpu_limit=4.0,
            timeout_seconds=1200
        )
        assert config.memory_limit == "8g"
        assert config.cpu_limit == 4.0
        assert config.timeout_seconds == 1200


class TestSandboxConfig:
    """Test sandbox configuration."""

    def test_default_sandbox_config(self):
        from kosmos.execution.sandbox import SandboxConfig

        config = SandboxConfig()
        assert config.auto_install_packages == True
        assert config.network_enabled == False

    def test_network_enabled(self):
        from kosmos.execution.sandbox import SandboxConfig

        config = SandboxConfig(network_enabled=True)
        assert config.network_enabled == True


# Integration tests that require Docker
@pytest.mark.skipif(
    not pytest.importorskip("docker", reason="Docker not available"),
    reason="Docker not available"
)
class TestSandboxedExecutorIntegration:
    """Integration tests requiring Docker."""

    @pytest.mark.asyncio
    async def test_simple_execution(self):
        from kosmos.execution.sandbox import SandboxedExecutor

        executor = SandboxedExecutor()
        await executor.initialize()

        try:
            result = await executor.execute_code("print('Hello from sandbox!')")
            assert "Hello from sandbox!" in result.stdout
        finally:
            await executor.cleanup()

    @pytest.mark.asyncio
    async def test_pandas_execution(self):
        from kosmos.execution.sandbox import SandboxedExecutor

        executor = SandboxedExecutor()
        await executor.initialize()

        try:
            code = """
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(df.sum())
"""
            result = await executor.execute_code(code)
            assert result.status.value == "completed"
        finally:
            await executor.cleanup()
```

### A.3 Integration with Research Workflow

Update `kosmos/workflow/research_loop.py` to use real executor:

```python
# Add to ResearchWorkflow.__init__
from kosmos.execution.sandbox import SandboxedExecutor, SandboxConfig

def __init__(self, ...):
    ...
    # Initialize sandbox executor for production
    if use_sandbox:
        sandbox_config = SandboxConfig(
            memory_limit="4g",
            timeout_seconds=600,
            auto_install_packages=True
        )
        self.executor = SandboxedExecutor(sandbox_config)
    else:
        self.executor = MockExecutor()  # For testing
```

---

## Phase B: Fix Legacy Test Dependencies

### B.1 Install Missing Packages

Create a test requirements file:

**File: `requirements-test.txt`**

```
# Test framework
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
pytest-timeout>=2.1.0

# Missing dependencies for legacy tests
arxiv>=1.4.0
scipy>=1.10.0
matplotlib>=3.7.0

# Optional heavy dependencies
# scanpy>=1.9.0  # Large, install separately if needed
# torch>=2.0.0   # Very large, install separately if needed
```

### B.2 Update pyproject.toml

Add optional dependency groups:

```toml
[project.optional-dependencies]
test = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "arxiv>=1.4.0",
    "scipy>=1.10.0",
    "matplotlib>=3.7.0",
]

science = [
    "scanpy>=1.9.0",
    "anndata>=0.9.0",
    "biopython>=1.81",
]

all = [
    "kosmos[test,science]",
]
```

### B.3 Add Conditional Imports

Update modules with missing imports to fail gracefully:

**Example: `kosmos/literature/arxiv_client.py`**

```python
try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False
    arxiv = None

class ArxivClient:
    def __init__(self):
        if not ARXIV_AVAILABLE:
            raise ImportError(
                "arxiv package required. Install with: pip install arxiv"
            )
```

---

## Phase C: CI/CD Pipeline

### C.1 GitHub Actions Workflow

**File: `.github/workflows/ci.yml`**

```yaml
name: Kosmos CI

on:
  push:
    branches: [master, main, develop]
  pull_request:
    branches: [master, main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install linters
        run: pip install black isort flake8
      - name: Check formatting
        run: |
          black --check kosmos/ tests/
          isort --check kosmos/ tests/

  test-core:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-asyncio pytest-cov

      - name: Run smoke tests
        run: python scripts/smoke_test.py

      - name: Run core tests
        run: |
          pytest tests/unit/compression/ \
                 tests/unit/orchestration/ \
                 tests/unit/validation/ \
                 tests/unit/workflow/ \
                 tests/unit/agents/test_skill_loader.py \
                 tests/unit/world_model/test_artifacts.py \
                 --cov=kosmos --cov-report=xml -v

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: coverage.xml

  test-integration:
    runs-on: ubuntu-latest
    needs: test-core
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-asyncio

      - name: Run integration tests
        run: pytest tests/integration/ tests/e2e/ -v

  test-legacy:
    runs-on: ubuntu-latest
    continue-on-error: true  # Don't fail CI on legacy test issues
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install all dependencies
        run: |
          pip install -e ".[test]"
          pip install arxiv scipy matplotlib

      - name: Run legacy tests
        run: |
          pytest tests/unit/literature/ \
                 tests/unit/safety/ \
                 --ignore=tests/unit/literature/test_unified_search.py \
                 -v || true

  build-docker:
    runs-on: ubuntu-latest
    needs: [test-core, test-integration]
    if: github.ref == 'refs/heads/master'
    steps:
      - uses: actions/checkout@v4

      - name: Build executor image
        run: |
          docker build -t kosmos/executor:latest -f docker/Dockerfile.executor docker/

      - name: Test executor image
        run: |
          docker run --rm kosmos/executor:latest python -c "import pandas; import numpy; print('OK')"
```

### C.2 Pre-commit Configuration

**File: `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.3.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: ["--max-line-length=100", "--ignore=E203,W503"]

  - repo: local
    hooks:
      - id: smoke-test
        name: Run Smoke Tests
        entry: python scripts/smoke_test.py
        language: python
        pass_filenames: false
        stages: [push]
```

---

## Phase D: Monitoring and Observability

### D.1 Prometheus Metrics

**File: `kosmos/monitoring/metrics.py`**

```python
"""
Prometheus metrics for Kosmos monitoring.
"""

from prometheus_client import Counter, Histogram, Gauge, Info

# System info
kosmos_info = Info('kosmos', 'Kosmos system information')

# Workflow metrics
workflows_total = Counter(
    'kosmos_workflows_total',
    'Total workflows started',
    ['status']
)

cycles_total = Counter(
    'kosmos_cycles_total',
    'Total research cycles completed',
    ['workflow_id']
)

tasks_total = Counter(
    'kosmos_tasks_total',
    'Total tasks executed',
    ['task_type', 'status']
)

# Performance metrics
cycle_duration_seconds = Histogram(
    'kosmos_cycle_duration_seconds',
    'Duration of research cycles',
    buckets=[60, 300, 600, 1800, 3600, 7200]
)

task_duration_seconds = Histogram(
    'kosmos_task_duration_seconds',
    'Duration of individual tasks',
    ['task_type'],
    buckets=[10, 30, 60, 120, 300, 600]
)

# Quality metrics
findings_total = Counter(
    'kosmos_findings_total',
    'Total findings generated',
    ['validation_status']
)

validation_score = Histogram(
    'kosmos_validation_score',
    'ScholarEval validation scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Resource metrics
active_workflows = Gauge(
    'kosmos_active_workflows',
    'Currently running workflows'
)

active_containers = Gauge(
    'kosmos_active_containers',
    'Currently running execution containers'
)

# Compression metrics
compression_ratio = Histogram(
    'kosmos_compression_ratio',
    'Context compression ratios achieved',
    buckets=[5, 10, 15, 20, 25, 30, 50, 100]
)
```

### D.2 Logging Configuration

**File: `kosmos/core/logging_config.py`**

```python
"""
Structured logging configuration for Kosmos.
"""

import logging
import json
from datetime import datetime
from typing import Any, Dict

class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        if hasattr(record, "workflow_id"):
            log_data["workflow_id"] = record.workflow_id
        if hasattr(record, "cycle"):
            log_data["cycle"] = record.cycle
        if hasattr(record, "task_id"):
            log_data["task_id"] = record.task_id

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)

def configure_logging(level: str = "INFO", json_format: bool = True):
    """Configure logging for Kosmos."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    handler = logging.StreamHandler()
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

    root_logger.addHandler(handler)
```

---

## Phase E: Production Checklist

### E.1 Pre-Production Verification

```bash
#!/bin/bash
# scripts/verify_production.sh

echo "=== Kosmos Production Verification ==="

# 1. Run smoke tests
echo "Running smoke tests..."
python scripts/smoke_test.py || exit 1

# 2. Run core tests
echo "Running core tests..."
pytest tests/unit/compression/ tests/unit/orchestration/ \
       tests/unit/validation/ tests/unit/workflow/ \
       tests/unit/agents/test_skill_loader.py \
       tests/unit/world_model/test_artifacts.py -v || exit 1

# 3. Run integration tests
echo "Running integration tests..."
pytest tests/integration/ tests/e2e/ -v || exit 1

# 4. Verify E2E workflow
echo "Running E2E verification..."
python scripts/verify_e2e.py --cycles 3 --tasks 5 || exit 1

# 5. Check Docker build (if available)
if command -v docker &> /dev/null; then
    echo "Building Docker image..."
    docker build -t kosmos/executor:test -f docker/Dockerfile.executor docker/ || exit 1

    echo "Testing Docker image..."
    docker run --rm kosmos/executor:test python -c "import pandas; print('OK')" || exit 1
fi

# 6. Check dependencies
echo "Checking dependencies..."
pip check || exit 1

echo "=== All Verifications Passed ==="
```

### E.2 Final Checklist

**Code Quality**:
- [ ] All 339+ tests passing
- [ ] E2E verification completes
- [ ] No critical linting errors
- [ ] Type hints on public APIs

**Gap Implementation**:
- [ ] Gap 0: Context compression working
- [ ] Gap 1: State manager functional
- [ ] Gap 2: Orchestration complete
- [ ] Gap 3: Skill loader active
- [ ] Gap 4: Docker execution working
- [ ] Gap 5: ScholarEval validation active

**Infrastructure**:
- [ ] Docker images built and tested
- [ ] CI/CD pipeline active
- [ ] Secrets configured (API keys)
- [ ] Resource limits defined

**Monitoring**:
- [ ] Prometheus metrics exposed
- [ ] Logging configured
- [ ] Health checks implemented

**Documentation**:
- [ ] README current
- [ ] API docs generated
- [ ] Deployment guide written
- [ ] Troubleshooting guide available

---

## Execution Instructions

### Step 1: Implement Gap 4 Execution Module
```bash
# Create directory structure
mkdir -p kosmos/execution
mkdir -p docker
mkdir -p tests/unit/execution

# Implement files in order:
# 1. kosmos/execution/__init__.py
# 2. kosmos/execution/docker_manager.py
# 3. kosmos/execution/jupyter_client.py
# 4. kosmos/execution/package_resolver.py
# 5. kosmos/execution/sandbox.py
# 6. docker/Dockerfile.executor
# 7. docker/docker-compose.yml
# 8. tests/unit/execution/test_sandbox.py
```

### Step 2: Fix Legacy Dependencies
```bash
# Create requirements-test.txt
# Update pyproject.toml with optional dependencies
# Add conditional imports to affected modules
```

### Step 3: Setup CI/CD
```bash
# Create .github/workflows/ci.yml
# Create .pre-commit-config.yaml
# Run: pre-commit install
```

### Step 4: Add Monitoring
```bash
# Create kosmos/monitoring/metrics.py
# Update kosmos/core/logging_config.py
# Add metrics to workflow components
```

### Step 5: Run Production Verification
```bash
chmod +x scripts/verify_production.sh
./scripts/verify_production.sh
```

---

## Success Criteria

This prompt is complete when:

1. **Gap 4 Complete**:
   - [ ] Docker executor implemented and tested
   - [ ] Package resolver working
   - [ ] Security constraints enforced
   - [ ] Integration with research workflow

2. **Tests Fixed**:
   - [ ] Legacy test dependencies resolved
   - [ ] All tests passing or properly skipped
   - [ ] Test coverage maintained

3. **CI/CD Active**:
   - [ ] GitHub Actions running on push/PR
   - [ ] Pre-commit hooks installed
   - [ ] Docker builds automated

4. **Monitoring Ready**:
   - [ ] Prometheus metrics exposed
   - [ ] Structured logging configured
   - [ ] Health checks implemented

5. **Production Verified**:
   - [ ] verify_production.sh passes
   - [ ] 5-cycle research workflow completes
   - [ ] Documentation updated

---

## Reference

- **Current Implementation**: See `IMPLEMENTATION_REPORT.md`
- **Test Status**: See `TESTS_STATUS.md`
- **Production Plan**: See `PRODUCTION_PLAN.md`
- **Arxiv Paper**: `paper/2511.02824v2.pdf`

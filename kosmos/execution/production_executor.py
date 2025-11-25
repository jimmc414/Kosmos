"""
Production-ready sandboxed executor combining all components.

This is the primary interface for executing code safely:
- Automatic container management with pooling
- Dependency resolution and installation
- Resource limits and security constraints
- Output capture and error handling

Usage:
    executor = ProductionExecutor()
    await executor.initialize()

    result = await executor.execute_code('''
        import pandas as pd
        df = pd.DataFrame({'a': [1, 2, 3]})
        print(df.describe())
        results = {'mean': df['a'].mean()}
    ''')

    print(result.stdout)
    print(result.return_value)  # {'mean': 2.0}

    await executor.cleanup()
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import logging
import time

from .docker_manager import DockerManager, ContainerConfig, ContainerInstance
from .jupyter_client import JupyterClient, ExecutionResult, ExecutionStatus
from .package_resolver import PackageResolver

logger = logging.getLogger(__name__)


@dataclass
class ProductionConfig:
    """Configuration for production executor."""
    # Resource limits
    memory_limit: str = "4g"
    cpu_limit: float = 2.0
    timeout_seconds: int = 600

    # Container pooling
    pool_size: int = 3

    # Package management
    auto_install_packages: bool = True
    package_install_timeout: int = 120

    # Security
    network_enabled: bool = False
    readonly_filesystem: bool = True

    # Docker image
    image: str = "kosmos-sandbox:latest"


class ProductionExecutor:
    """
    Production-ready sandboxed code executor.

    This executor provides:
    - Container pooling for low-latency execution
    - Automatic dependency detection and installation
    - Security isolation (no network, read-only fs)
    - Resource limits (CPU, memory, timeout)
    - Comprehensive output capture

    The executor manages a pool of pre-warmed containers to minimize
    cold-start latency when executing generated scientific code.

    Example:
        async def run_experiment():
            executor = ProductionExecutor(ProductionConfig(
                memory_limit="8g",
                timeout_seconds=1200
            ))
            await executor.initialize()

            try:
                result = await executor.execute_code('''
                    import pandas as pd
                    import numpy as np

                    # Generate some data
                    data = np.random.randn(1000, 5)
                    df = pd.DataFrame(data, columns=['a', 'b', 'c', 'd', 'e'])

                    # Compute correlations
                    correlations = df.corr()
                    print(correlations)

                    results = {'correlations': correlations.to_dict()}
                ''')

                if result.success:
                    print("Results:", result.return_value)
                else:
                    print("Error:", result.error_message)

            finally:
                await executor.cleanup()
    """

    def __init__(self, config: Optional[ProductionConfig] = None):
        """
        Initialize production executor.

        Args:
            config: Executor configuration
        """
        self.config = config or ProductionConfig()
        self._docker_manager: Optional[DockerManager] = None
        self._initialized = False
        self._execution_count = 0

    async def initialize(self):
        """Initialize the executor and warm up container pool."""
        if self._initialized:
            logger.debug("Executor already initialized")
            return

        logger.info("Initializing ProductionExecutor")

        # Create container configuration
        container_config = ContainerConfig(
            image=self.config.image,
            memory_limit=self.config.memory_limit,
            cpu_limit=self.config.cpu_limit,
            timeout_seconds=self.config.timeout_seconds,
            network_mode="bridge" if self.config.network_enabled else "none",
            readonly_rootfs=self.config.readonly_filesystem
        )

        # Initialize Docker manager with pool
        self._docker_manager = DockerManager(
            config=container_config,
            pool_size=self.config.pool_size
        )

        await self._docker_manager.initialize_pool()

        self._initialized = True
        logger.info("ProductionExecutor initialized successfully")

    async def execute_code(
        self,
        code: str,
        timeout: Optional[int] = None,
        install_packages: Optional[bool] = None
    ) -> ExecutionResult:
        """
        Execute code in sandboxed environment.

        Args:
            code: Python code to execute
            timeout: Optional timeout override (seconds)
            install_packages: Override auto_install_packages setting

        Returns:
            ExecutionResult with status, outputs, and any errors
        """
        if not self._initialized:
            await self.initialize()

        timeout = timeout or self.config.timeout_seconds
        should_install = (
            install_packages if install_packages is not None
            else self.config.auto_install_packages
        )

        start_time = time.time()
        self._execution_count += 1
        execution_id = self._execution_count

        logger.info(f"Execution {execution_id}: Starting code execution")

        # Get container from pool
        container = await self._docker_manager.get_container()
        container_healthy = True

        try:
            # Create clients for this container
            jupyter = JupyterClient(
                container.container_id,
                self._docker_manager.client
            )

            # Auto-install dependencies if enabled
            if should_install:
                resolver = PackageResolver(
                    self._docker_manager.client,
                    container.container_id
                )

                success, failed = await resolver.ensure_dependencies(code)
                if not success:
                    logger.warning(
                        f"Execution {execution_id}: Failed to install packages: {failed}"
                    )
                    # Continue execution - might still work with pre-installed packages

            # Execute code
            result = await jupyter.execute_code(code, timeout)

            execution_time = time.time() - start_time
            result.execution_time = execution_time

            if result.success:
                logger.info(
                    f"Execution {execution_id}: Completed successfully in {execution_time:.2f}s"
                )
            else:
                logger.warning(
                    f"Execution {execution_id}: Failed - {result.error_message}"
                )
                # Mark container as potentially unhealthy on error
                if result.status == ExecutionStatus.TIMEOUT:
                    container_healthy = False

            return result

        except Exception as e:
            logger.error(f"Execution {execution_id}: Unexpected error - {e}")
            container_healthy = False

            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error_message=str(e),
                execution_time=time.time() - start_time
            )

        finally:
            # Release container back to pool
            await self._docker_manager.release_container(
                container.container_id,
                healthy=container_healthy
            )

    async def execute_notebook(
        self,
        notebook_content: Dict[str, Any],
        timeout_per_cell: int = 300,
        stop_on_error: bool = True
    ) -> List[ExecutionResult]:
        """
        Execute a Jupyter notebook.

        Args:
            notebook_content: Parsed notebook JSON
            timeout_per_cell: Timeout for each cell
            stop_on_error: Whether to stop on first error

        Returns:
            List of ExecutionResults, one per code cell
        """
        if not self._initialized:
            await self.initialize()

        logger.info("Starting notebook execution")

        container = await self._docker_manager.get_container()
        container_healthy = True

        try:
            jupyter = JupyterClient(
                container.container_id,
                self._docker_manager.client
            )

            # Install all dependencies from notebook first
            if self.config.auto_install_packages:
                await self._install_notebook_dependencies(
                    container.container_id,
                    notebook_content
                )

            # Execute notebook
            results = await jupyter.execute_notebook(
                notebook_content,
                timeout_per_cell,
                stop_on_error
            )

            # Check if last result was a failure/timeout
            if results and results[-1].status in (
                ExecutionStatus.FAILED,
                ExecutionStatus.TIMEOUT
            ):
                container_healthy = False

            logger.info(f"Notebook execution complete: {len(results)} cells executed")
            return results

        except Exception as e:
            logger.error(f"Notebook execution error: {e}")
            container_healthy = False
            return [ExecutionResult(
                status=ExecutionStatus.FAILED,
                error_message=str(e)
            )]

        finally:
            await self._docker_manager.release_container(
                container.container_id,
                healthy=container_healthy
            )

    async def _install_notebook_dependencies(
        self,
        container_id: str,
        notebook_content: Dict[str, Any]
    ):
        """Install all dependencies needed by a notebook."""
        resolver = PackageResolver(
            self._docker_manager.client,
            container_id
        )

        # Extract code from all cells
        all_code = []
        for cell in notebook_content.get("cells", []):
            if cell.get("cell_type") == "code":
                source = cell.get("source", [])
                if isinstance(source, list):
                    all_code.append("".join(source))
                else:
                    all_code.append(source)

        combined_code = "\n".join(all_code)
        await resolver.ensure_dependencies(combined_code)

    async def check_health(self) -> Dict[str, Any]:
        """
        Check executor health and pool status.

        Returns:
            Dictionary with health status
        """
        if not self._initialized:
            return {
                "status": "not_initialized",
                "pool": None
            }

        pool_stats = self._docker_manager.get_pool_stats()

        return {
            "status": "healthy" if pool_stats["ready"] > 0 else "degraded",
            "initialized": self._initialized,
            "execution_count": self._execution_count,
            "pool": pool_stats
        }

    async def cleanup(self):
        """Cleanup all resources."""
        logger.info("Cleaning up ProductionExecutor")

        if self._docker_manager:
            await self._docker_manager.cleanup()

        self._initialized = False
        logger.info("ProductionExecutor cleaned up")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()


# Convenience function for one-off execution
async def execute_code_safely(
    code: str,
    timeout: int = 300,
    memory_limit: str = "4g"
) -> ExecutionResult:
    """
    Execute code safely in an isolated container.

    This is a convenience function for one-off executions.
    For multiple executions, use ProductionExecutor directly
    to benefit from container pooling.

    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds
        memory_limit: Container memory limit

    Returns:
        ExecutionResult

    Example:
        result = await execute_code_safely('''
            import pandas as pd
            df = pd.DataFrame({'x': [1, 2, 3]})
            results = {'sum': df['x'].sum()}
        ''')
        print(result.return_value)  # {'sum': 6}
    """
    config = ProductionConfig(
        memory_limit=memory_limit,
        timeout_seconds=timeout,
        pool_size=1  # Single container for one-off execution
    )

    async with ProductionExecutor(config) as executor:
        return await executor.execute_code(code, timeout)

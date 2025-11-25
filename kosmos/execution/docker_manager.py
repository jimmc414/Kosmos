"""
Docker container management for sandboxed code execution.

Manages container lifecycle:
- Pool of pre-warmed containers for performance
- Container creation with resource limits
- Cleanup after execution
- Health monitoring

This module provides production-ready container pooling to reduce
cold-start latency when executing generated scientific code.
"""

import asyncio
import docker
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import logging
import time
from enum import Enum
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class ContainerStatus(Enum):
    """Status of a container in the pool."""
    READY = "ready"
    IN_USE = "in_use"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"


@dataclass
class ContainerConfig:
    """Configuration for execution containers."""
    image: str = "kosmos-sandbox:latest"
    memory_limit: str = "4g"
    cpu_limit: float = 2.0
    timeout_seconds: int = 600
    network_mode: str = "none"  # Network isolation by default
    readonly_rootfs: bool = True
    working_dir: str = "/workspace"
    # Security options
    security_opt: List[str] = field(default_factory=lambda: ["no-new-privileges:true"])
    cap_drop: List[str] = field(default_factory=lambda: ["ALL"])
    # Temporary writable directories
    tmpfs: Dict[str, str] = field(default_factory=lambda: {
        "/tmp": "size=512m,mode=1777",
        "/home/sandbox/.local": "size=1g,mode=755"
    })


@dataclass
class ContainerInstance:
    """Represents a container in the pool."""
    container_id: str
    container: Any  # docker.models.containers.Container
    status: ContainerStatus
    created_at: float
    last_used_at: float
    config: ContainerConfig
    use_count: int = 0

    @property
    def age_seconds(self) -> float:
        """Get container age in seconds."""
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        """Get time since last use in seconds."""
        return time.time() - self.last_used_at


class DockerManager:
    """
    Manages Docker containers for code execution with pooling.

    Provides:
    - Container pool for reduced cold-start latency
    - Automatic container lifecycle management
    - Health monitoring and recovery
    - Resource limit enforcement

    Usage:
        manager = DockerManager()
        await manager.initialize_pool()

        container = await manager.get_container()
        # ... use container ...
        await manager.release_container(container.container_id)

        await manager.cleanup()
    """

    # Pool configuration
    DEFAULT_POOL_SIZE = 3
    MAX_CONTAINER_AGE_SECONDS = 3600  # 1 hour
    MAX_CONTAINER_USES = 100
    HEALTH_CHECK_INTERVAL = 60  # seconds

    def __init__(
        self,
        config: Optional[ContainerConfig] = None,
        pool_size: int = DEFAULT_POOL_SIZE
    ):
        """
        Initialize Docker manager.

        Args:
            config: Container configuration
            pool_size: Number of containers to keep in pool
        """
        self.config = config or ContainerConfig()
        self._pool_size = pool_size
        self._container_pool: Dict[str, ContainerInstance] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        self._health_check_task: Optional[asyncio.Task] = None

        # Initialize Docker client
        try:
            self.client = docker.from_env()
            logger.info("Docker client initialized successfully")
        except docker.errors.DockerException as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise RuntimeError(f"Docker not available: {e}")

    async def initialize_pool(self):
        """Pre-warm container pool for faster execution."""
        if self._initialized:
            return

        logger.info(f"Initializing container pool (size={self._pool_size})")

        # Verify image exists
        await self._verify_image()

        # Create initial containers
        creation_tasks = [
            self._create_container()
            for _ in range(self._pool_size)
        ]

        results = await asyncio.gather(*creation_tasks, return_exceptions=True)

        successful = sum(1 for r in results if isinstance(r, ContainerInstance))
        logger.info(f"Container pool initialized: {successful}/{self._pool_size} containers ready")

        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        self._initialized = True

    async def _verify_image(self):
        """Verify Docker image exists."""
        try:
            self.client.images.get(self.config.image)
            logger.info(f"Docker image '{self.config.image}' found")
        except docker.errors.ImageNotFound:
            logger.warning(f"Docker image '{self.config.image}' not found")
            raise RuntimeError(
                f"Docker image '{self.config.image}' not found. "
                f"Build it with: cd docker/sandbox && docker build -t {self.config.image} ."
            )

    async def _create_container(self) -> ContainerInstance:
        """Create a new container with security constraints."""
        try:
            container = self.client.containers.run(
                self.config.image,
                detach=True,
                mem_limit=self.config.memory_limit,
                nano_cpus=int(self.config.cpu_limit * 1e9),
                network_mode=self.config.network_mode,
                read_only=self.config.readonly_rootfs,
                working_dir=self.config.working_dir,
                security_opt=self.config.security_opt,
                cap_drop=self.config.cap_drop,
                tmpfs=self.config.tmpfs,
                environment={
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONUNBUFFERED": "1",
                    "MPLBACKEND": "Agg"
                },
                # Keep alive for reuse
                command=["tail", "-f", "/dev/null"],
                # Remove on exit if not managed
                auto_remove=False
            )

            now = time.time()
            instance = ContainerInstance(
                container_id=container.id,
                container=container,
                status=ContainerStatus.READY,
                created_at=now,
                last_used_at=now,
                config=self.config
            )

            async with self._lock:
                self._container_pool[container.id] = instance

            logger.debug(f"Created container {container.short_id}")
            return instance

        except Exception as e:
            logger.error(f"Failed to create container: {e}")
            raise

    async def get_container(self) -> ContainerInstance:
        """
        Get an available container from pool or create new.

        Returns:
            ContainerInstance ready for use
        """
        if not self._initialized:
            await self.initialize_pool()

        async with self._lock:
            # Find available container
            for cid, instance in self._container_pool.items():
                if instance.status == ContainerStatus.READY:
                    # Check if container is still healthy
                    if self._is_container_healthy(instance):
                        instance.status = ContainerStatus.IN_USE
                        instance.last_used_at = time.time()
                        instance.use_count += 1
                        logger.debug(f"Acquired container {instance.container.short_id}")
                        return instance
                    else:
                        # Mark as unhealthy for cleanup
                        instance.status = ContainerStatus.UNHEALTHY

        # Create new container if none available
        logger.info("No available containers in pool, creating new one")
        instance = await self._create_container()
        instance.status = ContainerStatus.IN_USE
        instance.use_count = 1
        return instance

    def _is_container_healthy(self, instance: ContainerInstance) -> bool:
        """Check if container is healthy and can be reused."""
        try:
            # Check container is running
            instance.container.reload()
            if instance.container.status != "running":
                return False

            # Check age
            if instance.age_seconds > self.MAX_CONTAINER_AGE_SECONDS:
                logger.debug(f"Container {instance.container.short_id} too old")
                return False

            # Check use count
            if instance.use_count >= self.MAX_CONTAINER_USES:
                logger.debug(f"Container {instance.container.short_id} exceeded use limit")
                return False

            return True

        except Exception as e:
            logger.warning(f"Health check failed for container: {e}")
            return False

    async def release_container(self, container_id: str, healthy: bool = True):
        """
        Release container back to pool or destroy if unhealthy.

        Args:
            container_id: ID of container to release
            healthy: Whether the container is still healthy
        """
        async with self._lock:
            if container_id not in self._container_pool:
                return

            instance = self._container_pool[container_id]

            if healthy and self._is_container_healthy(instance):
                instance.status = ContainerStatus.READY
                logger.debug(f"Released container {instance.container.short_id} back to pool")
            else:
                # Remove unhealthy container
                await self._remove_container(instance)

                # Replenish pool
                if len([i for i in self._container_pool.values()
                       if i.status == ContainerStatus.READY]) < self._pool_size:
                    asyncio.create_task(self._create_container())

    async def _remove_container(self, instance: ContainerInstance):
        """Remove and cleanup a container."""
        try:
            instance.container.stop(timeout=5)
            instance.container.remove(force=True)
            logger.debug(f"Removed container {instance.container.short_id}")
        except Exception as e:
            logger.warning(f"Error removing container: {e}")

        if instance.container_id in self._container_pool:
            del self._container_pool[instance.container_id]

    async def _health_check_loop(self):
        """Background task to monitor container health."""
        while True:
            try:
                await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)
                await self._run_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _run_health_checks(self):
        """Run health checks on all containers."""
        unhealthy = []

        async with self._lock:
            for cid, instance in self._container_pool.items():
                if instance.status == ContainerStatus.READY:
                    if not self._is_container_healthy(instance):
                        instance.status = ContainerStatus.UNHEALTHY
                        unhealthy.append(instance)

        # Remove unhealthy containers outside lock
        for instance in unhealthy:
            await self._remove_container(instance)

        # Replenish pool
        current_ready = len([
            i for i in self._container_pool.values()
            if i.status == ContainerStatus.READY
        ])

        if current_ready < self._pool_size:
            to_create = self._pool_size - current_ready
            logger.info(f"Replenishing pool with {to_create} containers")
            for _ in range(to_create):
                try:
                    await self._create_container()
                except Exception as e:
                    logger.error(f"Failed to replenish pool: {e}")

    @asynccontextmanager
    async def container_context(self):
        """
        Context manager for container acquisition and release.

        Usage:
            async with manager.container_context() as container:
                # use container
        """
        container = await self.get_container()
        try:
            yield container
        finally:
            await self.release_container(container.container_id)

    async def cleanup(self):
        """Cleanup all containers and resources."""
        logger.info("Cleaning up Docker manager")

        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Remove all containers
        async with self._lock:
            for cid in list(self._container_pool.keys()):
                instance = self._container_pool[cid]
                try:
                    instance.container.stop(timeout=5)
                    instance.container.remove(force=True)
                except Exception as e:
                    logger.warning(f"Error cleaning container {cid}: {e}")

            self._container_pool.clear()

        self._initialized = False
        logger.info("Docker manager cleaned up")

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics about the container pool."""
        ready = in_use = unhealthy = 0

        for instance in self._container_pool.values():
            if instance.status == ContainerStatus.READY:
                ready += 1
            elif instance.status == ContainerStatus.IN_USE:
                in_use += 1
            elif instance.status == ContainerStatus.UNHEALTHY:
                unhealthy += 1

        return {
            "total": len(self._container_pool),
            "ready": ready,
            "in_use": in_use,
            "unhealthy": unhealthy,
            "target_size": self._pool_size
        }

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.client.close()
        except Exception:
            pass

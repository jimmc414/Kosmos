"""
Tests for Docker container management.

Tests container pooling, lifecycle management, and health monitoring
using mocks to avoid requiring actual Docker installation.
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from kosmos.execution.docker_manager import (
    DockerManager,
    ContainerConfig,
    ContainerInstance,
    ContainerStatus,
)


class TestContainerConfig:
    """Tests for ContainerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ContainerConfig()

        assert config.image == "kosmos-sandbox:latest"
        assert config.memory_limit == "4g"
        assert config.cpu_limit == 2.0
        assert config.timeout_seconds == 600
        assert config.network_mode == "none"
        assert config.readonly_rootfs is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ContainerConfig(
            image="custom-image:v1",
            memory_limit="8g",
            cpu_limit=4.0,
            timeout_seconds=1200,
            network_mode="bridge"
        )

        assert config.image == "custom-image:v1"
        assert config.memory_limit == "8g"
        assert config.cpu_limit == 4.0
        assert config.timeout_seconds == 1200
        assert config.network_mode == "bridge"

    def test_security_defaults(self):
        """Test security-related defaults."""
        config = ContainerConfig()

        assert "no-new-privileges:true" in config.security_opt
        assert "ALL" in config.cap_drop
        assert "/tmp" in config.tmpfs


class TestContainerInstance:
    """Tests for ContainerInstance dataclass."""

    def test_instance_creation(self):
        """Test container instance creation."""
        import time
        now = time.time()

        instance = ContainerInstance(
            container_id="abc123",
            container=Mock(),
            status=ContainerStatus.READY,
            created_at=now,
            last_used_at=now,
            config=ContainerConfig()
        )

        assert instance.container_id == "abc123"
        assert instance.status == ContainerStatus.READY
        assert instance.use_count == 0

    def test_instance_age(self):
        """Test age calculation."""
        import time
        old_time = time.time() - 100

        instance = ContainerInstance(
            container_id="abc123",
            container=Mock(),
            status=ContainerStatus.READY,
            created_at=old_time,
            last_used_at=old_time,
            config=ContainerConfig()
        )

        assert instance.age_seconds >= 100

    def test_instance_idle_time(self):
        """Test idle time calculation."""
        import time
        now = time.time()
        old_time = now - 50

        instance = ContainerInstance(
            container_id="abc123",
            container=Mock(),
            status=ContainerStatus.READY,
            created_at=old_time,
            last_used_at=old_time,
            config=ContainerConfig()
        )

        assert instance.idle_seconds >= 50


class TestContainerStatus:
    """Tests for ContainerStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert ContainerStatus.READY.value == "ready"
        assert ContainerStatus.IN_USE.value == "in_use"
        assert ContainerStatus.UNHEALTHY.value == "unhealthy"
        assert ContainerStatus.STOPPED.value == "stopped"


class TestDockerManagerInit:
    """Tests for DockerManager initialization."""

    @patch('kosmos.execution.docker_manager.docker')
    def test_init_with_default_config(self, mock_docker):
        """Test initialization with default config."""
        mock_docker.from_env.return_value = Mock()

        manager = DockerManager()

        assert manager.config is not None
        assert manager._pool_size == DockerManager.DEFAULT_POOL_SIZE
        assert not manager._initialized

    @patch('kosmos.execution.docker_manager.docker')
    def test_init_with_custom_config(self, mock_docker):
        """Test initialization with custom config."""
        mock_docker.from_env.return_value = Mock()

        config = ContainerConfig(memory_limit="16g")
        manager = DockerManager(config=config, pool_size=5)

        assert manager.config.memory_limit == "16g"
        assert manager._pool_size == 5

    @patch('kosmos.execution.docker_manager.docker')
    def test_init_docker_unavailable(self, mock_docker):
        """Test initialization when Docker is not available."""
        # Create a proper exception class that inherits from BaseException
        class MockDockerException(Exception):
            pass

        mock_docker.errors.DockerException = MockDockerException
        mock_docker.from_env.side_effect = MockDockerException("Docker not running")

        with pytest.raises(RuntimeError, match="Docker not available"):
            DockerManager()


class TestDockerManagerPoolStats:
    """Tests for pool statistics."""

    @patch('kosmos.execution.docker_manager.docker')
    def test_get_pool_stats_empty(self, mock_docker):
        """Test pool stats when pool is empty."""
        mock_docker.from_env.return_value = Mock()

        manager = DockerManager()
        stats = manager.get_pool_stats()

        assert stats["total"] == 0
        assert stats["ready"] == 0
        assert stats["in_use"] == 0
        assert stats["unhealthy"] == 0
        assert stats["target_size"] == manager._pool_size

    @patch('kosmos.execution.docker_manager.docker')
    def test_get_pool_stats_with_containers(self, mock_docker):
        """Test pool stats with containers in pool."""
        mock_docker.from_env.return_value = Mock()

        manager = DockerManager()

        # Add mock containers to pool
        import time
        now = time.time()

        manager._container_pool["c1"] = ContainerInstance(
            container_id="c1",
            container=Mock(),
            status=ContainerStatus.READY,
            created_at=now,
            last_used_at=now,
            config=ContainerConfig()
        )
        manager._container_pool["c2"] = ContainerInstance(
            container_id="c2",
            container=Mock(),
            status=ContainerStatus.IN_USE,
            created_at=now,
            last_used_at=now,
            config=ContainerConfig()
        )

        stats = manager.get_pool_stats()

        assert stats["total"] == 2
        assert stats["ready"] == 1
        assert stats["in_use"] == 1


class TestDockerManagerContainerHealth:
    """Tests for container health checking."""

    @patch('kosmos.execution.docker_manager.docker')
    def test_healthy_container(self, mock_docker):
        """Test health check on healthy container."""
        mock_docker.from_env.return_value = Mock()

        manager = DockerManager()

        mock_container = Mock()
        mock_container.status = "running"
        mock_container.reload = Mock()

        import time
        now = time.time()

        instance = ContainerInstance(
            container_id="c1",
            container=mock_container,
            status=ContainerStatus.READY,
            created_at=now,
            last_used_at=now,
            config=ContainerConfig(),
            use_count=1
        )

        assert manager._is_container_healthy(instance) is True

    @patch('kosmos.execution.docker_manager.docker')
    def test_unhealthy_stopped_container(self, mock_docker):
        """Test health check on stopped container."""
        mock_docker.from_env.return_value = Mock()

        manager = DockerManager()

        mock_container = Mock()
        mock_container.status = "exited"
        mock_container.reload = Mock()

        import time
        now = time.time()

        instance = ContainerInstance(
            container_id="c1",
            container=mock_container,
            status=ContainerStatus.READY,
            created_at=now,
            last_used_at=now,
            config=ContainerConfig()
        )

        assert manager._is_container_healthy(instance) is False

    @patch('kosmos.execution.docker_manager.docker')
    def test_unhealthy_old_container(self, mock_docker):
        """Test health check on container exceeding max age."""
        mock_docker.from_env.return_value = Mock()

        manager = DockerManager()

        mock_container = Mock()
        mock_container.status = "running"
        mock_container.reload = Mock()

        import time
        # Created 2 hours ago (exceeds MAX_CONTAINER_AGE_SECONDS)
        old_time = time.time() - 7200

        instance = ContainerInstance(
            container_id="c1",
            container=mock_container,
            status=ContainerStatus.READY,
            created_at=old_time,
            last_used_at=old_time,
            config=ContainerConfig()
        )

        assert manager._is_container_healthy(instance) is False

    @patch('kosmos.execution.docker_manager.docker')
    def test_unhealthy_overused_container(self, mock_docker):
        """Test health check on container exceeding use limit."""
        mock_docker.from_env.return_value = Mock()

        manager = DockerManager()

        mock_container = Mock()
        mock_container.status = "running"
        mock_container.reload = Mock()

        import time
        now = time.time()

        instance = ContainerInstance(
            container_id="c1",
            container=mock_container,
            status=ContainerStatus.READY,
            created_at=now,
            last_used_at=now,
            config=ContainerConfig(),
            use_count=DockerManager.MAX_CONTAINER_USES + 1
        )

        assert manager._is_container_healthy(instance) is False


# Integration tests (require Docker daemon, skip if not available)
def docker_available():
    """Check if Docker daemon is available."""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not docker_available(),
    reason="Docker daemon not available"
)
class TestDockerManagerIntegration:
    """Integration tests requiring Docker daemon."""

    @pytest.mark.asyncio
    async def test_initialize_pool_no_image(self):
        """Test initialization fails gracefully without image."""
        from kosmos.execution.docker_manager import DockerManager, ContainerConfig

        config = ContainerConfig(image="nonexistent-image:latest")
        manager = DockerManager(config=config)

        with pytest.raises(RuntimeError, match="not found"):
            await manager.initialize_pool()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

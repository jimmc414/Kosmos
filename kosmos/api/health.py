"""
Health check endpoints for Kosmos AI Scientist.

Provides liveness, readiness, and metrics endpoints for monitoring.
"""

import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
import psutil
import platform

logger = logging.getLogger(__name__)


class HealthChecker:
    """
    Health checker for Kosmos system components.

    Provides comprehensive health checks for:
    - Database connectivity
    - Cache availability
    - External API accessibility
    - System resources (CPU, memory, disk)
    - Service readiness
    """

    def __init__(self):
        """Initialize health checker."""
        self.start_time = time.time()
        self.checks_performed = 0
        self.last_check_time: Optional[float] = None

    def get_basic_health(self) -> Dict[str, Any]:
        """
        Get basic liveness health check.

        Returns:
            Dictionary with basic health status
        """
        self.checks_performed += 1
        self.last_check_time = time.time()

        uptime = self.last_check_time - self.start_time

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": round(uptime, 2),
            "checks_performed": self.checks_performed,
            "service": "kosmos-ai-scientist",
            "version": self._get_version()
        }

    def get_readiness_check(self) -> Dict[str, Any]:
        """
        Get readiness check including dependencies.

        Checks if the service is ready to handle requests by verifying:
        - Database connection
        - Cache availability
        - External APIs

        Returns:
            Dictionary with readiness status and component details
        """
        self.checks_performed += 1
        self.last_check_time = time.time()

        components = {}
        all_ready = True

        # Check database
        db_status = self._check_database()
        components["database"] = db_status
        if db_status["status"] != "healthy":
            all_ready = False

        # Check cache
        cache_status = self._check_cache()
        components["cache"] = cache_status
        if cache_status["status"] != "healthy":
            all_ready = False

        # Check external APIs
        api_status = self._check_external_apis()
        components["external_apis"] = api_status
        if api_status["status"] != "healthy":
            all_ready = False

        # Check Neo4j (optional)
        neo4j_status = self._check_neo4j()
        components["neo4j"] = neo4j_status
        # Don't mark as not ready if Neo4j is down (it's optional)

        overall_status = "ready" if all_ready else "not_ready"

        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "components": components,
            "service": "kosmos-ai-scientist",
            "version": self._get_version()
        }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get system metrics.

        Returns:
            Dictionary with system resource metrics
        """
        self.checks_performed += 1
        self.last_check_time = time.time()

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()

        # Memory metrics
        memory = psutil.virtual_memory()
        memory_total = memory.total
        memory_available = memory.available
        memory_used = memory.used
        memory_percent = memory.percent

        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_total = disk.total
        disk_used = disk.used
        disk_free = disk.free
        disk_percent = disk.percent

        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        process_cpu_percent = process.cpu_percent()

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "platform": platform.system(),
                "platform_release": platform.release(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            },
            "cpu": {
                "percent": round(cpu_percent, 2),
                "count": cpu_count,
                "load_average": self._get_load_average()
            },
            "memory": {
                "total_bytes": memory_total,
                "available_bytes": memory_available,
                "used_bytes": memory_used,
                "percent": round(memory_percent, 2),
                "total_gb": round(memory_total / (1024**3), 2),
                "available_gb": round(memory_available / (1024**3), 2),
                "used_gb": round(memory_used / (1024**3), 2)
            },
            "disk": {
                "total_bytes": disk_total,
                "used_bytes": disk_used,
                "free_bytes": disk_free,
                "percent": round(disk_percent, 2),
                "total_gb": round(disk_total / (1024**3), 2),
                "used_gb": round(disk_used / (1024**3), 2),
                "free_gb": round(disk_free / (1024**3), 2)
            },
            "process": {
                "memory_rss_bytes": process_memory.rss,
                "memory_rss_mb": round(process_memory.rss / (1024**2), 2),
                "memory_vms_bytes": process_memory.vms,
                "memory_vms_mb": round(process_memory.vms / (1024**2), 2),
                "cpu_percent": round(process_cpu_percent, 2),
                "num_threads": process.num_threads(),
                "num_fds": self._get_num_fds(process)
            },
            "uptime_seconds": round(time.time() - self.start_time, 2)
        }

    def _check_database(self) -> Dict[str, Any]:
        """
        Check database connectivity.

        Returns:
            Dictionary with database status
        """
        try:
            from kosmos.db import get_session

            with get_session() as session:
                # Try a simple query
                start_time = time.time()
                session.execute("SELECT 1")
                query_time = time.time() - start_time

                return {
                    "status": "healthy",
                    "response_time_ms": round(query_time * 1000, 2),
                    "details": "Database connection successful"
                }

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "Database connection failed"
            }

    def _check_cache(self) -> Dict[str, Any]:
        """
        Check cache availability.

        Returns:
            Dictionary with cache status
        """
        try:
            # Check if Redis is enabled
            redis_enabled = os.getenv("REDIS_ENABLED", "false").lower() == "true"

            if redis_enabled:
                # Try to connect to Redis
                import redis
                redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

                start_time = time.time()
                client = redis.from_url(
                    redis_url,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                    decode_responses=True
                )
                client.ping()
                response_time = time.time() - start_time

                # Get cache info
                info = client.info()

                return {
                    "status": "healthy",
                    "type": "redis",
                    "response_time_ms": round(response_time * 1000, 2),
                    "details": "Redis connection successful",
                    "info": {
                        "version": info.get("redis_version"),
                        "used_memory_mb": round(info.get("used_memory", 0) / (1024**2), 2),
                        "connected_clients": info.get("connected_clients", 0)
                    }
                }
            else:
                # In-memory cache (always available)
                return {
                    "status": "healthy",
                    "type": "memory",
                    "details": "In-memory cache active"
                }

        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return {
                "status": "unhealthy",
                "type": "redis" if redis_enabled else "memory",
                "error": str(e),
                "details": "Cache connection failed"
            }

    def _check_external_apis(self) -> Dict[str, Any]:
        """
        Check external API accessibility.

        Returns:
            Dictionary with external API status
        """
        try:
            # Check if API key is configured
            api_key = os.getenv("ANTHROPIC_API_KEY")

            if not api_key:
                return {
                    "status": "unhealthy",
                    "details": "Anthropic API key not configured",
                    "error": "ANTHROPIC_API_KEY not set"
                }

            # Check if it's the CLI mode (all 9s)
            is_cli_mode = api_key == "9" * len(api_key)

            if is_cli_mode:
                return {
                    "status": "healthy",
                    "mode": "cli",
                    "details": "Claude Code CLI mode configured"
                }
            else:
                # For API mode, we don't want to make actual API calls in health checks
                # Just verify the key format
                if api_key.startswith("sk-ant-"):
                    return {
                        "status": "healthy",
                        "mode": "api",
                        "details": "Anthropic API key configured"
                    }
                else:
                    return {
                        "status": "warning",
                        "mode": "api",
                        "details": "API key format unexpected",
                        "warning": "Key doesn't start with sk-ant-"
                    }

        except Exception as e:
            logger.error(f"External API health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "External API check failed"
            }

    def _check_neo4j(self) -> Dict[str, Any]:
        """
        Check Neo4j connectivity (optional component).

        Returns:
            Dictionary with Neo4j status
        """
        try:
            from neo4j import GraphDatabase

            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD")

            if not password:
                return {
                    "status": "not_configured",
                    "details": "Neo4j password not configured (optional)"
                }

            start_time = time.time()
            driver = GraphDatabase.driver(uri, auth=(user, password))

            # Verify connectivity
            driver.verify_connectivity()
            response_time = time.time() - start_time

            driver.close()

            return {
                "status": "healthy",
                "response_time_ms": round(response_time * 1000, 2),
                "details": "Neo4j connection successful"
            }

        except Exception as e:
            logger.warning(f"Neo4j health check failed (optional component): {e}")
            return {
                "status": "unavailable",
                "details": "Neo4j not available (optional component)",
                "error": str(e)
            }

    def _get_version(self) -> str:
        """Get Kosmos version."""
        try:
            from kosmos import __version__
            return __version__
        except Exception:
            return "unknown"

    def _get_load_average(self) -> Optional[list]:
        """Get system load average (Unix only)."""
        try:
            if hasattr(os, 'getloadavg'):
                load = os.getloadavg()
                return [round(l, 2) for l in load]
            return None
        except Exception:
            return None

    def _get_num_fds(self, process) -> Optional[int]:
        """Get number of file descriptors (Unix only)."""
        try:
            if hasattr(process, 'num_fds'):
                return process.num_fds()
            return None
        except Exception:
            return None


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """
    Get or create the global health checker instance.

    Returns:
        HealthChecker instance
    """
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


# Convenience functions for endpoints
def get_basic_health() -> Dict[str, Any]:
    """Get basic health status."""
    return get_health_checker().get_basic_health()


def get_readiness_check() -> Dict[str, Any]:
    """Get readiness status."""
    return get_health_checker().get_readiness_check()


def get_metrics() -> Dict[str, Any]:
    """Get system metrics."""
    return get_health_checker().get_metrics()

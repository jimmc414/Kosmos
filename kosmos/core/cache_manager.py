"""
Global cache manager for orchestrating multiple cache types.

Provides centralized cache management, statistics aggregation,
and lifecycle operations across all cache instances.
"""

import logging
import threading
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from kosmos.core.cache import BaseCache, HybridCache, InMemoryCache, DiskCache
from kosmos.config import get_config

logger = logging.getLogger(__name__)


class CacheType(Enum):
    """Types of caches managed by the cache manager."""

    CLAUDE = "claude"
    EXPERIMENT = "experiment"
    EMBEDDING = "embedding"
    GENERAL = "general"
    LITERATURE = "literature"


class CacheManager:
    """
    Global cache manager for all caching operations.

    Coordinates multiple cache instances and provides unified
    access to caching across the entire system.
    """

    _instance: Optional['CacheManager'] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern to ensure only one cache manager exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the cache manager."""
        # Skip if already initialized (singleton)
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self._caches: Dict[CacheType, BaseCache] = {}
        self._config = get_config()

        # Initialize caches
        self._initialize_caches()

        logger.info("CacheManager initialized")

    def _initialize_caches(self):
        """Initialize all cache instances based on configuration."""
        cache_dir = Path(".kosmos_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Get cache configuration (48 hours default for new caches)
        ttl_seconds = getattr(
            self._config.performance,
            'cache_ttl',
            172800  # 48 hours
        )

        # Claude API Cache - Hybrid (memory + disk)
        # Hot responses in memory, all persisted to disk
        self._caches[CacheType.CLAUDE] = HybridCache(
            memory_size=1000,  # 1000 most recent responses in memory
            cache_dir=str(cache_dir / "claude"),
            ttl_seconds=ttl_seconds,
            max_size_mb=2000  # 2GB for Claude responses
        )
        logger.info("Initialized Claude API cache (hybrid)")

        # Experiment Results Cache - Disk only (large, persistent)
        self._caches[CacheType.EXPERIMENT] = DiskCache(
            cache_dir=str(cache_dir / "experiments"),
            ttl_seconds=ttl_seconds * 2,  # Longer TTL for experiments (96h)
            max_size_mb=3000  # 3GB for experiment results
        )
        logger.info("Initialized experiment results cache (disk)")

        # Embedding Cache - In-memory (small, frequently accessed)
        self._caches[CacheType.EMBEDDING] = InMemoryCache(
            max_size=5000,  # 5000 embeddings
            ttl_seconds=ttl_seconds * 7  # Week-long TTL (embeddings don't change)
        )
        logger.info("Initialized embedding cache (memory)")

        # General-purpose Cache - Hybrid
        self._caches[CacheType.GENERAL] = HybridCache(
            memory_size=500,
            cache_dir=str(cache_dir / "general"),
            ttl_seconds=ttl_seconds,
            max_size_mb=500  # 500MB
        )
        logger.info("Initialized general cache (hybrid)")

        # Note: Literature cache is managed separately by literature module

    def get_cache(self, cache_type: CacheType) -> Optional[BaseCache]:
        """
        Get a cache instance by type.

        Args:
            cache_type: Type of cache to retrieve

        Returns:
            Cache instance or None if not found
        """
        return self._caches.get(cache_type)

    def get(
        self,
        cache_type: CacheType,
        key: str
    ) -> Optional[Any]:
        """
        Retrieve a value from a cache.

        Args:
            cache_type: Type of cache to query
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        cache = self.get_cache(cache_type)
        if cache is None:
            logger.warning(f"Cache type {cache_type} not found")
            return None

        return cache.get(key)

    def set(
        self,
        cache_type: CacheType,
        key: str,
        value: Any
    ) -> bool:
        """
        Store a value in a cache.

        Args:
            cache_type: Type of cache to use
            key: Cache key
            value: Value to cache

        Returns:
            True if successful, False otherwise
        """
        cache = self.get_cache(cache_type)
        if cache is None:
            logger.warning(f"Cache type {cache_type} not found")
            return False

        return cache.set(key, value)

    def delete(
        self,
        cache_type: CacheType,
        key: str
    ) -> bool:
        """
        Delete a value from a cache.

        Args:
            cache_type: Type of cache
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        cache = self.get_cache(cache_type)
        if cache is None:
            logger.warning(f"Cache type {cache_type} not found")
            return False

        return cache.delete(key)

    def clear(self, cache_type: Optional[CacheType] = None) -> Dict[CacheType, int]:
        """
        Clear cache(s).

        Args:
            cache_type: Type of cache to clear. If None, clear all caches.

        Returns:
            Dictionary mapping cache type to number of entries cleared
        """
        results = {}

        if cache_type is not None:
            # Clear specific cache
            cache = self.get_cache(cache_type)
            if cache:
                count = cache.clear()
                results[cache_type] = count
                logger.info(f"Cleared {cache_type.value} cache: {count} entries")
        else:
            # Clear all caches
            for ctype, cache in self._caches.items():
                count = cache.clear()
                results[ctype] = count
                logger.info(f"Cleared {ctype.value} cache: {count} entries")

        return results

    def cleanup_expired(
        self,
        cache_type: Optional[CacheType] = None
    ) -> Dict[CacheType, int]:
        """
        Remove expired entries from cache(s).

        Args:
            cache_type: Type of cache to clean. If None, clean all caches.

        Returns:
            Dictionary mapping cache type to number of entries removed
        """
        results = {}

        if cache_type is not None:
            # Clean specific cache
            cache = self.get_cache(cache_type)
            if cache:
                count = cache.cleanup_expired()
                results[cache_type] = count
                if count > 0:
                    logger.info(f"Cleaned {cache_type.value} cache: {count} entries")
        else:
            # Clean all caches
            for ctype, cache in self._caches.items():
                count = cache.cleanup_expired()
                results[ctype] = count
                if count > 0:
                    logger.info(f"Cleaned {ctype.value} cache: {count} entries")

        return results

    def get_stats(
        self,
        cache_type: Optional[CacheType] = None
    ) -> Dict[str, Any]:
        """
        Get cache statistics.

        Args:
            cache_type: Type of cache to query. If None, get stats for all caches.

        Returns:
            Dictionary with cache statistics
        """
        if cache_type is not None:
            # Stats for specific cache
            cache = self.get_cache(cache_type)
            if cache:
                stats = cache.get_stats()
                stats['cache_type'] = cache_type.value
                return stats
            return {}

        # Stats for all caches
        all_stats = {
            'caches': {},
            'total_size': 0,
            'total_hits': 0,
            'total_misses': 0,
            'total_sets': 0,
            'total_evictions': 0,
            'total_errors': 0,
        }

        for ctype, cache in self._caches.items():
            cache_stats = cache.get_stats()
            all_stats['caches'][ctype.value] = cache_stats

            # Aggregate totals
            all_stats['total_size'] += cache_stats.get('size', 0)
            all_stats['total_hits'] += cache_stats.get('hits', 0)
            all_stats['total_misses'] += cache_stats.get('misses', 0)
            all_stats['total_sets'] += cache_stats.get('sets', 0)
            all_stats['total_evictions'] += cache_stats.get('evictions', 0)
            all_stats['total_errors'] += cache_stats.get('errors', 0)

        # Calculate overall hit rate
        total_requests = all_stats['total_hits'] + all_stats['total_misses']
        if total_requests > 0:
            all_stats['overall_hit_rate_percent'] = round(
                all_stats['total_hits'] / total_requests * 100,
                2
            )
        else:
            all_stats['overall_hit_rate_percent'] = 0.0

        return all_stats

    def get_cache_types(self) -> List[CacheType]:
        """
        Get list of all cache types.

        Returns:
            List of cache types
        """
        return list(self._caches.keys())

    def generate_key(self, *args, **kwargs) -> str:
        """
        Generate a cache key from arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Hexadecimal cache key
        """
        return BaseCache.generate_key(*args, **kwargs)

    def warm_up(
        self,
        cache_type: CacheType,
        data: Dict[str, Any]
    ) -> int:
        """
        Pre-populate a cache with data (cache warming).

        Args:
            cache_type: Type of cache to warm
            data: Dictionary of key-value pairs to cache

        Returns:
            Number of items cached
        """
        cache = self.get_cache(cache_type)
        if cache is None:
            logger.warning(f"Cache type {cache_type} not found")
            return 0

        count = 0
        for key, value in data.items():
            if cache.set(key, value):
                count += 1

        logger.info(f"Warmed {cache_type.value} cache with {count} entries")
        return count

    def get_size_breakdown(self) -> Dict[str, Any]:
        """
        Get cache size breakdown across all caches.

        Returns:
            Dictionary with size information for each cache
        """
        breakdown = {}

        for ctype, cache in self._caches.items():
            stats = cache.get_stats()
            breakdown[ctype.value] = {
                'entries': stats.get('size', 0),
                'type': cache.__class__.__name__,
            }

            # Add disk size for disk-based caches
            if hasattr(cache, 'cache_dir'):
                cache_dir = Path(cache.cache_dir)
                if cache_dir.exists():
                    total_bytes = sum(
                        f.stat().st_size
                        for f in cache_dir.rglob("*.pkl")
                    )
                    breakdown[ctype.value]['disk_mb'] = round(
                        total_bytes / (1024 * 1024),
                        2
                    )

            # Add hybrid cache breakdown
            if hasattr(cache, 'memory_cache') and hasattr(cache, 'disk_cache'):
                mem_stats = cache.memory_cache.get_stats()
                disk_stats = cache.disk_cache.get_stats()
                breakdown[ctype.value]['memory_entries'] = mem_stats.get('size', 0)
                breakdown[ctype.value]['disk_entries'] = disk_stats.get('size', 0)

        return breakdown

    def get_hit_rates(self) -> Dict[str, float]:
        """
        Get hit rates for all caches.

        Returns:
            Dictionary mapping cache type to hit rate percentage
        """
        hit_rates = {}

        for ctype, cache in self._caches.items():
            stats = cache.get_stats()
            hit_rates[ctype.value] = stats.get('hit_rate_percent', 0.0)

        return hit_rates

    def optimize(self):
        """
        Optimize all caches by cleaning up expired entries.

        This should be called periodically (e.g., daily) to maintain
        cache health and reclaim space.
        """
        logger.info("Starting cache optimization...")

        results = self.cleanup_expired()
        total_cleaned = sum(results.values())

        logger.info(f"Cache optimization complete: removed {total_cleaned} expired entries")

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all caches.

        Returns:
            Dictionary with health status for each cache
        """
        health = {
            'healthy': True,
            'caches': {},
            'warnings': []
        }

        for ctype, cache in self._caches.items():
            try:
                # Test basic operations
                test_key = "__health_check__"
                test_value = {"timestamp": "test"}

                cache.set(test_key, test_value)
                retrieved = cache.get(test_key)
                cache.delete(test_key)

                cache_health = {
                    'status': 'healthy',
                    'size': cache.size(),
                }

                # Check for issues
                stats = cache.get_stats()
                error_rate = stats.get('errors', 0) / max(stats.get('total_requests', 1), 1)

                if error_rate > 0.1:  # >10% error rate
                    cache_health['status'] = 'degraded'
                    health['warnings'].append(
                        f"{ctype.value} cache has high error rate: {error_rate:.1%}"
                    )

                # Check disk usage for disk caches
                if hasattr(cache, 'max_size_mb') and hasattr(cache, 'cache_dir'):
                    cache_dir = Path(cache.cache_dir)
                    if cache_dir.exists():
                        total_mb = sum(
                            f.stat().st_size for f in cache_dir.rglob("*.pkl")
                        ) / (1024 * 1024)
                        usage_percent = (total_mb / cache.max_size_mb) * 100

                        if usage_percent > 90:
                            cache_health['status'] = 'warning'
                            health['warnings'].append(
                                f"{ctype.value} cache is {usage_percent:.1f}% full"
                            )

                health['caches'][ctype.value] = cache_health

            except Exception as e:
                logger.error(f"Health check failed for {ctype.value} cache: {e}")
                health['healthy'] = False
                health['caches'][ctype.value] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }

        return health


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None
_manager_lock = threading.Lock()


def get_cache_manager() -> CacheManager:
    """
    Get or create the global cache manager instance.

    Returns:
        CacheManager instance
    """
    global _cache_manager

    if _cache_manager is None:
        with _manager_lock:
            if _cache_manager is None:
                _cache_manager = CacheManager()

    return _cache_manager


def reset_cache_manager():
    """Reset the global cache manager (useful for testing)."""
    global _cache_manager
    _cache_manager = None

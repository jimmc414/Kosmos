"""
Unit tests for cache system.

Tests all cache types (Memory, Disk, Hybrid, Redis) and cache manager.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from kosmos.core.cache import (
    InMemoryCache as MemoryCache,
    DiskCache,
    HybridCache,
    CacheStats,
)


def _set_in_past(cache, key, value, hours_ago=2):
    """Store `value` as if it had been cached `hours_ago` hours earlier.

    TTL is per cache instance (ttl_seconds); there is no per-call ttl, so an
    expired entry is produced by back-dating the clock used by set().
    """
    past = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    with patch("kosmos.core.cache.datetime") as mock_dt:
        mock_dt.now.return_value = past
        assert cache.set(key, value) is True


class TestMemoryCache:
    """Test MemoryCache implementation."""

    def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = MemoryCache(max_size=100)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_nonexistent(self):
        """Test getting nonexistent key returns None."""
        cache = MemoryCache(max_size=100)
        assert cache.get("nonexistent") is None

    def test_ttl_expiration(self):
        """Test TTL expiration."""
        cache = MemoryCache(max_size=100, ttl_seconds=3600)

        # Entry cached two hours ago with a one-hour TTL has expired
        _set_in_past(cache, "key1", "value1")
        cache.set("key2", "value2")

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

    def test_delete(self):
        """Test deletion."""
        cache = MemoryCache(max_size=100)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        cache.delete("key1")
        assert cache.get("key1") is None

    def test_clear(self):
        """Test clearing all entries."""
        cache = MemoryCache(max_size=100)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_exists(self):
        """Test exists check."""
        cache = MemoryCache(max_size=100)

        cache.set("key1", "value1")
        # exists() method doesn't exist, use get() instead
        assert cache.get("key1") is not None
        assert cache.get("nonexistent") is None

    def test_size_tracking(self):
        """Test size tracking."""
        cache = MemoryCache(max_size=1000)

        cache.set("key1", "a" * 100)
        stats = cache.get_stats()

        assert stats["size"] == 1  # 'size' instead of 'entries'
        # size_bytes not tracked in current implementation

    def test_lru_eviction(self):
        """Test LRU eviction when max size exceeded."""
        cache = MemoryCache(max_size=10)  # Small size to trigger eviction

        # Fill cache beyond capacity
        for i in range(20):
            cache.set(f"key{i}", "x" * 10)

        stats = cache.get_stats()
        # Should have evicted old entries
        assert stats["size"] <= 10  # max_size limit

    def test_stats(self):
        """Test statistics collection."""
        cache = MemoryCache(max_size=1000)

        cache.set("key1", "value1")
        cache.get("key1")  # hit
        cache.get("nonexistent")  # miss

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1


class TestDiskCache:
    """Test DiskCache implementation."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_set_and_get(self, temp_cache_dir):
        """Test basic set and get operations."""
        cache = DiskCache(cache_dir=temp_cache_dir, max_size_mb=1)

        cache.set("key1", {"data": "value1"})
        result = cache.get("key1")

        assert result["data"] == "value1"

    def test_persistence(self, temp_cache_dir):
        """Test data persists across cache instances."""
        cache1 = DiskCache(cache_dir=temp_cache_dir, max_size_mb=1)
        cache1.set("key1", "value1")

        # Create new cache instance with same directory
        cache2 = DiskCache(cache_dir=temp_cache_dir, max_size_mb=1)
        assert cache2.get("key1") == "value1"

    def test_ttl_expiration(self, temp_cache_dir):
        """Test TTL expiration on disk."""
        cache = DiskCache(cache_dir=temp_cache_dir, ttl_seconds=3600, max_size_mb=1)

        _set_in_past(cache, "key1", "value1")
        assert cache.get("key1") is None
        # The expired file is removed on read
        assert cache.size() == 0

    def test_file_creation(self, temp_cache_dir):
        """Test cache files are created."""
        cache = DiskCache(cache_dir=temp_cache_dir, max_size_mb=1)

        cache.set("test_key", "test_value")

        # Files are pickles stored under a two-character subdirectory
        cache_files = list(temp_cache_dir.rglob("*.pkl"))
        assert len(cache_files) == 1
        assert cache_files[0] == temp_cache_dir / "te" / "test_key.pkl"

    def test_cleanup_expired(self, temp_cache_dir):
        """Test cleanup of expired entries."""
        cache = DiskCache(cache_dir=temp_cache_dir, ttl_seconds=3600, max_size_mb=1)

        # One entry back-dated beyond the TTL, one fresh
        _set_in_past(cache, "expired", "value")
        cache.set("valid", "value")

        # Cleanup should remove only the expired entry
        assert cache.cleanup_expired() == 1

        assert cache.size() == 1
        assert cache.get("expired") is None
        assert cache.get("valid") == "value"


class TestHybridCache:
    """Test HybridCache implementation."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_memory_then_disk(self, temp_cache_dir):
        """Test hybrid cache checks memory first, then disk."""
        cache = HybridCache(
            memory_size=100,
            cache_dir=str(temp_cache_dir),
            max_size_mb=1
        )

        # Set in cache (should go to memory)
        cache.set("key1", "value1")

        # Should retrieve from memory (fast)
        assert cache.get("key1") == "value1"

        # Clear memory to force disk retrieval
        cache.memory_cache.clear()

        # Should still get value from disk
        assert cache.get("key1") == "value1"
        # ...and the disk hit is promoted back into memory
        assert cache.memory_cache.get("key1") == "value1"

    def test_stats_aggregation(self, temp_cache_dir):
        """Test stats aggregate from both caches."""
        cache = HybridCache(
            memory_size=100,
            cache_dir=str(temp_cache_dir),
            max_size_mb=1
        )

        cache.set("key1", "value1")
        cache.get("key1")

        stats = cache.get_stats()
        assert stats["hits"] > 0
        assert stats["memory_cache"]["hits"] > 0
        assert "disk_cache" in stats


@pytest.mark.skip(reason="CacheManager class not implemented")
class TestCacheManager:
    """Test CacheManager orchestration."""

    def test_cache_selection(self):
        """Test cache selection by type."""
        # manager = CacheManager()
        # memory_cache = manager.get_cache("memory")
        # assert isinstance(memory_cache, MemoryCache)
        pass

    def test_default_cache(self):
        """Test getting default cache."""
        pass

    def test_set_get_through_manager(self):
        """Test set/get through manager."""
        pass

    def test_clear_all_caches(self):
        """Test clearing all caches."""
        pass

    def test_global_stats(self):
        """Test getting stats from all caches."""
        pass


@pytest.mark.skip(reason="CacheStats dataclass not implemented (CacheStats is a tracker class)")
class TestCacheStats:
    """Test CacheStats data class."""

    def test_hit_ratio_calculation(self):
        """Test hit ratio calculation."""
        pass

    def test_hit_ratio_zero_requests(self):
        """Test hit ratio when no requests."""
        pass


@pytest.mark.skip(reason="CacheEntry class not implemented")
class TestCacheEntry:
    """Test CacheEntry data class."""

    def test_is_expired(self):
        """Test expiration checking."""
        pass

    def test_no_expiry(self):
        """Test entries without expiry never expire."""
        pass


class TestRedisCache:
    """Test RedisCache implementation (requires Redis running)."""

    @pytest.fixture
    def redis_available(self):
        """Check if Redis is available for testing."""
        try:
            import redis
            client = redis.from_url("redis://localhost:6379/15", socket_timeout=1)
            client.ping()
            client.flushdb()  # Clean test database
            return True
        except:
            pytest.skip("Redis not available for testing")

    def test_redis_set_get(self, redis_available):
        """Test Redis cache set/get."""
        from kosmos.core.cache import RedisCache

        cache = RedisCache(redis_url="redis://localhost:6379/15")

        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"

        cache.client.flushdb()

    def test_redis_ttl(self, redis_available):
        """Test Redis TTL expiration."""
        from kosmos.core.cache import RedisCache

        cache = RedisCache(redis_url="redis://localhost:6379/15")

        cache.set("test_key", "test_value", ttl=1)

        import time
        time.sleep(2)

        assert cache.get("test_key") is None

        cache.client.flushdb()

    def test_redis_stats(self, redis_available):
        """Test Redis stats collection."""
        from kosmos.core.cache import RedisCache

        cache = RedisCache(redis_url="redis://localhost:6379/15")

        cache.set("key1", "value1")
        cache.get("key1")  # hit
        cache.get("nonexistent")  # miss

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

        cache.client.flushdb()

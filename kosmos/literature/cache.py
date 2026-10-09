"""
HTTP caching layer for literature API clients.

Provides disk-based caching with configurable TTL to reduce API calls
and respect rate limits.
"""

import hashlib
import json
import os
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)

# Relative to the working directory unless KOSMOS_LITERATURE_CACHE_DIR names another place
# (tests/conftest.py points it into a temp dir so tests never write the repo's cache).
DEFAULT_CACHE_DIR = ".literature_cache"


def _default_cache_dir() -> str:
    return os.environ.get("KOSMOS_LITERATURE_CACHE_DIR") or DEFAULT_CACHE_DIR


class LiteratureCacheError(Exception):
    """Exception raised for cache-related errors."""
    pass


class LiteratureCache:
    """
    Disk-based cache for literature API responses.

    Implements a simple file-based caching system with TTL support.
    Each cached response is stored as a separate pickle file with metadata.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        ttl_hours: int = 48,
        max_cache_size_mb: int = 1000
    ):
        """
        Initialize the literature cache.

        Args:
            cache_dir: Directory to store cache files (default: KOSMOS_LITERATURE_CACHE_DIR,
                else .literature_cache in the working directory)
            ttl_hours: Time-to-live for cached responses in hours (default: 48)
            max_cache_size_mb: Maximum cache directory size in MB
        """
        cache_dir = cache_dir or _default_cache_dir()
        self.cache_dir = Path(cache_dir)
        self.ttl_hours = ttl_hours
        self.max_cache_size_mb = max_cache_size_mb

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized cache: dir={cache_dir}, ttl={ttl_hours}h")

    def _generate_cache_key(self, source: str, endpoint: str, params: Dict[str, Any]) -> str:
        """
        Generate a unique cache key for the request.

        Args:
            source: API source (e.g., "arxiv", "semantic_scholar")
            endpoint: API endpoint (e.g., "search", "paper")
            params: Request parameters

        Returns:
            Hexadecimal cache key
        """
        # Sort params for consistent key generation
        param_str = json.dumps(params, sort_keys=True)

        # Create hash from source + endpoint + params
        key_input = f"{source}:{endpoint}:{param_str}"
        cache_key = hashlib.sha256(key_input.encode()).hexdigest()

        return cache_key

    def _get_cache_path(self, cache_key: str) -> Path:
        """
        Get the file path for a cache key.

        Args:
            cache_key: Cache key

        Returns:
            Path to cache file
        """
        # Use first 2 chars of hash for subdirectory (distribute files)
        subdir = self.cache_dir / cache_key[:2]
        subdir.mkdir(exist_ok=True)

        return subdir / f"{cache_key}.pkl"

    def _is_expired(self, cached_at: datetime) -> bool:
        """
        Check if a cached item has expired.

        Args:
            cached_at: When the item was cached

        Returns:
            True if expired, False otherwise
        """
        expiry = cached_at + timedelta(hours=self.ttl_hours)
        return datetime.now(timezone.utc) > expiry

    def get(self, source: str, endpoint: str, params: Dict[str, Any]) -> Optional[Any]:
        """
        Retrieve a cached response.

        Args:
            source: API source
            endpoint: API endpoint
            params: Request parameters

        Returns:
            Cached response or None if not found/expired
        """
        cache_key = self._generate_cache_key(source, endpoint, params)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            logger.debug(f"Cache miss: {source}/{endpoint}")
            return None

        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)

            # Check if expired
            if self._is_expired(cached_data['cached_at']):
                logger.debug(f"Cache expired: {source}/{endpoint}")
                cache_path.unlink()  # Delete expired cache
                return None

            logger.debug(f"Cache hit: {source}/{endpoint}")
            return cached_data['response']

        except Exception as e:
            logger.warning(f"Error reading cache: {e}")
            # Delete corrupted cache file
            if cache_path.exists():
                cache_path.unlink()
            return None

    def set(self, source: str, endpoint: str, params: Dict[str, Any], response: Any):
        """
        Store a response in the cache.

        Args:
            source: API source
            endpoint: API endpoint
            params: Request parameters
            response: Response to cache
        """
        cache_key = self._generate_cache_key(source, endpoint, params)
        cache_path = self._get_cache_path(cache_key)

        try:
            cached_data = {
                'source': source,
                'endpoint': endpoint,
                'params': params,
                'response': response,
                'cached_at': datetime.now(timezone.utc)
            }

            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)

            logger.debug(f"Cached: {source}/{endpoint}")

            # Check cache size and cleanup if needed
            self._check_cache_size()

        except Exception as e:
            logger.warning(f"Error writing cache: {e}")
            # Don't fail the request if caching fails

    def invalidate(self, source: Optional[str] = None, endpoint: Optional[str] = None):
        """
        Invalidate cache entries.

        Args:
            source: If provided, only invalidate this source
            endpoint: If provided, only invalidate this endpoint (requires source)
        """
        count = 0

        for cache_file in self.cache_dir.rglob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                should_delete = True

                if source and cached_data['source'] != source:
                    should_delete = False

                if endpoint and cached_data.get('endpoint') != endpoint:
                    should_delete = False

                if should_delete:
                    cache_file.unlink()
                    count += 1

            except Exception as e:
                logger.warning(f"Error checking cache file {cache_file}: {e}")
                # Delete corrupted file
                cache_file.unlink()
                count += 1

        logger.info(f"Invalidated {count} cache entries")

    def clear(self):
        """Clear all cache entries."""
        count = 0
        for cache_file in self.cache_dir.rglob("*.pkl"):
            cache_file.unlink()
            count += 1

        logger.info(f"Cleared {count} cache entries")

    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        count = 0

        for cache_file in self.cache_dir.rglob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                if self._is_expired(cached_data['cached_at']):
                    cache_file.unlink()
                    count += 1

            except Exception:
                # Delete corrupted cache files
                cache_file.unlink()
                count += 1

        logger.info(f"Cleaned up {count} expired cache entries")
        return count

    def _check_cache_size(self):
        """Check cache size and cleanup if exceeds limit."""
        total_size_mb = sum(f.stat().st_size for f in self.cache_dir.rglob("*.pkl")) / (1024 * 1024)

        if total_size_mb > self.max_cache_size_mb:
            logger.warning(f"Cache size ({total_size_mb:.1f} MB) exceeds limit ({self.max_cache_size_mb} MB)")

            # Delete oldest files first
            cache_files = sorted(
                self.cache_dir.rglob("*.pkl"),
                key=lambda f: f.stat().st_mtime
            )

            deleted_mb = 0
            target_mb = self.max_cache_size_mb * 0.8  # Clean to 80% of max

            for cache_file in cache_files:
                if total_size_mb - deleted_mb <= target_mb:
                    break

                file_size_mb = cache_file.stat().st_size / (1024 * 1024)
                cache_file.unlink()
                deleted_mb += file_size_mb

            logger.info(f"Cleaned up {deleted_mb:.1f} MB from cache")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        cache_files = list(self.cache_dir.rglob("*.pkl"))
        total_size_mb = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)

        expired_count = 0
        for cache_file in cache_files:
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                if self._is_expired(cached_data['cached_at']):
                    expired_count += 1
            except Exception:
                expired_count += 1

        return {
            "total_entries": len(cache_files),
            "size_mb": round(total_size_mb, 2),
            "expired_entries": expired_count,
            "ttl_hours": self.ttl_hours,
            "cache_dir": str(self.cache_dir)
        }


# Singleton cache instance
_cache: Optional[LiteratureCache] = None


def get_cache(
    cache_dir: Optional[str] = None,
    ttl_hours: int = 48,
    max_cache_size_mb: int = 1000
) -> LiteratureCache:
    """
    Get or create the singleton cache instance.

    Args:
        cache_dir: Directory to store cache files
        ttl_hours: Time-to-live for cached responses in hours
        max_cache_size_mb: Maximum cache directory size in MB

    Returns:
        LiteratureCache instance
    """
    global _cache
    if _cache is None:
        _cache = LiteratureCache(
            cache_dir=cache_dir,
            ttl_hours=ttl_hours,
            max_cache_size_mb=max_cache_size_mb
        )
    return _cache


def reset_cache():
    """Reset the singleton cache (useful for testing)."""
    global _cache
    _cache = None

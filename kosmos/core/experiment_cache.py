"""
Experiment result caching with similarity-based reuse detection.

Provides intelligent caching for experiment results with the ability to
detect similar experiments and reuse results when appropriate. Uses SQLite
for persistent storage and embeddings for similarity matching.
"""

import hashlib
import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kosmos.core.cache_manager import CacheType, get_cache_manager

logger = logging.getLogger(__name__)


class ExperimentCacheEntry:
    """Represents a cached experiment result."""

    def __init__(
        self,
        experiment_id: str,
        hypothesis: str,
        parameters: Dict[str, Any],
        results: Dict[str, Any],
        execution_time: float,
        timestamp: datetime,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ):
        """
        Initialize experiment cache entry.

        Args:
            experiment_id: Unique identifier for the experiment
            hypothesis: The hypothesis being tested
            parameters: Experiment parameters/configuration
            results: Experiment results
            execution_time: Time taken to run experiment (seconds)
            timestamp: When the experiment was run
            metadata: Additional metadata (agent info, metrics, etc.)
            embedding: Optional embedding vector for similarity matching
        """
        self.experiment_id = experiment_id
        self.hypothesis = hypothesis
        self.parameters = parameters
        self.results = results
        self.execution_time = execution_time
        self.timestamp = timestamp
        self.metadata = metadata or {}
        self.embedding = embedding

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'experiment_id': self.experiment_id,
            'hypothesis': self.hypothesis,
            'parameters': self.parameters,
            'results': self.results,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'embedding': self.embedding,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentCacheEntry':
        """Create from dictionary."""
        return cls(
            experiment_id=data['experiment_id'],
            hypothesis=data['hypothesis'],
            parameters=data['parameters'],
            results=data['results'],
            execution_time=data['execution_time'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {}),
            embedding=data.get('embedding'),
        )


class ExperimentNormalizer:
    """
    Normalize experiment parameters for better matching.

    Handles parameter normalization, template extraction, and
    experiment fingerprinting for cache key generation.
    """

    @staticmethod
    def normalize_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize experiment parameters.

        Args:
            params: Original parameters

        Returns:
            Normalized parameters with sorted keys and standardized values
        """
        normalized = {}

        for key, value in sorted(params.items()):
            # Normalize numeric values to standard precision
            if isinstance(value, float):
                normalized[key] = round(value, 6)
            elif isinstance(value, dict):
                normalized[key] = ExperimentNormalizer.normalize_parameters(value)
            elif isinstance(value, list):
                # Sort lists for consistent ordering (if items are comparable)
                try:
                    normalized[key] = sorted(value)
                except TypeError:
                    normalized[key] = value
            else:
                normalized[key] = value

        return normalized

    @staticmethod
    def generate_fingerprint(
        hypothesis: str,
        parameters: Dict[str, Any]
    ) -> str:
        """
        Generate unique fingerprint for experiment.

        Args:
            hypothesis: Experiment hypothesis
            parameters: Experiment parameters

        Returns:
            SHA256 fingerprint
        """
        # Normalize parameters first
        norm_params = ExperimentNormalizer.normalize_parameters(parameters)

        # Create fingerprint from hypothesis + normalized params
        fingerprint_data = {
            'hypothesis': hypothesis.strip().lower(),
            'parameters': norm_params
        }

        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True, default=str)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()

    @staticmethod
    def extract_searchable_text(
        hypothesis: str,
        parameters: Dict[str, Any]
    ) -> str:
        """
        Extract searchable text for embedding generation.

        Args:
            hypothesis: Experiment hypothesis
            parameters: Experiment parameters

        Returns:
            Combined text for embedding
        """
        # Combine hypothesis with key parameters
        text_parts = [hypothesis]

        # Add important parameters as text
        for key, value in sorted(parameters.items()):
            if isinstance(value, (str, int, float, bool)):
                text_parts.append(f"{key}: {value}")

        return " | ".join(text_parts)


class ExperimentCache:
    """
    Intelligent cache for experiment results with similarity detection.

    Features:
    - SQLite-based persistent storage
    - Embedding-based similarity matching
    - Automatic experiment reuse detection
    - Configurable similarity thresholds
    - Metadata tracking and statistics
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        similarity_threshold: float = 0.90,
        enable_similarity: bool = True,
        max_similar_results: int = 5,
    ):
        """
        Initialize experiment cache.

        Args:
            cache_dir: Directory for SQLite database (default: .kosmos_cache/experiments)
            similarity_threshold: Minimum cosine similarity for reuse (0-1)
            enable_similarity: Enable similarity-based matching
            max_similar_results: Maximum similar experiments to return
        """
        self.similarity_threshold = similarity_threshold
        self.enable_similarity = enable_similarity
        self.max_similar_results = max_similar_results

        # Set up cache directory
        if cache_dir is None:
            cache_dir = str(Path(".kosmos_cache") / "experiments")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # SQLite database path
        self.db_path = self.cache_dir / "experiments.db"

        # Thread lock for database operations
        self._lock = threading.RLock()

        # Normalizer
        self.normalizer = ExperimentNormalizer()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.similar_hits = 0

        # Initialize database
        self._init_database()

        logger.info(
            f"ExperimentCache initialized: "
            f"db={self.db_path}, "
            f"similarity={enable_similarity}, "
            f"threshold={similarity_threshold}"
        )

    def _init_database(self):
        """Initialize SQLite database schema."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Create experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    hypothesis TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    results TEXT NOT NULL,
                    execution_time REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    embedding TEXT,
                    fingerprint TEXT NOT NULL,
                    searchable_text TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_fingerprint
                ON experiments(fingerprint)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON experiments(timestamp DESC)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_hypothesis
                ON experiments(hypothesis)
            """)

            # Create statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_stats (
                    stat_key TEXT PRIMARY KEY,
                    stat_value INTEGER DEFAULT 0,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Initialize stats
            cursor.execute("""
                INSERT OR IGNORE INTO cache_stats (stat_key, stat_value)
                VALUES ('total_experiments', 0),
                       ('cache_hits', 0),
                       ('cache_misses', 0),
                       ('similar_hits', 0)
            """)

            conn.commit()
            conn.close()

            logger.debug(f"Database initialized: {self.db_path}")

    def cache_result(
        self,
        hypothesis: str,
        parameters: Dict[str, Any],
        results: Dict[str, Any],
        execution_time: float,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> str:
        """
        Cache an experiment result.

        Args:
            hypothesis: The hypothesis tested
            parameters: Experiment parameters
            results: Experiment results
            execution_time: Time taken to run (seconds)
            metadata: Additional metadata
            embedding: Optional embedding vector for similarity

        Returns:
            Experiment ID
        """
        with self._lock:
            # Generate fingerprint and ID
            fingerprint = self.normalizer.generate_fingerprint(hypothesis, parameters)
            experiment_id = f"exp_{fingerprint[:16]}"

            # Extract searchable text
            searchable_text = self.normalizer.extract_searchable_text(
                hypothesis, parameters
            )

            # Create entry
            entry = ExperimentCacheEntry(
                experiment_id=experiment_id,
                hypothesis=hypothesis,
                parameters=parameters,
                results=results,
                execution_time=execution_time,
                timestamp=datetime.now(),
                metadata=metadata,
                embedding=embedding,
            )

            # Store in database
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()

                # Serialize complex fields
                params_json = json.dumps(parameters, default=str)
                results_json = json.dumps(results, default=str)
                metadata_json = json.dumps(metadata or {}, default=str)
                embedding_json = json.dumps(embedding) if embedding else None

                # Insert or replace
                cursor.execute("""
                    INSERT OR REPLACE INTO experiments
                    (experiment_id, hypothesis, parameters, results, execution_time,
                     timestamp, metadata, embedding, fingerprint, searchable_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment_id,
                    hypothesis,
                    params_json,
                    results_json,
                    execution_time,
                    entry.timestamp.isoformat(),
                    metadata_json,
                    embedding_json,
                    fingerprint,
                    searchable_text,
                ))

                # Update stats
                cursor.execute("""
                    UPDATE cache_stats
                    SET stat_value = stat_value + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE stat_key = 'total_experiments'
                """)

                conn.commit()
                conn.close()

                logger.info(f"Cached experiment: {experiment_id}")
                return experiment_id

            except Exception as e:
                logger.error(f"Failed to cache experiment: {e}")
                raise

    def get_cached_result(
        self,
        hypothesis: str,
        parameters: Dict[str, Any]
    ) -> Optional[ExperimentCacheEntry]:
        """
        Get cached experiment result by exact match.

        Args:
            hypothesis: The hypothesis
            parameters: Experiment parameters

        Returns:
            Cached entry if found, None otherwise
        """
        with self._lock:
            # Generate fingerprint
            fingerprint = self.normalizer.generate_fingerprint(hypothesis, parameters)

            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()

                # Query by fingerprint
                cursor.execute("""
                    SELECT experiment_id, hypothesis, parameters, results,
                           execution_time, timestamp, metadata, embedding
                    FROM experiments
                    WHERE fingerprint = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (fingerprint,))

                row = cursor.fetchone()
                conn.close()

                if row:
                    # Parse entry
                    entry = self._row_to_entry(row)
                    self.hits += 1
                    self._increment_stat('cache_hits')
                    logger.info(f"Cache hit: {entry.experiment_id}")
                    return entry
                else:
                    self.misses += 1
                    self._increment_stat('cache_misses')
                    logger.debug("Cache miss: no exact match")
                    return None

            except Exception as e:
                logger.error(f"Failed to get cached result: {e}")
                return None

    def find_similar(
        self,
        hypothesis: str,
        parameters: Dict[str, Any],
        embedding: Optional[List[float]] = None,
    ) -> List[Tuple[ExperimentCacheEntry, float]]:
        """
        Find similar cached experiments.

        Args:
            hypothesis: The hypothesis
            parameters: Experiment parameters
            embedding: Optional embedding for similarity matching

        Returns:
            List of (entry, similarity_score) tuples, sorted by similarity
        """
        if not self.enable_similarity:
            return []

        with self._lock:
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()

                # Get all experiments with embeddings
                cursor.execute("""
                    SELECT experiment_id, hypothesis, parameters, results,
                           execution_time, timestamp, metadata, embedding
                    FROM experiments
                    WHERE embedding IS NOT NULL
                    ORDER BY timestamp DESC
                """)

                rows = cursor.fetchall()
                conn.close()

                if not rows or embedding is None:
                    return []

                # Calculate similarities
                similar_experiments = []

                for row in rows:
                    entry = self._row_to_entry(row)

                    if entry.embedding:
                        # Cosine similarity
                        similarity = self._cosine_similarity(
                            embedding,
                            entry.embedding
                        )

                        if similarity >= self.similarity_threshold:
                            similar_experiments.append((entry, similarity))

                # Sort by similarity (highest first)
                similar_experiments.sort(key=lambda x: x[1], reverse=True)

                # Limit results
                similar_experiments = similar_experiments[:self.max_similar_results]

                if similar_experiments:
                    self.similar_hits += 1
                    self._increment_stat('similar_hits')
                    logger.info(
                        f"Found {len(similar_experiments)} similar experiments "
                        f"(best match: {similar_experiments[0][1]:.3f})"
                    )

                return similar_experiments

            except Exception as e:
                logger.error(f"Failed to find similar experiments: {e}")
                return []

    def _row_to_entry(self, row: Tuple) -> ExperimentCacheEntry:
        """Convert database row to ExperimentCacheEntry."""
        experiment_id, hypothesis, params_json, results_json, \
            execution_time, timestamp, metadata_json, embedding_json = row

        # Parse JSON fields
        parameters = json.loads(params_json)
        results = json.loads(results_json)
        metadata = json.loads(metadata_json) if metadata_json else {}
        embedding = json.loads(embedding_json) if embedding_json else None

        return ExperimentCacheEntry(
            experiment_id=experiment_id,
            hypothesis=hypothesis,
            parameters=parameters,
            results=results,
            execution_time=execution_time,
            timestamp=datetime.fromisoformat(timestamp),
            metadata=metadata,
            embedding=embedding,
        )

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (0-1)
        """
        if len(vec1) != len(vec2):
            return 0.0

        # Dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Magnitudes
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    def _increment_stat(self, stat_key: str):
        """Increment a statistic in the database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE cache_stats
                SET stat_value = stat_value + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE stat_key = ?
            """, (stat_key,))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to increment stat {stat_key}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get experiment cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()

                # Get stats from database
                cursor.execute("""
                    SELECT stat_key, stat_value
                    FROM cache_stats
                """)

                db_stats = {row[0]: row[1] for row in cursor.fetchall()}

                # Get total size
                cursor.execute("SELECT COUNT(*) FROM experiments")
                total_count = cursor.fetchone()[0]

                conn.close()

                # Calculate hit rate
                total_requests = (
                    db_stats.get('cache_hits', 0) +
                    db_stats.get('cache_misses', 0)
                )
                hit_rate = (
                    (db_stats.get('cache_hits', 0) / total_requests * 100)
                    if total_requests > 0
                    else 0.0
                )

                return {
                    'total_experiments': total_count,
                    'cache_hits': db_stats.get('cache_hits', 0),
                    'cache_misses': db_stats.get('cache_misses', 0),
                    'similar_hits': db_stats.get('similar_hits', 0),
                    'hit_rate_percent': round(hit_rate, 2),
                    'similarity_enabled': self.enable_similarity,
                    'similarity_threshold': self.similarity_threshold,
                    'database_path': str(self.db_path),
                    'database_size_mb': round(
                        self.db_path.stat().st_size / (1024 * 1024), 2
                    ) if self.db_path.exists() else 0,
                }

            except Exception as e:
                logger.error(f"Failed to get stats: {e}")
                return {
                    'error': str(e),
                    'total_experiments': 0,
                }

    def clear(self) -> int:
        """
        Clear all cached experiments.

        Returns:
            Number of experiments cleared
        """
        with self._lock:
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()

                # Get count before clearing
                cursor.execute("SELECT COUNT(*) FROM experiments")
                count = cursor.fetchone()[0]

                # Clear experiments
                cursor.execute("DELETE FROM experiments")

                # Reset stats
                cursor.execute("""
                    UPDATE cache_stats
                    SET stat_value = 0,
                        updated_at = CURRENT_TIMESTAMP
                """)

                conn.commit()
                conn.close()

                logger.info(f"Cleared {count} experiments from cache")
                return count

            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
                return 0

    def get_recent_experiments(
        self,
        limit: int = 10
    ) -> List[ExperimentCacheEntry]:
        """
        Get most recent experiments.

        Args:
            limit: Maximum number of experiments to return

        Returns:
            List of recent experiment entries
        """
        with self._lock:
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT experiment_id, hypothesis, parameters, results,
                           execution_time, timestamp, metadata, embedding
                    FROM experiments
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))

                rows = cursor.fetchall()
                conn.close()

                return [self._row_to_entry(row) for row in rows]

            except Exception as e:
                logger.error(f"Failed to get recent experiments: {e}")
                return []


# Global experiment cache instance
_experiment_cache: Optional[ExperimentCache] = None


def get_experiment_cache(
    similarity_threshold: float = 0.90,
    enable_similarity: bool = True
) -> ExperimentCache:
    """
    Get or create the global experiment cache instance.

    Args:
        similarity_threshold: Minimum similarity for reuse (0-1)
        enable_similarity: Enable similarity-based matching

    Returns:
        ExperimentCache instance
    """
    global _experiment_cache

    if _experiment_cache is None:
        _experiment_cache = ExperimentCache(
            similarity_threshold=similarity_threshold,
            enable_similarity=enable_similarity
        )

    return _experiment_cache


def reset_experiment_cache():
    """Reset the global experiment cache (useful for testing)."""
    global _experiment_cache
    _experiment_cache = None

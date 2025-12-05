"""
Prometheus metrics for Kosmos AI Scientist.

Provides comprehensive metrics collection for monitoring and observability.
"""

import logging
import time
from typing import Dict, Any, Optional
from contextlib import contextmanager

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus_client not installed, metrics disabled")

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Centralized metrics collector for Kosmos using Prometheus.

    Tracks:
    - Research cycles and iterations
    - Hypothesis generation and evaluation
    - Experiment execution
    - API calls and costs
    - Cache performance
    - System resources
    """

    def __init__(self, registry: Optional[Any] = None):
        """
        Initialize metrics collector.

        Args:
            registry: Prometheus registry (optional, creates new if None)
        """
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus metrics not available")
            self.enabled = False
            return

        self.enabled = True
        self.registry = registry or CollectorRegistry()

        # Research metrics
        self.research_cycles_total = Counter(
            'kosmos_research_cycles_total',
            'Total number of research cycles started',
            ['domain', 'status'],
            registry=self.registry
        )

        self.research_iterations_total = Counter(
            'kosmos_research_iterations_total',
            'Total number of research iterations',
            ['domain'],
            registry=self.registry
        )

        self.research_duration_seconds = Histogram(
            'kosmos_research_duration_seconds',
            'Research cycle duration in seconds',
            ['domain', 'status'],
            buckets=[60, 300, 600, 1800, 3600, 7200, 14400],  # 1min to 4hrs
            registry=self.registry
        )

        # Hypothesis metrics
        self.hypotheses_generated_total = Counter(
            'kosmos_hypotheses_generated_total',
            'Total number of hypotheses generated',
            ['domain', 'strategy'],
            registry=self.registry
        )

        self.hypotheses_tested_total = Counter(
            'kosmos_hypotheses_tested_total',
            'Total number of hypotheses tested',
            ['domain', 'outcome'],
            registry=self.registry
        )

        self.hypothesis_evaluation_duration_seconds = Histogram(
            'kosmos_hypothesis_evaluation_duration_seconds',
            'Hypothesis evaluation duration',
            buckets=[0.5, 1, 2, 5, 10, 30, 60],
            registry=self.registry
        )

        # Experiment metrics
        self.experiments_total = Counter(
            'kosmos_experiments_total',
            'Total number of experiments executed',
            ['domain', 'experiment_type', 'status'],
            registry=self.registry
        )

        self.experiments_active = Gauge(
            'kosmos_experiments_active',
            'Number of currently active experiments',
            registry=self.registry
        )

        self.experiment_duration_seconds = Histogram(
            'kosmos_experiment_duration_seconds',
            'Experiment execution duration',
            ['experiment_type'],
            buckets=[1, 5, 10, 30, 60, 300, 600, 1800],  # 1s to 30min
            registry=self.registry
        )

        # API call metrics
        self.api_calls_total = Counter(
            'kosmos_api_calls_total',
            'Total number of API calls',
            ['api', 'model', 'status'],
            registry=self.registry
        )

        self.api_tokens_total = Counter(
            'kosmos_api_tokens_total',
            'Total tokens used in API calls',
            ['api', 'model', 'token_type'],
            registry=self.registry
        )

        self.api_cost_usd_total = Counter(
            'kosmos_api_cost_usd_total',
            'Total API cost in USD',
            ['api', 'model'],
            registry=self.registry
        )

        self.api_latency_seconds = Histogram(
            'kosmos_api_latency_seconds',
            'API call latency',
            ['api', 'model'],
            buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
            registry=self.registry
        )

        # Cache metrics
        self.cache_operations_total = Counter(
            'kosmos_cache_operations_total',
            'Total cache operations',
            ['operation', 'cache_type', 'status'],
            registry=self.registry
        )

        self.cache_hit_ratio = Gauge(
            'kosmos_cache_hit_ratio',
            'Cache hit ratio (0.0-1.0)',
            ['cache_type'],
            registry=self.registry
        )

        self.cache_size_bytes = Gauge(
            'kosmos_cache_size_bytes',
            'Current cache size in bytes',
            ['cache_type'],
            registry=self.registry
        )

        self.cache_entries = Gauge(
            'kosmos_cache_entries',
            'Number of entries in cache',
            ['cache_type'],
            registry=self.registry
        )

        # Database metrics
        self.database_queries_total = Counter(
            'kosmos_database_queries_total',
            'Total database queries',
            ['operation', 'table', 'status'],
            registry=self.registry
        )

        self.database_query_duration_seconds = Histogram(
            'kosmos_database_query_duration_seconds',
            'Database query duration',
            ['operation', 'table'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
            registry=self.registry
        )

        self.database_connections_active = Gauge(
            'kosmos_database_connections_active',
            'Number of active database connections',
            registry=self.registry
        )

        # System resource metrics
        self.cpu_usage_percent = Gauge(
            'kosmos_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )

        self.memory_usage_bytes = Gauge(
            'kosmos_memory_usage_bytes',
            'Memory usage in bytes',
            ['type'],  # rss, vms
            registry=self.registry
        )

        self.disk_usage_bytes = Gauge(
            'kosmos_disk_usage_bytes',
            'Disk usage in bytes',
            ['mount_point', 'type'],  # type: total, used, free
            registry=self.registry
        )

        # Result quality metrics
        self.result_quality_score = Summary(
            'kosmos_result_quality_score',
            'Result quality score distribution',
            ['domain'],
            registry=self.registry
        )

        self.convergence_iterations = Histogram(
            'kosmos_convergence_iterations',
            'Number of iterations to convergence',
            ['domain'],
            buckets=[1, 3, 5, 10, 15, 20, 30, 50],
            registry=self.registry
        )

        # Info metric for version
        self.info = Info(
            'kosmos_info',
            'Kosmos service information',
            registry=self.registry
        )
        self._set_info()

        logger.info("Prometheus metrics collector initialized")

    def _set_info(self):
        """Set service info metric."""
        try:
            from kosmos import __version__
            import platform

            self.info.info({
                'version': __version__,
                'python_version': platform.python_version(),
                'platform': platform.system(),
                'architecture': platform.machine()
            })
        except Exception as e:
            logger.debug("Failed to collect system info for metrics: %s", e)

    # Research tracking
    def track_research_cycle(self, domain: str, status: str, duration: Optional[float] = None):
        """Track research cycle completion."""
        if not self.enabled:
            return

        self.research_cycles_total.labels(domain=domain, status=status).inc()
        if duration:
            self.research_duration_seconds.labels(domain=domain, status=status).observe(duration)

    def track_research_iteration(self, domain: str):
        """Track research iteration."""
        if not self.enabled:
            return

        self.research_iterations_total.labels(domain=domain).inc()

    # Hypothesis tracking
    def track_hypothesis_generated(self, domain: str, strategy: str):
        """Track hypothesis generation."""
        if not self.enabled:
            return

        self.hypotheses_generated_total.labels(domain=domain, strategy=strategy).inc()

    def track_hypothesis_tested(self, domain: str, outcome: str):
        """Track hypothesis test completion."""
        if not self.enabled:
            return

        self.hypotheses_tested_total.labels(domain=domain, outcome=outcome).inc()

    @contextmanager
    def track_hypothesis_evaluation(self):
        """Context manager to track hypothesis evaluation duration."""
        if not self.enabled:
            yield
            return

        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.hypothesis_evaluation_duration_seconds.observe(duration)

    # Experiment tracking
    def track_experiment_start(self, domain: str, experiment_type: str):
        """Track experiment start."""
        if not self.enabled:
            return

        self.experiments_active.inc()

    def track_experiment_complete(self, domain: str, experiment_type: str, status: str, duration: float):
        """Track experiment completion."""
        if not self.enabled:
            return

        self.experiments_total.labels(
            domain=domain,
            experiment_type=experiment_type,
            status=status
        ).inc()
        self.experiments_active.dec()
        self.experiment_duration_seconds.labels(experiment_type=experiment_type).observe(duration)

    # API tracking
    def track_api_call(self, api: str, model: str, status: str, latency: float,
                      input_tokens: int = 0, output_tokens: int = 0, cost_usd: float = 0.0):
        """Track API call metrics."""
        if not self.enabled:
            return

        self.api_calls_total.labels(api=api, model=model, status=status).inc()
        self.api_latency_seconds.labels(api=api, model=model).observe(latency)

        if input_tokens > 0:
            self.api_tokens_total.labels(api=api, model=model, token_type='input').inc(input_tokens)
        if output_tokens > 0:
            self.api_tokens_total.labels(api=api, model=model, token_type='output').inc(output_tokens)
        if cost_usd > 0:
            self.api_cost_usd_total.labels(api=api, model=model).inc(cost_usd)

    # Cache tracking
    def track_cache_operation(self, operation: str, cache_type: str, status: str):
        """Track cache operation."""
        if not self.enabled:
            return

        self.cache_operations_total.labels(
            operation=operation,
            cache_type=cache_type,
            status=status
        ).inc()

    def update_cache_stats(self, cache_type: str, hit_ratio: float, size_bytes: int, entries: int):
        """Update cache statistics."""
        if not self.enabled:
            return

        self.cache_hit_ratio.labels(cache_type=cache_type).set(hit_ratio)
        self.cache_size_bytes.labels(cache_type=cache_type).set(size_bytes)
        self.cache_entries.labels(cache_type=cache_type).set(entries)

    # Database tracking
    def track_database_query(self, operation: str, table: str, status: str, duration: float):
        """Track database query."""
        if not self.enabled:
            return

        self.database_queries_total.labels(
            operation=operation,
            table=table,
            status=status
        ).inc()
        self.database_query_duration_seconds.labels(
            operation=operation,
            table=table
        ).observe(duration)

    def update_database_connections(self, count: int):
        """Update active database connections count."""
        if not self.enabled:
            return

        self.database_connections_active.set(count)

    # System resource tracking
    def update_system_metrics(self, cpu_percent: float, memory_rss: int, memory_vms: int,
                             disk_total: int, disk_used: int, disk_free: int):
        """Update system resource metrics."""
        if not self.enabled:
            return

        self.cpu_usage_percent.set(cpu_percent)
        self.memory_usage_bytes.labels(type='rss').set(memory_rss)
        self.memory_usage_bytes.labels(type='vms').set(memory_vms)
        self.disk_usage_bytes.labels(mount_point='/', type='total').set(disk_total)
        self.disk_usage_bytes.labels(mount_point='/', type='used').set(disk_used)
        self.disk_usage_bytes.labels(mount_point='/', type='free').set(disk_free)

    # Quality tracking
    def track_result_quality(self, domain: str, score: float):
        """Track result quality score."""
        if not self.enabled:
            return

        self.result_quality_score.labels(domain=domain).observe(score)

    def track_convergence(self, domain: str, iterations: int):
        """Track convergence iterations."""
        if not self.enabled:
            return

        self.convergence_iterations.labels(domain=domain).observe(iterations)

    def export_metrics(self) -> bytes:
        """
        Export metrics in Prometheus format.

        Returns:
            Metrics in Prometheus text format
        """
        if not self.enabled:
            return b""

        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """Get content type for Prometheus metrics."""
        return CONTENT_TYPE_LATEST


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """
    Get or create the global metrics collector.

    Returns:
        MetricsCollector instance
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def export_metrics() -> bytes:
    """Export metrics in Prometheus format."""
    return get_metrics_collector().export_metrics()


def get_metrics_content_type() -> str:
    """Get content type for metrics."""
    return get_metrics_collector().get_content_type()

"""
Metrics collection for monitoring Kosmos performance.

Tracks:
- API calls and costs
- Experiment execution times
- Agent activity
- System health
- Cache performance and cost savings
- Budget limits and alerts
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import logging
from enum import Enum


logger = logging.getLogger(__name__)


class BudgetPeriod(Enum):
    """Budget tracking periods."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class BudgetAlert:
    """Represents a budget alert."""

    def __init__(
        self,
        threshold_percent: float,
        message: str,
        triggered_at: Optional[datetime] = None
    ):
        """
        Initialize budget alert.

        Args:
            threshold_percent: Threshold percentage (0-100)
            message: Alert message
            triggered_at: When alert was triggered
        """
        self.threshold_percent = threshold_percent
        self.message = message
        self.triggered_at = triggered_at or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'threshold_percent': self.threshold_percent,
            'message': self.message,
            'triggered_at': self.triggered_at.isoformat(),
        }


class MetricsCollector:
    """
    Collect and aggregate metrics.

    Thread-safe metrics collection with aggregation and export capabilities.

    Example:
        ```python
        from kosmos.core.metrics import get_metrics

        metrics = get_metrics()

        # Record API call
        metrics.record_api_call(
            model="claude-3-5-sonnet",
            input_tokens=100,
            output_tokens=50,
            duration_seconds=1.2
        )

        # Record experiment
        metrics.record_experiment(
            experiment_type="data_analysis",
            duration_seconds=30.5,
            status="success"
        )

        # Get statistics
        stats = metrics.get_statistics()
        print(stats["total_api_calls"])
        ```
    """

    def __init__(self):
        """Initialize metrics collector."""
        self._lock = threading.Lock()
        self.start_time = datetime.utcnow()

        # API metrics
        self.api_calls = 0
        self.api_errors = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_api_duration = 0.0
        self.api_call_history: List[Dict[str, Any]] = []

        # Experiment metrics
        self.experiments_started = 0
        self.experiments_completed = 0
        self.experiments_failed = 0
        self.total_experiment_duration = 0.0
        self.experiments_by_type = defaultdict(int)
        self.experiment_history: List[Dict[str, Any]] = []

        # Agent metrics
        self.agents_created = 0
        self.messages_sent = 0
        self.messages_received = 0
        self.agent_errors = 0

        # Cache metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_sets = 0
        self.cache_evictions = 0
        self.cache_errors = 0
        self.cache_hit_history: List[Dict[str, Any]] = []

        # Budget tracking
        self.budget_enabled = False
        self.budget_limit_usd: Optional[float] = None
        self.budget_limit_requests: Optional[int] = None
        self.budget_period = BudgetPeriod.DAILY
        self.budget_period_start = datetime.utcnow()
        self.budget_alert_thresholds = [50.0, 75.0, 90.0, 100.0]  # Percentage thresholds
        self.budget_alerts: List[BudgetAlert] = []
        self.alert_callbacks: List[Callable[[BudgetAlert], None]] = []

        # System metrics
        self.errors_encountered = 0
        self.warnings_logged = 0

        logger.info("Metrics collector initialized")

    # ========================================================================
    # API METRICS
    # ========================================================================

    def record_api_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration_seconds: float,
        success: bool = True
    ):
        """
        Record Claude API call.

        Args:
            model: Model used
            input_tokens: Input tokens
            output_tokens: Output tokens
            duration_seconds: Call duration
            success: Whether call succeeded
        """
        with self._lock:
            self.api_calls += 1
            if not success:
                self.api_errors += 1

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_api_duration += duration_seconds

            # Store in history
            self.api_call_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "duration_seconds": duration_seconds,
                "success": success
            })

            # Keep history limited to last 1000 calls
            if len(self.api_call_history) > 1000:
                self.api_call_history.pop(0)

    def get_api_statistics(self) -> Dict[str, Any]:
        """
        Get API call statistics.

        Returns:
            dict: API statistics
        """
        with self._lock:
            avg_duration = (self.total_api_duration / self.api_calls
                          if self.api_calls > 0 else 0)
            error_rate = (self.api_errors / self.api_calls
                        if self.api_calls > 0 else 0)

            # Estimate cost (Claude 3.5 Sonnet pricing)
            input_cost = (self.total_input_tokens / 1_000_000) * 3.0
            output_cost = (self.total_output_tokens / 1_000_000) * 15.0
            total_cost = input_cost + output_cost

            return {
                "total_calls": self.api_calls,
                "successful_calls": self.api_calls - self.api_errors,
                "failed_calls": self.api_errors,
                "error_rate": error_rate,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
                "total_duration_seconds": self.total_api_duration,
                "average_duration_seconds": avg_duration,
                "estimated_cost_usd": total_cost,
            }

    # ========================================================================
    # EXPERIMENT METRICS
    # ========================================================================

    def record_experiment_start(self, experiment_id: str, experiment_type: str):
        """
        Record experiment start.

        Args:
            experiment_id: Unique experiment ID
            experiment_type: Type of experiment
        """
        with self._lock:
            self.experiments_started += 1
            self.experiments_by_type[experiment_type] += 1

            self.experiment_history.append({
                "experiment_id": experiment_id,
                "experiment_type": experiment_type,
                "status": "running",
                "start_time": datetime.utcnow().isoformat(),
                "end_time": None,
                "duration_seconds": None
            })

    def record_experiment_end(
        self,
        experiment_id: str,
        duration_seconds: float,
        status: str = "success"
    ):
        """
        Record experiment completion.

        Args:
            experiment_id: Experiment ID
            duration_seconds: Total duration
            status: Final status (success or failure)
        """
        with self._lock:
            if status == "success":
                self.experiments_completed += 1
            else:
                self.experiments_failed += 1

            self.total_experiment_duration += duration_seconds

            # Update history
            for exp in reversed(self.experiment_history):
                if exp["experiment_id"] == experiment_id:
                    exp["status"] = status
                    exp["end_time"] = datetime.utcnow().isoformat()
                    exp["duration_seconds"] = duration_seconds
                    break

            # Keep history limited
            if len(self.experiment_history) > 1000:
                self.experiment_history.pop(0)

    def get_experiment_statistics(self) -> Dict[str, Any]:
        """
        Get experiment statistics.

        Returns:
            dict: Experiment statistics
        """
        with self._lock:
            avg_duration = (self.total_experiment_duration / self.experiments_completed
                          if self.experiments_completed > 0 else 0)
            success_rate = (self.experiments_completed / self.experiments_started
                          if self.experiments_started > 0 else 0)

            return {
                "experiments_started": self.experiments_started,
                "experiments_completed": self.experiments_completed,
                "experiments_failed": self.experiments_failed,
                "experiments_running": self.experiments_started - self.experiments_completed - self.experiments_failed,
                "success_rate": success_rate,
                "total_duration_seconds": self.total_experiment_duration,
                "average_duration_seconds": avg_duration,
                "experiments_by_type": dict(self.experiments_by_type),
            }

    # ========================================================================
    # AGENT METRICS
    # ========================================================================

    def record_agent_created(self):
        """Record agent creation."""
        with self._lock:
            self.agents_created += 1

    def record_message_sent(self):
        """Record inter-agent message sent."""
        with self._lock:
            self.messages_sent += 1

    def record_message_received(self):
        """Record inter-agent message received."""
        with self._lock:
            self.messages_received += 1

    def record_agent_error(self):
        """Record agent error."""
        with self._lock:
            self.agent_errors += 1

    def get_agent_statistics(self) -> Dict[str, Any]:
        """
        Get agent statistics.

        Returns:
            dict: Agent statistics
        """
        with self._lock:
            return {
                "agents_created": self.agents_created,
                "messages_sent": self.messages_sent,
                "messages_received": self.messages_received,
                "agent_errors": self.agent_errors,
            }

    # ========================================================================
    # CACHE METRICS
    # ========================================================================

    def record_cache_hit(self, cache_type: str = "general"):
        """
        Record cache hit.

        Args:
            cache_type: Type of cache (claude, experiment, embedding, etc.)
        """
        with self._lock:
            self.cache_hits += 1

            # Store in history
            self.cache_hit_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "cache_type": cache_type,
                "event_type": "hit"
            })

            # Keep history limited
            if len(self.cache_hit_history) > 1000:
                self.cache_hit_history.pop(0)

    def record_cache_miss(self, cache_type: str = "general"):
        """
        Record cache miss.

        Args:
            cache_type: Type of cache
        """
        with self._lock:
            self.cache_misses += 1

            # Store in history
            self.cache_hit_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "cache_type": cache_type,
                "event_type": "miss"
            })

            # Keep history limited
            if len(self.cache_hit_history) > 1000:
                self.cache_hit_history.pop(0)

    def record_cache_set(self, cache_type: str = "general"):
        """
        Record cache set operation.

        Args:
            cache_type: Type of cache
        """
        with self._lock:
            self.cache_sets += 1

    def record_cache_eviction(self, cache_type: str = "general"):
        """
        Record cache eviction.

        Args:
            cache_type: Type of cache
        """
        with self._lock:
            self.cache_evictions += 1

    def record_cache_error(self, cache_type: str = "general"):
        """
        Record cache error.

        Args:
            cache_type: Type of cache
        """
        with self._lock:
            self.cache_errors += 1

    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics with integration from cache manager.

        Returns:
            dict: Cache statistics
        """
        with self._lock:
            # Calculate hit rate
            total_cache_requests = self.cache_hits + self.cache_misses
            cache_hit_rate = (
                (self.cache_hits / total_cache_requests * 100)
                if total_cache_requests > 0
                else 0.0
            )

            base_stats = {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_sets": self.cache_sets,
                "cache_evictions": self.cache_evictions,
                "cache_errors": self.cache_errors,
                "total_cache_requests": total_cache_requests,
                "cache_hit_rate_percent": round(cache_hit_rate, 2),
            }

            # Try to get detailed cache stats from cache manager
            try:
                from kosmos.core.cache_manager import get_cache_manager

                cache_manager = get_cache_manager()
                detailed_stats = cache_manager.get_stats()

                # Merge detailed stats
                base_stats.update({
                    "cache_manager_stats": detailed_stats,
                    "total_cache_size": detailed_stats.get('total_size', 0),
                    "overall_hit_rate_percent": detailed_stats.get('overall_hit_rate_percent', 0.0),
                })

                # Calculate cost savings
                if 'caches' in detailed_stats:
                    # Get Claude cache stats if available
                    claude_cache_stats = detailed_stats['caches'].get('claude', {})
                    if claude_cache_stats:
                        # Estimate cost savings from cache hits
                        # Assume average request: 1000 input tokens, 500 output tokens
                        avg_input_tokens = 1000
                        avg_output_tokens = 500
                        hits = claude_cache_stats.get('hits', 0)

                        if hits > 0:
                            # Claude 3.5 Sonnet pricing: $3/M input, $15/M output
                            input_saved = (avg_input_tokens * hits / 1_000_000) * 3.0
                            output_saved = (avg_output_tokens * hits / 1_000_000) * 15.0
                            total_saved = input_saved + output_saved

                            base_stats['estimated_cost_saved_usd'] = round(total_saved, 2)
                            base_stats['cache_efficiency'] = 'high' if cache_hit_rate > 30 else 'moderate' if cache_hit_rate > 10 else 'low'

            except ImportError:
                # Cache manager not available
                logger.debug("Cache manager not available for detailed stats")
            except Exception as e:
                logger.error(f"Failed to get cache manager stats: {e}")

            return base_stats

    # ========================================================================
    # SYSTEM METRICS
    # ========================================================================

    def record_error(self):
        """Record system error."""
        with self._lock:
            self.errors_encountered += 1

    def record_warning(self):
        """Record warning."""
        with self._lock:
            self.warnings_logged += 1

    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get system statistics.

        Returns:
            dict: System statistics
        """
        with self._lock:
            uptime = (datetime.utcnow() - self.start_time).total_seconds()

            return {
                "start_time": self.start_time.isoformat(),
                "uptime_seconds": uptime,
                "errors_encountered": self.errors_encountered,
                "warnings_logged": self.warnings_logged,
            }

    # ========================================================================
    # BUDGET MANAGEMENT
    # ========================================================================

    def configure_budget(
        self,
        limit_usd: Optional[float] = None,
        limit_requests: Optional[int] = None,
        period: BudgetPeriod = BudgetPeriod.DAILY,
        alert_thresholds: Optional[List[float]] = None
    ):
        """
        Configure budget limits and alerts.

        Args:
            limit_usd: Budget limit in USD (for API mode)
            limit_requests: Budget limit in requests (for CLI mode)
            period: Budget tracking period
            alert_thresholds: List of percentage thresholds for alerts (0-100)

        Example:
            ```python
            metrics = get_metrics()

            # API mode: $100/day budget
            metrics.configure_budget(
                limit_usd=100.0,
                period=BudgetPeriod.DAILY,
                alert_thresholds=[50, 75, 90, 100]
            )

            # CLI mode: 1000 requests/hour
            metrics.configure_budget(
                limit_requests=1000,
                period=BudgetPeriod.HOURLY
            )
            ```
        """
        with self._lock:
            self.budget_enabled = True
            self.budget_limit_usd = limit_usd
            self.budget_limit_requests = limit_requests
            self.budget_period = period
            self.budget_period_start = datetime.utcnow()

            if alert_thresholds is not None:
                self.budget_alert_thresholds = sorted(alert_thresholds)

            # Clear existing alerts
            self.budget_alerts = []

            logger.info(
                f"Budget configured: "
                f"${limit_usd or 0}/period, "
                f"{limit_requests or 0} requests/period, "
                f"period={period.value}"
            )

    def add_alert_callback(self, callback: Callable[[BudgetAlert], None]):
        """
        Add callback function to be called when budget alerts are triggered.

        Args:
            callback: Function that takes BudgetAlert as argument

        Example:
            ```python
            def on_budget_alert(alert: BudgetAlert):
                print(f"ALERT: {alert.message}")
                send_email(alert.message)

            metrics.add_alert_callback(on_budget_alert)
            ```
        """
        with self._lock:
            self.alert_callbacks.append(callback)

    def check_budget(self) -> Dict[str, Any]:
        """
        Check current budget status and trigger alerts if needed.

        Returns:
            dict: Budget status including usage, limits, and alerts

        This method should be called periodically (e.g., after each API call)
        to check budget status and trigger alerts.
        """
        if not self.budget_enabled:
            return {
                'enabled': False,
                'message': 'Budget tracking not enabled'
            }

        with self._lock:
            # Check if we need to reset for new period
            self._check_period_reset()

            # Calculate current usage
            current_cost = self._calculate_period_cost()
            current_requests = self._calculate_period_requests()

            # Determine which limit to check
            if self.budget_limit_usd is not None:
                # API mode: check cost
                limit = self.budget_limit_usd
                usage = current_cost
                usage_type = 'cost_usd'
                usage_percent = (usage / limit * 100) if limit > 0 else 0
            elif self.budget_limit_requests is not None:
                # CLI mode: check requests
                limit = self.budget_limit_requests
                usage = current_requests
                usage_type = 'requests'
                usage_percent = (usage / limit * 100) if limit > 0 else 0
            else:
                return {
                    'enabled': True,
                    'message': 'No budget limit configured'
                }

            # Check thresholds and trigger alerts
            triggered_alerts = []
            for threshold in self.budget_alert_thresholds:
                if usage_percent >= threshold:
                    # Check if we already triggered this threshold
                    already_triggered = any(
                        alert.threshold_percent == threshold
                        for alert in self.budget_alerts
                    )

                    if not already_triggered:
                        # Trigger new alert
                        if usage_percent >= 100:
                            message = (
                                f"Budget limit exceeded! "
                                f"Usage: {usage:.2f}/{limit:.2f} {usage_type} "
                                f"({usage_percent:.1f}%)"
                            )
                        else:
                            message = (
                                f"Budget alert: {threshold}% threshold reached. "
                                f"Usage: {usage:.2f}/{limit:.2f} {usage_type} "
                                f"({usage_percent:.1f}%)"
                            )

                        alert = BudgetAlert(
                            threshold_percent=threshold,
                            message=message
                        )

                        self.budget_alerts.append(alert)
                        triggered_alerts.append(alert)

                        # Call callbacks
                        for callback in self.alert_callbacks:
                            try:
                                callback(alert)
                            except Exception as e:
                                logger.error(f"Budget alert callback failed: {e}")

                        logger.warning(message)

            return {
                'enabled': True,
                'period': self.budget_period.value,
                'period_start': self.budget_period_start.isoformat(),
                'limit_usd': self.budget_limit_usd,
                'limit_requests': self.budget_limit_requests,
                'current_cost_usd': round(current_cost, 2),
                'current_requests': current_requests,
                'usage_percent': round(usage_percent, 2),
                'alerts_triggered': [alert.to_dict() for alert in triggered_alerts],
                'total_alerts': len(self.budget_alerts),
                'budget_exceeded': usage_percent >= 100,
            }

    def _check_period_reset(self):
        """Check if budget period should be reset."""
        now = datetime.utcnow()
        time_since_start = now - self.budget_period_start

        should_reset = False

        if self.budget_period == BudgetPeriod.HOURLY:
            should_reset = time_since_start.total_seconds() >= 3600
        elif self.budget_period == BudgetPeriod.DAILY:
            should_reset = time_since_start.days >= 1
        elif self.budget_period == BudgetPeriod.WEEKLY:
            should_reset = time_since_start.days >= 7
        elif self.budget_period == BudgetPeriod.MONTHLY:
            should_reset = time_since_start.days >= 30

        if should_reset:
            logger.info(f"Resetting budget for new {self.budget_period.value} period")
            self.budget_period_start = now
            self.budget_alerts = []
            # Note: We don't reset API/request counters, just the budget period

    def _calculate_period_cost(self) -> float:
        """Calculate API cost for current budget period."""
        period_start = self.budget_period_start

        # Filter API calls within period
        period_calls = [
            call for call in self.api_call_history
            if datetime.fromisoformat(call["timestamp"]) >= period_start
        ]

        # Sum tokens
        total_input = sum(call.get("input_tokens", 0) for call in period_calls)
        total_output = sum(call.get("output_tokens", 0) for call in period_calls)

        # Calculate cost (Claude 3.5 Sonnet pricing)
        input_cost = (total_input / 1_000_000) * 3.0
        output_cost = (total_output / 1_000_000) * 15.0

        return input_cost + output_cost

    def _calculate_period_requests(self) -> int:
        """Calculate API requests for current budget period."""
        period_start = self.budget_period_start

        # Count API calls within period
        period_calls = [
            call for call in self.api_call_history
            if datetime.fromisoformat(call["timestamp"]) >= period_start
        ]

        return len(period_calls)

    def get_budget_status(self) -> Dict[str, Any]:
        """
        Get current budget status without triggering alerts.

        Returns:
            dict: Budget status
        """
        return self.check_budget()

    def reset_budget_alerts(self):
        """Clear all triggered budget alerts."""
        with self._lock:
            self.budget_alerts = []
            logger.info("Budget alerts reset")

    # ========================================================================
    # AGGREGATION
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get all statistics.

        Returns:
            dict: Complete statistics
        """
        stats = {
            "system": self.get_system_statistics(),
            "api": self.get_api_statistics(),
            "experiments": self.get_experiment_statistics(),
            "agents": self.get_agent_statistics(),
            "cache": self.get_cache_statistics(),
        }

        # Add budget status if enabled
        if self.budget_enabled:
            stats["budget"] = self.get_budget_status()

        return stats

    def get_recent_activity(self, minutes: int = 60) -> Dict[str, Any]:
        """
        Get recent activity in last N minutes.

        Args:
            minutes: Number of minutes to look back

        Returns:
            dict: Recent activity summary
        """
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)

        with self._lock:
            # Recent API calls
            recent_api_calls = [
                call for call in self.api_call_history
                if datetime.fromisoformat(call["timestamp"]) > cutoff_time
            ]

            # Recent experiments
            recent_experiments = [
                exp for exp in self.experiment_history
                if datetime.fromisoformat(exp["start_time"]) > cutoff_time
            ]

            # Recent cache activity
            recent_cache_events = [
                event for event in self.cache_hit_history
                if datetime.fromisoformat(event["timestamp"]) > cutoff_time
            ]

            recent_cache_hits = sum(
                1 for event in recent_cache_events if event["event_type"] == "hit"
            )
            recent_cache_misses = sum(
                1 for event in recent_cache_events if event["event_type"] == "miss"
            )
            recent_cache_total = recent_cache_hits + recent_cache_misses
            recent_cache_hit_rate = (
                (recent_cache_hits / recent_cache_total * 100)
                if recent_cache_total > 0
                else 0.0
            )

            return {
                "time_window_minutes": minutes,
                "recent_api_calls": len(recent_api_calls),
                "recent_experiments": len(recent_experiments),
                "recent_experiments_completed": sum(
                    1 for exp in recent_experiments if exp["status"] == "success"
                ),
                "recent_experiments_failed": sum(
                    1 for exp in recent_experiments if exp["status"] == "failure"
                ),
                "recent_cache_hits": recent_cache_hits,
                "recent_cache_misses": recent_cache_misses,
                "recent_cache_hit_rate_percent": round(recent_cache_hit_rate, 2),
            }

    def export_metrics(self) -> Dict[str, Any]:
        """
        Export all metrics for external monitoring.

        Returns:
            dict: Complete metrics dump
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "statistics": self.get_statistics(),
            "recent_activity": self.get_recent_activity(),
            "api_call_history": self.api_call_history[-100:],  # Last 100
            "experiment_history": self.experiment_history[-100:],  # Last 100
        }

    def reset(self):
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self.__init__()


# Singleton metrics collector
_metrics: Optional[MetricsCollector] = None


def get_metrics(reset: bool = False) -> MetricsCollector:
    """
    Get or create metrics collector singleton.

    Args:
        reset: If True, create new collector instance

    Returns:
        MetricsCollector: Metrics collector instance
    """
    global _metrics
    if _metrics is None or reset:
        _metrics = MetricsCollector()
    return _metrics

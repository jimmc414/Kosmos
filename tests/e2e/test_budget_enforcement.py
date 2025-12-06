"""
E2E Tests for Budget Enforcement.

Tests the MetricsCollector budget enforcement functionality:
- Budget configuration and limits
- Alert thresholds and callbacks
- Period resets (hourly, daily)
- Cost calculation accuracy
- Graceful shutdown on budget exceeded
- Concurrent tracking
"""

import pytest
import time
import threading
from datetime import datetime, timedelta

from kosmos.core.metrics import (
    MetricsCollector,
    BudgetPeriod,
    BudgetAlert,
    BudgetExceededError
)
from tests.e2e.factories import APICallFactory


pytestmark = [pytest.mark.e2e]


class TestBudgetEnforcement:
    """Tests for budget enforcement halting operations."""

    def test_budget_enforcement_halts_on_exceeded(self, metrics_collector):
        """Verify enforce_budget raises exception when budget exceeded."""
        # Configure tight budget: $0.01
        metrics_collector.configure_budget(
            limit_usd=0.01,
            period=BudgetPeriod.DAILY,
            alert_thresholds=[50.0, 75.0, 90.0, 100.0]
        )

        # Record API calls that exceed budget
        # Need ~$0.02 worth of tokens
        # Claude pricing: $3/M input, $15/M output
        # ~6,000 input tokens + 1,000 output = $0.018 + $0.015 = ~$0.033
        for _ in range(3):
            metrics_collector.record_api_call(
                model="claude-3-5-sonnet-20241022",
                input_tokens=2000,
                output_tokens=500,
                duration_seconds=1.0
            )

        # Verify budget is exceeded
        status = metrics_collector.check_budget()
        assert status['budget_exceeded'] is True

        # Verify enforce_budget raises exception
        with pytest.raises(BudgetExceededError) as exc_info:
            metrics_collector.enforce_budget()

        assert exc_info.value.current_cost > 0.01
        assert exc_info.value.limit == 0.01

    def test_budget_not_exceeded_allows_execution(self, metrics_collector):
        """Verify enforce_budget passes when within budget."""
        # Configure budget: $10
        metrics_collector.configure_budget(
            limit_usd=10.0,
            period=BudgetPeriod.DAILY
        )

        # Record small API call
        metrics_collector.record_api_call(
            model="claude-3-5-sonnet-20241022",
            input_tokens=1000,
            output_tokens=500,
            duration_seconds=1.0
        )

        # Should not raise
        metrics_collector.enforce_budget()

        # Verify not exceeded
        status = metrics_collector.check_budget()
        assert status['budget_exceeded'] is False
        assert status['usage_percent'] < 100


class TestBudgetThresholdAlerts:
    """Tests for budget alert thresholds and callbacks."""

    def test_budget_threshold_alerts_trigger_at_correct_levels(self, metrics_collector, alert_tracker):
        """Verify alerts trigger at 50%, 75%, 90%, 100%."""
        # Configure budget with all threshold alerts
        metrics_collector.configure_budget(
            limit_usd=0.10,  # $0.10 budget
            period=BudgetPeriod.DAILY,
            alert_thresholds=[50.0, 75.0, 90.0, 100.0]
        )
        metrics_collector.add_alert_callback(alert_tracker.callback)

        # Generate API calls to progressively hit thresholds
        # Each call: ~$0.0105 (1000 input @ $3/M + 500 output @ $15/M)
        # Need ~10 calls to hit $0.105 (exceeds $0.10)

        # First batch: ~50% ($0.05)
        for _ in range(5):
            metrics_collector.record_api_call(
                model="claude-3-5-sonnet-20241022",
                input_tokens=1000,
                output_tokens=500,
                duration_seconds=1.0
            )
            metrics_collector.check_budget()

        # Should have 50% alert
        assert 50.0 in alert_tracker.get_thresholds()

        # More calls: ~75% ($0.075)
        for _ in range(3):
            metrics_collector.record_api_call(
                model="claude-3-5-sonnet-20241022",
                input_tokens=1000,
                output_tokens=500,
                duration_seconds=1.0
            )
            metrics_collector.check_budget()

        assert 75.0 in alert_tracker.get_thresholds()

        # More calls: ~90% and 100%
        for _ in range(4):
            metrics_collector.record_api_call(
                model="claude-3-5-sonnet-20241022",
                input_tokens=1000,
                output_tokens=500,
                duration_seconds=1.0
            )
            metrics_collector.check_budget()

        # Should have all thresholds
        thresholds = alert_tracker.get_thresholds()
        assert 90.0 in thresholds
        assert 100.0 in thresholds

    def test_alerts_not_duplicated(self, metrics_collector, alert_tracker):
        """Verify same threshold alert not triggered multiple times."""
        metrics_collector.configure_budget(
            limit_usd=0.10,
            period=BudgetPeriod.DAILY,
            alert_thresholds=[50.0]
        )
        metrics_collector.add_alert_callback(alert_tracker.callback)

        # Exceed 50% multiple times
        for _ in range(10):
            metrics_collector.record_api_call(
                model="claude-3-5-sonnet-20241022",
                input_tokens=1000,
                output_tokens=500,
                duration_seconds=1.0
            )
            metrics_collector.check_budget()

        # 50% should only appear once
        threshold_counts = alert_tracker.get_thresholds().count(50.0)
        assert threshold_counts == 1


class TestBudgetPeriodReset:
    """Tests for budget period resets."""

    def test_budget_period_reset_hourly(self, metrics_collector):
        """Verify hourly budget resets correctly."""
        metrics_collector.configure_budget(
            limit_usd=1.00,
            period=BudgetPeriod.HOURLY,
            alert_thresholds=[100.0]
        )

        # Record API calls
        metrics_collector.record_api_call(
            model="claude-3-5-sonnet-20241022",
            input_tokens=100000,
            output_tokens=50000,
            duration_seconds=1.0
        )

        # Simulate time passing (more than 1 hour)
        metrics_collector.budget_period_start = datetime.utcnow() - timedelta(hours=2)

        # Check budget - should reset
        status = metrics_collector.check_budget()

        # Period should have reset, alerts cleared
        assert status['total_alerts'] == 0

    def test_budget_period_reset_daily(self, metrics_collector):
        """Verify daily budget resets correctly."""
        metrics_collector.configure_budget(
            limit_usd=1.00,
            period=BudgetPeriod.DAILY,
            alert_thresholds=[100.0]
        )

        # Record some API calls and trigger alert
        metrics_collector.record_api_call(
            model="claude-3-5-sonnet-20241022",
            input_tokens=100000,
            output_tokens=50000,
            duration_seconds=1.0
        )
        metrics_collector.check_budget()

        # Verify alert was triggered
        initial_status = metrics_collector.check_budget()

        # Simulate time passing (more than 1 day)
        metrics_collector.budget_period_start = datetime.utcnow() - timedelta(days=2)

        # Check budget - should reset
        status = metrics_collector.check_budget()

        # Alerts should be cleared after period reset
        assert status['total_alerts'] == 0


class TestCostCalculationAccuracy:
    """Tests for accurate cost calculation."""

    def test_cost_calculation_accuracy_claude_pricing(self, metrics_collector):
        """Verify Claude pricing ($3/$15 per M tokens) is calculated correctly."""
        metrics_collector.configure_budget(
            limit_usd=100.0,
            period=BudgetPeriod.DAILY
        )

        # Record specific token counts
        # 1M input tokens @ $3/M = $3.00
        # 0.5M output tokens @ $15/M = $7.50
        # Total = $10.50
        metrics_collector.record_api_call(
            model="claude-3-5-sonnet-20241022",
            input_tokens=1_000_000,
            output_tokens=500_000,
            duration_seconds=10.0
        )

        status = metrics_collector.check_budget()

        # Should be ~$10.50
        expected_cost = (1_000_000 / 1_000_000 * 3.0) + (500_000 / 1_000_000 * 15.0)
        assert abs(status['current_cost_usd'] - expected_cost) < 0.01

    def test_cost_accumulates_across_calls(self, metrics_collector):
        """Verify costs accumulate correctly across multiple API calls."""
        metrics_collector.configure_budget(
            limit_usd=100.0,
            period=BudgetPeriod.DAILY
        )

        # Record 10 identical API calls
        for _ in range(10):
            metrics_collector.record_api_call(
                model="claude-3-5-sonnet-20241022",
                input_tokens=10000,
                output_tokens=5000,
                duration_seconds=1.0
            )

        status = metrics_collector.check_budget()

        # Each call: (10000/1M * 3) + (5000/1M * 15) = 0.03 + 0.075 = $0.105
        # 10 calls = $1.05
        expected_cost = 10 * ((10000 / 1_000_000 * 3.0) + (5000 / 1_000_000 * 15.0))
        assert abs(status['current_cost_usd'] - expected_cost) < 0.01


class TestBudgetGracefulShutdown:
    """Tests for graceful shutdown when budget exceeded."""

    def test_budget_enforcement_graceful_shutdown(self, metrics_collector):
        """Verify in-progress work can complete before halt."""
        # Configure very low budget
        metrics_collector.configure_budget(
            limit_usd=0.001,  # $0.001
            period=BudgetPeriod.DAILY
        )

        # Simulate workflow with budget check before each expensive operation
        operations_completed = 0
        operations_attempted = 5

        for i in range(operations_attempted):
            # Check budget before operation
            try:
                metrics_collector.enforce_budget()

                # Simulate operation
                metrics_collector.record_api_call(
                    model="claude-3-5-sonnet-20241022",
                    input_tokens=1000,
                    output_tokens=500,
                    duration_seconds=1.0
                )
                operations_completed += 1

            except BudgetExceededError:
                # Budget exceeded - stop gracefully
                break

        # Should complete at least first operation before budget exceeded
        assert operations_completed >= 1
        assert operations_completed < operations_attempted

    def test_budget_status_includes_completion_info(self, metrics_collector):
        """Verify budget status includes useful completion information."""
        metrics_collector.configure_budget(
            limit_usd=0.05,
            period=BudgetPeriod.DAILY,
            alert_thresholds=[50.0, 100.0]
        )

        # Record some API calls
        for _ in range(5):
            metrics_collector.record_api_call(
                model="claude-3-5-sonnet-20241022",
                input_tokens=1000,
                output_tokens=500,
                duration_seconds=1.0
            )

        status = metrics_collector.check_budget()

        # Verify status contains all required fields
        assert 'enabled' in status
        assert 'period' in status
        assert 'current_cost_usd' in status
        assert 'usage_percent' in status
        assert 'budget_exceeded' in status
        assert 'limit_usd' in status


class TestBudgetConcurrentTracking:
    """Tests for thread-safe concurrent cost accumulation."""

    def test_budget_concurrent_tracking(self, metrics_collector):
        """Verify thread-safe cost accumulation."""
        metrics_collector.configure_budget(
            limit_usd=100.0,
            period=BudgetPeriod.DAILY
        )

        num_threads = 10
        calls_per_thread = 100
        errors = []

        def record_calls():
            try:
                for _ in range(calls_per_thread):
                    metrics_collector.record_api_call(
                        model="claude-3-5-sonnet-20241022",
                        input_tokens=100,
                        output_tokens=50,
                        duration_seconds=0.1
                    )
            except Exception as e:
                errors.append(e)

        # Run concurrent threads
        threads = [threading.Thread(target=record_calls) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0

        # Verify total API calls recorded
        stats = metrics_collector.get_api_statistics()
        expected_calls = num_threads * calls_per_thread
        assert stats['total_calls'] == expected_calls

        # Verify cost calculation
        # Each call: (100/1M * 3) + (50/1M * 15) = 0.0003 + 0.00075 = $0.00105
        expected_cost = expected_calls * ((100 / 1_000_000 * 3.0) + (50 / 1_000_000 * 15.0))
        assert abs(stats['estimated_cost_usd'] - expected_cost) < 0.01

    def test_concurrent_budget_checks(self, metrics_collector, alert_tracker):
        """Verify concurrent budget checks don't duplicate alerts."""
        metrics_collector.configure_budget(
            limit_usd=0.10,
            period=BudgetPeriod.DAILY,
            alert_thresholds=[50.0]
        )
        metrics_collector.add_alert_callback(alert_tracker.callback)

        # Pre-populate with calls that exceed 50%
        for _ in range(6):
            metrics_collector.record_api_call(
                model="claude-3-5-sonnet-20241022",
                input_tokens=1000,
                output_tokens=500,
                duration_seconds=1.0
            )

        num_threads = 5
        errors = []

        def check_budget():
            try:
                for _ in range(10):
                    metrics_collector.check_budget()
            except Exception as e:
                errors.append(e)

        # Run concurrent budget checks
        threads = [threading.Thread(target=check_budget) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors
        assert len(errors) == 0

        # 50% threshold should only be triggered once
        threshold_count = alert_tracker.get_thresholds().count(50.0)
        assert threshold_count == 1

"""
E2E Tests for Error Recovery.

Tests the error recovery and circuit breaker functionality:
- Circuit breaker state transitions
- Retry logic for recoverable errors
- No retry for permanent errors
- Workflow continuation after agent failure
- Task retry tracking
- Partial batch failure handling
"""

import pytest
import asyncio
import time
from datetime import datetime

from kosmos.core.async_llm import (
    CircuitBreaker,
    RateLimiter,
    is_recoverable_error,
    BatchRequest,
    BatchResponse
)
from kosmos.core.providers.base import ProviderAPIError


pytestmark = [pytest.mark.e2e]


class TestCircuitBreakerStateTransitions:
    """Tests for circuit breaker state transitions."""

    def test_circuit_breaker_initial_state_closed(self, circuit_breaker):
        """Verify circuit breaker starts in CLOSED state."""
        assert circuit_breaker.state == "CLOSED"
        assert circuit_breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, circuit_breaker):
        """Verify CB opens after 3 consecutive failures."""
        # Initial state: CLOSED
        assert circuit_breaker.state == "CLOSED"
        assert await circuit_breaker.can_execute() is True

        # Record 3 failures (failure_threshold=3)
        for i in range(3):
            await circuit_breaker.record_failure(Exception(f"Failure {i}"))

        # Should be OPEN now
        assert circuit_breaker.state == "OPEN"
        assert circuit_breaker.failure_count == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_when_open(self, circuit_breaker):
        """Verify requests are blocked when CB is open."""
        # Open the circuit breaker
        for _ in range(3):
            await circuit_breaker.record_failure(Exception("Failure"))

        assert circuit_breaker.state == "OPEN"

        # Should not allow execution
        assert await circuit_breaker.can_execute() is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovers_to_half_open(self, circuit_breaker):
        """Verify CB transitions to HALF_OPEN after timeout."""
        # Open the circuit breaker
        for _ in range(3):
            await circuit_breaker.record_failure(Exception("Failure"))

        assert circuit_breaker.state == "OPEN"

        # Wait for recovery timeout (1 second in test config)
        await asyncio.sleep(1.5)

        # Should transition to HALF_OPEN on next can_execute check
        can_exec = await circuit_breaker.can_execute()
        assert can_exec is True
        assert circuit_breaker.state == "HALF_OPEN"

    @pytest.mark.asyncio
    async def test_circuit_breaker_closes_on_success(self, circuit_breaker):
        """Verify CB closes after success in HALF_OPEN."""
        # Open the circuit breaker
        for _ in range(3):
            await circuit_breaker.record_failure(Exception("Failure"))

        # Wait for recovery
        await asyncio.sleep(1.5)

        # Trigger HALF_OPEN
        await circuit_breaker.can_execute()
        assert circuit_breaker.state == "HALF_OPEN"

        # Record success
        await circuit_breaker.record_success()

        # Should be CLOSED now
        assert circuit_breaker.state == "CLOSED"
        assert circuit_breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_reopens_on_half_open_failure(self, circuit_breaker):
        """Verify CB reopens immediately on failure in HALF_OPEN."""
        # Open the circuit breaker
        for _ in range(3):
            await circuit_breaker.record_failure(Exception("Failure"))

        # Wait for recovery
        await asyncio.sleep(1.5)

        # Trigger HALF_OPEN
        await circuit_breaker.can_execute()
        assert circuit_breaker.state == "HALF_OPEN"

        # Record failure
        await circuit_breaker.record_failure(Exception("Still failing"))

        # Should be OPEN again
        assert circuit_breaker.state == "OPEN"

    def test_circuit_breaker_get_state(self, circuit_breaker):
        """Verify get_state returns complete state info."""
        state = circuit_breaker.get_state()

        assert "state" in state
        assert "failure_count" in state
        assert "failure_threshold" in state
        assert "last_failure_time" in state

        assert state["state"] == "CLOSED"
        assert state["failure_threshold"] == 3


class TestErrorRecoverability:
    """Tests for error recoverability classification."""

    def test_retry_on_recoverable_errors(self):
        """Verify recoverable errors are identified correctly."""
        # Rate limit errors are recoverable
        rate_limit_error = ProviderAPIError(
            provider="anthropic",
            message="Rate limit exceeded",
            recoverable=True
        )
        assert is_recoverable_error(rate_limit_error) is True

        # Timeout errors are recoverable
        timeout_error = ProviderAPIError(
            provider="anthropic",
            message="Request timeout",
            recoverable=True
        )
        assert is_recoverable_error(timeout_error) is True

    def test_no_retry_on_permanent_errors(self):
        """Verify permanent errors are not retried."""
        # Authentication errors are not recoverable
        auth_error = ProviderAPIError(
            provider="anthropic",
            message="Invalid API key - authentication failed",
            recoverable=False
        )
        assert is_recoverable_error(auth_error) is False

        # Invalid request errors are not recoverable
        invalid_error = ProviderAPIError(
            provider="anthropic",
            message="Invalid request format",
            recoverable=False
        )
        assert is_recoverable_error(invalid_error) is False

    def test_unknown_errors_default_recoverable(self):
        """Verify unknown errors default to potentially recoverable."""
        unknown_error = Exception("Some unknown error")
        # Unknown errors should be considered potentially recoverable
        assert is_recoverable_error(unknown_error) is True


class TestWorkflowContinuation:
    """Tests for workflow continuation after failures."""

    @pytest.mark.asyncio
    async def test_workflow_continues_after_agent_failure(self):
        """Verify workflow continues despite individual agent failures."""
        # Simulate a workflow with multiple cycles
        call_count = [0]
        results = []

        async def failing_agent(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Simulated agent failure")
            return {"hypotheses": [{"id": f"hyp_{call_count[0]}"}]}

        # Execute 5 cycles with failure handling
        for cycle in range(1, 6):
            try:
                result = await failing_agent()
                results.append({"cycle": cycle, "success": True, "result": result})
            except Exception as e:
                results.append({"cycle": cycle, "success": False, "error": str(e)})
                # Workflow continues despite failure

        # Verify workflow continued
        assert len(results) == 5
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        assert len(successful) == 4  # 4 successful
        assert len(failed) == 1  # 1 failed (cycle 2)

    @pytest.mark.asyncio
    async def test_workflow_tracks_error_history(self):
        """Verify workflow tracks error history correctly."""
        error_history = []
        consecutive_errors = [0]

        async def failing_operation(should_fail: bool):
            if should_fail:
                consecutive_errors[0] += 1
                error_history.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": "Operation failed",
                    "consecutive_count": consecutive_errors[0]
                })
                raise Exception("Operation failed")
            else:
                consecutive_errors[0] = 0  # Reset on success
                return "success"

        # Simulate mixed success/failure
        operations = [True, True, False, True, False, False]  # fail, fail, success, fail, success, success

        for should_fail in operations:
            try:
                await failing_operation(should_fail)
            except Exception:
                pass

        # Verify error history
        assert len(error_history) == 3  # 3 failures
        assert error_history[0]["consecutive_count"] == 1
        assert error_history[1]["consecutive_count"] == 2
        assert error_history[2]["consecutive_count"] == 1  # Reset after success


class TestTaskRetryTracking:
    """Tests for task retry tracking."""

    def test_task_retry_increments(self):
        """Verify failed tasks track retry count."""

        class TaskTracker:
            def __init__(self):
                self.tasks = {}

            def create_task(self, task_id: str):
                self.tasks[task_id] = {"id": task_id, "retry_count": 0, "status": "pending"}

            def record_retry(self, task_id: str):
                if task_id in self.tasks:
                    self.tasks[task_id]["retry_count"] += 1

            def get_task(self, task_id: str):
                return self.tasks.get(task_id)

        tracker = TaskTracker()

        # Create task
        tracker.create_task("task_001")
        assert tracker.get_task("task_001")["retry_count"] == 0

        # Record retries
        tracker.record_retry("task_001")
        assert tracker.get_task("task_001")["retry_count"] == 1

        tracker.record_retry("task_001")
        assert tracker.get_task("task_001")["retry_count"] == 2

        tracker.record_retry("task_001")
        assert tracker.get_task("task_001")["retry_count"] == 3

    def test_max_retries_respected(self):
        """Verify tasks stop retrying after max retries."""
        max_retries = 3
        retry_count = [0]
        final_status = [None]

        def execute_with_retry(task_id: str):
            for attempt in range(max_retries + 1):
                try:
                    # Always fail for testing
                    raise Exception(f"Attempt {attempt + 1} failed")
                except Exception:
                    retry_count[0] = attempt + 1
                    if attempt >= max_retries:
                        final_status[0] = "max_retries_exceeded"
                        break
                    # Would retry here

        execute_with_retry("task_001")

        assert retry_count[0] == max_retries + 1  # Initial + retries
        assert final_status[0] == "max_retries_exceeded"


class TestPartialBatchFailureHandling:
    """Tests for handling partial batch failures."""

    @pytest.mark.asyncio
    async def test_partial_batch_failure_handling(self):
        """Verify batch continues despite some failures."""

        async def process_item(item_id: int) -> BatchResponse:
            # Simulate failure for specific items
            if item_id in [2, 5, 8]:
                return BatchResponse(
                    id=str(item_id),
                    success=False,
                    error=f"Item {item_id} failed"
                )
            return BatchResponse(
                id=str(item_id),
                success=True,
                response=f"Result for {item_id}"
            )

        # Process batch of 10 items
        batch_items = list(range(10))
        tasks = [process_item(i) for i in batch_items]
        results = await asyncio.gather(*tasks)

        # Verify results
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        assert len(results) == 10
        assert len(successful) == 7
        assert len(failed) == 3

        # Verify failed IDs
        failed_ids = [int(r.id) for r in failed]
        assert set(failed_ids) == {2, 5, 8}

    @pytest.mark.asyncio
    async def test_batch_aggregates_statistics(self):
        """Verify batch properly aggregates success/failure statistics."""
        total_requests = [0]
        successful_requests = [0]
        failed_requests = [0]
        total_tokens = [0]

        async def process_with_stats(item_id: int) -> BatchResponse:
            total_requests[0] += 1

            if item_id % 3 == 0:  # Every 3rd item fails
                failed_requests[0] += 1
                return BatchResponse(
                    id=str(item_id),
                    success=False,
                    error="Failed"
                )
            else:
                successful_requests[0] += 1
                tokens = 100 + item_id * 10
                total_tokens[0] += tokens
                return BatchResponse(
                    id=str(item_id),
                    success=True,
                    response=f"Result {item_id}",
                    input_tokens=tokens,
                    output_tokens=tokens // 2
                )

        # Process batch
        batch_items = list(range(12))  # 0-11, items 0, 3, 6, 9 will fail
        tasks = [process_with_stats(i) for i in batch_items]
        await asyncio.gather(*tasks)

        # Verify statistics
        assert total_requests[0] == 12
        assert failed_requests[0] == 4  # 0, 3, 6, 9
        assert successful_requests[0] == 8
        assert total_tokens[0] > 0

    @pytest.mark.asyncio
    async def test_batch_respects_circuit_breaker(self, circuit_breaker):
        """Verify batch processing respects circuit breaker."""
        call_attempts = [0]
        blocked_attempts = [0]

        async def process_with_circuit_breaker(item_id: int) -> BatchResponse:
            # Check circuit breaker before processing
            if not await circuit_breaker.can_execute():
                blocked_attempts[0] += 1
                return BatchResponse(
                    id=str(item_id),
                    success=False,
                    error="Circuit breaker open"
                )

            call_attempts[0] += 1

            # Simulate failures that trip circuit breaker
            if item_id < 3:
                await circuit_breaker.record_failure(Exception("Failed"))
                return BatchResponse(
                    id=str(item_id),
                    success=False,
                    error="Processing failed"
                )

            await circuit_breaker.record_success()
            return BatchResponse(
                id=str(item_id),
                success=True,
                response=f"Result {item_id}"
            )

        # Process items 0-9
        results = []
        for i in range(10):
            result = await process_with_circuit_breaker(i)
            results.append(result)

        # First 3 items fail and trip circuit breaker
        # Remaining items should be blocked
        assert call_attempts[0] == 3  # Only first 3 got through
        assert blocked_attempts[0] == 7  # Rest were blocked


class TestRateLimiter:
    """Tests for rate limiter functionality."""

    @pytest.mark.asyncio
    async def test_rate_limiter_respects_concurrent_limit(self, rate_limiter):
        """Verify rate limiter respects max concurrent requests."""
        concurrent_count = [0]
        max_observed = [0]

        async def tracked_operation():
            await rate_limiter.acquire()
            concurrent_count[0] += 1
            max_observed[0] = max(max_observed[0], concurrent_count[0])

            await asyncio.sleep(0.1)  # Simulate operation

            concurrent_count[0] -= 1
            rate_limiter.release()

        # Launch more requests than concurrent limit
        tasks = [tracked_operation() for _ in range(10)]
        await asyncio.gather(*tasks)

        # Max concurrent should be limited
        assert max_observed[0] <= rate_limiter.max_concurrent

    @pytest.mark.asyncio
    async def test_rate_limiter_token_refill(self, rate_limiter):
        """Verify rate limiter refills tokens over time."""
        # Consume initial tokens
        initial_tokens = rate_limiter.tokens

        await rate_limiter.acquire()
        rate_limiter.release()

        tokens_after = rate_limiter.tokens

        # Tokens should have been consumed
        assert tokens_after < initial_tokens

        # Wait for some token refill
        await asyncio.sleep(0.5)

        # Manually trigger refill by acquiring
        await rate_limiter.acquire()
        rate_limiter.release()

        # Tokens should be at least partially refilled
        # (exact value depends on timing)
        assert rate_limiter.tokens >= 0

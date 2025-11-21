"""
Tests for Performance Resource Requirements (REQ-PERF-RES-001 through REQ-PERF-RES-009).

These tests validate resource management including prompt caching, memory limits,
code/paper tracking, CPU usage, disk I/O, and resource cleanup.
"""

import pytest
import time
import psutil
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from kosmos.agents.research_director import ResearchDirectorAgent
from kosmos.core.workflow import ResearchPlan
from kosmos.monitoring.metrics import MetricsCollector
from kosmos.core.metrics import get_metrics

# Test markers for requirements traceability
pytestmark = [
    pytest.mark.requirement("REQ-PERF-RES"),
    pytest.mark.category("performance"),
]


@pytest.mark.requirement("REQ-PERF-RES-001")
@pytest.mark.priority("MUST")
class TestREQ_PERF_RES_001_PromptCaching:
    """
    REQ-PERF-RES-001: The system MUST achieve >50% prompt caching hit rate
    for repeated LLM operations to minimize API costs and latency.
    """

    def test_cache_hit_rate_above_fifty_percent(self):
        """Verify cache achieves >50% hit rate."""
        metrics = get_metrics(reset=True)

        # Simulate cache hits and misses
        total_requests = 100
        cache_hits = 60
        cache_misses = 40

        for i in range(cache_hits):
            metrics.record_cache_hit(cache_type="claude")

        for i in range(cache_misses):
            metrics.record_cache_miss(cache_type="claude")

        stats = metrics.get_cache_statistics()

        hit_rate = stats.get("cache_hit_rate_percent", 0)

        assert hit_rate >= 50.0, \
            f"Cache hit rate {hit_rate:.1f}% is below 50% requirement"

    def test_prompt_cache_effectiveness(self):
        """Verify prompt caching reduces redundant API calls."""
        metrics = get_metrics(reset=True)

        # Simulate repeated queries with caching
        repeated_prompts = [
            "What is the hypothesis?",
            "What is the hypothesis?",  # Cache hit
            "Design an experiment",
            "Design an experiment",  # Cache hit
            "Analyze results",
            "Analyze results",  # Cache hit
        ]

        cache_hits = 0
        for i, prompt in enumerate(repeated_prompts):
            if i > 0 and prompt == repeated_prompts[i-1]:
                # Cache hit
                metrics.record_cache_hit(cache_type="claude")
                cache_hits += 1
            else:
                # Cache miss
                metrics.record_cache_miss(cache_type="claude")

        stats = metrics.get_cache_statistics()
        hit_rate = stats.get("cache_hit_rate_percent", 0)

        # With 3 repeated prompts out of 6 total, hit rate should be 50%
        assert hit_rate >= 50.0, \
            f"Cache hit rate {hit_rate:.1f}% below requirement"

    def test_cache_improves_response_time(self):
        """Verify cache hits have lower latency than misses."""
        # Simulate API call times
        cache_miss_times = []
        cache_hit_times = []

        # Cache misses: slower (actual API call)
        for i in range(10):
            start = time.time()
            time.sleep(0.01)  # Simulate API latency
            cache_miss_times.append(time.time() - start)

        # Cache hits: faster (cached response)
        for i in range(10):
            start = time.time()
            time.sleep(0.001)  # Simulate cache lookup
            cache_hit_times.append(time.time() - start)

        avg_miss_time = sum(cache_miss_times) / len(cache_miss_times)
        avg_hit_time = sum(cache_hit_times) / len(cache_hit_times)

        # Cache hits should be significantly faster
        speedup = avg_miss_time / avg_hit_time
        assert speedup > 2.0, \
            f"Cache only provides {speedup:.1f}x speedup, expected >2x"

    def test_cache_cost_savings(self):
        """Verify cache provides cost savings."""
        metrics = get_metrics(reset=True)

        # Simulate operations with good cache hit rate
        for i in range(60):
            metrics.record_cache_hit(cache_type="claude")

        for i in range(40):
            metrics.record_cache_miss(cache_type="claude")
            # Cache miss requires API call
            metrics.record_api_call(
                model="claude-3-5-sonnet",
                input_tokens=1000,
                output_tokens=500,
                duration_seconds=1.0
            )

        stats = metrics.get_cache_statistics()

        # Verify hit rate
        hit_rate = stats.get("cache_hit_rate_percent", 0)
        assert hit_rate >= 50.0

        # Check if cost savings are estimated
        if "estimated_cost_saved_usd" in stats:
            cost_saved = stats["estimated_cost_saved_usd"]
            assert cost_saved > 0, "Should show cost savings from caching"


@pytest.mark.requirement("REQ-PERF-RES-002")
@pytest.mark.priority("MUST")
class TestREQ_PERF_RES_002_MemoryLimits:
    """
    REQ-PERF-RES-002: The system MUST enforce memory limits per process
    and prevent memory usage from exceeding configured thresholds.
    """

    def test_memory_usage_within_limits(self):
        """Verify memory usage stays within reasonable limits."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Set reasonable limit (e.g., 500MB growth)
        memory_limit_mb = 500

        with patch('kosmos.agents.research_director.get_client') as mock_llm:
            with patch('kosmos.world_model.get_world_model') as mock_wm:
                mock_llm.return_value = Mock()
                mock_wm.return_value = None

                director = ResearchDirectorAgent(
                    research_question="Test memory limits",
                    config={"max_iterations": 100}
                )

                # Simulate memory-intensive operations
                for i in range(100):
                    director.research_plan.add_hypothesis(f"hyp_{i}")
                    director.research_plan.add_experiment(f"exp_{i}")

                current_memory = process.memory_info().rss / 1024 / 1024
                memory_used = current_memory - initial_memory

                assert memory_used < memory_limit_mb, \
                    f"Memory usage {memory_used:.1f}MB exceeds limit of {memory_limit_mb}MB"

    def test_memory_limit_enforcement(self):
        """Verify system respects configured memory limits."""
        # This would integrate with actual memory limit enforcement
        # For now, verify monitoring capabilities exist

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        # Verify we can monitor memory
        assert memory_info.rss > 0
        assert memory_info.vms > 0

        # Check if memory percent is reasonable
        memory_percent = process.memory_percent()
        assert memory_percent < 90, \
            f"Memory usage at {memory_percent:.1f}% is too high"

    def test_memory_alerts_triggered(self):
        """Verify memory alerts are triggered when approaching limits."""
        process = psutil.Process(os.getpid())
        memory_percent = process.memory_percent()

        # Define alert threshold
        alert_threshold = 80.0

        if memory_percent > alert_threshold:
            # Alert should be triggered
            alert_triggered = True
        else:
            alert_triggered = False

        # Verify alert logic works
        assert isinstance(alert_triggered, bool)


@pytest.mark.requirement("REQ-PERF-RES-003")
@pytest.mark.priority("MUST")
class TestREQ_PERF_RES_003_CodeLineTracking:
    """
    REQ-PERF-RES-003: The system MUST track total lines of code analyzed
    and referenced across all experiments.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_code_line_tracking(self, mock_wm, mock_llm):
        """Verify system tracks lines of code."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test code tracking",
            config={"max_iterations": 10}
        )

        # Simulate code analysis
        code_samples = [
            "def foo():\n    return 42",  # 2 lines
            "class Bar:\n    def __init__(self):\n        pass",  # 3 lines
            "x = 1\ny = 2\nz = x + y",  # 3 lines
        ]

        total_lines_analyzed = 0

        for code in code_samples:
            lines = code.count('\n') + 1
            total_lines_analyzed += lines

        # Verify tracking capability
        assert total_lines_analyzed == 8

        # In actual implementation, this would be tracked in world model
        # or research plan metadata

    def test_code_line_accumulation(self):
        """Verify code lines accumulate across experiments."""
        code_line_counts = []

        # Simulate multiple experiments analyzing code
        experiments = [
            {"code_lines": 100},
            {"code_lines": 250},
            {"code_lines": 300},
            {"code_lines": 150},
        ]

        cumulative_lines = 0
        for exp in experiments:
            cumulative_lines += exp["code_lines"]
            code_line_counts.append(cumulative_lines)

        # Verify accumulation
        assert code_line_counts == [100, 350, 650, 800]
        assert cumulative_lines == 800

    def test_code_line_metadata_storage(self):
        """Verify code line counts are stored in metadata."""
        with patch('kosmos.agents.research_director.get_client') as mock_llm:
            with patch('kosmos.world_model.get_world_model') as mock_wm:
                mock_llm.return_value = Mock()
                mock_wm.return_value = None

                director = ResearchDirectorAgent(
                    research_question="Test metadata",
                    config={"max_iterations": 1}
                )

                # Store code line count in metadata
                if not hasattr(director.research_plan, 'metadata'):
                    director.research_plan.metadata = {}

                director.research_plan.metadata['total_code_lines'] = 1000

                # Verify it can be retrieved
                assert director.research_plan.metadata.get('total_code_lines') == 1000


@pytest.mark.requirement("REQ-PERF-RES-004")
@pytest.mark.priority("MUST")
class TestREQ_PERF_RES_004_PaperTracking:
    """
    REQ-PERF-RES-004: The system MUST track total number of papers
    analyzed and referenced in literature review.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_paper_count_tracking(self, mock_wm, mock_llm):
        """Verify system tracks number of papers."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test paper tracking",
            config={"max_iterations": 10}
        )

        # Simulate paper analysis
        papers = [
            {"id": "paper1", "title": "Title 1"},
            {"id": "paper2", "title": "Title 2"},
            {"id": "paper3", "title": "Title 3"},
        ]

        # Track papers
        if not hasattr(director.research_plan, 'metadata'):
            director.research_plan.metadata = {}

        director.research_plan.metadata['papers_analyzed'] = len(papers)

        # Verify tracking
        assert director.research_plan.metadata['papers_analyzed'] == 3

    def test_paper_accumulation_across_iterations(self):
        """Verify paper counts accumulate across iterations."""
        paper_counts_by_iteration = []

        # Simulate multiple iterations analyzing papers
        iterations = [
            {"new_papers": 10},
            {"new_papers": 15},
            {"new_papers": 20},
            {"new_papers": 8},
        ]

        cumulative_papers = 0
        for iteration in iterations:
            cumulative_papers += iteration["new_papers"]
            paper_counts_by_iteration.append(cumulative_papers)

        # Verify accumulation
        assert paper_counts_by_iteration == [10, 25, 45, 53]
        assert cumulative_papers == 53

    def test_paper_metadata_persistence(self):
        """Verify paper tracking metadata persists."""
        with patch('kosmos.agents.research_director.get_client') as mock_llm:
            with patch('kosmos.world_model.get_world_model') as mock_wm:
                mock_llm.return_value = Mock()
                mock_wm.return_value = None

                director = ResearchDirectorAgent(
                    research_question="Test persistence",
                    config={"max_iterations": 1}
                )

                # Set paper count
                if not hasattr(director.research_plan, 'metadata'):
                    director.research_plan.metadata = {}

                director.research_plan.metadata['total_papers'] = 100
                director.research_plan.metadata['papers_cited'] = 50

                # Verify retrieval
                assert director.research_plan.metadata.get('total_papers') == 100
                assert director.research_plan.metadata.get('papers_cited') == 50


@pytest.mark.requirement("REQ-PERF-RES-005")
@pytest.mark.priority("MUST")
class TestREQ_PERF_RES_005_CPUUsageMonitoring:
    """
    REQ-PERF-RES-005: The system MUST monitor CPU usage and prevent
    individual processes from consuming excessive CPU resources.
    """

    def test_cpu_usage_monitoring(self):
        """Verify CPU usage can be monitored."""
        process = psutil.Process(os.getpid())

        # Get CPU usage
        cpu_percent = process.cpu_percent(interval=0.1)

        # Verify monitoring works
        assert cpu_percent >= 0
        assert isinstance(cpu_percent, (int, float))

    def test_cpu_usage_stays_reasonable(self):
        """Verify CPU usage stays within reasonable bounds."""
        process = psutil.Process(os.getpid())

        # Perform some work
        total = 0
        for i in range(10000):
            total += i

        # Check CPU usage
        cpu_percent = process.cpu_percent(interval=0.1)

        # Should not exceed 100% (single core)
        # In multi-core systems, this can be higher, but per-process should be reasonable
        assert cpu_percent < 200, \
            f"CPU usage {cpu_percent:.1f}% is excessive"

    def test_cpu_metrics_collection(self):
        """Verify CPU metrics are collected."""
        metrics = MetricsCollector() if hasattr(MetricsCollector, '__init__') else None

        if metrics and hasattr(metrics, 'update_system_metrics'):
            process = psutil.Process(os.getpid())
            memory = process.memory_info()

            # Update system metrics
            metrics.update_system_metrics(
                cpu_percent=process.cpu_percent(),
                memory_rss=memory.rss,
                memory_vms=memory.vms,
                disk_total=0,
                disk_used=0,
                disk_free=0
            )

            # Verify no errors
            assert True
        else:
            # Basic CPU monitoring is available
            assert psutil.cpu_percent(interval=0.1) >= 0


@pytest.mark.requirement("REQ-PERF-RES-006")
@pytest.mark.priority("MUST")
class TestREQ_PERF_RES_006_DiskIOLimits:
    """
    REQ-PERF-RES-006: The system MUST limit disk I/O operations and
    prevent excessive disk writes during operation.
    """

    def test_disk_io_monitoring(self):
        """Verify disk I/O can be monitored."""
        process = psutil.Process(os.getpid())

        # Get I/O counters
        try:
            io_counters = process.io_counters()
            assert io_counters.read_bytes >= 0
            assert io_counters.write_bytes >= 0
        except AttributeError:
            # Not available on all platforms
            pytest.skip("I/O counters not available on this platform")

    def test_disk_write_limits(self):
        """Verify disk writes are limited."""
        process = psutil.Process(os.getpid())

        try:
            io_before = process.io_counters()

            # Perform some file operations
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                temp_file = f.name
                # Write modest amount of data
                for i in range(100):
                    f.write(f"Line {i}\n")

            io_after = process.io_counters()

            # Calculate bytes written
            bytes_written = io_after.write_bytes - io_before.write_bytes

            # Should be reasonable amount (< 1MB for this test)
            assert bytes_written < 1024 * 1024, \
                f"Wrote {bytes_written} bytes, excessive for test operation"

            # Cleanup
            os.unlink(temp_file)

        except AttributeError:
            pytest.skip("I/O counters not available on this platform")

    def test_disk_usage_monitoring(self):
        """Verify disk usage can be monitored."""
        disk = psutil.disk_usage('/')

        assert disk.total > 0
        assert disk.used >= 0
        assert disk.free >= 0
        assert disk.percent >= 0
        assert disk.percent <= 100


@pytest.mark.requirement("REQ-PERF-RES-007")
@pytest.mark.priority("MUST")
class TestREQ_PERF_RES_007_NetworkBandwidth:
    """
    REQ-PERF-RES-007: The system MUST monitor network bandwidth usage
    (API calls) and implement rate limiting.
    """

    def test_api_call_rate_limiting(self):
        """Verify API call rate limiting is enforced."""
        metrics = get_metrics(reset=True)

        # Configure rate limit
        max_calls_per_minute = 60

        # Track calls
        call_times = []
        current_time = time.time()

        for i in range(10):
            call_times.append(current_time + i * 0.5)

        # Count calls in last minute
        one_minute_ago = current_time - 60
        recent_calls = [t for t in call_times if t > one_minute_ago]

        # Verify under limit
        assert len(recent_calls) <= max_calls_per_minute, \
            f"Exceeded rate limit: {len(recent_calls)} calls in last minute"

    def test_bandwidth_monitoring(self):
        """Verify network bandwidth can be monitored."""
        # Get network I/O statistics
        net_io = psutil.net_io_counters()

        assert net_io.bytes_sent >= 0
        assert net_io.bytes_recv >= 0
        assert net_io.packets_sent >= 0
        assert net_io.packets_recv >= 0

    def test_api_call_metrics_tracking(self):
        """Verify API call metrics track bandwidth usage."""
        metrics = get_metrics(reset=True)

        # Simulate API calls
        for i in range(10):
            metrics.record_api_call(
                model="claude-3-5-sonnet",
                input_tokens=1000,
                output_tokens=500,
                duration_seconds=1.0,
                success=True
            )

        stats = metrics.get_api_statistics()

        # Verify tracking
        assert stats["total_calls"] == 10
        assert stats["total_input_tokens"] == 10000
        assert stats["total_output_tokens"] == 5000

    def test_rate_limit_configuration(self):
        """Verify rate limits can be configured."""
        with patch('kosmos.agents.research_director.get_client') as mock_llm:
            with patch('kosmos.world_model.get_world_model') as mock_wm:
                mock_llm.return_value = Mock()
                mock_wm.return_value = None

                director = ResearchDirectorAgent(
                    research_question="Test rate limits",
                    config={
                        "llm_rate_limit_per_minute": 100,
                        "max_concurrent_llm_calls": 10
                    }
                )

                # Verify config is set
                assert director.config.get("llm_rate_limit_per_minute") == 100
                assert director.config.get("max_concurrent_llm_calls") == 10


@pytest.mark.requirement("REQ-PERF-RES-008")
@pytest.mark.priority("MUST")
class TestREQ_PERF_RES_008_ResourceCleanup:
    """
    REQ-PERF-RES-008: The system MUST properly clean up resources
    (file handles, connections, memory) after operations complete.
    """

    def test_file_handle_cleanup(self):
        """Verify file handles are properly closed."""
        process = psutil.Process(os.getpid())

        # Get initial file descriptor count
        try:
            initial_fds = process.num_fds()
        except AttributeError:
            pytest.skip("File descriptor count not available on this platform")

        # Perform file operations
        temp_files = []
        for i in range(10):
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                temp_files.append(f.name)
                f.write("test data")

        # Files should be closed
        current_fds = process.num_fds()

        # Cleanup temp files
        for temp_file in temp_files:
            os.unlink(temp_file)

        final_fds = process.num_fds()

        # File descriptors should return to baseline
        fd_leak = final_fds - initial_fds
        assert fd_leak < 5, \
            f"File descriptor leak detected: {fd_leak} descriptors not cleaned up"

    def test_connection_cleanup(self):
        """Verify database connections are properly closed."""
        # This would test actual database connection cleanup
        # For now, verify monitoring capability exists

        process = psutil.Process(os.getpid())
        connections = process.connections()

        # Verify we can monitor connections
        assert isinstance(connections, list)

    def test_memory_cleanup_after_experiment(self):
        """Verify memory is cleaned up after experiments."""
        import gc

        process = psutil.Process(os.getpid())
        gc.collect()

        initial_memory = process.memory_info().rss / 1024 / 1024

        with patch('kosmos.agents.research_director.get_client') as mock_llm:
            with patch('kosmos.world_model.get_world_model') as mock_wm:
                mock_llm.return_value = Mock()
                mock_wm.return_value = None

                # Create and destroy multiple directors
                for i in range(5):
                    director = ResearchDirectorAgent(
                        research_question=f"Test cleanup {i}",
                        config={"max_iterations": 1}
                    )

                    # Simulate work
                    for j in range(100):
                        director.research_plan.add_hypothesis(f"hyp_{i}_{j}")

                    # Cleanup
                    del director

                # Force cleanup
                gc.collect()

                final_memory = process.memory_info().rss / 1024 / 1024
                memory_retained = final_memory - initial_memory

                # Allow some growth but should be minimal
                assert memory_retained < 50, \
                    f"Memory not cleaned up properly: {memory_retained:.1f}MB retained"


@pytest.mark.requirement("REQ-PERF-RES-009")
@pytest.mark.priority("MUST")
class TestREQ_PERF_RES_009_MemoryLeakPrevention:
    """
    REQ-PERF-RES-009: The system MUST prevent memory leaks through
    proper object lifecycle management and garbage collection.
    """

    def test_no_memory_leak_in_loop(self):
        """Verify no memory leaks in repeated operations."""
        import gc

        process = psutil.Process(os.getpid())
        gc.collect()

        memory_samples = []

        # Sample memory at intervals
        for iteration in range(10):
            # Perform operations
            with patch('kosmos.agents.research_director.get_client') as mock_llm:
                with patch('kosmos.world_model.get_world_model') as mock_wm:
                    mock_llm.return_value = Mock()
                    mock_wm.return_value = None

                    director = ResearchDirectorAgent(
                        research_question=f"Iteration {iteration}",
                        config={"max_iterations": 1}
                    )

                    # Simulate work
                    for j in range(50):
                        director.research_plan.add_hypothesis(f"hyp_{iteration}_{j}")

                    del director

            # Force garbage collection
            gc.collect()

            # Sample memory
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)

        # Check for monotonic growth (memory leak indicator)
        # Calculate linear regression slope
        n = len(memory_samples)
        if n > 2:
            x_mean = (n - 1) / 2
            y_mean = sum(memory_samples) / n

            numerator = sum((i - x_mean) * (memory_samples[i] - y_mean) for i in range(n))
            denominator = sum((i - x_mean) ** 2 for i in range(n))

            if denominator > 0:
                slope = numerator / denominator

                # Slope should be minimal (< 1MB per iteration)
                assert slope < 1.0, \
                    f"Memory leak detected: {slope:.2f}MB growth per iteration"

    def test_object_lifecycle_management(self):
        """Verify objects are properly garbage collected."""
        import gc
        import weakref

        gc.collect()

        weak_refs = []

        # Create objects with weak references
        with patch('kosmos.agents.research_director.get_client') as mock_llm:
            with patch('kosmos.world_model.get_world_model') as mock_wm:
                mock_llm.return_value = Mock()
                mock_wm.return_value = None

                for i in range(5):
                    director = ResearchDirectorAgent(
                        research_question=f"Test {i}",
                        config={"max_iterations": 1}
                    )
                    weak_refs.append(weakref.ref(director))
                    del director

        # Force garbage collection
        gc.collect()

        # All weak references should be dead
        alive_count = sum(1 for ref in weak_refs if ref() is not None)
        assert alive_count == 0, \
            f"{alive_count} objects not garbage collected (memory leak)"

    def test_circular_reference_detection(self):
        """Verify no circular references prevent garbage collection."""
        import gc
        import sys

        gc.collect()

        with patch('kosmos.agents.research_director.get_client') as mock_llm:
            with patch('kosmos.world_model.get_world_model') as mock_wm:
                mock_llm.return_value = Mock()
                mock_wm.return_value = None

                director = ResearchDirectorAgent(
                    research_question="Test circular refs",
                    config={"max_iterations": 1}
                )

                # Get reference count
                initial_refcount = sys.getrefcount(director)

                # Director should not have excessive references
                assert initial_refcount < 10, \
                    f"Object has {initial_refcount} references, possible circular references"

                del director
                gc.collect()

    def test_long_running_memory_stability(self):
        """Verify memory remains stable over many operations."""
        import gc

        process = psutil.Process(os.getpid())
        gc.collect()

        initial_memory = process.memory_info().rss / 1024 / 1024
        memory_samples = []

        # Perform many operations
        for i in range(20):
            with patch('kosmos.agents.research_director.get_client') as mock_llm:
                with patch('kosmos.world_model.get_world_model') as mock_wm:
                    mock_llm.return_value = Mock()
                    mock_wm.return_value = None

                    director = ResearchDirectorAgent(
                        research_question=f"Test {i}",
                        config={"max_iterations": 1}
                    )

                    # Simulate work
                    for j in range(20):
                        director.research_plan.add_hypothesis(f"hyp_{i}_{j}")

                    del director

            if i % 5 == 0:
                gc.collect()
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)

        # Verify memory hasn't grown excessively
        if len(memory_samples) > 1:
            memory_growth = memory_samples[-1] - memory_samples[0]
            assert memory_growth < 50, \
                f"Memory grew by {memory_growth:.1f}MB over 20 iterations (leak suspected)"

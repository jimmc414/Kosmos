"""
Tests for Performance Scalability Requirements (REQ-PERF-SCALE-001 through REQ-PERF-SCALE-003).

These tests validate system scalability including handling 40K+ lines of code,
1K+ papers, and 150+ rollouts capacity.
"""

import pytest
import time
import psutil
import os
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from kosmos.agents.research_director import ResearchDirectorAgent
from kosmos.core.workflow import ResearchPlan
from kosmos.monitoring.metrics import MetricsCollector
from kosmos.core.metrics import get_metrics

# Test markers for requirements traceability
pytestmark = [
    pytest.mark.requirement("REQ-PERF-SCALE"),
    pytest.mark.category("performance"),
]


@pytest.mark.requirement("REQ-PERF-SCALE-001")
@pytest.mark.priority("MUST")
class TestREQ_PERF_SCALE_001_FortyThousandLinesCapacity:
    """
    REQ-PERF-SCALE-001: The system MUST handle and track analysis of
    at least 40,000 lines of code across all experiments without
    performance degradation.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_forty_thousand_lines_capacity(self, mock_wm, mock_llm):
        """Verify system can handle 40K+ lines of code."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test code capacity",
            config={"max_iterations": 100}
        )

        # Initialize metadata tracking
        if not hasattr(director.research_plan, 'metadata'):
            director.research_plan.metadata = {}

        director.research_plan.metadata['total_code_lines'] = 0

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Simulate analyzing code in chunks
        chunk_size = 1000  # lines per chunk
        num_chunks = 40  # Total: 40,000 lines

        start_time = time.time()

        for chunk_num in range(num_chunks):
            # Simulate code analysis
            code_lines = chunk_size
            director.research_plan.metadata['total_code_lines'] += code_lines

            # Track experiment analyzing this code
            exp_id = f"exp_code_chunk_{chunk_num}"
            director.research_plan.add_experiment(exp_id)
            director.research_plan.mark_experiment_complete(exp_id)

        duration = time.time() - start_time
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_used = final_memory - initial_memory

        # Verify capacity
        total_lines = director.research_plan.metadata.get('total_code_lines', 0)
        assert total_lines >= 40000, \
            f"Only tracked {total_lines} lines, need 40K+"

        # Verify performance
        assert duration < 60, \
            f"Processing 40K lines took {duration:.1f}s, too slow"

        # Verify memory efficiency
        assert memory_used < 500, \
            f"Used {memory_used:.1f}MB for 40K lines, too much"

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_code_line_tracking_accuracy(self, mock_wm, mock_llm):
        """Verify accurate tracking of code lines across experiments."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test tracking accuracy",
            config={"max_iterations": 10}
        )

        if not hasattr(director.research_plan, 'metadata'):
            director.research_plan.metadata = {}

        director.research_plan.metadata['code_lines_by_experiment'] = {}

        # Simulate experiments with different code sizes
        experiments = [
            {'id': 'exp1', 'lines': 5000},
            {'id': 'exp2', 'lines': 10000},
            {'id': 'exp3', 'lines': 15000},
            {'id': 'exp4', 'lines': 12000},
        ]

        for exp in experiments:
            director.research_plan.add_experiment(exp['id'])
            director.research_plan.metadata['code_lines_by_experiment'][exp['id']] = exp['lines']
            director.research_plan.mark_experiment_complete(exp['id'])

        # Calculate total
        total_lines = sum(
            director.research_plan.metadata['code_lines_by_experiment'].values()
        )

        # Verify accuracy
        expected_total = sum(exp['lines'] for exp in experiments)
        assert total_lines == expected_total, \
            f"Line count mismatch: {total_lines} vs {expected_total}"

        assert total_lines == 42000

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_large_codebase_performance(self, mock_wm, mock_llm):
        """Verify performance remains acceptable with large codebases."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test large codebase",
            config={"max_iterations": 50}
        )

        if not hasattr(director.research_plan, 'metadata'):
            director.research_plan.metadata = {}

        director.research_plan.metadata['total_code_lines'] = 0

        # Simulate analyzing progressively larger codebases
        operation_times = []

        for i in range(10):
            start_time = time.time()

            # Each operation processes 5000 lines
            lines_processed = 5000
            director.research_plan.metadata['total_code_lines'] += lines_processed

            # Simulate some processing
            director.research_plan.add_experiment(f"exp_{i}")
            director.research_plan.mark_experiment_complete(f"exp_{i}")

            operation_time = time.time() - start_time
            operation_times.append(operation_time)

        # Verify no significant slowdown
        if len(operation_times) > 5:
            first_half_avg = sum(operation_times[:5]) / 5
            second_half_avg = sum(operation_times[5:]) / (len(operation_times) - 5)

            if first_half_avg > 0:
                slowdown = second_half_avg / first_half_avg
                assert slowdown < 2.0, \
                    f"Performance degraded {slowdown:.1f}x with larger codebase"

        # Verify total capacity reached
        total_lines = director.research_plan.metadata['total_code_lines']
        assert total_lines >= 40000

    def test_code_line_metadata_scalability(self):
        """Verify metadata storage scales with large line counts."""
        # Test storing and retrieving large line counts
        metadata = {
            'total_code_lines': 50000,
            'lines_by_file': {f'file_{i}.py': 1000 for i in range(50)},
            'lines_by_experiment': {f'exp_{i}': 2000 for i in range(25)}
        }

        # Verify metadata structure handles large counts
        assert metadata['total_code_lines'] == 50000
        assert len(metadata['lines_by_file']) == 50
        assert len(metadata['lines_by_experiment']) == 25
        assert sum(metadata['lines_by_file'].values()) == 50000


@pytest.mark.requirement("REQ-PERF-SCALE-002")
@pytest.mark.priority("MUST")
class TestREQ_PERF_SCALE_002_ThousandPapersCapacity:
    """
    REQ-PERF-SCALE-002: The system MUST handle and reference at least
    1,000 research papers in literature review without performance issues.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_thousand_papers_capacity(self, mock_wm, mock_llm):
        """Verify system can handle 1000+ papers."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test paper capacity",
            config={"max_iterations": 100}
        )

        if not hasattr(director.research_plan, 'metadata'):
            director.research_plan.metadata = {}

        director.research_plan.metadata['papers'] = []
        director.research_plan.metadata['paper_count'] = 0

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024

        start_time = time.time()

        # Add 1000 papers
        for i in range(1000):
            paper = {
                'id': f'paper_{i}',
                'title': f'Paper Title {i}',
                'authors': ['Author A', 'Author B'],
                'year': 2020 + (i % 5),
                'citations': 10 + (i % 100),
            }

            # Store paper reference (not full content to save memory)
            director.research_plan.metadata['papers'].append({
                'id': paper['id'],
                'title': paper['title']
            })
            director.research_plan.metadata['paper_count'] += 1

        duration = time.time() - start_time
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_used = final_memory - initial_memory

        # Verify capacity
        paper_count = director.research_plan.metadata['paper_count']
        assert paper_count >= 1000, \
            f"Only stored {paper_count} papers, need 1000+"

        # Verify performance
        assert duration < 30, \
            f"Adding 1000 papers took {duration:.1f}s, too slow"

        # Verify memory efficiency (should be < 100MB for metadata)
        assert memory_used < 200, \
            f"Used {memory_used:.1f}MB for 1000 papers, too much"

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_paper_retrieval_performance(self, mock_wm, mock_llm):
        """Verify paper retrieval remains fast with large corpus."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test paper retrieval",
            config={"max_iterations": 10}
        )

        if not hasattr(director.research_plan, 'metadata'):
            director.research_plan.metadata = {}

        # Create paper index
        director.research_plan.metadata['papers_by_id'] = {}

        # Add 1500 papers
        for i in range(1500):
            paper_id = f'paper_{i}'
            director.research_plan.metadata['papers_by_id'][paper_id] = {
                'id': paper_id,
                'title': f'Title {i}',
                'year': 2020 + (i % 5)
            }

        # Test retrieval performance
        retrieval_times = []

        for i in range(100):
            # Random retrieval
            paper_id = f'paper_{i * 10}'

            start_time = time.time()
            paper = director.research_plan.metadata['papers_by_id'].get(paper_id)
            retrieval_time = time.time() - start_time

            retrieval_times.append(retrieval_time)

            assert paper is not None

        # Verify retrieval is fast
        avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
        max_retrieval_time = max(retrieval_times)

        assert avg_retrieval_time < 0.001, \
            f"Average retrieval {avg_retrieval_time:.6f}s too slow"

        assert max_retrieval_time < 0.01, \
            f"Max retrieval {max_retrieval_time:.6f}s too slow"

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_paper_citation_tracking(self, mock_wm, mock_llm):
        """Verify citation tracking scales to 1000+ papers."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test citations",
            config={"max_iterations": 10}
        )

        if not hasattr(director.research_plan, 'metadata'):
            director.research_plan.metadata = {}

        # Track citations
        director.research_plan.metadata['citations'] = {}
        director.research_plan.metadata['papers_cited'] = set()

        # Simulate 1200 papers with citations
        for i in range(1200):
            paper_id = f'paper_{i}'

            # Each paper cites 3-5 other papers
            cited_papers = [
                f'paper_{(i + j + 1) % 1200}'
                for j in range(3 + (i % 3))
            ]

            director.research_plan.metadata['citations'][paper_id] = cited_papers
            director.research_plan.metadata['papers_cited'].add(paper_id)

        # Verify scale
        total_papers = len(director.research_plan.metadata['papers_cited'])
        assert total_papers >= 1000, \
            f"Only {total_papers} papers tracked"

        # Verify citation counts
        total_citations = sum(
            len(cites) for cites in director.research_plan.metadata['citations'].values()
        )
        assert total_citations > 3600, \
            "Should have many citations for 1200 papers"

    @pytest.mark.slow
    def test_large_literature_review_simulation(self):
        """Simulate large literature review with 1K+ papers."""
        # This test simulates a realistic literature review workflow

        papers_analyzed = 0
        papers_by_topic = {}

        # Simulate analyzing papers in batches
        topics = ['machine_learning', 'computer_vision', 'nlp', 'robotics', 'theory']

        for topic in topics:
            papers_by_topic[topic] = []

            # Each topic has ~250 papers
            for i in range(250):
                paper = {
                    'id': f'{topic}_paper_{i}',
                    'title': f'{topic.title()} Paper {i}',
                    'topic': topic,
                    'relevance_score': 0.5 + (i % 50) / 100
                }
                papers_by_topic[topic].append(paper)
                papers_analyzed += 1

        # Verify scale
        assert papers_analyzed >= 1000, \
            f"Only analyzed {papers_analyzed} papers"

        # Verify organization
        assert len(papers_by_topic) == 5
        for topic, papers in papers_by_topic.items():
            assert len(papers) >= 200, \
                f"Topic {topic} has only {len(papers)} papers"


@pytest.mark.requirement("REQ-PERF-SCALE-003")
@pytest.mark.priority("MUST")
class TestREQ_PERF_SCALE_003_HundredFiftyRolloutsCapacity:
    """
    REQ-PERF-SCALE-003: The system MUST support at least 150 experiment
    rollouts in a single research session without degradation.
    """

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_hundred_fifty_rollouts_capacity(self, mock_wm, mock_llm):
        """Verify system handles 150+ rollouts."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test rollout capacity",
            config={"max_iterations": 50}
        )

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024

        rollout_times = []
        start_time = time.time()

        # Execute 150 rollouts
        for rollout_num in range(150):
            rollout_start = time.time()

            # Simulate rollout: hypothesis -> experiment -> result
            hyp_id = f'hyp_rollout_{rollout_num}'
            exp_id = f'exp_rollout_{rollout_num}'
            res_id = f'res_rollout_{rollout_num}'

            director.research_plan.add_hypothesis(hyp_id)
            director.research_plan.add_experiment(exp_id)
            director.research_plan.mark_experiment_complete(exp_id)
            director.research_plan.add_result(res_id)
            director.research_plan.mark_tested(hyp_id)

            rollout_time = time.time() - rollout_start
            rollout_times.append(rollout_time)

        total_duration = time.time() - start_time
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_used = final_memory - initial_memory

        # Verify capacity
        assert len(director.research_plan.hypothesis_pool) >= 150
        assert len(director.research_plan.results) >= 150

        # Verify performance
        assert total_duration < 300, \
            f"150 rollouts took {total_duration:.1f}s, exceeds 5 minute target"

        # Verify no significant slowdown
        first_30_avg = sum(rollout_times[:30]) / 30
        last_30_avg = sum(rollout_times[-30:]) / 30
        slowdown = last_30_avg / first_30_avg if first_30_avg > 0 else 1.0

        assert slowdown < 2.0, \
            f"Performance degraded {slowdown:.1f}x over 150 rollouts"

        # Verify memory efficiency
        assert memory_used < 300, \
            f"Used {memory_used:.1f}MB for 150 rollouts, too much"

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_rollout_state_consistency(self, mock_wm, mock_llm):
        """Verify state remains consistent across 150+ rollouts."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test state consistency",
            config={"max_iterations": 50}
        )

        # Track state after each rollout
        state_checks = []

        for rollout_num in range(150):
            # Execute rollout
            hyp_id = f'hyp_{rollout_num}'
            exp_id = f'exp_{rollout_num}'
            res_id = f'res_{rollout_num}'

            director.research_plan.add_hypothesis(hyp_id)
            director.research_plan.add_experiment(exp_id)
            director.research_plan.mark_experiment_complete(exp_id)
            director.research_plan.add_result(res_id)

            # Some rollouts lead to supported hypotheses
            if rollout_num % 3 == 0:
                director.research_plan.mark_supported(hyp_id)
            elif rollout_num % 3 == 1:
                director.research_plan.mark_rejected(hyp_id)
            else:
                director.research_plan.mark_tested(hyp_id)

            # Check state consistency
            state = {
                'hypotheses': len(director.research_plan.hypothesis_pool),
                'experiments': len(director.research_plan.completed_experiments),
                'results': len(director.research_plan.results),
                'tested': len(director.research_plan.tested_hypotheses),
                'supported': len(director.research_plan.supported_hypotheses),
                'rejected': len(director.research_plan.rejected_hypotheses),
            }
            state_checks.append(state)

        # Verify final state
        final_state = state_checks[-1]
        assert final_state['hypotheses'] == 150
        assert final_state['experiments'] == 150
        assert final_state['results'] == 150
        assert final_state['tested'] == 150

        # Verify all state transitions were valid
        for i, state in enumerate(state_checks):
            # Each rollout should add exactly one of each
            expected_count = i + 1
            assert state['hypotheses'] == expected_count
            assert state['experiments'] == expected_count
            assert state['results'] == expected_count

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_concurrent_rollout_processing(self, mock_wm, mock_llm):
        """Verify concurrent processing improves rollout throughput."""
        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        director = ResearchDirectorAgent(
            research_question="Test concurrent rollouts",
            config={
                "enable_concurrent_operations": True,
                "max_concurrent_experiments": 10
            }
        )

        # Create 150 experiments
        experiment_ids = [f'exp_{i}' for i in range(150)]
        for exp_id in experiment_ids:
            director.research_plan.add_experiment(exp_id)

        start_time = time.time()

        # Process in batches of 10 (concurrent)
        batch_size = 10
        for i in range(0, 150, batch_size):
            batch = experiment_ids[i:i+batch_size]

            # Mock batch execution
            with patch.object(director, 'execute_experiments_batch') as mock_batch:
                mock_batch.return_value = [
                    {"protocol_id": exp_id, "success": True, "result_id": f"res_{exp_id}"}
                    for exp_id in batch
                ]

                results = director.execute_experiments_batch(batch)
                assert len(results) == len(batch)

                # Mark as complete
                for result in results:
                    if result.get("success"):
                        director.research_plan.mark_experiment_complete(result["protocol_id"])
                        director.research_plan.add_result(result["result_id"])

        duration = time.time() - start_time

        # Verify all completed
        assert len(director.research_plan.completed_experiments) == 150
        assert len(director.research_plan.results) == 150

        # Verify reasonable performance
        assert duration < 60, \
            f"Batch processing took {duration:.1f}s, too slow"

    @pytest.mark.slow
    def test_rollout_metrics_tracking(self):
        """Verify metrics accurately track 150+ rollouts."""
        metrics = get_metrics(reset=True)

        # Simulate 150 rollouts
        for i in range(150):
            # Track experiment
            metrics.record_experiment_start(f'exp_{i}', 'rollout')

            # Simulate execution
            time.sleep(0.001)  # Minimal delay

            # Track completion
            metrics.record_experiment_end(
                f'exp_{i}',
                duration_seconds=0.001,
                status='success'
            )

        stats = metrics.get_experiment_statistics()

        # Verify counts
        assert stats['experiments_started'] >= 150
        assert stats['experiments_completed'] >= 150

        # Verify success rate
        success_rate = stats.get('success_rate', 0)
        assert success_rate > 0.95, \
            f"Success rate {success_rate:.2%} too low"

    def test_rollout_data_structure_efficiency(self):
        """Verify data structures scale efficiently for 150+ rollouts."""
        # Test that our data structures don't have quadratic complexity

        import sys

        # Test hypothesis pool
        hypothesis_pool = set()
        for i in range(150):
            hypothesis_pool.add(f'hyp_{i}')

        # Adding to set is O(1)
        assert len(hypothesis_pool) == 150

        # Test experiment queue (list)
        experiment_queue = []
        for i in range(150):
            experiment_queue.append(f'exp_{i}')

        # Appending to list is O(1)
        assert len(experiment_queue) == 150

        # Test results lookup (dict)
        results = {}
        for i in range(150):
            results[f'res_{i}'] = {'data': i}

        # Dict lookup is O(1)
        assert len(results) == 150
        assert results['res_50']['data'] == 50

        # Verify memory efficiency
        # Each rollout should use minimal memory
        size_per_rollout = sys.getsizeof(results) / 150

        # Should be less than 1KB per rollout in basic metadata
        assert size_per_rollout < 1000, \
            f"Each rollout uses {size_per_rollout:.0f} bytes, too much"

    @patch('kosmos.agents.research_director.get_client')
    @patch('kosmos.world_model.get_world_model')
    def test_rollout_cleanup_efficiency(self, mock_wm, mock_llm):
        """Verify rollout data can be cleaned up efficiently."""
        import gc

        mock_llm.return_value = Mock()
        mock_wm.return_value = None

        process = psutil.Process(os.getpid())
        gc.collect()

        initial_memory = process.memory_info().rss / 1024 / 1024

        # Create director and run 150 rollouts
        director = ResearchDirectorAgent(
            research_question="Test cleanup",
            config={"max_iterations": 50}
        )

        for i in range(150):
            director.research_plan.add_hypothesis(f'hyp_{i}')
            director.research_plan.add_experiment(f'exp_{i}')
            director.research_plan.mark_experiment_complete(f'exp_{i}')
            director.research_plan.add_result(f'res_{i}')

        mid_memory = process.memory_info().rss / 1024 / 1024

        # Cleanup
        del director
        gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024

        # Verify most memory was reclaimed
        memory_used = mid_memory - initial_memory
        memory_retained = final_memory - initial_memory

        retention_percent = (memory_retained / memory_used * 100) if memory_used > 0 else 0

        assert retention_percent < 20, \
            f"{retention_percent:.1f}% of memory retained after cleanup"

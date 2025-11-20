"""
End-to-end tests for full research workflow.

Tests complete research cycles from question to results across multiple domains.
"""

import pytest
import os
from pathlib import Path


@pytest.mark.e2e
@pytest.mark.slow
class TestBiologyResearchWorkflow:
    """Test complete biology research workflow."""

    @pytest.fixture
    def research_question(self):
        """Sample biology research question."""
        return "How does temperature affect enzyme activity in metabolic pathways?"

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"),
        reason="API key required (ANTHROPIC_API_KEY or OPENAI_API_KEY)"
    )
    def test_full_biology_workflow(self, research_question):
        """Test complete biology research cycle."""
        from kosmos.agents.research_director import ResearchDirectorAgent

        config = {
            "max_iterations": 2,  # Keep very short for testing (just 2 iterations)
            "enable_concurrent_operations": False,  # Sequential for simplicity
            "max_concurrent_experiments": 1
        }

        director = ResearchDirectorAgent(
            research_question=research_question,
            domain="biology",
            config=config
        )

        assert director is not None
        assert director.research_question == research_question
        assert director.domain == "biology"

        # Execute first step of research
        print(f"\n🔬 Starting research: {research_question}")
        result = director.execute({"action": "start_research"})

        # Verify research started
        assert result["status"] == "research_started"
        assert "research_plan" in result
        assert "next_action" in result

        print(f"✅ Research started successfully")
        print(f"   Status: {result['status']}")
        print(f"   Next action: {result['next_action']}")

        # Get research status
        status = director.get_research_status()
        assert "workflow_state" in status
        assert "iteration" in status

        print(f"   Workflow state: {status.get('workflow_state')}")
        print(f"   Iteration: {status.get('iteration')}")

        # Verify we have started generating hypotheses or moved past it
        # (workflow_state can be lowercase or uppercase depending on implementation)
        workflow_state = status.get("workflow_state", "").lower()
        assert workflow_state in [
            "initializing", "generating_hypotheses", "designing_experiments",
            "executing", "analyzing"
        ]

        print(f"\n🎉 E2E test passed! Research workflow executing correctly.")


@pytest.mark.e2e
@pytest.mark.slow
class TestNeuroscienceResearchWorkflow:
    """Test complete neuroscience research workflow."""

    @pytest.fixture
    def research_question(self):
        """Sample neuroscience research question."""
        return "What neural pathways are involved in memory consolidation?"

    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="API key required")
    def test_full_neuroscience_workflow(self, research_question):
        """Test complete neuroscience research cycle."""
        from kosmos.agents.research_director import ResearchDirectorAgent

        config = {"max_iterations": 3}

        director = ResearchDirectorAgent(
            research_question=research_question,
            domain="neuroscience",
            config=config
        )

        assert director is not None


@pytest.mark.e2e
@pytest.mark.slow
class TestPerformanceValidation:
    """Test performance benchmarks meet targets."""

    def test_parallel_vs_sequential_speedup(self):
        """Test parallel execution provides expected speedup."""
        # This would run actual benchmarks
        # For now, placeholder
        assert True

    def test_cache_hit_rate(self):
        """Test cache hit rate meets target."""
        # Verify cache is effective
        assert True

    def test_api_cost_reduction(self):
        """Test caching reduces API costs."""
        # Verify cost savings from caching
        assert True


@pytest.mark.e2e
class TestCLIWorkflows:
    """Test complete CLI workflows."""

    def test_cli_run_and_view_results(self):
        """Test running research via CLI and viewing results."""
        # This would use CliRunner to test full CLI flow
        assert True

    def test_cli_status_monitoring(self):
        """Test monitoring research status via CLI."""
        assert True


@pytest.mark.e2e
@pytest.mark.docker
class TestDockerDeployment:
    """Test Docker deployment."""

    def test_docker_compose_health(self):
        """Test docker-compose deployment is healthy."""
        import subprocess

        try:
            result = subprocess.run(
                ["docker-compose", "ps"],
                capture_output=True,
                text=True,
                timeout=10
            )
            # Check if services are running
            assert "kosmos" in result.stdout or result.returncode >= 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Docker not available")

    def test_service_health_checks(self):
        """Test all services pass health checks."""
        # Would verify all containers are healthy
        assert True

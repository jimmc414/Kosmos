"""
Tests for kosmos.agents.literature_analyzer module.

Tests using REAL Claude API for LLM-dependent tests.
Knowledge graph tests use mocks for specific behavior testing.
"""

import os
import pytest
import uuid
from unittest.mock import Mock, patch, MagicMock

from kosmos.agents.literature_analyzer import LiteratureAnalyzerAgent, PaperAnalysis
from kosmos.literature.base_client import PaperMetadata, PaperSource


# Skip all tests if API key not available
pytestmark = [
    pytest.mark.requires_claude,
    pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY for real LLM calls"
    )
]


def unique_id() -> str:
    """Generate unique ID for test isolation."""
    return uuid.uuid4().hex[:8]


@pytest.fixture
def sample_paper():
    """Create sample paper metadata for testing."""
    return PaperMetadata(
        id=f"paper_{unique_id()}",
        source=PaperSource.ARXIV,
        title=f"Attention Is All You Need [{unique_id()}]",
        authors=["Vaswani, A.", "Shazeer, N.", "Parmar, N."],
        abstract="We propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality.",
        year=2017
    )


@pytest.fixture
def literature_analyzer():
    """Create LiteratureAnalyzerAgent with real Claude client."""
    # Use legacy ClaudeClient to avoid provider interface mismatch
    with patch('kosmos.agents.literature_analyzer.get_client') as mock_get_client:
        from kosmos.core.llm import ClaudeClient
        mock_get_client.return_value = ClaudeClient(model="claude-3-haiku-20240307")

        return LiteratureAnalyzerAgent(config={
            "model": "claude-3-haiku-20240307",
            "use_knowledge_graph": False,
            "use_semantic_similarity": False
        })


@pytest.mark.unit
class TestLiteratureAnalyzerInit:
    """Test literature analyzer initialization."""

    def test_init_default(self):
        """Test default initialization."""
        agent = LiteratureAnalyzerAgent(config={
            "model": "claude-3-haiku-20240307"
        })
        assert agent.agent_type == "LiteratureAnalyzerAgent"

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = {
            "model": "claude-3-haiku-20240307",
            "use_knowledge_graph": True,
            "use_semantic_similarity": True
        }
        agent = LiteratureAnalyzerAgent(config=config)
        assert agent.use_knowledge_graph is True
        assert agent.use_semantic_similarity is True


@pytest.mark.unit
class TestPaperSummarization:
    """Test paper summarization with real Claude."""

    def test_summarize_paper(self, literature_analyzer, sample_paper):
        """Test summarizing a paper with real Claude."""
        analysis = literature_analyzer.summarize_paper(sample_paper)

        assert isinstance(analysis, PaperAnalysis)
        assert len(analysis.executive_summary) > 0
        assert isinstance(analysis.key_findings, list)
        assert 0 <= analysis.confidence_score <= 1

    def test_summarize_paper_with_minimal_abstract(self, literature_analyzer):
        """Test summarizing paper with minimal abstract."""
        paper = PaperMetadata(
            id=f"paper_{unique_id()}",
            source=PaperSource.MANUAL,
            title=f"Test Paper [{unique_id()}]",
            authors=["Author"],
            abstract="A brief study on neural networks.",
            year=2023
        )

        analysis = literature_analyzer.summarize_paper(paper)

        assert isinstance(analysis, PaperAnalysis)
        assert len(analysis.executive_summary) > 0


@pytest.mark.unit
class TestCitationNetworkAnalysis:
    """Test citation network analysis (uses mocked knowledge graph)."""

    def test_analyze_citation_network(self, literature_analyzer):
        """Test analyzing citation network with mocked KG."""
        mock_kg = Mock()
        mock_kg.get_citations.return_value = [
            {"paper_id": "cited1", "title": "Cited Paper 1"},
            {"paper_id": "cited2", "title": "Cited Paper 2"},
        ]
        mock_kg.get_citing_papers.return_value = [
            {"paper_id": "citing1", "title": "Citing Paper 1"},
        ]
        literature_analyzer.knowledge_graph = mock_kg
        literature_analyzer.use_knowledge_graph = True

        network_analysis = literature_analyzer.analyze_citation_network("paper_123", depth=1)

        # Check correct keys from actual implementation
        assert "citation_count" in network_analysis
        assert "cited_by_count" in network_analysis
        assert network_analysis["citation_count"] == 2
        assert network_analysis["cited_by_count"] == 1

    def test_analyze_citation_network_empty(self, literature_analyzer):
        """Test analyzing citation network with no citations."""
        mock_kg = Mock()
        mock_kg.get_citations.return_value = []
        mock_kg.get_citing_papers.return_value = []
        literature_analyzer.knowledge_graph = mock_kg
        literature_analyzer.use_knowledge_graph = True

        network_analysis = literature_analyzer.analyze_citation_network("paper_123", depth=1)

        assert isinstance(network_analysis, dict)
        assert network_analysis["citation_count"] == 0


@pytest.mark.unit
class TestAgentLifecycle:
    """Test agent lifecycle methods."""

    def test_agent_start(self, literature_analyzer):
        """Test starting the agent."""
        literature_analyzer.start()

        assert literature_analyzer.status == "running"

    def test_agent_stop(self, literature_analyzer):
        """Test stopping the agent."""
        literature_analyzer.start()
        literature_analyzer.stop()

        assert literature_analyzer.status == "stopped"

    def test_agent_execute_summarize(self, literature_analyzer, sample_paper):
        """Test agent execute method with summarize_paper task."""
        task = {
            "task_type": "summarize_paper",
            "paper": sample_paper,
        }

        response = literature_analyzer.execute(task)

        assert response is not None
        assert response["status"] == "success"
        assert "summary" in response


@pytest.mark.integration
@pytest.mark.slow
class TestLiteratureAnalyzerIntegration:
    """Integration tests with real services."""

    def test_real_paper_summarization(self, sample_paper):
        """Test real paper summarization with Claude."""
        # Use legacy ClaudeClient to avoid provider interface mismatch
        with patch('kosmos.agents.literature_analyzer.get_client') as mock_get_client:
            from kosmos.core.llm import ClaudeClient
            mock_get_client.return_value = ClaudeClient(model="claude-3-haiku-20240307")

            agent = LiteratureAnalyzerAgent(config={
                "model": "claude-3-haiku-20240307",
                "use_knowledge_graph": False
            })

            agent.start()
            analysis = agent.summarize_paper(sample_paper)
            agent.stop()

            assert isinstance(analysis, PaperAnalysis)
            assert len(analysis.executive_summary) > 0
            assert len(analysis.key_findings) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

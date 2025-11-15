"""
Shared pytest fixtures for Kosmos tests.

This module provides common fixtures used across all test suites.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, Mock

import pytest

from kosmos.literature.base_client import PaperMetadata


# ============================================================================
# Path and File Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_papers_json(fixtures_dir: Path) -> Path:
    """Return path to sample papers JSON file."""
    return fixtures_dir / "sample_papers.json"


@pytest.fixture(scope="session")
def sample_arxiv_xml(fixtures_dir: Path) -> Path:
    """Return path to sample arXiv XML response."""
    return fixtures_dir / "sample_arxiv_response.xml"


@pytest.fixture(scope="session")
def sample_semantic_scholar_json(fixtures_dir: Path) -> Path:
    """Return path to sample Semantic Scholar JSON response."""
    return fixtures_dir / "sample_semantic_scholar_response.json"


@pytest.fixture(scope="session")
def sample_pubmed_xml(fixtures_dir: Path) -> Path:
    """Return path to sample PubMed XML response."""
    return fixtures_dir / "sample_pubmed_response.xml"


@pytest.fixture(scope="session")
def sample_bibtex(fixtures_dir: Path) -> Path:
    """Return path to sample BibTeX file."""
    return fixtures_dir / "sample.bib"


@pytest.fixture(scope="session")
def sample_ris(fixtures_dir: Path) -> Path:
    """Return path to sample RIS file."""
    return fixtures_dir / "sample.ris"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file for tests."""
    def _create_temp_file(filename: str, content: str = "") -> Path:
        file_path = temp_dir / filename
        file_path.write_text(content)
        return file_path
    return _create_temp_file


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def sample_papers_data(sample_papers_json: Path) -> List[Dict]:
    """Load sample papers data from JSON."""
    with open(sample_papers_json) as f:
        data = json.load(f)
    return data["papers"]


@pytest.fixture
def sample_paper_metadata(sample_papers_data: List[Dict]) -> PaperMetadata:
    """Return a single sample PaperMetadata object."""
    paper_dict = sample_papers_data[0]  # "Attention Is All You Need"
    return PaperMetadata(
        title=paper_dict["title"],
        authors=paper_dict["authors"],
        abstract=paper_dict["abstract"],
        year=paper_dict["year"],
        venue=paper_dict.get("venue"),
        doi=paper_dict.get("doi"),
        arxiv_id=paper_dict.get("arxiv_id"),
        pubmed_id=paper_dict.get("pubmed_id"),
        url=paper_dict.get("url"),
        pdf_url=paper_dict.get("pdf_url"),
        citation_count=paper_dict.get("citation_count", 0),
        source=paper_dict.get("source", "unknown"),
    )


@pytest.fixture
def sample_papers_list(sample_papers_data: List[Dict]) -> List[PaperMetadata]:
    """Return a list of sample PaperMetadata objects."""
    papers = []
    for paper_dict in sample_papers_data:
        papers.append(
            PaperMetadata(
                title=paper_dict["title"],
                authors=paper_dict["authors"],
                abstract=paper_dict["abstract"],
                year=paper_dict["year"],
                venue=paper_dict.get("venue"),
                doi=paper_dict.get("doi"),
                arxiv_id=paper_dict.get("arxiv_id"),
                pubmed_id=paper_dict.get("pubmed_id"),
                url=paper_dict.get("url"),
                pdf_url=paper_dict.get("pdf_url"),
                citation_count=paper_dict.get("citation_count", 0),
                source=paper_dict.get("source", "unknown"),
            )
        )
    return papers


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_client():
    """Mock Claude LLM client."""
    mock = Mock()
    mock.generate.return_value = "Mocked Claude response"
    mock.generate_structured.return_value = {
        "executive_summary": "This is a summary.",
        "key_findings": ["Finding 1", "Finding 2"],
        "methodology": "The methodology used.",
        "significance": "Significance of the work.",
        "limitations": ["Limitation 1"],
        "confidence_score": 0.85,
    }
    return mock


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic API client."""
    mock = Mock()
    mock.messages.create.return_value = Mock(
        content=[Mock(text="Mocked API response")],
        usage=Mock(input_tokens=100, output_tokens=50),
    )
    return mock


@pytest.fixture
def mock_knowledge_graph():
    """Mock knowledge graph."""
    mock = Mock()
    mock.add_paper.return_value = "paper_id_123"
    mock.add_concept.return_value = "concept_id_456"
    mock.add_citation.return_value = "citation_id_789"
    mock.get_citations.return_value = []
    mock.get_concept_papers.return_value = []
    mock.get_stats.return_value = {
        "total_papers": 100,
        "total_concepts": 50,
        "total_citations": 200,
    }
    return mock


@pytest.fixture
def mock_vector_db():
    """Mock vector database."""
    mock = Mock()
    mock.add_papers.return_value = None
    mock.search.return_value = []
    mock.get_paper_count.return_value = 0
    return mock


@pytest.fixture
def mock_concept_extractor():
    """Mock concept extractor."""
    mock = Mock()
    mock.extract_from_paper.return_value = Mock(
        paper_id="paper_123",
        concepts=[
            Mock(name="Machine Learning", category="Method", relevance=0.9),
            Mock(name="Neural Networks", category="Concept", relevance=0.85),
        ],
        methods=[
            Mock(name="Transformer Architecture", category="Architecture", relevance=0.95),
        ],
        relationships=[],
        extraction_time=1.5,
        model_used="claude-sonnet-4.5",
    )
    return mock


@pytest.fixture
def mock_cache():
    """Mock cache with simple dict-based storage."""
    cache_dict = {}

    class MockCache:
        def get(self, key):
            return cache_dict.get(key)

        def set(self, key, value, ttl=None):
            cache_dict[key] = value

        def delete(self, key):
            if key in cache_dict:
                del cache_dict[key]

        def clear(self):
            cache_dict.clear()

        def exists(self, key):
            return key in cache_dict

    return MockCache()


# ============================================================================
# API Response Fixtures
# ============================================================================

@pytest.fixture
def arxiv_response_xml(sample_arxiv_xml: Path) -> str:
    """Load sample arXiv XML response."""
    return sample_arxiv_xml.read_text()


@pytest.fixture
def semantic_scholar_response_json(sample_semantic_scholar_json: Path) -> Dict:
    """Load sample Semantic Scholar JSON response."""
    return json.loads(sample_semantic_scholar_json.read_text())


@pytest.fixture
def pubmed_response_xml(sample_pubmed_xml: Path) -> str:
    """Load sample PubMed XML response."""
    return sample_pubmed_xml.read_text()


# ============================================================================
# Environment and Configuration Fixtures
# ============================================================================

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    env_vars = {
        "ANTHROPIC_API_KEY": "test_api_key",
        "SEMANTIC_SCHOLAR_API_KEY": "test_s2_key",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "test_password",
        "CHROMA_PERSIST_DIR": ".chroma_test_db",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "model": "claude-sonnet-4-5",
        "max_tokens": 1024,
        "temperature": 0.7,
        "cache_ttl": 3600,
        "use_knowledge_graph": False,
        "use_semantic_similarity": False,
    }


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset all singleton instances before each test."""
    # This ensures tests don't interfere with each other
    yield
    # Reset singletons after test
    from kosmos.knowledge.graph import reset_knowledge_graph
    from kosmos.knowledge.vector_db import reset_vector_db
    from kosmos.knowledge.embeddings import reset_embedder
    from kosmos.knowledge.concept_extractor import reset_concept_extractor
    from kosmos.literature.reference_manager import reset_reference_manager
    from kosmos.world_model.factory import reset_world_model

    try:
        reset_knowledge_graph()
        reset_vector_db()
        reset_embedder()
        reset_concept_extractor()
        reset_reference_manager()
        reset_world_model()
    except Exception:
        # Silently ignore errors during cleanup
        pass


@pytest.fixture(autouse=True)
def cleanup_test_files(temp_dir):
    """Clean up any test files created during tests."""
    yield
    # Cleanup happens automatically with temp_dir


# ============================================================================
# Marker-based Skipping
# ============================================================================

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow"
    )
    config.addinivalue_line(
        "markers", "requires_api_key: mark test as requiring API keys"
    )
    config.addinivalue_line(
        "markers", "requires_neo4j: mark test as requiring Neo4j"
    )
    config.addinivalue_line(
        "markers", "requires_chromadb: mark test as requiring ChromaDB"
    )
    config.addinivalue_line(
        "markers", "requires_claude: mark test as requiring Claude API"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests based on markers if dependencies not available."""
    skip_api_key = pytest.mark.skip(reason="API keys not configured")
    skip_neo4j = pytest.mark.skip(reason="Neo4j not available")
    skip_chromadb = pytest.mark.skip(reason="ChromaDB not available")
    skip_claude = pytest.mark.skip(reason="Claude API not configured")

    # Check environment
    has_api_keys = os.getenv("ANTHROPIC_API_KEY") and os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    has_neo4j = os.getenv("NEO4J_URI")
    has_claude = os.getenv("ANTHROPIC_API_KEY")

    for item in items:
        if "requires_api_key" in item.keywords and not has_api_keys:
            item.add_marker(skip_api_key)
        if "requires_neo4j" in item.keywords and not has_neo4j:
            item.add_marker(skip_neo4j)
        if "requires_claude" in item.keywords and not has_claude:
            item.add_marker(skip_claude)

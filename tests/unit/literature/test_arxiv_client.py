"""
Tests for kosmos.literature.arxiv_client module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from kosmos.literature.arxiv_client import ArxivClient
from kosmos.literature.base_client import PaperMetadata, PaperSource


@pytest.fixture
def mock_config():
    """Mock config for consistent max_results."""
    mock_cfg = Mock()
    mock_cfg.literature.max_results_per_query = 100
    return mock_cfg


@pytest.fixture
def arxiv_client(mock_config):
    """Create ArxivClient instance for testing."""
    with patch('kosmos.literature.arxiv_client.get_config', return_value=mock_config):
        with patch('kosmos.literature.arxiv_client.get_cache') as mock_cache:
            mock_cache.return_value = None  # Disable caching for tests
            return ArxivClient(cache_enabled=False)


@pytest.fixture
def mock_arxiv_result():
    """Create a mock arxiv.Result object."""
    mock_result = Mock()
    mock_result.entry_id = "http://arxiv.org/abs/1706.03762v5"
    mock_result.title = "Attention Is All You Need"
    mock_result.summary = "We propose the Transformer architecture..."
    # Create mock authors with string name attribute
    author1 = Mock()
    author1.name = "Ashish Vaswani"
    author2 = Mock()
    author2.name = "Noam Shazeer"
    mock_result.authors = [author1, author2]
    mock_result.published = Mock(year=2017)
    mock_result.updated = Mock(isoformat=Mock(return_value="2017-06-12T00:00:00"))
    mock_result.journal_ref = "NeurIPS 2017"
    mock_result.doi = "10.5555/3295222.3295349"
    mock_result.pdf_url = "http://arxiv.org/pdf/1706.03762v5"
    mock_result.primary_category = "cs.CL"
    mock_result.categories = ["cs.CL", "cs.LG"]
    mock_result.comment = "Test comment"
    return mock_result


@pytest.mark.unit
class TestArxivClientInit:
    """Test ArxivClient initialization."""

    def test_init_default(self, mock_config):
        """Test default initialization."""
        with patch('kosmos.literature.arxiv_client.get_config', return_value=mock_config):
            with patch('kosmos.literature.arxiv_client.get_cache'):
                client = ArxivClient()
                assert client.max_results == 100  # From config

    def test_init_with_cache_disabled(self, mock_config):
        """Test initialization with cache disabled."""
        with patch('kosmos.literature.arxiv_client.get_config', return_value=mock_config):
            with patch('kosmos.literature.arxiv_client.get_cache') as mock_cache:
                client = ArxivClient(cache_enabled=False)
                mock_cache.assert_not_called()


@pytest.mark.unit
class TestArxivSearch:
    """Test arXiv search functionality."""

    def test_search_success(self, arxiv_client, mock_arxiv_result):
        """Test successful paper search."""
        with patch.object(arxiv_client, 'client') as mock_client:
            mock_client.results.return_value = [mock_arxiv_result]

            papers = arxiv_client.search("attention mechanism", max_results=1)

            assert len(papers) == 1
            assert isinstance(papers[0], PaperMetadata)
            assert papers[0].title == "Attention Is All You Need"
            assert papers[0].arxiv_id == "1706.03762"
            assert papers[0].year == 2017
            assert papers[0].source == PaperSource.ARXIV

    def test_search_empty_results(self, arxiv_client):
        """Test search with no results."""
        with patch.object(arxiv_client, 'client') as mock_client:
            mock_client.results.return_value = []

            papers = arxiv_client.search("nonexistent_query_xyz123")
            assert papers == []

    def test_search_error_handling(self, arxiv_client):
        """Clients raise; UnifiedLiteratureSearch isolates per source."""
        with patch.object(arxiv_client, 'client') as mock_client:
            mock_client.results.side_effect = Exception("API Error")

            with pytest.raises(Exception, match="API Error"):
                arxiv_client.search("test query")


@pytest.mark.unit
class TestArxivGetPaperById:
    """Test fetching papers by arXiv ID."""

    def test_get_paper_by_id_success(self, arxiv_client, mock_arxiv_result):
        """Test successfully fetching a paper by ID."""
        with patch.object(arxiv_client, 'client') as mock_client:
            mock_client.results.return_value = iter([mock_arxiv_result])

            paper = arxiv_client.get_paper_by_id("1706.03762")

            assert paper is not None
            assert paper.arxiv_id == "1706.03762"
            assert paper.title == "Attention Is All You Need"

    def test_get_paper_by_id_not_found(self, arxiv_client):
        """Test fetching a non-existent paper ID."""
        with patch.object(arxiv_client, 'client') as mock_client:
            mock_client.results.return_value = iter([])

            paper = arxiv_client.get_paper_by_id("9999.99999")
            assert paper is None

    def test_get_paper_by_id_strips_prefix(self, arxiv_client, mock_arxiv_result):
        """Test that arXiv: prefix is stripped from ID."""
        with patch.object(arxiv_client, 'client') as mock_client:
            mock_client.results.return_value = iter([mock_arxiv_result])

            paper = arxiv_client.get_paper_by_id("arXiv:1706.03762")

            assert paper is not None
            assert paper.arxiv_id == "1706.03762"


@pytest.mark.unit
class TestArxivMetadataConversion:
    """Test converting arXiv results to PaperMetadata."""

    def test_arxiv_to_metadata_complete(self, arxiv_client, mock_arxiv_result):
        """Test converting a complete arXiv result."""
        paper = arxiv_client._arxiv_to_metadata(mock_arxiv_result)

        assert paper.title == "Attention Is All You Need"
        assert paper.arxiv_id == "1706.03762"
        assert len(paper.authors) == 2
        assert paper.authors[0].name == "Ashish Vaswani"
        assert paper.year == 2017
        assert paper.journal == "NeurIPS 2017"
        assert paper.doi == "10.5555/3295222.3295349"
        assert paper.source == PaperSource.ARXIV
        assert "cs.cl" in paper.fields
        assert "cs.lg" in paper.fields

    def test_arxiv_to_metadata_minimal(self, arxiv_client):
        """Test converting a minimal arXiv result."""
        mock_result = Mock()
        mock_result.entry_id = "http://arxiv.org/abs/2301.00000"
        mock_result.title = "Minimal Paper"
        mock_result.summary = "A minimal paper."
        mock_result.authors = [Mock(name="Anonymous")]
        mock_result.published = Mock(year=2023)
        mock_result.updated = None
        mock_result.journal_ref = None
        mock_result.doi = None
        mock_result.pdf_url = "http://arxiv.org/pdf/2301.00000"
        mock_result.primary_category = "cs.AI"
        mock_result.categories = ["cs.AI"]
        mock_result.comment = None

        paper = arxiv_client._arxiv_to_metadata(mock_result)

        assert paper.title == "Minimal Paper"
        assert paper.arxiv_id == "2301.00000"
        assert paper.journal is None
        assert paper.doi is None
        assert paper.source == PaperSource.ARXIV


@pytest.mark.unit
class TestArxivBuildQuery:
    """Test query building functionality."""

    def test_build_query_simple(self, arxiv_client):
        """Test building a simple query."""
        query = arxiv_client._build_query("machine learning", None, None, None)
        assert "machine learning" in query

    def test_build_query_with_fields(self, arxiv_client):
        """Test building query with category filters."""
        query = arxiv_client._build_query("neural networks", ["cs.AI", "cs.LG"], None, None)
        assert "neural networks" in query
        assert "cat:cs.AI" in query
        assert "cat:cs.LG" in query


@pytest.mark.unit
class TestArxivCitations:
    """Test citation methods (which return empty for arXiv)."""

    def test_get_references_returns_empty(self, arxiv_client):
        """Test that get_paper_references returns empty list."""
        refs = arxiv_client.get_paper_references("1706.03762")
        assert refs == []

    def test_get_citations_returns_empty(self, arxiv_client):
        """Test that get_paper_citations returns empty list."""
        cites = arxiv_client.get_paper_citations("1706.03762")
        assert cites == []


@pytest.mark.unit
class TestArxivCategories:
    """Test category functionality."""

    def test_get_categories(self, arxiv_client):
        """Test getting arXiv categories."""
        categories = arxiv_client.get_categories()
        assert isinstance(categories, list)
        assert "cs.AI" in categories
        assert "cs.LG" in categories
        assert "cs.CL" in categories


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_network
class TestArxivClientIntegration:
    """Integration tests for ArxivClient (requires network)."""

    def test_real_arxiv_search(self):
        """Test real arXiv API search (requires network)."""
        client = ArxivClient()
        papers = client.search("transformer neural network", max_results=2)

        assert len(papers) > 0
        assert all(isinstance(p, PaperMetadata) for p in papers)
        assert all(p.arxiv_id is not None for p in papers)

    def test_real_get_paper_by_id(self):
        """Test fetching a real paper by ID."""
        client = ArxivClient()
        paper = client.get_paper_by_id("1706.03762")  # "Attention Is All You Need"

        assert paper is not None
        assert paper.arxiv_id == "1706.03762"
        assert "Attention" in paper.title
        assert paper.year == 2017

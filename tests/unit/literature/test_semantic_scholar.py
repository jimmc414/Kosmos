"""
Tests for kosmos.literature.semantic_scholar module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from kosmos.literature.semantic_scholar import SemanticScholarClient
from kosmos.literature.base_client import PaperMetadata, PaperSource


@pytest.fixture
def mock_config():
    """Mock config for consistent settings."""
    mock_cfg = Mock()
    mock_cfg.literature.max_results_per_query = 100
    mock_cfg.literature.semantic_scholar_api_key = None
    return mock_cfg


@pytest.fixture
def s2_client(mock_config):
    """Create SemanticScholarClient instance for testing."""
    with patch('kosmos.literature.semantic_scholar.get_config', return_value=mock_config):
        with patch('kosmos.literature.semantic_scholar.get_cache') as mock_cache:
            mock_cache.return_value = None  # Disable caching for tests
            with patch('kosmos.literature.semantic_scholar.SemanticScholar'):
                return SemanticScholarClient(api_key="test_api_key", cache_enabled=False)


@pytest.fixture
def mock_s2_paper():
    """Create a mock S2Paper object."""
    paper = Mock()
    paper.paperId = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    paper.title = "Attention Is All You Need"
    paper.abstract = "We propose the Transformer..."
    paper.year = 2017
    paper.publicationDate = "2017-06-12"
    paper.venue = "NeurIPS"
    paper.journal = {"name": "NeurIPS 2017"}
    paper.url = "https://semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    paper.citationCount = 98765
    paper.referenceCount = 42
    paper.influentialCitationCount = 5000
    paper.fieldsOfStudy = ["Computer Science", "Machine Learning"]
    paper.externalIds = {
        "ArXiv": "1706.03762",
        "DOI": "10.5555/3295222.3295349"
    }
    # Mock authors
    author1 = Mock()
    author1.name = "Ashish Vaswani"
    author1.authorId = "1234"
    author2 = Mock()
    author2.name = "Noam Shazeer"
    author2.authorId = "5678"
    paper.authors = [author1, author2]
    # Mock open access PDF
    paper.openAccessPdf = Mock(url="https://arxiv.org/pdf/1706.03762.pdf")
    return paper


@pytest.mark.unit
class TestSemanticScholarInit:
    """Test Semantic Scholar client initialization."""

    def test_init_with_api_key(self, mock_config):
        """Test initialization with API key."""
        with patch('kosmos.literature.semantic_scholar.get_config', return_value=mock_config):
            with patch('kosmos.literature.semantic_scholar.get_cache'):
                with patch('kosmos.literature.semantic_scholar.SemanticScholar'):
                    client = SemanticScholarClient(api_key="test_key")
                    assert client.api_key == "test_key"

    def test_init_without_api_key(self, mock_config):
        """Test initialization without API key."""
        with patch('kosmos.literature.semantic_scholar.get_config', return_value=mock_config):
            with patch('kosmos.literature.semantic_scholar.get_cache'):
                with patch('kosmos.literature.semantic_scholar.SemanticScholar'):
                    client = SemanticScholarClient()
                    assert client.api_key is None


@pytest.mark.unit
class TestSemanticScholarSearch:
    """Test Semantic Scholar search functionality."""

    def test_search_success(self, s2_client, mock_s2_paper):
        """Test successful paper search."""
        s2_client.client.search_paper.return_value = [mock_s2_paper]

        papers = s2_client.search("attention mechanism", max_results=3)

        assert len(papers) == 1
        assert isinstance(papers[0], PaperMetadata)
        assert papers[0].title == "Attention Is All You Need"
        assert papers[0].source == PaperSource.SEMANTIC_SCHOLAR

    def test_search_empty_results(self, s2_client):
        """Test search with no results."""
        s2_client.client.search_paper.return_value = []

        papers = s2_client.search("nonexistent_query_xyz")
        assert papers == []

    def test_search_with_api_error(self, s2_client):
        """Test search with API error: the client raises."""
        s2_client.client.search_paper.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            s2_client.search("test query")


@pytest.mark.unit
class TestSemanticScholarGetPaper:
    """Test fetching papers by ID."""

    def test_get_paper_by_id_success(self, s2_client, mock_s2_paper):
        """Test fetching paper by Semantic Scholar ID."""
        s2_client.client.get_paper.return_value = mock_s2_paper

        paper = s2_client.get_paper_by_id(mock_s2_paper.paperId)

        assert paper is not None
        assert paper.title == "Attention Is All You Need"
        assert paper.arxiv_id == "1706.03762"
        assert paper.doi == "10.5555/3295222.3295349"

    def test_get_paper_by_id_not_found(self, s2_client):
        """Test fetching non-existent paper."""
        s2_client.client.get_paper.return_value = None

        paper = s2_client.get_paper_by_id("nonexistent")
        assert paper is None


@pytest.mark.unit
class TestSemanticScholarCitations:
    """Test citation fetching functionality."""

    def test_get_citations_success(self, s2_client, mock_s2_paper):
        """Test fetching citations for a paper."""
        paper_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

        # Create mock citation result
        mock_citation = Mock()
        mock_citation.citingPaper = mock_s2_paper

        s2_client.client.get_paper_citations.return_value = [mock_citation]

        citations = s2_client.get_paper_citations(paper_id, max_cites=10)

        assert len(citations) == 1
        assert citations[0].title == "Attention Is All You Need"

    def test_get_references_success(self, s2_client, mock_s2_paper):
        """Test fetching references for a paper."""
        paper_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

        # Create mock reference result
        mock_reference = Mock()
        mock_reference.citedPaper = mock_s2_paper

        s2_client.client.get_paper_references.return_value = [mock_reference]

        references = s2_client.get_paper_references(paper_id, max_refs=10)

        assert len(references) == 1
        assert references[0].title == "Attention Is All You Need"


@pytest.mark.unit
class TestSemanticScholarMetadataConversion:
    """Test converting S2Paper to PaperMetadata."""

    def test_s2_to_metadata_complete(self, s2_client, mock_s2_paper):
        """Test converting a complete S2Paper."""
        paper = s2_client._s2_to_metadata(mock_s2_paper)

        assert paper.title == "Attention Is All You Need"
        assert paper.id == mock_s2_paper.paperId
        assert len(paper.authors) == 2
        assert paper.authors[0].name == "Ashish Vaswani"
        assert paper.arxiv_id == "1706.03762"
        assert paper.doi == "10.5555/3295222.3295349"
        assert paper.citation_count == 98765
        assert paper.source == PaperSource.SEMANTIC_SCHOLAR
        assert "computer science" in paper.fields

    def test_s2_to_metadata_minimal(self, s2_client):
        """Test converting a minimal S2Paper."""
        mock_paper = Mock()
        mock_paper.paperId = "456"
        mock_paper.title = "Minimal Paper"
        mock_paper.abstract = None
        mock_paper.year = 2023
        mock_paper.publicationDate = None
        mock_paper.venue = None
        mock_paper.journal = None
        mock_paper.url = None
        mock_paper.citationCount = None
        mock_paper.referenceCount = None
        mock_paper.influentialCitationCount = None
        mock_paper.fieldsOfStudy = None
        mock_paper.externalIds = None
        mock_paper.authors = None
        mock_paper.openAccessPdf = None

        paper = s2_client._s2_to_metadata(mock_paper)

        assert paper.title == "Minimal Paper"
        assert paper.year == 2023
        assert paper.abstract == ""
        assert paper.authors == []
        assert paper.citation_count == 0


@pytest.mark.unit
class TestSemanticScholarErrorHandling:
    """Test error handling."""

    def test_search_error_handling(self, s2_client):
        """Errors during search propagate (UnifiedLiteratureSearch isolates per source)."""
        s2_client.client.search_paper.side_effect = Exception("Network error")

        with pytest.raises(Exception, match="Network error"):
            s2_client.search("test query")

    def test_get_paper_error_handling(self, s2_client):
        """Errors during get_paper propagate."""
        s2_client.client.get_paper.side_effect = Exception("Network error")

        with pytest.raises(Exception, match="Network error"):
            s2_client.get_paper_by_id("test_id")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_api_key
@pytest.mark.requires_network
class TestSemanticScholarIntegration:
    """Integration tests (requires API key and network)."""

    def test_real_search(self):
        """Test real Semantic Scholar search."""
        client = SemanticScholarClient()
        papers = client.search("machine learning", max_results=2)

        assert len(papers) > 0
        assert all(isinstance(p, PaperMetadata) for p in papers)

    def test_real_get_paper(self):
        """Test fetching a real paper."""
        client = SemanticScholarClient()
        paper = client.get_paper_by_id("204e3073870fae3d05bcbc2f6a8e263d9b72e776")

        assert paper is not None
        assert "Attention" in paper.title

"""
Unit tests for ArxivHTTPClient.

Tests the Python 3.11+ compatible HTTP-based arXiv client.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Mock config before import
@pytest.fixture(autouse=True)
def mock_config():
    """Mock configuration for tests."""
    with patch('kosmos.literature.arxiv_http_client.get_config') as mock_get_config:
        mock_config = MagicMock()
        mock_config.literature.max_results_per_query = 100
        mock_get_config.return_value = mock_config
        yield mock_config


@pytest.fixture
def mock_cache():
    """Mock cache for tests."""
    with patch('kosmos.literature.arxiv_http_client.get_cache') as mock_get_cache:
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_get_cache.return_value = mock_cache
        yield mock_cache


class TestArxivHTTPClient:
    """Tests for ArxivHTTPClient."""

    def test_import(self):
        """Test that ArxivHTTPClient can be imported."""
        from kosmos.literature.arxiv_http_client import ArxivHTTPClient
        assert ArxivHTTPClient is not None

    def test_client_initialization(self, mock_cache):
        """Test client initialization."""
        from kosmos.literature.arxiv_http_client import ArxivHTTPClient

        client = ArxivHTTPClient(cache_enabled=False)
        assert client is not None
        assert client.http_client is not None
        client.close()

    def test_build_search_query_simple(self, mock_cache):
        """Test simple query building."""
        from kosmos.literature.arxiv_http_client import ArxivHTTPClient

        client = ArxivHTTPClient(cache_enabled=False)

        # Simple query without field prefix
        query = client._build_search_query("machine learning")
        assert "all:machine learning" in query

        client.close()

    def test_build_search_query_with_fields(self, mock_cache):
        """Test query building with category filters."""
        from kosmos.literature.arxiv_http_client import ArxivHTTPClient

        client = ArxivHTTPClient(cache_enabled=False)

        # Query with category filters
        query = client._build_search_query("neural network", fields=["cs.LG", "cs.AI"])
        assert "all:neural network" in query
        assert "cat:cs.LG" in query
        assert "cat:cs.AI" in query

        client.close()

    def test_normalize_arxiv_id(self, mock_cache):
        """Test arXiv ID normalization."""
        from kosmos.literature.arxiv_http_client import ArxivHTTPClient

        client = ArxivHTTPClient(cache_enabled=False)

        # Test various ID formats
        assert client._normalize_arxiv_id("2103.00020") == "2103.00020"
        assert client._normalize_arxiv_id("arXiv:2103.00020") == "2103.00020"
        assert client._normalize_arxiv_id("2103.00020v2") == "2103.00020"
        assert client._normalize_arxiv_id("hep-th/9901001") == "hep-th/9901001"

        client.close()

    def test_parse_datetime(self, mock_cache):
        """Test datetime parsing."""
        from kosmos.literature.arxiv_http_client import ArxivHTTPClient

        client = ArxivHTTPClient(cache_enabled=False)

        # Valid ISO format
        dt = client._parse_datetime("2023-01-15T10:30:00Z")
        assert dt.year == 2023
        assert dt.month == 1
        assert dt.day == 15

        # Empty string
        dt = client._parse_datetime("")
        assert dt.year == 1970

        client.close()

    def test_parse_atom_feed_empty(self, mock_cache):
        """Test parsing empty atom feed."""
        from kosmos.literature.arxiv_http_client import ArxivHTTPClient

        client = ArxivHTTPClient(cache_enabled=False)

        empty_feed = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
        </feed>"""

        results = client._parse_atom_feed(empty_feed)
        assert results == []

        client.close()

    def test_parse_atom_feed_with_entry(self, mock_cache):
        """Test parsing atom feed with an entry."""
        from kosmos.literature.arxiv_http_client import ArxivHTTPClient

        client = ArxivHTTPClient(cache_enabled=False)

        feed_with_entry = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom"
              xmlns:arxiv="http://arxiv.org/schemas/atom">
            <entry>
                <id>http://arxiv.org/abs/2103.00020</id>
                <title>Test Paper Title</title>
                <summary>This is a test abstract.</summary>
                <published>2021-03-01T00:00:00Z</published>
                <updated>2021-03-01T00:00:00Z</updated>
                <author><name>John Doe</name></author>
                <author><name>Jane Smith</name></author>
                <category term="cs.LG" />
                <category term="cs.AI" />
                <link title="pdf" href="http://arxiv.org/pdf/2103.00020v1"/>
            </entry>
        </feed>"""

        results = client._parse_atom_feed(feed_with_entry)
        assert len(results) == 1

        result = results[0]
        assert result.arxiv_id == "2103.00020"
        assert result.title == "Test Paper Title"
        assert result.abstract == "This is a test abstract."
        assert len(result.authors) == 2
        assert "John Doe" in result.authors
        assert "Jane Smith" in result.authors
        assert "cs.LG" in result.categories
        assert "cs.AI" in result.categories

        client.close()

    def test_get_categories(self, mock_cache):
        """Test category list retrieval."""
        from kosmos.literature.arxiv_http_client import ArxivHTTPClient

        client = ArxivHTTPClient(cache_enabled=False)

        categories = client.get_categories()
        assert isinstance(categories, list)
        assert len(categories) > 0
        assert "cs.AI" in categories
        assert "cs.LG" in categories

        client.close()

    def test_filter_by_year(self, mock_cache):
        """Test year filtering."""
        from kosmos.literature.arxiv_http_client import ArxivHTTPClient
        from kosmos.literature.base_client import PaperMetadata, PaperSource, Author

        client = ArxivHTTPClient(cache_enabled=False)

        # Create test papers
        papers = [
            PaperMetadata(
                id="1", source=PaperSource.ARXIV, title="Paper 1",
                abstract="", authors=[Author(name="Author")],
                year=2020
            ),
            PaperMetadata(
                id="2", source=PaperSource.ARXIV, title="Paper 2",
                abstract="", authors=[Author(name="Author")],
                year=2021
            ),
            PaperMetadata(
                id="3", source=PaperSource.ARXIV, title="Paper 3",
                abstract="", authors=[Author(name="Author")],
                year=2022
            ),
        ]

        # Filter from 2021
        filtered = client._filter_by_year(papers, year_from=2021, year_to=None)
        assert len(filtered) == 2
        assert all(p.year >= 2021 for p in filtered)

        # Filter to 2021
        filtered = client._filter_by_year(papers, year_from=None, year_to=2021)
        assert len(filtered) == 2
        assert all(p.year <= 2021 for p in filtered)

        # Filter range
        filtered = client._filter_by_year(papers, year_from=2020, year_to=2021)
        assert len(filtered) == 2

        client.close()

    def test_context_manager(self, mock_cache):
        """Test context manager usage."""
        from kosmos.literature.arxiv_http_client import ArxivHTTPClient

        with ArxivHTTPClient(cache_enabled=False) as client:
            assert client is not None
            categories = client.get_categories()
            assert len(categories) > 0


class TestArxivClientFallback:
    """Tests for ArxivClient fallback to ArxivHTTPClient."""

    def test_fallback_when_arxiv_not_available(self, mock_cache):
        """Test that ArxivClient falls back to HTTP client."""
        # This test verifies the fallback mechanism exists
        from kosmos.literature.arxiv_client import ArxivClient, HAS_ARXIV

        # If arxiv is not available, the client should use fallback
        client = ArxivClient(cache_enabled=False)

        if not HAS_ARXIV:
            assert client._using_fallback is True
            assert client._fallback_client is not None
        else:
            assert client._using_fallback is False

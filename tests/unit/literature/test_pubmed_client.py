"""
Tests for kosmos.literature.pubmed_client module.
"""

import pytest
from unittest.mock import Mock, patch
from Bio import Entrez

from kosmos.literature.pubmed_client import PubMedClient
from kosmos.literature.base_client import PaperMetadata, PaperSource


@pytest.fixture
def pubmed_client():
    """Create PubMedClient instance for testing."""
    return PubMedClient(email="test@example.com")


@pytest.mark.unit
class TestPubMedInit:
    """Test PubMed client initialization."""

    def test_init_with_email(self):
        """Test initialization with email."""
        client = PubMedClient(email="test@example.com")
        # Email is stored in Entrez.email, not on the client
        assert Entrez.email == "test@example.com"

    def test_init_sets_rate_limit(self):
        """Test that rate limit is set correctly."""
        client = PubMedClient(email="test@example.com")
        # Without API key, rate limit is 3 req/s
        assert client.rate_limit == 3 or client.rate_limit == 10


MEDLINE_RECORDS = [
    {
        "PMID": "23287718",
        "TI": "Multiplex genome engineering using CRISPR/Cas systems.",
        "AB": "Functional elucidation of causal genetic variants.",
        "AU": ["Cong L", "Ran FA"],
        "DP": "2013 Feb 15",
        "TA": "Science",
        "AID": ["10.1126/science.1231143 [doi]"],
        "MH": ["CRISPR-Cas Systems"],
    },
    {
        "PMID": "28753425",
        "TI": "A second CRISPR paper.",
        "AB": "Abstract.",
        "AU": ["Doe J"],
        "DP": "2017",
        "TA": "Nature",
    },
]


@pytest.mark.unit
class TestPubMedSearch:
    """Test PubMed search functionality (Entrez is reached through _do_esearch/_do_efetch)."""

    def test_search_success(self, pubmed_client):
        """Test successful PubMed search."""
        with patch.object(pubmed_client, "_do_esearch", return_value=["23287718", "28753425"]), \
             patch.object(pubmed_client, "_do_efetch", return_value=MEDLINE_RECORDS):
            papers = pubmed_client.search("CRISPR", max_results=2)

        assert [p.pubmed_id for p in papers] == ["23287718", "28753425"]
        assert all(isinstance(p, PaperMetadata) for p in papers)
        assert papers[0].doi == "10.1126/science.1231143"
        assert papers[0].year == 2013
        assert papers[0].source == PaperSource.PUBMED

    def test_search_empty_results(self, pubmed_client):
        """Test search with no results."""
        with patch.object(pubmed_client, "_do_esearch", return_value=[]), \
             patch.object(pubmed_client, "_do_efetch") as efetch:
            papers = pubmed_client.search("nonexistent_query_xyz")

        assert papers == []
        efetch.assert_not_called()

    def test_search_with_error(self, pubmed_client):
        """Clients raise; UnifiedLiteratureSearch isolates per source."""
        with patch.object(pubmed_client, "_do_esearch", side_effect=Exception("API Error")):
            with pytest.raises(Exception, match="API Error"):
                pubmed_client.search("test query")


@pytest.mark.unit
class TestPubMedGetPaper:
    """Test fetching papers by PubMed ID."""

    def test_get_paper_by_id_success(self, pubmed_client):
        """Test fetching paper by PubMed ID."""
        # Create mock paper metadata
        mock_paper = PaperMetadata(
            id="23287718",
            source=PaperSource.PUBMED,
            title="CRISPR Test",
            abstract="Test abstract",
            pubmed_id="23287718",
        )

        with patch.object(pubmed_client, '_fetch_paper_details', return_value=[mock_paper]):
            paper = pubmed_client.get_paper_by_id("23287718")

        assert paper is not None
        assert paper.pubmed_id == "23287718"

    def test_get_paper_by_id_not_found(self, pubmed_client):
        """Test fetching non-existent paper."""
        with patch.object(pubmed_client, '_fetch_paper_details', return_value=[]):
            paper = pubmed_client.get_paper_by_id("99999999")
        assert paper is None


@pytest.mark.unit
class TestPubMedRateLimiting:
    """Test rate limiting."""

    @patch('kosmos.literature.pubmed_client.time.sleep')
    def test_rate_limiting_delay(self, mock_sleep, pubmed_client):
        """Test that rate limiting adds delays."""
        # Make multiple requests (Entrez is reached through _do_esearch)
        with patch.object(pubmed_client, "_do_esearch", return_value=[]):
            pubmed_client.search("query1", max_results=1)
            pubmed_client.search("query2", max_results=1)

        # Should add delays between requests
        assert mock_sleep.called


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_network
class TestPubMedIntegration:
    """Integration tests (requires network)."""

    def test_real_search(self):
        """Test real PubMed search."""
        client = PubMedClient(email="test@example.com")
        papers = client.search("diabetes", max_results=2)

        assert len(papers) > 0
        assert all(isinstance(p, PaperMetadata) for p in papers)
        assert all(p.pubmed_id is not None for p in papers)

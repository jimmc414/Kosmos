"""
Tests for kosmos.literature.unified_search module.
"""

import pytest
from unittest.mock import Mock, patch

from kosmos.literature.unified_search import UnifiedLiteratureSearch
from kosmos.literature.base_client import PaperMetadata, PaperSource


def make_paper(pid, source, title="A Paper", **ids):
    return PaperMetadata(id=pid, source=source, title=title, abstract="", year=2023, **ids)


@pytest.fixture
def unified_search():
    """UnifiedLiteratureSearch with all three sources replaced by mocks."""
    search = UnifiedLiteratureSearch()
    search.clients = {
        PaperSource.ARXIV: Mock(),
        PaperSource.SEMANTIC_SCHOLAR: Mock(),
        PaperSource.PUBMED: Mock(),
    }
    for client in search.clients.values():
        client.search.return_value = []
    return search


@pytest.mark.unit
class TestUnifiedSearchInit:
    """Test unified search initialization."""

    def test_init_default(self):
        """All three sources are enabled by default."""
        search = UnifiedLiteratureSearch()
        assert set(search.clients) == {
            PaperSource.ARXIV, PaperSource.SEMANTIC_SCHOLAR, PaperSource.PUBMED
        }

    def test_init_with_custom_sources(self):
        """Disabled sources get no client."""
        search = UnifiedLiteratureSearch(pubmed_enabled=False)
        assert PaperSource.ARXIV in search.clients
        assert PaperSource.SEMANTIC_SCHOLAR in search.clients
        assert PaperSource.PUBMED not in search.clients


@pytest.mark.unit
class TestUnifiedSearch:
    """Test unified search functionality."""

    def test_search_all_sources(self, unified_search):
        """Every enabled source is searched and the results are merged."""
        clients = unified_search.clients
        clients[PaperSource.ARXIV].search.return_value = [make_paper("a", PaperSource.ARXIV, "Alpha", arxiv_id="1")]
        clients[PaperSource.SEMANTIC_SCHOLAR].search.return_value = [make_paper("s", PaperSource.SEMANTIC_SCHOLAR, "Beta", doi="10.1/b")]
        clients[PaperSource.PUBMED].search.return_value = [make_paper("p", PaperSource.PUBMED, "Gamma", pubmed_id="3")]

        papers = unified_search.search("machine learning", max_results_per_source=10)

        assert {p.id for p in papers} == {"a", "s", "p"}
        assert all(c.search.called for c in clients.values())

    def test_search_specific_sources(self, unified_search):
        """Only the requested sources are searched."""
        clients = unified_search.clients
        clients[PaperSource.ARXIV].search.return_value = [make_paper("a", PaperSource.ARXIV, "Alpha", arxiv_id="1")]
        clients[PaperSource.SEMANTIC_SCHOLAR].search.return_value = [make_paper("s", PaperSource.SEMANTIC_SCHOLAR, "Beta", doi="10.1/b")]

        papers = unified_search.search(
            "test query", sources=[PaperSource.ARXIV, PaperSource.SEMANTIC_SCHOLAR]
        )

        assert {p.id for p in papers} == {"a", "s"}
        assert not clients[PaperSource.PUBMED].search.called

    def test_deduplication(self, unified_search):
        """The same DOI from two sources is returned once."""
        clients = unified_search.clients
        clients[PaperSource.ARXIV].search.return_value = [make_paper("a", PaperSource.ARXIV, "Same Paper", doi="10.1234/same")]
        clients[PaperSource.SEMANTIC_SCHOLAR].search.return_value = [make_paper("s", PaperSource.SEMANTIC_SCHOLAR, "Same Paper", doi="10.1234/SAME")]

        papers = unified_search.search("test")

        assert len(papers) == 1

    def test_search_with_errors(self, unified_search):
        """A raising client is isolated: the other sources still return."""
        clients = unified_search.clients
        clients[PaperSource.ARXIV].search.side_effect = Exception("API Error")
        clients[PaperSource.PUBMED].search.return_value = [make_paper("p", PaperSource.PUBMED, "Gamma", pubmed_id="3")]

        papers = unified_search.search("test query")

        assert [p.id for p in papers] == ["p"]

    def test_search_with_only_a_failing_source(self, unified_search):
        """Searching only a raising source returns an empty list instead of raising."""
        unified_search.clients[PaperSource.ARXIV].search.side_effect = Exception("API Error")

        assert unified_search.search("test query", sources=[PaperSource.ARXIV]) == []


@pytest.mark.unit
class TestUnifiedSearchDeduplication:
    """Test deduplication strategies."""

    def test_deduplicate_by_doi(self, unified_search):
        papers = [
            make_paper("1", PaperSource.ARXIV, "Paper 1", doi="10.1234/test"),
            make_paper("2", PaperSource.SEMANTIC_SCHOLAR, "Paper 1 Duplicate", doi="10.1234/test"),
        ]
        assert len(unified_search._deduplicate_papers(papers)) == 1

    def test_deduplicate_by_arxiv_id(self, unified_search):
        papers = [
            make_paper("1", PaperSource.ARXIV, "Paper 1", arxiv_id="2301.00001"),
            make_paper("2", PaperSource.SEMANTIC_SCHOLAR, "Paper 1", arxiv_id="2301.00001"),
        ]
        assert len(unified_search._deduplicate_papers(papers)) == 1

    def test_deduplicate_by_title_similarity(self, unified_search):
        """Titles that differ only in case and punctuation are duplicates."""
        papers = [
            make_paper("1", PaperSource.ARXIV, "Attention Is All You Need"),
            make_paper("2", PaperSource.SEMANTIC_SCHOLAR, "Attention is all you need!"),
        ]
        assert len(unified_search._deduplicate_papers(papers)) == 1

    def test_distinct_titles_are_kept(self, unified_search):
        papers = [
            make_paper("1", PaperSource.ARXIV, "Attention Is All You Need"),
            make_paper("2", PaperSource.ARXIV, "Attention Is Not All You Need"),
        ]
        assert len(unified_search._deduplicate_papers(papers)) == 2


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_network
class TestUnifiedSearchIntegration:
    """Integration tests."""

    def test_real_unified_search(self):
        """Test real unified search across sources."""
        search = UnifiedLiteratureSearch()
        papers = search.search("transformer neural network", max_results=5)

        assert len(papers) > 0
        assert all(isinstance(p, PaperMetadata) for p in papers)
        # Should have papers from multiple sources
        sources = set(p.source for p in papers)
        assert len(sources) > 1

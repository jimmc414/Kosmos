"""
Tests for kosmos.literature.citations module.
"""

import pytest
from pathlib import Path

from kosmos.literature.citations import CitationParser, CitationFormatter, CitationNetwork
from kosmos.literature.base_client import PaperMetadata


@pytest.fixture
def citation_parser():
    """Create CitationParser instance."""
    return CitationParser()


@pytest.fixture
def citation_formatter():
    """Create CitationFormatter instance."""
    return CitationFormatter()


@pytest.fixture
def citation_network():
    """Create CitationNetwork instance."""
    return CitationNetwork()


@pytest.mark.unit
class TestCitationParser:
    """Test citation parsing."""

    def test_parse_bibtex(self, citation_parser, sample_bibtex):
        """Test parsing BibTeX file."""
        papers = citation_parser.parse_bibtex(str(sample_bibtex))

        assert len(papers) > 0
        assert all(isinstance(p, PaperMetadata) for p in papers)
        # Check that Attention paper is present
        attention_papers = [p for p in papers if "Attention" in p.title]
        assert len(attention_papers) > 0

    def test_parse_ris(self, citation_parser, sample_ris):
        """Test parsing RIS file."""
        papers = citation_parser.parse_ris(str(sample_ris))

        assert len(papers) > 0
        assert all(isinstance(p, PaperMetadata) for p in papers)

    def test_parse_bibtex_with_invalid_file(self, citation_parser, temp_file):
        """Test parsing invalid BibTeX file."""
        invalid_file = temp_file("invalid.bib", "invalid content @@@")

        papers = citation_parser.parse_bibtex(str(invalid_file))
        # Should return empty list or handle gracefully
        assert isinstance(papers, list)

    def test_parse_ris_with_missing_file(self, citation_parser):
        """Test parsing non-existent RIS file."""
        with pytest.raises(FileNotFoundError):
            citation_parser.parse_ris("nonexistent.ris")


@pytest.mark.unit
class TestCitationFormatter:
    """Test citation formatting."""

    def test_format_apa(self, citation_formatter, sample_paper_metadata):
        """Test APA citation formatting."""
        citation = citation_formatter.format_citation(sample_paper_metadata, style="apa")

        assert isinstance(citation, str)
        assert sample_paper_metadata.authors[0].name in citation
        assert str(sample_paper_metadata.year) in citation
        assert sample_paper_metadata.title in citation

    def test_format_chicago(self, citation_formatter, sample_paper_metadata):
        """Test Chicago citation formatting."""
        citation = citation_formatter.format_citation(sample_paper_metadata, style="chicago")

        assert isinstance(citation, str)
        assert sample_paper_metadata.title in citation

    def test_format_ieee(self, citation_formatter, sample_paper_metadata):
        """Test IEEE citation formatting."""
        citation = citation_formatter.format_citation(sample_paper_metadata, style="ieee")

        assert isinstance(citation, str)
        assert str(sample_paper_metadata.year) in citation

    def test_format_harvard(self, citation_formatter, sample_paper_metadata):
        """Test Harvard citation formatting."""
        citation = citation_formatter.format_citation(sample_paper_metadata, style="harvard")

        assert isinstance(citation, str)

    def test_format_vancouver(self, citation_formatter, sample_paper_metadata):
        """Test Vancouver citation formatting."""
        citation = citation_formatter.format_citation(sample_paper_metadata, style="vancouver")

        assert isinstance(citation, str)

    def test_format_invalid_style(self, citation_formatter, sample_paper_metadata):
        """Test formatting with invalid style defaults to APA."""
        citation = citation_formatter.format_citation(
            sample_paper_metadata, style="invalid_style"
        )

        # Should default to APA
        assert isinstance(citation, str)

    def test_to_bibtex(self, citation_formatter, sample_paper_metadata):
        """Test converting to BibTeX format."""
        bibtex = citation_formatter.to_bibtex(sample_paper_metadata)

        assert "@" in bibtex
        assert sample_paper_metadata.title in bibtex
        assert str(sample_paper_metadata.year) in bibtex

    def test_to_ris(self, citation_formatter, sample_paper_metadata):
        """Test converting to RIS format."""
        ris = citation_formatter.to_ris(sample_paper_metadata)

        assert "TY  -" in ris
        assert "TI  -" in ris
        assert "AU  -" in ris
        assert "ER  -" in ris
        assert sample_paper_metadata.title in ris


@pytest.mark.unit
class TestCitationNetwork:
    """Test citation network analysis."""

    def test_build_network(self, citation_network, sample_papers_list):
        """Test building citation network."""
        graph = citation_network.build_network(sample_papers_list)

        assert graph is not None
        # Should have nodes for each paper
        assert graph.number_of_nodes() >= len(sample_papers_list)

    def test_identify_seminal_papers(self, citation_network, sample_papers_list):
        """Test identifying seminal papers."""
        graph = citation_network.build_network(sample_papers_list)
        seminal = citation_network.identify_seminal_papers(graph, top_n=3)

        assert len(seminal) <= 3
        assert all(isinstance(item, tuple) for item in seminal)

    def test_get_citation_path(self, citation_network, sample_papers_list):
        """A path exists only along citation edges."""
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c")])

        assert citation_network.get_citation_path(graph, "a", "c") == ["a", "b", "c"]
        assert citation_network.get_citation_path(graph, "c", "a") is None

    def test_analyze_influence(self, citation_network):
        """Influence metrics report degree, PageRank and betweenness for a node."""
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_edges_from([("a", "hub"), ("b", "hub"), ("hub", "c")])

        metrics = citation_network.analyze_influence(graph, "hub")

        assert metrics["in_degree"] == 2
        assert metrics["out_degree"] == 1
        assert metrics["betweenness_centrality"] > 0
        assert metrics["pagerank"] > 0
        assert citation_network.analyze_influence(graph, "missing")["in_degree"] == 0


@pytest.mark.unit
class TestCitationFormatConversion:
    """Test converting between citation formats."""

    def test_bibtex_to_paper_to_ris(
        self, citation_parser, citation_formatter, sample_bibtex
    ):
        """Test round-trip conversion BibTeX -> Paper -> RIS."""
        # Parse BibTeX
        papers = citation_parser.parse_bibtex(str(sample_bibtex))
        assert len(papers) > 0

        # Convert to RIS
        ris = citation_formatter.to_ris(papers[0])
        assert "TY  -" in ris
        assert papers[0].title in ris

    def test_format_consistency(self, citation_formatter, sample_paper_metadata):
        """Test that all formats can be generated for same paper."""
        styles = ["apa", "chicago", "ieee", "harvard", "vancouver"]

        for style in styles:
            citation = citation_formatter.format_citation(sample_paper_metadata, style=style)
            assert isinstance(citation, str)
            assert len(citation) > 0


@pytest.mark.integration
class TestCitationIntegration:
    """Integration tests for citations."""

    def test_parse_and_format_bibtex(self, citation_parser, citation_formatter, sample_bibtex):
        """Test parsing BibTeX and formatting citations."""
        papers = citation_parser.parse_bibtex(str(sample_bibtex))
        citations = [
            citation_formatter.format_citation(p, style="apa") for p in papers[:3]
        ]

        assert len(citations) > 0
        assert all(isinstance(c, str) for c in citations)

    def test_build_citation_network_from_bibtex(
        self, citation_parser, citation_network, sample_bibtex
    ):
        """Test building citation network from BibTeX."""
        papers = citation_parser.parse_bibtex(str(sample_bibtex))
        graph = citation_network.build_network(papers)

        assert graph.number_of_nodes() >= len(papers)

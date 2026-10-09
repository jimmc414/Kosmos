"""
Citation parsing, formatting, validation, and network analysis.

Supports:
- BibTeX and RIS parsing/generation
- Multiple citation styles (APA, Chicago, IEEE, Harvard, Vancouver)
- Citation network analysis
- Format validation
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase
import networkx as nx

from kosmos.literature.base_client import PaperMetadata, Author, PaperSource

logger = logging.getLogger(__name__)


class CitationParser:
    """
    Parse citations from various formats.

    Supports BibTeX, RIS, and text extraction.
    """

    def __init__(self):
        """Initialize citation parser."""
        logger.info("Initialized CitationParser")

    def parse_bibtex(self, file_path: str) -> List[PaperMetadata]:
        """
        Parse BibTeX file.

        Args:
            file_path: Path to .bib file

        Returns:
            List of PaperMetadata objects

        Example:
            ```python
            parser = CitationParser()
            papers = parser.parse_bibtex("references.bib")
            ```
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as bibtex_file:
                parser = BibTexParser(common_strings=True)
                bib_database = bibtexparser.load(bibtex_file, parser)

            papers = []
            for entry in bib_database.entries:
                paper = self._bibtex_entry_to_paper(entry)
                if paper:
                    papers.append(paper)

            logger.info(f"Parsed {len(papers)} papers from {file_path}")
            return papers

        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"BibTeX parsing failed: {e}")
            return []

    def parse_bibtex_string(self, bibtex_str: str) -> Optional[PaperMetadata]:
        """
        Parse single BibTeX entry from string.

        Args:
            bibtex_str: BibTeX entry as string

        Returns:
            PaperMetadata object or None

        Example:
            ```python
            bibtex = '''@article{key2024,
                title={Example},
                author={Doe, John},
                year={2024}
            }'''
            paper = parser.parse_bibtex_string(bibtex)
            ```
        """
        try:
            parser = BibTexParser(common_strings=True)
            bib_database = bibtexparser.loads(bibtex_str, parser)

            if bib_database.entries:
                return self._bibtex_entry_to_paper(bib_database.entries[0])

            return None

        except Exception as e:
            logger.error(f"BibTeX string parsing failed: {e}")
            return None

    def parse_ris(self, file_path: str) -> List[PaperMetadata]:
        """
        Parse RIS file.

        Args:
            file_path: Path to .ris file

        Returns:
            List of PaperMetadata objects

        Example:
            ```python
            papers = parser.parse_ris("references.ris")
            ```
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as ris_file:
                ris_content = ris_file.read()

            papers = []
            # Split by ER  - (end of record)
            entries = re.split(r'ER\s*-\s*\n', ris_content)

            for entry in entries:
                if entry.strip():
                    paper = self._ris_entry_to_paper(entry)
                    if paper:
                        papers.append(paper)

            logger.info(f"Parsed {len(papers)} papers from {file_path}")
            return papers

        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"RIS parsing failed: {e}")
            return []

    def extract_citations_from_text(self, text: str) -> List[str]:
        """
        Extract citation strings from text using regex patterns.

        Args:
            text: Text containing citations

        Returns:
            List of citation strings

        Example:
            ```python
            text = "According to Smith et al. (2020), ..."
            citations = parser.extract_citations_from_text(text)
            ```
        """
        citations = []

        # Pattern: Author et al. (Year)
        pattern1 = r'([A-Z][a-z]+(?:\set\sal\.)?)\s*\((\d{4})\)'
        matches1 = re.findall(pattern1, text)
        citations.extend([f"{author} ({year})" for author, year in matches1])

        # Pattern: [Author, Year]
        pattern2 = r'\[([A-Z][a-z]+(?:\set\sal\.)?),\s*(\d{4})\]'
        matches2 = re.findall(pattern2, text)
        citations.extend([f"{author} ({year})" for author, year in matches2])

        # Pattern: (Author1 & Author2, Year)
        pattern3 = r'\(([A-Z][a-z]+\s*&\s*[A-Z][a-z]+),\s*(\d{4})\)'
        matches3 = re.findall(pattern3, text)
        citations.extend([f"{authors} ({year})" for authors, year in matches3])

        return list(set(citations))  # Remove duplicates

    def _bibtex_entry_to_paper(self, entry: Dict[str, Any]) -> Optional[PaperMetadata]:
        """Convert BibTeX entry to PaperMetadata."""
        try:
            # Parse authors
            authors = []
            if 'author' in entry:
                author_str = entry['author']
                # Split by ' and '
                for author_name in author_str.split(' and '):
                    # Handle "Last, First" or "First Last" formats
                    if ',' in author_name:
                        parts = author_name.split(',')
                        name = f"{parts[1].strip()} {parts[0].strip()}"
                    else:
                        name = author_name.strip()

                    authors.append(Author(name=name))

            # Parse year
            year = None
            if 'year' in entry:
                try:
                    year = int(entry['year'])
                except ValueError:
                    pass

            # Build PaperMetadata
            paper = PaperMetadata(
                id=entry.get('ID', ''),
                source=PaperSource.OTHER,
                title=entry.get('title', ''),
                abstract=entry.get('abstract', ''),
                authors=authors,
                year=year,
                doi=entry.get('doi'),
                url=entry.get('url'),
                journal=entry.get('journal'),
                venue=entry.get('booktitle') or entry.get('journal'),
                raw_data=entry
            )

            return paper

        except Exception as e:
            logger.debug(f"Error converting BibTeX entry: {e}")
            return None

    def _ris_entry_to_paper(self, entry: str) -> Optional[PaperMetadata]:
        """Convert RIS entry to PaperMetadata."""
        try:
            lines = entry.strip().split('\n')
            data = {}

            for line in lines:
                if '  - ' in line:
                    tag, value = line.split('  - ', 1)
                    tag = tag.strip()
                    value = value.strip()

                    if tag in data:
                        if isinstance(data[tag], list):
                            data[tag].append(value)
                        else:
                            data[tag] = [data[tag], value]
                    else:
                        data[tag] = value

            # Parse authors (AU tag)
            authors = []
            if 'AU' in data:
                author_list = data['AU'] if isinstance(data['AU'], list) else [data['AU']]
                for author_name in author_list:
                    authors.append(Author(name=author_name))

            # Parse year (PY tag)
            year = None
            if 'PY' in data:
                try:
                    year = int(data['PY'].split('/')[0])  # Handle PY  - 2024/01/15
                except (ValueError, IndexError):
                    pass

            # Build PaperMetadata
            paper = PaperMetadata(
                id=data.get('ID', ''),
                source=PaperSource.OTHER,
                title=data.get('TI', ''),
                abstract=data.get('AB', ''),
                authors=authors,
                year=year,
                doi=data.get('DO'),
                url=data.get('UR'),
                journal=data.get('JO'),
                venue=data.get('JO') or data.get('T2'),
                raw_data=data
            )

            return paper

        except Exception as e:
            logger.debug(f"Error converting RIS entry: {e}")
            return None


class CitationFormatter:
    """
    Generate formatted citations in multiple styles.

    Supports APA, Chicago, IEEE, Harvard, Vancouver.
    """

    def __init__(self):
        """Initialize citation formatter."""
        logger.info("Initialized CitationFormatter")

    def format_citation(
        self,
        paper: PaperMetadata,
        style: str = "apa"
    ) -> str:
        """
        Format citation in specified style.

        Args:
            paper: PaperMetadata object
            style: Citation style (apa, chicago, ieee, harvard, vancouver)

        Returns:
            Formatted citation string

        Example:
            ```python
            formatter = CitationFormatter()
            apa = formatter.format_citation(paper, style="apa")
            ieee = formatter.format_citation(paper, style="ieee")
            ```
        """
        style = style.lower()

        formatters = {
            "apa": self._format_apa,
            "chicago": self._format_chicago,
            "ieee": self._format_ieee,
            "harvard": self._format_harvard,
            "vancouver": self._format_vancouver
        }

        formatter_func = formatters.get(style, self._format_apa)
        return formatter_func(paper)

    def to_bibtex(self, paper: PaperMetadata) -> str:
        """
        Generate BibTeX entry.

        Args:
            paper: PaperMetadata object

        Returns:
            BibTeX string

        Example:
            ```python
            bibtex = formatter.to_bibtex(paper)
            ```
        """
        # Determine entry type
        entry_type = "article"
        if paper.venue and "conference" in paper.venue.lower():
            entry_type = "inproceedings"

        # Generate cite key (author_year)
        cite_key = self._generate_cite_key(paper)

        # Build entry
        entry = f"@{entry_type}{{{cite_key},\n"

        if paper.title:
            entry += f"  title = {{{paper.title}}},\n"

        if paper.authors:
            author_str = " and ".join([a.name for a in paper.authors])
            entry += f"  author = {{{author_str}}},\n"

        if paper.year:
            entry += f"  year = {{{paper.year}}},\n"

        if paper.journal:
            entry += f"  journal = {{{paper.journal}}},\n"

        if paper.venue:
            entry += f"  booktitle = {{{paper.venue}}},\n"

        if paper.doi:
            entry += f"  doi = {{{paper.doi}}},\n"

        if paper.url:
            entry += f"  url = {{{paper.url}}},\n"

        if paper.abstract:
            entry += f"  abstract = {{{paper.abstract}}},\n"

        entry += "}\n"

        return entry

    def to_ris(self, paper: PaperMetadata) -> str:
        """
        Generate RIS entry.

        Args:
            paper: PaperMetadata object

        Returns:
            RIS string

        Example:
            ```python
            ris = formatter.to_ris(paper)
            ```
        """
        lines = []

        # Type
        lines.append("TY  - JOUR")  # Journal article

        # Title
        if paper.title:
            lines.append(f"TI  - {paper.title}")

        # Authors
        for author in (paper.authors or []):
            lines.append(f"AU  - {author.name}")

        # Year
        if paper.year:
            lines.append(f"PY  - {paper.year}")

        # Journal
        if paper.journal:
            lines.append(f"JO  - {paper.journal}")

        # DOI
        if paper.doi:
            lines.append(f"DO  - {paper.doi}")

        # URL
        if paper.url:
            lines.append(f"UR  - {paper.url}")

        # Abstract
        if paper.abstract:
            lines.append(f"AB  - {paper.abstract}")

        # End of record
        lines.append("ER  - ")

        return "\n".join(lines) + "\n"

    def generate_bibliography(
        self,
        papers: List[PaperMetadata],
        style: str = "apa",
        sort_by: str = "author"
    ) -> str:
        """
        Generate formatted bibliography.

        Args:
            papers: List of papers
            style: Citation style
            sort_by: Sort order (author, year, title)

        Returns:
            Formatted bibliography string

        Example:
            ```python
            bib = formatter.generate_bibliography(
                papers,
                style="apa",
                sort_by="year"
            )
            ```
        """
        # Filter out None papers
        valid_papers = [p for p in papers if p is not None]

        # Sort papers
        if sort_by == "author":
            sorted_papers = sorted(valid_papers, key=lambda p: p.authors[0].name if p.authors and len(p.authors) > 0 else "")
        elif sort_by == "year":
            sorted_papers = sorted(valid_papers, key=lambda p: p.year or 0, reverse=True)
        elif sort_by == "title":
            sorted_papers = sorted(valid_papers, key=lambda p: p.title or "")
        else:
            sorted_papers = valid_papers

        # Format each citation
        citations = []
        for paper in sorted_papers:
            citation = self.format_citation(paper, style=style)
            citations.append(citation)

        return "\n\n".join(citations)

    # Style-specific formatters

    def _format_apa(self, paper: PaperMetadata) -> str:
        """APA style formatting."""
        parts = []

        # Authors
        if paper.authors:
            if len(paper.authors) == 1:
                author_str = f"{paper.authors[0].name}"
            elif len(paper.authors) == 2:
                author_str = f"{paper.authors[0].name} & {paper.authors[1].name}"
            else:
                author_str = f"{paper.authors[0].name} et al."
            parts.append(author_str)

        # Year
        year_str = f"({paper.year})" if paper.year else "(n.d.)"
        parts.append(year_str)

        # Title
        if paper.title:
            parts.append(f"{paper.title}.")

        # Journal
        if paper.journal:
            journal_str = f"*{paper.journal}*"
            parts.append(journal_str)

        # DOI
        if paper.doi:
            parts.append(f"https://doi.org/{paper.doi}")

        return " ".join(parts)

    def _format_chicago(self, paper: PaperMetadata) -> str:
        """Chicago style formatting."""
        parts = []

        # Authors (Last, First format)
        if paper.authors:
            author_str = paper.authors[0].name
            if len(paper.authors) > 1:
                author_str += " et al."
            parts.append(f"{author_str}.")

        # Title
        if paper.title:
            parts.append(f'"{paper.title}."')

        # Journal
        if paper.journal:
            journal_str = f"*{paper.journal}*"
            if paper.year:
                journal_str += f" ({paper.year})"
            parts.append(journal_str + ".")

        return " ".join(parts)

    def _format_ieee(self, paper: PaperMetadata) -> str:
        """IEEE style formatting."""
        parts = []

        # Authors (initials)
        if paper.authors:
            if len(paper.authors) <= 3:
                author_str = ", ".join([a.name for a in paper.authors])
            else:
                author_str = f"{paper.authors[0].name} et al."
            parts.append(author_str + ",")

        # Title
        if paper.title:
            parts.append(f'"{paper.title},"')

        # Journal (or venue), then year
        venue = paper.journal or paper.venue
        if venue:
            journal_str = f"*{venue}*"
            if paper.year:
                journal_str += f", {paper.year}"
            parts.append(journal_str + ".")
        elif paper.year:
            parts.append(f"{paper.year}.")

        return " ".join(parts)

    def _format_harvard(self, paper: PaperMetadata) -> str:
        """Harvard style formatting."""
        # Similar to APA
        return self._format_apa(paper)

    def _format_vancouver(self, paper: PaperMetadata) -> str:
        """Vancouver style formatting (numbered)."""
        parts = []

        # Authors
        if paper.authors:
            if len(paper.authors) <= 6:
                author_str = ", ".join([a.name for a in paper.authors])
            else:
                author_str = ", ".join([a.name for a in paper.authors[:6]]) + ", et al."
            parts.append(author_str + ".")

        # Title
        if paper.title:
            parts.append(f"{paper.title}.")

        # Journal
        if paper.journal:
            journal_str = f"{paper.journal}."
            if paper.year:
                journal_str += f" {paper.year}"
            parts.append(journal_str + ".")

        return " ".join(parts)

    def _generate_cite_key(self, paper: PaperMetadata) -> str:
        """Generate BibTeX cite key (author_year)."""
        if paper.authors and paper.year:
            author_last = paper.authors[0].name.split()[-1]
            return f"{author_last.lower()}{paper.year}"
        elif paper.authors:
            author_last = paper.authors[0].name.split()[-1]
            return f"{author_last.lower()}unknown"
        else:
            return "unknownauthor"


class CitationNetwork:
    """
    Build and analyze citation networks.

    Uses NetworkX for graph analysis.
    """

    def __init__(self, use_knowledge_graph: bool = False):
        """
        Initialize citation network.

        Args:
            use_knowledge_graph: Whether to integrate with Neo4j
        """
        self.use_knowledge_graph = use_knowledge_graph

        if use_knowledge_graph:
            try:
                from kosmos.knowledge.graph import get_knowledge_graph
                self.knowledge_graph = get_knowledge_graph()
            except Exception as e:
                logger.warning(f"Knowledge graph unavailable: {e}")
                self.knowledge_graph = None

        logger.info("Initialized CitationNetwork")

    def build_network(
        self,
        papers: List[PaperMetadata],
        fetch_citations: bool = False
    ) -> nx.DiGraph:
        """
        Build citation network graph.

        Args:
            papers: List of papers
            fetch_citations: Whether to fetch missing citations

        Returns:
            NetworkX directed graph

        Example:
            ```python
            network = CitationNetwork()
            graph = network.build_network(papers)
            print(f"Nodes: {graph.number_of_nodes()}")
            print(f"Edges: {graph.number_of_edges()}")
            ```
        """
        G = nx.DiGraph()

        # Add papers as nodes
        for paper in papers:
            G.add_node(
                paper.primary_identifier,
                title=paper.title,
                year=paper.year,
                citation_count=paper.citation_count
            )

        # Add citation edges (if available from knowledge graph)
        if self.use_knowledge_graph and self.knowledge_graph:
            for paper in papers:
                try:
                    citations = self.knowledge_graph.get_citations(
                        paper.primary_identifier,
                        depth=1
                    )

                    for cited in citations:
                        cited_id = cited["paper"].get("id")
                        if cited_id:
                            G.add_edge(paper.primary_identifier, cited_id)

                except Exception as e:
                    logger.debug(f"Citation fetch failed: {e}")

        logger.info(f"Built citation network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def get_citation_path(
        self,
        graph: nx.DiGraph,
        paper1_id: str,
        paper2_id: str
    ) -> Optional[List[str]]:
        """
        Find shortest citation path between two papers.

        Args:
            graph: Citation network graph
            paper1_id: First paper ID
            paper2_id: Second paper ID

        Returns:
            List of paper IDs forming path, or None if no path exists
        """
        try:
            path = nx.shortest_path(graph, paper1_id, paper2_id)
            return path
        except nx.NetworkXNoPath:
            return None

    def analyze_influence(
        self,
        graph: nx.DiGraph,
        paper_id: str
    ) -> Dict[str, Any]:
        """
        Analyze paper's influence in citation network.

        Args:
            graph: Citation network
            paper_id: Paper to analyze

        Returns:
            Dictionary with influence metrics

        Example:
            ```python
            metrics = network.analyze_influence(graph, "arxiv:1234.5678")
            print(f"Betweenness: {metrics['betweenness_centrality']}")
            ```
        """
        metrics = {}

        # Betweenness centrality
        try:
            betweenness = nx.betweenness_centrality(graph)
            metrics["betweenness_centrality"] = betweenness.get(paper_id, 0.0)
        except Exception as e:
            logger.debug(f"Betweenness calculation failed: {e}")

        # PageRank
        try:
            pagerank = nx.pagerank(graph)
            metrics["pagerank"] = pagerank.get(paper_id, 0.0)
        except Exception as e:
            logger.debug(f"PageRank calculation failed: {e}")

        # Degree centrality
        metrics["in_degree"] = graph.in_degree(paper_id) if graph.has_node(paper_id) else 0
        metrics["out_degree"] = graph.out_degree(paper_id) if graph.has_node(paper_id) else 0

        return metrics

    def identify_seminal_papers(
        self,
        graph: nx.DiGraph,
        top_n: int = 10
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Identify most influential papers.

        Args:
            graph: Citation network
            top_n: Number of papers to return

        Returns:
            List of (paper_id, metrics) tuples

        Example:
            ```python
            seminal = network.identify_seminal_papers(graph, top_n=10)
            for paper_id, metrics in seminal:
                print(f"{paper_id}: PageRank={metrics['pagerank']:.4f}")
            ```
        """
        # Calculate PageRank
        try:
            pagerank = nx.pagerank(graph)

            # Sort by PageRank
            ranked = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

            # Get top N with metrics
            results = []
            for paper_id, pr_score in ranked[:top_n]:
                metrics = {
                    "pagerank": pr_score,
                    "in_degree": graph.in_degree(paper_id),
                    "out_degree": graph.out_degree(paper_id)
                }
                results.append((paper_id, metrics))

            return results

        except Exception as e:
            logger.error(f"Seminal paper identification failed: {e}")
            return []


class CitationValidator:
    """
    Validate citation formats and data.
    """

    def __init__(self):
        """Initialize citation validator."""
        logger.info("Initialized CitationValidator")

    def validate_bibtex(self, bibtex_str: str) -> Tuple[bool, List[str]]:
        """
        Validate BibTeX format.

        Args:
            bibtex_str: BibTeX string to validate

        Returns:
            Tuple of (is_valid, error_messages)

        Example:
            ```python
            validator = CitationValidator()
            is_valid, errors = validator.validate_bibtex(bibtex_str)
            if not is_valid:
                print("Errors:", errors)
            ```
        """
        errors = []

        try:
            parser = BibTexParser()
            bibtexparser.loads(bibtex_str, parser)

            # Check for required fields
            if '@' not in bibtex_str:
                errors.append("Missing entry type (e.g., @article)")

            if '{' not in bibtex_str or '}' not in bibtex_str:
                errors.append("Missing or unbalanced braces")

            return (len(errors) == 0, errors)

        except Exception as e:
            errors.append(f"Parse error: {str(e)}")
            return (False, errors)

    def validate_ris(self, ris_str: str) -> Tuple[bool, List[str]]:
        """
        Validate RIS format.

        Args:
            ris_str: RIS string to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check for required tags
        required_tags = ['TY', 'ER']

        for tag in required_tags:
            if f"{tag}  - " not in ris_str:
                errors.append(f"Missing required tag: {tag}")

        # Check format of tags
        lines = ris_str.strip().split('\n')
        for line in lines:
            if line.strip() and '  - ' not in line:
                errors.append(f"Invalid line format: {line}")
                break

        return (len(errors) == 0, errors)

    def validate_citation_data(
        self,
        citation: Dict[str, Any],
        required_fields: List[str] = ["title", "year"]
    ) -> Tuple[bool, List[str]]:
        """
        Validate citation has required fields.

        Args:
            citation: Citation data dictionary
            required_fields: List of required field names

        Returns:
            Tuple of (is_valid, missing_fields)
        """
        missing = []

        for field in required_fields:
            if field not in citation or not citation[field]:
                missing.append(field)

        return (len(missing) == 0, missing)


# Utility functions

def papers_to_bibtex(papers: List[PaperMetadata], output_file: str):
    """
    Export papers to BibTeX file.

    Args:
        papers: List of papers
        output_file: Output file path

    Example:
        ```python
        from kosmos.literature.citations import papers_to_bibtex
        papers_to_bibtex(papers, "my_library.bib")
        ```
    """
    formatter = CitationFormatter()

    with open(output_file, 'w', encoding='utf-8') as f:
        for paper in papers:
            bibtex = formatter.to_bibtex(paper)
            f.write(bibtex + "\n")

    logger.info(f"Exported {len(papers)} papers to {output_file}")


def papers_to_ris(papers: List[PaperMetadata], output_file: str):
    """
    Export papers to RIS file.

    Args:
        papers: List of papers
        output_file: Output file path

    Example:
        ```python
        from kosmos.literature.citations import papers_to_ris
        papers_to_ris(papers, "my_library.ris")
        ```
    """
    formatter = CitationFormatter()

    with open(output_file, 'w', encoding='utf-8') as f:
        for paper in papers:
            ris = formatter.to_ris(paper)
            f.write(ris + "\n")

    logger.info(f"Exported {len(papers)} papers to {output_file}")

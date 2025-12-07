"""
Direct HTTP-based arXiv API client.

This module provides an arXiv client that uses the arXiv API directly via HTTP,
avoiding the sgmllib3k dependency issue in the official arxiv package on Python 3.11+.

API Documentation: https://info.arxiv.org/help/api/user-manual.html
"""

import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
from urllib.parse import urlencode, quote

import httpx

from kosmos.literature.base_client import (
    BaseLiteratureClient,
    PaperMetadata,
    PaperSource,
    Author
)
from kosmos.literature.cache import get_cache
from kosmos.config import get_config

logger = logging.getLogger(__name__)

# ArXiv API constants
ARXIV_API_BASE = "http://export.arxiv.org/api/query"
ARXIV_RATE_LIMIT_SECONDS = 3.0  # ArXiv asks for no more than 1 request per 3 seconds
MAX_RESULTS_PER_PAGE = 2000  # ArXiv API limit

# XML namespaces used in ArXiv Atom feed
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
OPENSEARCH_NS = "{http://a9.com/-/spec/opensearch/1.1/}"


@dataclass
class ArxivSearchResult:
    """Intermediate representation of an arXiv search result."""
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    primary_category: str
    published: datetime
    updated: datetime
    doi: Optional[str]
    journal_ref: Optional[str]
    comment: Optional[str]
    pdf_url: str
    entry_url: str


class ArxivHTTPClient(BaseLiteratureClient):
    """
    Direct HTTP-based arXiv API client.

    This client uses httpx to query the arXiv API directly, parsing the
    Atom 1.0 XML response without relying on the arxiv Python package
    (which has sgmllib3k compatibility issues on Python 3.11+).

    Features:
    - No sgmllib3k dependency
    - Built-in rate limiting (3 second delay between requests)
    - Response caching
    - Full search query syntax support

    Example:
        ```python
        client = ArxivHTTPClient()

        # Simple search
        papers = client.search("large language models", max_results=10)

        # Category-specific search
        papers = client.search("quantum computing", fields=["quant-ph"])

        # Get specific paper
        paper = client.get_paper_by_id("2103.00020")
        ```
    """

    def __init__(self, api_key: Optional[str] = None, cache_enabled: bool = True):
        """
        Initialize the arXiv HTTP client.

        Args:
            api_key: Not used for arXiv (public API), kept for interface consistency
            cache_enabled: Whether to enable caching for API responses
        """
        super().__init__(api_key=api_key, cache_enabled=cache_enabled)

        # Get configuration
        config = get_config()
        self.max_results = config.literature.max_results_per_query

        # Initialize HTTP client with timeout
        self.http_client = httpx.Client(
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
            headers={
                "User-Agent": "Kosmos-AI-Scientist/1.0 (https://github.com/jimmc414/Kosmos)"
            }
        )

        # Initialize cache if enabled
        self.cache = get_cache() if cache_enabled else None

        # Track last request time for rate limiting
        self._last_request_time = 0.0

        self.logger.info("Initialized arXiv HTTP client (Python 3.11+ compatible)")

    def _rate_limit(self) -> None:
        """
        Enforce rate limiting per arXiv's Terms of Use.

        ArXiv asks for no more than one request every 3 seconds.
        """
        current_time = time.time()
        elapsed = current_time - self._last_request_time

        if elapsed < ARXIV_RATE_LIMIT_SECONDS:
            sleep_time = ARXIV_RATE_LIMIT_SECONDS - elapsed
            self.logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self._last_request_time = time.time()

    def search(
        self,
        query: str,
        max_results: int = 10,
        fields: Optional[List[str]] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        **kwargs
    ) -> List[PaperMetadata]:
        """
        Search for papers on arXiv using direct HTTP API.

        Args:
            query: Search query (supports arXiv query syntax)
            max_results: Maximum number of results to return
            fields: Optional filter by categories (e.g., ["cs.AI", "cs.LG"])
            year_from: Optional start year filter (post-filtering)
            year_to: Optional end year filter (post-filtering)
            **kwargs: Additional options:
                - sort_by: "relevance", "lastUpdatedDate", "submittedDate"
                - sort_order: "ascending", "descending"
                - start: Starting index for pagination

        Returns:
            List of PaperMetadata objects

        Example:
            ```python
            # Simple search
            papers = client.search("transformer attention mechanism", max_results=20)

            # Advanced search with categories
            papers = client.search(
                "neural network",
                fields=["cs.LG", "cs.AI"],
                year_from=2022,
                max_results=50
            )
            ```
        """
        if not self._validate_query(query):
            return []

        # Check cache
        cache_params = {
            "query": query,
            "max_results": max_results,
            "fields": fields,
            "year_from": year_from,
            "year_to": year_to
        }

        if self.cache:
            cached_result = self.cache.get("arxiv_http", "search", cache_params)
            if cached_result is not None:
                self.logger.debug("Cache hit for arXiv search")
                return cached_result

        try:
            # Build search query
            search_query = self._build_search_query(query, fields)

            # Build API parameters
            params = {
                "search_query": search_query,
                "start": kwargs.get("start", 0),
                "max_results": min(max_results, self.max_results, MAX_RESULTS_PER_PAGE),
            }

            # Add sorting
            sort_by = kwargs.get("sort_by", "relevance")
            sort_order = kwargs.get("sort_order", "descending")

            if sort_by != "relevance":
                params["sortBy"] = sort_by
                params["sortOrder"] = sort_order

            # Rate limit before request
            self._rate_limit()

            # Make API request
            url = f"{ARXIV_API_BASE}?{urlencode(params)}"
            self.logger.debug(f"ArXiv API request: {url}")

            response = self.http_client.get(url)
            response.raise_for_status()

            # Parse XML response
            results = self._parse_atom_feed(response.text)

            # Convert to PaperMetadata
            papers = [self._result_to_metadata(r) for r in results]

            # Post-filter by year if specified
            if year_from or year_to:
                papers = self._filter_by_year(papers, year_from, year_to)

            # Cache results
            if self.cache:
                self.cache.set("arxiv_http", "search", cache_params, papers)

            self.logger.info(f"Found {len(papers)} papers on arXiv for query: {query}")
            return papers

        except httpx.HTTPStatusError as e:
            self._handle_api_error(e, f"search query='{query}'")
            return []
        except Exception as e:
            self._handle_api_error(e, f"search query='{query}'")
            return []

    def get_paper_by_id(self, paper_id: str) -> Optional[PaperMetadata]:
        """
        Retrieve a specific paper by arXiv ID.

        Args:
            paper_id: arXiv ID (e.g., "2103.00020" or "arXiv:2103.00020")

        Returns:
            PaperMetadata object or None if not found

        Example:
            ```python
            paper = client.get_paper_by_id("2103.00020")
            if paper:
                print(f"Title: {paper.title}")
            ```
        """
        # Normalize paper ID
        paper_id = self._normalize_arxiv_id(paper_id)

        # Check cache
        cache_params = {"paper_id": paper_id}

        if self.cache:
            cached_result = self.cache.get("arxiv_http", "get_paper", cache_params)
            if cached_result is not None:
                return cached_result

        try:
            # Build API URL with id_list parameter
            params = {
                "id_list": paper_id,
                "max_results": 1
            }

            # Rate limit before request
            self._rate_limit()

            url = f"{ARXIV_API_BASE}?{urlencode(params)}"
            self.logger.debug(f"ArXiv API request: {url}")

            response = self.http_client.get(url)
            response.raise_for_status()

            # Parse XML response
            results = self._parse_atom_feed(response.text)

            if not results:
                self.logger.warning(f"Paper not found: {paper_id}")
                return None

            paper = self._result_to_metadata(results[0])

            # Cache result
            if self.cache:
                self.cache.set("arxiv_http", "get_paper", cache_params, paper)

            return paper

        except Exception as e:
            self._handle_api_error(e, f"get_paper_by_id id={paper_id}")
            return None

    def get_paper_references(self, paper_id: str, max_refs: int = 50) -> List[PaperMetadata]:
        """
        Get papers cited by the given paper.

        Note: arXiv API doesn't provide citation information directly.
        Use Semantic Scholar API for citation data.

        Args:
            paper_id: arXiv ID
            max_refs: Maximum number of references (unused)

        Returns:
            Empty list (arXiv doesn't provide citations)
        """
        self.logger.warning("arXiv API does not provide citation data. Use Semantic Scholar instead.")
        return []

    def get_paper_citations(self, paper_id: str, max_cites: int = 50) -> List[PaperMetadata]:
        """
        Get papers that cite the given paper.

        Note: arXiv API doesn't provide citation information directly.
        Use Semantic Scholar API for citation data.

        Args:
            paper_id: arXiv ID
            max_cites: Maximum number of citations (unused)

        Returns:
            Empty list (arXiv doesn't provide citations)
        """
        self.logger.warning("arXiv API does not provide citation data. Use Semantic Scholar instead.")
        return []

    def _build_search_query(
        self,
        query: str,
        fields: Optional[List[str]] = None
    ) -> str:
        """
        Build arXiv query string with category filters.

        ArXiv query syntax:
        - all:keyword - Search all fields
        - ti:keyword - Title only
        - au:author - Author name
        - abs:keyword - Abstract only
        - cat:cs.AI - Category filter
        - AND, OR, ANDNOT - Boolean operators

        Args:
            query: Base query string
            fields: arXiv categories to filter by

        Returns:
            Formatted query string
        """
        # If query doesn't have field prefix, search all fields
        if ":" not in query:
            query = f"all:{query}"

        # Add category filters
        if fields:
            category_queries = [f"cat:{field}" for field in fields]
            category_part = f"({' OR '.join(category_queries)})"
            query = f"({query}) AND {category_part}"

        return query

    def _normalize_arxiv_id(self, paper_id: str) -> str:
        """
        Normalize arXiv ID to standard format.

        Handles:
        - "arXiv:2103.00020" -> "2103.00020"
        - "2103.00020v2" -> "2103.00020"
        - "hep-th/9901001" -> "hep-th/9901001"

        Args:
            paper_id: Raw paper ID

        Returns:
            Normalized paper ID
        """
        # Remove "arXiv:" prefix
        paper_id = paper_id.replace("arXiv:", "").replace("arxiv:", "").strip()

        # Remove version suffix (e.g., "v2")
        paper_id = re.sub(r'v\d+$', '', paper_id)

        return paper_id

    def _parse_atom_feed(self, xml_content: str) -> List[ArxivSearchResult]:
        """
        Parse arXiv Atom feed XML into search results.

        Args:
            xml_content: Raw XML response from arXiv API

        Returns:
            List of ArxivSearchResult objects
        """
        results = []

        try:
            root = ET.fromstring(xml_content)

            # Find all entry elements
            for entry in root.findall(f"{ATOM_NS}entry"):
                result = self._parse_entry(entry)
                if result:
                    results.append(result)

        except ET.ParseError as e:
            self.logger.error(f"Failed to parse arXiv XML: {e}")

        return results

    def _parse_entry(self, entry: ET.Element) -> Optional[ArxivSearchResult]:
        """
        Parse a single Atom entry into ArxivSearchResult.

        Args:
            entry: XML Element for an entry

        Returns:
            ArxivSearchResult or None if parsing fails
        """
        try:
            # Extract arXiv ID from entry URL
            entry_url = entry.findtext(f"{ATOM_NS}id", "")
            arxiv_id = entry_url.split("/abs/")[-1] if "/abs/" in entry_url else ""

            # Remove version from ID
            arxiv_id = re.sub(r'v\d+$', '', arxiv_id)

            # Title (clean whitespace)
            title = entry.findtext(f"{ATOM_NS}title", "")
            title = " ".join(title.split())

            # Abstract/summary (clean whitespace)
            abstract = entry.findtext(f"{ATOM_NS}summary", "")
            abstract = " ".join(abstract.split())

            # Authors
            authors = []
            for author in entry.findall(f"{ATOM_NS}author"):
                name = author.findtext(f"{ATOM_NS}name", "")
                if name:
                    authors.append(name)

            # Categories
            categories = []
            primary_category = ""
            for category in entry.findall(f"{ATOM_NS}category"):
                term = category.get("term", "")
                if term:
                    categories.append(term)
            # Primary category from arxiv namespace
            primary_cat_elem = entry.find(f"{ARXIV_NS}primary_category")
            if primary_cat_elem is not None:
                primary_category = primary_cat_elem.get("term", "")
            elif categories:
                primary_category = categories[0]

            # Dates
            published_str = entry.findtext(f"{ATOM_NS}published", "")
            updated_str = entry.findtext(f"{ATOM_NS}updated", "")

            published = self._parse_datetime(published_str)
            updated = self._parse_datetime(updated_str)

            # DOI (from arxiv namespace)
            doi = entry.findtext(f"{ARXIV_NS}doi")

            # Journal reference
            journal_ref = entry.findtext(f"{ARXIV_NS}journal_ref")

            # Comment
            comment = entry.findtext(f"{ARXIV_NS}comment")

            # PDF URL
            pdf_url = ""
            for link in entry.findall(f"{ATOM_NS}link"):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")
                    break

            return ArxivSearchResult(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=authors,
                categories=categories,
                primary_category=primary_category,
                published=published,
                updated=updated,
                doi=doi,
                journal_ref=journal_ref,
                comment=comment,
                pdf_url=pdf_url,
                entry_url=entry_url
            )

        except Exception as e:
            self.logger.error(f"Failed to parse entry: {e}")
            return None

    def _parse_datetime(self, dt_str: str) -> datetime:
        """
        Parse ISO datetime string from arXiv.

        Args:
            dt_str: ISO format datetime string

        Returns:
            datetime object (defaults to epoch if parsing fails)
        """
        if not dt_str:
            return datetime(1970, 1, 1)

        try:
            # Handle timezone suffix
            dt_str = dt_str.replace("Z", "+00:00")
            return datetime.fromisoformat(dt_str)
        except ValueError:
            try:
                # Try without timezone
                return datetime.fromisoformat(dt_str[:19])
            except ValueError:
                return datetime(1970, 1, 1)

    def _result_to_metadata(self, result: ArxivSearchResult) -> PaperMetadata:
        """
        Convert ArxivSearchResult to PaperMetadata.

        Args:
            result: ArxivSearchResult object

        Returns:
            PaperMetadata object
        """
        # Convert author strings to Author objects
        authors = [Author(name=name) for name in result.authors]

        return PaperMetadata(
            id=result.arxiv_id,
            source=PaperSource.ARXIV,
            doi=result.doi,
            arxiv_id=result.arxiv_id,
            title=result.title,
            abstract=result.abstract,
            authors=authors,
            publication_date=result.published,
            journal=result.journal_ref,
            year=result.published.year if result.published else None,
            url=result.entry_url,
            pdf_url=result.pdf_url,
            fields=[cat.lower() for cat in result.categories],
            raw_data={
                "entry_id": result.entry_url,
                "updated": result.updated.isoformat() if result.updated else None,
                "comment": result.comment,
                "primary_category": result.primary_category
            }
        )

    def _filter_by_year(
        self,
        papers: List[PaperMetadata],
        year_from: Optional[int],
        year_to: Optional[int]
    ) -> List[PaperMetadata]:
        """
        Filter papers by publication year.

        Args:
            papers: List of papers to filter
            year_from: Minimum year (inclusive)
            year_to: Maximum year (inclusive)

        Returns:
            Filtered list of papers
        """
        filtered = []
        for paper in papers:
            if paper.year is None:
                continue

            if year_from and paper.year < year_from:
                continue
            if year_to and paper.year > year_to:
                continue

            filtered.append(paper)

        return filtered

    def get_categories(self) -> List[str]:
        """
        Get list of common arXiv categories.

        Returns:
            List of category codes

        Categories reference: https://arxiv.org/category_taxonomy
        """
        return [
            # Computer Science
            "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", "cs.RO",
            "cs.HC", "cs.SE", "cs.DB", "cs.IR",
            # Physics
            "physics.gen-ph", "physics.comp-ph", "quant-ph",
            "cond-mat", "hep-th", "hep-ph", "astro-ph",
            # Biology
            "q-bio.BM", "q-bio.GN", "q-bio.NC", "q-bio.QM",
            # Mathematics
            "math.ST", "math.OC", "math.PR",
            # Statistics
            "stat.ML", "stat.TH", "stat.ME",
            # Economics/Finance
            "econ.EM", "q-fin.ST"
        ]

    def close(self) -> None:
        """Close the HTTP client connection."""
        self.http_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

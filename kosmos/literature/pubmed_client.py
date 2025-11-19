"""
PubMed API client for searching and retrieving biomedical literature.

Uses Biopython's Entrez utilities to access NCBI PubMed database.
"""

from Bio import Entrez, Medline
from typing import List, Optional
from datetime import datetime
import time

from kosmos.literature.base_client import (
    BaseLiteratureClient,
    PaperMetadata,
    PaperSource,
    Author
)
from kosmos.literature.cache import get_cache
from kosmos.config import get_config


class PubMedClient(BaseLiteratureClient):
    """
    Client for interacting with PubMed/NCBI E-utilities API.

    PubMed is the premier database for biomedical literature with over 35M citations.

    API Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
    """

    def __init__(self, api_key: Optional[str] = None, cache_enabled: bool = True, email: Optional[str] = None):
        """
        Initialize the PubMed client.

        Args:
            api_key: Optional NCBI API key (increases rate limit to 10 req/sec)
            cache_enabled: Whether to enable caching for API responses
            email: Email address (recommended by NCBI for tracking)

        Note:
            NCBI requires an email address for E-utilities. Without an API key,
            limit is 3 requests/second. With API key, limit is 10 requests/second.
        """
        super().__init__(api_key=api_key, cache_enabled=cache_enabled)

        # Get configuration
        config = get_config()
        self.max_results = config.literature.max_results_per_query

        # Configure Entrez
        Entrez.api_key = api_key or config.literature.pubmed_api_key
        Entrez.email = email or config.literature.pubmed_email or "kosmos@example.com"

        # Rate limiting (requests per second)
        self.rate_limit = 10 if Entrez.api_key else 3
        self.min_delay = 1.0 / self.rate_limit
        self.last_request_time = 0

        # Initialize cache if enabled
        self.cache = get_cache() if cache_enabled else None

        self.logger.info(
            f"Initialized PubMed client (email={Entrez.email}, "
            f"rate_limit={self.rate_limit} req/s)"
        )

    def _rate_limit_delay(self):
        """Apply rate limiting delay."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self.last_request_time = time.time()

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
        Search for papers on PubMed.

        Args:
            query: Search query (supports PubMed query syntax)
            max_results: Maximum number of results to return
            fields: Not used for PubMed (kept for interface consistency)
            year_from: Optional start year filter
            year_to: Optional end year filter
            **kwargs: Additional options:
                - retmax: Max results to retrieve at once (default: 100)
                - sort: Sort order ("pub_date", "relevance")

        Returns:
            List of PaperMetadata objects

        Example:
            ```python
            client = PubMedClient(email="your-email@example.com")

            # Simple search
            papers = client.search("CRISPR gene editing", max_results=10)

            # Year range search
            papers = client.search(
                "machine learning diagnosis",
                year_from=2020,
                year_to=2024,
                sort="pub_date"
            )
            ```
        """
        if not self._validate_query(query):
            return []

        # Check cache
        cache_params = {
            "query": query,
            "max_results": max_results,
            "year_from": year_from,
            "year_to": year_to
        }

        if self.cache:
            cached_result = self.cache.get("pubmed", "search", cache_params)
            if cached_result is not None:
                return cached_result

        try:
            # Build query with date filter
            search_query = self._build_query(query, year_from, year_to)

            # Search for PMIDs
            self._rate_limit_delay()
            handle = Entrez.esearch(
                db="pubmed",
                term=search_query,
                retmax=min(max_results, self.max_results),
                sort=kwargs.get("sort", "relevance")
            )
            record = Entrez.read(handle)
            handle.close()

            # Validate response structure
            if not record or "IdList" not in record:
                self.logger.warning(f"Invalid PubMed response for query: {query}")
                return []

            pmids = record["IdList"]

            if not pmids:
                self.logger.info(f"No results found for query: {query}")
                return []

            # Fetch paper details
            papers = self._fetch_paper_details(pmids)

            # Cache results
            if self.cache:
                self.cache.set("pubmed", "search", cache_params, papers)

            self.logger.info(f"Found {len(papers)} papers on PubMed for query: {query}")
            return papers

        except Exception as e:
            self._handle_api_error(e, f"search query='{query}'")
            return []

    def get_paper_by_id(self, paper_id: str) -> Optional[PaperMetadata]:
        """
        Retrieve a specific paper by PubMed ID (PMID).

        Args:
            paper_id: PubMed ID (e.g., "19872477" or "PMID:19872477")

        Returns:
            PaperMetadata object or None if not found

        Example:
            ```python
            paper = client.get_paper_by_id("19872477")
            ```
        """
        # Remove "PMID:" prefix if present
        pmid = paper_id.replace("PMID:", "").strip()

        # Check cache
        cache_params = {"paper_id": pmid}

        if self.cache:
            cached_result = self.cache.get("pubmed", "get_paper", cache_params)
            if cached_result is not None:
                return cached_result

        try:
            papers = self._fetch_paper_details([pmid])

            if not papers:
                self.logger.warning(f"Paper not found: {pmid}")
                return None

            paper = papers[0]

            # Cache result
            if self.cache:
                self.cache.set("pubmed", "get_paper", cache_params, paper)

            return paper

        except Exception as e:
            self._handle_api_error(e, f"get_paper_by_id id={pmid}")
            return None

    def get_paper_references(self, paper_id: str, max_refs: int = 50) -> List[PaperMetadata]:
        """
        Get papers cited by the given paper.

        Args:
            paper_id: PubMed ID
            max_refs: Maximum number of references to return

        Returns:
            List of PaperMetadata objects for referenced papers

        Example:
            ```python
            references = client.get_paper_references("19872477")
            ```
        """
        pmid = paper_id.replace("PMID:", "").strip()

        # Check cache
        cache_params = {"paper_id": pmid, "max_refs": max_refs}

        if self.cache:
            cached_result = self.cache.get("pubmed", "get_references", cache_params)
            if cached_result is not None:
                return cached_result

        try:
            # Use elink to get references
            self._rate_limit_delay()
            handle = Entrez.elink(
                dbfrom="pubmed",
                db="pubmed",
                id=pmid,
                linkname="pubmed_pubmed_refs"
            )
            record = Entrez.read(handle)
            handle.close()

            # Validate response structure
            if not record or len(record) == 0:
                return []

            link_set_db = record[0].get("LinkSetDb")
            if not link_set_db or len(link_set_db) == 0:
                return []

            links = link_set_db[0].get("Link", [])
            if not links:
                return []

            # Extract PMIDs of references
            ref_pmids = [link["Id"] for link in links if "Id" in link][:max_refs]

            # Fetch paper details
            papers = self._fetch_paper_details(ref_pmids)

            # Cache results
            if self.cache:
                self.cache.set("pubmed", "get_references", cache_params, papers)

            self.logger.info(f"Retrieved {len(papers)} references for PMID {pmid}")
            return papers

        except Exception as e:
            self._handle_api_error(e, f"get_paper_references id={pmid}")
            return []

    def get_paper_citations(self, paper_id: str, max_cites: int = 50) -> List[PaperMetadata]:
        """
        Get papers that cite the given paper.

        Args:
            paper_id: PubMed ID
            max_cites: Maximum number of citations to return

        Returns:
            List of PaperMetadata objects for citing papers

        Example:
            ```python
            citations = client.get_paper_citations("19872477")
            ```
        """
        pmid = paper_id.replace("PMID:", "").strip()

        # Check cache
        cache_params = {"paper_id": pmid, "max_cites": max_cites}

        if self.cache:
            cached_result = self.cache.get("pubmed", "get_citations", cache_params)
            if cached_result is not None:
                return cached_result

        try:
            # Use elink to get citations
            self._rate_limit_delay()
            handle = Entrez.elink(
                dbfrom="pubmed",
                db="pubmed",
                id=pmid,
                linkname="pubmed_pubmed_citedin"
            )
            record = Entrez.read(handle)
            handle.close()

            if not record or not record[0].get("LinkSetDb"):
                return []

            # Extract PMIDs of citing papers
            cite_pmids = [link["Id"] for link in record[0]["LinkSetDb"][0]["Link"]][:max_cites]

            # Fetch paper details
            papers = self._fetch_paper_details(cite_pmids)

            # Cache results
            if self.cache:
                self.cache.set("pubmed", "get_citations", cache_params, papers)

            self.logger.info(f"Retrieved {len(papers)} citations for PMID {pmid}")
            return papers

        except Exception as e:
            self._handle_api_error(e, f"get_paper_citations id={pmid}")
            return []

    def _build_query(self, query: str, year_from: Optional[int], year_to: Optional[int]) -> str:
        """
        Build PubMed query with date filters.

        Args:
            query: Base query
            year_from: Start year
            year_to: End year

        Returns:
            Formatted query string
        """
        if year_from and year_to:
            return f"{query} AND {year_from}:{year_to}[pdat]"
        elif year_from:
            return f"{query} AND {year_from}:3000[pdat]"
        elif year_to:
            return f"{query} AND 1800:{year_to}[pdat]"
        return query

    def _fetch_paper_details(self, pmids: List[str]) -> List[PaperMetadata]:
        """
        Fetch detailed information for a list of PMIDs.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of PaperMetadata objects
        """
        if not pmids:
            return []

        try:
            # Fetch in batches of 100
            papers = []
            batch_size = 100

            for i in range(0, len(pmids), batch_size):
                batch_pmids = pmids[i:i + batch_size]

                self._rate_limit_delay()
                handle = Entrez.efetch(
                    db="pubmed",
                    id=",".join(batch_pmids),
                    rettype="medline",
                    retmode="text"
                )

                records = Medline.parse(handle)

                for record in records:
                    paper = self._medline_to_metadata(record)
                    if paper:
                        papers.append(paper)

                handle.close()

            return papers

        except Exception as e:
            self.logger.error(f"Error fetching paper details: {e}")
            return []

    def _medline_to_metadata(self, record: dict) -> Optional[PaperMetadata]:
        """
        Convert Medline record to PaperMetadata.

        Args:
            record: Medline record dictionary

        Returns:
            PaperMetadata object or None if invalid
        """
        try:
            # Extract PMID
            pmid = record.get("PMID", "")
            if not pmid:
                return None

            # Extract authors
            authors = []
            for author_name in record.get("AU", []):
                authors.append(Author(name=author_name))

            # Parse publication date
            pub_date = None
            date_str = record.get("DP", "")
            try:
                # Try to parse "2023 Jan 15" format
                pub_date = datetime.strptime(date_str.split()[0], "%Y")
            except (ValueError, IndexError):
                pass

            # Extract year
            year = pub_date.year if pub_date else None

            # Get DOI
            doi = None
            for aid in record.get("AID", []):
                if "[doi]" in aid.lower():
                    doi = aid.replace("[doi]", "").strip()
                    break

            return PaperMetadata(
                id=pmid,
                source=PaperSource.PUBMED,
                doi=doi,
                pubmed_id=pmid,
                title=record.get("TI", ""),
                abstract=record.get("AB", ""),
                authors=authors,
                publication_date=pub_date,
                journal=record.get("TA", ""),
                year=year,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                keywords=record.get("MH", []),  # MeSH terms as keywords
                raw_data=record
            )

        except Exception as e:
            self.logger.warning(f"Error parsing Medline record: {e}")
            return None

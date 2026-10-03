"""
Novelty Checker for hypotheses.

Checks if generated hypotheses are novel by:
1. Searching existing literature for similar claims
2. Comparing semantic similarity with known hypotheses
3. Detecting prior art
4. Generating novelty scores and reports
"""

import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

from kosmos.models.hypothesis import Hypothesis, NoveltyReport
from kosmos.literature.unified_search import UnifiedLiteratureSearch
from kosmos.literature.base_client import PaperMetadata, PaperSource
from kosmos.knowledge.embeddings import get_embedder
from kosmos.knowledge.vector_db import get_vector_db
from kosmos.db.models import Hypothesis as DBHypothesis
from kosmos.db import get_session

logger = logging.getLogger(__name__)

# TF-IDF similarity when no embedding model is loaded; Jaccard if scikit-learn is missing
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    TfidfVectorizer = None

# Most recent same-domain hypotheses compared against each new one
MAX_EXISTING_HYPOTHESES = 500


def _clamp_similarity(similarity: float) -> float:
    """Clamp to [0, 1]; a non-finite value (nan from 0/0) means no evidence of similarity."""
    if not np.isfinite(similarity):
        return 0.0
    return float(max(0.0, min(1.0, similarity)))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [0, 1]; 0.0 when either vector is zero."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return _clamp_similarity(np.dot(a, b) / (na * nb))


def _fit_tfidf(corpus: List[str]):
    """Fit TF-IDF on the corpus; None when nothing but stop words remains."""
    try:
        return TfidfVectorizer(
            ngram_range=(1, 2), stop_words="english", sublinear_tf=True
        ).fit_transform(corpus)
    except ValueError:  # empty vocabulary
        return None


def _row_cosine(matrix, i: int, j: int) -> float:
    """Cosine of two TF-IDF rows (rows are L2-normalized, so the dot product)."""
    return _clamp_similarity(matrix[i].multiply(matrix[j]).sum())


class NoveltyChecker:
    """
    Check novelty of hypotheses against existing work.

    Uses semantic similarity, literature search, and prior art detection
    to assess how novel a hypothesis is.

    Example:
        ```python
        checker = NoveltyChecker(similarity_threshold=0.75)

        report = checker.check_novelty(hypothesis)

        if report.is_novel:
            print(f"Novel hypothesis! Score: {report.novelty_score}")
        else:
            print(f"Similar to: {report.similar_papers[0]['title']}")
        ```
    """

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        max_similar_papers: int = 10,
        use_vector_db: bool = True
    ):
        """
        Initialize novelty checker.

        Args:
            similarity_threshold: Similarity above this is considered "similar" (0.0-1.0)
            max_similar_papers: Maximum similar papers to return in report
            use_vector_db: Whether to use vector DB for similarity search
        """
        self.similarity_threshold = similarity_threshold
        self.max_similar_papers = max_similar_papers
        self.use_vector_db = use_vector_db

        # Components. An embedder without a loaded model returns zero vectors,
        # so treat it as absent and use TF-IDF similarity instead.
        self.literature_search = UnifiedLiteratureSearch()
        emb = get_embedder() if use_vector_db else None
        self.embedder = emb if (emb is not None and emb.is_available) else None
        self.vector_db = get_vector_db() if use_vector_db else None

        # TF-IDF state for the no-embedding path: the hypothesis pool of the
        # current check, fitted once, plus a cache for pairs outside the pool
        self._pool_texts: List[str] = []
        self._pool_matrix = None
        self._pool_index: Dict[str, int] = {}
        self._pair_cache: Dict[Tuple[str, str], float] = {}

        logger.info(f"Initialized NoveltyChecker with threshold={similarity_threshold}")

    def check_novelty(self, hypothesis: Hypothesis) -> NoveltyReport:
        """
        Check novelty of a hypothesis.

        Args:
            hypothesis: Hypothesis to check

        Returns:
            NoveltyReport: Detailed novelty analysis

        Example:
            ```python
            report = checker.check_novelty(hypothesis)
            print(f"Novelty: {report.novelty_score:.2f}")
            print(f"Prior art: {report.prior_art_detected}")
            ```
        """
        logger.info(f"Checking novelty for hypothesis: {hypothesis.statement[:50]}...")
        self._set_similarity_pool([])

        # Step 1: Search literature for similar work
        similar_papers = self._search_similar_literature(hypothesis)

        # Step 2: Check against existing hypotheses in database
        similar_hypotheses = self._check_existing_hypotheses(hypothesis)

        # Step 3: Compute semantic similarity scores
        max_paper_similarity = 0.0
        if similar_papers:
            max_paper_similarity = max(
                self._compute_similarity(hypothesis, paper)
                for paper in similar_papers
            )

        max_hypothesis_similarity = 0.0
        if similar_hypotheses:
            max_hypothesis_similarity = max(
                self._compute_hypothesis_similarity(hypothesis, existing)
                for existing in similar_hypotheses
            )

        max_similarity = max(max_paper_similarity, max_hypothesis_similarity)

        # Step 4: Detect prior art (near-duplicates)
        prior_art_detected = max_similarity >= self.similarity_threshold

        # Step 5: Calculate novelty score
        # Score decreases as similarity increases
        # 1.0 = completely novel, 0.0 = exact duplicate
        if max_similarity >= 0.95:
            novelty_score = 0.0  # Essentially a duplicate
        elif max_similarity >= self.similarity_threshold:
            # Linear decay from threshold to 0.95
            novelty_score = 1.0 - ((max_similarity - self.similarity_threshold) / (0.95 - self.similarity_threshold))
            novelty_score = max(0.0, novelty_score * 0.5)  # Cap at 0.5 for similar work
        else:
            # Below threshold: high novelty score
            novelty_score = 1.0 - (max_similarity * 0.5)  # Scale so 0.0 similarity = 1.0 novelty

        novelty_score = max(0.0, min(1.0, novelty_score))

        # Step 6: Generate human-readable summary
        summary = self._generate_summary(
            novelty_score=novelty_score,
            prior_art_detected=prior_art_detected,
            max_similarity=max_similarity,
            similar_papers=similar_papers,
            similar_hypotheses=similar_hypotheses
        )

        # Step 7: Prepare similar work details (filter out None papers)
        similar_papers_info = [
            {
                "title": paper.title or "Untitled",
                "authors": (paper.authors or [])[:3],  # First 3 authors
                "year": paper.year,
                "source": paper.source,
                "similarity": self._compute_similarity(hypothesis, paper),
                "doi": paper.doi,
                "arxiv_id": paper.arxiv_id
            }
            for paper in similar_papers[:self.max_similar_papers]
            if paper is not None and paper.title
        ]

        similar_hypotheses_info = [
            {
                "statement": hyp.statement,
                "domain": hyp.domain,
                "created_at": hyp.created_at.isoformat() if hyp.created_at else None,
                "similarity": self._compute_hypothesis_similarity(hypothesis, hyp),
                "id": hyp.id
            }
            for hyp in similar_hypotheses[:5]  # Limit to 5
        ]

        # Update hypothesis with novelty score
        hypothesis.novelty_score = novelty_score

        return NoveltyReport(
            hypothesis_id=hypothesis.id or "unknown",
            novelty_score=novelty_score,
            similar_hypotheses=similar_hypotheses_info,
            similar_papers=similar_papers_info,
            max_similarity=max_similarity,
            prior_art_detected=prior_art_detected,
            is_novel=novelty_score >= (1.0 - self.similarity_threshold),  # Inverse of threshold
            novelty_threshold_used=self.similarity_threshold,
            summary=summary
        )

    def _search_similar_literature(self, hypothesis: Hypothesis) -> List[PaperMetadata]:
        """
        Search literature for papers related to the hypothesis.

        Args:
            hypothesis: Hypothesis to search for

        Returns:
            List[PaperMetadata]: Similar papers
        """
        try:
            # Use vector DB for semantic search if available
            if self.use_vector_db and self.vector_db and self.embedder:
                return self._vector_search_papers(hypothesis)

            # Fallback: keyword search
            query = f"{hypothesis.statement} {hypothesis.rationale[:100]}"
            papers = self.literature_search.search(
                query=query,
                max_results=20
            )

            logger.info(f"Found {len(papers)} similar papers via keyword search")

            # Index retrieved papers into vector DB for future semantic searches
            if papers and self.vector_db:
                try:
                    self.vector_db.add_papers(papers)
                    logger.info(f"Indexed {len(papers)} papers into vector DB")
                except Exception as e:
                    logger.warning(f"Failed to index papers into vector DB: {e}")

            return papers

        except Exception as e:
            logger.error(f"Error searching literature: {e}", exc_info=True)
            return []

    def _vector_search_papers(self, hypothesis: Hypothesis) -> List[PaperMetadata]:
        """
        Use vector database to find semantically similar papers.

        Args:
            hypothesis: Hypothesis to search for

        Returns:
            List[PaperMetadata]: Similar papers
        """
        try:
            # Create search query from hypothesis
            query = f"{hypothesis.statement}. {hypothesis.rationale}"

            # Search vector DB
            results = self.vector_db.search(query, top_k=20)

            # Convert results to PaperMetadata. The stored metadata holds no
            # abstract or authors (vector_db._paper_metadata); the document
            # text is "title [SEP] abstract".
            papers = []
            for result in results:
                metadata = result.get("metadata") or {}
                title = metadata.get("title", "")
                paper_id = (
                    metadata.get("id")
                    or metadata.get("doi")
                    or metadata.get("arxiv_id")
                    or f"vec_{hashlib.sha1(title.encode('utf-8')).hexdigest()[:12]}"
                )
                try:
                    source = PaperSource(metadata.get("source", "unknown"))
                except ValueError:
                    source = PaperSource.UNKNOWN
                paper = PaperMetadata(
                    id=paper_id,
                    source=source,
                    title=title,
                    abstract=result.get("document", ""),
                    year=metadata.get("year") or None,
                    doi=metadata.get("doi"),
                    arxiv_id=metadata.get("arxiv_id"),
                    pubmed_id=metadata.get("pubmed_id"),
                )
                papers.append(paper)

            logger.info(f"Found {len(papers)} similar papers via vector search")
            return papers

        except Exception as e:
            logger.error(f"Error in vector search: {e}", exc_info=True)
            return []

    def _check_existing_hypotheses(self, hypothesis: Hypothesis) -> List[Hypothesis]:
        """
        Check against existing hypotheses in database.

        Args:
            hypothesis: Hypothesis to check

        Returns:
            List[Hypothesis]: Similar existing hypotheses
        """
        try:
            with get_session() as session:
                # Most recent hypotheses in the same domain, excluding this one
                query = session.query(DBHypothesis).filter(
                    DBHypothesis.domain == hypothesis.domain
                )
                if hypothesis.id:
                    query = query.filter(DBHypothesis.id != hypothesis.id)
                db_hypotheses = query.order_by(
                    DBHypothesis.created_at.desc()
                ).limit(MAX_EXISTING_HYPOTHESES).all()

                # Convert to Pydantic models
                existing_hypotheses = []
                for db_hyp in db_hypotheses:
                    hyp = Hypothesis(
                        id=db_hyp.id,
                        research_question=db_hyp.research_question,
                        statement=db_hyp.statement,
                        rationale=db_hyp.rationale,
                        domain=db_hyp.domain,
                        created_at=db_hyp.created_at,
                        updated_at=db_hyp.updated_at
                    )
                    existing_hypotheses.append(hyp)

                self._set_similarity_pool(
                    [hypothesis.statement] + [h.statement for h in existing_hypotheses]
                )

                # Filter by similarity (lower threshold for preliminary filtering),
                # then sort highest first
                scored = []
                for existing in existing_hypotheses:
                    similarity = self._compute_hypothesis_similarity(hypothesis, existing)
                    if similarity >= 0.5:
                        scored.append((similarity, existing))
                scored.sort(key=lambda pair: pair[0], reverse=True)
                similar = [existing for _, existing in scored]

                logger.info(f"Found {len(similar)} similar existing hypotheses")
                return similar

        except Exception as e:
            logger.error(f"Error checking existing hypotheses: {e}", exc_info=True)
            return []

    def _compute_similarity(
        self,
        hypothesis: Hypothesis,
        paper: PaperMetadata
    ) -> float:
        """
        Compute semantic similarity between hypothesis and paper.

        Args:
            hypothesis: Hypothesis
            paper: Paper to compare

        Returns:
            float: Similarity score (0.0-1.0)
        """
        try:
            # Guard against None paper or missing title
            if paper is None or not paper.title:
                return 0.0

            paper_title = paper.title or ""
            paper_abstract = paper.abstract or ""

            if not self.embedder:
                # Fallback: simple keyword overlap
                return self._keyword_similarity(hypothesis.statement, paper_title + " " + paper_abstract)

            # Use embeddings for semantic similarity
            hyp_text = f"{hypothesis.statement}. {hypothesis.rationale}"
            paper_text = f"{paper_title}. {paper_abstract}"

            hyp_embedding = self.embedder.embed_query(hyp_text)
            paper_embedding = self.embedder.embed_query(paper_text)

            return _cosine_similarity(hyp_embedding, paper_embedding)

        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

    def _compute_hypothesis_similarity(
        self,
        hyp1: Hypothesis,
        hyp2: Hypothesis
    ) -> float:
        """
        Compute similarity between two hypotheses.

        Args:
            hyp1: First hypothesis
            hyp2: Second hypothesis

        Returns:
            float: Similarity score (0.0-1.0)
        """
        try:
            if not self.embedder:
                # Fallback: simple keyword similarity
                return self._keyword_similarity(hyp1.statement, hyp2.statement)

            # Use embeddings
            text1 = f"{hyp1.statement}. {hyp1.rationale}"
            text2 = f"{hyp2.statement}. {hyp2.rationale}"

            emb1 = self.embedder.embed_query(text1)
            emb2 = self.embedder.embed_query(text2)

            return _cosine_similarity(emb1, emb2)

        except Exception as e:
            logger.error(f"Error computing hypothesis similarity: {e}")
            return 0.0

    def _set_similarity_pool(self, texts: List[str]) -> None:
        """
        Set the hypothesis pool that TF-IDF weights are fitted on.

        Fits once so every pair inside the pool is a lookup; pairs with a text
        outside the pool (papers) are fitted on demand and cached.

        Args:
            texts: Statements in the current pool (empty to reset)
        """
        self._pool_texts = list(dict.fromkeys(t for t in texts if t and t.strip()))
        self._pool_matrix = None
        self._pool_index = {}
        self._pair_cache = {}
        if TfidfVectorizer is None or len(self._pool_texts) < 2:
            return
        self._pool_matrix = _fit_tfidf(self._pool_texts)
        if self._pool_matrix is not None:
            self._pool_index = {t: i for i, t in enumerate(self._pool_texts)}

    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """
        TF-IDF cosine similarity (fallback when no embedding model is loaded).

        Uses word unigrams and bigrams with English stop words removed, fitted
        on the pair plus the current hypothesis pool, so words every hypothesis
        in the domain shares carry little weight. Falls back to Jaccard word
        overlap when scikit-learn is unavailable.

        Args:
            text1: First text
            text2: Second text

        Returns:
            float: Similarity score (0.0-1.0)
        """
        if TfidfVectorizer is None:
            return self._jaccard_similarity(text1, text2)
        if not (text1 and text1.strip() and text2 and text2.strip()):
            return 0.0

        i, j = self._pool_index.get(text1), self._pool_index.get(text2)
        if i is not None and j is not None:
            return _row_cosine(self._pool_matrix, i, j)

        key = (text1, text2)
        if key not in self._pair_cache:
            corpus = [text1, text2] + [t for t in self._pool_texts if t not in key]
            matrix = _fit_tfidf(corpus)
            self._pair_cache[key] = _row_cosine(matrix, 0, 1) if matrix is not None else 0.0
        return self._pair_cache[key]

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Jaccard word-overlap similarity (used when scikit-learn is unavailable).

        Args:
            text1: First text
            text2: Second text

        Returns:
            float: Similarity score (0.0-1.0)
        """
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Remove common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "by", "with"}
        words1 = words1 - stopwords
        words2 = words2 - stopwords

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _generate_summary(
        self,
        novelty_score: float,
        prior_art_detected: bool,
        max_similarity: float,
        similar_papers: List[PaperMetadata],
        similar_hypotheses: List[Hypothesis]
    ) -> str:
        """
        Generate human-readable novelty summary.

        Args:
            novelty_score: Calculated novelty score
            prior_art_detected: Whether prior art was detected
            max_similarity: Maximum similarity found
            similar_papers: Similar papers
            similar_hypotheses: Similar existing hypotheses

        Returns:
            str: Summary text
        """
        if prior_art_detected:
            if similar_hypotheses:
                return (
                    f"LOW NOVELTY (score: {novelty_score:.2f}). "
                    f"Very similar to existing hypothesis: '{similar_hypotheses[0].statement[:80]}...'. "
                    f"Maximum similarity: {max_similarity:.2f}. "
                    f"Consider revising or expanding this hypothesis."
                )
            elif similar_papers and len(similar_papers) > 0:
                paper_title = similar_papers[0].title if similar_papers[0].title else "Untitled"
                paper_year = similar_papers[0].year if similar_papers[0].year else "unknown"
                return (
                    f"LOW NOVELTY (score: {novelty_score:.2f}). "
                    f"Very similar to existing work: '{paper_title}' ({paper_year}). "
                    f"Maximum similarity: {max_similarity:.2f}. "
                    f"This hypothesis may already be addressed in the literature."
                )
            else:
                return (
                    f"LOW NOVELTY (score: {novelty_score:.2f}). "
                    f"High similarity detected (max: {max_similarity:.2f}) but source unclear."
                )

        elif novelty_score >= 0.8:
            return (
                f"HIGH NOVELTY (score: {novelty_score:.2f}). "
                f"This hypothesis appears highly novel with low similarity to existing work (max: {max_similarity:.2f}). "
                f"No significant prior art detected."
            )

        elif novelty_score >= 0.6:
            summary = (
                f"MODERATE NOVELTY (score: {novelty_score:.2f}). "
                f"Some similar work exists (max similarity: {max_similarity:.2f}), "
                f"but this hypothesis offers a distinct perspective. "
            )

            if similar_papers:
                summary += f"Related papers found: {len(similar_papers)}. "

            if similar_hypotheses:
                summary += f"Similar existing hypotheses: {len(similar_hypotheses)}. "

            return summary + "Consider emphasizing the novel aspects."

        else:
            return (
                f"MODERATE-LOW NOVELTY (score: {novelty_score:.2f}). "
                f"Considerable similarity to existing work (max: {max_similarity:.2f}). "
                f"Found {len(similar_papers)} related papers and {len(similar_hypotheses)} similar hypotheses. "
                f"Consider how this hypothesis extends or differs from prior work."
            )


def check_hypothesis_novelty(
    hypothesis: Hypothesis,
    similarity_threshold: float = 0.75
) -> NoveltyReport:
    """
    Convenience function to check hypothesis novelty.

    Args:
        hypothesis: Hypothesis to check
        similarity_threshold: Similarity threshold (default: 0.75)

    Returns:
        NoveltyReport: Novelty analysis report

    Example:
        ```python
        report = check_hypothesis_novelty(my_hypothesis)
        if report.is_novel:
            print("Novel hypothesis!")
        ```
    """
    checker = NoveltyChecker(similarity_threshold=similarity_threshold)
    return checker.check_novelty(hypothesis)

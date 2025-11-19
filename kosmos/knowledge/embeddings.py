"""
SPECTER-based embeddings for scientific papers.

Uses the allenai/specter model optimized for scientific document similarity.
SPECTER is trained on citation graphs and produces 768-dimensional embeddings.
"""

from typing import List, Optional, Union
import numpy as np
from pathlib import Path
import logging

from kosmos.literature.base_client import PaperMetadata

logger = logging.getLogger(__name__)

# Optional dependency - sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    logger.warning("sentence_transformers not installed. Install with: pip install sentence-transformers")
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None


class PaperEmbedder:
    """
    Generate embeddings for scientific papers using SPECTER.

    SPECTER (Scientific Paper Embeddings using Citation-informed TransformERs)
    produces high-quality embeddings for scientific papers that capture semantic
    similarity better than general-purpose models.

    Model: allenai/specter (768-dim)
    Paper: https://arxiv.org/abs/2004.07180
    """

    def __init__(
        self,
        model_name: str = "allenai/specter",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the paper embedder.

        Args:
            model_name: Model name or path (default: "allenai/specter")
            cache_dir: Directory to cache model files (default: ~/.cache/huggingface)
            device: Device to use ("cuda", "cpu", or None for auto)

        Note:
            First run will download ~440MB model. Subsequent runs use cached version.
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            logger.warning("SentenceTransformers not available. PaperEmbedder will not function.")
            self.model = None
            self.model_name = model_name
            self.embedding_dim = 768  # Default SPECTER dimension
            return

        self.model_name = model_name

        # Set cache directory if provided
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading SPECTER model: {model_name}")
        logger.info("First run may take a few minutes to download model (~440MB)")

        try:
            self.model = SentenceTransformer(
                model_name,
                cache_folder=cache_dir,
                device=device
            )
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

            logger.info(
                f"Loaded SPECTER model (embedding_dim={self.embedding_dim}, "
                f"device={self.model.device})"
            )

        except Exception as e:
            logger.error(f"Error loading SPECTER model: {e}")
            raise

    def embed_paper(self, paper: PaperMetadata) -> np.ndarray:
        """
        Generate embedding for a single paper.

        Uses title + abstract for embedding generation (SPECTER's recommended input).

        Args:
            paper: PaperMetadata object

        Returns:
            768-dimensional embedding vector

        Example:
            ```python
            embedder = PaperEmbedder()
            embedding = embedder.embed_paper(paper)
            print(embedding.shape)  # (768,)
            ```
        """
        text = self._paper_to_text(paper)

        # Check if model is available
        if self.model is None:
            logger.warning("Embedding model not available. Returning zero vector.")
            return np.zeros(self.embedding_dim, dtype=np.float32)

        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embedding

        except Exception as e:
            logger.error(f"Error embedding paper {paper.id}: {e}")
            # Return zero vector on error
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def embed_papers(
        self,
        papers: List[PaperMetadata],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple papers in batches.

        Args:
            papers: List of PaperMetadata objects
            batch_size: Batch size for encoding (default: 32)
            show_progress: Whether to show progress bar

        Returns:
            Array of shape (n_papers, 768)

        Example:
            ```python
            embeddings = embedder.embed_papers(papers, batch_size=32)
            print(embeddings.shape)  # (n_papers, 768)

            # Calculate similarity between first two papers
            similarity = np.dot(embeddings[0], embeddings[1])
            ```
        """
        if not papers:
            return np.array([])

        # Check if model is available
        if self.model is None:
            logger.warning("Embedding model not available. Returning zero vectors.")
            return np.zeros((len(papers), self.embedding_dim), dtype=np.float32)

        texts = [self._paper_to_text(paper) for paper in papers]

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress
            )

            logger.info(f"Generated embeddings for {len(papers)} papers")
            return embeddings

        except Exception as e:
            logger.error(f"Error embedding papers: {e}")
            # Return zero vectors on error
            return np.zeros((len(papers), self.embedding_dim), dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.

        Args:
            query: Search query string

        Returns:
            768-dimensional embedding vector

        Example:
            ```python
            query_embedding = embedder.embed_query("machine learning for drug discovery")

            # Find most similar papers
            similarities = np.dot(paper_embeddings, query_embedding)
            top_indices = np.argsort(similarities)[::-1][:5]
            ```
        """
        # Check if model is available
        if self.model is None:
            logger.warning("Embedding model not available. Returning zero vector.")
            return np.zeros(self.embedding_dim, dtype=np.float32)

        try:
            embedding = self.model.encode(
                query,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embedding

        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between -1 and 1 (higher = more similar)

        Example:
            ```python
            similarity = embedder.compute_similarity(emb1, emb2)
            print(f"Similarity: {similarity:.3f}")
            ```
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        return float(similarity)

    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        paper_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar papers to a query.

        Args:
            query_embedding: Query embedding vector (768,)
            paper_embeddings: Paper embeddings array (n_papers, 768)
            top_k: Number of top results to return

        Returns:
            List of (index, similarity_score) tuples, sorted by similarity

        Example:
            ```python
            query_emb = embedder.embed_query("CRISPR gene editing")
            paper_embs = embedder.embed_papers(papers)

            top_papers = embedder.find_most_similar(query_emb, paper_embs, top_k=5)

            for idx, score in top_papers:
                print(f"{papers[idx].title}: {score:.3f}")
            ```
        """
        if len(paper_embeddings) == 0:
            return []

        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Normalize papers
        paper_norms = paper_embeddings / (
            np.linalg.norm(paper_embeddings, axis=1, keepdims=True) + 1e-8
        )

        # Compute similarities
        similarities = np.dot(paper_norms, query_norm)

        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return (index, score) tuples
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]

        return results

    def _paper_to_text(self, paper: PaperMetadata) -> str:
        """
        Convert paper to text for embedding.

        SPECTER is trained on title + abstract, so we use that format.

        Args:
            paper: PaperMetadata object

        Returns:
            Formatted text string
        """
        # SPECTER format: title [SEP] abstract
        title = paper.title.strip() if paper.title else ""
        abstract = paper.abstract.strip() if paper.abstract else ""

        if title and abstract:
            # Truncate abstract if too long (SPECTER has 512 token limit)
            if len(abstract.split()) > 400:
                abstract_words = abstract.split()[:400]
                abstract = " ".join(abstract_words) + "..."

            return f"{title} [SEP] {abstract}"

        elif title:
            return title

        elif abstract:
            return abstract

        else:
            logger.warning(f"Paper {paper.id} has no title or abstract")
            return ""


# Singleton embedder instance
_embedder: Optional[PaperEmbedder] = None


def get_embedder(
    model_name: str = "allenai/specter",
    cache_dir: Optional[str] = None,
    device: Optional[str] = None
) -> PaperEmbedder:
    """
    Get or create the singleton embedder instance.

    Args:
        model_name: Model name or path
        cache_dir: Cache directory for model files
        device: Device to use

    Returns:
        PaperEmbedder instance
    """
    global _embedder
    if _embedder is None:
        _embedder = PaperEmbedder(
            model_name=model_name,
            cache_dir=cache_dir,
            device=device
        )
    return _embedder


def reset_embedder():
    """Reset the singleton embedder (useful for testing)."""
    global _embedder
    _embedder = None

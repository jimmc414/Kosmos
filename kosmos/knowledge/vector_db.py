"""
Vector database interface using ChromaDB.

Stores and retrieves paper embeddings for semantic search.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
from pathlib import Path
import logging

from kosmos.literature.base_client import PaperMetadata
from kosmos.knowledge.embeddings import get_embedder
from kosmos.config import get_config

logger = logging.getLogger(__name__)

# Optional dependency - chromadb
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    logger.warning("chromadb not installed. Install with: pip install chromadb")
    HAS_CHROMADB = False
    chromadb = None
    Settings = None


class PaperVectorDB:
    """
    Vector database for storing and searching paper embeddings.

    Uses ChromaDB for persistent vector storage with semantic search capabilities.
    """

    def __init__(
        self,
        collection_name: str = "papers",
        persist_directory: Optional[str] = None,
        reset: bool = False
    ):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the collection (default: "papers")
            persist_directory: Directory to persist database (default: from config)
            reset: Whether to reset/clear the collection on init

        Example:
            ```python
            db = PaperVectorDB()

            # Add papers
            db.add_papers(papers)

            # Search
            results = db.search("machine learning", top_k=10)
            ```
        """
        if not HAS_CHROMADB:
            logger.warning("ChromaDB not available. PaperVectorDB will not function.")
            self.client = None
            self.collection = None
            self.collection_name = collection_name
            return

        config = get_config()

        # Set persist directory
        if persist_directory is None:
            persist_directory = config.vector_db.chroma_persist_directory

        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        self.collection_name = collection_name

        if reset:
            try:
                self.client.delete_collection(name=collection_name)
                logger.info(f"Reset collection: {collection_name}")
            except Exception:
                pass

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        # Initialize embedder
        self.embedder = get_embedder()

        logger.info(
            f"Initialized PaperVectorDB (collection={collection_name}, "
            f"persist_dir={persist_directory}, count={self.collection.count()})"
        )

    def add_paper(
        self,
        paper: PaperMetadata,
        embedding: Optional[np.ndarray] = None
    ):
        """
        Add a single paper to the vector database.

        Args:
            paper: PaperMetadata object
            embedding: Optional pre-computed embedding (if None, will compute)

        Example:
            ```python
            db.add_paper(paper)
            ```
        """
        self.add_papers([paper], embeddings=[embedding] if embedding is not None else None)

    def add_papers(
        self,
        papers: List[PaperMetadata],
        embeddings: Optional[np.ndarray] = None,
        batch_size: int = 100
    ):
        """
        Add multiple papers to the vector database.

        Args:
            papers: List of PaperMetadata objects
            embeddings: Optional pre-computed embeddings (if None, will compute)
            batch_size: Batch size for insertion

        Example:
            ```python
            # Let it compute embeddings
            db.add_papers(papers)

            # Or provide pre-computed embeddings
            embeddings = embedder.embed_papers(papers)
            db.add_papers(papers, embeddings=embeddings)
            ```
        """
        if not papers:
            return

        # Compute embeddings if not provided
        if embeddings is None:
            logger.info(f"Computing embeddings for {len(papers)} papers")
            embeddings = self.embedder.embed_papers(papers, show_progress=True)

        # Prepare data
        ids = [self._paper_id(paper) for paper in papers]
        metadatas = [self._paper_metadata(paper) for paper in papers]
        documents = [self._paper_document(paper) for paper in papers]

        # Check if collection is available
        if self.collection is None:
            logger.warning("Vector database collection not available. Cannot add papers.")
            return

        # Add in batches
        for i in range(0, len(papers), batch_size):
            batch_end = min(i + batch_size, len(papers))

            self.collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end].tolist(),
                metadatas=metadatas[i:batch_end],
                documents=documents[i:batch_end]
            )

        logger.info(f"Added {len(papers)} papers to vector database")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for papers by query string.

        Args:
            query: Search query
            top_k: Number of top results to return
            filters: Optional metadata filters (e.g., {"domain": "biology", "year": {"$gte": 2020}})

        Returns:
            List of search results with paper metadata and similarity scores

        Example:
            ```python
            # Simple search
            results = db.search("CRISPR gene editing", top_k=10)

            # With filters
            results = db.search(
                "quantum computing",
                top_k=5,
                filters={"domain": "physics", "year": {"$gte": 2020}}
            )

            for result in results:
                print(f"{result['title']}: {result['score']:.3f}")
            ```
        """
        # Check if collection is available
        if self.collection is None:
            logger.warning("Vector database collection not available. Cannot search.")
            return []

        # Compute query embedding
        query_embedding = self.embedder.embed_query(query)

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filters
        )

        # Format results
        formatted_results = []

        if results and results["ids"] and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                result = {
                    "id": results["ids"][0][i],
                    "score": float(1 - results["distances"][0][i]) if "distances" in results else 1.0,  # Convert distance to similarity
                    "metadata": results["metadatas"][0][i] if "metadatas" in results else {},
                    "document": results["documents"][0][i] if "documents" in results else ""
                }
                formatted_results.append(result)

        return formatted_results

    def search_by_paper(
        self,
        paper: PaperMetadata,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find similar papers to a given paper.

        Args:
            paper: Paper to find similar papers for
            top_k: Number of top results to return
            filters: Optional metadata filters

        Returns:
            List of similar papers

        Example:
            ```python
            similar_papers = db.search_by_paper(paper, top_k=5)
            ```
        """
        # Get paper embedding
        paper_embedding = self.embedder.embed_paper(paper)

        # Search
        results = self.collection.query(
            query_embeddings=[paper_embedding.tolist()],
            n_results=top_k + 1,  # +1 to account for self-match
            where=filters
        )

        # Format results (exclude self)
        formatted_results = []
        paper_id = self._paper_id(paper)

        if results and results["ids"] and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                result_id = results["ids"][0][i]

                # Skip self
                if result_id == paper_id:
                    continue

                result = {
                    "id": result_id,
                    "score": float(1 - results["distances"][0][i]) if "distances" in results else 1.0,
                    "metadata": results["metadatas"][0][i] if "metadatas" in results else {},
                    "document": results["documents"][0][i] if "documents" in results else ""
                }
                formatted_results.append(result)

                if len(formatted_results) >= top_k:
                    break

        return formatted_results

    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a paper by ID.

        Args:
            paper_id: Paper identifier

        Returns:
            Paper data or None if not found
        """
        try:
            result = self.collection.get(ids=[paper_id])

            if result and result["ids"] and len(result["ids"]) > 0:
                return {
                    "id": result["ids"][0],
                    "metadata": result["metadatas"][0] if "metadatas" in result else {},
                    "document": result["documents"][0] if "documents" in result else ""
                }

            return None

        except Exception as e:
            logger.error(f"Error getting paper {paper_id}: {e}")
            return None

    def delete_paper(self, paper_id: str):
        """
        Delete a paper from the database.

        Args:
            paper_id: Paper identifier
        """
        try:
            self.collection.delete(ids=[paper_id])
            logger.info(f"Deleted paper {paper_id}")
        except Exception as e:
            logger.error(f"Error deleting paper {paper_id}: {e}")

    def count(self) -> int:
        """
        Get the number of papers in the database.

        Returns:
            Paper count
        """
        if self.collection is None:
            logger.warning("Vector database collection not available.")
            return 0
        return self.collection.count()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "collection_name": self.collection_name,
            "paper_count": self.count(),
            "embedding_dim": self.embedder.embedding_dim
        }

    def clear(self):
        """Clear all papers from the database."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Cleared collection: {self.collection_name}")

    def _paper_id(self, paper: PaperMetadata) -> str:
        """
        Generate unique ID for a paper.

        Uses primary identifier (DOI > arXiv > PubMed > source ID).

        Args:
            paper: PaperMetadata object

        Returns:
            Unique paper ID
        """
        return f"{paper.source.value}:{paper.primary_identifier}"

    def _paper_metadata(self, paper: PaperMetadata) -> Dict[str, Any]:
        """
        Extract metadata for ChromaDB storage.

        Args:
            paper: PaperMetadata object

        Returns:
            Metadata dictionary
        """
        metadata = {
            "source": paper.source.value,
            "title": paper.title[:500] if paper.title else "",  # ChromaDB has size limits
            "year": paper.year or 0,
            "citation_count": paper.citation_count,
            "domain": paper.fields[0] if paper.fields else "unknown"
        }

        # Add identifiers
        if paper.doi:
            metadata["doi"] = paper.doi
        if paper.arxiv_id:
            metadata["arxiv_id"] = paper.arxiv_id
        if paper.pubmed_id:
            metadata["pubmed_id"] = paper.pubmed_id

        return metadata

    def _paper_document(self, paper: PaperMetadata) -> str:
        """
        Create document text for storage.

        Args:
            paper: PaperMetadata object

        Returns:
            Document string
        """
        # Store title + abstract
        parts = []

        if paper.title:
            parts.append(paper.title)

        if paper.abstract:
            # Truncate abstract if too long
            abstract = paper.abstract[:1000] + "..." if len(paper.abstract) > 1000 else paper.abstract
            parts.append(abstract)

        return " [SEP] ".join(parts)


# Singleton vector database instance
_vector_db: Optional[PaperVectorDB] = None


def get_vector_db(
    collection_name: str = "papers",
    persist_directory: Optional[str] = None,
    reset: bool = False
) -> PaperVectorDB:
    """
    Get or create the singleton vector database instance.

    Args:
        collection_name: Collection name
        persist_directory: Persistence directory
        reset: Whether to reset the collection

    Returns:
        PaperVectorDB instance
    """
    global _vector_db
    if _vector_db is None or reset:
        _vector_db = PaperVectorDB(
            collection_name=collection_name,
            persist_directory=persist_directory,
            reset=reset
        )
    return _vector_db


def reset_vector_db():
    """Reset the singleton vector database (useful for testing)."""
    global _vector_db
    _vector_db = None

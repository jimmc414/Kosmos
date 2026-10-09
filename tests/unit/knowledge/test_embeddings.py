"""
Tests for kosmos.knowledge.embeddings module.

Model tests use REAL SentenceTransformer embeddings (not mocks) with the
smaller all-MiniLM-L6-v2 model. sentence-transformers is the optional
"embeddings" extra: when it cannot be imported, those tests skip and the
zero-vector fallback tests (which always run) cover the degraded path.
"""

import pytest
import numpy as np
import uuid

import kosmos.knowledge.embeddings as embeddings_module
from kosmos.knowledge.embeddings import PaperEmbedder, HAS_SENTENCE_TRANSFORMERS
from kosmos.literature.base_client import PaperMetadata, PaperSource


# The module's own import probe: it also catches a broken transformers/torch
# install (which raises more than ImportError), so pytest.importorskip alone
# is not sufficient here.
requires_sentence_transformers = pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS,
    reason="sentence-transformers (optional 'embeddings' extra) is not importable",
)


def unique_text(base: str) -> str:
    """Add unique suffix to avoid cache hits."""
    return f"{base} [test-id: {uuid.uuid4().hex[:8]}]"


@pytest.fixture
def paper_embedder():
    """Create PaperEmbedder instance with real SentenceTransformer."""
    # Use smaller model for faster tests
    embedder = PaperEmbedder(model_name="all-MiniLM-L6-v2")
    return embedder


@pytest.fixture
def specter_embedder():
    """Create PaperEmbedder with SPECTER model."""
    embedder = PaperEmbedder(model_name="allenai/specter")
    return embedder


@pytest.fixture
def fallback_embedder(monkeypatch):
    """PaperEmbedder in the degraded mode used when sentence-transformers is absent."""
    monkeypatch.setattr(embeddings_module, "HAS_SENTENCE_TRANSFORMERS", False)
    return PaperEmbedder()


@pytest.mark.unit
class TestFallbackWithoutSentenceTransformers:
    """Degraded behavior when sentence-transformers is unavailable (always runs)."""

    def test_init_without_model(self, fallback_embedder):
        assert fallback_embedder.model is None
        assert fallback_embedder.model_name == "allenai/specter"
        assert fallback_embedder.embedding_dim == 768  # SPECTER default
        assert fallback_embedder.is_available is False

    def test_init_custom_model_name_kept(self, monkeypatch):
        monkeypatch.setattr(embeddings_module, "HAS_SENTENCE_TRANSFORMERS", False)
        embedder = PaperEmbedder(model_name="all-MiniLM-L6-v2")
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.model is None
        assert embedder.is_available is False

    def test_embed_query_returns_zero_vector(self, fallback_embedder):
        embedding = fallback_embedder.embed_query(unique_text("test query"))

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)
        assert embedding.dtype == np.float32
        assert not embedding.any()

    def test_embed_paper_returns_zero_vector(self, fallback_embedder, sample_paper_metadata):
        embedding = fallback_embedder.embed_paper(sample_paper_metadata)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)
        assert not embedding.any()

    def test_embed_papers_returns_zero_matrix(self, fallback_embedder, sample_papers_list):
        embeddings = fallback_embedder.embed_papers(sample_papers_list)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (len(sample_papers_list), 768)
        assert not embeddings.any()

    def test_embed_papers_empty_list(self, fallback_embedder):
        embeddings = fallback_embedder.embed_papers([])

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.size == 0

    def test_failed_probe_unregisters_partial_modules(self, monkeypatch):
        """A failed sentence-transformers probe must not leave a half-imported
        ``transformers`` in sys.modules: shap.TreeExplainer inspects
        sys.modules['transformers'] and crashed on it (MaterialsOptimizer.shap_analysis)."""
        import importlib
        import sys
        import types

        class _BrokenProbeFinder:
            def find_spec(self, name, path=None, target=None):
                if name == "sentence_transformers":
                    # Simulate a broken install: transformers gets registered, then the import fails.
                    sys.modules["transformers"] = types.ModuleType("transformers")
                    sys.modules["transformers.modeling_utils"] = types.ModuleType("transformers.modeling_utils")
                    raise ModuleNotFoundError("Could not import module 'PreTrainedModel'")
                return None

        for name in [m for m in sys.modules if m.split(".")[0] in ("sentence_transformers", "transformers")]:
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.setattr(sys, "meta_path", [_BrokenProbeFinder()] + sys.meta_path)
        try:
            importlib.reload(embeddings_module)
            assert embeddings_module.HAS_SENTENCE_TRANSFORMERS is False
            assert embeddings_module.SentenceTransformer is None
            assert "transformers" not in sys.modules
            assert "transformers.modeling_utils" not in sys.modules
        finally:
            monkeypatch.undo()
            importlib.reload(embeddings_module)

    def test_zero_vectors_have_zero_similarity(self, fallback_embedder):
        """Zero-vector fallback must never look like a match."""
        emb1 = fallback_embedder.embed_query("machine learning")
        emb2 = fallback_embedder.embed_query("machine learning")

        assert fallback_embedder.compute_similarity(emb1, emb2) == 0.0


@pytest.mark.unit
@requires_sentence_transformers
class TestPaperEmbedderInit:
    """Test paper embedder initialization."""

    def test_init_default(self):
        """Test default initialization."""
        embedder = PaperEmbedder()
        assert embedder.model_name == "allenai/specter"
        assert embedder.model is not None

    def test_init_custom_model(self):
        """Test initialization with custom model."""
        embedder = PaperEmbedder(model_name="all-MiniLM-L6-v2")
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.model is not None


@pytest.mark.unit
@requires_sentence_transformers
class TestEmbeddingGeneration:
    """Test embedding generation."""

    def test_embed_query(self, paper_embedder):
        """Test embedding a query."""
        embedding = paper_embedder.embed_query(unique_text("test query"))

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 384  # MiniLM dimension
        assert embedding.dtype == np.float32 or embedding.dtype == np.float64

    def test_embed_paper(self, paper_embedder, sample_paper_metadata):
        """Test embedding a paper."""
        embedding = paper_embedder.embed_paper(sample_paper_metadata)

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 384  # MiniLM dimension

    def test_embed_papers_batch(self, paper_embedder, sample_papers_list):
        """Test batch embedding of papers."""
        embeddings = paper_embedder.embed_papers(sample_papers_list)

        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings) == len(sample_papers_list)
        assert embeddings.shape[1] == 384  # MiniLM dimension

    def test_embed_empty_query(self, paper_embedder):
        """Test embedding empty query."""
        embedding = paper_embedder.embed_query("")

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 384


@pytest.mark.unit
@requires_sentence_transformers
class TestEmbeddingBehavior:
    """Test embedding behavior."""

    def test_multiple_queries_different_results(self, paper_embedder):
        """Test that different queries produce different embeddings."""
        emb1 = paper_embedder.embed_query(unique_text("machine learning"))
        emb2 = paper_embedder.embed_query(unique_text("quantum physics"))

        # Different queries should have different embeddings
        assert not np.allclose(emb1, emb2)

    def test_similar_queries_similar_embeddings(self, paper_embedder):
        """Test that similar queries produce similar embeddings."""
        emb1 = paper_embedder.embed_query("neural network deep learning")
        emb2 = paper_embedder.embed_query("deep learning neural network")

        # Similar queries should have high cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        assert similarity > 0.8  # High similarity


@pytest.mark.unit
class TestEmbeddingSimilarity:
    """Test similarity calculations."""

    def test_compute_similarity(self, fallback_embedder):
        """Test similarity calculation."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])

        similarity = fallback_embedder.compute_similarity(vec1, vec2)

        assert 0.99 <= similarity <= 1.01  # Should be 1.0 (identical)

    def test_compute_similarity_orthogonal(self, fallback_embedder):
        """Test similarity for orthogonal vectors."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])

        similarity = fallback_embedder.compute_similarity(vec1, vec2)

        assert -0.01 <= similarity <= 0.01  # Should be 0.0 (orthogonal)

    def test_find_most_similar(self, fallback_embedder):
        """Test finding most similar papers."""
        # Create mock embeddings array
        paper_embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ])

        query_embedding = np.array([1.0, 0.0, 0.0])
        similar = fallback_embedder.find_most_similar(
            query_embedding, paper_embeddings, top_k=2
        )

        assert len(similar) <= 2
        assert all(isinstance(item, tuple) for item in similar)
        # First result should be most similar (index 0)
        assert similar[0][0] == 0


@pytest.mark.unit
@requires_sentence_transformers
class TestSpecterModel:
    """Test SPECTER model specifically."""

    def test_specter_embedding_dimension(self, specter_embedder):
        """Test SPECTER embedding dimension is 768."""
        embedding = specter_embedder.embed_query("test query")

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 768  # SPECTER dimension

    def test_specter_paper_embedding(self, specter_embedder, sample_paper_metadata):
        """Test SPECTER paper embedding."""
        embedding = specter_embedder.embed_paper(sample_paper_metadata)

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 768

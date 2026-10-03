"""Tests for novelty_checker module."""

import numpy as np
import pytest
from unittest.mock import MagicMock, Mock, patch

import kosmos.db as kosmos_db
from kosmos.db import get_session, init_database, operations
from kosmos.hypothesis.novelty_checker import NoveltyChecker
from kosmos.knowledge.embeddings import reset_embedder
from kosmos.literature.base_client import PaperMetadata, PaperSource
from kosmos.models.hypothesis import Hypothesis

@pytest.fixture
def novelty_checker():
    return NoveltyChecker(similarity_threshold=0.75, use_vector_db=False)

@pytest.fixture
def sample_hypothesis():
    return Hypothesis(
        research_question="Test question",
        statement="Attention mechanism improves transformer performance",
        rationale="Prior work shows attention captures dependencies",
        domain="machine_learning"
    )

@pytest.mark.unit
class TestNoveltyChecker:
    def test_init(self, novelty_checker):
        assert novelty_checker.similarity_threshold == 0.75

    @patch('kosmos.hypothesis.novelty_checker.UnifiedLiteratureSearch')
    @patch('kosmos.hypothesis.novelty_checker.get_session')
    def test_check_novelty_high(self, mock_session, mock_search, sample_hypothesis):
        mock_search_inst = Mock()
        mock_search_inst.search.return_value = []
        mock_search.return_value = mock_search_inst

        mock_sess = MagicMock()
        mock_sess.query.return_value.filter.return_value.order_by.return_value \
            .limit.return_value.all.return_value = []
        mock_session.return_value.__enter__.return_value = mock_sess

        # Built inside the patches so no live literature search runs
        novelty_checker = NoveltyChecker(similarity_threshold=0.75, use_vector_db=False)
        report = novelty_checker.check_novelty(sample_hypothesis)

        assert report.novelty_score >= 0.8
        assert report.is_novel is True
        assert len(report.similar_papers) == 0

    def test_keyword_similarity(self, novelty_checker):
        text1 = "attention mechanism transformer neural network"
        text2 = "transformer attention model deep learning"
        similarity = novelty_checker._keyword_similarity(text1, text2)
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0  # Some overlap exists


@pytest.mark.unit
class TestPaperIndexing:
    """Test that papers from keyword fallback are indexed into vector DB (D3 fix)."""

    @patch('kosmos.hypothesis.novelty_checker.UnifiedLiteratureSearch')
    def test_keyword_fallback_indexes_papers(self, mock_search):
        """Papers from keyword fallback should be indexed into vector DB."""
        from kosmos.literature.base_client import PaperMetadata, PaperSource

        mock_papers = [
            PaperMetadata(
                id="paper-1",
                source=PaperSource.SEMANTIC_SCHOLAR,
                title="Paper 1",
                authors=[],
                abstract="Abstract about transformers",
            ),
            PaperMetadata(
                id="paper-2",
                source=PaperSource.SEMANTIC_SCHOLAR,
                title="Paper 2",
                authors=[],
                abstract="Abstract about attention",
            ),
        ]

        mock_search_inst = Mock()
        mock_search_inst.search.return_value = mock_papers
        mock_search.return_value = mock_search_inst

        mock_vdb_inst = Mock()

        # use_vector_db=False so embedder is None (skips vector search path)
        # then manually set vector_db so indexing still works
        checker = NoveltyChecker(use_vector_db=False)
        checker.vector_db = mock_vdb_inst

        hypothesis = Hypothesis(
            research_question="Test question",
            statement="Transformers use attention mechanisms",
            rationale="Based on Vaswani et al. attention is all you need",
            domain="machine_learning",
        )

        papers = checker._search_similar_literature(hypothesis)

        assert len(papers) == 2
        # Verify add_papers was called with the retrieved papers
        mock_vdb_inst.add_papers.assert_called_once_with(mock_papers)

    @patch('kosmos.hypothesis.novelty_checker.UnifiedLiteratureSearch')
    def test_keyword_fallback_no_vector_db_skips_indexing(self, mock_search):
        """When vector DB is not available, indexing should be skipped."""
        from kosmos.literature.base_client import PaperMetadata, PaperSource

        mock_papers = [
            PaperMetadata(
                id="paper-1",
                source=PaperSource.SEMANTIC_SCHOLAR,
                title="Paper 1",
                authors=[],
                abstract="Abstract",
            ),
        ]

        mock_search_inst = Mock()
        mock_search_inst.search.return_value = mock_papers
        mock_search.return_value = mock_search_inst

        checker = NoveltyChecker(use_vector_db=False)

        hypothesis = Hypothesis(
            research_question="Test question",
            statement="Some hypothesis",
            rationale="Some rationale about the world and how it works",
            domain="biology",
        )

        papers = checker._search_similar_literature(hypothesis)

        assert len(papers) == 1
        # vector_db is None, so add_papers should not be called
        assert checker.vector_db is None

    @patch('kosmos.hypothesis.novelty_checker.UnifiedLiteratureSearch')
    def test_indexing_error_does_not_crash(self, mock_search):
        """If indexing fails, it should log warning but not crash."""
        from kosmos.literature.base_client import PaperMetadata, PaperSource

        mock_papers = [
            PaperMetadata(id="paper-1", source=PaperSource.SEMANTIC_SCHOLAR, title="Paper 1", authors=[], abstract="Abs"),
        ]

        mock_search_inst = Mock()
        mock_search_inst.search.return_value = mock_papers
        mock_search.return_value = mock_search_inst

        mock_vdb_inst = Mock()
        mock_vdb_inst.add_papers.side_effect = RuntimeError("ChromaDB error")

        # use_vector_db=False so we go through keyword path, then set vector_db manually
        checker = NoveltyChecker(use_vector_db=False)
        checker.vector_db = mock_vdb_inst

        hypothesis = Hypothesis(
            research_question="Test question",
            statement="Some hypothesis statement here",
            rationale="Rationale for the hypothesis being tested",
            domain="chemistry",
        )

        # Should not raise despite indexing failure
        papers = checker._search_similar_literature(hypothesis)
        assert len(papers) == 1


def _climate_hypothesis(statement, rationale="Radiative forcing from greenhouse gases warms the lower atmosphere"):
    return Hypothesis(
        research_question="Does CO2 predict temperature anomaly?",
        statement=statement,
        rationale=rationale,
        domain="climate",
    )


@pytest.fixture
def no_embeddings():
    """sentence-transformers absent, no vector DB, literature search mocked to return nothing."""
    reset_embedder()
    with patch('kosmos.knowledge.embeddings.HAS_SENTENCE_TRANSFORMERS', False), \
         patch('kosmos.hypothesis.novelty_checker.get_vector_db', return_value=None), \
         patch('kosmos.hypothesis.novelty_checker.UnifiedLiteratureSearch') as mock_search:
        mock_search.return_value.search.return_value = []
        yield mock_search
    reset_embedder()


@pytest.fixture
def in_memory_db():
    """Point kosmos.db at a fresh in-memory SQLite database; restore the old engine afterwards."""
    saved = (kosmos_db._engine, kosmos_db._SessionLocal)
    init_database("sqlite:///:memory:")
    yield
    kosmos_db.reset_database()
    kosmos_db._engine, kosmos_db._SessionLocal = saved


@pytest.mark.unit
class TestNoveltyWithoutEmbeddings:
    """P2-5: novelty scoring is correct when sentence-transformers is missing."""

    def test_embedder_is_none_without_model(self, no_embeddings):
        assert NoveltyChecker().embedder is None

    def test_tfidf_similarity_separates_copies_from_unrelated(self, no_embeddings):
        checker = NoveltyChecker()
        h = _climate_hypothesis("Atmospheric CO2 concentration positively predicts global temperature anomaly")
        h_copy = h.model_copy()
        unrelated = Hypothesis(
            research_question="What shapes the infant gut microbiome?",
            statement="Antibiotic exposure in infancy decreases gut microbiome diversity",
            rationale="Broad-spectrum antibiotics deplete commensal bacterial populations",
            domain="biology",
        )

        assert checker._compute_hypothesis_similarity(h, h_copy) > 0.95
        assert checker._compute_hypothesis_similarity(h, unrelated) < 0.2

    def test_zero_vector_embedder_gives_zero_similarity(self, no_embeddings):
        checker = NoveltyChecker()
        checker.embedder = Mock(model=object(), embed_query=Mock(return_value=np.zeros(768)))
        h = _climate_hypothesis("Atmospheric CO2 concentration positively predicts global temperature anomaly")
        paper = PaperMetadata(id="p1", source=PaperSource.ARXIV, title="T", abstract="A")

        # As in production: pytest.ini turns numpy's 0/0 warning into an exception
        # that the similarity methods would catch, masking a nan result
        with np.errstate(divide='ignore', invalid='ignore'):
            assert checker._compute_similarity(h, paper) == 0.0
            assert checker._compute_hypothesis_similarity(h, h.model_copy()) == 0.0

    def test_vector_search_builds_valid_papers(self, no_embeddings):
        checker = NoveltyChecker()
        checker.vector_db = Mock(search=Mock(return_value=[
            {'metadata': {'title': 'T', 'abstract': 'A', 'doi': '10.1/x'}},
            {'metadata': {'title': 'No identifiers', 'source': 'not-a-source'}, 'document': 'No identifiers [SEP] body'},
        ]))
        h = _climate_hypothesis("Atmospheric CO2 concentration positively predicts global temperature anomaly")

        papers = checker._vector_search_papers(h)

        assert len(papers) == 2
        assert papers[0].id == '10.1/x'
        assert papers[0].source == PaperSource.UNKNOWN
        assert papers[1].id.startswith('vec_')
        assert papers[1].source == PaperSource.UNKNOWN
        assert papers[1].abstract == 'No identifiers [SEP] body'

    def test_distinct_hypothesis_survives_generator_filter(self, no_embeddings, in_memory_db):
        from kosmos.agents.hypothesis_generator import HypothesisGeneratorAgent

        with get_session() as session:
            operations.create_hypothesis(
                session, id="hyp-existing", research_question="Does CO2 predict temperature anomaly?",
                statement="Atmospheric CO2 concentration positively predicts global temperature anomaly",
                rationale="Radiative forcing from greenhouse gases warms the lower atmosphere",
                domain="climate",
            )

        new = {
            "statement": "Higher volcanic aerosol index will lower the temperature anomaly in the following year",
            "rationale": "Stratospheric sulfate aerosols reflect incoming sunlight and cool the surface",
            "confidence_score": 0.6,
            "testability_score": 0.8,
            "suggested_experiment_types": ["data_analysis"],
        }
        with patch('kosmos.agents.hypothesis_generator.get_client') as mock_client:
            mock_client.return_value.generate_structured.return_value = {"hypotheses": [new]}
            agent = HypothesisGeneratorAgent(config={"use_literature_context": False})
            with np.errstate(divide='ignore', invalid='ignore'):  # production numpy behaviour
                response = agent.generate_hypotheses(
                    "Does CO2 predict temperature anomaly?", domain="climate", store_in_db=False
                )

        assert len(response.hypotheses) == 1
        assert response.hypotheses[0].novelty_score > 0.5

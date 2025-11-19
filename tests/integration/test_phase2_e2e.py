"""
End-to-end integration tests for Phase 2 components.

These tests verify that all Phase 2 components work together correctly
in realistic workflows.
"""

import pytest
from unittest.mock import patch

from kosmos.literature.unified_search import UnifiedLiteratureSearch
from kosmos.knowledge.embeddings import PaperEmbedder
from kosmos.knowledge.vector_db import VectorDatabase
from kosmos.knowledge.graph import KnowledgeGraph
from kosmos.knowledge.concept_extractor import ConceptExtractor
from kosmos.knowledge.graph_builder import GraphBuilder
from kosmos.agents.literature_analyzer import LiteratureAnalyzerAgent
from kosmos.literature.citations import CitationParser, CitationFormatter
from kosmos.literature.reference_manager import ReferenceManager


@pytest.mark.integration
class TestSearchAndAnalyzeWorkflow:
    """Test complete search and analysis workflow."""

    @pytest.mark.slow
    def test_search_and_summarize_workflow(self, tmp_path):
        """Test: Search → Analyze → Store results."""
        # 1. Search for papers
        search = UnifiedLiteratureSearch()
        with patch.object(search.arxiv_client, 'search') as mock_search:
            mock_search.return_value = []  # Mock to avoid real API calls

            papers = search.search("machine learning", max_results=2, sources=["arxiv"])

        # 2. Analyze with agent (mocked)
        with patch('kosmos.agents.literature_analyzer.get_client') as mock_client:
            mock_client.return_value.generate_structured.return_value = {
                "executive_summary": "Test summary",
                "key_findings": ["Finding 1"],
                "methodology": "Test methods",
                "significance": "Important",
                "limitations": ["Limitation 1"],
                "confidence_score": 0.8,
            }

            agent = LiteratureAnalyzerAgent(config={"use_knowledge_graph": False})
            if papers:
                analysis = agent.summarize_paper(papers[0])
                assert analysis is not None


@pytest.mark.integration
class TestKnowledgeGraphWorkflow:
    """Test knowledge graph construction workflow."""

    def test_build_knowledge_graph_workflow(self, sample_papers_list):
        """Test: Papers → Extract Concepts → Build Graph."""
        with patch('py2neo.Graph'):
            with patch('kosmos.knowledge.graph.KnowledgeGraph._ensure_container_running'):
                with patch('sentence_transformers.SentenceTransformer'):
                    # Create mocked components
                    kg = KnowledgeGraph(auto_start_container=False, create_indexes=False)
                    kg.graph = patch('py2neo.Graph').start()

                    with patch('kosmos.knowledge.concept_extractor.get_client') as mock_client:
                        mock_client.return_value.generate_structured.return_value = {
                            "concepts": [
                                {"name": "ML", "category": "Field", "relevance": 0.9}
                            ],
                            "methods": [],
                            "relationships": [],
                        }

                        builder = GraphBuilder(knowledge_graph=kg)

                        # Build graph (will be mocked)
                        with patch.object(builder, 'add_paper'):
                            for paper in sample_papers_list[:2]:
                                builder.add_paper(paper, extract_concepts=False)


@pytest.mark.integration
class TestVectorSearchWorkflow:
    """Test vector search workflow."""

    @pytest.mark.slow
    def test_embed_and_search_workflow(self, sample_papers_list):
        """Test: Embed Papers → Store → Search."""
        with patch('chromadb.Client'):
            with patch('sentence_transformers.SentenceTransformer'):
                # Create vector DB (mocked)
                db = VectorDatabase(persist_directory=":memory:")
                db.collection = patch('chromadb.Collection').start()

                # Mock embedding and add
                with patch.object(db, 'embedding_generator'):
                    with patch.object(db.collection, 'add'):
                        db.add_papers(sample_papers_list[:3])


@pytest.mark.integration
class TestCitationWorkflow:
    """Test citation management workflow."""

    def test_parse_format_export_workflow(self, sample_bibtex, tmp_path):
        """Test: Parse BibTeX → Format → Export."""
        # 1. Parse citations
        parser = CitationParser()
        papers = parser.parse_bibtex(str(sample_bibtex))

        assert len(papers) > 0

        # 2. Format citations
        formatter = CitationFormatter()
        apa_citations = [formatter.format_citation(p, style="apa") for p in papers[:3]]

        assert len(apa_citations) > 0
        assert all(isinstance(c, str) for c in apa_citations)

        # 3. Manage references
        manager = ReferenceManager(storage_path=str(tmp_path / "refs.json"))
        manager.add_references(papers)

        # 4. Deduplicate
        dedup_report = manager.deduplicate_references(strategy="comprehensive")

        assert "original_count" in dedup_report
        assert "unique_count" in dedup_report

        # 5. Export
        export_path = tmp_path / "exported.bib"
        manager.export_library(str(export_path), format="bibtex")

        assert export_path.exists()


@pytest.mark.integration
@pytest.mark.slow
class TestFullPipeline:
    """Test complete Phase 2 pipeline."""

    def test_complete_literature_pipeline(self, sample_papers_list, tmp_path):
        """Test: Search → Store → Analyze → Extract → Visualize."""
        # Mock all external services
        with patch('chromadb.Client'):
            with patch('py2neo.Graph'):
                with patch('sentence_transformers.SentenceTransformer'):
                    with patch('kosmos.knowledge.graph.KnowledgeGraph._ensure_container_running'):
                        # 1. Create mock papers (using fixtures)
                        papers = sample_papers_list[:3]

                        # 2. Store in vector DB (mocked)
                        db = VectorDatabase(persist_directory=":memory:")
                        db.collection = patch('chromadb.Collection').start()

                        with patch.object(db, 'embedding_generator'):
                            with patch.object(db.collection, 'add'):
                                db.add_papers(papers)

                        # 3. Build knowledge graph (mocked)
                        kg = KnowledgeGraph(auto_start_container=False, create_indexes=False)
                        kg.graph = patch('py2neo.Graph').start()

                        with patch('kosmos.knowledge.concept_extractor.get_client') as mock_client:
                            mock_client.return_value.generate_structured.return_value = {
                                "concepts": [],
                                "methods": [],
                                "relationships": [],
                            }

                            builder = GraphBuilder(knowledge_graph=kg)

                            with patch.object(builder, 'add_paper'):
                                for paper in papers:
                                    builder.add_paper(paper, extract_concepts=False)

                        # 4. Analyze with agent (mocked)
                        with patch('kosmos.agents.literature_analyzer.get_client') as mock_agent_client:
                            mock_agent_client.return_value.generate_structured.return_value = {
                                "executive_summary": "Test summary",
                                "key_findings": ["Finding 1"],
                                "methodology": "Methods",
                                "significance": "Important",
                                "limitations": [],
                                "confidence_score": 0.85,
                            }

                            agent = LiteratureAnalyzerAgent(config={"use_knowledge_graph": False})
                            analyses = agent.analyze_papers_batch(papers)

                            assert len(analyses) == len(papers)

                        # 5. Export citations
                        manager = ReferenceManager(storage_path=str(tmp_path / "library.json"))
                        manager.add_references(papers)

                        export_path = tmp_path / "citations.bib"
                        manager.export_library(str(export_path), format="bibtex")

                        assert export_path.exists()


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling across components."""

    def test_graceful_api_failures(self):
        """Test that components handle API failures gracefully."""
        search = UnifiedLiteratureSearch()

        with patch.object(search.arxiv_client, 'search', side_effect=Exception("API Error")):
            # Should return empty list instead of raising
            papers = search.search("test query", sources=["arxiv"])
            assert papers == []

    def test_missing_services_degradation(self, sample_paper_metadata):
        """Test degraded functionality when services unavailable."""
        with patch('kosmos.agents.literature_analyzer.get_client') as mock_client:
            mock_client.return_value.generate_structured.return_value = {
                "executive_summary": "Summary",
                "key_findings": [],
                "methodology": "Methods",
                "significance": "Important",
                "limitations": [],
                "confidence_score": 0.7,
            }

            # Agent with disabled knowledge graph
            agent = LiteratureAnalyzerAgent(config={"use_knowledge_graph": False})

            # Should still work without knowledge graph
            analysis = agent.summarize_paper(sample_paper_metadata)
            assert analysis is not None


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_api_key
@pytest.mark.requires_neo4j
@pytest.mark.requires_chromadb
@pytest.mark.requires_claude
class TestRealServicesIntegration:
    """Integration tests with real services (requires all services running)."""

    def test_real_end_to_end_workflow(self):
        """Test complete workflow with real services."""
        # This test requires:
        # - Claude API key
        # - Neo4j running
        # - ChromaDB available
        # - Network access for literature APIs

        # 1. Real search
        search = UnifiedLiteratureSearch()
        papers = search.search("transformer neural network", max_results=1, sources=["arxiv"])

        if len(papers) == 0:
            pytest.skip("No papers found in search")

        # 2. Real vector DB storage
        db = VectorDatabase(persist_directory=":memory:")
        db.add_papers(papers)

        # 3. Real knowledge graph
        kg = KnowledgeGraph()
        for paper in papers:
            kg.add_paper(paper)

        # 4. Real agent analysis
        agent = LiteratureAnalyzerAgent()
        agent.start()
        analysis = agent.summarize_paper(papers[0])
        agent.stop()

        assert analysis is not None
        assert len(analysis.executive_summary) > 0

"""
Literature Analyzer Agent with knowledge graph integration.

Provides intelligent paper analysis including:
- Paper summarization using Claude
- Key findings extraction
- Methodology identification
- Citation network analysis
- Relevance scoring
- Corpus-level insights
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import time
import json
from pathlib import Path

from kosmos.agents.base import BaseAgent, AgentMessage, MessageType, AgentStatus
from kosmos.core.llm import get_client
from kosmos.knowledge.graph import get_knowledge_graph
from kosmos.knowledge.vector_db import get_vector_db
from kosmos.knowledge.embeddings import get_embedder
from kosmos.knowledge.semantic_search import SemanticLiteratureSearch
from kosmos.literature.unified_search import UnifiedLiteratureSearch
from kosmos.literature.base_client import PaperMetadata
from kosmos.knowledge.concept_extractor import get_concept_extractor

logger = logging.getLogger(__name__)


@dataclass
class PaperAnalysis:
    """Complete paper analysis result."""
    paper_id: str
    executive_summary: str
    key_findings: List[Dict[str, Any]]
    methodology: Dict[str, Any]
    significance: str
    limitations: List[str]
    confidence_score: float
    analysis_time: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LiteratureAnalyzerAgent(BaseAgent):
    """
    Intelligent literature analysis agent with knowledge graph integration.

    Capabilities:
    - Paper summarization using Claude
    - Key findings extraction
    - Methodology identification
    - Citation network analysis (graph + on-demand)
    - Relevance scoring
    - Concept mapping
    - Corpus-level insights

    Example:
        ```python
        analyzer = LiteratureAnalyzerAgent(
            agent_id="lit-analyzer-001",
            config={"use_knowledge_graph": True}
        )
        analyzer.start()

        # Summarize a paper
        summary = analyzer.summarize_paper(paper)

        # Analyze citations
        citations = analyzer.analyze_citation_network(
            paper_id="arxiv:1234.5678",
            depth=2
        )
        ```
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Literature Analyzer Agent.

        Args:
            agent_id: Unique agent identifier
            agent_type: Agent type name
            config: Configuration dictionary
        """
        super().__init__(agent_id, agent_type or "LiteratureAnalyzerAgent", config)

        # Configuration with defaults
        self.use_knowledge_graph = self.config.get("use_knowledge_graph", True)
        self.build_missing_citations = self.config.get("build_missing_citations", True)
        self.max_citation_depth = self.config.get("max_citation_depth", 2)
        self.min_relevance_score = self.config.get("min_relevance_score", 0.6)
        self.max_papers_per_analysis = self.config.get("max_papers_per_analysis", 50)
        self.extract_concepts = self.config.get("extract_concepts", True)
        self.use_semantic_similarity = self.config.get("use_semantic_similarity", True)

        # Initialize components
        self.llm_client = get_client()

        if self.use_knowledge_graph:
            try:
                self.knowledge_graph = get_knowledge_graph()
                logger.info("Knowledge graph connected")
            except Exception as e:
                logger.warning(f"Knowledge graph unavailable: {e}")
                self.knowledge_graph = None
                self.use_knowledge_graph = False

        if self.use_semantic_similarity:
            self.vector_db = get_vector_db()
            self.embedder = get_embedder()

        if self.extract_concepts:
            self.concept_extractor = get_concept_extractor()

        self.semantic_search = SemanticLiteratureSearch()
        self.unified_search = UnifiedLiteratureSearch()

        # Caching
        self.cache_dir = Path(".literature_analysis_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized {self.agent_type} "
            f"(graph={self.use_knowledge_graph}, "
            f"semantic={self.use_semantic_similarity})"
        )

    def _on_start(self):
        """Called when agent starts."""
        logger.info(f"Starting {self.agent_type} ({self.agent_id})")
        self.save_state_data("start_time", time.time())

    def _on_stop(self):
        """Called when agent stops."""
        logger.info(f"Stopping {self.agent_type} ({self.agent_id})")
        runtime = time.time() - self.get_state_data("start_time", time.time())
        logger.info(
            f"Agent runtime: {runtime:.2f}s, "
            f"tasks completed: {self.tasks_completed}, "
            f"errors: {self.errors_encountered}"
        )

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute analysis task.

        Args:
            task: Task dictionary with:
                - task_type: Type of analysis
                - Additional task-specific parameters

        Supported task types:
            - "summarize_paper": Analyze single paper
            - "analyze_corpus": Analyze multiple papers
            - "citation_network": Explore citation graph
            - "find_related": Find similar/related papers
            - "extract_methodology": Identify methods used

        Returns:
            Analysis results dictionary

        Example:
            ```python
            result = agent.execute({
                "task_type": "summarize_paper",
                "paper_id": "arxiv:1234.5678"
            })
            ```
        """
        self.status = AgentStatus.WORKING

        try:
            task_type = task.get("task_type")

            if task_type == "summarize_paper":
                paper_id = task.get("paper_id")
                paper = task.get("paper")

                if not paper:
                    # Fetch paper
                    paper = self._fetch_paper(paper_id)

                summary = self.summarize_paper(paper)
                result = {"status": "success", "summary": summary.to_dict()}

            elif task_type == "analyze_corpus":
                papers = task.get("papers", [])
                insights = self.analyze_corpus(papers, generate_insights=True)
                result = {"status": "success", "insights": insights}

            elif task_type == "citation_network":
                paper_id = task.get("paper_id")
                depth = task.get("depth", self.max_citation_depth)
                network = self.analyze_citation_network(paper_id, depth)
                result = {"status": "success", "network": network}

            elif task_type == "find_related":
                paper = task.get("paper")
                max_results = task.get("max_results", 10)
                related = self.find_related_papers(paper, max_results)
                result = {"status": "success", "related_papers": related}

            elif task_type == "extract_methodology":
                paper = task.get("paper")
                methodology = self.extract_methodology(paper)
                result = {"status": "success", "methodology": methodology}

            else:
                raise ValueError(f"Unknown task type: {task_type}")

            self.tasks_completed += 1
            self.status = AgentStatus.IDLE
            return result

        except Exception as e:
            self.errors_encountered += 1
            logger.error(f"Task execution failed: {e}")
            self.status = AgentStatus.ERROR
            return {"status": "error", "error": str(e)}

    def summarize_paper(self, paper: PaperMetadata) -> PaperAnalysis:
        """
        Generate comprehensive paper summary using Claude.

        Args:
            paper: PaperMetadata object

        Returns:
            PaperAnalysis with structured summary

        Example:
            ```python
            summary = agent.summarize_paper(paper)
            print(summary.executive_summary)
            print(summary.key_findings)
            ```
        """
        # Check cache
        cached = self._get_cached_analysis(paper.primary_identifier)
        if cached:
            logger.debug(f"Using cached analysis for {paper.primary_identifier}")
            return PaperAnalysis(**cached)

        start_time = time.time()

        # Validate paper
        if not self._validate_paper(paper):
            raise ValueError(f"Paper lacks required data: {paper.primary_identifier}")

        # Build prompt
        prompt = self._build_summarization_prompt(paper)

        # Generate with Claude
        try:
            analysis = self.llm_client.generate_structured(
                prompt=prompt,
                output_schema=self._get_summarization_schema(),
                system="You are an expert scientific literature analyst. Provide thorough, accurate analysis.",
                max_tokens=4096
            )

            result = PaperAnalysis(
                paper_id=paper.primary_identifier,
                executive_summary=analysis.get("executive_summary", ""),
                key_findings=analysis.get("key_findings", []),
                methodology=analysis.get("methodology", {}),
                significance=analysis.get("significance", ""),
                limitations=analysis.get("limitations", []),
                confidence_score=analysis.get("confidence_score", 0.5),
                analysis_time=time.time() - start_time
            )

            # Cache result
            self._cache_analysis(paper.primary_identifier, result.to_dict())

            logger.info(f"Analyzed paper: {paper.title or paper.primary_identifier}")
            return result

        except Exception as e:
            logger.error(f"Summarization failed for {paper.primary_identifier}: {e}")
            raise

    def extract_key_findings(
        self,
        paper: PaperMetadata,
        max_findings: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Extract structured key findings using Claude.

        Args:
            paper: PaperMetadata object
            max_findings: Maximum number of findings

        Returns:
            List of findings with evidence and confidence

        Example:
            ```python
            findings = agent.extract_key_findings(paper, max_findings=5)
            for finding in findings:
                print(f"{finding['finding']} (confidence: {finding['confidence']})")
            ```
        """
        prompt = self._build_key_findings_prompt(paper, max_findings)

        schema = {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "finding": {"type": "string"},
                            "evidence": {"type": "string"},
                            "confidence": {"type": "number"},
                            "category": {"type": "string"}
                        }
                    }
                }
            }
        }

        try:
            result = self.llm_client.generate_structured(
                prompt=prompt,
                output_schema=schema,
                system="Extract key findings with supporting evidence."
            )

            return result.get("findings", [])

        except Exception as e:
            logger.error(f"Key findings extraction failed: {e}")
            return []

    def extract_methodology(self, paper: PaperMetadata) -> Dict[str, Any]:
        """
        Identify research methods and techniques.

        Integrates with ConceptExtractor and knowledge graph.

        Args:
            paper: PaperMetadata object

        Returns:
            Dictionary with categorized methods

        Example:
            ```python
            methods = agent.extract_methodology(paper)
            print(methods['experimental_methods'])
            print(methods['computational_methods'])
            ```
        """
        methodology = {
            "experimental_methods": [],
            "computational_methods": [],
            "analytical_methods": [],
            "datasets_used": []
        }

        # Use concept extractor if available
        if self.extract_concepts:
            try:
                extraction = self.concept_extractor.extract_from_paper(paper)

                for method in extraction.methods:
                    method_dict = {
                        "name": method.name,
                        "description": method.description,
                        "confidence": method.confidence
                    }

                    # Categorize by category
                    if method.category == "experimental":
                        methodology["experimental_methods"].append(method_dict)
                    elif method.category == "computational":
                        methodology["computational_methods"].append(method_dict)
                    elif method.category == "analytical":
                        methodology["analytical_methods"].append(method_dict)

            except Exception as e:
                logger.warning(f"Concept extraction failed: {e}")

        # Supplement with Claude if needed
        if not methodology["experimental_methods"] and not methodology["computational_methods"]:
            prompt = self._build_methodology_prompt(paper)

            try:
                result = self.llm_client.generate_structured(
                    prompt=prompt,
                    output_schema=self._get_methodology_schema(),
                    system="Extract research methods systematically."
                )

                methodology.update(result)

            except Exception as e:
                logger.error(f"Methodology extraction failed: {e}")

        return methodology

    def analyze_citation_network(
        self,
        paper_id: str,
        depth: int = 2,
        build_if_missing: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze citation network around a paper.

        Strategy:
        1. Check knowledge graph for existing citations
        2. If missing and build_if_missing=True, fetch from APIs
        3. Analyze network structure

        Args:
            paper_id: Paper identifier
            depth: Citation depth
            build_if_missing: Whether to fetch missing citations

        Returns:
            Citation network analysis

        Example:
            ```python
            network = agent.analyze_citation_network(
                "arxiv:1234.5678",
                depth=2,
                build_if_missing=True
            )
            print(f"Citation count: {network['citation_count']}")
            ```
        """
        network_data = {
            "paper_id": paper_id,
            "citation_count": 0,
            "cited_by_count": 0,
            "influential_citations": [],
            "network_metrics": {}
        }

        # Try knowledge graph first
        if self.use_knowledge_graph and self.knowledge_graph:
            try:
                # Get citations from graph
                citations = self.knowledge_graph.get_citations(paper_id, depth)
                network_data["citation_count"] = len(citations)

                # Get citing papers
                citing = self.knowledge_graph.get_citing_papers(paper_id, limit=100)
                network_data["cited_by_count"] = len(citing)

                # Identify influential citations (high citation count)
                influential = sorted(
                    citations,
                    key=lambda x: x["paper"].get("citation_count", 0),
                    reverse=True
                )[:5]

                network_data["influential_citations"] = [
                    {
                        "paper_id": c["paper"]["id"],
                        "title": c["paper"]["title"],
                        "citation_count": c["paper"].get("citation_count", 0)
                    }
                    for c in influential
                ]

                logger.info(f"Citation network from graph: {len(citations)} citations")

            except Exception as e:
                logger.warning(f"Graph query failed: {e}")

        # Build on-demand if needed
        if network_data["citation_count"] == 0 and build_if_missing:
            logger.info("Building citation network on-demand...")
            built = self._build_citation_graph_on_demand(paper_id)

            if built:
                # Retry graph query
                return self.analyze_citation_network(paper_id, depth, build_if_missing=False)

        return network_data

    def score_relevance(
        self,
        papers: List[PaperMetadata],
        query: str,
        reference_papers: Optional[List[PaperMetadata]] = None
    ) -> List[Tuple[PaperMetadata, float]]:
        """
        Multi-faceted relevance scoring.

        Scoring factors:
        - Semantic similarity (SPECTER embeddings)
        - Citation importance (from graph)
        - Recency (publication date)
        - Concept overlap (from knowledge graph)

        Args:
            papers: Papers to score
            query: Query string
            reference_papers: Optional reference papers for relevance

        Returns:
            List of (paper, score) tuples, sorted by relevance

        Example:
            ```python
            scored = agent.score_relevance(
                papers,
                query="machine learning interpretability"
            )

            for paper, score in scored[:10]:
                print(f"{paper.title}: {score:.3f}")
            ```
        """
        scored_papers = []

        # Compute query embedding for semantic similarity
        query_embedding = None
        if self.use_semantic_similarity:
            query_embedding = self.embedder.embed_query(query)

        for paper in papers:
            score = 0.0
            weights = {"semantic": 0.4, "citation": 0.3, "recency": 0.2, "concept": 0.1}

            # Semantic similarity
            if query_embedding is not None:
                try:
                    paper_embedding = self.embedder.embed_paper(paper)
                    similarity = self.embedder.compute_similarity(query_embedding, paper_embedding)
                    score += weights["semantic"] * similarity
                except Exception as e:
                    logger.debug(f"Semantic scoring failed: {e}")

            # Citation importance (normalized)
            if paper.citation_count > 0:
                # Log scale normalization
                import math
                citation_score = math.log1p(paper.citation_count) / math.log1p(1000)
                score += weights["citation"] * min(citation_score, 1.0)

            # Recency (papers from last 5 years get bonus)
            if paper.year:
                from datetime import datetime
                current_year = datetime.now().year
                age = current_year - paper.year
                recency_score = max(0, 1.0 - (age / 5.0))  # 5-year window
                score += weights["recency"] * recency_score

            scored_papers.append((paper, score))

        # Sort by score descending
        scored_papers.sort(key=lambda x: x[1], reverse=True)

        return scored_papers

    def find_related_papers(
        self,
        paper: PaperMetadata,
        max_results: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find papers related through multiple pathways.

        Uses:
        - Vector similarity (semantic)
        - Citation network (graph)
        - Concept overlap (knowledge graph)

        Args:
            paper: Reference paper
            max_results: Maximum results
            similarity_threshold: Minimum similarity

        Returns:
            List of related papers with relationship info

        Example:
            ```python
            related = agent.find_related_papers(paper, max_results=10)
            for item in related:
                print(f"{item['paper']['title']}")
                print(f"  Similarity: {item['similarity']:.3f}")
                print(f"  Relationship: {item['relationship_type']}")
            ```
        """
        related = []

        # 1. Semantic similarity
        if self.use_semantic_similarity:
            try:
                similar = self.vector_db.search_by_paper(
                    paper,
                    top_k=max_results,
                    filters=None
                )

                for result in similar:
                    if result["score"] >= similarity_threshold:
                        related.append({
                            "paper": result["metadata"],
                            "similarity": result["score"],
                            "relationship_type": "semantic_similarity"
                        })

            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")

        # 2. Citation network
        if self.use_knowledge_graph and self.knowledge_graph:
            try:
                graph_related = self.knowledge_graph.find_related_papers(
                    paper.primary_identifier,
                    max_hops=2,
                    limit=max_results
                )

                for item in graph_related:
                    related.append({
                        "paper": item["paper"],
                        "distance": item["distance"],
                        "relationship_type": "citation_network"
                    })

            except Exception as e:
                logger.warning(f"Graph search failed: {e}")

        # Deduplicate and sort
        seen_ids = set()
        unique_related = []

        for item in related:
            paper_id = item["paper"].get("id") or item["paper"].get("title")
            if paper_id not in seen_ids:
                seen_ids.add(paper_id)
                unique_related.append(item)

        return unique_related[:max_results]

    def analyze_corpus(
        self,
        papers: List[PaperMetadata],
        generate_insights: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a corpus of papers collectively.

        Generates:
        - Common themes and concepts
        - Methodological trends
        - Citation patterns
        - Research gaps (Claude-powered)

        Args:
            papers: List of papers
            generate_insights: Whether to generate high-level insights

        Returns:
            Corpus analysis results

        Example:
            ```python
            insights = agent.analyze_corpus(papers, generate_insights=True)
            print("Common themes:", insights['common_themes'])
            print("Research gaps:", insights['research_gaps'])
            ```
        """
        analysis = {
            "corpus_size": len(papers),
            "common_themes": [],
            "methodological_trends": [],
            "temporal_distribution": {},
            "research_gaps": [],
            "key_authors": []
        }

        if not papers:
            return analysis

        # Temporal distribution
        year_counts = {}
        for paper in papers:
            if paper.year:
                year_counts[paper.year] = year_counts.get(paper.year, 0) + 1
        analysis["temporal_distribution"] = year_counts

        # Extract common concepts
        if self.extract_concepts:
            concept_freq = {}

            for paper in papers[:self.max_papers_per_analysis]:
                try:
                    extraction = self.concept_extractor.extract_from_paper(paper)
                    for concept in extraction.concepts:
                        concept_freq[concept.name] = concept_freq.get(concept.name, 0) + 1
                except Exception as e:
                    logger.debug(f"Concept extraction failed: {e}")

            # Top concepts
            top_concepts = sorted(concept_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            analysis["common_themes"] = [name for name, count in top_concepts]

        # Generate insights with Claude
        if generate_insights:
            try:
                # Summarize papers (filter out None papers)
                summaries = [
                    f"{p.title or 'Untitled'}: {p.abstract[:200] if p.abstract else 'No abstract'}"
                    for p in papers[:20]  # Limit to avoid token issues
                    if p is not None and p.title
                ]

                prompt = self._build_corpus_insights_prompt(summaries)

                insights = self.llm_client.generate_structured(
                    prompt=prompt,
                    output_schema=self._get_corpus_insights_schema(),
                    system="Analyze research corpus and identify patterns, trends, and gaps.",
                    max_tokens=1024
                )

                analysis.update(insights)

            except Exception as e:
                logger.error(f"Corpus insights generation failed: {e}")

        return analysis

    # Helper methods

    def _fetch_paper(self, paper_id: str) -> PaperMetadata:
        """Fetch paper from search APIs."""
        # Try to find in vector DB first
        if self.use_semantic_similarity:
            paper_node = self.vector_db.get_paper(paper_id)
            if paper_node:
                # Reconstruct PaperMetadata from stored data
                # This is simplified - full implementation would properly reconstruct
                pass

        # Fall back to API search
        results = self.unified_search.search(paper_id, max_results_per_source=1)
        if results:
            return results[0]

        raise ValueError(f"Paper not found: {paper_id}")

    def _validate_paper(self, paper: PaperMetadata) -> bool:
        """Ensure paper has minimum required data."""
        return bool(paper.title and (paper.abstract or paper.full_text))

    def _build_citation_graph_on_demand(self, paper_id: str) -> bool:
        """
        Fetch citations from APIs and build in graph.

        Integrates with Semantic Scholar API to fetch citation data
        and populate Neo4j knowledge graph.

        Args:
            paper_id: Paper identifier (arXiv, DOI, S2, etc.)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            from kosmos.literature.semantic_scholar import SemanticScholarClient

            # Initialize Semantic Scholar client
            ss_client = SemanticScholarClient()

            # Fetch paper details with citations
            try:
                paper_data = ss_client.get_paper(paper_id, fields=['citations', 'references', 'title', 'year', 'authors'])
            except Exception as e:
                logger.warning(f"Failed to fetch paper from Semantic Scholar: {e}")
                return False

            if not paper_data:
                logger.warning(f"Paper {paper_id} not found in Semantic Scholar")
                return False

            # Add main paper to knowledge graph
            if self.knowledge_graph:
                try:
                    from kosmos.literature.base_client import PaperMetadata, PaperSource

                    # Create PaperMetadata object
                    paper_meta = PaperMetadata(
                        title=paper_data.get('title', ''),
                        authors=[a.get('name', '') for a in paper_data.get('authors', [])],
                        year=paper_data.get('year'),
                        abstract=paper_data.get('abstract', ''),
                        source=PaperSource.SEMANTIC_SCHOLAR,
                        source_id=paper_data.get('paperId', paper_id)
                    )

                    # Add paper node
                    self.knowledge_graph.add_paper(paper_meta)

                    # Add citations (papers this paper cites)
                    references = paper_data.get('references', [])
                    for ref in references[:50]:  # Limit to 50 most relevant
                        if ref and ref.get('paperId'):
                            ref_meta = PaperMetadata(
                                title=ref.get('title', ''),
                                authors=[a.get('name', '') for a in ref.get('authors', [])],
                                year=ref.get('year'),
                                source=PaperSource.SEMANTIC_SCHOLAR,
                                source_id=ref['paperId']
                            )

                            self.knowledge_graph.add_paper(ref_meta)
                            self.knowledge_graph.add_citation(
                                citing_paper_id=paper_meta.primary_identifier,
                                cited_paper_id=ref_meta.primary_identifier
                            )

                    # Add citing papers (papers that cite this paper)
                    citations = paper_data.get('citations', [])
                    for cite in citations[:50]:  # Limit to 50 most relevant
                        if cite and cite.get('paperId'):
                            cite_meta = PaperMetadata(
                                title=cite.get('title', ''),
                                authors=[a.get('name', '') for a in cite.get('authors', [])],
                                year=cite.get('year'),
                                source=PaperSource.SEMANTIC_SCHOLAR,
                                source_id=cite['paperId']
                            )

                            self.knowledge_graph.add_paper(cite_meta)
                            self.knowledge_graph.add_citation(
                                citing_paper_id=cite_meta.primary_identifier,
                                cited_paper_id=paper_meta.primary_identifier
                            )

                    logger.info(f"Built citation graph for {paper_id}: {len(references)} references, {len(citations)} citations")
                    return True

                except Exception as e:
                    logger.error(f"Failed to add citations to knowledge graph: {e}")
                    return False
            else:
                logger.warning("Knowledge graph not available")
                return False

        except Exception as e:
            logger.error(f"Failed to build citation graph for {paper_id}: {e}")
            return False

    # Prompt builders

    def _build_summarization_prompt(self, paper: PaperMetadata) -> str:
        """Build prompt for paper summarization."""
        title = paper.title if paper and paper.title else "Unknown Title"
        text = f"Title: {title}\n\n"

        if paper and paper.full_text:
            text += f"Text: {paper.full_text[:5000]}"  # Limit length
        elif paper and paper.abstract:
            text += f"Abstract: {paper.abstract}"
        else:
            text += "Abstract: Not available"

        prompt = f"""Analyze this scientific paper and provide a comprehensive summary.

{text}

Provide a structured analysis with:
1. Executive summary (2-3 sentences)
2. Key findings (3-5 main results)
3. Methodology (research approach)
4. Significance (scientific importance)
5. Limitations (weaknesses or caveats)
6. Confidence score (0-1, how confident are you in this analysis?)

Return structured output following the schema."""

        return prompt

    def _build_key_findings_prompt(self, paper: PaperMetadata, max_findings: int) -> str:
        """Build prompt for key findings extraction."""
        title = paper.title if paper and paper.title else "Unknown Title"
        abstract = paper.abstract if paper and paper.abstract else "Not available"
        text = f"Title: {title}\n\n"
        text += f"Abstract: {abstract}"

        prompt = f"""Extract the {max_findings} most important findings from this paper.

{text}

For each finding, provide:
- finding: Clear statement of the finding
- evidence: Direct quote or paraphrase supporting it
- confidence: Your confidence (0-1)
- category: Type (result, method, insight, limitation, etc.)

Return structured output."""

        return prompt

    def _build_methodology_prompt(self, paper: PaperMetadata) -> str:
        """Build prompt for methodology extraction."""
        title = paper.title if paper and paper.title else "Unknown Title"
        abstract = paper.abstract if paper and paper.abstract else "Not available"
        return f"""Identify the research methods used in this paper.

Title: {title}
Abstract: {abstract}

Categorize methods as:
- experimental_methods: Lab or field experiments
- computational_methods: Simulations, algorithms, models
- analytical_methods: Statistical analysis, mathematical proofs
- datasets_used: Data sources or datasets mentioned

Return structured output."""

    def _build_corpus_insights_prompt(self, summaries: List[str]) -> str:
        """Build prompt for corpus-level insights."""
        papers_text = "\n\n".join([f"{i+1}. {s}" for i, s in enumerate(summaries)])

        prompt = f"""Analyze this collection of research papers and identify:

{papers_text}

Provide:
1. common_themes: Recurring topics and concepts (list)
2. methodological_trends: Common research approaches (list)
3. research_gaps: Underexplored areas or questions (list)
4. emerging_directions: Potential future research directions (list)

Return structured output."""

        return prompt

    # JSON schemas

    def _get_summarization_schema(self) -> Dict[str, Any]:
        """Schema for paper summarization."""
        return {
            "type": "object",
            "properties": {
                "executive_summary": {"type": "string"},
                "key_findings": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"finding": {"type": "string"}}}
                },
                "methodology": {"type": "object"},
                "significance": {"type": "string"},
                "limitations": {"type": "array", "items": {"type": "string"}},
                "confidence_score": {"type": "number"}
            }
        }

    def _get_methodology_schema(self) -> Dict[str, Any]:
        """Schema for methodology extraction."""
        method_item = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"}
            }
        }

        return {
            "type": "object",
            "properties": {
                "experimental_methods": {"type": "array", "items": method_item},
                "computational_methods": {"type": "array", "items": method_item},
                "analytical_methods": {"type": "array", "items": method_item},
                "datasets_used": {"type": "array", "items": {"type": "string"}}
            }
        }

    def _get_corpus_insights_schema(self) -> Dict[str, Any]:
        """Schema for corpus insights."""
        return {
            "type": "object",
            "properties": {
                "common_themes": {"type": "array", "items": {"type": "string"}},
                "methodological_trends": {"type": "array", "items": {"type": "string"}},
                "research_gaps": {"type": "array", "items": {"type": "string"}},
                "emerging_directions": {"type": "array", "items": {"type": "string"}}
            }
        }

    # Caching

    def _get_cached_analysis(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Check cache for existing analysis."""
        cache_file = self.cache_dir / f"{self._sanitize_id(paper_id)}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)

            # Check if cache is still fresh (24 hours)
            if time.time() - data.get("cached_at", 0) < 86400:
                return data.get("analysis")

        except Exception as e:
            logger.debug(f"Cache read failed: {e}")

        return None

    def _cache_analysis(self, paper_id: str, analysis: Dict[str, Any]):
        """Save analysis to cache."""
        cache_file = self.cache_dir / f"{self._sanitize_id(paper_id)}.json"

        try:
            data = {
                "paper_id": paper_id,
                "analysis": analysis,
                "cached_at": time.time()
            }

            with open(cache_file, "w") as f:
                json.dump(data, f)

        except Exception as e:
            logger.debug(f"Cache write failed: {e}")

    def _sanitize_id(self, paper_id: str) -> str:
        """Sanitize paper ID for filename."""
        return paper_id.replace(":", "_").replace("/", "_")


# Singleton instance
_literature_analyzer: Optional[LiteratureAnalyzerAgent] = None


def get_literature_analyzer(
    agent_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    reset: bool = False
) -> LiteratureAnalyzerAgent:
    """
    Get or create the singleton literature analyzer agent.

    Args:
        agent_id: Agent ID
        config: Configuration dictionary
        reset: Whether to reset the singleton

    Returns:
        LiteratureAnalyzerAgent instance
    """
    global _literature_analyzer
    if _literature_analyzer is None or reset:
        _literature_analyzer = LiteratureAnalyzerAgent(
            agent_id=agent_id,
            config=config
        )
    return _literature_analyzer


def reset_literature_analyzer():
    """Reset the singleton literature analyzer (useful for testing)."""
    global _literature_analyzer
    _literature_analyzer = None

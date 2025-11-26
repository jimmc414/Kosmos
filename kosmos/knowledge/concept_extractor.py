"""
Claude-powered concept and method extraction from scientific papers.

Uses Claude API to extract structured information including:
- Scientific concepts and their relationships
- Methods and techniques used
- Relevance scores and context
"""

import json
import logging
import hashlib
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import time

import anthropic

from kosmos.config import get_config
from kosmos.literature.base_client import PaperMetadata

logger = logging.getLogger(__name__)


@dataclass
class ExtractedConcept:
    """Represents an extracted scientific concept."""
    name: str
    description: str
    domain: str
    relevance: float  # 0-1 score

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedMethod:
    """Represents an extracted method or technique."""
    name: str
    description: str
    category: str  # experimental, computational, analytical, theoretical
    confidence: float  # 0-1 score

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConceptRelationship:
    """Represents a relationship between concepts."""
    concept1: str
    concept2: str
    relationship_type: str  # SUBTOPIC_OF, RELATED_TO, PREREQUISITE_FOR, etc.
    strength: float  # 0-1 score

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionResult:
    """Complete extraction result for a paper."""
    paper_id: str
    concepts: List[ExtractedConcept]
    methods: List[ExtractedMethod]
    relationships: List[ConceptRelationship]
    extraction_time: float
    model_used: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "concepts": [c.to_dict() for c in self.concepts],
            "methods": [m.to_dict() for m in self.methods],
            "relationships": [r.to_dict() for r in self.relationships],
            "extraction_time": self.extraction_time,
            "model_used": self.model_used
        }


class ConceptExtractor:
    """
    Claude-powered extractor for scientific concepts and methods.

    Uses Claude API with specialized prompts to extract structured information
    from scientific papers.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250929",
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        max_tokens: int = 4096,
        temperature: float = 0.0
    ):
        """
        Initialize concept extractor.

        Args:
            api_key: Anthropic API key (default: from config)
            model: Claude model to use
            cache_dir: Cache directory for extractions
            use_cache: Whether to use caching
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature (0 for deterministic)

        Example:
            ```python
            extractor = ConceptExtractor()

            # Extract from paper
            result = extractor.extract_from_paper(paper_metadata)

            # Get concepts
            for concept in result.concepts:
                print(f"{concept.name}: {concept.description}")
            ```
        """
        config = get_config()

        # API setup - requires Anthropic API key
        self.api_key = api_key or (config.claude.api_key if config.claude else None)
        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required for ConceptExtractor. "
                "Either pass api_key parameter or set ANTHROPIC_API_KEY environment variable."
            )
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Caching setup
        self.use_cache = use_cache
        if use_cache:
            self.cache_dir = Path(cache_dir or ".concept_extraction_cache")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # seconds between requests

        logger.info(f"Initialized ConceptExtractor (model={model})")

    def extract_from_paper(
        self,
        paper: PaperMetadata,
        include_relationships: bool = True,
        max_concepts: int = 10,
        max_methods: int = 5
    ) -> ExtractionResult:
        """
        Extract concepts and methods from a paper.

        Args:
            paper: PaperMetadata object
            include_relationships: Whether to extract concept relationships
            max_concepts: Maximum number of concepts to extract
            max_methods: Maximum number of methods to extract

        Returns:
            ExtractionResult with concepts, methods, and relationships

        Example:
            ```python
            result = extractor.extract_from_paper(paper)

            print(f"Found {len(result.concepts)} concepts")
            print(f"Found {len(result.methods)} methods")
            ```
        """
        # Check cache
        if self.use_cache:
            cached = self._get_from_cache(paper.primary_identifier)
            if cached:
                logger.debug(f"Using cached extraction for {paper.primary_identifier}")
                return cached

        # Rate limiting
        self._rate_limit()

        start_time = time.time()

        # Extract concepts and methods
        logger.info(f"Extracting concepts from: {paper.title}")

        concepts_and_methods = self._extract_concepts_and_methods(
            paper,
            max_concepts=max_concepts,
            max_methods=max_methods
        )

        concepts = concepts_and_methods["concepts"]
        methods = concepts_and_methods["methods"]

        # Extract relationships if requested
        relationships = []
        if include_relationships and len(concepts) > 1:
            relationships = self._extract_relationships(paper, concepts)

        extraction_time = time.time() - start_time

        result = ExtractionResult(
            paper_id=paper.primary_identifier,
            concepts=concepts,
            methods=methods,
            relationships=relationships,
            extraction_time=extraction_time,
            model_used=self.model
        )

        # Cache result
        if self.use_cache:
            self._save_to_cache(paper.primary_identifier, result)

        logger.info(
            f"Extracted {len(concepts)} concepts, {len(methods)} methods "
            f"in {extraction_time:.2f}s"
        )

        return result

    def extract_from_papers(
        self,
        papers: List[PaperMetadata],
        show_progress: bool = True
    ) -> List[ExtractionResult]:
        """
        Extract from multiple papers (batch processing).

        Args:
            papers: List of papers
            show_progress: Whether to show progress

        Returns:
            List of extraction results
        """
        results = []

        for i, paper in enumerate(papers):
            if show_progress:
                logger.info(f"Processing paper {i + 1}/{len(papers)}: {paper.title}")

            try:
                result = self.extract_from_paper(paper)
                results.append(result)
            except Exception as e:
                logger.error(f"Error extracting from paper {paper.primary_identifier}: {e}")
                # Create empty result on error
                results.append(ExtractionResult(
                    paper_id=paper.primary_identifier,
                    concepts=[],
                    methods=[],
                    relationships=[],
                    extraction_time=0.0,
                    model_used=self.model
                ))

        return results

    def _extract_concepts_and_methods(
        self,
        paper: PaperMetadata,
        max_concepts: int,
        max_methods: int
    ) -> Dict[str, Any]:
        """
        Extract concepts and methods using Claude.

        Args:
            paper: Paper metadata
            max_concepts: Maximum concepts
            max_methods: Maximum methods

        Returns:
            Dictionary with concepts and methods lists
        """
        # Build prompt
        prompt = self._build_concept_extraction_prompt(
            paper,
            max_concepts=max_concepts,
            max_methods=max_methods
        )

        # Call Claude API
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            response_text = message.content[0].text

            # Parse JSON response
            parsed = self._parse_json_response(response_text)

            # Convert to dataclasses
            concepts = [
                ExtractedConcept(
                    name=c["name"],
                    description=c["description"],
                    domain=c.get("domain", "unknown"),
                    relevance=c.get("relevance", 0.5)
                )
                for c in parsed.get("concepts", [])
            ]

            methods = [
                ExtractedMethod(
                    name=m["name"],
                    description=m["description"],
                    category=m.get("category", "unknown"),
                    confidence=m.get("confidence", 0.5)
                )
                for m in parsed.get("methods", [])
            ]

            return {"concepts": concepts, "methods": methods}

        except Exception as e:
            logger.error(f"Error in Claude API call: {e}")
            return {"concepts": [], "methods": []}

    def _extract_relationships(
        self,
        paper: PaperMetadata,
        concepts: List[ExtractedConcept]
    ) -> List[ConceptRelationship]:
        """
        Extract relationships between concepts.

        Args:
            paper: Paper metadata
            concepts: List of extracted concepts

        Returns:
            List of concept relationships
        """
        if len(concepts) < 2:
            return []

        # Build prompt
        prompt = self._build_relationship_prompt(paper, concepts)

        # Call Claude API
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            response_text = message.content[0].text
            parsed = self._parse_json_response(response_text)

            # Convert to dataclasses
            relationships = [
                ConceptRelationship(
                    concept1=r["concept1"],
                    concept2=r["concept2"],
                    relationship_type=r.get("type", "RELATED_TO"),
                    strength=r.get("strength", 0.5)
                )
                for r in parsed.get("relationships", [])
            ]

            return relationships

        except Exception as e:
            logger.error(f"Error extracting relationships: {e}")
            return []

    def _build_concept_extraction_prompt(
        self,
        paper: PaperMetadata,
        max_concepts: int,
        max_methods: int
    ) -> str:
        """
        Build prompt for concept and method extraction.

        Args:
            paper: Paper metadata
            max_concepts: Maximum concepts
            max_methods: Maximum methods

        Returns:
            Formatted prompt string
        """
        # Use title and abstract (or full text if available)
        text = f"Title: {paper.title}\n\n"

        if paper.full_text:
            # Use first 3000 characters of full text
            text += f"Text: {paper.full_text[:3000]}"
        elif paper.abstract:
            text += f"Abstract: {paper.abstract}"
        else:
            text += "Abstract: Not available"

        prompt = f"""You are a scientific literature analyst. Extract key concepts and methods from the following paper.

{text}

Extract up to {max_concepts} key scientific concepts and up to {max_methods} methods/techniques.

For each concept, provide:
- name: Concept name (concise, 2-5 words)
- description: Brief explanation (1-2 sentences)
- domain: Scientific domain (e.g., biology, physics, computer_science)
- relevance: Relevance score 0-1 (how central is this concept to the paper?)

For each method, provide:
- name: Method name (concise, 2-5 words)
- description: Brief explanation (1-2 sentences)
- category: One of [experimental, computational, analytical, theoretical]
- confidence: Confidence score 0-1 (how certain are you this method is used?)

Return ONLY a JSON object in this exact format (no additional text):
{{
  "concepts": [
    {{"name": "...", "description": "...", "domain": "...", "relevance": 0.9}},
    ...
  ],
  "methods": [
    {{"name": "...", "description": "...", "category": "...", "confidence": 0.9}},
    ...
  ]
}}

Be specific and focus on the most important concepts and methods. Prioritize clarity and relevance."""

        return prompt

    def _build_relationship_prompt(
        self,
        paper: PaperMetadata,
        concepts: List[ExtractedConcept]
    ) -> str:
        """
        Build prompt for relationship extraction.

        Args:
            paper: Paper metadata
            concepts: List of concepts

        Returns:
            Formatted prompt string
        """
        concept_list = "\n".join([f"- {c.name}: {c.description}" for c in concepts])

        prompt = f"""Given these scientific concepts extracted from a paper:

{concept_list}

Identify relationships between these concepts. For each relationship, specify:
- concept1: First concept name (exactly as listed above)
- concept2: Second concept name (exactly as listed above)
- type: Relationship type - one of:
  - SUBTOPIC_OF: concept1 is a subtopic/specialization of concept2
  - PREREQUISITE_FOR: concept1 is required to understand concept2
  - RELATED_TO: concepts are related but neither is more specific
  - ENABLES: concept1 enables or makes concept2 possible
- strength: Relationship strength 0-1

Only include strong, clear relationships. Return ONLY a JSON object:
{{
  "relationships": [
    {{"concept1": "...", "concept2": "...", "type": "...", "strength": 0.9}},
    ...
  ]
}}

If no clear relationships exist, return {{"relationships": []}}"""

        return prompt

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON from Claude response.

        Handles cases where response includes markdown code blocks.

        Args:
            response_text: Raw response from Claude

        Returns:
            Parsed JSON dictionary
        """
        # Remove markdown code blocks if present
        text = response_text.strip()

        if text.startswith("```json"):
            text = text[7:]  # Remove ```json
        elif text.startswith("```"):
            text = text[3:]  # Remove ```

        if text.endswith("```"):
            text = text[:-3]  # Remove closing ```

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {text}")
            return {"concepts": [], "methods": [], "relationships": []}

    def _rate_limit(self):
        """Apply rate limiting to API requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _get_cache_key(self, paper_id: str) -> str:
        """Generate cache key for a paper."""
        return hashlib.sha256(
            f"{paper_id}:{self.model}".encode()
        ).hexdigest()

    def _get_from_cache(self, paper_id: str) -> Optional[ExtractionResult]:
        """
        Get extraction result from cache.

        Args:
            paper_id: Paper identifier

        Returns:
            Cached result or None
        """
        if not self.cache_dir:
            return None

        cache_key = self._get_cache_key(paper_id)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)

            # Reconstruct result
            result = ExtractionResult(
                paper_id=data["paper_id"],
                concepts=[
                    ExtractedConcept(**c) for c in data["concepts"]
                ],
                methods=[
                    ExtractedMethod(**m) for m in data["methods"]
                ],
                relationships=[
                    ConceptRelationship(**r) for r in data["relationships"]
                ],
                extraction_time=data["extraction_time"],
                model_used=data["model_used"]
            )

            return result

        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
            return None

    def _save_to_cache(self, paper_id: str, result: ExtractionResult):
        """
        Save extraction result to cache.

        Args:
            paper_id: Paper identifier
            result: Extraction result
        """
        if not self.cache_dir:
            return

        cache_key = self._get_cache_key(paper_id)
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            with open(cache_file, "w") as f:
                json.dump(result.to_dict(), f, indent=2)

            logger.debug(f"Cached extraction for {paper_id}")

        except Exception as e:
            logger.error(f"Error saving to cache: {e}")

    def clear_cache(self):
        """Clear extraction cache."""
        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("Cleared extraction cache")


# Singleton instance
_concept_extractor: Optional[ConceptExtractor] = None


def get_concept_extractor(
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-5-20250929",
    reset: bool = False
) -> ConceptExtractor:
    """
    Get or create the singleton concept extractor instance.

    Args:
        api_key: Anthropic API key
        model: Claude model to use
        reset: Whether to reset the singleton

    Returns:
        ConceptExtractor instance
    """
    global _concept_extractor
    if _concept_extractor is None or reset:
        _concept_extractor = ConceptExtractor(
            api_key=api_key,
            model=model
        )
    return _concept_extractor


def reset_concept_extractor():
    """Reset the singleton concept extractor (useful for testing)."""
    global _concept_extractor
    _concept_extractor = None

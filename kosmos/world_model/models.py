"""
World Model Data Models.

This module defines the core data models for the world model:
- Entity: Knowledge graph entities (papers, concepts, authors, etc.)
- Relationship: Connections between entities
- Annotation: User annotations on entities

DESIGN RATIONALE:
- Dataclasses for simplicity and validation
- Confidence scores for uncertainty representation
- Project namespacing for multi-project support
- Timestamps for provenance
- Validation in __post_init__ for data integrity
- to_dict/from_dict for serialization (export/import)

EDUCATIONAL NOTE:
This is a simple but extensible design. Production Mode (Phase 4) will add:
- PROV-O standard provenance
- Embedding vectors
- Quality scores
But the core models remain the same (backward compatible).
"""

import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# Export format version for compatibility checking
EXPORT_FORMAT_VERSION = "1.0"


@dataclass
class Annotation:
    """
    User annotation on an entity.

    Annotations allow users to add notes, corrections, or context to entities.
    This supports the curation workflow (Phase 2).

    Attributes:
        text: The annotation content
        created_by: Who created the annotation (email/username)
        created_at: When the annotation was created

    Example:
        ann = Annotation(
            text="This paper is seminal in the field",
            created_by="researcher@university.edu"
        )
    """

    text: str
    created_by: str
    created_at: Optional[datetime] = None

    def __post_init__(self):
        """Validate annotation."""
        if not self.text or not self.text.strip():
            raise ValueError("Annotation text cannot be empty")
        if not self.created_by or not self.created_by.strip():
            raise ValueError("Annotation must have a creator")

        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Entity:
    """
    Knowledge graph entity (paper, concept, experiment, etc.).

    DESIGN RATIONALE:
    - id: Auto-generated UUID if not provided (stable references)
    - type: Explicit entity type for filtering and querying
    - properties: Flexible dict for entity-specific attributes
    - confidence: Allows representing uncertainty (0.0-1.0)
    - project: Enables multi-project isolation
    - verified: User curation support (Phase 2)
    - annotations: Inline notes (Phase 2)

    ENTITY TYPES:
    - Paper: Research paper from literature
    - Concept: Scientific concept or term
    - Author: Paper author
    - Experiment: Experiment design or result
    - Hypothesis: Scientific hypothesis
    - Finding: Research finding
    - Dataset: Referenced dataset
    - Method: Experimental method

    Attributes:
        type: Entity type (Paper, Concept, Author, etc.)
        properties: Dict of entity-specific properties
        id: Unique identifier (auto-generated if not provided)
        confidence: Confidence score 0.0-1.0 (default 1.0)
        project: Project namespace for isolation
        created_at: Creation timestamp
        updated_at: Last update timestamp
        created_by: Creator identifier (agent name, user email)
        verified: Whether entity has been manually verified
        annotations: List of user annotations

    Example:
        paper = Entity(
            type="Paper",
            properties={
                "title": "Attention Is All You Need",
                "authors": ["Vaswani et al."],
                "year": 2017,
                "doi": "10.5555/3295222.3295349"
            },
            confidence=0.95,
            project="transformers_research",
            created_by="literature_agent"
        )
    """

    type: str
    properties: Dict[str, Any]
    id: Optional[str] = None
    confidence: float = 1.0
    project: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None
    verified: bool = False
    annotations: List[Annotation] = field(default_factory=list)

    # Standard entity types (can be extended)
    VALID_TYPES = {
        "Paper",
        "Concept",
        "Author",
        "Experiment",
        "Hypothesis",
        "Finding",
        "Dataset",
        "Method",
        # Research workflow entities
        "ResearchQuestion",
        "ExperimentProtocol",
        "ExperimentResult",
    }

    def __post_init__(self):
        """Validate and initialize entity."""
        # Generate ID if not provided
        if self.id is None:
            self.id = str(uuid.uuid4())

        # Validate type
        if not self.type:
            raise ValueError("Entity type is required")

        # Warn if non-standard type (but allow it for extensibility)
        if self.type not in self.VALID_TYPES:
            warnings.warn(
                f"Entity type '{self.type}' is not a standard type. "
                f"Standard types: {', '.join(sorted(self.VALID_TYPES))}",
                UserWarning,
                stacklevel=2,
            )

        # Validate confidence
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        # Set timestamps
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at

        # Validate properties
        if not isinstance(self.properties, dict):
            raise ValueError("Properties must be a dictionary")

    @classmethod
    def from_hypothesis(cls, hypothesis: Any, created_by: str = "hypothesis_generator") -> "Entity":
        """
        Create Entity from Hypothesis Pydantic model.

        Args:
            hypothesis: Hypothesis model from kosmos.models.hypothesis
            created_by: Agent that created this entity

        Returns:
            Entity representing the hypothesis

        Example:
            from kosmos.models.hypothesis import Hypothesis
            hypothesis = Hypothesis(statement="...", rationale="...", domain="...")
            entity = Entity.from_hypothesis(hypothesis, created_by="HypothesisGeneratorAgent")
        """
        properties = {
            "research_question": hypothesis.research_question,
            "statement": hypothesis.statement,
            "rationale": hypothesis.rationale,
            "domain": hypothesis.domain,
            "status": hypothesis.status.value if hasattr(hypothesis.status, 'value') else str(hypothesis.status),
        }

        # Add scores if present
        if hypothesis.testability_score is not None:
            properties["testability_score"] = hypothesis.testability_score
        if hypothesis.novelty_score is not None:
            properties["novelty_score"] = hypothesis.novelty_score
        if hypothesis.confidence_score is not None:
            properties["confidence_score"] = hypothesis.confidence_score
        if hypothesis.priority_score is not None:
            properties["priority_score"] = hypothesis.priority_score

        # Add evolution tracking
        if hypothesis.parent_hypothesis_id:
            properties["parent_hypothesis_id"] = hypothesis.parent_hypothesis_id
        properties["generation"] = hypothesis.generation
        properties["refinement_count"] = hypothesis.refinement_count

        # Add related papers
        if hypothesis.related_papers:
            properties["related_papers"] = hypothesis.related_papers

        return cls(
            id=hypothesis.id,
            type="Hypothesis",
            properties=properties,
            confidence=hypothesis.confidence_score or 1.0,
            created_at=hypothesis.created_at,
            updated_at=hypothesis.updated_at,
            created_by=created_by,
        )

    @classmethod
    def from_protocol(cls, protocol: Any, created_by: str = "experiment_designer") -> "Entity":
        """
        Create Entity from ExperimentProtocol Pydantic model.

        Args:
            protocol: ExperimentProtocol model from kosmos.models.experiment
            created_by: Agent that created this entity

        Returns:
            Entity representing the experiment protocol

        Example:
            from kosmos.models.experiment import ExperimentProtocol
            protocol = ExperimentProtocol(name="...", hypothesis_id="...", ...)
            entity = Entity.from_protocol(protocol, created_by="ExperimentDesignerAgent")
        """
        properties = {
            "name": protocol.name,
            "hypothesis_id": protocol.hypothesis_id,
            "experiment_type": protocol.experiment_type.value if hasattr(protocol.experiment_type, 'value') else str(protocol.experiment_type),
            "domain": protocol.domain,
            "description": protocol.description,
            "objective": protocol.objective,
        }

        # Add rigor score if present
        if hasattr(protocol, 'rigor_score') and protocol.rigor_score is not None:
            properties["rigor_score"] = protocol.rigor_score

        # Add template info if present
        if hasattr(protocol, 'template_name') and protocol.template_name:
            properties["template_name"] = protocol.template_name

        # Note: Not storing full steps/variables to keep entity lightweight
        # These details remain in SQLAlchemy model

        return cls(
            id=protocol.id,
            type="ExperimentProtocol",
            properties=properties,
            confidence=properties.get("rigor_score", 1.0),
            created_at=protocol.created_at if hasattr(protocol, 'created_at') else None,
            updated_at=protocol.updated_at if hasattr(protocol, 'updated_at') else None,
            created_by=created_by,
        )

    @classmethod
    def from_result(cls, result: Any, created_by: str = "executor") -> "Entity":
        """
        Create Entity from ExperimentResult Pydantic model.

        Args:
            result: ExperimentResult model from kosmos.models.result
            created_by: Agent that created this entity

        Returns:
            Entity representing the experiment result

        Example:
            from kosmos.models.result import ExperimentResult
            result = ExperimentResult(experiment_id="...", protocol_id="...", ...)
            entity = Entity.from_result(result, created_by="Executor")
        """
        properties = {
            "experiment_id": result.experiment_id,
            "protocol_id": result.protocol_id,
            "status": result.status.value if hasattr(result.status, 'value') else str(result.status),
        }

        # Add hypothesis link if present
        if hasattr(result, 'hypothesis_id') and result.hypothesis_id:
            properties["hypothesis_id"] = result.hypothesis_id

        # Add support/refute information
        if hasattr(result, 'supports_hypothesis') and result.supports_hypothesis is not None:
            properties["supports_hypothesis"] = result.supports_hypothesis

        # Add interpretation if present
        if hasattr(result, 'interpretation') and result.interpretation:
            properties["interpretation"] = result.interpretation

        # Add summary if present
        if hasattr(result, 'summary') and result.summary:
            properties["summary"] = result.summary

        return cls(
            id=result.id,
            type="ExperimentResult",
            properties=properties,
            confidence=1.0,
            created_at=result.created_at if hasattr(result, 'created_at') else None,
            updated_at=result.updated_at if hasattr(result, 'updated_at') else None,
            created_by=created_by,
        )

    @classmethod
    def from_research_question(
        cls,
        question_text: str,
        domain: Optional[str] = None,
        created_by: str = "research_director"
    ) -> "Entity":
        """
        Create Entity for a research question.

        Args:
            question_text: The research question text
            domain: Scientific domain
            created_by: Agent that created this entity

        Returns:
            Entity representing the research question

        Example:
            entity = Entity.from_research_question(
                "How do transformers learn long-range dependencies?",
                domain="machine_learning",
                created_by="ResearchDirectorAgent"
            )
        """
        properties = {"text": question_text}
        if domain:
            properties["domain"] = domain

        return cls(
            type="ResearchQuestion",
            properties=properties,
            created_by=created_by,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entity to dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON export

        Example:
            entity_dict = entity.to_dict()
            with open('export.json', 'w') as f:
                json.dump(entity_dict, f)
        """
        return {
            "id": self.id,
            "type": self.type,
            "properties": self.properties,
            "confidence": self.confidence,
            "project": self.project,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
            "verified": self.verified,
            "annotations": [
                {
                    "text": ann.text,
                    "created_by": ann.created_by,
                    "created_at": ann.created_at.isoformat() if ann.created_at else None,
                }
                for ann in self.annotations
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """
        Create entity from dictionary (for import).

        Args:
            data: Dictionary representation from export

        Returns:
            Entity instance

        Example:
            with open('export.json') as f:
                data = json.load(f)
            entity = Entity.from_dict(data)
        """
        # Parse timestamps
        created_at = None
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])

        updated_at = None
        if data.get("updated_at"):
            updated_at = datetime.fromisoformat(data["updated_at"])

        # Parse annotations
        annotations = []
        for ann_data in data.get("annotations", []):
            ann_created_at = None
            if ann_data.get("created_at"):
                ann_created_at = datetime.fromisoformat(ann_data["created_at"])

            annotations.append(
                Annotation(
                    text=ann_data["text"],
                    created_by=ann_data["created_by"],
                    created_at=ann_created_at,
                )
            )

        return cls(
            id=data["id"],
            type=data["type"],
            properties=data["properties"],
            confidence=data.get("confidence", 1.0),
            project=data.get("project"),
            created_at=created_at,
            updated_at=updated_at,
            created_by=data.get("created_by"),
            verified=data.get("verified", False),
            annotations=annotations,
        )


@dataclass
class Relationship:
    """
    Relationship between two entities.

    DESIGN RATIONALE:
    - source_id/target_id: Connect entities
    - type: Explicit relationship semantics
    - properties: Flexible attributes (e.g., citation context)
    - confidence: Uncertainty representation

    RELATIONSHIP TYPES:
    - CITES: Paper cites another paper
    - AUTHOR_OF: Author wrote paper
    - MENTIONS: Paper mentions concept
    - RELATES_TO: Concept relates to concept
    - SUPPORTS: Finding supports hypothesis
    - REFUTES: Finding refutes hypothesis
    - USES_METHOD: Experiment uses method
    - PRODUCED_BY: Finding from experiment
    - DERIVED_FROM: Entity derived from others (provenance)

    Attributes:
        source_id: ID of the source entity
        target_id: ID of the target entity
        type: Relationship type
        id: Unique identifier (auto-generated)
        properties: Additional relationship attributes
        confidence: Confidence score 0.0-1.0
        created_at: Creation timestamp
        created_by: Creator identifier

    Example:
        citation = Relationship(
            source_id="paper1_id",
            target_id="paper2_id",
            type="CITES",
            properties={"section": "introduction", "context": "builds on"},
            confidence=1.0,
            created_by="citation_extractor"
        )
    """

    source_id: str
    target_id: str
    type: str
    id: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    created_at: Optional[datetime] = None
    created_by: Optional[str] = None

    # Standard relationship types
    VALID_TYPES = {
        "CITES",
        "AUTHOR_OF",
        "MENTIONS",
        "RELATES_TO",
        "SUPPORTS",
        "REFUTES",
        "USES_METHOD",
        "PRODUCED_BY",
        "DERIVED_FROM",
        # Research workflow relationships
        "SPAWNED_BY",
        "TESTS",
        "REFINED_FROM",
    }

    def __post_init__(self):
        """Validate and initialize relationship."""
        # Generate ID if not provided
        if self.id is None:
            self.id = str(uuid.uuid4())

        # Validate IDs
        if not self.source_id or not self.target_id:
            raise ValueError("Source and target IDs are required")

        # Validate type
        if not self.type:
            raise ValueError("Relationship type is required")

        # Warn if non-standard type
        if self.type not in self.VALID_TYPES:
            warnings.warn(
                f"Relationship type '{self.type}' is not standard. "
                f"Standard types: {', '.join(sorted(self.VALID_TYPES))}",
                UserWarning,
                stacklevel=2,
            )

        # Validate confidence
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        # Set timestamp
        if self.created_at is None:
            self.created_at = datetime.now()

    @classmethod
    def with_provenance(
        cls,
        source_id: str,
        target_id: str,
        rel_type: str,
        agent: str,
        confidence: float = 1.0,
        **metadata: Any
    ) -> "Relationship":
        """
        Create relationship with rich provenance metadata.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            rel_type: Relationship type
            agent: Agent that created this relationship
            confidence: Confidence score (0.0-1.0)
            **metadata: Additional provenance metadata (e.g., p_value, effect_size, iteration)

        Returns:
            Relationship with provenance metadata

        Example:
            # Hypothesis support relationship
            rel = Relationship.with_provenance(
                source_id=result_id,
                target_id=hypothesis_id,
                rel_type="SUPPORTS",
                agent="DataAnalystAgent",
                confidence=0.95,
                p_value=0.001,
                effect_size=0.78,
                iteration=3
            )

            # Hypothesis spawning relationship
            rel = Relationship.with_provenance(
                source_id=hypothesis_id,
                target_id=question_id,
                rel_type="SPAWNED_BY",
                agent="HypothesisGeneratorAgent",
                generation=1
            )
        """
        # Create properties dict with timestamp and agent
        properties = {
            "agent": agent,
            "timestamp": datetime.now().isoformat(),
        }

        # Add all additional metadata
        properties.update(metadata)

        return cls(
            source_id=source_id,
            target_id=target_id,
            type=rel_type,
            properties=properties,
            confidence=confidence,
            created_by=agent,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert relationship to dictionary.

        Returns:
            Dictionary representation suitable for JSON export
        """
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type,
            "properties": self.properties,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        """
        Create relationship from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Relationship instance
        """
        created_at = None
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])

        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            type=data["type"],
            properties=data.get("properties", {}),
            confidence=data.get("confidence", 1.0),
            created_at=created_at,
            created_by=data.get("created_by"),
        )

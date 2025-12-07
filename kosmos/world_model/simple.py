"""
Neo4j-based world model implementation (Simple Mode).

This module implements the WorldModelStorage interface using Neo4j as the backend.
It wraps the existing kosmos.knowledge.graph.KnowledgeGraph to provide the world
model abstraction layer.

DESIGN PATTERN: Adapter Pattern
- WorldModelStorage: Target interface
- Neo4jWorldModel: Adapter
- KnowledgeGraph: Adaptee (existing code)

This allows us to use the existing, production-tested KnowledgeGraph implementation
while providing the new world model interface for future evolution.

EDUCATIONAL NOTE:
Instead of rewriting graph operations, we wrap existing code. This is:
- Faster to implement (reuse existing code)
- Lower risk (existing code is tested)
- Easier to evolve (can refactor internals later)

See: https://refactoring.guru/design-patterns/adapter
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from kosmos.knowledge import get_knowledge_graph, KnowledgeGraph
from kosmos.literature.base_client import PaperMetadata, PaperSource
from kosmos.world_model.interface import EntityManager, WorldModelStorage
from kosmos.world_model.models import (
    EXPORT_FORMAT_VERSION,
    Annotation,
    Entity,
    Relationship,
)

logger = logging.getLogger(__name__)


class Neo4jWorldModel(WorldModelStorage, EntityManager):
    """
    Simple Mode implementation using Neo4j.

    DESIGN RATIONALE:
    - Wraps existing kosmos/knowledge/graph.py (1,000+ lines of working code)
    - Single database, easy setup
    - Sufficient for <10K entities
    - Used by 90% of researchers

    STORAGE DETAILS:
    - Backend: Single Neo4j database
    - Node labels: Paper, Concept, Author, Method, Experiment, Hypothesis, Finding, Dataset
    - Relationship types: CITES, MENTIONS, RELATES_TO, SUPPORTS, REFUTES, etc.
    - Properties: JSON dict stored in Neo4j node properties
    - Indexes: Created automatically for performance

    IMPLEMENTATION NOTES:
    - Uses existing KnowledgeGraph singleton for connection management
    - Maps Entity/Relationship models to Neo4j nodes/relationships
    - Implements merge logic for intelligent accumulation
    - Project isolation via graph namespaces (node property filtering)

    Example:
        wm = Neo4jWorldModel()

        # Add entity
        paper = Entity(
            type="Paper",
            properties={"title": "Test", "year": 2024}
        )
        entity_id = wm.add_entity(paper)

        # Export graph
        wm.export_graph("backup.json")
    """

    def __init__(self):
        """
        Initialize with existing knowledge graph.

        Uses the singleton KnowledgeGraph for connection management,
        ensuring all components share the same Neo4j connection.
        """
        self.graph = get_knowledge_graph()  # Existing singleton
        logger.info("Neo4jWorldModel initialized (Simple Mode)")

    def add_entity(self, entity: Entity, merge: bool = True) -> str:
        """
        Add entity to graph.

        This method maps our generic Entity model to the existing KnowledgeGraph
        node types. For standard types (Paper, Concept, Author, Method), we use
        the existing optimized methods. For custom types, we create generic nodes.

        Args:
            entity: Entity to add
            merge: If True, merge with existing entity (prevents duplicates)

        Returns:
            Entity ID

        Implementation Strategy:
            - Paper → use graph.create_paper()
            - Concept → use graph.create_concept()
            - Author → use graph.create_author()
            - Method → use graph.create_method()
            - Other types → create generic labeled node
        """
        if entity.type == "Paper":
            return self._add_paper_entity(entity, merge)
        elif entity.type == "Concept":
            return self._add_concept_entity(entity, merge)
        elif entity.type == "Author":
            return self._add_author_entity(entity, merge)
        elif entity.type == "Method":
            return self._add_method_entity(entity, merge)
        else:
            # Generic entity types (Experiment, Hypothesis, Finding, Dataset, custom)
            return self._add_generic_entity(entity, merge)

    def _add_paper_entity(self, entity: Entity, merge: bool) -> str:
        """Add Paper entity using existing graph methods."""
        # Extract paper-specific properties
        title = entity.properties.get("title", "Untitled")
        authors = entity.properties.get("authors", [])
        year = entity.properties.get("year")
        doi = entity.properties.get("doi")
        abstract = entity.properties.get("abstract", "")

        # Create PaperMetadata object for create_paper method
        paper = PaperMetadata(
            id=entity.id,
            source=PaperSource.MANUAL,  # Default source for world model entities
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            doi=doi,
        )

        # Use existing create_paper method (handles merge logic)
        node = self.graph.create_paper(paper=paper, merge=merge)

        return entity.id

    def _add_concept_entity(self, entity: Entity, merge: bool) -> str:
        """Add Concept entity using existing graph methods."""
        name = entity.properties.get("name", entity.id)
        description = entity.properties.get("description")
        domain = entity.properties.get("domain")

        # Use existing create_concept method (no metadata parameter)
        node = self.graph.create_concept(
            name=name,
            description=description,
            domain=domain,
            merge=merge
        )

        return entity.id

    def _add_author_entity(self, entity: Entity, merge: bool) -> str:
        """Add Author entity using existing graph methods."""
        name = entity.properties.get("name", entity.id)
        affiliation = entity.properties.get("affiliation")
        email = entity.properties.get("email")

        metadata = {
            "confidence": entity.confidence,
            "project": entity.project,
            "created_by": entity.created_by,
            "entity_id": entity.id,
        }

        node = self.graph.create_author(
            name=name,
            affiliation=affiliation,
            email=email,
            metadata=metadata,
            merge=merge
        )

        return entity.id

    def _add_method_entity(self, entity: Entity, merge: bool) -> str:
        """Add Method entity using existing graph methods."""
        name = entity.properties.get("name", entity.id)
        description = entity.properties.get("description")
        category = entity.properties.get("category")

        metadata = {
            "confidence": entity.confidence,
            "project": entity.project,
            "created_by": entity.created_by,
            "entity_id": entity.id,
        }

        node = self.graph.create_method(
            name=name,
            description=description,
            category=category,
            metadata=metadata,
            merge=merge
        )

        return entity.id

    def _add_generic_entity(self, entity: Entity, merge: bool) -> str:
        """
        Add generic entity type (Experiment, Hypothesis, Finding, Dataset, custom).

        For entity types not in the standard KnowledgeGraph, we create
        generic nodes with the appropriate label.
        """
        # Build Cypher query to create or merge node
        if merge:
            # MERGE creates if not exists, matches if exists
            cypher = f"""
            MERGE (n:{entity.type} {{entity_id: $entity_id}})
            SET n.properties = $properties,
                n.confidence = $confidence,
                n.project = $project,
                n.created_by = $created_by,
                n.verified = $verified,
                n.created_at = $created_at,
                n.updated_at = $updated_at
            RETURN n.entity_id as entity_id
            """
        else:
            # CREATE always creates new node
            cypher = f"""
            CREATE (n:{entity.type} {{entity_id: $entity_id}})
            SET n.properties = $properties,
                n.confidence = $confidence,
                n.project = $project,
                n.created_by = $created_by,
                n.verified = $verified,
                n.created_at = $created_at,
                n.updated_at = $updated_at
            RETURN n.entity_id as entity_id
            """

        # Convert properties to JSON string for Neo4j storage
        result = self.graph.graph.run(
            cypher,
            entity_id=entity.id,
            properties=json.dumps(entity.properties),  # Store as JSON string
            confidence=entity.confidence,
            project=entity.project,
            created_by=entity.created_by,
            verified=entity.verified,
            created_at=entity.created_at.isoformat() if entity.created_at else None,
            updated_at=entity.updated_at.isoformat() if entity.updated_at else None
        ).data()

        if result:
            return result[0]["entity_id"]
        else:
            return entity.id

    def get_entity(self, entity_id: str, project: Optional[str] = None) -> Optional[Entity]:
        """
        Retrieve entity by ID.

        Args:
            entity_id: Unique entity identifier
            project: Optional project filter (None = any project)

        Returns:
            Entity if found, None otherwise

        Implementation:
            - Try standard node types (Paper, Concept, Author, Method)
            - If not found, query for generic entity types
            - Convert Neo4j node to Entity model
        """
        # Try standard types first
        node = self.graph.get_paper(entity_id)
        if node:
            return self._node_to_entity(node, "Paper")

        # Try other standard types
        # Note: KnowledgeGraph uses 'name' as primary key for Concept/Author/Method
        # We'll need to search by entity_id in metadata instead

        # Query for any node with matching entity_id
        cypher = """
        MATCH (n)
        WHERE n.entity_id = $entity_id OR n.paper_id = $entity_id
        RETURN n, labels(n) as labels
        LIMIT 1
        """

        result = self.graph.graph.run(cypher, entity_id=entity_id).data()

        if result:
            node = result[0]["n"]
            labels = result[0]["labels"]
            # Use the first label as entity type
            entity_type = labels[0] if labels else "Unknown"
            return self._node_to_entity(node, entity_type)

        return None

    def _node_to_entity(self, node: Dict[str, Any], entity_type: str) -> Entity:
        """Convert Neo4j node to Entity model."""
        # Extract properties
        properties_str = node.get("properties", "{}")
        if isinstance(properties_str, str):
            properties = json.loads(properties_str) if properties_str else {}
        else:
            properties = properties_str

        # For Paper nodes, properties are stored directly as node attributes
        if entity_type == "Paper":
            properties = {
                "title": node.get("title"),
                "authors": node.get("authors", []),
                "year": node.get("year"),
                "doi": node.get("doi"),
                "abstract": node.get("abstract"),
            }

        # Parse timestamps
        created_at = None
        if node.get("created_at"):
            try:
                created_at = datetime.fromisoformat(node["created_at"])
            except (ValueError, TypeError):
                pass

        updated_at = None
        if node.get("updated_at"):
            try:
                updated_at = datetime.fromisoformat(node["updated_at"])
            except (ValueError, TypeError):
                pass

        # Load annotations from node property
        annotations = []
        annotations_raw = node.get('annotations', [])
        if annotations_raw:
            for ann_json in annotations_raw:
                try:
                    if isinstance(ann_json, str):
                        ann_dict = json.loads(ann_json)
                    else:
                        ann_dict = ann_json

                    ann_created_at = None
                    if ann_dict.get('created_at'):
                        try:
                            ann_created_at = datetime.fromisoformat(ann_dict['created_at'])
                        except ValueError:
                            pass

                    annotations.append(Annotation(
                        text=ann_dict['text'],
                        created_by=ann_dict['created_by'],
                        created_at=ann_created_at
                    ))
                except Exception as e:
                    logger.debug(f"Failed to parse annotation in node: {e}")

        return Entity(
            id=node.get("entity_id", node.get("paper_id")),
            type=entity_type,
            properties=properties,
            confidence=node.get("confidence", 1.0),
            project=node.get("project"),
            created_at=created_at,
            updated_at=updated_at,
            created_by=node.get("created_by"),
            verified=node.get("verified", False),
            annotations=annotations,
        )

    def update_entity(self, entity_id: str, updates: Dict[str, Any]) -> None:
        """
        Update entity properties.

        Args:
            entity_id: Entity to update
            updates: Dict of property updates (merged with existing)

        Raises:
            EntityNotFoundError: If entity doesn't exist
        """
        # Query for node
        cypher = """
        MATCH (n)
        WHERE n.entity_id = $entity_id OR n.paper_id = $entity_id
        SET n += $updates
        SET n.updated_at = $updated_at
        RETURN count(n) as count
        """

        result = self.graph.graph.run(
            cypher,
            entity_id=entity_id,
            updates=updates,
            updated_at=datetime.now().isoformat()
        ).data()

        if not result or result[0]["count"] == 0:
            raise ValueError(f"Entity not found: {entity_id}")

    def delete_entity(self, entity_id: str) -> None:
        """
        Delete entity and all its relationships.

        Args:
            entity_id: Entity to delete

        Raises:
            EntityNotFoundError: If entity doesn't exist
        """
        # DETACH DELETE removes node and all its relationships
        cypher = """
        MATCH (n)
        WHERE n.entity_id = $entity_id OR n.paper_id = $entity_id
        DETACH DELETE n
        RETURN count(n) as count
        """

        result = self.graph.graph.run(cypher, entity_id=entity_id).data()

        if not result or result[0]["count"] == 0:
            raise ValueError(f"Entity not found: {entity_id}")

        logger.info(f"Deleted entity: {entity_id}")

    def add_relationship(self, relationship: Relationship) -> str:
        """
        Add relationship between two entities.

        Args:
            relationship: Relationship to add

        Returns:
            Relationship ID

        Raises:
            ValueError: If relationship validation fails
            EntityNotFoundError: If source or target entity doesn't exist

        Implementation:
            - Use existing create_citation, create_authored, etc. for standard types
            - Use generic relationship creation for custom types
        """
        # Use existing methods for standard relationship types
        if relationship.type == "CITES":
            # Note: create_citation expects paper_id, cited_paper_id
            self.graph.create_citation(
                paper_id=relationship.source_id,
                cited_paper_id=relationship.target_id,
                context=relationship.properties.get("context"),
                section=relationship.properties.get("section")
            )
            return relationship.id

        elif relationship.type == "AUTHOR_OF":
            self.graph.create_authored(
                author_name=relationship.source_id,
                paper_id=relationship.target_id
            )
            return relationship.id

        else:
            # Generic relationship creation
            return self._add_generic_relationship(relationship)

    def _add_generic_relationship(self, relationship: Relationship) -> str:
        """Add generic relationship type."""
        cypher = f"""
        MATCH (source), (target)
        WHERE (source.entity_id = $source_id OR source.paper_id = $source_id)
        AND (target.entity_id = $target_id OR target.paper_id = $target_id)
        CREATE (source)-[r:{relationship.type}]->(target)
        SET r.relationship_id = $relationship_id,
            r.properties = $properties,
            r.confidence = $confidence,
            r.created_at = $created_at,
            r.created_by = $created_by
        RETURN r.relationship_id as relationship_id
        """

        result = self.graph.graph.run(
            cypher,
            source_id=relationship.source_id,
            target_id=relationship.target_id,
            relationship_id=relationship.id,
            properties=json.dumps(relationship.properties),
            confidence=relationship.confidence,
            created_at=relationship.created_at.isoformat() if relationship.created_at else None,
            created_by=relationship.created_by
        ).data()

        if not result:
            raise ValueError(
                f"Could not create relationship: source={relationship.source_id}, "
                f"target={relationship.target_id}"
            )

        return result[0]["relationship_id"]

    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """
        Retrieve relationship by ID.

        Args:
            relationship_id: Relationship identifier

        Returns:
            Relationship if found, None otherwise
        """
        cypher = """
        MATCH (source)-[r]->(target)
        WHERE r.relationship_id = $relationship_id
        RETURN r, type(r) as rel_type,
               source.entity_id as source_id, source.paper_id as source_paper_id,
               target.entity_id as target_id, target.paper_id as target_paper_id
        """

        result = self.graph.graph.run(cypher, relationship_id=relationship_id).data()

        if not result:
            return None

        row = result[0]
        rel = row["r"]

        # Parse properties
        properties_str = rel.get("properties", "{}")
        if isinstance(properties_str, str):
            properties = json.loads(properties_str) if properties_str else {}
        else:
            properties = properties_str

        # Parse created_at
        created_at = None
        if rel.get("created_at"):
            try:
                created_at = datetime.fromisoformat(rel["created_at"])
            except (ValueError, TypeError):
                pass

        return Relationship(
            id=relationship_id,
            source_id=row.get("source_id") or row.get("source_paper_id"),
            target_id=row.get("target_id") or row.get("target_paper_id"),
            type=row["rel_type"],
            properties=properties,
            confidence=rel.get("confidence", 1.0),
            created_at=created_at,
            created_by=rel.get("created_by")
        )

    def query_related_entities(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "outgoing",
        max_depth: int = 1,
    ) -> List[Entity]:
        """
        Query entities related to a given entity.

        Args:
            entity_id: Starting entity
            relationship_type: Optional filter by relationship type
            direction: "outgoing", "incoming", or "both"
            max_depth: Maximum traversal depth (1 = direct neighbors)

        Returns:
            List of related entities
        """
        # Build Cypher query based on direction
        if direction == "outgoing":
            pattern = f"(start)-[r]->{{1,{max_depth}}}(related)"
        elif direction == "incoming":
            pattern = f"(related)-[r]->{{1,{max_depth}}}(start)"
        else:  # both
            pattern = f"(start)-[r*1..{max_depth}]-(related)"

        # Add relationship type filter if specified
        if relationship_type:
            pattern = pattern.replace("[r", f"[r:{relationship_type}")

        cypher = f"""
        MATCH {pattern}
        WHERE start.entity_id = $entity_id OR start.paper_id = $entity_id
        RETURN DISTINCT related, labels(related) as labels
        LIMIT 100
        """

        results = self.graph.graph.run(cypher, entity_id=entity_id).data()

        entities = []
        for row in results:
            node = row["related"]
            labels = row["labels"]
            entity_type = labels[0] if labels else "Unknown"
            entities.append(self._node_to_entity(node, entity_type))

        return entities

    def export_graph(self, filepath: str, project: Optional[str] = None) -> None:
        """
        Export knowledge graph to JSON file.

        Args:
            filepath: Output file path (.json)
            project: Optional project filter (None = all projects)

        Format:
            {
                "version": "1.0",
                "exported_at": "2024-01-15T10:30:00",
                "source": "kosmos",
                "mode": "simple",
                "project": "my_project",
                "statistics": {...},
                "entities": [...],
                "relationships": [...]
            }
        """
        logger.info(f"Exporting graph to {filepath}")

        # Get statistics
        stats = self.get_statistics(project)

        # Build project filter clause
        project_filter = f"WHERE n.project = '{project}'" if project else ""

        # Get all entities
        entity_cypher = f"""
        MATCH (n)
        {project_filter}
        RETURN n, labels(n) as labels
        """

        entity_results = self.graph.graph.run(entity_cypher).data()

        entities = []
        for row in entity_results:
            node = row["n"]
            labels = row["labels"]
            entity_type = labels[0] if labels else "Unknown"

            # Convert to Entity model then to dict for export
            entity = self._node_to_entity(node, entity_type)
            entities.append(entity.to_dict())

        # Get all relationships
        rel_cypher = """
        MATCH (source)-[r]->(target)
        RETURN r, type(r) as rel_type,
               source.entity_id as source_id, source.paper_id as source_paper_id,
               target.entity_id as target_id, target.paper_id as target_paper_id
        """

        rel_results = self.graph.graph.run(rel_cypher).data()

        relationships = []
        for row in rel_results:
            rel = row["r"]
            properties_str = rel.get("properties", "{}")
            if isinstance(properties_str, str):
                properties = json.loads(properties_str) if properties_str else {}
            else:
                properties = properties_str

            relationships.append({
                "id": rel.get("relationship_id", str(id(rel))),
                "source_id": row.get("source_id") or row.get("source_paper_id"),
                "target_id": row.get("target_id") or row.get("target_paper_id"),
                "type": row["rel_type"],
                "properties": properties,
                "confidence": rel.get("confidence", 1.0),
                "created_at": rel.get("created_at"),
                "created_by": rel.get("created_by")
            })

        # Create export data
        export_data = {
            "version": EXPORT_FORMAT_VERSION,
            "exported_at": datetime.now().isoformat(),
            "source": "kosmos",
            "mode": "simple",
            "project": project,
            "statistics": stats,
            "entities": entities,
            "relationships": relationships
        }

        # Write to file
        filepath_obj = Path(filepath)
        filepath_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(
            f"Exported {len(entities)} entities and {len(relationships)} relationships"
        )

    def import_graph(self, filepath: str, clear: bool = False, project: Optional[str] = None) -> None:
        """
        Import knowledge graph from JSON file.

        Args:
            filepath: Input file path
            clear: If True, clear existing graph before import
            project: Optional project to import into (overrides project in file)

        Implementation:
            1. Load and validate file
            2. Clear graph if requested
            3. Import entities first (to satisfy relationship constraints)
            4. Import relationships (after entities exist)
            5. Log statistics
        """
        logger.info(f"Importing graph from {filepath}")

        # Load file
        with open(filepath, "r") as f:
            data = json.load(f)

        # Validate format version
        if data.get("version") != EXPORT_FORMAT_VERSION:
            logger.warning(
                f"Import file version {data.get('version')} may not be compatible "
                f"with current version {EXPORT_FORMAT_VERSION}"
            )

        # Clear if requested
        if clear:
            if project:
                # Clear only specific project
                logger.warning(f"Clearing project: {project}")
                cypher = "MATCH (n {project: $project}) DETACH DELETE n"
                self.graph.graph.run(cypher, project=project)
            else:
                # Clear all
                logger.warning("Clearing all graph data")
                self.graph.clear_graph()

        # Import entities
        entities = data.get("entities", [])
        logger.info(f"Importing {len(entities)} entities...")

        for entity_data in entities:
            # Override project if specified
            if project:
                entity_data["project"] = project

            entity = Entity.from_dict(entity_data)
            self.add_entity(entity, merge=True)  # Merge to avoid duplicates

        # Import relationships
        relationships = data.get("relationships", [])
        logger.info(f"Importing {len(relationships)} relationships...")

        for rel_data in relationships:
            rel = Relationship.from_dict(rel_data)
            try:
                self.add_relationship(rel)
            except Exception as e:
                logger.warning(f"Failed to import relationship {rel.id}: {e}")

        logger.info("Import complete")

    def get_statistics(self, project: Optional[str] = None) -> Dict[str, Any]:
        """
        Get knowledge graph statistics.

        Args:
            project: Optional project filter

        Returns:
            Dictionary with counts and types
        """
        project_filter = f"WHERE n.project = '{project}'" if project else ""

        # Total entity count
        cypher_total = f"MATCH (n) {project_filter} RETURN count(n) as count"
        total_result = self.graph.graph.run(cypher_total).data()
        entity_count = total_result[0]["count"] if total_result else 0

        # Entity type breakdown
        cypher_types = f"""
        MATCH (n)
        {project_filter}
        RETURN labels(n) as labels, count(*) as count
        """
        type_results = self.graph.graph.run(cypher_types).data()

        entity_types = {}
        for row in type_results:
            labels = row["labels"]
            label = labels[0] if labels else "Unknown"
            entity_types[label] = row["count"]

        # Relationship count
        cypher_rels = "MATCH ()-[r]->() RETURN count(r) as count"
        rel_result = self.graph.graph.run(cypher_rels).data()
        relationship_count = rel_result[0]["count"] if rel_result else 0

        # Relationship type breakdown
        cypher_rel_types = """
        MATCH ()-[r]->()
        RETURN type(r) as rel_type, count(*) as count
        """
        rel_type_results = self.graph.graph.run(cypher_rel_types).data()

        relationship_types = {}
        for row in rel_type_results:
            relationship_types[row["rel_type"]] = row["count"]

        # Get projects list
        cypher_projects = "MATCH (n) WHERE n.project IS NOT NULL RETURN DISTINCT n.project as project"
        project_results = self.graph.graph.run(cypher_projects).data()
        projects = [row["project"] for row in project_results]

        # Query Neo4j database storage size
        storage_size_mb = self._get_neo4j_storage_size()

        return {
            "entity_count": entity_count,
            "relationship_count": relationship_count,
            "entity_types": entity_types,
            "relationship_types": relationship_types,
            "projects": projects,
            "storage_size_mb": storage_size_mb,
        }

    def _get_neo4j_storage_size(self) -> float:
        """
        Query Neo4j database storage size in MB.

        Attempts multiple methods to get the storage size:
        1. APOC monitor.store() procedure (if APOC is installed)
        2. Database statistics (Neo4j 4.x+)
        3. Estimation based on entity/relationship counts

        Returns:
            Storage size in MB, or 0.0 if unable to determine
        """
        try:
            # Method 1: Try APOC procedure (most accurate)
            try:
                cypher_apoc = "CALL apoc.monitor.store() YIELD totalStoreSize RETURN totalStoreSize"
                result = self.graph.graph.run(cypher_apoc).data()
                if result and "totalStoreSize" in result[0]:
                    size_bytes = result[0]["totalStoreSize"]
                    return round(size_bytes / (1024 * 1024), 2)
            except Exception:
                pass  # APOC not available

            # Method 2: Try db.stats (Neo4j 4.x+)
            try:
                cypher_stats = "CALL db.stats.retrieve('GRAPH COUNTS') YIELD data RETURN data"
                result = self.graph.graph.run(cypher_stats).data()
                if result:
                    # Stats available but don't include exact size
                    # Fall through to estimation
                    pass
            except Exception:
                pass  # db.stats not available

            # Method 3: Estimate based on counts
            # Average node size: ~500 bytes, Average relationship: ~200 bytes
            try:
                cypher_counts = """
                MATCH (n) WITH count(n) as nodes
                MATCH ()-[r]->() WITH nodes, count(r) as rels
                RETURN nodes, rels
                """
                result = self.graph.graph.run(cypher_counts).data()
                if result:
                    nodes = result[0].get("nodes", 0)
                    rels = result[0].get("rels", 0)
                    # Estimate: 500 bytes/node + 200 bytes/relationship + 20% overhead
                    estimated_bytes = (nodes * 500 + rels * 200) * 1.2
                    return round(estimated_bytes / (1024 * 1024), 2)
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"Could not determine Neo4j storage size: {e}")

        return 0.0

    def reset(self, project: Optional[str] = None) -> None:
        """
        Clear all knowledge graph data.

        WARNING: This is a destructive operation!

        Args:
            project: Optional project to reset (None = reset ALL data)
        """
        if project:
            logger.warning(f"Resetting project: {project}")
            cypher = "MATCH (n {project: $project}) DETACH DELETE n"
            self.graph.graph.run(cypher, project=project)
        else:
            logger.warning("Resetting ALL graph data")
            self.graph.clear_graph()

    def close(self) -> None:
        """
        Close connections and cleanup resources.

        Note: The underlying KnowledgeGraph uses py2neo which manages
        connections internally. No explicit close needed.
        """
        logger.info("Neo4jWorldModel closed")

    # EntityManager interface methods (Phase 2 - Curation)

    def verify_entity(self, entity_id: str, verified_by: str) -> None:
        """
        Mark entity as manually verified.

        Args:
            entity_id: Entity to verify
            verified_by: Who verified (email/username)
        """
        cypher = """
        MATCH (n)
        WHERE n.entity_id = $entity_id OR n.paper_id = $entity_id
        SET n.verified = true, n.verified_by = $verified_by, n.verified_at = $verified_at
        RETURN count(n) as count
        """

        result = self.graph.graph.run(
            cypher,
            entity_id=entity_id,
            verified_by=verified_by,
            verified_at=datetime.now().isoformat()
        ).data()

        if not result or result[0]["count"] == 0:
            raise ValueError(f"Entity not found: {entity_id}")

        logger.info(f"Entity {entity_id} verified by {verified_by}")

    def add_annotation(self, entity_id: str, annotation: Annotation) -> None:
        """
        Add annotation to entity.

        Stores annotations as a JSON array in the Neo4j node's 'annotations' property.
        Each annotation is serialized as a JSON string within the array.

        Args:
            entity_id: ID of entity to annotate (entity_id or paper_id)
            annotation: Annotation object to add

        Note:
            - Creates annotations array if it doesn't exist
            - Appends to existing annotations array
            - Timestamps are ISO 8601 formatted
        """
        if not self.connected:
            logger.warning(
                f"Not connected to Neo4j, annotation not persisted for {entity_id}"
            )
            return

        # Serialize annotation to JSON-compatible dict
        ann_dict = {
            'text': annotation.text,
            'created_by': annotation.created_by,
            'created_at': (annotation.created_at or datetime.utcnow()).isoformat(),
            'annotation_id': str(uuid.uuid4())  # Unique ID for each annotation
        }

        # Cypher query: append to annotations array, create if null
        query = """
        MATCH (n)
        WHERE n.entity_id = $entity_id OR n.paper_id = $entity_id
        SET n.annotations = CASE
            WHEN n.annotations IS NULL THEN [$annotation]
            ELSE n.annotations + $annotation
        END,
        n.updated_at = $updated_at
        RETURN count(n) as updated, n.entity_id as eid
        """

        try:
            result = self.graph.run(
                query,
                entity_id=entity_id,
                annotation=json.dumps(ann_dict),
                updated_at=datetime.utcnow().isoformat()
            ).data()

            if result and result[0]['updated'] > 0:
                logger.info(
                    f"Annotation added to {entity_id} by {annotation.created_by}: "
                    f"{annotation.text[:50]}{'...' if len(annotation.text) > 50 else ''}"
                )
            else:
                logger.warning(f"Entity not found for annotation: {entity_id}")

        except Exception as e:
            logger.error(f"Failed to add annotation to {entity_id}: {e}")
            raise

    def get_annotations(self, entity_id: str) -> List[Annotation]:
        """
        Get all annotations for an entity.

        Retrieves and deserializes the annotations array from the Neo4j node.

        Args:
            entity_id: ID of entity to query (entity_id or paper_id)

        Returns:
            List of Annotation objects, empty list if none or on error

        Note:
            - Gracefully handles missing annotations property
            - Skips malformed annotation entries with warning
            - Returns annotations in order they were added
        """
        if not self.connected:
            logger.debug(f"Not connected to Neo4j, returning empty annotations for {entity_id}")
            return []

        query = """
        MATCH (n)
        WHERE n.entity_id = $entity_id OR n.paper_id = $entity_id
        RETURN n.annotations as annotations
        """

        try:
            result = self.graph.run(query, entity_id=entity_id).data()

            if not result:
                logger.debug(f"Entity not found: {entity_id}")
                return []

            annotations_raw = result[0].get('annotations')
            if not annotations_raw:
                return []

            annotations = []
            for i, ann_json in enumerate(annotations_raw):
                try:
                    # Parse JSON string to dict
                    if isinstance(ann_json, str):
                        ann_dict = json.loads(ann_json)
                    else:
                        ann_dict = ann_json

                    # Reconstruct Annotation object
                    created_at = None
                    if ann_dict.get('created_at'):
                        try:
                            created_at = datetime.fromisoformat(ann_dict['created_at'])
                        except ValueError:
                            logger.debug(f"Invalid created_at format: {ann_dict['created_at']}")

                    annotations.append(Annotation(
                        text=ann_dict['text'],
                        created_by=ann_dict['created_by'],
                        created_at=created_at
                    ))

                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning(
                        f"Failed to parse annotation {i} for {entity_id}: {e}"
                    )
                    continue

            return annotations

        except Exception as e:
            logger.error(f"Failed to get annotations for {entity_id}: {e}")
            return []

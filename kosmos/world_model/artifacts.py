"""
Artifact-based State Manager for Kosmos.

Implements hybrid 4-layer architecture for research state management:
- Layer 1: JSON artifacts (human-readable debugging)
- Layer 2: Knowledge graph (structural queries) - optional
- Layer 3: Vector store (semantic search) - optional
- Layer 4: Citation tracking (evidence chains)

Gap addressed: Gap 1 (State Manager Architecture)
Pattern source: Hybrid approach combining kosmos-claude-skills-mcp + paper requirements

Key Design Decision: JSON artifacts as primary storage, with optional graph layers.

Why Hybrid?
- JSON artifacts: Easy debugging, version control, human inspection
- Knowledge graph: Powerful queries, relationship traversal
- Best of both worlds: Start simple, add graph when needed

Architecture Philosophy:
- Minimum viable: JSON artifacts work for prototyping
- Production scale: Graph needed for complex queries
- Gradual adoption: Add layers as requirements grow
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict, field
from enum import Enum

logger = logging.getLogger(__name__)


class UpdateType(str, Enum):
    """
    World Model update categories (Issue #55).

    Paper Claim: "The World Model integrates findings using three categories:
    Confirmation (data supports hypothesis), Conflict (data contradicts literature),
    Pruning (hypothesis refuted)"
    """
    CONFIRMATION = "confirmation"  # Data supports hypothesis
    CONFLICT = "conflict"          # Data contradicts existing evidence
    PRUNING = "pruning"            # Hypothesis refuted


@dataclass
class Finding:
    """Container for research findings."""
    finding_id: str
    cycle: int
    task_id: int
    summary: str
    statistics: Dict[str, Any]
    methods: Optional[str] = None
    interpretation: Optional[str] = None
    evidence_type: str = "data_analysis"
    notebook_path: Optional[str] = None
    citations: Optional[List[Dict]] = None
    scholar_eval: Optional[Dict] = None
    metadata: Optional[Dict] = None
    timestamp: Optional[str] = None
    hypothesis_id: Optional[str] = None  # Issue #55: Link finding to hypothesis
    refutes_hypothesis: bool = False  # Issue #55: Flag for hypothesis refutation
    confidence: float = 0.5  # Issue #55: Confidence score for the finding
    # Issue #60: Figure tracking for research artifacts
    figure_paths: Optional[List[str]] = None  # List of generated figure file paths
    figure_metadata: Optional[Dict] = None  # Figure metadata: {type, caption, dpi, etc.}
    # Issue #61: Notebook tracking for research artifacts
    notebook_metadata: Optional[Dict] = None  # Notebook metadata: {kernel, line_count, cell_count, etc.}
    # Issue #70: Null model validation for statistical grounding
    null_model_result: Optional[Dict] = None  # Null model validation: {p_value, persists_in_noise, etc.}
    # Issue #63: Failure mode detection
    failure_detection_result: Optional[Dict] = None  # Failure modes: {over_interpretation, invented_metrics, rabbit_hole}
    # Issue #62: Code line provenance
    code_provenance: Optional[Dict] = None  # CodeProvenance: {notebook_path, cell_index, start_line, end_line, code_snippet}
    # Issue #65: Expert validation tracking for accuracy benchmarking
    expert_validated: bool = False  # Whether expert has validated this finding
    validation_accurate: Optional[bool] = None  # Expert assessment of accuracy
    validation_timestamp: Optional[str] = None  # When validation was recorded
    validation_notes: Optional[str] = None  # Expert notes on validation

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.timestamp is None:
            data['timestamp'] = datetime.now().isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'Finding':
        """Create Finding from dictionary."""
        return cls(**data)


@dataclass
class Hypothesis:
    """Container for research hypotheses."""
    hypothesis_id: str
    statement: str
    status: str  # "supported", "refuted", "unknown"
    domain: Optional[str] = None
    confidence: float = 0.0
    supporting_evidence: Optional[List[str]] = None
    refuting_evidence: Optional[List[str]] = None
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'Hypothesis':
        """Create Hypothesis from dictionary."""
        return cls(**data)


@dataclass
class FindingIntegrationResult:
    """
    Result of integrating a finding into the world model (Issue #55).

    Captures the update type (confirmation/conflict/pruning) and provides
    reasoning for the categorization.
    """
    success: bool
    update_type: UpdateType
    affected_hypotheses: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "update_type": self.update_type.value,
            "affected_hypotheses": self.affected_hypotheses,
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }


class ArtifactStateManager:
    """
    Hybrid State Manager for Kosmos research system.

    Provides 4-layer architecture:
    1. JSON artifacts (always active)
    2. Knowledge graph (optional, via existing world_model)
    3. Vector store (optional, for semantic search)
    4. Citation tracking (integrated in JSON)

    Design Goals:
    - Human-readable artifacts for debugging
    - Complete traceability (claim → evidence)
    - Coherent state across 20 research cycles
    - Fast context retrieval for task generation
    - Conflict detection and resolution
    """

    def __init__(
        self,
        artifacts_dir: str = "artifacts",
        world_model=None,
        vector_store=None
    ):
        """
        Initialize Artifact State Manager.

        Args:
            artifacts_dir: Directory for JSON artifacts
            world_model: Optional knowledge graph (from kosmos.world_model)
            vector_store: Optional vector store for semantic search
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Optional advanced layers
        self.world_model = world_model
        self.vector_store = vector_store

        # In-memory caches
        self._findings_cache: Dict[str, Finding] = {}
        self._hypotheses_cache: Dict[str, Hypothesis] = {}

        logger.info(
            f"Initialized ArtifactStateManager at {self.artifacts_dir}"
            f" (graph={'enabled' if world_model else 'disabled'}, "
            f"vectors={'enabled' if vector_store else 'disabled'})"
        )

    # ============================================================================
    # Layer 1: JSON Artifacts (Core Functionality)
    # ============================================================================

    async def save_finding_artifact(
        self,
        cycle: int,
        task_id: int,
        finding: Dict
    ) -> Path:
        """
        Save finding as JSON artifact.

        Args:
            cycle: Research cycle number
            task_id: Task identifier within cycle
            finding: Finding dictionary with summary, statistics, etc.

        Returns:
            Path to saved artifact file
        """
        # Create cycle directory
        cycle_dir = self.artifacts_dir / f"cycle_{cycle}"
        cycle_dir.mkdir(parents=True, exist_ok=True)

        # Generate finding ID if not present
        if 'finding_id' not in finding:
            finding['finding_id'] = f"cycle{cycle}_task{task_id}"

        # Add cycle/task metadata
        finding['cycle'] = cycle
        finding['task_id'] = task_id

        # Create Finding object
        finding_obj = Finding.from_dict(finding) if isinstance(finding, dict) else finding

        # Save as JSON
        artifact_path = cycle_dir / f"task_{task_id}_finding.json"
        with open(artifact_path, 'w') as f:
            json.dump(finding_obj.to_dict(), f, indent=2)

        # Cache in memory
        self._findings_cache[finding_obj.finding_id] = finding_obj

        # Index to graph if available (Layer 2)
        if self.world_model:
            await self._index_finding_to_graph(finding_obj)

        # Index to vector store if available (Layer 3)
        if self.vector_store:
            await self._index_finding_to_vectors(finding_obj)

        logger.debug(f"Saved finding artifact: {artifact_path}")
        return artifact_path

    async def save_hypothesis(self, hypothesis: Dict) -> str:
        """
        Save hypothesis to artifacts.

        Args:
            hypothesis: Hypothesis dictionary

        Returns:
            Hypothesis ID
        """
        # Create hypotheses directory
        hypotheses_dir = self.artifacts_dir / "hypotheses"
        hypotheses_dir.mkdir(parents=True, exist_ok=True)

        # Create Hypothesis object
        hyp_obj = Hypothesis.from_dict(hypothesis)

        # Save as JSON
        hyp_path = hypotheses_dir / f"{hyp_obj.hypothesis_id}.json"
        with open(hyp_path, 'w') as f:
            json.dump(hyp_obj.to_dict(), f, indent=2)

        # Cache in memory
        self._hypotheses_cache[hyp_obj.hypothesis_id] = hyp_obj

        return hyp_obj.hypothesis_id

    def get_finding(self, finding_id: str) -> Optional[Finding]:
        """
        Retrieve finding by ID.

        Args:
            finding_id: Finding identifier

        Returns:
            Finding object or None if not found
        """
        # Check cache first
        if finding_id in self._findings_cache:
            return self._findings_cache[finding_id]

        # Search filesystem
        for cycle_dir in self.artifacts_dir.glob("cycle_*"):
            for artifact_file in cycle_dir.glob("task_*_finding.json"):
                with open(artifact_file, 'r') as f:
                    data = json.load(f)
                    if data.get('finding_id') == finding_id:
                        finding = Finding.from_dict(data)
                        self._findings_cache[finding_id] = finding
                        return finding

        return None

    def get_all_cycle_findings(self, cycle: int) -> List[Finding]:
        """
        Get all findings from a specific cycle.

        Args:
            cycle: Cycle number

        Returns:
            List of Finding objects
        """
        findings = []
        cycle_dir = self.artifacts_dir / f"cycle_{cycle}"

        if not cycle_dir.exists():
            return findings

        for artifact_file in cycle_dir.glob("task_*_finding.json"):
            try:
                with open(artifact_file, 'r') as f:
                    data = json.load(f)
                    finding = Finding.from_dict(data)
                    findings.append(finding)
                    # Cache
                    self._findings_cache[finding.finding_id] = finding
            except Exception as e:
                logger.error(f"Failed to load finding from {artifact_file}: {e}")

        return findings

    def get_validated_findings(self) -> List[Finding]:
        """
        Get all findings that passed ScholarEval validation.

        Returns:
            List of validated Finding objects
        """
        all_findings = self.get_all_findings()
        validated = [
            f for f in all_findings
            if f.scholar_eval and f.scholar_eval.get('passes_threshold', False)
        ]
        return validated

    def get_all_findings(self) -> List[Finding]:
        """
        Get all findings across all cycles.

        Returns:
            List of all Finding objects
        """
        findings = []
        for cycle_dir in sorted(self.artifacts_dir.glob("cycle_*")):
            cycle_num = int(cycle_dir.name.split('_')[1])
            cycle_findings = self.get_all_cycle_findings(cycle_num)
            findings.extend(cycle_findings)
        return findings

    # ============================================================================
    # Context Retrieval for Task Generation
    # ============================================================================

    def get_cycle_context(self, cycle: int, lookback: int = 3) -> Dict:
        """
        Get context for task generation in a cycle.

        Provides:
        - Recent findings (last N cycles)
        - Unsupported hypotheses
        - Validated discoveries
        - Statistics summary

        Args:
            cycle: Current cycle number
            lookback: Number of past cycles to include (default: 3)

        Returns:
            Dictionary with context information
        """
        # Get recent findings
        recent_findings = []
        for c in range(max(1, cycle - lookback), cycle + 1):
            cycle_findings = self.get_all_cycle_findings(c)
            recent_findings.extend(cycle_findings)

        # Get unsupported hypotheses
        unsupported_hypotheses = self._get_unsupported_hypotheses()

        # Get validated discoveries
        validated_discoveries = self.get_validated_findings()

        # Compute statistics
        total_findings = len(self.get_all_findings())
        validated_count = len(validated_discoveries)
        validation_rate = validated_count / total_findings if total_findings > 0 else 0

        return {
            "cycle": cycle,
            "findings_count": len(recent_findings),
            "recent_findings": [f.to_dict() for f in recent_findings[-10:]],  # Last 10
            "unsupported_hypotheses": [h.to_dict() for h in unsupported_hypotheses],
            "validated_discoveries": [f.to_dict() for f in validated_discoveries],
            "statistics": {
                "total_findings": total_findings,
                "validated_findings": validated_count,
                "validation_rate": validation_rate,
                "cycles_completed": cycle - 1
            }
        }

    def _get_unsupported_hypotheses(self) -> List[Hypothesis]:
        """Get hypotheses that lack sufficient supporting evidence."""
        hypotheses_dir = self.artifacts_dir / "hypotheses"
        if not hypotheses_dir.exists():
            return []

        unsupported = []
        for hyp_file in hypotheses_dir.glob("*.json"):
            try:
                with open(hyp_file, 'r') as f:
                    data = json.load(f)
                    hyp = Hypothesis.from_dict(data)
                    if hyp.status == "unknown" or hyp.confidence < 0.5:
                        unsupported.append(hyp)
            except Exception as e:
                logger.error(f"Failed to load hypothesis from {hyp_file}: {e}")

        return unsupported

    # ============================================================================
    # Cycle Summarization
    # ============================================================================

    async def generate_cycle_summary(self, cycle: int) -> str:
        """
        Generate markdown summary for a cycle.

        Args:
            cycle: Cycle number

        Returns:
            Markdown-formatted summary string
        """
        findings = self.get_all_cycle_findings(cycle)

        summary = f"# Cycle {cycle} Summary\n\n"
        summary += f"**Date**: {datetime.now().strftime('%Y-%m-%d')}\n"
        summary += f"**Total Findings**: {len(findings)}\n"

        # Count validated findings
        validated = [
            f for f in findings
            if f.scholar_eval and f.scholar_eval.get('passes_threshold', False)
        ]
        summary += f"**Validated Findings**: {len(validated)}\n"
        validation_rate = (len(validated) / len(findings) * 100) if findings else 0
        summary += f"**Validation Rate**: {validation_rate:.1f}%\n\n"

        # Key findings
        summary += "## Key Findings\n\n"
        for i, finding in enumerate(validated[:5], 1):  # Top 5 validated
            summary += f"### Finding {i}: {finding.summary.split('.')[0]}\n\n"

            # Statistics
            if finding.statistics:
                summary += "**Statistics**:\n"
                for key, value in finding.statistics.items():
                    if isinstance(value, float):
                        summary += f"- {key}: {value:.4f}\n"
                    else:
                        summary += f"- {key}: {value}\n"
                summary += "\n"

            # Evidence with code provenance (Issue #62)
            if finding.code_provenance:
                prov = finding.code_provenance
                hyperlink = f"{prov['notebook_path']}#cell={prov['cell_index']}&line={prov['start_line']}"
                filename = prov['notebook_path'].split('/')[-1]
                summary += f"**Code Citation**: [{filename}]({hyperlink})"
                if prov.get('start_line') and prov.get('end_line'):
                    summary += f" (lines {prov['start_line']}-{prov['end_line']})"
                summary += "\n\n"
            elif finding.notebook_path:
                summary += f"**Evidence**: `{finding.notebook_path}`\n\n"

        # Save summary to file
        cycle_dir = self.artifacts_dir / f"cycle_{cycle}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        summary_path = cycle_dir / "summary.md"
        with open(summary_path, 'w') as f:
            f.write(summary)

        return summary

    # ============================================================================
    # Layer 2: Knowledge Graph Integration (Optional)
    # ============================================================================

    async def _index_finding_to_graph(self, finding: Finding):
        """
        Index finding to knowledge graph.

        Creates:
        - Finding node with properties
        - DERIVES_FROM relationship to evidence
        - SUPPORTS/REFUTES relationships to hypotheses

        Args:
            finding: Finding object to index
        """
        if not self.world_model:
            return

        try:
            from kosmos.world_model import Entity, Relationship

            # Create Finding entity
            finding_entity = Entity(
                type="Finding",
                properties={
                    "finding_id": finding.finding_id,
                    "summary": finding.summary,
                    "cycle": finding.cycle,
                    "task_id": finding.task_id,
                    "confidence": finding.scholar_eval.get('overall_score', 0)
                    if finding.scholar_eval else 0,
                    "timestamp": finding.timestamp or datetime.now().isoformat()
                }
            )

            entity_id = self.world_model.add_entity(finding_entity)

            # Create relationship to evidence if notebook_path exists
            if finding.notebook_path:
                evidence_entity = Entity(
                    type="Evidence",
                    properties={
                        "path": finding.notebook_path,
                        "type": finding.evidence_type
                    }
                )
                evidence_id = self.world_model.add_entity(evidence_entity)

                derives_from = Relationship(
                    source_id=entity_id,
                    target_id=evidence_id,
                    type="DERIVES_FROM",
                    properties={"evidence_type": finding.evidence_type}
                )
                self.world_model.add_relationship(derives_from)

            logger.debug(f"Indexed finding {finding.finding_id} to knowledge graph")

        except Exception as e:
            logger.warning(f"Failed to index finding to graph: {e}", exc_info=True)
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise

    async def _index_finding_to_vectors(self, finding: Finding):
        """Index finding to vector store for semantic search."""
        if not self.vector_store:
            return

        # Future implementation: Use sentence-transformers
        # to create embeddings and store in vector DB
        pass

    # ============================================================================
    # Conflict Detection (Issue #55)
    # ============================================================================

    async def add_finding_with_conflict_check(
        self, finding: Dict
    ) -> FindingIntegrationResult:
        """
        Add finding with automatic conflict detection (Issue #55).

        Implements the paper's three update categories:
        - Confirmation: Data supports hypothesis
        - Conflict: Data contradicts existing evidence
        - Pruning: Hypothesis refuted

        Args:
            finding: Finding dictionary with keys:
                - cycle, task_id: Location identifiers
                - hypothesis_id: Optional linked hypothesis
                - statistics: Dict with effect_size, p_value, etc.
                - refutes_hypothesis: Optional bool flag

        Returns:
            FindingIntegrationResult with update_type and reasoning
        """
        # Save the finding
        cycle = finding.get('cycle', 0)
        task_id = finding.get('task_id', 0)
        await self.save_finding_artifact(cycle, task_id, finding)

        # Determine update type
        update_type = UpdateType.CONFIRMATION
        affected = []
        reasoning = ""

        # Check for statistical contradictions
        new_stats = finding.get('statistics', {})
        existing_findings = await self._get_related_findings(finding)

        for existing in existing_findings:
            existing_stats = existing.get('statistics', {})

            # Check effect direction contradiction
            new_effect = new_stats.get('effect_size', 0)
            old_effect = existing_stats.get('effect_size', 0)

            # Non-zero effects with opposite signs indicate conflict
            if new_effect != 0 and old_effect != 0 and (new_effect * old_effect < 0):
                update_type = UpdateType.CONFLICT
                reasoning = (
                    f"Effect direction contradiction: new effect={new_effect:.3f} "
                    f"vs existing effect={old_effect:.3f}"
                )
                logger.warning(f"[CONFLICT] {reasoning}")
                break

            # Check p-value contradiction (significant vs non-significant)
            new_p = new_stats.get('p_value')
            old_p = existing_stats.get('p_value')
            if new_p is not None and old_p is not None:
                new_sig = new_p < 0.05
                old_sig = old_p < 0.05
                if new_sig != old_sig:
                    update_type = UpdateType.CONFLICT
                    reasoning = (
                        f"Significance contradiction: new p={new_p:.4f} "
                        f"({'significant' if new_sig else 'non-significant'}) vs "
                        f"existing p={old_p:.4f} "
                        f"({'significant' if old_sig else 'non-significant'})"
                    )
                    logger.warning(f"[CONFLICT] {reasoning}")
                    break

        # Check for hypothesis refutation (pruning)
        hypothesis_id = finding.get('hypothesis_id')
        if hypothesis_id and finding.get('refutes_hypothesis', False):
            update_type = UpdateType.PRUNING
            affected.append(hypothesis_id)
            reasoning = f"Hypothesis {hypothesis_id} refuted by experimental evidence"
            logger.info(f"[PRUNING] {reasoning}")

        # Log the categorization
        if update_type == UpdateType.CONFIRMATION and not reasoning:
            reasoning = "Finding supports existing hypotheses"
            logger.debug(f"[CONFIRMATION] {reasoning}")

        return FindingIntegrationResult(
            success=True,
            update_type=update_type,
            affected_hypotheses=affected,
            confidence=finding.get('confidence', 0.5),
            reasoning=reasoning
        )

    async def _get_related_findings(self, finding: Dict) -> List[Dict]:
        """
        Get findings related to the same hypothesis or topic.

        Args:
            finding: The new finding to compare against

        Returns:
            List of related findings as dicts from cache
        """
        hypothesis_id = finding.get('hypothesis_id')
        if not hypothesis_id:
            return []

        related = []
        for fid, cached in self._findings_cache.items():
            # Cache stores Finding objects, convert to dict for comparison
            cached_hyp_id = cached.hypothesis_id if hasattr(cached, 'hypothesis_id') else None
            if cached_hyp_id == hypothesis_id:
                related.append(cached.to_dict() if hasattr(cached, 'to_dict') else cached)
        return related

    # ============================================================================
    # Export/Import
    # ============================================================================

    def export_artifacts(self, output_path: str):
        """
        Export all artifacts to a single JSON file.

        Args:
            output_path: Path to output JSON file
        """
        export_data = {
            "findings": [f.to_dict() for f in self.get_all_findings()],
            "hypotheses": list(self._hypotheses_cache.values()),
            "export_timestamp": datetime.now().isoformat()
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported artifacts to {output_path}")

    def get_statistics(self) -> Dict:
        """
        Get statistics about stored artifacts.

        Returns:
            Dictionary with counts and metadata
        """
        all_findings = self.get_all_findings()
        validated = self.get_validated_findings()

        cycles = set(f.cycle for f in all_findings)

        return {
            "total_findings": len(all_findings),
            "validated_findings": len(validated),
            "validation_rate": len(validated) / len(all_findings) if all_findings else 0,
            "cycles_completed": max(cycles) if cycles else 0,
            "total_hypotheses": len(self._hypotheses_cache),
            "artifacts_directory": str(self.artifacts_dir)
        }

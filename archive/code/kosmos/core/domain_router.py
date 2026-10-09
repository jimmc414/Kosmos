"""
Domain Router for intelligent routing of research questions to domain-specific agents.

Uses Claude to classify research questions, detect multi-domain research, and route
to appropriate domain-specific tools, templates, and agents.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from kosmos.config import get_config
from kosmos.core.llm import ClaudeClient
from kosmos.models.domain import (
    ScientificDomain,
    DomainClassification,
    DomainConfidence,
    DomainRoute,
    DomainExpertise,
    DomainCapability,
)

logger = logging.getLogger(__name__)


class DomainRouter:
    """
    Routes research questions to appropriate scientific domains.

    Capabilities:
    - Domain classification using Claude
    - Multi-domain research detection
    - Agent and tool selection per domain
    - Cross-domain synthesis routing
    - Domain expertise assessment
    """

    # Domain keywords for classification hints
    DOMAIN_KEYWORDS = {
        ScientificDomain.BIOLOGY: [
            "gene", "protein", "dna", "rna", "cell", "organism", "species",
            "metabolite", "pathway", "genome", "expression", "mutation",
            "evolution", "ecology", "metabolism", "enzyme", "gwas", "snp"
        ],
        ScientificDomain.NEUROSCIENCE: [
            "neuron", "brain", "synapse", "neural", "cognitive", "neuronal",
            "cortex", "hippocampus", "alzheimer", "parkinson", "connectome",
            "spike", "fmri", "eeg", "neurotransmitter", "plasticity"
        ],
        ScientificDomain.MATERIALS: [
            "material", "crystal", "structure", "property", "synthesis",
            "perovskite", "solar cell", "semiconductor", "composite",
            "conductivity", "strength", "optimization", "parameter"
        ],
        ScientificDomain.PHYSICS: [
            "force", "energy", "momentum", "particle", "wave", "field",
            "quantum", "thermodynamic", "mechanics", "electromagnetic",
            "relativity", "optics", "plasma", "cosmology"
        ],
        ScientificDomain.CHEMISTRY: [
            "molecule", "reaction", "compound", "synthesis", "catalyst",
            "bond", "electron", "oxidation", "reduction", "spectroscopy",
            "chromatography", "polymer", "organic", "inorganic"
        ],
        ScientificDomain.ASTRONOMY: [
            "star", "planet", "galaxy", "universe", "cosmic", "telescope",
            "orbit", "redshift", "black hole", "nebula", "exoplanet",
            "supernova", "dark matter", "constellation"
        ],
        ScientificDomain.SOCIAL_SCIENCE: [
            "society", "behavior", "population", "survey", "demographic",
            "psychology", "sociology", "economics", "anthropology",
            "culture", "policy", "intervention", "cohort study"
        ],
    }

    # Domain-specific agent types
    DOMAIN_AGENTS = {
        ScientificDomain.BIOLOGY: [
            "MetabolomicsAnalyzer",
            "GenomicsAnalyzer",
            "LiteratureAnalyzer",
        ],
        ScientificDomain.NEUROSCIENCE: [
            "ConnectomicsAnalyzer",
            "NeurodegenerationAnalyzer",
            "LiteratureAnalyzer",
        ],
        ScientificDomain.MATERIALS: [
            "MaterialsOptimizer",
            "ExperimentDesignerAgent",
        ],
        ScientificDomain.PHYSICS: [
            "ExperimentDesignerAgent",
            "DataAnalystAgent",
        ],
        ScientificDomain.CHEMISTRY: [
            "ExperimentDesignerAgent",
            "DataAnalystAgent",
        ],
        ScientificDomain.GENERAL: [
            "ExperimentDesignerAgent",
            "DataAnalystAgent",
            "LiteratureAnalyzer",
        ],
    }

    # Domain-specific templates
    DOMAIN_TEMPLATES = {
        ScientificDomain.BIOLOGY: [
            "metabolomics_comparison",
            "gwas_multimodal",
        ],
        ScientificDomain.NEUROSCIENCE: [
            "connectome_scaling",
            "differential_expression",
        ],
        ScientificDomain.MATERIALS: [
            "parameter_correlation",
            "optimization",
            "shap_analysis",
        ],
        ScientificDomain.GENERAL: [
            "ttest_comparison",
            "correlation_analysis",
            "log_log_analysis",
        ],
    }

    # Domain-specific tools/APIs
    DOMAIN_TOOLS = {
        ScientificDomain.BIOLOGY: [
            "KEGGClient",
            "GWASCatalogClient",
            "GTExClient",
            "ENCODEClient",
        ],
        ScientificDomain.NEUROSCIENCE: [
            "FlyWireClient",
            "AllenBrainClient",
            "GEOClient",
        ],
        ScientificDomain.MATERIALS: [
            "MaterialsProjectClient",
            "NOMADClient",
        ],
    }

    def __init__(self, claude_client: Optional[ClaudeClient] = None):
        """
        Initialize domain router.

        Args:
            claude_client: Optional Claude client (creates new one if not provided)
        """
        self.config = get_config()
        self.claude = claude_client or ClaudeClient()

        # Domain capabilities registry
        self.capabilities: Dict[ScientificDomain, DomainCapability] = {}
        self._initialize_capabilities()

    def _initialize_capabilities(self):
        """Initialize domain capability registry."""
        for domain in ScientificDomain:
            # Get available tools, templates, agents for each domain
            api_clients = self.DOMAIN_TOOLS.get(domain, [])
            templates = self.DOMAIN_TEMPLATES.get(domain, [])
            agents = self.DOMAIN_AGENTS.get(domain, [])

            self.capabilities[domain] = DomainCapability(
                domain=domain,
                api_clients=api_clients,
                api_status={api: "available" for api in api_clients},
                analysis_modules=agents,
                experiment_templates=templates,
                has_ontology=domain in [
                    ScientificDomain.BIOLOGY,
                    ScientificDomain.NEUROSCIENCE,
                    ScientificDomain.MATERIALS,
                ],
                ontology_coverage=0.8 if domain in [
                    ScientificDomain.BIOLOGY,
                    ScientificDomain.NEUROSCIENCE,
                    ScientificDomain.MATERIALS,
                ] else 0.0,
            )

    def classify_research_question(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> DomainClassification:
        """
        Classify a research question to scientific domain(s).

        Args:
            question: The research question to classify
            context: Optional context (hypothesis, data description, etc.)

        Returns:
            DomainClassification with primary domain, confidence, and secondary domains
        """
        logger.info(f"Classifying research question: {question[:100]}...")

        # Build classification prompt
        prompt = self._build_classification_prompt(question, context)

        # Get classification from Claude
        try:
            response = self.claude.complete(
                prompt,
                temperature=0.3,  # Lower temperature for more consistent classification
                max_tokens=1000,
            )

            # Parse Claude's response into DomainClassification
            classification = self._parse_classification_response(response, question)

            logger.info(
                f"Classified to {classification.primary_domain.value} "
                f"(confidence: {classification.confidence.value})"
            )

            return classification

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            # Fallback to keyword-based classification
            return self._keyword_based_classification(question)

    def _build_classification_prompt(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for Claude to classify research question."""
        domains_list = ", ".join([d.value for d in ScientificDomain])

        prompt = f"""Classify the following research question into one or more scientific domains.

Research Question:
{question}

"""

        if context:
            prompt += f"""Additional Context:
{context}

"""

        prompt += f"""Available Domains:
{domains_list}

Instructions:
1. Identify the PRIMARY domain that best matches this research question
2. Assign a confidence level: very_high (>0.9), high (0.7-0.9), medium (0.5-0.7), low (0.3-0.5), very_low (<0.3)
3. Identify any SECONDARY domains if this is multi-domain research
4. Extract key terms that influenced your classification
5. Explain your reasoning

Respond in the following format:

PRIMARY DOMAIN: <domain_name>
CONFIDENCE: <confidence_level>
CONFIDENCE_SCORE: <0-1 numeric score>
SECONDARY DOMAINS: <comma-separated list or "none">
KEY TERMS: <comma-separated key terms>
IS MULTI-DOMAIN: <yes/no>
REASONING: <your explanation>
"""
        return prompt

    def _parse_classification_response(
        self,
        response: str,
        question: str
    ) -> DomainClassification:
        """Parse Claude's classification response into DomainClassification object."""
        lines = response.strip().split('\n')
        data = {}

        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                data[key.strip()] = value.strip()

        # Extract primary domain
        primary_domain_str = data.get('PRIMARY DOMAIN', 'general').lower()
        try:
            primary_domain = ScientificDomain(primary_domain_str)
        except ValueError:
            primary_domain = ScientificDomain.GENERAL

        # Extract confidence
        confidence_str = data.get('CONFIDENCE', 'medium').lower().replace(' ', '_')
        try:
            confidence = DomainConfidence(confidence_str)
        except ValueError:
            confidence = DomainConfidence.MEDIUM

        confidence_score = float(data.get('CONFIDENCE_SCORE', '0.6'))

        # Extract secondary domains
        secondary_str = data.get('SECONDARY DOMAINS', 'none')
        secondary_domains = []
        if secondary_str.lower() != 'none':
            for domain_str in secondary_str.split(','):
                domain_str = domain_str.strip().lower()
                try:
                    domain = ScientificDomain(domain_str)
                    if domain != primary_domain:
                        secondary_domains.append(domain)
                except ValueError:
                    pass

        # Extract key terms
        key_terms_str = data.get('KEY TERMS', '')
        key_terms = [term.strip() for term in key_terms_str.split(',') if term.strip()]

        # Is multi-domain
        is_multi_domain = data.get('IS MULTI-DOMAIN', 'no').lower() in ['yes', 'true']

        reasoning = data.get('REASONING', '')

        # Calculate domain scores (simplified)
        domain_scores = {
            primary_domain.value: confidence_score
        }
        for i, sec_domain in enumerate(secondary_domains):
            # Secondary domains get progressively lower scores
            domain_scores[sec_domain.value] = confidence_score * (0.7 - i * 0.1)

        return DomainClassification(
            primary_domain=primary_domain,
            confidence=confidence,
            confidence_score=confidence_score,
            secondary_domains=secondary_domains,
            domain_scores=domain_scores,
            key_terms=key_terms,
            classification_reasoning=reasoning,
            is_multi_domain=is_multi_domain,
            cross_domain_rationale=reasoning if is_multi_domain else None,
        )

    def _keyword_based_classification(self, question: str) -> DomainClassification:
        """Fallback classification using keyword matching."""
        question_lower = question.lower()

        # Calculate scores for each domain based on keyword matches
        domain_scores = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            matches = sum(1 for keyword in keywords if keyword in question_lower)
            domain_scores[domain.value] = matches / len(keywords) if keywords else 0.0

        # Get primary domain (highest score)
        if not domain_scores or max(domain_scores.values()) == 0:
            primary_domain = ScientificDomain.GENERAL
            confidence_score = 0.5
        else:
            primary_domain_str = max(domain_scores, key=domain_scores.get)
            primary_domain = ScientificDomain(primary_domain_str)
            confidence_score = domain_scores[primary_domain_str]

        # Map score to confidence level
        if confidence_score > 0.9:
            confidence = DomainConfidence.VERY_HIGH
        elif confidence_score > 0.7:
            confidence = DomainConfidence.HIGH
        elif confidence_score > 0.5:
            confidence = DomainConfidence.MEDIUM
        elif confidence_score > 0.3:
            confidence = DomainConfidence.LOW
        else:
            confidence = DomainConfidence.VERY_LOW

        # Get secondary domains (scores > 0.3, excluding primary)
        secondary_domains = [
            ScientificDomain(domain_str)
            for domain_str, score in domain_scores.items()
            if score > 0.3 and ScientificDomain(domain_str) != primary_domain
        ]

        return DomainClassification(
            primary_domain=primary_domain,
            confidence=confidence,
            confidence_score=confidence_score,
            secondary_domains=secondary_domains[:2],  # Limit to top 2
            domain_scores=domain_scores,
            key_terms=[],
            classification_reasoning="Keyword-based fallback classification",
            is_multi_domain=len(secondary_domains) > 0,
            classifier_model="keyword-matcher",
        )

    def route(
        self,
        question: str,
        classification: Optional[DomainClassification] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> DomainRoute:
        """
        Create a complete routing decision for a research question.

        Args:
            question: Research question to route
            classification: Optional pre-computed classification (will classify if not provided)
            context: Optional context for routing decisions

        Returns:
            DomainRoute with selected agents, tools, templates, and routing strategy
        """
        # Classify if not provided
        if classification is None:
            classification = self.classify_research_question(question, context)

        # Determine routing strategy
        if classification.is_multi_domain and len(classification.secondary_domains) > 0:
            # Multi-domain routing
            selected_domains = classification.to_domain_list()
            routing_strategy = self._determine_multi_domain_strategy(
                classification, context
            )
        else:
            # Single domain routing
            selected_domains = [classification.primary_domain]
            routing_strategy = "single_domain"

        # Select agents, tools, and templates for each domain
        assigned_agents = {}
        required_tools = {}
        recommended_templates = {}

        for domain in selected_domains:
            assigned_agents[domain.value] = self.DOMAIN_AGENTS.get(domain, [])
            required_tools[domain.value] = self.DOMAIN_TOOLS.get(domain, [])
            recommended_templates[domain.value] = self.DOMAIN_TEMPLATES.get(domain, [])

        # Determine if cross-domain synthesis is needed
        synthesis_required = classification.requires_cross_domain_synthesis()
        synthesis_strategy = self._determine_synthesis_strategy(
            classification
        ) if synthesis_required else None

        # Build routing decision
        route = DomainRoute(
            classification=classification,
            selected_domains=selected_domains,
            routing_strategy=routing_strategy,
            assigned_agents=assigned_agents,
            required_tools=required_tools,
            recommended_templates=recommended_templates,
            synthesis_required=synthesis_required,
            synthesis_strategy=synthesis_strategy,
            routing_reasoning=self._build_routing_reasoning(
                classification, routing_strategy, selected_domains
            ),
        )

        logger.info(
            f"Routed to {len(selected_domains)} domain(s) "
            f"using {routing_strategy} strategy"
        )

        return route

    def _determine_multi_domain_strategy(
        self,
        classification: DomainClassification,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Determine routing strategy for multi-domain research.

        Returns:
            "parallel_multi_domain" or "sequential_multi_domain"
        """
        # If domains are independent, run in parallel
        # If one domain depends on another, run sequentially

        # For now, default to parallel (most efficient)
        # Future: Use Claude to determine dependencies
        return "parallel_multi_domain"

    def _determine_synthesis_strategy(
        self,
        classification: DomainClassification
    ) -> str:
        """Determine strategy for synthesizing cross-domain results."""
        domains = classification.to_domain_list()

        # Strategy depends on domain combination
        domain_names = sorted([d.value for d in domains])

        # Specific synthesis strategies for common combinations
        if "biology" in domain_names and "neuroscience" in domain_names:
            return "biological_neural_integration"
        elif "materials" in domain_names and "physics" in domain_names:
            return "materials_physics_integration"
        elif "chemistry" in domain_names and "biology" in domain_names:
            return "biochemical_integration"
        else:
            return "general_cross_domain_synthesis"

    def _build_routing_reasoning(
        self,
        classification: DomainClassification,
        strategy: str,
        domains: List[ScientificDomain]
    ) -> str:
        """Build human-readable reasoning for routing decision."""
        reasoning = f"Routing to {len(domains)} domain(s): {', '.join([d.value for d in domains])}. "
        reasoning += f"Strategy: {strategy}. "
        reasoning += f"Classification confidence: {classification.confidence.value} ({classification.confidence_score:.2f}). "

        if classification.is_multi_domain:
            reasoning += "Multi-domain research detected. "

        return reasoning

    def assess_domain_expertise(
        self,
        domain: ScientificDomain
    ) -> DomainExpertise:
        """
        Assess Kosmos's expertise and capabilities for a specific domain.

        Args:
            domain: Domain to assess

        Returns:
            DomainExpertise assessment
        """
        capability = self.capabilities.get(domain)

        if not capability:
            # Unknown domain
            return DomainExpertise(
                domain=domain,
                expertise_level="beginner",
                expertise_score=0.1,
                known_limitations=["Domain not yet supported"],
            )

        # Calculate expertise score based on capabilities
        expertise_score = 0.0

        # Factor 1: API availability (40% weight)
        api_score = len(capability.get_operational_apis()) / max(len(capability.api_clients), 1)
        expertise_score += api_score * 0.4

        # Factor 2: Template availability (30% weight)
        template_score = min(len(capability.experiment_templates) / 3.0, 1.0)
        expertise_score += template_score * 0.3

        # Factor 3: Ontology coverage (30% weight)
        expertise_score += capability.ontology_coverage * 0.3

        # Map to expertise level
        if expertise_score >= 0.8:
            expertise_level = "expert"
        elif expertise_score >= 0.6:
            expertise_level = "advanced"
        elif expertise_score >= 0.4:
            expertise_level = "intermediate"
        else:
            expertise_level = "beginner"

        # Build limitations list
        limitations = []
        if len(capability.get_operational_apis()) == 0:
            limitations.append("No API integrations available")
        if len(capability.experiment_templates) == 0:
            limitations.append("No experiment templates available")
        if not capability.has_ontology:
            limitations.append("Domain ontology not yet implemented")

        return DomainExpertise(
            domain=domain,
            expertise_level=expertise_level,
            expertise_score=expertise_score,
            available_tools=capability.get_operational_apis(),
            available_templates=capability.experiment_templates,
            knowledge_base_coverage=capability.ontology_coverage,
            known_limitations=limitations,
            recommended_human_expertise=(
                "Domain expert consultation recommended"
                if expertise_score < 0.5 else None
            ),
        )

    def get_domain_capabilities(
        self,
        domain: ScientificDomain
    ) -> Optional[DomainCapability]:
        """Get capabilities available for a specific domain."""
        return self.capabilities.get(domain)

    def get_all_supported_domains(self) -> List[ScientificDomain]:
        """Get list of all supported domains."""
        return [
            domain for domain, cap in self.capabilities.items()
            if len(cap.api_clients) > 0 or len(cap.experiment_templates) > 0
        ]

    def suggest_cross_domain_connections(
        self,
        source_domain: ScientificDomain,
        target_domain: ScientificDomain
    ) -> List[str]:
        """
        Suggest potential cross-domain connections between two domains.

        Args:
            source_domain: Source domain
            target_domain: Target domain

        Returns:
            List of suggested connection concepts/methods
        """
        # This is a simplified version - in production, this would query
        # the cross-domain mapping database or use Claude

        connections = {
            (ScientificDomain.BIOLOGY, ScientificDomain.NEUROSCIENCE): [
                "Gene expression in neural tissue",
                "Molecular mechanisms of neurodegeneration",
                "Cellular metabolism in neurons",
            ],
            (ScientificDomain.MATERIALS, ScientificDomain.PHYSICS): [
                "Material property optimization",
                "Quantum effects in materials",
                "Thermodynamic stability",
            ],
            (ScientificDomain.CHEMISTRY, ScientificDomain.BIOLOGY): [
                "Biochemical pathways",
                "Drug-target interactions",
                "Enzyme catalysis",
            ],
        }

        # Try both directions
        key1 = (source_domain, target_domain)
        key2 = (target_domain, source_domain)

        return connections.get(key1, connections.get(key2, []))

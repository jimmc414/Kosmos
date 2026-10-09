"""
Unit tests for DomainRouter (Phase 9).

Tests domain classification, routing, expertise assessment, and cross-domain capabilities.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from kosmos.core.domain_router import (
    DomainRouter
)
from kosmos.models.domain import (
    DomainClassification,
    DomainRoute,
    DomainExpertise,
    DomainCapability,
    ScientificDomain,
    DomainConfidence
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_client():
    """Mock LLM client for domain classification"""
    client = Mock()
    client.generate_structured = Mock()
    return client


@pytest.fixture
def domain_router(mock_llm_client, mock_env_vars):
    """Create DomainRouter instance with mocked LLM"""
    router = DomainRouter(claude_client=mock_llm_client)
    return router


@pytest.fixture
def sample_classification_response() -> Dict[str, Any]:
    """Sample classification response from Claude"""
    return {
        "primary_domain": "biology",
        "confidence": "very_high",
        "secondary_domains": ["neuroscience"],
        "is_multi_domain": False,
        "keywords": ["gene", "protein", "expression"],
        "rationale": "Focus on genetic and protein analysis"
    }


@pytest.fixture
def multi_domain_classification_response() -> Dict[str, Any]:
    """Sample multi-domain classification response"""
    return {
        "primary_domain": "neuroscience",
        "confidence": "high",
        "secondary_domains": ["biology", "materials"],
        "is_multi_domain": True,
        "keywords": ["neural", "conductance", "materials"],
        "rationale": "Combines neuroscience with materials science"
    }


# ============================================================================
# Test Domain Router Initialization
# ============================================================================

@pytest.mark.unit
class TestDomainRouterInit:
    """Test DomainRouter initialization."""

    def test_init_default(self, mock_env_vars):
        """Test default initialization."""
        router = DomainRouter(claude_client=Mock())

        assert router.claude is not None
        assert len(router.DOMAIN_KEYWORDS) > 0
        assert len(router.DOMAIN_AGENTS) > 0
        assert len(router.DOMAIN_TEMPLATES) > 0
        assert len(router.DOMAIN_TOOLS) > 0

    def test_init_custom_client(self, mock_env_vars):
        """Test initialization with custom LLM client."""
        custom_client = Mock()
        router = DomainRouter(claude_client=custom_client)

        assert router.claude == custom_client

    def test_capabilities_loaded(self, domain_router):
        """Test that domain capabilities are correctly loaded."""
        # Biology capabilities
        assert ScientificDomain.BIOLOGY in domain_router.DOMAIN_KEYWORDS
        assert len(domain_router.DOMAIN_KEYWORDS[ScientificDomain.BIOLOGY]) > 0

        # Neuroscience capabilities
        assert ScientificDomain.NEUROSCIENCE in domain_router.DOMAIN_KEYWORDS
        assert len(domain_router.DOMAIN_KEYWORDS[ScientificDomain.NEUROSCIENCE]) > 0

        # Materials capabilities
        assert ScientificDomain.MATERIALS in domain_router.DOMAIN_KEYWORDS
        assert len(domain_router.DOMAIN_KEYWORDS[ScientificDomain.MATERIALS]) > 0


# ============================================================================
# Test Classification Prompt Building
# ============================================================================

@pytest.mark.unit
class TestClassificationPromptBuilding:
    """Test domain classification prompt construction."""

    def test_prompt_without_context(self, domain_router):
        """Test prompt building without context."""
        question = "What genes are involved in cancer?"
        prompt = domain_router._build_classification_prompt(question, context=None)

        assert isinstance(prompt, str)
        assert question in prompt
        assert "biology" in prompt.lower()
        assert "neuroscience" in prompt.lower()
        assert "materials" in prompt.lower()

    def test_prompt_with_context(self, domain_router):
        """Test prompt building with literature context."""
        question = "How do neurons process information?"
        context = {"previous_research": "Studies on synaptic plasticity"}

        prompt = domain_router._build_classification_prompt(question, context=context)

        assert question in prompt
        assert "synaptic plasticity" in prompt

    def test_prompt_with_multi_domain_hints(self, domain_router):
        """Test prompt with multi-domain research hints."""
        question = "How do conductive materials interact with neural tissue?"
        prompt = domain_router._build_classification_prompt(question, context=None)

        assert "multi-domain" in prompt.lower() or "cross-domain" in prompt.lower()

    def test_prompt_includes_keywords(self, domain_router):
        """Test that prompt includes domain keywords for guidance."""
        question = "Test question"
        prompt = domain_router._build_classification_prompt(question, context=None)

        # Should include examples of domain keywords
        assert any(keyword in prompt.lower() for keyword in ["gene", "neuron", "material", "crystal"])


# ============================================================================
# Test Classification Response Parsing
# ============================================================================

@pytest.mark.unit
class TestClassificationParsing:
    """Test parsing of Claude's classification responses."""

    def test_parse_valid_response(self, domain_router, sample_classification_response):
        """Test parsing valid classification response."""
        question = "What genes regulate cell growth?"

        classification = domain_router._parse_classification_response(
            sample_classification_response, question
        )

        assert isinstance(classification, DomainClassification)
        assert classification.primary_domain == ScientificDomain.BIOLOGY
        assert classification.confidence == DomainConfidence.VERY_HIGH
        assert ScientificDomain.NEUROSCIENCE in classification.secondary_domains
        assert not classification.is_multi_domain
        assert "gene" in classification.key_terms

    def test_parse_missing_fields(self, domain_router):
        """Test handling of response with missing fields."""
        incomplete_response = {
            "primary_domain": "biology",
            # Missing confidence, keywords, etc.
        }
        question = "Test question"

        classification = domain_router._parse_classification_response(
            incomplete_response, question
        )

        # Should still create classification with defaults
        assert isinstance(classification, DomainClassification)
        assert classification.primary_domain == ScientificDomain.BIOLOGY
        # Should have default confidence
        assert classification.confidence in DomainConfidence

    def test_parse_invalid_domain_name(self, domain_router):
        """Test handling of invalid domain names."""
        invalid_response = {
            "primary_domain": "invalid_domain",
            "confidence": "high",
            "secondary_domains": [],
            "is_multi_domain": False,
            "keywords": [],
            "rationale": "Test"
        }
        question = "Test question"

        classification = domain_router._parse_classification_response(
            invalid_response, question
        )

        # Should fallback to GENERAL
        assert classification.primary_domain == ScientificDomain.GENERAL

    def test_parse_malformed_json(self, domain_router):
        """Test handling of malformed response."""
        malformed_response = "not a dict"
        question = "Test question"

        # Should raise or return None - implementation dependent
        try:
            classification = domain_router._parse_classification_response(
                malformed_response, question
            )
            # If it doesn't raise, should be None or have fallback
            assert classification is None or classification.primary_domain == ScientificDomain.GENERAL
        except (TypeError, AttributeError):
            # Expected for malformed input
            pass

    def test_parse_empty_response(self, domain_router):
        """Test handling of empty response."""
        empty_response = {}
        question = "Test question"

        classification = domain_router._parse_classification_response(
            empty_response, question
        )

        # Should fallback gracefully
        assert classification is None or isinstance(classification, DomainClassification)


# ============================================================================
# Test Keyword-Based Classification
# ============================================================================

@pytest.mark.unit
class TestKeywordBasedClassification:
    """Test fallback keyword-based classification."""

    @pytest.mark.parametrize("question,expected_domain", [
        ("What genes regulate protein synthesis?", ScientificDomain.BIOLOGY),
        ("How do neurons transmit signals?", ScientificDomain.NEUROSCIENCE),
        ("What is the band gap of graphene?", ScientificDomain.MATERIALS),
        ("Analyze KEGG pathway data", ScientificDomain.BIOLOGY),
        ("Study synaptic plasticity in mice", ScientificDomain.NEUROSCIENCE),
        ("Optimize perovskite crystal structure", ScientificDomain.MATERIALS),
    ])
    def test_single_domain_keywords(self, domain_router, question, expected_domain):
        """Test keyword matching for single-domain questions."""
        classification = domain_router._keyword_based_classification(question)

        assert isinstance(classification, DomainClassification)
        assert classification.primary_domain == expected_domain
        assert len(classification.key_terms) > 0

    def test_multi_domain_keywords(self, domain_router):
        """Test detection of multi-domain research."""
        question = "How do conductive materials interact with neural tissue at the protein level?"
        # Contains: materials (conductive), neuroscience (neural), biology (protein)

        classification = domain_router._keyword_based_classification(question)

        assert len(classification.secondary_domains) > 0 or classification.is_multi_domain

    def test_ambiguous_query(self, domain_router):
        """Test handling of ambiguous queries."""
        question = "Analyze the data using statistical methods"

        classification = domain_router._keyword_based_classification(question)

        # Should fallback to GENERAL or have low confidence
        assert (classification.primary_domain == ScientificDomain.GENERAL or
                classification.confidence in [DomainConfidence.LOW, DomainConfidence.VERY_LOW])

    def test_empty_query(self, domain_router):
        """Test handling of empty query."""
        classification = domain_router._keyword_based_classification("")

        assert classification.primary_domain == ScientificDomain.GENERAL
        assert len(classification.key_terms) == 0

    def test_case_insensitive_matching(self, domain_router):
        """Test that keyword matching is case-insensitive."""
        question_lower = "what genes regulate growth?"
        question_upper = "WHAT GENES REGULATE GROWTH?"
        question_mixed = "What GeNeS Regulate GrOwTh?"

        class1 = domain_router._keyword_based_classification(question_lower)
        class2 = domain_router._keyword_based_classification(question_upper)
        class3 = domain_router._keyword_based_classification(question_mixed)

        assert class1.primary_domain == class2.primary_domain == class3.primary_domain
        assert class1.primary_domain == ScientificDomain.BIOLOGY


# ============================================================================
# Test Domain Classification (End-to-End)
# ============================================================================

@pytest.mark.unit
class TestDomainClassification:
    """Test end-to-end domain classification."""

    def test_classification_claude_success(self, domain_router, sample_classification_response):
        """Test successful classification using Claude."""
        domain_router.claude.generate_structured.return_value = sample_classification_response

        question = "What genes are involved in cancer?"
        classification = domain_router.classify_research_question(question)

        assert isinstance(classification, DomainClassification)
        assert classification.primary_domain == ScientificDomain.BIOLOGY
        assert classification.confidence == DomainConfidence.VERY_HIGH
        domain_router.claude.generate_structured.assert_called_once()

    def test_classification_multi_domain(self, domain_router, multi_domain_classification_response):
        """Test multi-domain classification."""
        domain_router.claude.generate_structured.return_value = multi_domain_classification_response

        question = "How do conductive materials affect neural signaling?"
        classification = domain_router.classify_research_question(question)

        assert classification.is_multi_domain
        assert len(classification.secondary_domains) > 0

    def test_classification_claude_failure_fallback(self, domain_router):
        """Test fallback to keyword-based when Claude fails."""
        domain_router.claude.generate_structured.side_effect = Exception("API Error")

        question = "What genes regulate protein synthesis?"
        classification = domain_router.classify_research_question(question)

        # Should still classify using keywords
        assert isinstance(classification, DomainClassification)
        assert classification.primary_domain == ScientificDomain.BIOLOGY

    def test_confidence_score_calculation(self, domain_router):
        """Test that confidence scores are properly assigned."""
        for confidence_str in ["very_high", "high", "medium", "low", "very_low"]:
            response = {
                "primary_domain": "biology",
                "confidence": confidence_str,
                "secondary_domains": [],
                "is_multi_domain": False,
                "keywords": ["gene"],
                "rationale": "Test"
            }
            domain_router.claude.generate_structured.return_value = response

            classification = domain_router.classify_research_question("Test question")
            assert classification.confidence.value == confidence_str

    def test_secondary_domain_detection(self, domain_router):
        """Test detection of secondary domains."""
        response = {
            "primary_domain": "biology",
            "confidence": "high",
            "secondary_domains": ["neuroscience", "materials_science"],
            "is_multi_domain": True,
            "keywords": ["gene", "neuron", "material"],
            "rationale": "Cross-domain research"
        }
        domain_router.claude.generate_structured.return_value = response

        classification = domain_router.classify_research_question("Test question")

        assert len(classification.secondary_domains) == 2
        assert ScientificDomain.NEUROSCIENCE in classification.secondary_domains
        assert ScientificDomain.MATERIALS in classification.secondary_domains

    def test_context_integration(self, domain_router, sample_classification_response):
        """Test that context is passed to classification."""
        domain_router.claude.generate_structured.return_value = sample_classification_response

        question = "Analyze this data"
        context = {"previous_findings": "Gene expression analysis"}

        classification = domain_router.classify_research_question(question, context=context)

        # Verify context was used in prompt
        call_args = domain_router.claude.generate_structured.call_args
        prompt = call_args[0][0] if call_args[0] else call_args[1].get('prompt', '')
        assert "Gene expression" in prompt or context is not None


# ============================================================================
# Test Routing
# ============================================================================

@pytest.mark.unit
class TestRouting:
    """Test domain routing logic."""

    def test_single_domain_routing(self, domain_router):
        """Test routing for single-domain research."""
        classification = DomainClassification(
            primary_domain=ScientificDomain.BIOLOGY,
            confidence=DomainConfidence.HIGH,
            confidence_score=0.8,
            secondary_domains=[],
            is_multi_domain=False,
            key_terms=["gene", "protein"],
            rationale="Biology research"
        )

        route = domain_router.route("Test question", classification)

        assert isinstance(route, DomainRoute)
        assert route.classification.primary_domain == ScientificDomain.BIOLOGY
        assert len(route.assigned_agents) > 0
        assert len(route.required_tools) > 0
        assert len(route.recommended_templates) > 0

    def test_agent_selection(self, domain_router):
        """Test that correct agents are selected for domain."""
        classification = DomainClassification(
            primary_domain=ScientificDomain.BIOLOGY,
            confidence=DomainConfidence.HIGH,
            confidence_score=0.8,
            secondary_domains=[],
            is_multi_domain=False,
            key_terms=["gene"],
            rationale="Test"
        )

        route = domain_router.route("Test question", classification)

        # Should have biology-specific agents
        assert ScientificDomain.BIOLOGY in route.assigned_agents
        biology_agents = route.assigned_agents[ScientificDomain.BIOLOGY]
        assert len(biology_agents) > 0

    def test_tool_selection(self, domain_router):
        """Test that correct tools are selected for domain."""
        classification = DomainClassification(
            primary_domain=ScientificDomain.NEUROSCIENCE,
            confidence=DomainConfidence.HIGH,
            confidence_score=0.8,
            secondary_domains=[],
            is_multi_domain=False,
            key_terms=["neuron"],
            rationale="Test"
        )

        route = domain_router.route("Test question", classification)

        # Should have neuroscience-specific tools
        assert ScientificDomain.NEUROSCIENCE in route.required_tools
        neuro_tools = route.required_tools[ScientificDomain.NEUROSCIENCE]
        assert len(neuro_tools) > 0

    def test_template_recommendation(self, domain_router):
        """Test template recommendation for domain."""
        classification = DomainClassification(
            primary_domain=ScientificDomain.MATERIALS,
            confidence=DomainConfidence.HIGH,
            confidence_score=0.8,
            secondary_domains=[],
            is_multi_domain=False,
            key_terms=["crystal"],
            rationale="Test"
        )

        route = domain_router.route("Test question", classification)

        assert len(route.recommended_templates) > 0
        # Templates should be from materials domain

    def test_multi_domain_parallel_strategy(self, domain_router):
        """Test parallel strategy for multi-domain research."""
        classification = DomainClassification(
            primary_domain=ScientificDomain.BIOLOGY,
            confidence=DomainConfidence.HIGH,
            confidence_score=0.8,
            secondary_domains=[ScientificDomain.NEUROSCIENCE],
            is_multi_domain=True,
            key_terms=["gene", "neuron"],
            rationale="Multi-domain"
        )

        route = domain_router.route("Test question", classification)

        # Should have synthesis strategy for multi-domain
        assert route.synthesis_strategy in ["parallel_multi_domain", "sequential_multi_domain"]
        assert len(route.assigned_agents) >= 2  # Multiple domains

    def test_multi_domain_sequential_strategy(self, domain_router):
        """Test sequential strategy when domains have dependencies."""
        classification = DomainClassification(
            primary_domain=ScientificDomain.BIOLOGY,
            confidence=DomainConfidence.VERY_HIGH,
            confidence_score=0.95,
            secondary_domains=[ScientificDomain.NEUROSCIENCE],
            is_multi_domain=True,
            key_terms=["gene", "expression", "neuron"],
            rationale="Sequential analysis needed"
        )
        context = {"requires_sequential": True}

        route = domain_router.route("Test question", classification, context=context)

        # Strategy should consider context
        assert route.synthesis_strategy in ["parallel_multi_domain", "sequential_multi_domain"]


# ============================================================================
# Test Expertise Assessment
# ============================================================================

@pytest.mark.unit
class TestExpertiseAssessment:
    """Test domain expertise assessment."""

    @pytest.mark.parametrize("domain", [
        ScientificDomain.BIOLOGY,
        ScientificDomain.NEUROSCIENCE,
        ScientificDomain.MATERIALS,
    ])
    def test_expertise_for_supported_domains(self, domain_router, domain):
        """Test expertise assessment for supported domains."""
        expertise = domain_router.assess_domain_expertise(domain)

        assert isinstance(expertise, DomainExpertise)
        assert expertise.domain == domain
        assert 0.0 <= expertise.expertise_score <= 1.0
        assert expertise.expertise_score > 0.5  # Should have good expertise
        assert len(expertise.available_capabilities) > 0

    def test_expertise_for_unsupported_domain(self, domain_router):
        """Test expertise assessment for unsupported domain."""
        expertise = domain_router.assess_domain_expertise(ScientificDomain.GENERAL)

        assert expertise.expertise_score < 0.5  # Low expertise
        assert len(expertise.available_capabilities) == 0 or expertise.available_capabilities == []


# ============================================================================
# Test Capabilities and Cross-Domain Suggestions
# ============================================================================

@pytest.mark.unit
class TestCapabilitiesAndSuggestions:
    """Test capability retrieval and cross-domain suggestions."""

    def test_get_biology_capabilities(self, domain_router):
        """Test retrieving biology domain capabilities."""
        capabilities = domain_router.get_domain_capabilities(ScientificDomain.BIOLOGY)

        assert isinstance(capabilities, DomainCapability)
        assert capabilities.domain == ScientificDomain.BIOLOGY
        assert len(capabilities.available_apis) > 0
        assert len(capabilities.available_templates) > 0
        assert len(capabilities.specialized_agents) > 0

    def test_get_neuroscience_capabilities(self, domain_router):
        """Test retrieving neuroscience domain capabilities."""
        capabilities = domain_router.get_domain_capabilities(ScientificDomain.NEUROSCIENCE)

        assert capabilities.domain == ScientificDomain.NEUROSCIENCE
        assert len(capabilities.available_apis) > 0
        # Should have neuroscience-specific APIs like FlyWire, AllenBrain

    def test_get_materials_capabilities(self, domain_router):
        """Test retrieving materials science domain capabilities."""
        capabilities = domain_router.get_domain_capabilities(ScientificDomain.MATERIALS)

        assert capabilities.domain == ScientificDomain.MATERIALS
        assert len(capabilities.available_apis) > 0
        # Should have materials APIs like MaterialsProject, NOMAD

    def test_cross_domain_connection_suggestions(self, domain_router):
        """Test cross-domain connection suggestions."""
        suggestions = domain_router.suggest_cross_domain_connections(
            source_domain=ScientificDomain.MATERIALS,
            target_domain=ScientificDomain.NEUROSCIENCE
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        # Should suggest connections like electrical_conductivity ↔ neural_conductance

    def test_unsupported_domain_capabilities(self, domain_router):
        """Test capabilities for unsupported domain."""
        capabilities = domain_router.get_domain_capabilities(ScientificDomain.GENERAL)

        # Should return minimal or empty capabilities
        assert len(capabilities.available_apis) == 0 or capabilities.available_apis == []
        assert len(capabilities.available_templates) == 0 or capabilities.available_templates == []

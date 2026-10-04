"""
Experiment Designer Agent.

Designs experimental protocols from hypotheses using templates and Claude,
with resource estimation and scientific rigor validation.
"""

import logging
import time
import uuid
import json
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime, timezone

from kosmos.agents.base import BaseAgent, AgentMessage, MessageType, AgentStatus
from kosmos.core.llm import get_client
from kosmos.core.prompts import EXPERIMENT_DESIGNER
from kosmos.utils.compat import model_to_dict
from kosmos.models.hypothesis import Hypothesis, ExperimentType
from kosmos.models.experiment import (
    ExperimentProtocol,
    ExperimentDesignRequest,
    ExperimentDesignResponse,
    ProtocolStep,
    Variable,
    VariableType,
    ControlGroup,
    ResourceRequirements,
    StatisticalTestSpec,
    StatisticalTest,
    ValidationCheck,
)
from kosmos.experiments.templates.base import (
    TemplateBase,
    get_template_registry,
    TemplateCustomizationParams,
)
from kosmos.db.models import (
    Hypothesis as DBHypothesis,
    Experiment as DBExperiment,
    ExperimentStatus,
)
from kosmos.db import get_session

if TYPE_CHECKING:
    from kosmos.execution.data_schema import DatasetSchema

logger = logging.getLogger(__name__)


class UnboundVariableError(ValueError):
    """An independent or dependent variable could not be bound to a dataset column."""

    def __init__(
        self,
        hypothesis_id: Optional[str],
        unbound_names: List[str],
        available_columns: List[str],
        reason: Optional[str] = None,
    ):
        self.hypothesis_id = hypothesis_id
        self.unbound_names = list(unbound_names)
        self.available_columns = list(available_columns)
        self.reason = reason or "no matching dataset column"
        super().__init__(
            f"Hypothesis {hypothesis_id}: variables {self.unbound_names} cannot be bound "
            f"({self.reason}); available columns: {self.available_columns}"
        )


class ExperimentDesignerAgent(BaseAgent):
    """
    Agent for designing experimental protocols.

    Capabilities:
    - Select appropriate experiment templates
    - Generate detailed experimental protocols
    - Customize protocols for specific hypotheses
    - Validate scientific rigor
    - Estimate resource requirements
    - Store protocols in database

    Example:
        ```python
        agent = ExperimentDesignerAgent(config={
            "require_control_group": True,
            "min_rigor_score": 0.7
        })
        agent.start()

        # Design experiment for hypothesis
        response = agent.design_experiment(
            hypothesis=hypothesis,
            preferred_type=ExperimentType.COMPUTATIONAL
        )

        print(f"Protocol: {response.protocol.name}")
        print(f"Rigor Score: {response.rigor_score}")
        print(f"Cost: ${response.estimated_cost_usd}")
        ```
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Experiment Designer Agent.

        Args:
            agent_id: Unique agent identifier
            agent_type: Agent type name
            config: Configuration dictionary
        """
        super().__init__(agent_id, agent_type or "ExperimentDesignerAgent", config)

        # Configuration
        self.require_control_group = self.config.get("require_control_group", True)
        self.require_power_analysis = self.config.get("require_power_analysis", True)
        self.min_rigor_score = self.config.get("min_rigor_score", 0.6)
        self.use_templates = self.config.get("use_templates", True)
        self.use_llm_enhancement = self.config.get("use_llm_enhancement", True)

        # Components
        self.llm_client = get_client()
        self.template_registry = get_template_registry()

        logger.info(f"Initialized ExperimentDesignerAgent {self.agent_id}")

    def execute(self, message):
        """
        Execute agent task from message.

        Args:
            message: AgentMessage with task details

        Returns:
            AgentMessage: Response message with results
        """
        self.status = AgentStatus.WORKING

        try:
            task_type = message.content.get("task_type")

            if task_type == "design_experiment":
                hypothesis_id = message.content.get("hypothesis_id")
                hypothesis = message.content.get("hypothesis")
                preferred_type = message.content.get("preferred_experiment_type")
                max_cost = message.content.get("max_cost_usd")
                max_duration = message.content.get("max_duration_days")

                response = self.design_experiment(
                    hypothesis=hypothesis,
                    hypothesis_id=hypothesis_id,
                    preferred_experiment_type=preferred_type,
                    max_cost_usd=max_cost,
                    max_duration_days=max_duration
                )

                return AgentMessage(
                    type=MessageType.RESPONSE,
                    from_agent=self.agent_id,
                    to_agent=message.from_agent,
                    content={"response": model_to_dict(response)},
                    correlation_id=message.correlation_id
                )

            else:
                raise ValueError(f"Unknown task type: {task_type}")

        except Exception as e:
            logger.error(f"Error executing task: {e}", exc_info=True)
            self.status = AgentStatus.ERROR
            return AgentMessage(
                type=MessageType.ERROR,
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                content={"error": str(e)},
                correlation_id=message.correlation_id
            )

        finally:
            self.status = AgentStatus.IDLE

    def design_experiment(
        self,
        hypothesis: Optional[Hypothesis] = None,
        hypothesis_id: Optional[str] = None,
        preferred_experiment_type: Optional[ExperimentType] = None,
        max_cost_usd: Optional[float] = None,
        max_duration_days: Optional[float] = None,
        store_in_db: bool = True,
        *,
        dataset_schema: Optional["DatasetSchema"] = None,
    ) -> ExperimentDesignResponse:
        """
        Design an experimental protocol for a hypothesis.

        Args:
            hypothesis: Hypothesis object (provide this OR hypothesis_id)
            hypothesis_id: Hypothesis ID to load from database
            preferred_experiment_type: Preferred experiment type
            max_cost_usd: Maximum cost constraint
            max_duration_days: Maximum duration constraint
            store_in_db: Whether to store protocol in database
            dataset_schema: Schema of the supplied dataset. When given, the
                protocol is designed by the LLM (templates are skipped) and every
                independent and dependent variable must bind to a real column.

        Returns:
            ExperimentDesignResponse with protocol and metadata

        Raises:
            ValueError: If hypothesis invalid or design fails
            UnboundVariableError: If dataset_schema is given and a variable has
                no matching column
        """
        start_time = time.time()

        logger.info(f"Designing experiment for hypothesis: {hypothesis_id or 'provided'}")

        # Step 1: Load hypothesis if ID provided
        if hypothesis is None and hypothesis_id:
            hypothesis = self._load_hypothesis(hypothesis_id)
        elif hypothesis is None:
            raise ValueError("Must provide either hypothesis object or hypothesis_id")

        # Step 2: Select experiment type
        experiment_type = self._select_experiment_type(hypothesis, preferred_experiment_type)
        logger.info(f"Selected experiment type: {experiment_type.value}")

        # Step 3: Select or generate protocol
        if dataset_schema is not None:
            # Templates use placeholder variables that name no real column
            protocol = self._generate_with_claude(
                hypothesis=hypothesis,
                experiment_type=experiment_type,
                max_cost_usd=max_cost_usd,
                max_duration_days=max_duration_days,
                dataset_schema=dataset_schema,
            )
        elif self.use_templates:
            protocol = self._generate_from_template(
                hypothesis=hypothesis,
                experiment_type=experiment_type,
                max_cost_usd=max_cost_usd,
                max_duration_days=max_duration_days
            )
        else:
            protocol = self._generate_with_claude(
                hypothesis=hypothesis,
                experiment_type=experiment_type,
                max_cost_usd=max_cost_usd,
                max_duration_days=max_duration_days
            )

        # Step 4: Enhance with LLM if enabled
        if self.use_llm_enhancement and self.use_templates and dataset_schema is None:
            protocol = self._enhance_protocol_with_llm(protocol, hypothesis)

        # Step 4b: Power analysis — calculate required sample size
        try:
            from kosmos.experiments.statistical_power import PowerAnalyzer
            power_analyzer = PowerAnalyzer()
            for test_spec in protocol.statistical_tests:
                expected_es = test_spec.expected_effect_size or 0.5
                test_type_str = test_spec.test_type.value if hasattr(test_spec.test_type, 'value') else str(test_spec.test_type)
                try:
                    if 't_test' in test_type_str:
                        required_n = power_analyzer.ttest_sample_size(
                            effect_size=expected_es, power=test_spec.required_power, alpha=test_spec.alpha
                        )
                    elif 'correlation' in test_type_str:
                        required_n = power_analyzer.correlation_sample_size(
                            effect_size=expected_es, power=test_spec.required_power, alpha=test_spec.alpha
                        )
                    elif 'anova' in test_type_str:
                        num_groups = len(test_spec.groups) if test_spec.groups else 3
                        required_n = power_analyzer.anova_sample_size(
                            effect_size=expected_es, num_groups=num_groups,
                            power=test_spec.required_power, alpha=test_spec.alpha
                        )
                    else:
                        required_n = None

                    if required_n and (not protocol.sample_size or required_n > protocol.sample_size):
                        protocol.sample_size = required_n
                        protocol.sample_size_rationale = (
                            f"Power analysis: n={required_n} for {test_type_str} "
                            f"(effect_size={expected_es}, power={test_spec.required_power}, alpha={test_spec.alpha})"
                        )
                        protocol.power_analysis_performed = True
                        logger.info(f"Power analysis: updated sample_size to {required_n}")
                except Exception as pa_err:
                    logger.warning(f"Power analysis failed for {test_type_str}: {pa_err}")
        except ImportError:
            logger.warning("PowerAnalyzer unavailable, skipping power analysis")

        # Step 5: Validate protocol
        validation_result = self._validate_protocol(protocol)

        # Step 6: Calculate metrics
        rigor_score = self._calculate_rigor_score(protocol, validation_result)
        completeness_score = self._calculate_completeness_score(protocol)
        feasibility = self._assess_feasibility(protocol, max_cost_usd, max_duration_days)

        # Step 7: Store in database if requested
        if store_in_db:
            self._store_protocol(protocol, hypothesis)

        # Step 8: Create response
        design_time = time.time() - start_time

        response = ExperimentDesignResponse(
            protocol=protocol,
            hypothesis_id=hypothesis.id or "",
            design_time_seconds=design_time,
            template_used=protocol.template_name,
            validation_passed=validation_result["passed"],
            validation_warnings=validation_result["warnings"],
            validation_errors=validation_result["errors"],
            rigor_score=rigor_score,
            completeness_score=completeness_score,
            estimated_cost_usd=protocol.resource_requirements.estimated_cost_usd,
            estimated_duration_days=protocol.resource_requirements.estimated_duration_days,
            feasibility_assessment=feasibility,
            recommendations=self._generate_recommendations(protocol, validation_result),
            warnings=validation_result["warnings"]
        )

        logger.info(
            f"Designed protocol '{protocol.name}' in {design_time:.2f}s "
            f"(rigor: {rigor_score:.2f}, feasibility: {feasibility})"
        )

        return response

    def design_experiments(
        self,
        hypotheses: List[Hypothesis],
        preferred_experiment_type: Optional[ExperimentType] = None,
        max_cost_usd: Optional[float] = None,
        max_duration_days: Optional[float] = None,
        store_in_db: bool = True
    ) -> List[ExperimentDesignResponse]:
        """
        Design experimental protocols for multiple hypotheses.

        This is a convenience wrapper around design_experiment() for batch
        processing of multiple hypotheses.

        Args:
            hypotheses: List of Hypothesis objects to design experiments for
            preferred_experiment_type: Preferred experiment type for all
            max_cost_usd: Maximum cost constraint per experiment
            max_duration_days: Maximum duration constraint per experiment
            store_in_db: Whether to store protocols in database

        Returns:
            List of ExperimentDesignResponse objects, one per hypothesis

        Example:
            ```python
            responses = agent.design_experiments(
                hypotheses=[h1, h2, h3],
                preferred_experiment_type=ExperimentType.COMPUTATIONAL
            )
            for resp in responses:
                print(f"Protocol: {resp.protocol.name}")
            ```
        """
        responses = []
        for hypothesis in hypotheses:
            try:
                response = self.design_experiment(
                    hypothesis=hypothesis,
                    preferred_experiment_type=preferred_experiment_type,
                    max_cost_usd=max_cost_usd,
                    max_duration_days=max_duration_days,
                    store_in_db=store_in_db
                )
                responses.append(response)
            except Exception as e:
                logger.warning(f"Failed to design experiment for hypothesis {hypothesis.id}: {e}")
                # Continue with remaining hypotheses

        logger.info(f"Designed {len(responses)} experiments for {len(hypotheses)} hypotheses")
        return responses

    def _load_hypothesis(self, hypothesis_id: str) -> Hypothesis:
        """Load hypothesis from database."""
        with get_session() as session:
            db_hypothesis = session.query(DBHypothesis).filter_by(id=hypothesis_id).first()
            if not db_hypothesis:
                raise ValueError(f"Hypothesis {hypothesis_id} not found in database")

            # Convert to Pydantic model
            # priority_score may not exist in DB schema; derive from available scores
            priority = getattr(db_hypothesis, 'priority_score', None)
            if priority is None:
                # Derive from testability + novelty (equally weighted)
                t = db_hypothesis.testability_score or 0.5
                n = db_hypothesis.novelty_score or 0.5
                priority = (t + n) / 2.0

            return Hypothesis(
                id=db_hypothesis.id,
                research_question=db_hypothesis.research_question,
                statement=db_hypothesis.statement,
                rationale=db_hypothesis.rationale,
                domain=db_hypothesis.domain or "general",
                testability_score=db_hypothesis.testability_score,
                novelty_score=db_hypothesis.novelty_score,
                priority_score=priority,
            )

    def _select_experiment_type(
        self,
        hypothesis: Hypothesis,
        preferred: Optional[ExperimentType]
    ) -> ExperimentType:
        """Select the most appropriate experiment type."""
        if preferred:
            return preferred

        # Use hypothesis suggestion if available
        if hypothesis.suggested_experiment_types:
            return hypothesis.suggested_experiment_types[0]

        # Default based on domain heuristics
        domain_defaults = {
            "machine_learning": ExperimentType.COMPUTATIONAL,
            "artificial_intelligence": ExperimentType.COMPUTATIONAL,
            "computer_science": ExperimentType.COMPUTATIONAL,
            "statistics": ExperimentType.DATA_ANALYSIS,
            "data_science": ExperimentType.DATA_ANALYSIS,
            "psychology": ExperimentType.DATA_ANALYSIS,
            "neuroscience": ExperimentType.DATA_ANALYSIS,
        }

        return domain_defaults.get(
            hypothesis.domain.lower(),
            ExperimentType.COMPUTATIONAL  # Default fallback
        )

    def _generate_from_template(
        self,
        hypothesis: Hypothesis,
        experiment_type: ExperimentType,
        max_cost_usd: Optional[float],
        max_duration_days: Optional[float]
    ) -> ExperimentProtocol:
        """Generate protocol from template."""
        # Find best template
        template = self.template_registry.find_best_template(hypothesis, experiment_type)

        if not template:
            logger.warning(f"No template found for {experiment_type.value}, falling back to LLM")
            return self._generate_with_claude(
                hypothesis, experiment_type, max_cost_usd, max_duration_days
            )

        logger.info(f"Using template: {template.metadata.name}")

        # Customize template
        params = TemplateCustomizationParams(
            hypothesis=hypothesis,
            max_cost_usd=max_cost_usd,
            max_duration_days=max_duration_days
        )

        protocol = template.generate_protocol(params)
        protocol.template_name = template.metadata.name
        protocol.template_version = template.metadata.version

        return protocol

    def _generate_with_claude(
        self,
        hypothesis: Hypothesis,
        experiment_type: ExperimentType,
        max_cost_usd: Optional[float],
        max_duration_days: Optional[float],
        dataset_schema: Optional["DatasetSchema"] = None,
    ) -> ExperimentProtocol:
        """Generate protocol using Claude LLM."""
        logger.info("Generating protocol with Claude")

        dataset_context = ""
        if dataset_schema is not None:
            dataset_context = (
                "\n" + dataset_schema.to_prompt_block() + "\n\n"
                "Every independent and dependent variable MUST include "
                "\"column\": \"<exact column name from the dataset>\"; do not invent columns.\n"
            )

        # Build prompt
        prompt = EXPERIMENT_DESIGNER.format(
            hypothesis_statement=hypothesis.statement,
            hypothesis_rationale=hypothesis.rationale,
            domain=hypothesis.domain,
            experiment_type=experiment_type.value,
            max_cost_usd=max_cost_usd or "unlimited",
            max_duration_days=max_duration_days or "flexible",
            research_question=hypothesis.research_question,
            dataset_context=dataset_context,
        )

        # Define expected JSON schema for structured output
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "objective": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_number": {"type": "integer"},
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "action": {"type": "string"},
                            "expected_duration_minutes": {"type": "number"},
                        },
                        "required": ["step_number", "title", "description", "action"]
                    }
                },
                "variables": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "description": {"type": "string"},
                            "column": {"type": "string"},
                        },
                    },
                },
                "control_groups": {"type": "array"},
                "statistical_tests": {"type": "array"},
                "sample_size": {"type": "integer"},
                "resource_estimates": {"type": "object"},
            },
            "required": ["name", "description", "objective", "steps"]
        }

        try:
            # Get structured output from Claude
            # Use higher max_tokens for experiment protocols which can be verbose
            response = self.llm_client.generate_structured(
                prompt=prompt,
                schema=schema,
                system=EXPERIMENT_DESIGNER.system_prompt,  # Fixed: was system_prompt
                max_tokens=8192  # Increased from default 4096 for detailed protocols
            )

            # Parse and validate protocol
            protocol_data = response if isinstance(response, dict) else json.loads(response)

            # Validate protocol data to prevent NoneType errors downstream
            if not protocol_data or not isinstance(protocol_data, dict):
                logger.error(f"Failed to generate valid protocol data: {response}")
                raise ValueError("LLM returned invalid protocol data")

            protocol = self._parse_claude_protocol(
                protocol_data, hypothesis, experiment_type, dataset_schema=dataset_schema
            )

            return protocol

        except UnboundVariableError:
            raise
        except Exception as e:
            logger.error(f"Error generating protocol with Claude: {e}")
            raise ValueError(f"Failed to generate protocol: {e}")

    def _parse_claude_protocol(
        self,
        data: Dict[str, Any],
        hypothesis: Hypothesis,
        experiment_type: ExperimentType,
        dataset_schema: Optional["DatasetSchema"] = None,
    ) -> ExperimentProtocol:
        """Parse Claude's response into ExperimentProtocol.

        With dataset_schema, each variable's "column" (or, failing that, its
        name) is resolved against the dataset columns and stored in
        Variable.column; an unbound independent or dependent variable raises
        UnboundVariableError.
        """
        # Parse steps
        steps = []
        for step_data in data.get("steps", []):
            steps.append(ProtocolStep(
                step_number=step_data.get("step_number", len(steps) + 1),
                title=step_data.get("title", "") or "Untitled Step",
                description=step_data.get("description", ""),
                action=step_data.get("action", ""),
                expected_duration_minutes=step_data.get("expected_duration_minutes"),
                requires_steps=step_data.get("requires_steps", []),
                expected_output=step_data.get("expected_output"),
                validation_check=step_data.get("validation_check"),
                code_template=step_data.get("code_template"),
                library_imports=step_data.get("library_imports", []),
            ))

        # Parse variables
        variables = {}
        for var_name, var_data in data.get("variables", {}).items():
            if isinstance(var_data, dict):
                # Coerce values to list — LLMs sometimes return description
                # strings like "Variable (from dataset)" instead of lists
                raw_values = var_data.get("values")
                if raw_values is None:
                    coerced_values = None
                elif isinstance(raw_values, list):
                    coerced_values = raw_values
                elif isinstance(raw_values, str):
                    coerced_values = None  # Description strings aren't usable as values
                else:
                    coerced_values = [raw_values]  # Single scalar value

                variables[var_name] = Variable(
                    name=var_name,
                    type=VariableType(var_data.get("type", "independent")),
                    description=var_data.get("description", f"Variable: {var_name}"),
                    values=coerced_values,
                    fixed_value=var_data.get("fixed_value"),
                    unit=var_data.get("unit"),
                    measurement_method=var_data.get("measurement_method"),
                    column=self._bind_column(var_data.get("column"), var_name, dataset_schema),
                )

        # Validate: generate minimal default steps if LLM returned empty list
        if not steps:
            logger.warning("LLM returned empty steps, generating defaults from hypothesis")
            steps = [
                ProtocolStep(
                    step_number=1, title="Load data",
                    description=f"Load and prepare data for {hypothesis.domain} analysis",
                    action="Load dataset",
                ),
                ProtocolStep(
                    step_number=2, title="Run analysis",
                    description=f"Execute {experiment_type.value} analysis",
                    action=f"Run {experiment_type.value} analysis",
                ),
                ProtocolStep(
                    step_number=3, title="Statistical tests",
                    description="Run statistical tests and compute effect sizes",
                    action="Compute statistics",
                ),
            ]

        # Validate: generate minimal default variables if LLM returned empty dict
        if not variables:
            logger.warning("LLM returned empty variables, generating defaults from hypothesis")
            variables = {
                "independent_var": Variable(
                    name="independent_var",
                    type=VariableType.INDEPENDENT,
                    description=f"Primary independent variable for {hypothesis.domain} analysis",
                ),
                "dependent_var": Variable(
                    name="dependent_var",
                    type=VariableType.DEPENDENT,
                    description=f"Primary dependent variable for {hypothesis.domain} analysis",
                ),
            }

        if dataset_schema is not None:
            self._check_bindings(variables, hypothesis, dataset_schema)

        # Parse control groups
        control_groups = []
        for cg_data in data.get("control_groups", []):
            if isinstance(cg_data, dict):
                control_groups.append(ControlGroup(
                    name=cg_data.get("name", "control"),
                    description=cg_data.get("description", "Control group"),
                    variables=cg_data.get("variables", {}),
                    rationale=cg_data.get("rationale", "Standard control group"),
                    sample_size=cg_data.get("sample_size"),
                ))

        # Parse statistical tests
        statistical_tests = []
        for test_data in data.get("statistical_tests", []):
            if isinstance(test_data, dict):
                test_type_str = test_data.get("test_type", "t_test")
                try:
                    test_type = StatisticalTest(test_type_str)
                except ValueError:
                    test_type = StatisticalTest.CUSTOM

                statistical_tests.append(StatisticalTestSpec(
                    test_type=test_type,
                    description=test_data.get("description", "Statistical test"),
                    null_hypothesis=test_data.get("null_hypothesis", "H0: No effect"),
                    alternative=test_data.get("alternative", "two-sided"),
                    alpha=test_data.get("alpha", 0.05),
                    variables=test_data.get("variables", []),
                    groups=test_data.get("groups"),
                    correction_method=test_data.get("correction_method"),
                    required_power=test_data.get("required_power", 0.8),
                    expected_effect_size=test_data.get("expected_effect_size"),
                ))

        # Parse resource estimates
        resource_data = data.get("resource_estimates", {})
        resource_requirements = ResourceRequirements(
            compute_hours=resource_data.get("compute_hours"),
            memory_gb=resource_data.get("memory_gb"),
            gpu_required=resource_data.get("gpu_required", False),
            gpu_memory_gb=resource_data.get("gpu_memory_gb"),
            estimated_cost_usd=resource_data.get("estimated_cost_usd"),
            api_calls_estimated=resource_data.get("api_calls_estimated"),
            estimated_duration_days=resource_data.get("estimated_duration_days"),
            required_data_sources=resource_data.get("required_data_sources", []),
            required_datasets=resource_data.get("required_datasets", []),
            data_size_gb=resource_data.get("data_size_gb"),
            required_libraries=resource_data.get("required_libraries", []),
            python_version=resource_data.get("python_version"),
            can_parallelize=resource_data.get("can_parallelize", False),
            parallelization_factor=resource_data.get("parallelization_factor"),
        )

        # Create protocol
        protocol = ExperimentProtocol(
            name=data.get("name", f"Experiment for: {hypothesis.statement[:50]}"),
            hypothesis_id=hypothesis.id or "",
            experiment_type=experiment_type,
            domain=hypothesis.domain,
            description=data.get("description", ""),
            objective=data.get("objective", f"Test hypothesis: {hypothesis.statement}"),
            steps=steps,
            variables=variables,
            control_groups=control_groups,
            statistical_tests=statistical_tests,
            sample_size=data.get("sample_size"),
            sample_size_rationale=data.get("sample_size_rationale"),
            power_analysis_performed=data.get("power_analysis_performed", False),
            resource_requirements=resource_requirements,
            validation_checks=[],
            random_seed=data.get("random_seed"),
            reproducibility_notes=data.get("reproducibility_notes"),
        )

        return protocol

    @staticmethod
    def _bind_column(
        proposed: Any,
        var_name: str,
        dataset_schema: Optional["DatasetSchema"],
    ) -> Optional[str]:
        """Resolve a variable's proposed column, then its name, against the dataset."""
        if dataset_schema is None:
            return None
        for candidate in (proposed, var_name):
            column = dataset_schema.resolve_column(candidate if isinstance(candidate, str) else None)
            if column is not None:
                return column
        return None

    @staticmethod
    def _check_bindings(
        variables: Dict[str, Variable],
        hypothesis: Hypothesis,
        dataset_schema: "DatasetSchema",
    ) -> None:
        """Raise UnboundVariableError unless every independent and dependent variable is bound."""
        used = [v for v in variables.values()
                if v.type in (VariableType.INDEPENDENT, VariableType.DEPENDENT)]
        unbound = [v.name for v in used if v.column is None]
        if unbound:
            raise UnboundVariableError(hypothesis.id, unbound, dataset_schema.columns)

        indep = [v for v in used if v.type == VariableType.INDEPENDENT]
        dep = [v for v in used if v.type == VariableType.DEPENDENT]
        if not indep or not dep:
            missing = "independent" if not indep else "dependent"
            raise UnboundVariableError(
                hypothesis.id, [v.name for v in used], dataset_schema.columns,
                reason=f"no {missing} variable",
            )
        if indep[0].column == dep[0].column:
            raise UnboundVariableError(
                hypothesis.id, [indep[0].name, dep[0].name], dataset_schema.columns,
                reason=f"independent and dependent both bound to column {indep[0].column!r}",
            )

    def _enhance_protocol_with_llm(
        self,
        protocol: ExperimentProtocol,
        hypothesis: Hypothesis
    ) -> ExperimentProtocol:
        """Enhance template-generated protocol with LLM insights."""
        logger.info("Enhancing protocol with LLM")

        # Ask Claude for enhancements
        prompt = f"""Review and enhance this experimental protocol.

Hypothesis: {hypothesis.statement}
Rationale: {hypothesis.rationale}
Domain: {hypothesis.domain}

Current Protocol:
- Name: {protocol.name}
- Steps: {len(protocol.steps)}
- Controls: {len(protocol.control_groups)}

Suggest specific improvements for:
1. Additional control groups if needed
2. Additional variables to track
3. Potential confounding factors
4. Improved validation checks

Return ONLY a JSON object with suggested enhancements (keep it concise).
"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=1000)
            # Parse and apply enhancements (simplified for now)
            # In production, would parse JSON and selectively apply
            logger.info("LLM enhancements applied")
        except Exception as e:
            logger.warning(f"Failed to enhance protocol with LLM: {e}")

        return protocol

    def _validate_protocol(self, protocol: ExperimentProtocol) -> Dict[str, Any]:
        """Validate protocol using ExperimentValidator with inline fallback."""
        try:
            from kosmos.experiments.validator import ExperimentValidator
            validator = ExperimentValidator(
                require_control_group=self.require_control_group,
                require_power_analysis=self.require_power_analysis,
                min_rigor_score=self.min_rigor_score,
            )
            report = validator.validate(protocol)
            return {
                "passed": report.validation_passed,
                "errors": [c.message for c in report.checks_performed
                           if c.status == "failed" and c.severity == "error"],
                "warnings": report.recommendations,
                "rigor_score": report.rigor_score,
            }
        except Exception as e:
            logger.warning("ExperimentValidator failed, using fallback: %s", e)
            return self._validate_protocol_inline(protocol)

    def _validate_protocol_inline(self, protocol: ExperimentProtocol) -> Dict[str, Any]:
        """Inline fallback validation for scientific rigor."""
        errors = []
        warnings = []

        # Check control groups
        if self.require_control_group and not protocol.has_control_group():
            errors.append("Protocol must include at least one control group")

        # Check sample size
        if protocol.sample_size and protocol.sample_size < 10:
            warnings.append(f"Small sample size ({protocol.sample_size}) may lack statistical power")

        # Check variables
        if not protocol.get_independent_variables():
            errors.append("Protocol must define at least one independent variable")

        if not protocol.get_dependent_variables():
            errors.append("Protocol must define at least one dependent variable")

        # Check statistical tests
        if not protocol.statistical_tests:
            warnings.append("No statistical tests defined - consider adding hypothesis tests")

        # Check steps
        if len(protocol.steps) < 3:
            warnings.append("Protocol has very few steps - ensure completeness")

        # Check resource estimates
        if not protocol.resource_requirements.estimated_duration_days:
            warnings.append("No duration estimate provided")

        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    def _calculate_rigor_score(
        self,
        protocol: ExperimentProtocol,
        validation: Dict[str, Any]
    ) -> float:
        """Calculate scientific rigor score (0.0-1.0)."""
        # Use ExperimentValidator's score if available
        if "rigor_score" in validation:
            return validation["rigor_score"]

        score = 1.0

        # Penalties for validation issues
        score -= len(validation["errors"]) * 0.3
        score -= len(validation["warnings"]) * 0.1

        # Bonuses for good practices
        if protocol.has_control_group():
            score += 0.1
        if protocol.power_analysis_performed:
            score += 0.1
        if protocol.random_seed is not None:
            score += 0.05
        if len(protocol.statistical_tests) >= 2:
            score += 0.05

        return max(0.0, min(1.0, score))

    def _calculate_completeness_score(self, protocol: ExperimentProtocol) -> float:
        """Calculate protocol completeness score (0.0-1.0)."""
        score = 0.0
        total_checks = 10

        # Check various aspects
        if len(protocol.steps) >= 5:
            score += 1
        if len(protocol.variables) >= 3:
            score += 1
        if protocol.has_control_group():
            score += 1
        if protocol.statistical_tests:
            score += 1
        if protocol.sample_size:
            score += 1
        if protocol.sample_size_rationale:
            score += 1
        if protocol.resource_requirements.estimated_cost_usd:
            score += 1
        if protocol.resource_requirements.estimated_duration_days:
            score += 1
        if protocol.random_seed is not None:
            score += 1
        if protocol.reproducibility_notes:
            score += 1

        return score / total_checks

    def _assess_feasibility(
        self,
        protocol: ExperimentProtocol,
        max_cost: Optional[float],
        max_duration: Optional[float]
    ) -> str:
        """Assess experiment feasibility: High/Medium/Low."""
        cost = protocol.resource_requirements.estimated_cost_usd or 0
        duration = protocol.resource_requirements.estimated_duration_days or 0

        # Check constraints
        if max_cost and cost > max_cost:
            return "Low"
        if max_duration and duration > max_duration:
            return "Low"

        # Assess based on absolute values
        if cost < 50 and duration < 7:
            return "High"
        elif cost < 200 and duration < 30:
            return "Medium"
        else:
            return "Low"

    def _generate_recommendations(
        self,
        protocol: ExperimentProtocol,
        validation: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for improving protocol."""
        recommendations = []

        if validation["errors"]:
            recommendations.append("Fix validation errors before proceeding")

        if not protocol.power_analysis_performed:
            recommendations.append("Perform power analysis to determine adequate sample size")

        if protocol.random_seed is None:
            recommendations.append("Set random seed for reproducibility")

        if not protocol.reproducibility_notes:
            recommendations.append("Add reproducibility notes (dependencies, environment, etc.)")

        if len(protocol.control_groups) == 0:
            recommendations.append("Consider adding control group(s) for comparison")

        return recommendations

    def _store_protocol(self, protocol: ExperimentProtocol, hypothesis: Hypothesis) -> None:
        """Store protocol in database."""
        try:
            with get_session() as session:
                # Create database experiment
                # Note: DB model doesn't have 'name' column, use description instead
                db_experiment = DBExperiment(
                    id=protocol.id or str(uuid.uuid4()),
                    hypothesis_id=protocol.hypothesis_id,
                    description=protocol.description,
                    experiment_type=protocol.experiment_type.value,
                    status=ExperimentStatus.CREATED.value,
                    protocol=protocol.to_dict(),
                    domain=protocol.domain,
                    created_at=datetime.now(timezone.utc),
                )

                session.add(db_experiment)
                session.commit()

                protocol.id = db_experiment.id
                logger.info(f"Stored protocol {protocol.id} in database")

        except Exception as e:
            # Don't fail if database isn't available - the protocol is still valid
            error_str = str(e)
            if "Database not initialized" in error_str:
                logger.warning(f"Database not initialized, skipping protocol storage")
                if not protocol.id:
                    protocol.id = str(uuid.uuid4())
            elif "invalid keyword argument" in error_str:
                # Schema mismatch - log warning but don't fail
                logger.warning(f"Database schema mismatch, skipping protocol storage: {e}")
                if not protocol.id:
                    protocol.id = str(uuid.uuid4())
            else:
                logger.error(f"Error storing protocol: {e}")
                raise

    def list_templates(
        self,
        experiment_type: Optional[ExperimentType] = None,
        domain: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available templates.

        Args:
            experiment_type: Filter by experiment type
            domain: Filter by domain

        Returns:
            List of template metadata dictionaries
        """
        if experiment_type:
            templates = self.template_registry.get_templates_by_type(experiment_type)
        elif domain:
            templates = self.template_registry.get_templates_by_domain(domain)
        else:
            templates = list(self.template_registry)

        return [model_to_dict(template.metadata) for template in templates]

"""
Prompt templates for different agent types and research tasks.

This module provides reusable, structured prompts for:
- Hypothesis generation
- Experimental design
- Data analysis
- Literature analysis
- Result interpretation
"""

from typing import Dict, List, Optional, Any
from string import Template


class PromptTemplate:
    """
    A template for generating prompts with variable substitution.

    Example:
        ```python
        template = PromptTemplate(
            name="hypothesis_generator",
            template="Generate a hypothesis about ${topic} in ${domain}",
            variables=["topic", "domain"]
        )
        prompt = template.render(topic="dark matter", domain="astrophysics")
        ```
    """

    def __init__(
        self,
        name: str,
        template: str,
        variables: List[str],
        system_prompt: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Initialize a prompt template.

        Args:
            name: Unique template name
            template: Template string with ${variable} placeholders
            variables: List of required variable names
            system_prompt: Optional system prompt
            description: Optional description of template purpose
        """
        self.name = name
        self.template_str = template
        self.variables = variables
        self.system_prompt = system_prompt
        self.description = description
        self._template = Template(template)

    def render(self, **kwargs) -> str:
        """
        Render the template with provided variables.

        Args:
            **kwargs: Variable values

        Returns:
            str: Rendered prompt

        Raises:
            KeyError: If required variable is missing
        """
        # Check all required variables are provided
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise KeyError(f"Missing required variables: {missing}")

        return self._template.safe_substitute(**kwargs)

    def format(self, **kwargs) -> str:
        """
        Alias for render() to match common string formatting convention.

        Args:
            **kwargs: Variable values

        Returns:
            str: Rendered prompt
        """
        return self.render(**kwargs)

    def get_full_prompt(self, **kwargs) -> Dict[str, str]:
        """
        Get both system and user prompts.

        Returns:
            dict: {"system": str, "prompt": str}
        """
        return {
            "system": self.system_prompt or "",
            "prompt": self.render(**kwargs)
        }


# ============================================================================
# HYPOTHESIS GENERATION TEMPLATES
# ============================================================================

HYPOTHESIS_GENERATOR = PromptTemplate(
    name="hypothesis_generator",
    system_prompt="""You are a scientific hypothesis generator powered by Claude. Your role is to:
1. Analyze the research question and existing literature
2. Generate novel, testable hypotheses
3. Provide clear scientific rationale for each hypothesis
4. Assess testability and feasibility
5. Suggest appropriate experiment types

Guidelines for Good Hypotheses:
- Make specific, falsifiable predictions
- Use clear, unambiguous language
- Ground rationale in scientific theory or existing evidence
- Focus on testable relationships (not just observations)
- Avoid vague qualifiers like "maybe", "might", "possibly"

Experiment Types:
- computational: Simulations, algorithms, mathematical proofs
- data_analysis: Statistical analysis of existing datasets
- literature_synthesis: Systematic review, meta-analysis

Output Format (JSON):
{
  "hypotheses": [
    {
      "statement": "Clear, specific hypothesis statement with concrete prediction",
      "rationale": "Scientific justification grounded in theory or evidence (50-200 words)",
      "confidence_score": 0.0-1.0,
      "testability_score": 0.0-1.0,
      "suggested_experiment_types": ["computational", "data_analysis", "literature_synthesis"]
    }
  ]
}

Example:
{
  "hypotheses": [
    {
      "statement": "Increasing the number of attention heads from 8 to 16 in transformer models will improve performance on long-sequence tasks by 15-25%",
      "rationale": "Attention mechanisms allow transformers to capture long-range dependencies. Prior work (Vaswani et al. 2017) showed that multiple attention heads enable the model to attend to different aspects simultaneously. Increasing heads should provide richer representations for long sequences, where capturing diverse contextual relationships is crucial. However, diminishing returns may occur beyond 16 heads due to redundancy.",
      "confidence_score": 0.75,
      "testability_score": 0.90,
      "suggested_experiment_types": ["computational", "data_analysis"]
    }
  ]
}""",
    template="""Research Question: ${research_question}

Domain: ${domain}

Number of Hypotheses Requested: ${num_hypotheses}

Literature Context:
${literature_context}

Task:
Generate ${num_hypotheses} diverse, testable hypotheses that address this research question.

For each hypothesis:
1. **Statement**: A clear, specific, falsifiable prediction (not a question)
   - Include concrete, measurable outcomes where possible
   - Make directional predictions (increases, decreases, causes, leads to)
   - Avoid vague language (maybe, might, possibly)

2. **Rationale**: Scientific justification (50-200 words)
   - Reference relevant theory, prior work, or mechanisms
   - Explain WHY you expect this relationship
   - Cite literature context if applicable
   - Acknowledge potential limitations

3. **Confidence Score** (0.0-1.0): Your confidence in the hypothesis based on:
   - Strength of theoretical foundation
   - Quality of supporting evidence
   - Clarity of predicted mechanism

4. **Testability Score** (0.0-1.0): How testable this hypothesis is
   - 0.8-1.0: Easily testable with available methods/data
   - 0.5-0.7: Testable but requires significant resources or setup
   - 0.0-0.4: Difficult to test or requires unavailable resources

5. **Suggested Experiment Types**: List 1-2 appropriate experiment types
   - computational: Use if hypothesis involves simulation, algorithmic analysis, mathematical proof
   - data_analysis: Use if hypothesis can be tested with existing datasets
   - literature_synthesis: Use if hypothesis requires systematic review of existing literature

Diversity:
Ensure hypotheses explore different aspects or mechanisms related to the research question. Don't generate near-duplicates.

Output the hypotheses as a JSON object with the exact structure specified in the system prompt.""",
    variables=["research_question", "domain", "num_hypotheses", "literature_context"],
    description="Generate scientific hypotheses from research questions with structured output"
)

# ============================================================================
# EXPERIMENTAL DESIGN TEMPLATES
# ============================================================================

EXPERIMENT_DESIGNER = PromptTemplate(
    name="experiment_designer",
    system_prompt="""You are an experimental design expert powered by Claude. Your role is to:
1. Convert hypotheses into detailed experimental protocols
2. Define experimental variables (independent, dependent, control)
3. Specify control groups and experimental conditions
4. Select appropriate statistical methods
5. Estimate resource requirements accurately
6. Ensure scientific rigor and reproducibility

Guidelines for Good Experimental Design:
- Define clear, measurable variables
- Include appropriate control groups
- Specify sample size with power analysis rationale
- Choose statistical tests that match data and hypothesis
- Make protocols reproducible with detailed steps
- Estimate resources realistically (time, compute, cost)

Variable Types:
- independent: Variables you manipulate (e.g., attention heads: [8, 12, 16])
- dependent: Variables you measure (e.g., model_accuracy, inference_time)
- control: Variables held constant (e.g., dataset, random_seed)
- confounding: Potential confounds to track

Statistical Tests:
- t_test: Compare two group means
- anova: Compare multiple group means
- correlation: Measure relationship strength
- regression: Model relationships
- chi_square: Test categorical associations
- mann_whitney, kruskal_wallis, wilcoxon: Non-parametric alternatives

Output Format (JSON):
{
  "name": "Experiment name",
  "description": "Comprehensive experiment description (50+ words)",
  "objective": "What this experiment aims to accomplish",
  "steps": [
    {
      "step_number": 1,
      "title": "Step title",
      "description": "What to do in this step",
      "action": "Concrete action or command",
      "expected_duration_minutes": 30,
      "requires_steps": [],
      "expected_output": "What this produces",
      "validation_check": "How to verify success",
      "library_imports": ["numpy", "pandas"]
    }
  ],
  "variables": {
    "variable_name": {
      "name": "variable_name",
      "type": "independent",
      "description": "Clear variable description",
      "values": [value1, value2, value3],
      "unit": "unit_name",
      "column": "exact_dataset_column_name (required when a dataset is provided)"
    }
  },
  "control_groups": [
    {
      "name": "baseline_control",
      "description": "What this control represents",
      "variables": {"var_name": "baseline_value"},
      "rationale": "Why this control is needed",
      "sample_size": 30
    }
  ],
  "statistical_tests": [
    {
      "test_type": "t_test",
      "description": "What we're testing",
      "null_hypothesis": "H0: No difference in means",
      "alternative": "two-sided",
      "alpha": 0.05,
      "variables": ["outcome_variable"],
      "groups": ["experimental", "control"],
      "required_power": 0.8,
      "expected_effect_size": 0.5
    }
  ],
  "sample_size": 90,
  "sample_size_rationale": "Power analysis: 30 per group for 80% power to detect medium effect (d=0.5) at alpha=0.05",
  "power_analysis_performed": true,
  "resource_estimates": {
    "compute_hours": 24.0,
    "memory_gb": 16,
    "gpu_required": false,
    "estimated_cost_usd": 50.0,
    "estimated_duration_days": 2.0,
    "required_libraries": ["numpy", "scipy", "pandas", "scikit-learn"],
    "python_version": "3.9+",
    "can_parallelize": true,
    "parallelization_factor": 4
  },
  "random_seed": 42,
  "reproducibility_notes": "Fix random seed, record library versions, use same dataset version"
}

Example (Machine Learning):
{
  "name": "Transformer Attention Head Count Experiment",
  "description": "Test whether increasing attention heads from 8 to 16 improves transformer performance on long-sequence tasks by training models with different head counts and comparing their performance on standardized benchmarks.",
  "objective": "Determine if increasing attention heads significantly improves long-sequence task performance",
  "steps": [
    {
      "step_number": 1,
      "title": "Data Preparation",
      "description": "Load and preprocess long-sequence benchmark dataset",
      "action": "Load WikiText-103 dataset, filter for sequences >512 tokens, split 80/10/10",
      "expected_duration_minutes": 45,
      "expected_output": "train.pkl, val.pkl, test.pkl with N=5000 sequences each",
      "validation_check": "Verify all sequences >512 tokens, no data leakage between splits",
      "library_imports": ["pandas", "datasets", "torch"]
    },
    {
      "step_number": 2,
      "title": "Model Training",
      "description": "Train transformer models with 8, 12, and 16 attention heads",
      "action": "For each head count: initialize model, train for 20 epochs with Adam optimizer (lr=1e-4), save best checkpoint",
      "expected_duration_minutes": 1200,
      "requires_steps": [1],
      "expected_output": "3 trained model checkpoints",
      "validation_check": "Training loss converged, validation perplexity stable",
      "library_imports": ["torch", "transformers"]
    },
    {
      "step_number": 3,
      "title": "Performance Evaluation",
      "description": "Evaluate each model on test set",
      "action": "For each model: compute accuracy, perplexity, inference time on test set",
      "expected_duration_minutes": 90,
      "requires_steps": [2],
      "expected_output": "results.csv with metrics per model",
      "validation_check": "All models evaluated on same test set",
      "library_imports": ["torch", "pandas"]
    },
    {
      "step_number": 4,
      "title": "Statistical Analysis",
      "description": "Compare performance across attention head counts",
      "action": "Run one-way ANOVA on accuracy scores, post-hoc pairwise t-tests with Bonferroni correction",
      "expected_duration_minutes": 15,
      "requires_steps": [3],
      "expected_output": "Statistical test results with p-values",
      "validation_check": "Check ANOVA assumptions (normality, homogeneity of variance)",
      "library_imports": ["scipy", "statsmodels"]
    }
  ],
  "variables": {
    "attention_heads": {
      "name": "attention_heads",
      "type": "independent",
      "description": "Number of attention heads in transformer model",
      "values": [8, 12, 16],
      "unit": "count"
    },
    "accuracy": {
      "name": "accuracy",
      "type": "dependent",
      "description": "Model accuracy on long-sequence task",
      "unit": "percentage",
      "measurement_method": "Exact match accuracy on test set"
    },
    "perplexity": {
      "name": "perplexity",
      "type": "dependent",
      "description": "Language model perplexity",
      "unit": "perplexity",
      "measurement_method": "Exp(cross-entropy loss) on test set"
    },
    "dataset": {
      "name": "dataset",
      "type": "control",
      "description": "Training dataset",
      "fixed_value": "WikiText-103",
      "unit": "dataset_name"
    },
    "random_seed": {
      "name": "random_seed",
      "type": "control",
      "description": "Random seed for reproducibility",
      "fixed_value": 42,
      "unit": "integer"
    }
  },
  "control_groups": [
    {
      "name": "baseline_8_heads",
      "description": "Standard 8-head transformer (Vaswani et al. 2017 baseline)",
      "variables": {"attention_heads": 8},
      "rationale": "Industry standard baseline for comparison",
      "sample_size": 30
    }
  ],
  "statistical_tests": [
    {
      "test_type": "anova",
      "description": "Compare mean accuracy across all attention head counts",
      "null_hypothesis": "H0: No difference in mean accuracy across 8, 12, 16 heads",
      "alternative": "two-sided",
      "alpha": 0.05,
      "variables": ["accuracy"],
      "groups": ["8_heads", "12_heads", "16_heads"],
      "correction_method": "bonferroni",
      "required_power": 0.8,
      "expected_effect_size": 0.5
    },
    {
      "test_type": "t_test",
      "description": "Compare 16 heads vs 8 heads (primary comparison)",
      "null_hypothesis": "H0: No difference between 16 and 8 heads",
      "alternative": "two-sided",
      "alpha": 0.05,
      "variables": ["accuracy"],
      "groups": ["16_heads", "8_heads"],
      "required_power": 0.8,
      "expected_effect_size": 0.5
    }
  ],
  "sample_size": 90,
  "sample_size_rationale": "30 replications per condition (8/12/16 heads) for 80% power to detect medium effect size (d=0.5) at alpha=0.05 using ANOVA",
  "power_analysis_performed": true,
  "resource_estimates": {
    "compute_hours": 80.0,
    "memory_gb": 32,
    "gpu_required": true,
    "gpu_memory_gb": 24,
    "estimated_cost_usd": 120.0,
    "estimated_duration_days": 3.5,
    "required_libraries": ["torch", "transformers", "datasets", "scipy", "statsmodels", "pandas", "numpy"],
    "python_version": "3.9+",
    "can_parallelize": true,
    "parallelization_factor": 3
  },
  "random_seed": 42,
  "reproducibility_notes": "Fix all random seeds (Python, NumPy, PyTorch), record exact library versions (requirements.txt), use deterministic algorithms where possible, save training hyperparameters"
}""",
    template="""Research Question: ${research_question}

Hypothesis Statement: ${hypothesis_statement}

Hypothesis Rationale: ${hypothesis_rationale}

Domain: ${domain}

Experiment Type: ${experiment_type}
${dataset_context}
Resource Constraints:
- Max Cost: ${max_cost_usd} USD
- Max Duration: ${max_duration_days} days

Task:
Design a detailed experimental protocol to test this hypothesis.

Requirements:
1. **Name**: A clear, descriptive experiment name
2. **Description**: Comprehensive description (50+ words) of what the experiment does
3. **Objective**: What this experiment aims to accomplish
4. **Steps**: Detailed protocol steps (at least 3-5 steps)
   - Each step should have: step_number, title, description, concrete action
   - Include expected duration, dependencies, validation checks
   - Add library imports needed for each step
5. **Variables**: Define all experimental variables with types
   - Independent variables: What you manipulate (with specific values to test)
   - Dependent variables: What you measure (with measurement methods)
   - Control variables: What you hold constant (with fixed values)
6. **Control Groups**: At least one control group with clear rationale
   - Specify what variables are set to baseline/control values
   - Justify why this control is scientifically necessary
7. **Statistical Tests**: Appropriate tests for your hypothesis
   - Match test type to data (parametric vs non-parametric)
   - Define null hypothesis, alpha level, expected effect size
   - Include power analysis parameters (required power, effect size)
8. **Sample Size**: Calculate with power analysis rationale
   - Justify sample size for adequate statistical power
   - Typical: 80% power, alpha=0.05, medium effect size
9. **Resource Estimates**: Realistic estimates
   - Compute hours, memory requirements, GPU if needed
   - Cost estimate, duration in days
   - Required libraries and Python version
   - Parallelization potential
10. **Reproducibility**: Random seed and reproducibility notes

Constraints:
- Stay within max cost (${max_cost_usd} USD) if specified
- Stay within max duration (${max_duration_days} days) if specified
- Make protocol detailed enough for autonomous execution
- Ensure all steps are concrete and actionable
- Use appropriate statistical methods for the hypothesis type

Output the experiment protocol as a JSON object with the exact structure specified in the system prompt.""",
    variables=["hypothesis_statement", "hypothesis_rationale", "domain", "experiment_type", "research_question", "max_cost_usd", "max_duration_days", "dataset_context"],
    description="Design detailed experimental protocols from hypotheses with full specifications"
)

# ============================================================================
# DATA ANALYSIS TEMPLATES
# ============================================================================

DATA_ANALYST = PromptTemplate(
    name="data_analyst",
    system_prompt="""You are a data analysis expert. Your role is to:
1. Interpret experimental results scientifically
2. Identify patterns, trends, and anomalies
3. Assess statistical significance
4. Connect results back to original hypothesis
5. Suggest follow-up analyses if needed

Be rigorous, objective, and transparent about limitations.""",
    template="""Original Hypothesis: ${hypothesis}

Experiment Performed: ${experiment_description}

Results:
${results_data}

Statistical Tests: ${statistical_tests}

Please analyze these results:
1. Summarize key findings
2. Assess statistical significance
3. Determine if hypothesis is supported, rejected, or inconclusive
4. Identify any patterns or unexpected results
5. Note limitations or confounding factors
6. Suggest follow-up experiments if needed

${analysis_constraints}""",
    variables=["hypothesis", "experiment_description", "results_data", "statistical_tests", "analysis_constraints"],
    description="Analyze and interpret experimental results"
)

# ============================================================================
# LITERATURE ANALYSIS TEMPLATES
# ============================================================================

LITERATURE_ANALYZER = PromptTemplate(
    name="literature_analyzer",
    system_prompt="""You are a scientific literature analyst. Your role is to:
1. Extract key findings from papers
2. Identify methodologies and approaches
3. Assess relevance to research question
4. Detect gaps in existing literature
5. Synthesize information across multiple papers

Be thorough, accurate, and cite sources appropriately.""",
    template="""Research Question: ${research_question}

Papers to Analyze:
${papers_list}

Please analyze this literature:
1. Summarize key findings from each paper
2. Extract relevant methodologies
3. Identify common themes and contradictions
4. Assess gaps in current knowledge
5. Determine novelty of our research question
6. Suggest promising directions

${specific_questions}""",
    variables=["research_question", "papers_list", "specific_questions"],
    description="Analyze scientific literature"
)

PAPER_SUMMARIZER = PromptTemplate(
    name="paper_summarizer",
    system_prompt="""You are an expert at summarizing scientific papers. Extract:
1. Main research question
2. Key methods used
3. Primary findings
4. Limitations
5. Relevance to given domain

Be concise but comprehensive.""",
    template="""Paper Title: ${title}

Abstract: ${abstract}

Domain Context: ${domain}

${full_text}

Provide a structured summary:
1. Research Question: What problem does this paper address?
2. Methods: What approaches/techniques were used?
3. Key Findings: What were the main results?
4. Limitations: What are the acknowledged limitations?
5. Relevance: How relevant is this to ${domain} research (0-1 score)?""",
    variables=["title", "abstract", "domain", "full_text"],
    description="Summarize scientific papers"
)

# ============================================================================
# RESEARCH DIRECTOR TEMPLATES
# ============================================================================

RESEARCH_DIRECTOR = PromptTemplate(
    name="research_director",
    system_prompt="""You are a research director orchestrating autonomous scientific discovery. Your role is to:
1. Assess current research progress
2. Decide next steps in the research cycle
3. Determine when to pivot vs. persist
4. Detect convergence or diminishing returns
5. Coordinate multiple research threads

Be strategic, adaptive, and evidence-based in your decisions.""",
    template="""Research Question: ${research_question}

Current Progress:
${progress_summary}

Recent Results:
${recent_results}

Available Actions:
${available_actions}

Decide the next action:
1. Review current state and progress toward answering the research question
2. Analyze recent results for insights
3. Select the most promising next action
4. Provide rationale for your choice
5. Set success criteria for this action

Output Format (JSON):
{
  "next_action": "action_name",
  "rationale": "Why this action",
  "expected_outcome": "What we hope to learn",
  "success_criteria": "How to evaluate success",
  "should_continue": true/false
}""",
    variables=["research_question", "progress_summary", "recent_results", "available_actions"],
    description="Orchestrate research workflow and decide next steps"
)

# ============================================================================
# CODE GENERATION TEMPLATES
# ============================================================================

CODE_GENERATOR = PromptTemplate(
    name="code_generator",
    system_prompt="""You are a scientific code generator. Your role is to:
1. Generate correct, efficient Python code
2. Use appropriate scientific libraries (numpy, scipy, pandas, sklearn)
3. Include error handling and validation
4. Add clear comments and docstrings
5. Follow best practices for reproducibility

Only generate code that is safe to execute.""",
    template="""Task: ${task_description}

Required Analysis:
${analysis_type}

Input Data Format:
${data_format}

Expected Output:
${expected_output}

Available Libraries: ${libraries}

Generate Python code that:
1. Loads and validates input data
2. Performs the required analysis
3. Handles errors gracefully
4. Returns results in specified format
5. Includes docstring explaining the code

Constraints:
${constraints}""",
    variables=["task_description", "analysis_type", "data_format", "expected_output", "libraries", "constraints"],
    description="Generate scientific analysis code"
)


# ============================================================================
# TEMPLATE REGISTRY
# ============================================================================

TEMPLATE_REGISTRY: Dict[str, PromptTemplate] = {
    "hypothesis_generator": HYPOTHESIS_GENERATOR,
    "experiment_designer": EXPERIMENT_DESIGNER,
    "data_analyst": DATA_ANALYST,
    "literature_analyzer": LITERATURE_ANALYZER,
    "paper_summarizer": PAPER_SUMMARIZER,
    "research_director": RESEARCH_DIRECTOR,
    "code_generator": CODE_GENERATOR,
}


def get_template(name: str) -> PromptTemplate:
    """
    Get a prompt template by name.

    Args:
        name: Template name

    Returns:
        PromptTemplate: The requested template

    Raises:
        KeyError: If template not found
    """
    if name not in TEMPLATE_REGISTRY:
        available = ", ".join(TEMPLATE_REGISTRY.keys())
        raise KeyError(f"Template '{name}' not found. Available: {available}")
    return TEMPLATE_REGISTRY[name]


def list_templates() -> List[str]:
    """
    List all available template names.

    Returns:
        List[str]: Template names
    """
    return list(TEMPLATE_REGISTRY.keys())

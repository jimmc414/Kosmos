"""
Code generation for experiment execution.

Generates executable Python code from experiment protocols using:
1. Template-based generation for common patterns (from kosmos-figures)
2. LLM-based generation for novel experiments
3. Hybrid approach combining both

Based on patterns from docs/integration-plan.md.
"""

import ast
from typing import Dict, List, Optional, Any, Callable
import logging
from pathlib import Path

from kosmos.models.experiment import ExperimentProtocol, ProtocolStep, ExperimentType
from kosmos.models.hypothesis import Hypothesis
from kosmos.core.llm import ClaudeClient
from kosmos.core.prompts import EXPERIMENT_DESIGNER

logger = logging.getLogger(__name__)


class CodeTemplate:
    """Base class for code generation templates."""

    def __init__(self, name: str, experiment_type: ExperimentType):
        """
        Initialize code template.

        Args:
            name: Template name
            experiment_type: Type of experiment this template handles
        """
        self.name = name
        self.experiment_type = experiment_type

    def matches(self, protocol: ExperimentProtocol) -> bool:
        """Check if this template matches the protocol."""
        return protocol.experiment_type == self.experiment_type

    def generate(self, protocol: ExperimentProtocol) -> str:
        """Generate code from protocol."""
        raise NotImplementedError


class TTestComparisonCodeTemplate(CodeTemplate):
    """
    Template for t-test comparison experiments.

    Pattern from: kosmos-figures Figure_2_hypothermia_nucleotide_salvage
    """

    def __init__(self):
        super().__init__("ttest_comparison", ExperimentType.DATA_ANALYSIS)

    def matches(self, protocol: ExperimentProtocol) -> bool:
        """Check if protocol needs t-test comparison."""
        if protocol.experiment_type != ExperimentType.DATA_ANALYSIS:
            return False

        # Check for t-test in statistical tests
        for test in protocol.statistical_tests:
            # Use .value to get the string value from the enum if it is an enum, else use str()
            test_type_str = test.test_type.value if hasattr(test.test_type, 'value') else str(test.test_type)
            if 't_test' in test_type_str.lower() or 't-test' in test_type_str.lower():
                return True

        return False

    def generate(self, protocol: ExperimentProtocol) -> str:
        """Generate t-test comparison code."""
        # Extract variable information
        indep_vars = [v for v in protocol.variables.values() if v.type.value == 'independent']
        dep_vars = [v for v in protocol.variables.values() if v.type.value == 'dependent']

        group_var = indep_vars[0].name if indep_vars else 'group'
        measure_var = dep_vars[0].name if dep_vars else 'measurement'

        # Get groups from control groups
        groups = []
        if protocol.control_groups:
            groups.append(protocol.control_groups[0].name)
        groups.append('experimental')  # Default experimental group

        code_lines = [
            "# T-Test Comparison Analysis",
            "# Generated from protocol template",
            "",
            "import pandas as pd",
            "import numpy as np",
            "from scipy import stats",
            "from kosmos.execution.data_analysis import DataAnalyzer",
            "",
            "# Load data",
            f"# Expected format: CSV with columns '{group_var}' and '{measure_var}'",
            "df = pd.read_csv('data.csv')",
            "",
            "# Clean data",
            "df = df.dropna()",
            "",
            "# Perform t-test comparison",
            "analyzer = DataAnalyzer()",
            f"result = analyzer.ttest_comparison(",
            f"    df, '{group_var}', '{measure_var}',",
            f"    groups=('{groups[1]}', '{groups[0]}'),",
            f"    log_transform={'True' if any('log' in s.action.lower() for s in protocol.steps) else 'False'}",
            ")",
            "",
            "# Print results",
            "print(f\"T-statistic: {{result['t_statistic']:.4f}}\")",
            "print(f\"P-value: {{result['p_value']:.6f}}\")",
            "print(f\"Significance: {{result['significance_label']}}\")",
            "print(f\"Mean difference: {{result['mean_difference']:.4f}}\")",
            "",
            "# Return results for collection",
            "results = result"
        ]

        return "\n".join(code_lines)


class CorrelationAnalysisCodeTemplate(CodeTemplate):
    """
    Template for correlation analysis experiments.

    Pattern from: kosmos-figures Figure_3_perovskite_solar_cell
    """

    def __init__(self):
        super().__init__("correlation_analysis", ExperimentType.DATA_ANALYSIS)

    def matches(self, protocol: ExperimentProtocol) -> bool:
        """Check if protocol needs correlation analysis."""
        if protocol.experiment_type != ExperimentType.DATA_ANALYSIS:
            return False

        # Check for correlation in statistical tests or protocol name
        for test in protocol.statistical_tests:
            # Use .value to get the string value from the enum if it is an enum, else use str()
            test_type_str = test.test_type.value if hasattr(test.test_type, 'value') else str(test.test_type)
            if 'correlation' in test_type_str.lower() or 'regression' in test_type_str.lower():
                return True

        return 'correlation' in protocol.name.lower()

    def generate(self, protocol: ExperimentProtocol) -> str:
        """Generate correlation analysis code."""
        # Get variables
        vars_list = list(protocol.variables.keys())
        x_var = vars_list[0] if len(vars_list) > 0 else 'x'
        y_var = vars_list[1] if len(vars_list) > 1 else 'y'

        # Determine correlation method
        method = 'pearson'
        for test in protocol.statistical_tests:
            # Use .value to get the string value from the enum if it is an enum, else use str()
            test_type_str = test.test_type.value if hasattr(test.test_type, 'value') else str(test.test_type)
            if 'spearman' in test_type_str.lower():
                method = 'spearman'
                break

        code_lines = [
            "# Correlation Analysis",
            "# Generated from protocol template",
            "",
            "import pandas as pd",
            "import numpy as np",
            "from scipy import stats",
            "from kosmos.execution.data_analysis import DataAnalyzer",
            "",
            "# Load data",
            f"# Expected format: CSV with columns '{x_var}' and '{y_var}'",
            "df = pd.read_csv('data.csv')",
            "",
            "# Clean data",
            "df = df.dropna()",
            "",
            "# Perform correlation analysis",
            "analyzer = DataAnalyzer()",
            f"result = analyzer.correlation_analysis(",
            f"    df, '{x_var}', '{y_var}',",
            f"    method='{method}'",
            ")",
            "",
            "# Print results",
            "print(f\"Correlation ({method}): {{result['correlation']:.4f}}\")",
            "print(f\"P-value: {{result['p_value']:.6f}}\")",
            "print(f\"R-squared: {{result['r_squared']:.4f}}\")",
            "print(f\"Significance: {{result['significance']}}\")",
            "print(f\"Regression equation: {{result['equation']}}\")",
            "",
            "# Return results",
            "results = result"
        ]

        return "\n".join(code_lines)


class LogLogScalingCodeTemplate(CodeTemplate):
    """
    Template for log-log scaling analysis.

    Pattern from: kosmos-figures Figure_4_neural_network
    """

    def __init__(self):
        super().__init__("log_log_scaling", ExperimentType.DATA_ANALYSIS)

    def matches(self, protocol: ExperimentProtocol) -> bool:
        """Check if protocol needs log-log scaling analysis."""
        # Check for keywords in name or description
        keywords = ['scaling', 'power law', 'log-log', 'power-law']

        text = f"{protocol.name} {protocol.description}".lower()

        return any(keyword in text for keyword in keywords)

    def generate(self, protocol: ExperimentProtocol) -> str:
        """Generate log-log scaling analysis code."""
        vars_list = list(protocol.variables.keys())
        x_var = vars_list[0] if len(vars_list) > 0 else 'x'
        y_var = vars_list[1] if len(vars_list) > 1 else 'y'

        code_lines = [
            "# Log-Log Scaling Analysis",
            "# Generated from protocol template",
            "",
            "import pandas as pd",
            "import numpy as np",
            "from scipy import stats",
            "from kosmos.execution.data_analysis import DataAnalyzer, DataCleaner",
            "",
            "# Load data",
            f"# Expected format: CSV with columns '{x_var}' and '{y_var}'",
            "df = pd.read_csv('data.csv')",
            "",
            "# Clean data - remove NaN and non-positive values (required for log-log)",
            "df = DataCleaner.filter_positive(df, ['" + x_var + "', '" + y_var + "'])",
            "",
            "# Perform log-log scaling analysis",
            "analyzer = DataAnalyzer()",
            f"result = analyzer.log_log_scaling_analysis(df, '{x_var}', '{y_var}')",
            "",
            "# Print results",
            "print(f\"Spearman correlation: {{result['spearman_rho']:.4f}}\")",
            "print(f\"P-value: {{result['p_value']:.6f}}\")",
            "print(f\"Power law equation: {{result['equation']}}\")",
            "print(f\"Exponent: {{result['power_law_exponent']:.4f}}\")",
            "print(f\"R-squared: {{result['r_squared']:.4f}}\")",
            "",
            "# Return results",
            "results = result"
        ]

        return "\n".join(code_lines)


class MLExperimentCodeTemplate(CodeTemplate):
    """Template for machine learning experiments."""

    def __init__(self):
        super().__init__("ml_experiment", ExperimentType.COMPUTATIONAL)

    def matches(self, protocol: ExperimentProtocol) -> bool:
        """Check if protocol is ML experiment."""
        keywords = ['machine learning', 'classification', 'regression', 'cross-validation', 'model']

        text = f"{protocol.name} {protocol.description}".lower()

        return any(keyword in text for keyword in keywords)

    def generate(self, protocol: ExperimentProtocol) -> str:
        """Generate ML experiment code."""
        code_lines = [
            "# Machine Learning Experiment",
            "# Generated from protocol template",
            "",
            "import pandas as pd",
            "import numpy as np",
            "from sklearn.model_selection import train_test_split",
            "from sklearn.linear_model import LogisticRegression",
            "from kosmos.execution.ml_experiments import MLAnalyzer",
            "",
            "# Load data",
            "df = pd.read_csv('data.csv')",
            "",
            "# Prepare features and target",
            "# Assuming last column is target",
            "X = df.iloc[:, :-1]",
            "y = df.iloc[:, -1]",
            "",
            "# Initialize ML analyzer",
            "analyzer = MLAnalyzer(random_state=42)",
            "",
            "# Run complete experiment with cross-validation",
            "model = LogisticRegression(max_iter=1000)",
            "results = analyzer.run_experiment(",
            "    model, X, y,",
            "    test_size=0.2,",
            "    cv=5,",
            "    task_type='classification',",
            "    scale_features=True",
            ")",
            "",
            "# Print results",
            "print(f\"Test Accuracy: {{results['train_test_results']['accuracy']:.4f}}\")",
            "print(f\"CV Mean Score: {{results['cv_results']['mean_score']:.4f}}\")",
            "print(f\"F1 Score: {{results['train_test_results']['f1_score']:.4f}}\")",
            "",
            "# Return results",
            "results = results"
        ]

        return "\n".join(code_lines)


class ExperimentCodeGenerator:
    """
    Generates executable Python code from experiment protocols.

    Uses hybrid approach:
    1. Template matching for common patterns
    2. LLM generation for novel experiments
    3. Optional LLM enhancement of templates
    """

    def __init__(
        self,
        use_templates: bool = True,
        use_llm: bool = True,
        llm_enhance_templates: bool = False,
        llm_client: Optional[ClaudeClient] = None
    ):
        """
        Initialize code generator.

        Args:
            use_templates: If True, try template matching first
            use_llm: If True, use LLM for novel cases or fallback
            llm_enhance_templates: If True, enhance template code with LLM
            llm_client: Optional Claude client (created if not provided)
        """
        self.use_templates = use_templates
        self.use_llm = use_llm
        self.llm_enhance_templates = llm_enhance_templates

        # Initialize LLM client with error handling
        if use_llm and llm_client is None:
            try:
                self.llm_client = ClaudeClient()
            except (ValueError, Exception) as e:
                logger.warning(f"Failed to initialize ClaudeClient: {e}. LLM generation will be disabled.")
                self.llm_client = None
                self.use_llm = False
        else:
            self.llm_client = llm_client if use_llm else None

        # Initialize templates
        self.templates: List[CodeTemplate] = []
        if use_templates:
            self._register_templates()

    def _register_templates(self):
        """Register all available code templates."""
        self.templates = [
            TTestComparisonCodeTemplate(),
            CorrelationAnalysisCodeTemplate(),
            LogLogScalingCodeTemplate(),
            MLExperimentCodeTemplate()
        ]

        logger.info(f"Registered {len(self.templates)} code templates")

    def generate(self, protocol: ExperimentProtocol) -> str:
        """
        Generate code from protocol using hybrid approach.

        Args:
            protocol: Experiment protocol

        Returns:
            Generated Python code as string
        """
        code = None

        # Step 1: Try template matching
        if self.use_templates:
            template = self._match_template(protocol)
            if template:
                logger.info(f"Using template: {template.name}")
                code = template.generate(protocol)

                # Optionally enhance with LLM
                if self.llm_enhance_templates and self.llm_client:
                    code = self._enhance_with_llm(code, protocol)

        # Step 2: Fall back to LLM generation
        if code is None and self.use_llm:
            logger.info("No template matched, using LLM generation")
            code = self._generate_with_llm(protocol)

        # Step 3: Fallback to basic template
        if code is None:
            logger.warning("No code generated, using basic template")
            code = self._generate_basic_template(protocol)

        # Validate syntax
        self._validate_syntax(code)

        return code

    def _match_template(self, protocol: ExperimentProtocol) -> Optional[CodeTemplate]:
        """Find best matching template for protocol."""
        for template in self.templates:
            if template.matches(protocol):
                return template
        return None

    def _generate_with_llm(self, protocol: ExperimentProtocol) -> str:
        """Generate code using Claude LLM."""
        prompt = self._create_code_generation_prompt(protocol)

        try:
            response = self.llm_client.generate(prompt)

            # Extract code from response (may be in code blocks)
            code = self._extract_code_from_response(response)

            return code

        except Exception as e:
            logger.error(f"LLM code generation failed: {e}")
            return None

    def _create_code_generation_prompt(self, protocol: ExperimentProtocol) -> str:
        """Create prompt for LLM code generation."""
        steps_text = "\n".join([
            f"{i+1}. {step.title}: {step.action}"
            for i, step in enumerate(protocol.steps)
        ])

        variables_text = "\n".join([
            f"- {name} ({var.type.value}): {var.description}"
            for name, var in protocol.variables.items()
        ])

        tests_text = "\n".join([
            f"- {test.test_type}: {test.description}"
            for test in protocol.statistical_tests
        ])

        prompt = f"""Generate executable Python code for this experiment:

**Experiment:** {protocol.name}
**Type:** {protocol.experiment_type.value}
**Description:** {protocol.description}

**Steps:**
{steps_text}

**Variables:**
{variables_text}

**Statistical Tests:**
{tests_text}

Generate complete, executable Python code that:
1. Loads data from 'data.csv'
2. Implements each protocol step
3. Performs the specified statistical tests
4. Returns results in a dictionary

Use these libraries: pandas, numpy, scipy.stats
Use kosmos.execution.data_analysis.DataAnalyzer for statistical tests
Include comments explaining each section

Return ONLY the Python code, no explanations."""

        return prompt

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Look for code blocks
        if "```python" in response:
            # Extract from python code block
            start = response.find("```python") + 9
            end = response.find("```", start)
            code = response[start:end].strip()
        elif "```" in response:
            # Extract from generic code block
            start = response.find("```") + 3
            end = response.find("```", start)
            code = response[start:end].strip()
        else:
            # Assume entire response is code
            code = response.strip()

        return code

    def _enhance_with_llm(self, template_code: str, protocol: ExperimentProtocol) -> str:
        """Enhance template code with LLM additions."""
        prompt = f"""Enhance this experiment code for better results:

**Protocol:** {protocol.name}
**Description:** {protocol.description}

**Current Code:**
```python
{template_code}
```

Enhance the code to:
1. Add any domain-specific preprocessing
2. Add robustness checks
3. Add additional relevant statistics
4. Keep the same structure

Return the enhanced Python code only."""

        try:
            response = self.llm_client.generate(prompt)
            enhanced_code = self._extract_code_from_response(response)
            return enhanced_code
        except Exception as e:
            logger.warning(f"LLM enhancement failed, using original template: {e}")
            return template_code

    def _generate_basic_template(self, protocol: ExperimentProtocol) -> str:
        """Generate basic fallback template."""
        code_lines = [
            "# Basic Experiment Template",
            "# Minimal fallback when no specific template matches",
            "",
            "import pandas as pd",
            "import numpy as np",
            "",
            "# Load data",
            "df = pd.read_csv('data.csv')",
            "",
            "# Process data",
            "df = df.dropna()",
            "",
            "print(f\"Loaded {len(df)} samples\")",
            "print(f\"Columns: {list(df.columns)}\")",
            "",
            "# Basic statistics",
            "print(df.describe())",
            "",
            "# Return data",
            "results = {'data': df.to_dict(), 'shape': df.shape}"
        ]

        return "\n".join(code_lines)

    @staticmethod
    def _validate_syntax(code: str) -> None:
        """Validate Python syntax of generated code."""
        try:
            ast.parse(code)
            logger.info("Code syntax validation passed")
        except SyntaxError as e:
            logger.error(f"Generated code has syntax error: {e}")
            raise ValueError(f"Invalid Python syntax in generated code: {e}")

    def save_code(self, code: str, file_path: str) -> None:
        """Save generated code to file."""
        with open(file_path, 'w') as f:
            f.write(code)
        logger.info(f"Saved generated code to {file_path}")

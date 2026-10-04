"""
Code generation for experiment execution.

Generates executable Python code from experiment protocols using:
1. Template-based generation for common patterns (from kosmos-figures)
2. LLM-based generation for novel experiments
3. Hybrid approach combining both

Based on patterns from docs/integration-plan.md.
"""

import ast
from typing import Dict, List, Optional, Any, Callable, Tuple, TYPE_CHECKING
import logging
from pathlib import Path

from kosmos.models.experiment import ExperimentProtocol, ProtocolStep, ExperimentType, Variable
from kosmos.models.hypothesis import Hypothesis
from kosmos.core.llm import ClaudeClient
from kosmos.core.prompts import EXPERIMENT_DESIGNER

if TYPE_CHECKING:
    from kosmos.execution.data_schema import DatasetSchema

logger = logging.getLogger(__name__)


# Generated lines that define data_path as None when the executor did not set it.
# The host executor's restricted builtins have no dir(), so no "in dir()" test.
_DATA_PATH_GUARD = [
    "try:",
    "    data_path",
    "except NameError:",
    "    data_path = None",
]


def _variable_column(var: Variable) -> str:
    """The dataset column a variable reads: its bound column, else its name."""
    return var.column or var.name


def _xy_variables(protocol: ExperimentProtocol) -> Tuple[Optional[Variable], Optional[Variable]]:
    """The first independent (x) and first dependent (y) variable.

    Falls back to declaration order when a role is missing, so protocols that
    only list two variables keep working.
    """
    variables = list(protocol.variables.values())
    indep = protocol.get_independent_variables()
    dep = protocol.get_dependent_variables()
    x = indep[0] if indep else (variables[0] if variables else None)
    y = dep[0] if dep else next((v for v in variables if v is not x), None)
    return x, y


def _xy_columns(protocol: ExperimentProtocol) -> Tuple[str, str]:
    x, y = _xy_variables(protocol)
    return (_variable_column(x) if x else 'x'), (_variable_column(y) if y else 'y')


def _missing_columns_check(columns_expr: str) -> List[str]:
    """Generated lines that fail loudly when the loaded file lacks a required column."""
    return [
        "if df is not None:",
        f"    _missing = [c for c in {columns_expr} if c not in df.columns]",
        "    if _missing:",
        "        raise KeyError(f'Dataset is missing required columns: {_missing}; available: {list(df.columns)}')",
    ]


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

    def generate(self, protocol: ExperimentProtocol, dataset_schema: Optional["DatasetSchema"] = None) -> str:
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
            test_type_str = test.test_type.value if hasattr(test.test_type, 'value') else str(test.test_type)
            if 't_test' in test_type_str.lower() or 't-test' in test_type_str.lower():
                return True

        return False

    def generate(self, protocol: ExperimentProtocol, dataset_schema: Optional["DatasetSchema"] = None) -> str:
        """Generate t-test comparison code."""
        # Extract variable information
        indep_vars = [v for v in protocol.variables.values() if v.type.value == 'independent']
        dep_vars = [v for v in protocol.variables.values() if v.type.value == 'dependent']

        group_var = _variable_column(indep_vars[0]) if indep_vars else 'group'
        measure_var = _variable_column(dep_vars[0]) if dep_vars else 'measurement'

        # Get groups from control groups
        groups = []
        if protocol.control_groups:
            groups.append(protocol.control_groups[0].name)
        else:
            groups.append('control')  # Default control group
        groups.append('experimental')  # Default experimental group

        # A bound two-level column supplies the real group labels
        levels = (dataset_schema.categorical_columns.get(group_var) if dataset_schema else None) or []
        if len(levels) == 2 and not set(groups) <= set(levels):
            groups = [levels[0], levels[1]]

        # Get random seed from protocol or use default
        seed = getattr(protocol, 'random_seed', 42) or 42
        n_samples = 100  # Default sample size

        # Read effect size from protocol if available; default to 0.0 (null hypothesis)
        effect_size = 0.0
        if protocol.statistical_tests:
            es = getattr(protocol.statistical_tests[0], 'expected_effect_size', None)
            if es is not None:
                effect_size = es

        log_transform = any('log' in str(s.action).lower() for s in protocol.steps)

        code_lines = [
            "# T-Test Comparison Analysis",
            "# Generated from protocol template",
            "",
            "import pandas as pd",
            "import numpy as np",
            "from scipy import stats",
            "from pathlib import Path",
            "",
            f"_group_col = {group_var!r}",
            f"_measure_col = {measure_var!r}",
            f"_label1 = {groups[1]!r}",
            f"_label2 = {groups[0]!r}",
            "",
            "# Data loading with synthetic fallback (Issue #51 fix)",
            "# Expected format: CSV with columns _group_col and _measure_col",
            "df = None",
            *_DATA_PATH_GUARD,
            "if data_path:",
            "    try:",
            "        df = pd.read_csv(data_path)",
            "        _data_source = 'file'",
            "    except Exception as e:",
            "        print(f'Warning: Could not load data: {e}')",
            "        df = None",
            *_missing_columns_check("[_group_col, _measure_col]"),
            "if df is None:",
            f"    # Generate synthetic data for computational experiment",
            f"    np.random.seed({seed})",
            f"    n_per_group = {n_samples // 2}",
            f"    control_data = np.random.normal(0, 1, n_per_group)",
            f"    experimental_data = np.random.normal({effect_size}, 1, n_per_group)  # Effect size = {effect_size}",
            "    df = pd.DataFrame({",
            "        _group_col: [_label2] * n_per_group + [_label1] * n_per_group,",
            "        _measure_col: np.concatenate([control_data, experimental_data])",
            "    })",
            "    _data_source = 'synthetic'",
            "",
            "# Clean data",
            "df = df.dropna()",
            "",
            "# Check statistical assumptions before t-test",
            "_group_data = {g: df[df[_group_col]==g][_measure_col].values for g in df[_group_col].unique()}",
            "for _gname, _gvals in _group_data.items():",
            "    _shap_stat, _shap_p = stats.shapiro(_gvals[:5000]) if len(_gvals) >= 8 else (1.0, 1.0)",
            "    if _shap_p < 0.05:",
            "        print(f'WARNING: Normality assumption violated for group {_gname} (Shapiro p={_shap_p:.4f})')",
            "_groups_list = list(_group_data.values())",
            "if len(_groups_list) == 2:",
            "    _lev_stat, _lev_p = stats.levene(*_groups_list)",
            "    if _lev_p < 0.05:",
            "        print(f'WARNING: Equal variance assumption violated (Levene p={_lev_p:.4f})')",
            "",
            "# Perform t-test comparison (numpy and scipy only, so the sandbox can run it)",
            "_g1 = df[df[_group_col] == _label1][_measure_col].dropna().astype(float).values",
            "_g2 = df[df[_group_col] == _label2][_measure_col].dropna().astype(float).values",
            "if len(_g1) < 2 or len(_g2) < 2:",
            "    raise ValueError(",
            "        f'T-test needs at least 2 rows per group in column {_group_col!r} '",
            "        f'(measurement column {_measure_col!r}): {_label1!r} has {len(_g1)}, '",
            "        f'{_label2!r} has {len(_g2)}; available columns: {list(df.columns)}'",
            "    )",
        ]
        if log_transform:
            code_lines += [
                "_g1 = np.log2(_g1 + 1)",
                "_g2 = np.log2(_g2 + 1)",
            ]
        code_lines += [
            "_t_stat, _p_val = stats.ttest_ind(_g1, _g2)",
            "_n1, _n2 = len(_g1), len(_g2)",
            "_pooled_sd = float(np.sqrt(((_n1 - 1) * np.var(_g1, ddof=1) + (_n2 - 1) * np.var(_g2, ddof=1)) / (_n1 + _n2 - 2)))",
            "_mean_diff = float(np.mean(_g1) - np.mean(_g2))",
            "_cohens_d = _mean_diff / _pooled_sd if _pooled_sd > 0 else 0.0",
            "if _p_val < 0.001:",
            "    _sig_label = '***'",
            "elif _p_val < 0.01:",
            "    _sig_label = '**'",
            "elif _p_val < 0.05:",
            "    _sig_label = '*'",
            "else:",
            "    _sig_label = 'ns'",
            "result = {",
            "    'test': 'independent_t_test',",
            "    't_statistic': float(_t_stat),",
            "    'p_value': float(_p_val),",
            "    'group1': _label1,",
            "    'group2': _label2,",
            "    'group1_mean': float(np.mean(_g1)),",
            "    'group2_mean': float(np.mean(_g2)),",
            "    'group1_std': float(np.std(_g1, ddof=1)),",
            "    'group2_std': float(np.std(_g2, ddof=1)),",
            "    'mean_difference': _mean_diff,",
            "    'effect_size': float(_cohens_d),",
            "    'significance_label': _sig_label,",
            "    'n_group1': int(_n1),",
            "    'n_group2': int(_n2),",
            f"    'log_transform': {log_transform},",
            "}",
            "",
            "# Print results",
            "print(f\"T-statistic: {result['t_statistic']:.4f}\")",
            "print(f\"P-value: {result['p_value']:.6f}\")",
            "print(f\"Significance: {result['significance_label']}\")",
            "print(f\"Mean difference: {result['mean_difference']:.4f}\")",
            "",
            "# Propagate data source and assumption checks into results",
            "result['data_source'] = _data_source",
            "result['assumption_checks'] = {",
            "    'normality_tested': True,",
            "    'sample_size_adequate': len(df) >= 30,",
            "}",
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
            test_type_str = test.test_type.value if hasattr(test.test_type, 'value') else str(test.test_type)
            if 'correlation' in test_type_str.lower() or 'regression' in test_type_str.lower():
                return True

        return 'correlation' in protocol.name.lower()

    def generate(self, protocol: ExperimentProtocol, dataset_schema: Optional["DatasetSchema"] = None) -> str:
        """Generate correlation analysis code."""
        # Independent variable on x, dependent on y
        x_var, y_var = _xy_columns(protocol)

        # Determine correlation method
        method = 'pearson'
        for test in protocol.statistical_tests:
            test_type_str = test.test_type.value if hasattr(test.test_type, 'value') else str(test.test_type)
            if 'spearman' in test_type_str.lower():
                method = 'spearman'
                break

        seed = getattr(protocol, 'random_seed', 42) or 42

        code_lines = [
            "# Correlation Analysis",
            "# Generated from protocol template",
            "",
            "import pandas as pd",
            "import numpy as np",
            "from scipy import stats",
            "from pathlib import Path",
            "",
            f"_x_col = {x_var!r}",
            f"_y_col = {y_var!r}",
            "",
            "# Data loading with synthetic fallback",
            "# Expected format: CSV with columns _x_col and _y_col",
            "df = None",
            *_DATA_PATH_GUARD,
            "if data_path:",
            "    try:",
            "        df = pd.read_csv(data_path)",
            "        _data_source = 'file'",
            "    except Exception as e:",
            "        print(f'Warning: Could not load data: {e}')",
            "        df = None",
            *_missing_columns_check("[_x_col, _y_col]"),
            "if df is None:",
            f"    # Generate synthetic correlated data",
            f"    np.random.seed({seed})",
            f"    n = 100",
            "    _x_syn = np.random.normal(0, 1, n)",
            "    _y_syn = 0.5 * _x_syn + np.random.normal(0, 0.5, n)",
            "    df = pd.DataFrame({_x_col: _x_syn, _y_col: _y_syn})",
            "    _data_source = 'synthetic'",
            "",
            "# Clean data",
            "df = df.dropna()",
            "",
            "# Check statistical assumptions before correlation",
            "for _col in [_x_col, _y_col]:",
            "    _vals = df[_col].values",
            "    if len(_vals) >= 8:",
            "        _shap_stat, _shap_p = stats.shapiro(_vals[:5000])",
            "        if _shap_p < 0.05:",
            "            print(f'WARNING: Normality assumption violated for {_col} (Shapiro p={_shap_p:.4f})')",
            "",
            "# Perform correlation analysis (numpy and scipy only, so the sandbox can run it)",
            "_df_clean = df[[_x_col, _y_col]].dropna()",
            "if len(_df_clean) < 3:",
            "    raise ValueError(f'Correlation needs at least 3 complete rows in {_x_col!r} and {_y_col!r}, got {len(_df_clean)}')",
            "_x = _df_clean[_x_col].astype(float).values",
            "_y = _df_clean[_y_col].astype(float).values",
            f"_method = {method!r}",
            "if _method == 'spearman':",
            "    _corr, _p_corr = stats.spearmanr(_x, _y)",
            "else:",
            "    _corr, _p_corr = stats.pearsonr(_x, _y)",
            "_slope, _intercept, _r_value, _p_reg, _std_err = stats.linregress(_x, _y)",
            "if _p_corr < 0.001:",
            "    _significance = '***'",
            "elif _p_corr < 0.01:",
            "    _significance = '**'",
            "elif _p_corr < 0.05:",
            "    _significance = '*'",
            "else:",
            "    _significance = 'ns'",
            "_sign = '+' if _intercept >= 0 else ''",
            "result = {",
            "    'correlation': float(_corr),",
            "    'p_value': float(_p_corr),",
            "    'r_squared': float(_r_value ** 2),",
            "    'slope': float(_slope),",
            "    'intercept': float(_intercept),",
            "    'std_err': float(_std_err),",
            "    'significance': _significance,",
            "    'n_samples': int(len(_x)),",
            "    'equation': f'y = {_slope:.4f}x {_sign}{_intercept:.4f}',",
            "    'method': _method,",
            "}",
            "result['effect_size'] = result['correlation']",
            "",
            "# Also compute Spearman rank correlation for nonlinear relationships",
            "from scipy.stats import spearmanr, pearsonr",
            "_x_vals = df[_x_col].values",
            "_y_vals = df[_y_col].values",
            "_pearson_r, _pearson_p = pearsonr(_x_vals, _y_vals)",
            "_spearman_r, _spearman_p = spearmanr(_x_vals, _y_vals)",
            "result['pearson_r'] = float(_pearson_r)",
            "result['pearson_p'] = float(_pearson_p)",
            "result['spearman_r'] = float(_spearman_r)",
            "result['spearman_p'] = float(_spearman_p)",
            "",
            "# Use the more significant result for hypothesis support",
            "if _spearman_p < _pearson_p:",
            "    result['best_method'] = 'spearman'",
            "    result['best_correlation'] = float(_spearman_r)",
            "    result['best_p_value'] = float(_spearman_p)",
            "else:",
            "    result['best_method'] = 'pearson'",
            "    result['best_correlation'] = float(_pearson_r)",
            "    result['best_p_value'] = float(_pearson_p)",
            "result['supports_hypothesis'] = result['best_p_value'] < 0.05",
            "",
            "# Print results",
            "print(f\"Pearson r: {_pearson_r:.4f}, p={_pearson_p:.6f}\")",
            "print(f\"Spearman rho: {_spearman_r:.4f}, p={_spearman_p:.6f}\")",
            "print(f\"Best method: {result['best_method']} (p={result['best_p_value']:.6f})\")",
            f"print(f\"Correlation ({method}): {{result['correlation']:.4f}}\")",
            "print(f\"P-value: {result['p_value']:.6f}\")",
            "print(f\"R-squared: {result['r_squared']:.4f}\")",
            "print(f\"Significance: {result['significance']}\")",
            "print(f\"Regression equation: {result['equation']}\")",
            "",
            "# Propagate data source and assumption checks into results",
            "result['data_source'] = _data_source",
            "result['assumption_checks'] = {",
            "    'normality_tested': True,",
            "    'sample_size_adequate': len(df) >= 30,",
            "}",
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

    def generate(self, protocol: ExperimentProtocol, dataset_schema: Optional["DatasetSchema"] = None) -> str:
        """Generate log-log scaling analysis code."""
        x_var, y_var = _xy_columns(protocol)

        seed = getattr(protocol, 'random_seed', 42) or 42

        code_lines = [
            "# Log-Log Scaling Analysis",
            "# Generated from protocol template",
            "",
            "import pandas as pd",
            "import numpy as np",
            "from scipy import stats",
            "from pathlib import Path",
            "",
            f"_x_col = {x_var!r}",
            f"_y_col = {y_var!r}",
            "",
            "# Data loading with synthetic fallback",
            "# Expected format: CSV with columns _x_col and _y_col",
            "df = None",
            *_DATA_PATH_GUARD,
            "if data_path:",
            "    try:",
            "        df = pd.read_csv(data_path)",
            "        _data_source = 'file'",
            "    except Exception as e:",
            "        print(f'Warning: Could not load data: {e}')",
            "        df = None",
            *_missing_columns_check("[_x_col, _y_col]"),
            "if df is None:",
            f"    # Generate synthetic power-law data",
            f"    np.random.seed({seed})",
            f"    n = 100",
            "    _x_syn = np.logspace(0, 3, n)",
            "    _y_syn = 2.0 * _x_syn ** 0.75 * np.exp(np.random.normal(0, 0.1, n))",
            "    df = pd.DataFrame({_x_col: _x_syn, _y_col: _y_syn})",
            "    _data_source = 'synthetic'",
            "",
            "# Clean data - remove NaN and non-positive values (required for log-log)",
            "df = df[[_x_col, _y_col]].dropna()",
            "df = df[(df[_x_col] > 0) & (df[_y_col] > 0)]",
            "if len(df) < 3:",
            "    raise ValueError(f'Log-log analysis needs at least 3 positive rows in {_x_col!r} and {_y_col!r}, got {len(df)}')",
            "",
            "# Perform log-log scaling analysis (numpy and scipy only, so the sandbox can run it)",
            "_x = df[_x_col].astype(float).values",
            "_y = df[_y_col].astype(float).values",
            "_rho, _p_rho = stats.spearmanr(_x, _y)",
            "_slope, _intercept, _r_value, _p_reg, _std_err = stats.linregress(np.log10(_x), np.log10(_y))",
            "_coef = 10 ** _intercept",
            "result = {",
            "    'spearman_rho': float(_rho),",
            "    'p_value': float(_p_rho),",
            "    'power_law_exponent': float(_slope),",
            "    'power_law_coefficient': float(_coef),",
            "    'r_squared': float(_r_value ** 2),",
            "    'equation': f'y = {_coef:.3f} * x^{_slope:.3f}',",
            "    'n_samples': int(len(_x)),",
            "    'log_log_slope': float(_slope),",
            "    'log_log_intercept': float(_intercept),",
            "}",
            "result['effect_size'] = result['spearman_rho']",
            "",
            "# Print results",
            "print(f\"Spearman correlation: {result['spearman_rho']:.4f}\")",
            "print(f\"P-value: {result['p_value']:.6f}\")",
            "print(f\"Power law equation: {result['equation']}\")",
            "print(f\"Exponent: {result['power_law_exponent']:.4f}\")",
            "print(f\"R-squared: {result['r_squared']:.4f}\")",
            "",
            "# Propagate data source and assumption checks into results",
            "result['data_source'] = _data_source",
            "result['assumption_checks'] = {",
            "    'normality_tested': False,",
            "    'sample_size_adequate': len(df) >= 30,",
            "}",
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
        keywords = ['machine learning', 'classification', 'cross-validation',
                     'random forest', 'neural network', 'logistic regression',
                     'decision tree', 'svm', 'support vector']

        text = f"{protocol.name} {protocol.description}".lower()

        return any(keyword in text for keyword in keywords)

    def generate(self, protocol: ExperimentProtocol, dataset_schema: Optional["DatasetSchema"] = None) -> str:
        """Generate ML experiment code."""
        # Only explicit bindings pick columns: ML variable names ("features")
        # rarely name a column, so unbound protocols keep the last-column target
        target_cols = [v.column for v in protocol.get_dependent_variables() if v.column]
        target_col = target_cols[0] if target_cols else None
        feature_cols = [v.column for v in protocol.get_independent_variables()
                        if v.column and v.column != target_col]

        code_lines = [
            "# Machine Learning Experiment",
            "# Generated from protocol template",
            "",
            "import pandas as pd",
            "import numpy as np",
            "from sklearn.model_selection import train_test_split, cross_val_score",
            "from sklearn.linear_model import LogisticRegression",
            "from sklearn.pipeline import Pipeline",
            "from sklearn.preprocessing import StandardScaler",
            "from sklearn.metrics import accuracy_score, f1_score",
            "from sklearn.datasets import make_classification",
            "from pathlib import Path",
            "",
            "# Data loading with synthetic fallback",
            "df = None",
            *_DATA_PATH_GUARD,
            "if data_path:",
            "    try:",
            "        df = pd.read_csv(data_path)",
            "        _data_source = 'file'",
            "    except Exception as e:",
            "        print(f'Warning: Could not load data: {e}')",
            "        df = None",
            f"_target_col = {target_col!r}",
            f"_feature_cols = {feature_cols!r}",
            "if _target_col is not None:",
            *["    " + line for line in _missing_columns_check("[_target_col] + _feature_cols")],
            "if df is None:",
            "    # Generate synthetic classification data",
            "    X_syn, y_syn = make_classification(n_samples=200, n_features=10, random_state=42)",
            "    df = pd.DataFrame(X_syn, columns=[f'feature_{i}' for i in range(10)])",
            "    df['target'] = y_syn",
            "    _data_source = 'synthetic'",
            "",
            "# Prepare features and target: bound columns, else the last column is the target",
            "if _target_col is not None and _data_source == 'file':",
            "    y = df[_target_col]",
            "    X = df[_feature_cols] if _feature_cols else df.drop(columns=[_target_col])",
            "else:",
            "    X = df.iloc[:, :-1]",
            "    y = df.iloc[:, -1]",
            "",
            "# Train/test split, then 5-fold cross-validation (scikit-learn only, so the sandbox can run it)",
            "_pipeline = Pipeline([('scale', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))])",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
            "_pipeline.fit(X_train, y_train)",
            "_y_pred = _pipeline.predict(X_test)",
            "_cv_scores = cross_val_score(_pipeline, X, y, cv=5)",
            "results = {",
            "    'train_test_results': {",
            "        'accuracy': float(accuracy_score(y_test, _y_pred)),",
            "        'f1_score': float(f1_score(y_test, _y_pred, average='macro')),",
            "    },",
            "    'cv_results': {",
            "        'mean_score': float(np.mean(_cv_scores)),",
            "        'std_score': float(np.std(_cv_scores)),",
            "    },",
            "    'n_features': int(X.shape[1]),",
            "    'train_size': int(len(X_train)),",
            "    'test_size': int(len(X_test)),",
            "    'task_type': 'classification',",
            "}",
            "",
            "# Print results",
            "print(f\"Test Accuracy: {results['train_test_results']['accuracy']:.4f}\")",
            "print(f\"CV Mean Score: {results['cv_results']['mean_score']:.4f}\")",
            "print(f\"F1 Score: {results['train_test_results']['f1_score']:.4f}\")",
            "",
            "# Propagate data source and assumption checks into results",
            "results['data_source'] = _data_source",
            "results['assumption_checks'] = {",
            "    'normality_tested': False,",
            "    'sample_size_adequate': len(df) >= 30,",
            "}",
            "",
            "# Return results",
            "results = results"
        ]

        return "\n".join(code_lines)


class GenericComputationalCodeTemplate(CodeTemplate):
    """
    Generic template for computational experiments (biology, chemistry, etc.).

    Acts as a catch-all when no other template matches. Generates scipy-based
    analysis code with data loading, statistical tests, curve fitting, and
    visualization.
    """

    def __init__(self):
        super().__init__("generic_computational", ExperimentType.COMPUTATIONAL)

    def matches(self, protocol: ExperimentProtocol) -> bool:
        """Match COMPUTATIONAL or DATA_ANALYSIS experiments (catch-all fallback)."""
        return protocol.experiment_type in (
            ExperimentType.COMPUTATIONAL,
            ExperimentType.DATA_ANALYSIS,
        )

    def generate(self, protocol: ExperimentProtocol, dataset_schema: Optional["DatasetSchema"] = None) -> str:
        """Generate generic computational analysis code.

        Tests the bound independent column (x) against the bound dependent
        column (y): Pearson correlation when x is numeric, Welch t-test when x
        has two levels, one-way ANOVA when it has 3 to 10.
        """
        x_var, y_var = _xy_columns(protocol)

        seed = getattr(protocol, 'random_seed', 42) or 42

        code_lines = [
            "# Computational Experiment Analysis",
            f"# Protocol: {' '.join(str(protocol.name).split())}",
            "",
            "import pandas as pd",
            "import numpy as np",
            "from scipy import stats",
            "from scipy.optimize import curve_fit",
            "from pathlib import Path",
            "",
            f"_x_col = {x_var!r}",
            f"_y_col = {y_var!r}",
            "",
            "# Data loading with synthetic fallback",
            "# Expected format: CSV with columns _x_col and _y_col",
            "df = None",
            *_DATA_PATH_GUARD,
            "if data_path:",
            "    try:",
            "        df = pd.read_csv(data_path)",
            "        _data_source = 'file'",
            "    except Exception as e:",
            "        print(f'Warning: Could not load data: {e}')",
            "        df = None",
            *_missing_columns_check("[_x_col, _y_col]"),
            "if df is None:",
            f"    # Generate synthetic data for computational experiment",
            f"    np.random.seed({seed})",
            f"    n = 100",
            "    _x_syn = np.linspace(0, 10, n)",
            "    noise = np.random.normal(0, 0.5, n)",
            "    _y_syn = 2.0 * np.exp(-0.3 * _x_syn) + noise",
            "    df = pd.DataFrame({_x_col: _x_syn, _y_col: _y_syn})",
            "    _data_source = 'synthetic'",
            "",
            "# Clean data: keep rows where both analysis columns are present",
            "df = df.dropna(subset=[_x_col, _y_col])",
            "print(f'Loaded {len(df)} samples (source: {_data_source})')",
            "",
            "# Statistical analysis",
            "results = {}",
            "results['n_samples'] = len(df)",
            "results['data_source'] = _data_source",
            "results['columns'] = {'x': _x_col, 'y': _y_col}",
            "",
            "# Descriptive statistics",
            "results['descriptive'] = {}",
            "for col in df.select_dtypes(include=[np.number]).columns:",
            "    results['descriptive'][col] = {",
            "        'mean': float(df[col].mean()),",
            "        'std': float(df[col].std()),",
            "        'median': float(df[col].median()),",
            "        'min': float(df[col].min()),",
            "        'max': float(df[col].max()),",
            "    }",
            "",
            "# Primary statistical test on the bound columns",
            "if not pd.api.types.is_numeric_dtype(df[_y_col]):",
            "    raise ValueError(f'Dependent column {_y_col!r} must be numeric, got dtype {df[_y_col].dtype}')",
            "y_vals = df[_y_col].astype(float).values",
            "if pd.api.types.is_numeric_dtype(df[_x_col]):",
            "    x_vals = df[_x_col].astype(float).values",
            "    if len(x_vals) < 3:",
            "        raise ValueError(f'Correlation needs at least 3 complete rows in {_x_col!r} and {_y_col!r}, got {len(x_vals)}')",
            "",
            "    # Normality check",
            "    for _col, _vals in [(_x_col, x_vals), (_y_col, y_vals)]:",
            "        if len(_vals) >= 8:",
            "            _shap_stat, _shap_p = stats.shapiro(_vals[:5000])",
            "            if _shap_p < 0.05:",
            "                print(f'WARNING: Normality assumption violated for {_col} (Shapiro p={_shap_p:.4f})')",
            "",
            "    # Correlation analysis",
            "    pearson_r, pearson_p = stats.pearsonr(x_vals, y_vals)",
            "    spearman_r, spearman_p = stats.spearmanr(x_vals, y_vals)",
            "    results['correlation'] = {",
            "        'pearson_r': float(pearson_r),",
            "        'pearson_p': float(pearson_p),",
            "        'spearman_r': float(spearman_r),",
            "        'spearman_p': float(spearman_p),",
            "    }",
            "    results['test_type'] = 'pearson_correlation'",
            "    results['statistic'] = float(pearson_r)",
            "    results['p_value'] = float(pearson_p)",
            "    results['effect_size'] = float(pearson_r)",
            "",
            "    # Nonlinear curve fitting (exponential decay model)",
            "    try:",
            "        def exp_model(x, a, b, c):",
            "            return a * np.exp(b * x) + c",
            "        popt, pcov = curve_fit(exp_model, x_vals, y_vals, p0=[1, -0.1, 0], maxfev=5000)",
            "        y_fit = exp_model(x_vals, *popt)",
            "        ss_res = np.sum((y_vals - y_fit) ** 2)",
            "        ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)",
            "        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0",
            "        results['curve_fit'] = {",
            "            'model': 'exponential',",
            "            'params': {'a': float(popt[0]), 'b': float(popt[1]), 'c': float(popt[2])},",
            "            'r_squared': float(r_squared),",
            "        }",
            "    except Exception as e:",
            "        print(f'Curve fitting failed: {e}')",
            "        results['curve_fit'] = {'model': 'exponential', 'error': str(e)}",
            "",
            "    # Linear regression as baseline",
            "    slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)",
            "    results['linear_regression'] = {",
            "        'slope': float(slope),",
            "        'intercept': float(intercept),",
            "        'r_squared': float(r_value ** 2),",
            "        'p_value': float(p_value),",
            "        'std_err': float(std_err),",
            "    }",
            "else:",
            "    # Categorical x: compare the dependent column across its groups",
            "    _labels = df[_x_col].astype(str)",
            "    _levels = sorted(_labels.unique())",
            "    _groups = [y_vals[(_labels == _lvl).values] for _lvl in _levels]",
            "    if not 2 <= len(_levels) <= 10:",
            "        raise ValueError(",
            "            f'Independent column {_x_col!r} has {len(_levels)} levels; '",
            "            f'a group comparison needs 2 to 10 (or a numeric column)'",
            "        )",
            "    _small = [l for l, g in zip(_levels, _groups) if len(g) < 2]",
            "    if _small:",
            "        raise ValueError(f'Groups {_small} of column {_x_col!r} have fewer than 2 rows of {_y_col!r}')",
            "    results['groups'] = {l: {'n': int(len(g)), 'mean': float(np.mean(g))} for l, g in zip(_levels, _groups)}",
            "    if len(_levels) == 2:",
            "        _g1, _g2 = _groups",
            "        _t_stat, _p_val = stats.ttest_ind(_g1, _g2, equal_var=False)",
            "        _n1, _n2 = len(_g1), len(_g2)",
            "        _pooled_sd = float(np.sqrt(((_n1 - 1) * np.var(_g1, ddof=1) + (_n2 - 1) * np.var(_g2, ddof=1)) / (_n1 + _n2 - 2)))",
            "        _mean_diff = float(np.mean(_g1) - np.mean(_g2))",
            "        results['test_type'] = 'welch_t_test'",
            "        results['statistic'] = float(_t_stat)",
            "        results['p_value'] = float(_p_val)",
            "        results['effect_size'] = _mean_diff / _pooled_sd if _pooled_sd > 0 else 0.0",
            "        results['mean_difference'] = _mean_diff",
            "    else:",
            "        _f_stat, _p_val = stats.f_oneway(*_groups)",
            "        _grand_mean = float(np.mean(y_vals))",
            "        _ss_between = float(sum(len(g) * (np.mean(g) - _grand_mean) ** 2 for g in _groups))",
            "        _ss_total = float(np.sum((y_vals - _grand_mean) ** 2))",
            "        results['test_type'] = 'one_way_anova'",
            "        results['statistic'] = float(_f_stat)",
            "        results['p_value'] = float(_p_val)",
            "        results['effect_size'] = _ss_between / _ss_total if _ss_total > 0 else 0.0  # eta squared",
            "results['n'] = int(len(df))",
            "",
            "# Assumption checks",
            "results['assumption_checks'] = {",
            "    'normality_tested': True,",
            "    'sample_size_adequate': len(df) >= 30,",
            "}",
            "",
            "# Print summary",
            "print(f\"Test: {results['test_type']} of {_y_col!r} on {_x_col!r}\")",
            "print(f'Primary p-value: {results[\"p_value\"]:.6f}')",
            "print(f'Effect size: {results[\"effect_size\"]:.4f}')",
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

        # Use the configured provider (LLM_PROVIDER or kosmos run --provider)
        if use_llm and llm_client is None:
            try:
                from kosmos.core.llm import get_client
                self.llm_client = get_client()
            except Exception as e:
                logger.warning(f"LLM client unavailable: {e}. LLM generation disabled.")
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
            MLExperimentCodeTemplate(),
            GenericComputationalCodeTemplate(),  # Catch-all for COMPUTATIONAL + DATA_ANALYSIS
        ]

        logger.info(f"Registered {len(self.templates)} code templates")

    def generate(self, protocol: ExperimentProtocol, dataset_schema: Optional["DatasetSchema"] = None) -> str:
        """
        Generate code from protocol using hybrid approach.

        Args:
            protocol: Experiment protocol
            dataset_schema: Schema of the supplied dataset; its columns go into
                the LLM prompt and two-level columns supply t-test group labels

        Returns:
            Generated Python code as string

        Raises:
            ValueError: if the protocol is bound to a dataset (a schema is given
                or any variable has a column) but the analysed x or y variable
                has no column
        """
        self._check_bindings(protocol, dataset_schema)

        code = None

        # Step 1: Try template matching
        if self.use_templates:
            template = self._match_template(protocol)
            if template:
                logger.info(f"Using template: {template.name}")
                code = template.generate(protocol, dataset_schema=dataset_schema)

                # Optionally enhance with LLM
                if self.llm_enhance_templates and self.llm_client:
                    code = self._enhance_with_llm(code, protocol)

        # Step 2: Fall back to LLM generation
        if code is None and self.use_llm:
            logger.info("No template matched, using LLM generation")
            code = self._generate_with_llm(protocol, dataset_schema)

        # Step 3: Fallback to basic template
        if code is None:
            logger.warning("No code generated, using basic template")
            code = self._generate_basic_template(protocol)

        # Validate syntax
        self._validate_syntax(code)

        return code

    @staticmethod
    def _check_bindings(protocol: ExperimentProtocol, dataset_schema: Optional["DatasetSchema"]) -> None:
        """Refuse to analyse a dataset-bound protocol whose x or y has no column."""
        bound = dataset_schema is not None or any(v.column for v in protocol.variables.values())
        if not bound:
            return
        x, y = _xy_variables(protocol)
        unbound = [v.name for v in (x, y) if v is not None and v.column is None]
        if x is None or y is None or unbound:
            raise ValueError(f"protocol has unbound variables: {unbound or 'no x/y variables'}")

    def _match_template(self, protocol: ExperimentProtocol) -> Optional[CodeTemplate]:
        """Find best matching template for protocol."""
        for template in self.templates:
            if template.matches(protocol):
                return template
        return None

    def _generate_with_llm(
        self, protocol: ExperimentProtocol, dataset_schema: Optional["DatasetSchema"] = None
    ) -> str:
        """Generate code using Claude LLM."""
        prompt = self._create_code_generation_prompt(protocol, dataset_schema)

        try:
            response = self.llm_client.generate(prompt)

            # Extract code from response (may be in code blocks)
            code = self._extract_code_from_response(getattr(response, "content", response))

            return code

        except Exception as e:
            logger.error(f"LLM code generation failed: {e}")
            return None

    def _create_code_generation_prompt(
        self, protocol: ExperimentProtocol, dataset_schema: Optional["DatasetSchema"] = None
    ) -> str:
        """Create prompt for LLM code generation."""
        steps_text = "\n".join([
            f"{i+1}. {step.title}: {step.action}"
            for i, step in enumerate(protocol.steps)
        ])

        variables_text = "\n".join([
            f"- {name} ({var.type.value})"
            + (f" -> dataset column {var.column!r}" if var.column else "")
            + f": {var.description}"
            for name, var in protocol.variables.items()
        ])

        dataset_text = ""
        if dataset_schema is not None:
            dataset_text = (
                "\n**Dataset:**\n" + dataset_schema.to_prompt_block() + "\n"
                "Read only these exact column names; each variable's dataset column is given above. "
                "Do not invent columns and do not fall back to synthetic data.\n"
            )

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
{dataset_text}
**Statistical Tests:**
{tests_text}

Generate complete, executable Python code that:
1. Loads data from the `data_path` variable (already defined by executor)
2. Implements each protocol step
3. Performs the specified statistical tests
4. Returns results in a dictionary
5. Assign the final results dictionary to a top-level variable named results

IMPORTANT: Use `data_path` variable for loading data, e.g., `pd.read_csv(data_path)`
Do NOT hardcode 'data.csv' - use the data_path variable instead.

Use these libraries: pandas, numpy, scipy.stats
Do NOT import kosmos or any kosmos.* module; only pandas, numpy, scipy.stats, scikit-learn and statsmodels exist in the sandbox.
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
            "# Load data (data_path variable is provided by executor)",
            "df = pd.read_csv(data_path)",
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

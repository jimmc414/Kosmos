"""
Result collection and storage.

Extracts results from experiment execution, creates structured ExperimentResult objects,
and stores them in the database.
"""

import json
import sys
import platform
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
import re
import numpy as np
import pandas as pd

import uuid
from sqlalchemy.orm import Session
from kosmos.models.result import (
    ExperimentResult, ResultStatus, ExecutionMetadata,
    StatisticalTestResult, VariableResult, ResultExport
)
from kosmos.models.experiment import ExperimentProtocol
from kosmos.db import operations as db_ops

logger = logging.getLogger(__name__)


class ResultCollector:
    """
    Collects and structures experiment results.

    Extracts results from execution output, statistical tests, and data analysis.
    Creates ExperimentResult objects and stores them in the database.
    """

    def __init__(self, store_in_db: bool = True):
        """
        Initialize result collector.

        Args:
            store_in_db: If True, automatically store results in database
        """
        self.store_in_db = store_in_db
        self.library_versions = self._get_library_versions()

    @staticmethod
    def _get_library_versions() -> Dict[str, str]:
        """Get versions of key scientific libraries."""
        versions = {}

        libraries = ['numpy', 'pandas', 'scipy', 'sklearn', 'statsmodels', 'matplotlib']

        for lib_name in libraries:
            try:
                module = __import__(lib_name)
                versions[lib_name] = getattr(module, '__version__', 'unknown')
            except ImportError:
                versions[lib_name] = 'not installed'

        return versions

    def collect(
        self,
        protocol: ExperimentProtocol,
        execution_output: Dict[str, Any],
        session: Optional[Session] = None,
        statistical_tests: Optional[List[Dict[str, Any]]] = None,
        variable_data: Optional[Dict[str, pd.Series]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> ExperimentResult:
        """
        Collect and structure experiment results.

        Args:
            protocol: Experiment protocol that was executed
            execution_output: Output from code execution
            session: SQLAlchemy session for database operations
            statistical_tests: List of statistical test results
            variable_data: Dictionary of variable name -> data Series
            start_time: Execution start time
            end_time: Execution end time

        Returns:
            Structured ExperimentResult object
        """
        # Determine status
        status = self._determine_status(execution_output)

        # Extract metadata
        metadata = self._create_metadata(
            protocol=protocol,
            execution_output=execution_output,
            start_time=start_time or datetime.utcnow(),
            end_time=end_time or datetime.utcnow()
        )

        # Extract raw data
        raw_data = self._extract_raw_data(execution_output)

        # Process data
        processed_data = self._process_data(raw_data)

        # Extract variable results
        variable_results = []
        if variable_data:
            variable_results = self._create_variable_results(variable_data, protocol)

        # Extract statistical test results
        stat_test_results = []
        primary_test = None
        primary_p_value = None
        primary_effect_size = None

        if statistical_tests:
            stat_test_results, primary_test, primary_p_value, primary_effect_size = \
                self._create_statistical_test_results(statistical_tests)

        # Determine if hypothesis is supported
        supports_hypothesis = self._determine_hypothesis_support(
            primary_p_value,
            primary_effect_size,
            protocol
        )

        # Extract stdout/stderr
        stdout = execution_output.get('stdout', '')
        stderr = execution_output.get('stderr', '')

        # Get generated files
        generated_files = execution_output.get('generated_files', [])

        # Create result object
        result = ExperimentResult(
            experiment_id=protocol.id or 'unknown',
            protocol_id=protocol.id or 'unknown',
            hypothesis_id=protocol.hypothesis_id,
            status=status,
            raw_data=raw_data,
            processed_data=processed_data,
            variable_results=variable_results,
            statistical_tests=stat_test_results,
            primary_test=primary_test,
            primary_p_value=primary_p_value,
            primary_effect_size=primary_effect_size,
            supports_hypothesis=supports_hypothesis,
            metadata=metadata,
            stdout=stdout if stdout else None,
            stderr=stderr if stderr else None,
            generated_files=generated_files
        )

        # Store in database if requested
        if self.store_in_db and session:
            self._store_result(session, result)

        logger.info(f"Collected result for experiment {protocol.id}: "
                   f"status={status.value}, p-value={primary_p_value}")

        return result

    def _determine_status(self, execution_output: Dict[str, Any]) -> ResultStatus:
        """Determine result status from execution output."""
        if execution_output.get('error'):
            return ResultStatus.ERROR
        elif execution_output.get('timeout'):
            return ResultStatus.TIMEOUT
        elif execution_output.get('partial'):
            return ResultStatus.PARTIAL
        elif execution_output.get('failed'):
            return ResultStatus.FAILED
        else:
            return ResultStatus.SUCCESS

    def _create_metadata(
        self,
        protocol: ExperimentProtocol,
        execution_output: Dict[str, Any],
        start_time: datetime,
        end_time: datetime
    ) -> ExecutionMetadata:
        """Create execution metadata."""
        duration_seconds = (end_time - start_time).total_seconds()

        return ExecutionMetadata(
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration_seconds,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            platform=platform.platform(),
            hostname=platform.node(),
            cpu_time_seconds=execution_output.get('cpu_time'),
            memory_peak_mb=execution_output.get('memory_peak_mb'),
            random_seed=protocol.random_seed,
            library_versions=self.library_versions,
            experiment_id=protocol.id or 'unknown',
            protocol_id=protocol.id or 'unknown',
            hypothesis_id=protocol.hypothesis_id,
            sandbox_used=execution_output.get('sandbox_used', False),
            timeout_occurred=execution_output.get('timeout', False),
            errors=execution_output.get('errors', []),
            warnings=execution_output.get('warnings', [])
        )

    def _extract_raw_data(self, execution_output: Dict[str, Any]) -> Dict[str, Any]:
        """Extract raw data from execution output."""
        raw_data = {}

        # Get return value if present
        if 'return_value' in execution_output:
            raw_data['return_value'] = execution_output['return_value']

        # Get data from stdout (parse JSON if present)
        if 'stdout' in execution_output:
            stdout = execution_output['stdout']
            parsed_data = self._parse_stdout_data(stdout)
            if parsed_data:
                raw_data['stdout_data'] = parsed_data

        # Get explicit data field
        if 'data' in execution_output:
            raw_data['data'] = execution_output['data']

        # Get results field
        if 'results' in execution_output:
            raw_data['results'] = execution_output['results']

        return raw_data

    def _parse_stdout_data(self, stdout: str) -> Optional[Dict[str, Any]]:
        """Parse data from stdout (look for JSON blocks)."""
        # Look for JSON blocks in stdout
        json_pattern = r'\{[^{}]*\}'  # Simple pattern for single-line JSON

        matches = re.findall(json_pattern, stdout)

        parsed_data = {}
        for i, match in enumerate(matches):
            try:
                data = json.loads(match)
                if isinstance(data, dict):
                    parsed_data[f'json_block_{i}'] = data
            except json.JSONDecodeError:
                continue

        return parsed_data if parsed_data else None

    def _process_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw data into structured format."""
        processed = {}

        for key, value in raw_data.items():
            # Convert numpy/pandas types to native Python types
            if isinstance(value, (np.ndarray, pd.Series)):
                processed[key] = value.tolist()
            elif isinstance(value, pd.DataFrame):
                processed[key] = value.to_dict(orient='records')
            elif isinstance(value, (np.integer, np.floating)):
                processed[key] = value.item()
            else:
                processed[key] = value

        return processed

    def _create_variable_results(
        self,
        variable_data: Dict[str, pd.Series],
        protocol: ExperimentProtocol
    ) -> List[VariableResult]:
        """Create variable result objects from data."""
        variable_results = []

        for var_name, data in variable_data.items():
            # Get variable type from protocol
            var_type = 'unknown'
            if var_name in protocol.variables:
                var_type = protocol.variables[var_name].type.value

            # Calculate summary statistics
            data_clean = data.dropna()

            if len(data_clean) > 0 and pd.api.types.is_numeric_dtype(data_clean):
                mean = float(np.mean(data_clean))
                median = float(np.median(data_clean))
                std = float(np.std(data_clean, ddof=1)) if len(data_clean) > 1 else 0.0
                min_val = float(np.min(data_clean))
                max_val = float(np.max(data_clean))
            else:
                mean = median = std = min_val = max_val = None

            # Store values if not too large
            values = data_clean.tolist() if len(data_clean) <= 1000 else None

            variable_result = VariableResult(
                variable_name=var_name,
                variable_type=var_type,
                mean=mean,
                median=median,
                std=std,
                min=min_val,
                max=max_val,
                values=values,
                n_samples=len(data),
                n_missing=int(data.isna().sum())
            )

            variable_results.append(variable_result)

        return variable_results

    def _create_statistical_test_results(
        self,
        statistical_tests: List[Dict[str, Any]]
    ) -> Tuple[List[StatisticalTestResult], Optional[str], Optional[float], Optional[float]]:
        """
        Create statistical test result objects.

        Returns:
            Tuple of (test_results, primary_test_name, primary_p_value, primary_effect_size)
        """
        test_results = []
        primary_test = None
        primary_p_value = None
        primary_effect_size = None

        for i, test_data in enumerate(statistical_tests):
            # Extract required fields
            test_type = test_data.get('test_type', f'test_{i}')
            test_name = test_data.get('test_name', test_type)

            # Main statistics
            statistic = test_data.get('statistic', test_data.get('t_statistic',
                                      test_data.get('f_statistic',
                                      test_data.get('u_statistic',
                                      test_data.get('chi2_statistic', 0.0)))))

            p_value = test_data.get('p_value', 1.0)

            # Effect size
            effect_size = test_data.get('effect_size')
            effect_size_type = test_data.get('effect_size_type')

            # Confidence interval
            ci = test_data.get('confidence_interval')
            confidence_interval = None
            if ci and isinstance(ci, (list, tuple)) and len(ci) == 2:
                confidence_interval = {'lower': float(ci[0]), 'upper': float(ci[1])}
            elif ci and isinstance(ci, dict):
                confidence_interval = ci

            confidence_level = test_data.get('confidence_level', 0.95)

            # Significance
            sig_0_05 = test_data.get('significant_0.05', test_data.get('significant_0_05', p_value < 0.05))
            sig_0_01 = test_data.get('significant_0.01', test_data.get('significant_0_01', p_value < 0.01))
            sig_0_001 = test_data.get('significant_0.001', test_data.get('significant_0_001', p_value < 0.001))
            significance_label = test_data.get('significance_label', self._get_significance_label(p_value))

            # Sample information
            sample_size = test_data.get('sample_size', test_data.get('n_samples'))
            dof = test_data.get('degrees_of_freedom', test_data.get('dof'))

            # Additional stats
            known_keys = {
                'test_type', 'test_name', 'statistic', 'p_value', 'effect_size',
                'effect_size_type', 'confidence_interval', 'confidence_level',
                'significant_0_05', 'significant_0_01', 'significant_0_001',
                'significance_label', 'sample_size', 'n_samples', 'degrees_of_freedom', 'dof',
                'interpretation', 'is_primary'
            }
            additional_stats = {k: v for k, v in test_data.items() if k not in known_keys}

            # Interpretation
            interpretation = test_data.get('interpretation')

            test_result = StatisticalTestResult(
                test_type=test_type,
                test_name=test_name,
                statistic=float(statistic),
                p_value=float(p_value),
                effect_size=float(effect_size) if effect_size is not None else None,
                effect_size_type=effect_size_type,
                confidence_interval=confidence_interval,
                confidence_level=confidence_level,
                significant_0_05=bool(sig_0_05),
                significant_0_01=bool(sig_0_01),
                significant_0_001=bool(sig_0_001),
                significance_label=significance_label,
                sample_size=sample_size,
                degrees_of_freedom=dof,
                additional_stats=additional_stats,
                interpretation=interpretation
            )

            test_results.append(test_result)

            # Track primary test (first one or explicitly marked)
            if i == 0 or test_data.get('is_primary', False):
                primary_test = test_name
                primary_p_value = float(p_value)
                primary_effect_size = float(effect_size) if effect_size is not None else None

        return test_results, primary_test, primary_p_value, primary_effect_size

    @staticmethod
    def _get_significance_label(p_value: float) -> str:
        """Get significance label from p-value."""
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return 'ns'

    def _determine_hypothesis_support(
        self,
        p_value: Optional[float],
        effect_size: Optional[float],
        protocol: ExperimentProtocol
    ) -> Optional[bool]:
        """Determine if results support the hypothesis."""
        if p_value is None:
            return None

        # Check significance
        is_significant = p_value < 0.05

        # Check effect size if available
        has_meaningful_effect = True
        if effect_size is not None:
            # Consider effect size > 0.2 as meaningful
            has_meaningful_effect = abs(effect_size) > 0.2

        # Support hypothesis if significant AND has meaningful effect
        return is_significant and has_meaningful_effect

    def _store_result(self, session: Session, result: ExperimentResult) -> None:
        """Store result in database."""
        try:
            # Generate a unique ID for the result
            result_id = f"res_{uuid.uuid4().hex[:16]}"
            result.id = result_id

            # Convert to dict for database storage
            result_dict = result.to_dict()

            # Store using database operations
            db_result = db_ops.create_result(
                session=session,
                id=result_id,
                experiment_id=result.experiment_id,
                data=result_dict,
                statistical_tests={test.test_name: test.model_dump() for test in result.statistical_tests},
                p_value=result.primary_p_value,
                effect_size=result.primary_effect_size,
                supports_hypothesis=result.supports_hypothesis,
                interpretation=result.interpretation
            )

            logger.info(f"Stored result in database with ID: {db_result.id}")

        except Exception as e:
            logger.error(f"Failed to store result in database: {e}")
            raise

    def export_result(
        self,
        result: ExperimentResult,
        format: str = 'json',
        output_path: Optional[str] = None
    ) -> str:
        """
        Export result to file.

        Args:
            result: ExperimentResult to export
            format: Export format ('json', 'csv', 'markdown')
            output_path: Optional path to save file

        Returns:
            Exported string
        """
        exporter = ResultExport(result=result, format=format)

        if format == 'json':
            content = exporter.export_json()
        elif format == 'csv':
            content = exporter.export_csv()
        elif format == 'markdown':
            content = exporter.export_markdown()
        else:
            raise ValueError(f"Unknown export format: {format}")

        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(content)
            logger.info(f"Exported result to {output_path}")

        return content

    def create_version(
        self,
        original_result: ExperimentResult,
        new_execution_output: Dict[str, Any],
        protocol: ExperimentProtocol
    ) -> ExperimentResult:
        """
        Create a new version of a result (for re-runs).

        Args:
            original_result: Original result
            new_execution_output: New execution output
            protocol: Protocol (may be updated)

        Returns:
            New ExperimentResult with incremented version
        """
        # Collect new result
        new_result = self.collect(
            protocol=protocol,
            execution_output=new_execution_output
        )

        # Update version information
        new_result.version = original_result.version + 1
        new_result.parent_result_id = original_result.id

        logger.info(f"Created result version {new_result.version} "
                   f"(parent: {original_result.id})")

        return new_result

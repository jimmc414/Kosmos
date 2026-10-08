"""
Code safety validator with ethical guidelines.

Enhanced version of code validation with:
- Syntax and security checks
- Ethical research guidelines validation
- Risk level assessment
- Experiment approval gate logic
"""

import ast
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

from kosmos.models.safety import (
    SafetyReport, SafetyViolation, ViolationType, RiskLevel,
    EthicalGuideline, ApprovalRequest, ApprovalStatus
)
from kosmos.utils.compat import model_to_dict
from kosmos.config import get_config

logger = logging.getLogger(__name__)


class CodeValidator:
    """
    Validates generated code for safety, security, and ethical compliance.

    Enhanced version with ethical guidelines and risk assessment.
    """

    # Dangerous modules that should not be imported
    DANGEROUS_MODULES = [
        'os', 'subprocess', 'sys', 'shutil', 'importlib',
        'socket', 'urllib', 'requests', 'http', 'ftplib',
        '__import__', 'eval', 'exec', 'compile', 'pickle'
    ]

    # Dangerous functions/operations
    DANGEROUS_PATTERNS = [
        'open(',  # File operations (except specific allowed cases)
        'eval(',
        'exec(',
        'compile(',
        '__import__',
        'globals(',
        'locals(',
        'vars(',
        'delattr(',
        'setattr(',
    ]

    # Network-related keywords (warnings)
    NETWORK_KEYWORDS = ['socket', 'http', 'urllib', 'requests', 'ftp']

    def __init__(
        self,
        ethical_guidelines_path: Optional[str] = None,
        allow_file_read: bool = True,
        allow_file_write: bool = False,
        allow_network: bool = False
    ):
        """
        Initialize code validator.

        Args:
            ethical_guidelines_path: Path to JSON file with ethical guidelines
            allow_file_read: Allow read-only file operations
            allow_file_write: Allow file write operations
            allow_network: Allow network operations
        """
        self.allow_file_read = allow_file_read
        self.allow_file_write = allow_file_write
        self.allow_network = allow_network

        # Load ethical guidelines
        self.ethical_guidelines = self._load_ethical_guidelines(ethical_guidelines_path)

        logger.info(
            f"CodeValidator initialized (file_read={allow_file_read}, "
            f"file_write={allow_file_write}, network={allow_network}, "
            f"guidelines={len(self.ethical_guidelines)})"
        )

    def _load_ethical_guidelines(self, path: Optional[str]) -> List[EthicalGuideline]:
        """Load ethical guidelines from JSON file or use defaults."""
        guidelines = []

        # Try to load from file
        if isinstance(path, (str, os.PathLike)) and Path(path).exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    for item in data.get('guidelines', []):
                        guidelines.append(EthicalGuideline(**item))
                logger.info(f"Loaded {len(guidelines)} ethical guidelines from {path}")
            except Exception as e:
                logger.error(f"Error loading ethical guidelines: {e}")

        # Use default guidelines if none loaded
        if not guidelines:
            guidelines = self._get_default_ethical_guidelines()
            logger.info(f"Using {len(guidelines)} default ethical guidelines")

        return guidelines

    def _get_default_ethical_guidelines(self) -> List[EthicalGuideline]:
        """Get default ethical research guidelines."""
        return [
            EthicalGuideline(
                guideline_id="no_harm",
                category="general",
                description="Research must not cause harm to individuals or communities",
                required=True,
                validation_method="keyword",
                keywords=["harm", "danger", "toxic", "weapon", "exploit"],
                severity_if_violated=RiskLevel.CRITICAL
            ),
            EthicalGuideline(
                guideline_id="privacy",
                category="data_privacy",
                description="Must protect personally identifiable information (PII)",
                required=True,
                validation_method="keyword",
                keywords=["pii", "ssn", "credit card", "password", "email"],
                severity_if_violated=RiskLevel.HIGH
            ),
            EthicalGuideline(
                guideline_id="informed_consent",
                category="human_subjects",
                description="Human subjects research requires informed consent",
                required=True,
                validation_method="keyword",
                keywords=["human subjects", "participants", "survey", "experiment on"],
                severity_if_violated=RiskLevel.HIGH
            ),
            EthicalGuideline(
                guideline_id="animal_welfare",
                category="animal_welfare",
                description="Animal research must follow welfare guidelines",
                required=True,
                validation_method="keyword",
                keywords=["animal testing", "animal experiments", "in vivo"],
                severity_if_violated=RiskLevel.HIGH
            ),
            EthicalGuideline(
                guideline_id="environmental",
                category="environmental",
                description="Must consider environmental impact",
                required=False,
                validation_method="keyword",
                keywords=["toxic waste", "pollution", "environmental damage"],
                severity_if_violated=RiskLevel.MEDIUM
            ),
        ]

    def validate(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SafetyReport:
        """
        Validate code for safety, security, and ethical compliance.

        Args:
            code: Python code to validate
            context: Optional context about the code (experiment description, etc.)

        Returns:
            SafetyReport with validation results
        """
        violations = []
        warnings = []
        checks_performed = []

        # 1. Syntax check
        syntax_violations = self._check_syntax(code)
        violations.extend(syntax_violations)
        checks_performed.append("syntax")

        # 2. Dangerous imports check
        import_violations = self._check_dangerous_imports(code)
        violations.extend(import_violations)
        checks_performed.append("dangerous_imports")

        # 3. Dangerous patterns check
        pattern_violations, pattern_warnings = self._check_dangerous_patterns(code)
        violations.extend(pattern_violations)
        warnings.extend(pattern_warnings)
        checks_performed.append("dangerous_patterns")

        # 4. Network operations check
        network_warnings = self._check_network_operations(code)
        warnings.extend(network_warnings)
        checks_performed.append("network_operations")

        # 5. AST call analysis (getattr/setattr/delattr calls, dunder access)
        ast_call_violations = self._check_ast_calls(code)
        violations.extend(ast_call_violations)
        checks_performed.append("ast_call_analysis")

        # 6. Ethical guidelines check
        ethical_violations = self._check_ethical_guidelines(code, context)
        violations.extend(ethical_violations)
        checks_performed.append("ethical_guidelines")

        # 7. Assess risk level
        risk_level = self._assess_risk_level(violations)

        # Create report
        passed = len(violations) == 0
        report = SafetyReport(
            passed=passed,
            risk_level=risk_level,
            violations=violations,
            warnings=warnings,
            checks_performed=checks_performed,
            metadata={
                "code_length": len(code),
                "allow_file_read": self.allow_file_read,
                "allow_file_write": self.allow_file_write,
                "allow_network": self.allow_network,
                "context": context or {}
            }
        )

        logger.info(f"Code validation: {report.summary()}")

        return report

    def _check_syntax(self, code: str) -> List[SafetyViolation]:
        """Check for syntax errors."""
        violations = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            violations.append(SafetyViolation(
                type=ViolationType.DANGEROUS_CODE,
                severity=RiskLevel.HIGH,
                message=f"Syntax error: {e}",
                location=f"line {e.lineno}" if hasattr(e, 'lineno') else None
            ))
        return violations

    def _check_dangerous_imports(self, code: str) -> List[SafetyViolation]:
        """Check for dangerous module imports using AST parsing."""
        violations = []
        try:
            import ast
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.DANGEROUS_MODULES or any(alias.name.startswith(f"{m}.") for m in self.DANGEROUS_MODULES):
                            violations.append(SafetyViolation(
                                type=ViolationType.DANGEROUS_CODE,
                                severity=RiskLevel.CRITICAL,
                                message=f"Dangerous import detected: {alias.name}",
                                details={"module": alias.name}
                            ))
                elif isinstance(node, ast.ImportFrom):
                    if node.module and (node.module in self.DANGEROUS_MODULES or any(node.module.startswith(f"{m}.") for m in self.DANGEROUS_MODULES)):
                        violations.append(SafetyViolation(
                            type=ViolationType.DANGEROUS_CODE,
                            severity=RiskLevel.CRITICAL,
                            message=f"Dangerous import detected: from {node.module}",
                            details={"module": node.module}
                        ))
        except SyntaxError:
            # Fall back to string matching if code has syntax errors
            for module in self.DANGEROUS_MODULES:
                if f"import {module}" in code or f"from {module}" in code:
                    violations.append(SafetyViolation(
                        type=ViolationType.DANGEROUS_CODE,
                        severity=RiskLevel.CRITICAL,
                        message=f"Dangerous import detected: {module}",
                        details={"module": module}
                    ))
        return violations

    def _check_dangerous_patterns(self, code: str) -> tuple:
        """Check for dangerous code patterns."""
        violations = []
        warnings = []

        for pattern in self.DANGEROUS_PATTERNS:
            if pattern in code:
                # Special case: allow open() for reading if permitted
                if pattern == 'open(':
                    if self.allow_file_write:
                        warnings.append(f"File operation detected: {pattern}")
                    elif self.allow_file_read:
                        # Check if it's read-only (contains "'r'" or no mode specified)
                        if re.search(r'open\s*\([^)]*(?:["\'][waxWAX][+b]*["\']|mode\s*=\s*["\'][waxWAX])', code):
                            violations.append(SafetyViolation(
                                type=ViolationType.FILE_SYSTEM_ACCESS,
                                severity=RiskLevel.HIGH,
                                message="Write mode file operations not allowed",
                                details={"pattern": pattern}
                            ))
                        else:
                            warnings.append(f"File read operation detected: {pattern}")
                    else:
                        violations.append(SafetyViolation(
                            type=ViolationType.FILE_SYSTEM_ACCESS,
                            severity=RiskLevel.HIGH,
                            message="File operations not allowed",
                            details={"pattern": pattern}
                        ))
                else:
                    violations.append(SafetyViolation(
                        type=ViolationType.DANGEROUS_CODE,
                        severity=RiskLevel.CRITICAL,
                        message=f"Dangerous operation detected: {pattern}",
                        details={"pattern": pattern}
                    ))

        return violations, warnings

    def _check_ast_calls(self, code: str) -> List[SafetyViolation]:
        """Check for dangerous function calls and dunder attribute access via AST.

        Detects:
        - Calls to getattr(), setattr(), delattr() that bypass static analysis
        - Access to dangerous dunder attributes: __dict__, __class__,
          __builtins__, __subclasses__
        """
        violations = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Syntax errors are caught by _check_syntax; skip AST analysis
            return violations

        # Dangerous built-in reflection calls
        DANGEROUS_CALLS = {'getattr', 'setattr', 'delattr'}

        # Dangerous dunder attributes
        DANGEROUS_DUNDERS = {'__dict__', '__class__', '__builtins__', '__subclasses__'}

        for node in ast.walk(tree):
            # Check for getattr/setattr/delattr calls
            if isinstance(node, ast.Call):
                func = node.func
                func_name = None
                if isinstance(func, ast.Name):
                    func_name = func.id
                elif isinstance(func, ast.Attribute):
                    func_name = func.attr

                if func_name in DANGEROUS_CALLS:
                    lineno = getattr(node, 'lineno', None)
                    violations.append(SafetyViolation(
                        type=ViolationType.DANGEROUS_CODE,
                        severity=RiskLevel.CRITICAL,
                        message=f"Dangerous reflection call detected: {func_name}()",
                        location=f"line {lineno}" if lineno else None,
                        details={"call": func_name}
                    ))

            # Check for dunder attribute access (e.g., obj.__dict__)
            if isinstance(node, ast.Attribute):
                if node.attr in DANGEROUS_DUNDERS:
                    lineno = getattr(node, 'lineno', None)
                    violations.append(SafetyViolation(
                        type=ViolationType.DANGEROUS_CODE,
                        severity=RiskLevel.CRITICAL,
                        message=f"Dangerous dunder access detected: .{node.attr}",
                        location=f"line {lineno}" if lineno else None,
                        details={"attribute": node.attr}
                    ))

        return violations

    def _check_network_operations(self, code: str) -> List[str]:
        """Check for network operations (warnings only)."""
        warnings = []
        if not self.allow_network:
            for keyword in self.NETWORK_KEYWORDS:
                if keyword in code.lower():
                    warnings.append(f"Potential network operation detected: {keyword}")
        return warnings

    def _check_ethical_guidelines(
        self,
        code: str,
        context: Optional[Dict[str, Any]]
    ) -> List[SafetyViolation]:
        """Check code against ethical research guidelines."""
        violations = []

        # Combine code and context for checking
        text_to_check = code.lower()
        if context:
            description = context.get('description', '') or context.get('hypothesis', '')
            if description:
                text_to_check += " " + str(description).lower()

        for guideline in self.ethical_guidelines:
            if guideline.validation_method == "keyword":
                # Check if any keywords appear in the text
                for keyword in guideline.keywords:
                    if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text_to_check):
                        if guideline.required:
                            violations.append(SafetyViolation(
                                type=ViolationType.ETHICAL_VIOLATION,
                                severity=guideline.severity_if_violated,
                                message=f"Potential ethical violation: {guideline.description}",
                                details={
                                    "guideline_id": guideline.guideline_id,
                                    "category": guideline.category,
                                    "keyword": keyword
                                }
                            ))
                        break  # Only report once per guideline

        return violations

    def _assess_risk_level(self, violations: List[SafetyViolation]) -> RiskLevel:
        """Assess overall risk level based on violations."""
        if not violations:
            return RiskLevel.LOW

        # Find highest severity violation
        severity_order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        max_severity = RiskLevel.LOW

        for violation in violations:
            if severity_order.index(violation.severity) > severity_order.index(max_severity):
                max_severity = violation.severity

        return max_severity

    def requires_approval(self, report: SafetyReport) -> bool:
        """
        Determine if code requires human approval based on safety report.

        Args:
            report: SafetyReport from validation

        Returns:
            True if human approval is required
        """
        try:
            config = get_config()
        except Exception:
            logger.warning("Config not available in requires_approval(); defaulting to require approval")
            return True

        # Always require approval if configured
        if config.safety.require_human_approval:
            return True

        # Require approval for high/critical risk
        if report.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            return True

        # Require approval if there are critical violations
        if report.has_critical_violations():
            return True

        # Require approval for ethical violations
        ethical_violations = report.get_violations_by_type(ViolationType.ETHICAL_VIOLATION)
        if ethical_violations:
            return True

        return False

    def create_approval_request(
        self,
        code: str,
        report: SafetyReport,
        context: Optional[Dict[str, Any]] = None
    ) -> ApprovalRequest:
        """
        Create an approval request for code that needs human review.

        Args:
            code: Code to be approved
            report: Safety report for the code
            context: Additional context

        Returns:
            ApprovalRequest object
        """
        import uuid

        request_id = f"approval_{uuid.uuid4().hex[:8]}"

        # Build description
        description_parts = ["Code requires approval due to:"]
        for violation in report.violations:
            description_parts.append(f"- {violation.message}")

        description = "\n".join(description_parts)

        # Determine reason
        reasons = []
        if report.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            reasons.append(f"{report.risk_level.value} risk level")
        if report.has_violations():
            reasons.append(f"{len(report.violations)} safety violations")

        reason = ", ".join(reasons)

        return ApprovalRequest(
            request_id=request_id,
            operation_type="code_execution",
            operation_description=description,
            risk_level=report.risk_level,
            reason_for_approval=reason,
            context={
                "code": code[:500],  # First 500 chars
                "report": model_to_dict(report),
                **(context or {})
            }
        )

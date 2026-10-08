"""
Safety-related data models.

Pydantic models for safety reports, incidents, approvals, and ethical guidelines.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class RiskLevel(str, Enum):
    """Risk level for experiments and operations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ViolationType(str, Enum):
    """Types of safety violations."""

    DANGEROUS_CODE = "dangerous_code"
    RESOURCE_LIMIT = "resource_limit"
    ETHICAL_VIOLATION = "ethical_violation"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    NETWORK_ACCESS = "network_access"
    FILE_SYSTEM_ACCESS = "file_system_access"
    SUSPICIOUS_PATTERN = "suspicious_pattern"


class ApprovalStatus(str, Enum):
    """Status of approval requests."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class SafetyViolation(BaseModel):
    """Detailed information about a safety violation."""

    model_config = ConfigDict(frozen=False)

    type: ViolationType
    severity: RiskLevel
    message: str
    location: Optional[str] = None  # File path, line number, or code location
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class SafetyReport(BaseModel):
    """Report from safety validation checks."""

    model_config = ConfigDict(frozen=False)

    passed: bool
    risk_level: RiskLevel
    violations: List[SafetyViolation] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    checks_performed: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    def has_violations(self) -> bool:
        """Check if report has any violations."""
        return len(self.violations) > 0

    def has_critical_violations(self) -> bool:
        """Check if report has any critical or high-risk violations."""
        return any(
            v.severity in [RiskLevel.CRITICAL, RiskLevel.HIGH]
            for v in self.violations
        )

    def get_violations_by_type(self, violation_type: ViolationType) -> List[SafetyViolation]:
        """Get violations of a specific type."""
        return [v for v in self.violations if v.type == violation_type]

    def summary(self) -> str:
        """Generate human-readable summary."""
        if self.passed:
            return f"✓ Safety checks passed (Risk: {self.risk_level.value})"
        else:
            violation_count = len(self.violations)
            warning_count = len(self.warnings)
            return (
                f"✗ Safety checks failed (Risk: {self.risk_level.value}): "
                f"{violation_count} violations, {warning_count} warnings"
            )


class SafetyIncident(BaseModel):
    """Record of a safety incident for logging and audit trail."""

    model_config = ConfigDict(frozen=False)

    incident_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    violation: Optional[SafetyViolation] = None  # None for an emergency stop
    context: Dict[str, Any] = Field(default_factory=dict)
    action_taken: str  # What was done in response
    experiment_id: Optional[str] = None
    hypothesis_id: Optional[str] = None
    user_notified: bool = False
    resolved: bool = False
    resolution_notes: Optional[str] = None


class EthicalGuideline(BaseModel):
    """Ethical research guideline for validation."""

    model_config = ConfigDict(frozen=False)

    guideline_id: str
    category: str  # e.g., "human_subjects", "animal_welfare", "data_privacy", "environmental"
    description: str
    required: bool = True
    validation_method: str  # "keyword", "llm", "manual"
    keywords: List[str] = Field(default_factory=list)  # For keyword-based validation
    severity_if_violated: RiskLevel = RiskLevel.HIGH


class ApprovalRequest(BaseModel):
    """Request for human approval of an operation."""

    model_config = ConfigDict(frozen=False)

    request_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    operation_type: str  # "experiment_execution", "hypothesis_generation", "data_access"
    operation_description: str
    risk_level: RiskLevel
    reason_for_approval: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: Optional[str] = None
    approval_timestamp: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    timeout_seconds: int = 3600  # Auto-reject after 1 hour by default
    context: Dict[str, Any] = Field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if approval request has expired."""
        if self.status != ApprovalStatus.PENDING:
            return False

        elapsed = (datetime.now() - self.timestamp).total_seconds()
        return elapsed > self.timeout_seconds

    def approve(self, approved_by: str):
        """Approve the request."""
        self.status = ApprovalStatus.APPROVED
        self.approved_by = approved_by
        self.approval_timestamp = datetime.now()

    def reject(self, rejected_by: str, reason: str):
        """Reject the request."""
        self.status = ApprovalStatus.REJECTED
        self.approved_by = rejected_by
        self.approval_timestamp = datetime.now()
        self.rejection_reason = reason

    def mark_expired(self):
        """Mark request as expired."""
        if self.is_expired():
            self.status = ApprovalStatus.EXPIRED


class ResourceLimit(BaseModel):
    """Resource consumption limits for experiments."""

    model_config = ConfigDict(frozen=False)

    max_cpu_cores: Optional[float] = None
    max_memory_mb: Optional[int] = None
    max_execution_time_seconds: Optional[int] = None
    max_disk_usage_mb: Optional[int] = None
    max_network_bandwidth_mbps: Optional[float] = None
    allow_network_access: bool = False
    allow_file_write: bool = False
    allow_subprocess: bool = False


class EmergencyStopStatus(BaseModel):
    """Status of emergency stop mechanism."""

    model_config = ConfigDict(frozen=False)

    is_active: bool = False
    triggered_at: Optional[datetime] = None
    triggered_by: Optional[str] = None  # "signal", "flag_file", "api", "user"
    reason: Optional[str] = None
    affected_experiments: List[str] = Field(default_factory=list)

    def trigger(self, triggered_by: str, reason: str, affected_experiments: List[str] = None):
        """Trigger emergency stop."""
        self.is_active = True
        self.triggered_at = datetime.now()
        self.triggered_by = triggered_by
        self.reason = reason
        if affected_experiments:
            self.affected_experiments = affected_experiments

    def reset(self):
        """Reset emergency stop."""
        self.is_active = False
        self.triggered_at = None
        self.triggered_by = None
        self.reason = None
        self.affected_experiments = []

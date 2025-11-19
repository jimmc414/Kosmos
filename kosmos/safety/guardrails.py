"""
Safety guardrails for autonomous research.

Implements:
- Emergency stop mechanism (signals + flag file)
- Resource consumption limits
- Safety incident logging
- Code validation coordination
- Experiment approval gates
"""

import signal
import logging
import json
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import contextmanager

from kosmos.models.safety import (
    SafetyReport, SafetyIncident, ViolationType, RiskLevel,
    ResourceLimit, EmergencyStopStatus
)
from kosmos.safety.code_validator import CodeValidator
from kosmos.config import get_config

logger = logging.getLogger(__name__)


class SafetyGuardrails:
    """
    Comprehensive safety guardrails for autonomous research.

    Provides:
    - Code safety validation
    - Resource limit enforcement
    - Emergency stop mechanism (signals + flag file)
    - Safety incident logging
    """

    # Path to emergency stop flag file
    STOP_FLAG_FILE = Path(".kosmos_emergency_stop")

    def __init__(
        self,
        incident_log_path: Optional[str] = None,
        enable_signal_handlers: bool = True
    ):
        """
        Initialize safety guardrails.

        Args:
            incident_log_path: Path to safety incident log file
            enable_signal_handlers: Register signal handlers for emergency stop
        """
        config = get_config()

        # Initialize code validator
        self.code_validator = CodeValidator(
            ethical_guidelines_path=getattr(config.safety, 'ethical_guidelines_path', None),
            allow_file_read=True,
            allow_file_write=False,
            allow_network=False
        )

        # Incident logging
        self.incident_log_path = incident_log_path or "safety_incidents.jsonl"
        self.incidents: List[SafetyIncident] = []

        # Emergency stop status
        self.emergency_stop = EmergencyStopStatus()

        # Register signal handlers
        if enable_signal_handlers:
            self._register_signal_handlers()

        # Resource limits (from config)
        self.default_resource_limits = ResourceLimit(
            max_cpu_cores=getattr(config.safety, 'max_cpu_cores', None),
            max_memory_mb=getattr(config.safety, 'max_memory_mb', 2048),
            max_execution_time_seconds=getattr(config.safety, 'max_execution_time', 300),
            allow_network_access=False,
            allow_file_write=False,
            allow_subprocess=False
        )

        logger.info("SafetyGuardrails initialized")

    def _register_signal_handlers(self):
        """Register signal handlers for emergency stop."""
        def signal_handler(signum, frame):
            logger.warning(f"Emergency stop signal received: {signum}")
            self.trigger_emergency_stop(
                triggered_by="signal",
                reason=f"Signal {signum} received"
            )

        # Register SIGTERM and SIGINT
        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
            logger.info("Signal handlers registered for emergency stop")
        except Exception as e:
            logger.warning(f"Could not register signal handlers: {e}")

    def validate_code(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SafetyReport:
        """
        Validate code for safety.

        Args:
            code: Python code to validate
            context: Optional context (experiment description, hypothesis, etc.)

        Returns:
            SafetyReport with validation results
        """
        # Check for emergency stop
        if self.is_emergency_stop_active():
            raise RuntimeError(
                f"Emergency stop active: {self.emergency_stop.reason}"
            )

        # Validate code
        report = self.code_validator.validate(code, context)

        # Log incidents if violations found
        if not report.passed:
            self._log_violations(report, code, context)

        return report

    def enforce_resource_limits(
        self,
        requested_limits: Optional[ResourceLimit] = None
    ) -> ResourceLimit:
        """
        Enforce resource limits, using defaults if not specified.

        Args:
            requested_limits: Requested resource limits

        Returns:
            Enforced resource limits (may be capped)
        """
        if requested_limits is None:
            return self.default_resource_limits

        # Cap requested limits to defaults
        # Use explicit None checks instead of truthiness to handle 0 values correctly
        def get_value_or_inf(val):
            # Treat 0 as a valid limit (meaning no resources allowed)
            # Only treat None as unlimited
            return val if val is not None else float('inf')

        enforced = ResourceLimit(
            max_cpu_cores=min(
                get_value_or_inf(requested_limits.max_cpu_cores),
                get_value_or_inf(self.default_resource_limits.max_cpu_cores)
            ) if self.default_resource_limits.max_cpu_cores is not None else requested_limits.max_cpu_cores,
            max_memory_mb=min(
                get_value_or_inf(requested_limits.max_memory_mb),
                get_value_or_inf(self.default_resource_limits.max_memory_mb)
            ) if self.default_resource_limits.max_memory_mb is not None else requested_limits.max_memory_mb,
            max_execution_time_seconds=min(
                get_value_or_inf(requested_limits.max_execution_time_seconds),
                get_value_or_inf(self.default_resource_limits.max_execution_time_seconds)
            ) if self.default_resource_limits.max_execution_time_seconds is not None else requested_limits.max_execution_time_seconds,
            allow_network_access=requested_limits.allow_network_access and self.default_resource_limits.allow_network_access,
            allow_file_write=requested_limits.allow_file_write and self.default_resource_limits.allow_file_write,
            allow_subprocess=requested_limits.allow_subprocess and self.default_resource_limits.allow_subprocess
        )

        logger.debug(f"Enforced resource limits: {enforced.model_dump()}")
        return enforced

    def check_emergency_stop(self):
        """
        Check for emergency stop conditions.

        Checks both signal status and flag file.

        Raises:
            RuntimeError: If emergency stop is active
        """
        # Check flag file
        if self.STOP_FLAG_FILE.exists():
            if not self.emergency_stop.is_active:
                logger.warning("Emergency stop flag file detected")
                self.trigger_emergency_stop(
                    triggered_by="flag_file",
                    reason="Emergency stop flag file created"
                )

        # Raise exception if stop is active
        if self.is_emergency_stop_active():
            raise RuntimeError(
                f"Emergency stop active (triggered by {self.emergency_stop.triggered_by}): "
                f"{self.emergency_stop.reason}"
            )

    def is_emergency_stop_active(self) -> bool:
        """Check if emergency stop is currently active."""
        # Update from flag file if exists
        if self.STOP_FLAG_FILE.exists() and not self.emergency_stop.is_active:
            self.trigger_emergency_stop(
                triggered_by="flag_file",
                reason="Emergency stop flag file detected"
            )

        return self.emergency_stop.is_active

    def trigger_emergency_stop(
        self,
        triggered_by: str,
        reason: str,
        affected_experiments: Optional[List[str]] = None
    ):
        """
        Trigger emergency stop.

        Args:
            triggered_by: Source of stop (signal, flag_file, api, user)
            reason: Reason for emergency stop
            affected_experiments: List of affected experiment IDs
        """
        self.emergency_stop.trigger(
            triggered_by=triggered_by,
            reason=reason,
            affected_experiments=affected_experiments or []
        )

        logger.critical(
            f"EMERGENCY STOP TRIGGERED by {triggered_by}: {reason}"
        )

        # Create flag file if not exists
        if not self.STOP_FLAG_FILE.exists():
            try:
                self.STOP_FLAG_FILE.write_text(
                    json.dumps({
                        "triggered_at": self.emergency_stop.triggered_at.isoformat(),
                        "triggered_by": triggered_by,
                        "reason": reason
                    }, indent=2)
                )
            except Exception as e:
                logger.error(f"Could not create stop flag file: {e}")

        # Log as critical incident
        incident = SafetyIncident(
            incident_id=f"emergency_stop_{uuid.uuid4().hex[:8]}",
            violation=None,  # No specific violation
            context={
                "triggered_by": triggered_by,
                "reason": reason,
                "affected_experiments": affected_experiments or []
            },
            action_taken="Emergency stop triggered - all operations halted"
        )
        self._log_incident(incident)

    def reset_emergency_stop(self):
        """Reset emergency stop status."""
        self.emergency_stop.reset()

        # Remove flag file
        if self.STOP_FLAG_FILE.exists():
            try:
                self.STOP_FLAG_FILE.unlink()
                logger.info("Emergency stop flag file removed")
            except Exception as e:
                logger.error(f"Could not remove stop flag file: {e}")

        logger.info("Emergency stop reset")

    @contextmanager
    def safety_context(self, experiment_id: Optional[str] = None):
        """
        Context manager for safe code execution.

        Automatically checks emergency stop before and after execution.

        Args:
            experiment_id: Optional experiment ID for tracking

        Yields:
            None

        Raises:
            RuntimeError: If emergency stop is active

        Example:
            ```python
            with guardrails.safety_context(experiment_id="exp_123"):
                # Execute code
                result = execute_experiment()
            ```
        """
        # Check before execution
        self.check_emergency_stop()

        try:
            yield
        except Exception as e:
            # Check if exception is due to emergency stop
            if self.is_emergency_stop_active():
                logger.error(f"Execution interrupted by emergency stop: {e}")
                raise
            # Re-raise other exceptions
            raise
        finally:
            # Check after execution
            if self.is_emergency_stop_active():
                logger.warning(
                    f"Emergency stop detected after execution "
                    f"(experiment: {experiment_id})"
                )

    def _log_violations(
        self,
        report: SafetyReport,
        code: str,
        context: Optional[Dict[str, Any]]
    ):
        """Log safety violations as incidents."""
        for violation in report.violations:
            incident = SafetyIncident(
                incident_id=f"incident_{uuid.uuid4().hex[:8]}",
                violation=violation,
                context={
                    "code_snippet": code[:200],  # First 200 chars
                    "context": context or {},
                    "report_summary": report.summary()
                },
                action_taken="Code execution blocked",
                experiment_id=context.get('experiment_id') if context else None,
                hypothesis_id=context.get('hypothesis_id') if context else None
            )
            self._log_incident(incident)

    def _log_incident(self, incident: SafetyIncident):
        """Log safety incident to file and memory."""
        # Add to memory
        self.incidents.append(incident)

        # Write to log file (JSONL format)
        try:
            with open(self.incident_log_path, 'a') as f:
                f.write(json.dumps(incident.model_dump(), default=str) + '\n')
            logger.info(f"Safety incident logged: {incident.incident_id}")
        except Exception as e:
            logger.error(f"Error logging incident: {e}")

    def get_recent_incidents(
        self,
        limit: int = 10,
        severity: Optional[RiskLevel] = None
    ) -> List[SafetyIncident]:
        """
        Get recent safety incidents.

        Args:
            limit: Maximum number of incidents to return
            severity: Filter by violation severity

        Returns:
            List of recent incidents
        """
        incidents = self.incidents

        # Filter by severity if specified
        if severity:
            incidents = [
                i for i in incidents
                if i.violation and i.violation.severity == severity
            ]

        # Return most recent
        return incidents[-limit:]

    def get_incident_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of safety incidents.

        Returns:
            Dictionary with incident statistics
        """
        if not self.incidents:
            return {
                "total_incidents": 0,
                "by_type": {},
                "by_severity": {},
                "unresolved": 0
            }

        # Count by type
        by_type = {}
        by_severity = {}
        unresolved = 0

        for incident in self.incidents:
            if incident.violation:
                # Count by type
                vtype = incident.violation.type.value
                by_type[vtype] = by_type.get(vtype, 0) + 1

                # Count by severity
                sev = incident.violation.severity.value
                by_severity[sev] = by_severity.get(sev, 0) + 1

            # Count unresolved
            if not incident.resolved:
                unresolved += 1

        return {
            "total_incidents": len(self.incidents),
            "by_type": by_type,
            "by_severity": by_severity,
            "unresolved": unresolved,
            "emergency_stops": sum(
                1 for i in self.incidents
                if 'emergency_stop' in i.incident_id
            )
        }

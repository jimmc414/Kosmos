"""
Human oversight and review workflow.

Implements:
- Approval workflow (blocking, queue-based, automatic modes)
- Manual review interface (CLI)
- Override capabilities
- Audit trail logging
- Human feedback integration
"""

import logging
import json
import uuid
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum
from pathlib import Path

from kosmos.config import SafetyConfig
from kosmos.models.safety import ApprovalRequest, ApprovalStatus, RiskLevel

logger = logging.getLogger(__name__)


class ApprovalMode(str, Enum):
    """Approval workflow mode."""

    BLOCKING = "blocking"  # Pause and wait for approval
    QUEUE = "queue"  # Queue requests, continue other work
    AUTOMATIC = "automatic"  # Auto-approve with logging
    DISABLED = "disabled"  # No approval needed


class HumanFeedback(Enum):
    """Types of human feedback."""

    APPROVAL = "approval"
    REJECTION = "rejection"
    MODIFICATION = "modification"
    ESCALATION = "escalation"


class AuditEntry:
    """Audit trail entry."""

    def __init__(
        self,
        action: str,
        user: str,
        request_id: str,
        details: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ):
        self.action = action
        self.user = user
        self.request_id = request_id
        self.details = details
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action,
            "user": self.user,
            "request_id": self.request_id,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class HumanReviewWorkflow:
    """
    Manages human oversight and approval workflow.

    Supports multiple approval modes and maintains audit trail.
    """

    def __init__(
        self,
        mode: ApprovalMode = ApprovalMode.BLOCKING,
        audit_log_path: Optional[str] = None,
        auto_approve_low_risk: bool = True,
        notification_callback: Optional[Callable] = None
    ):
        """
        Initialize human review workflow.

        Args:
            mode: Approval workflow mode
            audit_log_path: Path to audit log file
            auto_approve_low_risk: Automatically approve low-risk requests
            notification_callback: Callback for sending notifications
        """
        self.mode = mode
        # Default from SafetyConfig.audit_log_path (AUDIT_LOG_PATH), not a fixed cwd file
        self.audit_log_path = audit_log_path or SafetyConfig().audit_log_path
        self.auto_approve_low_risk = auto_approve_low_risk
        self.notification_callback = notification_callback

        # Pending approval requests (for queue mode)
        self.pending_requests: Dict[str, ApprovalRequest] = {}

        # Approved/rejected requests cache
        self.processed_requests: Dict[str, ApprovalRequest] = {}

        # Audit trail
        self.audit_trail: List[AuditEntry] = []

        logger.info(
            f"HumanReviewWorkflow initialized (mode={mode.value}, "
            f"auto_approve_low_risk={auto_approve_low_risk})"
        )

    def request_approval(
        self,
        request: ApprovalRequest,
        user: str = "system"
    ) -> ApprovalRequest:
        """
        Request human approval for an operation.

        Args:
            request: Approval request
            user: User requesting approval

        Returns:
            ApprovalRequest with status updated

        Raises:
            RuntimeError: If approval is denied in blocking mode
        """
        # Log audit entry
        self._audit("approval_requested", user, request.request_id, {
            "operation": request.operation_type,
            "risk_level": request.risk_level.value
        })

        # Handle based on mode
        if self.mode == ApprovalMode.DISABLED:
            request.approve("automatic")
            return request

        if self.mode == ApprovalMode.AUTOMATIC:
            return self._auto_approve(request, user)

        if self.mode == ApprovalMode.BLOCKING:
            return self._blocking_approval(request, user)

        if self.mode == ApprovalMode.QUEUE:
            return self._queue_approval(request, user)

        raise ValueError(f"Unknown approval mode: {self.mode}")

    def _auto_approve(
        self,
        request: ApprovalRequest,
        user: str
    ) -> ApprovalRequest:
        """Automatically approve request with logging."""
        # Auto-approve low risk if enabled
        if self.auto_approve_low_risk and request.risk_level == RiskLevel.LOW:
            request.approve("automatic")
            self._audit("auto_approved", "automatic", request.request_id, {
                "reason": "low_risk"
            })
        else:
            # Still approve but log for review
            request.approve("automatic")
            self._audit("auto_approved", "automatic", request.request_id, {
                "risk_level": request.risk_level.value,
                "requires_review": True
            })

            # Send notification if callback provided
            if self.notification_callback:
                self.notification_callback(
                    f"Auto-approved {request.risk_level.value} risk operation: "
                    f"{request.operation_description[:100]}"
                )

        self.processed_requests[request.request_id] = request
        return request

    def _blocking_approval(
        self,
        request: ApprovalRequest,
        user: str
    ) -> ApprovalRequest:
        """Blocking approval - wait for user input."""
        # Check if should auto-approve
        if self.auto_approve_low_risk and request.risk_level == RiskLevel.LOW:
            return self._auto_approve(request, user)

        logger.info(
            f"Approval required for {request.operation_type} "
            f"(risk: {request.risk_level.value})"
        )

        # Display request details
        self._display_approval_request(request)

        # Get user input (CLI)
        decision = self._get_user_decision(request)

        if decision == "approve":
            request.approve(user)
            self._audit("approved", user, request.request_id, {})
            self.processed_requests[request.request_id] = request
            return request
        else:
            reason = input("Reason for rejection: ").strip() or "User rejected"
            request.reject(user, reason)
            self._audit("rejected", user, request.request_id, {"reason": reason})
            self.processed_requests[request.request_id] = request
            raise RuntimeError(f"Approval denied: {reason}")

    def _queue_approval(
        self,
        request: ApprovalRequest,
        user: str
    ) -> ApprovalRequest:
        """Queue approval - add to queue and continue."""
        # Check if should auto-approve
        if self.auto_approve_low_risk and request.risk_level == RiskLevel.LOW:
            return self._auto_approve(request, user)

        # Add to pending queue
        self.pending_requests[request.request_id] = request

        self._audit("queued", user, request.request_id, {})

        logger.info(
            f"Approval request queued: {request.request_id} "
            f"({len(self.pending_requests)} pending)"
        )

        # Send notification if callback provided
        if self.notification_callback:
            self.notification_callback(
                f"Approval required: {request.operation_description[:100]}"
            )

        return request

    def _display_approval_request(self, request: ApprovalRequest):
        """Display approval request to user (CLI)."""
        print("\n" + "=" * 70)
        print("APPROVAL REQUIRED")
        print("=" * 70)
        print(f"Operation: {request.operation_type}")
        print(f"Risk Level: {request.risk_level.value}")
        print(f"Reason: {request.reason_for_approval}")
        print(f"\nDescription:")
        print(request.operation_description)
        print("=" * 70)

    def _get_user_decision(self, request: ApprovalRequest) -> str:
        """Get user decision via CLI input."""
        while True:
            response = input("\nApprove this operation? (yes/no): ").strip().lower()
            if response in ["yes", "y", "approve"]:
                return "approve"
            elif response in ["no", "n", "reject", "deny"]:
                return "reject"
            else:
                print("Please enter 'yes' or 'no'")

    def process_pending_requests(
        self,
        batch_size: int = 10
    ) -> List[ApprovalRequest]:
        """
        Process pending approval requests in queue mode.

        Args:
            batch_size: Maximum number of requests to process

        Returns:
            List of processed requests
        """
        if self.mode != ApprovalMode.QUEUE:
            logger.warning("process_pending_requests only applies to queue mode")
            return []

        processed = []
        requests_to_process = list(self.pending_requests.values())[:batch_size]

        for request in requests_to_process:
            # Check if expired
            if request.is_expired():
                request.mark_expired()
                self._audit("expired", "system", request.request_id, {})
                self.processed_requests[request.request_id] = request
                del self.pending_requests[request.request_id]
                processed.append(request)
                continue

            # Display and get decision
            self._display_approval_request(request)
            decision = self._get_user_decision(request)

            if decision == "approve":
                request.approve("user")
                self._audit("approved", "user", request.request_id, {})
            else:
                reason = input("Reason for rejection: ").strip() or "User rejected"
                request.reject("user", reason)
                self._audit("rejected", "user", request.request_id, {"reason": reason})

            # Move to processed
            self.processed_requests[request.request_id] = request
            del self.pending_requests[request.request_id]
            processed.append(request)

        return processed

    def override_decision(
        self,
        request_id: str,
        user: str,
        new_status: ApprovalStatus,
        reason: str
    ):
        """
        Override a previous approval decision.

        Args:
            request_id: Request ID to override
            user: User making override
            new_status: New approval status
            reason: Reason for override
        """
        # Find request
        request = None
        if request_id in self.processed_requests:
            request = self.processed_requests[request_id]
        elif request_id in self.pending_requests:
            request = self.pending_requests[request_id]
        else:
            raise ValueError(f"Request {request_id} not found")

        old_status = request.status

        # Update status
        if new_status == ApprovalStatus.APPROVED:
            request.approve(user)
        elif new_status == ApprovalStatus.REJECTED:
            request.reject(user, reason)

        # Log override
        self._audit("override", user, request_id, {
            "old_status": old_status.value,
            "new_status": new_status.value,
            "reason": reason
        })

        logger.warning(
            f"Approval decision overridden by {user}: {request_id} "
            f"({old_status.value} -> {new_status.value})"
        )

    def get_pending_count(self) -> int:
        """Get number of pending approval requests."""
        return len(self.pending_requests)

    def get_pending_requests(
        self,
        risk_level: Optional[RiskLevel] = None
    ) -> List[ApprovalRequest]:
        """
        Get pending approval requests.

        Args:
            risk_level: Optional filter by risk level

        Returns:
            List of pending requests
        """
        requests = list(self.pending_requests.values())

        if risk_level:
            requests = [r for r in requests if r.risk_level == risk_level]

        return requests

    def _audit(
        self,
        action: str,
        user: str,
        request_id: str,
        details: Dict[str, Any]
    ):
        """Add entry to audit trail."""
        entry = AuditEntry(
            action=action,
            user=user,
            request_id=request_id,
            details=details
        )

        self.audit_trail.append(entry)

        # Write to file
        try:
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(entry.to_dict(), default=str) + '\n')
        except Exception as e:
            logger.error(f"Error writing to audit log: {e}")

    def get_audit_trail(
        self,
        request_id: Optional[str] = None,
        user: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditEntry]:
        """
        Get audit trail entries.

        Args:
            request_id: Optional filter by request ID
            user: Optional filter by user
            limit: Maximum entries to return

        Returns:
            List of audit entries
        """
        entries = self.audit_trail

        # Filter
        if request_id:
            entries = [e for e in entries if e.request_id == request_id]
        if user:
            entries = [e for e in entries if e.user == user]

        # Return most recent
        return entries[-limit:]

    def get_approval_stats(self) -> Dict[str, Any]:
        """
        Get approval statistics.

        Returns:
            Dictionary with stats
        """
        total_processed = len(self.processed_requests)
        approved = sum(
            1 for r in self.processed_requests.values()
            if r.status == ApprovalStatus.APPROVED
        )
        rejected = sum(
            1 for r in self.processed_requests.values()
            if r.status == ApprovalStatus.REJECTED
        )
        expired = sum(
            1 for r in self.processed_requests.values()
            if r.status == ApprovalStatus.EXPIRED
        )

        return {
            "mode": self.mode.value,
            "pending": len(self.pending_requests),
            "total_processed": total_processed,
            "approved": approved,
            "rejected": rejected,
            "expired": expired,
            "approval_rate": approved / total_processed if total_processed > 0 else 0.0
        }

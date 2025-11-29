"""
Alerting system for Kosmos AI Scientist.

Provides alert definitions and notification handlers for critical events.
"""

import logging
import os
import json
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class Alert:
    """Represents a system alert."""

    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: AlertStatus = AlertStatus.ACTIVE
    details: Dict[str, Any] = field(default_factory=dict)
    alert_id: Optional[str] = None

    def __post_init__(self):
        """Generate alert ID after initialization."""
        if not self.alert_id:
            self.alert_id = f"{self.name}_{int(self.timestamp.timestamp())}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "details": self.details
        }


@dataclass
class AlertRule:
    """Defines conditions for triggering alerts."""

    name: str
    condition: Callable[[], bool]
    severity: AlertSeverity
    message_template: str
    cooldown_seconds: int = 300  # 5 minutes default cooldown
    last_triggered: Optional[datetime] = None

    def should_trigger(self) -> bool:
        """
        Check if alert should be triggered.

        Returns:
            True if alert should trigger
        """
        # Check cooldown
        if self.last_triggered:
            cooldown_end = self.last_triggered + timedelta(seconds=self.cooldown_seconds)
            if datetime.utcnow() < cooldown_end:
                return False

        # Check condition
        try:
            return self.condition()
        except Exception as e:
            logger.error(f"Error evaluating alert condition for {self.name}: {e}")
            return False

    def trigger(self, details: Optional[Dict[str, Any]] = None) -> Alert:
        """
        Trigger the alert.

        Args:
            details: Additional context for the alert

        Returns:
            Alert instance
        """
        self.last_triggered = datetime.utcnow()
        return Alert(
            name=self.name,
            severity=self.severity,
            message=self.message_template,
            details=details or {}
        )


class AlertManager:
    """
    Manages alerts and notifications for Kosmos.

    Handles:
    - Alert rule evaluation
    - Alert triggering and resolution
    - Notification routing
    - Alert history
    """

    def __init__(self):
        """Initialize alert manager."""
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: List[Callable[[Alert], None]] = []

        # Initialize default alert rules
        self._initialize_default_rules()

        logger.info("Alert manager initialized")

    def _initialize_default_rules(self):
        """Initialize default alert rules."""

        # Database connection failure
        self.alert_rules.append(AlertRule(
            name="database_connection_failed",
            condition=self._check_database_connection,
            severity=AlertSeverity.CRITICAL,
            message_template="Database connection failed",
            cooldown_seconds=60
        ))

        # High API failure rate
        self.alert_rules.append(AlertRule(
            name="high_api_failure_rate",
            condition=self._check_api_failure_rate,
            severity=AlertSeverity.ERROR,
            message_template="High API failure rate detected",
            cooldown_seconds=300
        ))

        # API rate limit approaching
        self.alert_rules.append(AlertRule(
            name="api_rate_limit_warning",
            condition=self._check_api_rate_limit,
            severity=AlertSeverity.WARNING,
            message_template="Approaching API rate limit",
            cooldown_seconds=600
        ))

        # High memory usage
        self.alert_rules.append(AlertRule(
            name="high_memory_usage",
            condition=self._check_memory_usage,
            severity=AlertSeverity.WARNING,
            message_template="High memory usage detected",
            cooldown_seconds=300
        ))

        # High disk usage
        self.alert_rules.append(AlertRule(
            name="high_disk_usage",
            condition=self._check_disk_usage,
            severity=AlertSeverity.WARNING,
            message_template="High disk usage detected",
            cooldown_seconds=600
        ))

        # Experiment failure rate
        self.alert_rules.append(AlertRule(
            name="high_experiment_failure_rate",
            condition=self._check_experiment_failure_rate,
            severity=AlertSeverity.ERROR,
            message_template="High experiment failure rate",
            cooldown_seconds=300
        ))

        # Cache unavailable
        self.alert_rules.append(AlertRule(
            name="cache_unavailable",
            condition=self._check_cache_availability,
            severity=AlertSeverity.WARNING,
            message_template="Cache service unavailable",
            cooldown_seconds=120
        ))

    def add_alert_rule(self, rule: AlertRule):
        """Add a custom alert rule."""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")

    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """
        Add a notification handler.

        Args:
            handler: Function that takes an Alert and sends notification
        """
        self.notification_handlers.append(handler)
        logger.info(f"Added notification handler: {handler.__name__}")

    def evaluate_rules(self):
        """Evaluate all alert rules and trigger alerts if needed."""
        for rule in self.alert_rules:
            if rule.should_trigger():
                alert = rule.trigger()
                self._handle_alert(alert)

    def _handle_alert(self, alert: Alert):
        """
        Handle a triggered alert.

        Args:
            alert: Alert to handle
        """
        # Store in active alerts
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)

        # Trim history if too long
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]

        logger.warning(f"Alert triggered: {alert.name} - {alert.message}")

        # Send notifications
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")

    def resolve_alert(self, alert_id: str):
        """Mark alert as resolved."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            del self.active_alerts[alert_id]
            logger.info(f"Alert resolved: {alert.name}")

    def acknowledge_alert(self, alert_id: str):
        """Mark alert as acknowledged."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            logger.info(f"Alert acknowledged: {alert.name}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return self.alert_history[-limit:]

    # Alert condition checkers
    def _check_database_connection(self) -> bool:
        """Check if database connection is failing."""
        try:
            from kosmos.api.health import get_health_checker
            health_checker = get_health_checker()
            db_status = health_checker._check_database()
            return db_status["status"] != "healthy"
        except Exception as e:
            logger.warning("Database connection check error: %s", e, exc_info=True)
            return False

    def _check_api_failure_rate(self) -> bool:
        """Check if API failure rate is high."""
        # This would integrate with metrics collector
        # For now, return False (placeholder)
        return False

    def _check_api_rate_limit(self) -> bool:
        """Check if approaching API rate limit."""
        # This would integrate with metrics/rate limiter
        # For now, return False (placeholder)
        return False

    def _check_memory_usage(self) -> bool:
        """Check if memory usage is high."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            # Alert if memory usage > 85%
            return memory.percent > 85
        except Exception as e:
            logger.warning("Memory usage check error: %s", e, exc_info=True)
            return False

    def _check_disk_usage(self) -> bool:
        """Check if disk usage is high."""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            # Alert if disk usage > 90%
            return disk.percent > 90
        except Exception as e:
            logger.warning("Disk usage check error: %s", e, exc_info=True)
            return False

    def _check_experiment_failure_rate(self) -> bool:
        """Check if experiment failure rate is high."""
        # This would integrate with metrics collector
        # For now, return False (placeholder)
        return False

    def _check_cache_availability(self) -> bool:
        """Check if cache is unavailable."""
        try:
            from kosmos.api.health import get_health_checker
            health_checker = get_health_checker()
            cache_status = health_checker._check_cache()
            return cache_status["status"] == "unhealthy"
        except Exception as e:
            logger.warning("Cache availability check error: %s", e, exc_info=True)
            return False


# Notification handlers
def log_notification_handler(alert: Alert):
    """Log alert to logger (always enabled)."""
    log_func = {
        AlertSeverity.INFO: logger.info,
        AlertSeverity.WARNING: logger.warning,
        AlertSeverity.ERROR: logger.error,
        AlertSeverity.CRITICAL: logger.critical
    }.get(alert.severity, logger.info)

    log_func(f"ALERT [{alert.severity.value.upper()}]: {alert.message} | {alert.details}")


def email_notification_handler(alert: Alert):
    """
    Send email notification (requires configuration).

    Environment variables:
    - ALERT_EMAIL_ENABLED: true/false
    - ALERT_EMAIL_TO: recipient email
    - ALERT_EMAIL_FROM: sender email
    - SMTP_HOST: SMTP server
    - SMTP_PORT: SMTP port
    - SMTP_USER: SMTP username
    - SMTP_PASSWORD: SMTP password
    """
    if not os.getenv("ALERT_EMAIL_ENABLED", "false").lower() == "true":
        return

    try:
        import smtplib
        from email.message import EmailMessage

        msg = EmailMessage()
        msg['Subject'] = f"[{alert.severity.value.upper()}] Kosmos Alert: {alert.name}"
        msg['From'] = os.getenv("ALERT_EMAIL_FROM", "alerts@kosmos.ai")
        msg['To'] = os.getenv("ALERT_EMAIL_TO", "admin@example.com")

        body = f"""
Kosmos Alert

Severity: {alert.severity.value.upper()}
Alert: {alert.name}
Message: {alert.message}
Timestamp: {alert.timestamp.isoformat()}

Details:
{json.dumps(alert.details, indent=2)}

Alert ID: {alert.alert_id}
"""
        msg.set_content(body)

        # Send email
        smtp_host = os.getenv("SMTP_HOST", "localhost")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")

        with smtplib.SMTP(smtp_host, smtp_port) as smtp:
            smtp.starttls()
            if smtp_user and smtp_password:
                smtp.login(smtp_user, smtp_password)
            smtp.send_message(msg)

        logger.info(f"Email notification sent for alert: {alert.name}")

    except Exception as e:
        logger.error(f"Failed to send email notification: {e}")


def slack_notification_handler(alert: Alert):
    """
    Send Slack notification (requires configuration).

    Environment variables:
    - ALERT_SLACK_ENABLED: true/false
    - SLACK_WEBHOOK_URL: Slack webhook URL
    """
    if not os.getenv("ALERT_SLACK_ENABLED", "false").lower() == "true":
        return

    try:
        import requests

        webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        if not webhook_url:
            logger.warning("SLACK_WEBHOOK_URL not configured")
            return

        # Color coding by severity
        color = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9900",
            AlertSeverity.ERROR: "#ff0000",
            AlertSeverity.CRITICAL: "#8b0000"
        }.get(alert.severity, "#808080")

        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": f"{alert.severity.value.upper()}: {alert.name}",
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Timestamp",
                            "value": alert.timestamp.isoformat(),
                            "short": True
                        },
                        {
                            "title": "Alert ID",
                            "value": alert.alert_id,
                            "short": True
                        }
                    ],
                    "footer": "Kosmos Alert System",
                    "ts": int(alert.timestamp.timestamp())
                }
            ]
        }

        # Add details if present
        if alert.details:
            payload["attachments"][0]["fields"].append({
                "title": "Details",
                "value": f"```{json.dumps(alert.details, indent=2)}```",
                "short": False
            })

        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()

        logger.info(f"Slack notification sent for alert: {alert.name}")

    except Exception as e:
        logger.error(f"Failed to send Slack notification: {e}")


def pagerduty_notification_handler(alert: Alert):
    """
    Send PagerDuty notification (requires configuration).

    Environment variables:
    - ALERT_PAGERDUTY_ENABLED: true/false
    - PAGERDUTY_INTEGRATION_KEY: PagerDuty integration key
    """
    if not os.getenv("ALERT_PAGERDUTY_ENABLED", "false").lower() == "true":
        return

    # Only trigger PagerDuty for ERROR and CRITICAL
    if alert.severity not in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
        return

    try:
        import requests

        integration_key = os.getenv("PAGERDUTY_INTEGRATION_KEY")
        if not integration_key:
            logger.warning("PAGERDUTY_INTEGRATION_KEY not configured")
            return

        payload = {
            "routing_key": integration_key,
            "event_action": "trigger",
            "dedup_key": alert.alert_id,
            "payload": {
                "summary": f"{alert.severity.value.upper()}: {alert.message}",
                "severity": alert.severity.value,
                "source": "kosmos-ai-scientist",
                "timestamp": alert.timestamp.isoformat(),
                "custom_details": alert.details
            }
        }

        response = requests.post(
            "https://events.pagerduty.com/v2/enqueue",
            json=payload,
            timeout=10
        )
        response.raise_for_status()

        logger.info(f"PagerDuty notification sent for alert: {alert.name}")

    except Exception as e:
        logger.error(f"Failed to send PagerDuty notification: {e}")


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """
    Get or create the global alert manager.

    Returns:
        AlertManager instance
    """
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()

        # Register default notification handlers
        _alert_manager.add_notification_handler(log_notification_handler)

        # Register optional handlers if enabled
        if os.getenv("ALERT_EMAIL_ENABLED", "false").lower() == "true":
            _alert_manager.add_notification_handler(email_notification_handler)

        if os.getenv("ALERT_SLACK_ENABLED", "false").lower() == "true":
            _alert_manager.add_notification_handler(slack_notification_handler)

        if os.getenv("ALERT_PAGERDUTY_ENABLED", "false").lower() == "true":
            _alert_manager.add_notification_handler(pagerduty_notification_handler)

    return _alert_manager


def evaluate_alerts():
    """Evaluate all alert rules."""
    get_alert_manager().evaluate_rules()


def get_active_alerts() -> List[Dict[str, Any]]:
    """Get all active alerts."""
    alerts = get_alert_manager().get_active_alerts()
    return [alert.to_dict() for alert in alerts]


def get_alert_history(limit: int = 100) -> List[Dict[str, Any]]:
    """Get alert history."""
    alerts = get_alert_manager().get_alert_history(limit)
    return [alert.to_dict() for alert in alerts]

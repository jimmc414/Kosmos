"""
Structured logging system with JSON and text output support.

Provides:
- JSON structured logging for machine parsing
- Human-readable text logging for development
- Experiment run tracking
- Log aggregation and search
- Debug mode
"""

import logging
import logging.handlers
import json
import sys
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from enum import Enum


class LogFormat(str, Enum):
    """Log format options."""
    JSON = "json"
    TEXT = "text"


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.

    Produces machine-parseable JSON logs.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record

        Returns:
            str: JSON formatted log
        """
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add workflow context fields
        if hasattr(record, "workflow_id"):
            log_data["workflow_id"] = record.workflow_id
        if hasattr(record, "cycle"):
            log_data["cycle"] = record.cycle
        if hasattr(record, "task_id"):
            log_data["task_id"] = record.task_id

        # Add extra fields
        if hasattr(record, "extra"):
            log_data["extra"] = record.extra

        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """
    Human-readable text formatter with colors (for terminals).

    Produces readable logs for development.
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",      # Reset
    }

    def __init__(self, use_colors: bool = True):
        """
        Initialize text formatter.

        Args:
            use_colors: Whether to use ANSI colors
        """
        super().__init__(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with optional colors.

        Args:
            record: Log record

        Returns:
            str: Formatted log message
        """
        if self.use_colors and sys.stdout.isatty():
            color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
            reset = self.COLORS["RESET"]
            record.levelname = f"{color}{record.levelname}{reset}"

        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_format: LogFormat = LogFormat.JSON,
    log_file: Optional[str] = None,
    debug_mode: bool = False
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format (json or text)
        log_file: Optional log file path
        debug_mode: Enable debug mode with verbose output

    Returns:
        logging.Logger: Root logger

    Example:
        ```python
        from kosmos.core.logging import setup_logging, LogFormat

        # JSON logging to file
        setup_logging(
            level="INFO",
            log_format=LogFormat.JSON,
            log_file="logs/kosmos.log"
        )

        # Text logging to stdout for development
        setup_logging(
            level="DEBUG",
            log_format=LogFormat.TEXT,
            debug_mode=True
        )
        ```
    """
    # Get root logger
    root_logger = logging.getLogger()

    # Set level
    if debug_mode:
        level = "DEBUG"
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers = []

    # Create formatter
    if log_format == LogFormat.JSON:
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter(use_colors=True)

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if log_file specified)
    if log_file:
        # Create log directory
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Rotating file handler (10MB max, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Log startup
    root_logger.info("Logging initialized", extra={
        "level": level,
        "format": log_format,
        "file": log_file,
        "debug_mode": debug_mode
    })

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get logger for specific module.

    Args:
        name: Logger name (usually __name__)

    Returns:
        logging.Logger: Logger instance

    Example:
        ```python
        from kosmos.core.logging import get_logger

        logger = get_logger(__name__)
        logger.info("Starting hypothesis generation")
        logger.debug("Processing 10 papers")
        ```
    """
    return logging.getLogger(name)


class ExperimentLogger:
    """
    Logger for experiment runs with structured output.

    Tracks experiment progress and results.

    Example:
        ```python
        exp_logger = ExperimentLogger(experiment_id="exp-123")
        exp_logger.start()
        exp_logger.log_hypothesis("Dark matter hypothesis")
        exp_logger.log_result({"accuracy": 0.95})
        exp_logger.end(status="success")
        ```
    """

    def __init__(self, experiment_id: str, logger_name: str = "kosmos.experiment"):
        """
        Initialize experiment logger.

        Args:
            experiment_id: Unique experiment ID
            logger_name: Logger name to use
        """
        self.experiment_id = experiment_id
        self.logger = logging.getLogger(logger_name)
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.events: list[Dict[str, Any]] = []

    def start(self):
        """Log experiment start."""
        self.start_time = datetime.utcnow()
        self.logger.info(f"Experiment {self.experiment_id} started", extra={
            "experiment_id": self.experiment_id,
            "event": "start",
            "timestamp": self.start_time.isoformat()
        })

    def log_hypothesis(self, hypothesis: str):
        """Log hypothesis."""
        event = {
            "event": "hypothesis",
            "hypothesis": hypothesis,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.events.append(event)
        self.logger.info(f"Hypothesis: {hypothesis}", extra={
            "experiment_id": self.experiment_id,
            **event
        })

    def log_experiment_design(self, design: Dict[str, Any]):
        """Log experiment design."""
        event = {
            "event": "experiment_design",
            "design": design,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.events.append(event)
        self.logger.info("Experiment design created", extra={
            "experiment_id": self.experiment_id,
            **event
        })

    def log_execution_start(self):
        """Log execution start."""
        event = {
            "event": "execution_start",
            "timestamp": datetime.utcnow().isoformat()
        }
        self.events.append(event)
        self.logger.info("Execution started", extra={
            "experiment_id": self.experiment_id,
            **event
        })

    def log_result(self, result: Dict[str, Any]):
        """Log experiment result."""
        event = {
            "event": "result",
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.events.append(event)
        self.logger.info("Result obtained", extra={
            "experiment_id": self.experiment_id,
            **event
        })

    def log_error(self, error: str):
        """Log error."""
        event = {
            "event": "error",
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.events.append(event)
        self.logger.error(f"Error: {error}", extra={
            "experiment_id": self.experiment_id,
            **event
        })

    def end(self, status: str = "success"):
        """
        Log experiment end.

        Args:
            status: Final status (success, failure, error)
        """
        self.end_time = datetime.utcnow()
        duration = (self.end_time - self.start_time).total_seconds() if self.start_time else 0

        self.logger.info(f"Experiment {self.experiment_id} ended: {status}", extra={
            "experiment_id": self.experiment_id,
            "event": "end",
            "status": status,
            "duration_seconds": duration,
            "timestamp": self.end_time.isoformat(),
            "total_events": len(self.events)
        })

    def get_summary(self) -> Dict[str, Any]:
        """
        Get experiment summary.

        Returns:
            dict: Experiment summary
        """
        duration = (self.end_time - self.start_time).total_seconds() if (self.start_time and self.end_time) else 0

        return {
            "experiment_id": self.experiment_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": duration,
            "total_events": len(self.events),
            "events": self.events
        }


def configure_from_config():
    """
    Configure logging from Kosmos config.

    Example:
        ```python
        from kosmos.config import get_config
        from kosmos.core.logging import configure_from_config

        configure_from_config()
        ```
    """
    from kosmos.config import get_config

    config = get_config()
    setup_logging(
        level=config.logging.level,
        log_format=LogFormat(config.logging.format),
        log_file=config.logging.file,
        debug_mode=config.logging.debug_mode
    )

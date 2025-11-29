# Debug Mode Implementation Prompt

## Context

You have access to the Kosmos AI Scientist repository. A comprehensive debug mode implementation plan has been created at `DEBUG_MODE_IMPLEMENTATION_PLAN.md`. Your task is to implement the changes described in that plan.

## Prerequisites

Before starting, read these files to understand the current state:
1. `DEBUG_MODE_IMPLEMENTATION_PLAN.md` - The full implementation plan
2. `debug_mode_analysis.md` - Prior analysis and context
3. `kosmos/config.py` - Current configuration system
4. `kosmos/core/logging.py` - Current logging infrastructure
5. `kosmos/cli/main.py` - CLI entry point with existing flags

## Implementation Tasks

### Phase 1: Configuration Extension (Tier 1 - Start Here)

**Task 1.1: Extend LoggingConfig in `kosmos/config.py`**

Add these fields to the existing `LoggingConfig` class (around line 330):

```python
# Add after existing fields (level, format, file, debug_mode)
debug_level: Literal[0, 1, 2, 3] = Field(
    default=0,
    description="Debug verbosity: 0=off, 1=critical path, 2=full trace, 3=data dumps",
    alias="DEBUG_LEVEL"
)

debug_modules: Annotated[Optional[List[str]], BeforeValidator(parse_comma_separated)] = Field(
    default=None,
    description="Modules to debug (None=all when debug_mode=True)",
    alias="DEBUG_MODULES"
)

log_llm_calls: bool = Field(
    default=False,
    description="Log LLM request/response summaries",
    alias="LOG_LLM_CALLS"
)

log_agent_messages: bool = Field(
    default=False,
    description="Log inter-agent message routing",
    alias="LOG_AGENT_MESSAGES"
)

log_workflow_transitions: bool = Field(
    default=False,
    description="Log state machine transitions with timing",
    alias="LOG_WORKFLOW_TRANSITIONS"
)

stage_tracking_enabled: bool = Field(
    default=False,
    description="Enable real-time stage tracking output",
    alias="STAGE_TRACKING_ENABLED"
)

stage_tracking_file: str = Field(
    default="logs/stages.jsonl",
    description="Stage tracking output file",
    alias="STAGE_TRACKING_FILE"
)
```

**Task 1.2: Extend CLI flags in `kosmos/cli/main.py`**

Update the `main()` callback function (around line 69) to add new options:

```python
@app.callback()
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
    trace: bool = typer.Option(False, "--trace", help="Enable trace-level logging (maximum verbosity)"),
    debug_level: int = typer.Option(0, "--debug-level", "-dl", help="Debug level 0-3"),
    debug_modules: Optional[str] = typer.Option(None, "--debug-modules", help="Comma-separated modules to debug"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress non-essential output"),
):
```

Also update the `setup_logging()` function to handle the new `trace` flag:
```python
def setup_logging(verbose: bool = False, debug: bool = False, trace: bool = False):
    """Configure logging for CLI."""
    from kosmos.cli.utils import get_log_dir

    log_dir = get_log_dir()
    log_file = log_dir / "kosmos.log"

    if trace:
        level = logging.DEBUG
        # Also enable all debug toggles when trace is set
    elif debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    # ... rest of function
```

---

### Phase 2: Create Stage Tracker

**Task 2.1: Create new file `kosmos/core/stage_tracker.py`**

```python
"""
Real-time stage tracking for debug observability.

Provides context managers and utilities for tracking multi-step research processes.
"""

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Literal, List

logger = logging.getLogger(__name__)


@dataclass
class StageEvent:
    """Represents a single stage event for tracking."""
    timestamp: str
    process_id: str
    stage: str
    status: Literal["started", "completed", "failed", "skipped"]
    duration_ms: Optional[int] = None
    iteration: int = 0
    substage: Optional[str] = None
    parent_stage: Optional[str] = None
    output_summary: Optional[str] = None
    error: Optional[Dict[str, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), default=str)


class StageTracker:
    """
    Context manager for tracking stage execution with timing.

    Usage:
        tracker = StageTracker(process_id="research_123")
        with tracker.track("GENERATE_HYPOTHESIS", hypothesis_count=5):
            # do work
            pass
    """

    def __init__(
        self,
        process_id: str,
        output_file: Optional[str] = None,
        emit_to_stdout: bool = False,
        enabled: bool = True
    ):
        self.process_id = process_id
        self.output_file = output_file or "logs/stages.jsonl"
        self.emit_to_stdout = emit_to_stdout
        self.enabled = enabled
        self._stage_stack: List[str] = []
        self.current_iteration = 0
        self._events: List[StageEvent] = []

        # Ensure output directory exists
        if self.enabled and self.output_file:
            Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)

    def set_iteration(self, iteration: int):
        """Update current iteration number."""
        self.current_iteration = iteration

    @contextmanager
    def track(self, stage: str, **metadata):
        """
        Track a stage with timing and status.

        Args:
            stage: Stage name (e.g., "GENERATE_HYPOTHESIS")
            **metadata: Additional metadata to include in event
        """
        if not self.enabled:
            yield None
            return

        start = time.time()
        event = StageEvent(
            timestamp=datetime.utcnow().isoformat() + "Z",
            process_id=self.process_id,
            stage=stage,
            status="started",
            iteration=self.current_iteration,
            parent_stage=self._stage_stack[-1] if self._stage_stack else None,
            metadata=metadata
        )

        self._emit(event)
        self._stage_stack.append(stage)

        try:
            yield event
            event.status = "completed"
            event.duration_ms = int((time.time() - start) * 1000)
        except Exception as e:
            event.status = "failed"
            event.duration_ms = int((time.time() - start) * 1000)
            event.error = {
                "type": type(e).__name__,
                "message": str(e)[:500]
            }
            raise
        finally:
            self._stage_stack.pop()
            self._emit(event)
            self._events.append(event)

    def log_substage(self, substage: str, parent_stage: str, **metadata):
        """Log a substage event without context manager."""
        if not self.enabled:
            return

        event = StageEvent(
            timestamp=datetime.utcnow().isoformat() + "Z",
            process_id=self.process_id,
            stage=parent_stage,
            substage=substage,
            status="completed",
            iteration=self.current_iteration,
            metadata=metadata
        )
        self._emit(event)

    def _emit(self, event: StageEvent):
        """Emit stage event to configured outputs."""
        event_json = event.to_json()

        if self.emit_to_stdout:
            print(f"[STAGE] {event_json}")

        if self.output_file:
            try:
                with open(self.output_file, "a") as f:
                    f.write(event_json + "\n")
            except Exception as e:
                logger.warning(f"Failed to write stage event: {e}")

    def get_events(self) -> List[StageEvent]:
        """Get all recorded events."""
        return self._events.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of tracked stages."""
        completed = [e for e in self._events if e.status == "completed"]
        failed = [e for e in self._events if e.status == "failed"]

        total_duration = sum(e.duration_ms or 0 for e in completed)

        return {
            "process_id": self.process_id,
            "total_stages": len(self._events),
            "completed": len(completed),
            "failed": len(failed),
            "total_duration_ms": total_duration,
            "iterations": self.current_iteration
        }


# Singleton instance
_tracker: Optional[StageTracker] = None


def get_stage_tracker(process_id: Optional[str] = None) -> StageTracker:
    """Get or create stage tracker singleton."""
    global _tracker

    if _tracker is None or (process_id and _tracker.process_id != process_id):
        # Check config for settings
        try:
            from kosmos.config import get_config
            config = get_config()
            _tracker = StageTracker(
                process_id=process_id or f"research_{int(time.time())}",
                output_file=config.logging.stage_tracking_file,
                enabled=config.logging.stage_tracking_enabled
            )
        except Exception:
            # Fallback if config not available
            _tracker = StageTracker(
                process_id=process_id or f"research_{int(time.time())}",
                enabled=False
            )

    return _tracker
```

---

### Phase 3: Add Critical Path Instrumentation

**Task 3.1: Instrument `kosmos/agents/research_director.py`**

Add imports at top:
```python
from kosmos.core.stage_tracker import get_stage_tracker
```

Modify `decide_next_action()` method (around line 1200):
```python
def decide_next_action(self) -> NextAction:
    """Decide what to do next based on current workflow state and research plan."""
    current_state = self.workflow.current_state

    # Enhanced debug logging
    logger.debug(
        "[DECISION] decide_next_action: state=%s, iteration=%d/%d, "
        "hypotheses=%d, untested=%d, experiments_queued=%d",
        current_state.value,
        self.research_plan.iteration_count,
        self.research_plan.max_iterations,
        len(self.research_plan.hypothesis_pool),
        len(self.research_plan.get_untested_hypotheses()),
        len(self.research_plan.experiment_queue)
    )

    # ... rest of existing logic
```

Modify `_execute_next_action()` method (around line 1266):
```python
def _execute_next_action(self, action: NextAction):
    """Execute the decided next action."""
    logger.info("[ACTION] Executing: %s", action.value)

    tracker = get_stage_tracker()
    with tracker.track(f"ACTION_{action.value}", action=action.value):
        # ... existing switch/case logic
```

**Task 3.2: Instrument `kosmos/cli/commands/run.py`**

Add at the start of the research loop (around line 247):
```python
# Add import at top
from kosmos.core.stage_tracker import get_stage_tracker

# In run_with_progress function, before while loop:
tracker = get_stage_tracker(process_id=f"research_{int(time.time())}")

while iteration < max_iterations:
    loop_iteration_start = time.time()

    # Update tracker
    tracker.set_iteration(iteration)

    with tracker.track("RESEARCH_ITERATION", iteration=iteration):
        # ... existing loop body

        # At end of loop iteration, add:
        loop_duration = time.time() - loop_iteration_start
        logger.info(
            "[ITER %d/%d] state=%s, hyps=%d, exps=%d, duration=%.2fs",
            iteration, max_iterations,
            status.get('workflow_state'),
            status.get('hypothesis_pool_size', 0),
            status.get('experiments_completed', 0),
            loop_duration
        )
```

**Task 3.3: Instrument `kosmos/core/workflow.py`**

Modify `transition_to()` method (around line 260):
```python
def transition_to(
    self,
    target_state: WorkflowState,
    action: str = "",
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """Transition to a new state with enhanced logging."""
    if not self.can_transition_to(target_state):
        raise ValueError(...)

    # Calculate time in previous state
    time_in_state = 0.0
    if self.transition_history:
        last_transition = self.transition_history[-1]
        time_in_state = (datetime.utcnow() - last_transition.timestamp).total_seconds()

    # Enhanced transition logging
    logger.debug(
        "[WORKFLOW] Transition: %s -> %s (was in %s for %.2fs) action='%s'",
        self.current_state.value,
        target_state.value,
        self.current_state.value,
        time_in_state,
        action
    )

    # ... rest of existing logic
```

---

### Phase 4: Add LLM Call Instrumentation

**Task 4.1: Instrument `kosmos/core/providers/anthropic.py`**

Modify the `generate()` method (around line 124):
```python
def generate(
    self,
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    stop_sequences: Optional[List[str]] = None,
    **kwargs
) -> LLMResponse:
    """Generate text from Claude with enhanced logging."""
    try:
        # Check if LLM call logging is enabled
        from kosmos.config import get_config
        config = get_config()
        log_llm = config.logging.log_llm_calls

        # Pre-call logging
        if log_llm:
            logger.debug(
                "[LLM] Request: model=%s, prompt_len=%d, system_len=%d, "
                "max_tokens=%d, temp=%.2f",
                self.model,
                len(prompt),
                len(system or ""),
                max_tokens,
                temperature
            )

        start_time = time.time()

        # ... existing API call logic ...

        # Post-call logging
        latency_ms = int((time.time() - start_time) * 1000)
        if log_llm:
            logger.debug(
                "[LLM] Response: model=%s, in_tokens=%d, out_tokens=%d, "
                "latency=%dms, finish=%s",
                selected_model,
                usage_stats.input_tokens,
                usage_stats.output_tokens,
                latency_ms,
                response.stop_reason if hasattr(response, 'stop_reason') else "unknown"
            )

        # ... rest of method
```

Apply similar instrumentation to:
- `kosmos/core/providers/openai.py`
- `kosmos/core/providers/litellm_provider.py`

---

### Phase 5: Fix Error Handling

**Task 5.1: Fix bare except clauses in `kosmos/monitoring/alerts.py`**

Find and replace bare `except:` with `except Exception:` and add logging:
```python
# Before:
except:
    pass

# After:
except Exception as e:
    logger.warning("Alert handler error: %s", e, exc_info=True)
```

**Task 5.2: Add error context to `kosmos/agents/research_director.py`**

In the database initialization block (around line 104):
```python
# Before:
except RuntimeError:
    # Database already initialized
    pass

# After:
except RuntimeError as e:
    if "already initialized" not in str(e).lower():
        logger.warning("Database init RuntimeError: %s", e)
```

---

## Testing the Implementation

After implementing each phase, verify:

1. **Configuration loads correctly**:
```bash
python -c "from kosmos.config import get_config; c = get_config(); print(c.logging.debug_level)"
```

2. **CLI flags work**:
```bash
kosmos --help  # Should show new flags
kosmos --trace doctor  # Should produce verbose output
```

3. **Stage tracking produces output**:
```bash
kosmos run --trace "Test question" --max-iterations 1
cat logs/stages.jsonl  # Should have stage events
```

4. **Debug logs appear**:
```bash
LOG_LEVEL=DEBUG kosmos run "Test question" --max-iterations 1 2>&1 | grep "\[DECISION\]\|\[ACTION\]\|\[LLM\]"
```

---

## Commit Guidelines

Make separate commits for each phase:
1. "Add debug configuration fields to LoggingConfig and CLI"
2. "Add StageTracker for real-time observability"
3. "Instrument critical path: research_director, run, workflow"
4. "Add LLM call logging to providers"
5. "Fix bare except clauses and add error context"

Do NOT include "Co-Authored-By: Claude" in commit messages.

---

## Success Criteria

Implementation is complete when:
- [ ] `--trace` flag enables maximum debug output
- [ ] `--debug-level 1|2|3` provides tiered verbosity
- [ ] Stage tracking writes to `logs/stages.jsonl`
- [ ] Research loop iterations show timing and state
- [ ] LLM calls log token counts and latency
- [ ] No more bare `except:` clauses
- [ ] All tests pass: `pytest tests/`

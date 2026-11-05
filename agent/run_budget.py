"""Wall-clock run budget helpers.

``agent.run_budget_seconds`` has two jobs: the loop nudges the model to wrap up at 80%
and caps provider timeouts to what is left. Neither bounds the turn, so a slow-but-busy
run (180 calls at ~86-220s each) outlives any sane cap. ``run_budget_expired`` is the
hard stop the loop and the finalizer both read.
"""

from __future__ import annotations

import time
from typing import Any, Optional


def run_budget_elapsed(agent: Any) -> Optional[float]:
    """Seconds since the run budget started, or ``None`` when no budget is armed."""
    if not getattr(agent, "run_budget_seconds", None):
        return None
    started = getattr(agent, "_run_budget_started_at", None)
    if not started:
        return None
    try:
        return time.time() - float(started)
    except (TypeError, ValueError):
        return None


def run_budget_expired(agent: Any) -> bool:
    """True once the wall clock has passed ``agent.run_budget_seconds``."""
    elapsed = run_budget_elapsed(agent)
    if elapsed is None:
        return False
    try:
        return elapsed >= float(agent.run_budget_seconds)
    except (TypeError, ValueError):
        return False


def run_budget_exit_reason(agent: Any) -> str:
    """``run_budget_exhausted(<elapsed>s/<budget>s)`` for the ledger and the log line."""
    elapsed = run_budget_elapsed(agent) or 0.0
    return f"run_budget_exhausted({int(elapsed)}s/{int(float(agent.run_budget_seconds))}s)"

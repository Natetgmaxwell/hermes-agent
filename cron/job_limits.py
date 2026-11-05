"""Per-job run limits and quiet hours for cron jobs.

A cron job may cap its own run in the two dimensions that actually bound a scheduled
agent: the number of provider calls (``max_turns``) and the wall clock
(``max_runtime_seconds``). Both are per-job overrides of the global
``agent.max_turns`` / ``agent.run_budget_seconds``; an absent key falls back to the
global setting, so a job that sets neither behaves exactly as before.

The wall-clock cap exists because the inactivity watchdog cannot bound a run: a job
making a 180-call turn at ~86-220s per call is never *idle* for 600s, so it ran for
4h19m without tripping anything (cron job ``ae9c87cd6272``, 2026-10-28).

``quiet_hours`` suppresses failure *notices* inside a local-time window (wrapping
midnight). The failure itself is still recorded — job ``last_failure``, the incident
ledger and the execution ledger all keep it — so the morning briefing path still sees
what broke; only the page is held. ``off`` (or an empty value) disables the window for
one job, so a job that must alert at night can opt out.
"""

from __future__ import annotations

import re
from datetime import datetime, time
from typing import Any, Callable, Optional

# Opt-in, deliberately. The motivating incident (cron job ae9c87cd6272, 2026-10-28) wants
# 22:00-07:00, but a code-level always-on window makes every failure-delivery assertion in
# the suite depend on the wall clock — 35 tests across four files start failing between
# 22:00 and 07:00 UTC. So the window is set by the operator, globally
# (``cron.quiet_hours``) or per job; unset means "alert whenever it fails".
DEFAULT_QUIET_HOURS = ""

# A cap of 0 or less is meaningless (a run with no provider call cannot do anything);
# reject it at create time rather than storing a job that can never finish.
_MAX_CAP = 1_000_000

_QUIET_DISABLED = frozenset({"", "off", "none", "false", "disabled", "no"})
_CLOCK = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


def _normalize_cap(value: Any, field: str, *, unit: str) -> Optional[int]:
    """Shared ``max_turns``/``max_runtime_seconds`` validation: None/empty clears, else a
    positive whole number."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):  # bool is an int subclass; ``--max-turns True`` is a typo
        raise ValueError(f"{field} must be a whole number of {unit}, not a boolean.")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = float(text) if "." in text else int(text)
        except ValueError:
            raise ValueError(f"{field} must be a whole number of {unit}, got {value!r}.") from None
    if isinstance(value, float):
        if value != int(value):
            raise ValueError(f"{field} must be a whole number of {unit}, got {value!r}.")
        value = int(value)
    if not isinstance(value, int):
        raise ValueError(f"{field} must be a whole number of {unit}, got {type(value).__name__}.")
    if value <= 0 or value > _MAX_CAP:
        raise ValueError(f"{field} must be between 1 and {_MAX_CAP} {unit}, got {value}.")
    return value


def normalize_max_turns(value: Any) -> Optional[int]:
    """Per-job provider-call cap (``None`` = inherit the global turn limit)."""
    return _normalize_cap(value, "max_turns", unit="turns")


def normalize_max_runtime_seconds(value: Any) -> Optional[int]:
    """Per-job wall-clock cap in seconds (``None`` = inherit the global run budget)."""
    return _normalize_cap(value, "max_runtime_seconds", unit="seconds")


def normalize_quiet_hours(value: Any) -> Optional[str]:
    """``"HH:MM-HH:MM"`` (wrapping midnight allowed) or an explicit ``off``.

    Returns the canonical text, or ``None`` for "unset, inherit the global window". An
    explicit ``off`` is stored as-is so a job can opt out of a global window.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = ",".join(str(part).strip() for part in value if str(part).strip())
    text = str(value).strip().lower().replace(" ", "")
    if not text:
        return None
    if text in _QUIET_DISABLED:
        return "off"
    start, _, end = text.partition("-")
    if not parse_clock(start) or not parse_clock(end):
        raise ValueError(
            f"Invalid quiet_hours {value!r}. Use \"HH:MM-HH:MM\" (e.g. \"22:00-07:00\") "
            "or \"off\"."
        )
    return f"{start.zfill(5)}-{end.zfill(5)}"


def parse_clock(text: Any) -> Optional[time]:
    """``"HH:MM"`` → :class:`datetime.time`, else ``None``."""
    match = _CLOCK.match(str(text or "").strip())
    if not match:
        return None
    return time(int(match.group(1)), int(match.group(2)))


def quiet_hours_window(spec: Any) -> Optional[tuple[time, time]]:
    """Parse a window into ``(start, end)``; ``None`` when disabled or unparseable."""
    if spec is None:
        return None
    text = str(spec).strip().lower().replace(" ", "")
    if not text or text in _QUIET_DISABLED:
        return None
    start, _, end = text.partition("-")
    parsed_start, parsed_end = parse_clock(start), parse_clock(end)
    if parsed_start is None or parsed_end is None:
        return None
    return parsed_start, parsed_end


def in_quiet_hours(spec: Any, *, now: Optional[datetime] = None) -> bool:
    """True when *now* (default: local time) falls inside the window.

    A window whose end is at or before its start wraps midnight (``22:00-07:00``).
    An equal start/end is an empty window, not an all-day one.
    """
    window = quiet_hours_window(spec)
    if window is None:
        return False
    start, end = window
    if start == end:
        return False
    clock = (now or datetime.now()).time()
    if start < end:
        return start <= clock < end
    return clock >= start or clock < end


def resolve_quiet_hours(job: dict, cfg: dict) -> str:
    """Effective window for a job: its own value, else ``cron.quiet_hours``, else the default.

    An explicit per-job ``off`` wins over a configured global window.
    """
    own = job.get("quiet_hours") if isinstance(job, dict) else None
    if own is not None:
        return str(own)
    cron_cfg = cfg.get("cron") if isinstance(cfg, dict) else None
    configured = cron_cfg.get("quiet_hours") if isinstance(cron_cfg, dict) else None
    if configured is not None:
        return str(configured)
    return DEFAULT_QUIET_HOURS


def resolve_turn_cap(job: dict, cfg: dict) -> Any:
    """The turn-cap value to hand ``resolve_turn_limit``: the job's own cap, else the
    global ``agent.max_turns`` / ``max_turns`` (``None`` = unlimited)."""
    own = job.get("max_turns") if isinstance(job, dict) else None
    if own is not None:
        return own
    if isinstance(cfg, dict):
        agent_cfg = cfg.get("agent")
        if isinstance(agent_cfg, dict) and agent_cfg.get("max_turns") is not None:
            return agent_cfg.get("max_turns")
        return cfg.get("max_turns")
    return None


def resolve_runtime_cap(job: dict, cfg: Optional[dict] = None) -> Optional[float]:
    """Effective wall-clock cap in seconds, or ``None`` for unbounded.

    The job's own ``max_runtime_seconds`` wins; otherwise ``agent.run_budget_seconds``
    (the global soft budget) is reused as the hard cap, since a run that ignores the
    soft wrap-up must still be bounded.
    """
    own = job.get("max_runtime_seconds") if isinstance(job, dict) else None
    if own is not None:
        return float(own)
    if isinstance(cfg, dict):
        agent_cfg = cfg.get("agent")
        value = agent_cfg.get("run_budget_seconds") if isinstance(agent_cfg, dict) else None
        if value is None:
            value = cfg.get("run_budget_seconds")
        if value:
            try:
                seconds = float(value)
            except (TypeError, ValueError):
                return None
            return seconds if seconds > 0 else None
    return None


def elapsed_text(seconds: float) -> str:
    """``"1h 12m"`` / ``"4m 19s"`` — the ledger's and the notice's elapsed wording."""
    total = max(0, round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def wall_clock_stop_reason(cap_seconds: float, elapsed_seconds: float) -> str:
    """Ledger stop reason for a run the scheduler cut at its wall-clock cap."""
    return (
        f"wall_clock_cap: reached the {elapsed_text(cap_seconds)} cap after "
        f"{elapsed_text(elapsed_seconds)}"
    )


def quiet_hours_hold_reason(spec: str, *, now: Optional[datetime] = None) -> str:
    """Operator-facing note recorded when a failure notice is held overnight."""
    clock = (now or datetime.now()).strftime("%H:%M")
    return f"quiet hours ({spec}): failure notice held until the window ends (now {clock})"


def failure_notice_hold_reason(job: dict, cfg: Optional[dict] = None) -> Optional[str]:
    """The reason to withhold this job's failure notice, or ``None`` to send it.

    ``cfg`` is the loaded config; omitted (the delivery path) it is read here so the
    caller stays a single branch.
    """
    if cfg is None:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
    window = resolve_quiet_hours(job, cfg)
    return quiet_hours_hold_reason(window) if in_quiet_hours(window) else None


def normalize_reasoning_effort(value: Any) -> Optional[str]:
    """Spelling-only validation via the shared parser (cron knob never stricter/looser than
    config.yaml); model capability is deliberately NOT checked (model unknowable at create time,
    transports clamp at send time). None for unset, lowercase level, or ValueError."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    from hermes_constants import parse_reasoning_effort

    if parse_reasoning_effort(text) is None:
        raise ValueError(
            f"Invalid reasoning_effort {value!r}. Valid levels: "
            "none, minimal, low, medium, high, xhigh, max, ultra "
            "(empty string clears the override).")
    if text in {"false", "disabled"}:
        return "none"
    return text


def resolve_cron_turn_limit(cfg: dict, job: dict, resolve_turn_limit: Callable[[Any], Any]) -> Any:
    """``resolve_turn_limit`` applied to the job's effective cap (scheduler seam)."""
    return resolve_turn_limit(resolve_turn_cap(job, cfg))


def cap_stop_reason(result: Any) -> Optional[str]:
    """Ledger stop reason for a run the agent ended at a cap, else ``None``.

    Read off the turn's own exit reason so the scheduler never has to reconstruct what
    happened: the finalizer records which cap fired and how much of it was used.
    """
    if not isinstance(result, dict):
        return None
    reason = str(result.get("turn_exit_reason") or "")
    if not reason.startswith("run_budget_exhausted("):
        return None
    return f"wall_clock_cap: {reason}"

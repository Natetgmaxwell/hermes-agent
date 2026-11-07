"""Reading a dead Kanban worker's exit: the sentinel exit codes, the exit
trailer in the worker's own log, and its last captured words.

Split out of ``hermes_cli.kanban_db`` (the sentinel codes) and
``hermes_cli.kanban_db_dispatch`` (the log readers) — the dispatcher classifies
what these return, this module only interprets. ``kanban_db`` re-exports the
codes as the facade, so ``kb.KANBAN_*_EXIT_CODE`` keeps resolving.
"""

from __future__ import annotations

import re
from typing import Optional

from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER


# Worker exit "provider rate-limited": released WITHOUT counting a failure (the
# breaker must never trip on a throttle). 75 == BSD EX_TEMPFAIL.
KANBAN_RATE_LIMIT_EXIT_CODE = 75

# Worker exit "provider rejected the configuration": credential revoked (401/403), model gone
# (404), TLS chain broken — a retry cannot fix it, so the dispatcher parks the card blocked on
# the FIRST occurrence instead of spending ``failure_limit`` identical spawns. 78 == BSD EX_CONFIG.
KANBAN_TERMINAL_PROVIDER_EXIT_CODE = 78

# Worker exit "the owner refused this attempt's admission": the receipt settled ``unknown``
# ("Kanban execution unknown; retained by owner") — the attempt NEVER RAN, so the host's
# condition is not the card's work failing. Released WITHOUT counting a failure, booked
# ``owner_refused`` with ``infrastructure: true``, and the respawn guard spaces the retries.
# 77: adjacent to the other sentinels, no BSD meaning to collide with.
KANBAN_OWNER_REFUSED_EXIT_CODE = 77


_EXIT_TRAILER_RE = re.compile(
    r"^" + re.escape(KANBAN_WORKER_EXIT_TRAILER) + r"(\d+)\s*$", re.MULTILINE,
)


def _worker_log_exit_code(task_id: str, board: Optional[str] = None) -> Optional[int]:
    """Exit code from the trailer the worker CLI wrote to its own log; None when absent.

    The durable twin of the dispatcher's reap registry: written by the worker itself
    (``hermes_cli.quiet_single_query.exit_single_query``), so it is there whether
    or not the process running the sweep ever reaped the worker. Last trailer
    wins — the log is append-mode across re-runs.
    """
    from hermes_cli import kanban_db as _kb
    try:
        raw = _kb.read_worker_log(task_id, tail_bytes=4000, board=board)
    except Exception:
        return None
    matches = _EXIT_TRAILER_RE.findall(raw or "")
    return int(matches[-1]) if matches else None


# Rich panel/rule chrome around the rendered response, and the CLI's own preamble lines.
_LOG_CHROME = re.compile(r"[─━═╭╮╰╯│┃┌┐└┘]+|☤\s*Hermes")


def _exit_summary_marker() -> str:
    """The CLI exit-summary header (``cli_session_mixin.show_exit_summary``), in the active language."""
    from agent.i18n import t
    return t("cli.session.exit_resume_hint")


def _log_noise_prefixes() -> tuple[str, ...]:
    from agent.i18n import t
    return ("session_id:", "Query:", t("cli.chat.initializing_agent"))


def _worker_final_output(task_id: str, board: Optional[str] = None) -> str:
    """Best-effort read of a dead worker's last printed text, for the board diagnostic.

    A ``chat -q`` worker's stdout/stderr are redirected to its per-task log
    (``_default_spawn``), so when it exits without a terminal board call the
    reason is usually sitting there: the model's own explanation of why it
    could not comply (#88603), or the rendered provider error (#46593). The reap used to
    discard it in favour of a canned message on every retry. Trims the CLI exit
    summary, rule lines and the ``session_id:`` trailer; returns "" (never raises)
    on a missing/empty log.

    ``board`` must come from the dispatching tick: ambient current-board resolution
    is wrong for every board but the one the dispatcher thread happens to call
    "current", so the log would silently not be found.
    """
    from hermes_cli import kanban_db as _kb
    try:
        raw = _kb.read_worker_log(task_id, tail_bytes=4000, board=board)
    except Exception:
        return ""
    if not raw:
        return ""
    raw = _EXIT_TRAILER_RE.sub("", raw)
    cut = raw.rfind(_exit_summary_marker())
    if cut != -1:
        raw = raw[:cut]
    lines = []
    for ln in raw.splitlines():
        ln = _LOG_CHROME.sub("", ln).strip()
        if ln and not ln.startswith(_log_noise_prefixes()):
            lines.append(ln)
    return " ".join(lines)[-400:]

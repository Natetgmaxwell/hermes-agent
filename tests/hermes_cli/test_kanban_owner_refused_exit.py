"""An owner's admission refusal is infrastructure, not the card's work failing.

The review-lane submitter exits with ``KANBAN_OWNER_REFUSED_EXIT_CODE`` when the
owner's receipt settles ``unknown`` ("Kanban owner refused: Kanban execution
unknown; retained by owner"). That refusal means the attempt never ran — the
host's owner refused it — so booking it ``crashed`` spent the card's
work-failure budget until the breaker parked cards whose work was already on
disk (2026-11-05 strand). The sentinel exit books ``owner_refused`` with
``infrastructure: true``: consecutive_failures untouched, the card returns to
its source lane, and the respawn guard spaces the retries.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_worker_client as kwc
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    kbd._recent_worker_exits.clear()
    kb.init_db()
    return home


def _dead_review_worker_with_log(conn, tid: str, pid: int, rc: int) -> None:
    """Drive ``tid`` into the review lane, then claim it for a reviewer worker
    that already exited ``rc`` and left the refusal in its log."""
    kb.claim_task(conn, tid)
    run = kb.get_task(conn, tid)
    assert run is not None and run.current_run_id is not None
    assert kb.request_review(
        conn, tid, summary="Implementation complete", reviewer="reviewer",
        expected_run_id=run.current_run_id,
    )
    assert kb.claim_review_task(conn, tid) is not None
    conn.execute(
        "UPDATE tasks SET worker_pid=?, worker_started_at=NULL, started_at=? WHERE id=?",
        (pid, int(time.time()) - 120, tid),
    )
    conn.commit()
    log = kb.worker_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"Session: local-x\nKanban owner refused: Kanban execution unknown; retained by owner\n"
                f"\n{KANBAN_WORKER_EXIT_TRAILER}{rc}\n")


def test_review_lane_refusal_is_infrastructure_not_a_failure(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="review me", assignee="a")
        # One failure already spent: a second crash books the card ``blocked``
        # under the default 2-failure budget — the exact strand of 2026-11-05.
        conn.execute("UPDATE tasks SET consecutive_failures = 1 WHERE id = ?", (tid,))
        conn.commit()
        # The literal sentinel, not the constant: on the pre-fix code this same
        # stub shows the defect (a bare nonzero exit spending the budget).
        _dead_review_worker_with_log(conn, tid, 72001, 77)

        crashed = kbd.detect_crashed_workers(conn)

        task = kb.get_task(conn, tid)
        event = conn.execute(
            "SELECT kind FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,),
        ).fetchone()
        run = conn.execute(
            "SELECT outcome, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,),
        ).fetchone()

        # Not a crash, not a failure: the refusal never ran the attempt.
        assert crashed == []
        assert task.status == "review"  # back in its source lane, dispatchable
        assert task.consecutive_failures == 1  # budget untouched
        assert event["kind"] == "owner_refused"
        assert run["outcome"] == "owner_refused"
        assert kb._json_dict(run["metadata"]).get("infrastructure") is True
        assert "owner admission refusal" in (task.last_failure_error or "")
        # Spaced retries: the guard holds the re-spawn for the cooldown window.
        assert kbd.check_respawn_guard(conn, tid, lane="review") == "infrastructure_cooldown"


def test_refused_worker_exit_is_distinct_from_a_crash(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        conn.execute("UPDATE tasks SET consecutive_failures = 1 WHERE id = ?", (tid,))
        conn.commit()
        _dead_review_worker_with_log(conn, tid, 72002, kb.KANBAN_OWNER_REFUSED_EXIT_CODE)
        # The refusal is visible to the tick's telemetry, not only the board.
        kbd.detect_crashed_workers(conn)
        result = kbd.DispatchResult()
        result.owner_refused.extend(getattr(kbd.detect_crashed_workers, "_last_owner_refused", []))
        assert result.owner_refused == [tid]
        assert "owner_refused=1" in kbd.describe_suppression([result])


def test_submitter_exits_sentinel_on_unknown_admission(monkeypatch, capsys):
    """The worker client's own exit: the unknown-admission refusal maps to the
    sentinel exit code, not a generic 1 the sweep must book as a crash."""
    for key, value in {
        "HERMES_KANBAN_BOARD": "b", "HERMES_KANBAN_TASK": "t_x",
        "HERMES_KANBAN_RUN_ID": "9", "HERMES_KANBAN_CLAIM_LOCK": "h:1",
        "HERMES_KANBAN_DB": "/nonexistent/board.db",
    }.items():
        monkeypatch.setenv(key, value)

    async def refused(params, board_db):
        raise kwc.OwnerRefusedAdmission("Kanban execution unknown; retained by owner")
        raise AssertionError  # pragma: no cover

    monkeypatch.setattr(kwc, "run", refused)
    assert kwc._submit() == kb.KANBAN_OWNER_REFUSED_EXIT_CODE
    assert "Kanban owner refused: Kanban execution unknown" in capsys.readouterr().err


def test_plain_crash_in_the_review_lane_still_spends_the_budget(kanban_home):
    """The fence is not weakened: a real worker crash (bare nonzero exit, the
    pre-fix refusal shape) still books ``crashed`` and trips the breaker, and a
    LIVE run's admission lock is untouched by this change."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="review me too", assignee="a")
        conn.execute("UPDATE tasks SET consecutive_failures = 1 WHERE id = ?", (tid,))
        conn.commit()
        _dead_review_worker_with_log(conn, tid, 72003, 1)

        crashed = kbd.detect_crashed_workers(conn)

        task = kb.get_task(conn, tid)
        assert crashed == [tid]
        assert task is not None and task.status == "blocked"  # budget spent, breaker tripped
        assert task.consecutive_failures == 2


def test_exit_code_kind_recognises_the_sentinel():
    assert kbd._exit_code_kind(kb.KANBAN_OWNER_REFUSED_EXIT_CODE) == (
        "owner_refused", kb.KANBAN_OWNER_REFUSED_EXIT_CODE,
    )

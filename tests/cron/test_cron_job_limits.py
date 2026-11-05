"""Per-job run caps and quiet hours (cron/job_limits.py + the cron seams that honour them).

Bitten by cron job ``ae9c87cd6272`` (2026-10-28 22:00 AEST): a scheduled run made 180
provider calls at 86-220s each and ran for 4h19m. Nothing bounded it — the inactivity
watchdog only fires on *silence*, and the global turn limit was the only call cap. These
tests pin the two per-job knobs that do bound it, plus the overnight failure-notice hold.
"""

import pytest

from cron.job_limits import (
    DEFAULT_QUIET_HOURS, cap_stop_reason, in_quiet_hours, normalize_max_runtime_seconds,
    normalize_max_turns, normalize_quiet_hours, resolve_quiet_hours, resolve_runtime_cap,
    resolve_turn_cap, wall_clock_stop_reason,
)

from datetime import datetime


# ---- normalizers ---------------------------------------------------------------------

def test_max_turns_accepts_whole_numbers_and_clears_on_empty():
    assert normalize_max_turns(40) == 40
    assert normalize_max_turns("40") == 40
    assert normalize_max_turns(40.0) == 40
    assert normalize_max_turns(None) is None
    assert normalize_max_turns("") is None


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "abc"])
def test_max_turns_rejects_values_that_cannot_bound_a_run(bad):
    with pytest.raises(ValueError):
        normalize_max_turns(bad)


def test_max_runtime_seconds_uses_the_same_grammar():
    assert normalize_max_runtime_seconds(1800) == 1800
    assert normalize_max_runtime_seconds("1800") == 1800
    assert normalize_max_runtime_seconds(None) is None
    with pytest.raises(ValueError):
        normalize_max_runtime_seconds(0)


def test_quiet_hours_parses_a_window_and_stores_off_explicitly():
    assert normalize_quiet_hours("22:00-07:00") == "22:00-07:00"
    assert normalize_quiet_hours("9:00-17:30") == "09:00-17:30"
    assert normalize_quiet_hours("off") == "off"
    assert normalize_quiet_hours(None) is None
    with pytest.raises(ValueError):
        normalize_quiet_hours("10pm-7am")


# ---- window semantics ----------------------------------------------------------------

def test_window_wraps_midnight():
    spec = "22:00-07:00"
    assert in_quiet_hours(spec, now=datetime(2026, 10, 28, 23, 30)) is True
    assert in_quiet_hours(spec, now=datetime(2026, 10, 29, 2, 15)) is True
    assert in_quiet_hours(spec, now=datetime(2026, 10, 29, 7, 0)) is False
    assert in_quiet_hours(spec, now=datetime(2026, 10, 29, 14, 0)) is False


def test_off_and_empty_disable_the_window():
    assert in_quiet_hours("off", now=datetime(2026, 10, 29, 2, 0)) is False
    assert in_quiet_hours(None, now=datetime(2026, 10, 29, 2, 0)) is False


def test_equal_bounds_are_an_empty_window_not_all_day():
    assert in_quiet_hours("09:00-09:00", now=datetime(2026, 10, 29, 9, 0)) is False


# ---- resolution: per job, then global, then default ----------------------------------

def test_a_job_cap_overrides_the_global_turn_limit():
    assert resolve_turn_cap({"max_turns": 40}, {"agent": {"max_turns": 180}}) == 40
    assert resolve_turn_cap({}, {"agent": {"max_turns": 180}}) == 180
    assert resolve_turn_cap({}, {}) is None


def test_runtime_cap_prefers_the_job_then_the_global_budget():
    assert resolve_runtime_cap({"max_runtime_seconds": 900}, {"agent": {"run_budget_seconds": 1800}}) == 900
    assert resolve_runtime_cap({}, {"agent": {"run_budget_seconds": 1800}}) == 1800
    assert resolve_runtime_cap({}, {}) is None


def test_quiet_hours_is_opt_in_and_a_job_can_opt_out():
    """Unset means "alert whenever it fails": a code-level always-on window would make every
    failure-delivery assertion depend on the wall clock."""
    assert resolve_quiet_hours({}, {}) == DEFAULT_QUIET_HOURS == ""
    assert in_quiet_hours(resolve_quiet_hours({}, {}), now=datetime(2026, 10, 29, 2, 0)) is False
    assert resolve_quiet_hours({}, {"cron": {"quiet_hours": "23:00-06:00"}}) == "23:00-06:00"
    assert resolve_quiet_hours({"quiet_hours": "off"}, {"cron": {"quiet_hours": "23:00-06:00"}}) == "off"


def test_ledger_stop_reason_names_the_cap():
    reason = cap_stop_reason({"turn_exit_reason": "run_budget_exhausted(1800s/1800s)"})
    assert reason.startswith("wall_clock_cap:")
    assert "1800s/1800s" in reason
    assert cap_stop_reason({"turn_exit_reason": "max_iterations_reached(180/180)"}) is None
    assert cap_stop_reason({"turn_exit_reason": ""}) is None


def test_wall_clock_stop_reason_reads_as_elapsed_time():
    assert "1h 12m" in wall_clock_stop_reason(3600, 4320)


# ---- round-trip through the job store ------------------------------------------------

def _store(tmp_path, monkeypatch):
    from cron import jobs

    monkeypatch.setattr(jobs, "CRON_DIR", tmp_path)
    monkeypatch.setattr(jobs, "JOBS_FILE", tmp_path / "jobs.json")
    return jobs


def test_create_job_round_trips_the_caps(tmp_path, monkeypatch):
    jobs = _store(tmp_path, monkeypatch)

    job = jobs.create_job(
        prompt="digest the inbox", schedule="every 1h", name="capped",
        max_turns=40, max_runtime_seconds=1800, quiet_hours="23:00-06:00")

    assert job["max_turns"] == 40
    assert job["max_runtime_seconds"] == 1800
    assert job["quiet_hours"] == "23:00-06:00"
    stored = jobs.get_job(job["id"])
    assert (stored["max_turns"], stored["max_runtime_seconds"]) == (40, 1800)


def test_job_without_caps_stores_no_cap_keys(tmp_path, monkeypatch):
    jobs = _store(tmp_path, monkeypatch)

    job = jobs.create_job(prompt="plain", schedule="every 1h", name="uncapped")

    for key in ("max_turns", "max_runtime_seconds", "quiet_hours"):
        assert key not in job, f"{key} must stay absent so the job inherits config"


def test_the_scheduler_seam_applies_the_jobs_own_cap_first():
    """The binding cron/scheduler.py reads: the job's cap wins, absent falls to config."""
    from hermes_cli.config import resolve_turn_limit

    from cron.job_limits import resolve_cron_turn_limit

    cfg = {"agent": {"max_turns": 180}}
    assert resolve_cron_turn_limit(cfg, {"max_turns": 40}, resolve_turn_limit) == 40
    assert resolve_cron_turn_limit(cfg, {}, resolve_turn_limit) == 180


def test_the_ledger_keeps_the_cap_stop_reason(tmp_path, monkeypatch):
    """A run the wall clock cut is COMPLETED with its stop reason on the execution row,
    so the run history can show why a finished run ended early."""
    from cron import executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    record = executions.create_execution("capjob", source="tick")
    reason = "wall_clock_cap: run_budget_exhausted(1810s/1800s)"
    finished = executions.finish_execution(record["id"], success=True, stop_reason=reason)

    assert finished is not None
    assert finished["status"] == "completed"
    assert finished["stop_reason"] == reason


def test_update_job_can_set_and_clear_a_cap(tmp_path, monkeypatch):
    jobs = _store(tmp_path, monkeypatch)
    job = jobs.create_job(prompt="x", schedule="every 1h", name="edit-me")

    jobs.update_job(job["id"], {"max_turns": 12})
    assert jobs.get_job(job["id"])["max_turns"] == 12

    jobs.update_job(job["id"], {"max_turns": 0})
    # Clearing leaves the job inheriting agent.max_turns (resolve_turn_cap treats None that way).
    assert jobs.get_job(job["id"]).get("max_turns") is None
    assert resolve_turn_cap(jobs.get_job(job["id"]), {"agent": {"max_turns": 180}}) == 180

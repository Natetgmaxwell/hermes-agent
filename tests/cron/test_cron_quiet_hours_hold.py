"""A failure notice inside the job's quiet-hours window is held, not lost.

The failure itself must survive: only the page is suppressed, so the morning briefing path
still sees what broke. This pins both halves — the notice is not sent, and the job/incident
record keeps the failure — plus that a window which does not cover "now" still delivers.
"""

import cron.scheduler_delivery as sd


def _job(**over):
    job = {"id": "quiet123", "name": "nightly", "deliver": "local", "enabled": True}
    job.update(over)
    return job


def test_failure_notice_is_held_inside_the_window(monkeypatch):
    sent: list = []
    monkeypatch.setattr(sd, "_resolve_delivery_targets", lambda job, *, for_failure=False: sent.append(1) or [])
    monkeypatch.setattr(sd, "_record_delivery_verification", lambda job, targets: None)
    monkeypatch.setattr(sd, "_unresolved_delivery_outcome", lambda job, for_failure: None)
    # A window covering every clock reading keeps this test off the wall clock.
    job = _job(quiet_hours="00:00-23:59")

    error = sd._deliver_result(job, "Job 'nightly' failed", for_failure=True)

    assert error is None, "a held notice is not a delivery error"
    assert sent == [], "no target may be contacted inside quiet hours"
    assert "quiet hours" in job["_failure_notice_held"]


def test_failure_notice_is_delivered_outside_the_window(monkeypatch):
    touched: list = []
    monkeypatch.setattr(
        sd, "_resolve_delivery_targets",
        lambda job, *, for_failure=False: touched.append(for_failure) or [])
    monkeypatch.setattr(sd, "_record_delivery_verification", lambda job, targets: None)
    monkeypatch.setattr(sd, "_unresolved_delivery_outcome", lambda job, for_failure: None)
    job = _job(quiet_hours="off")

    sd._deliver_result(job, "Job 'nightly' failed", for_failure=True)

    assert touched == [True], "the delivery path must run when the notice is not held"
    assert "_failure_notice_held" not in job


def test_success_output_is_never_held(monkeypatch):
    """Quiet hours govern failure notices only — a nightly digest still arrives."""
    touched: list = []
    monkeypatch.setattr(
        sd, "_resolve_delivery_targets",
        lambda job, *, for_failure=False: touched.append(for_failure) or [])
    monkeypatch.setattr(sd, "_record_delivery_verification", lambda job, targets: None)
    monkeypatch.setattr(sd, "_unresolved_delivery_outcome", lambda job, for_failure: None)

    sd._deliver_result(_job(quiet_hours="00:00-23:59"), "digest body", for_failure=False)

    assert touched == [False]

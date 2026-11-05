"""Regression for the cron cap defect: a turn that exhausts its API-call budget with a
pending tool result must END WITH TEXT, not the generic "No reply: the turn stopped while
a tool result was still pending" close.

Measured failure this pins (cron job ``ae9c87cd6272``, 2026-10-28 22:00 AEST tick)::

    Turn ended: reason=pending_tool_result api_calls=180/180 budget=180/180
    RuntimeError: No reply: the turn stopped while a tool result was still pending ...

The budget-exhausted summary path (``_handle_max_iterations``) never ran: the generic
pending-tool-result branch minted the failure close first, so the run was recorded as a
hard failure and the model was never asked to summarise.
"""

from types import SimpleNamespace
import time

import pytest

from agent.turn_finalizer import finalize_turn


class _CapAgent:
    """Minimal agent double for the post-loop finalizer at budget exhaustion."""

    def __init__(self, *, max_iterations=180, budget_remaining=0, explainer=True,
                 run_budget_seconds=None, budget_started_ago=0.0):
        self.max_iterations = max_iterations
        self.run_budget_seconds = run_budget_seconds
        self._run_budget_started_at = (
            time.time() - budget_started_ago) if run_budget_seconds else None
        self.iteration_budget = SimpleNamespace(
            remaining=budget_remaining, used=max_iterations, max_total=max_iterations)
        self.quiet_mode = True
        self.model = "cap-model"
        self.provider = "cap-provider"
        self.base_url = ""
        self.session_id = "sess-cap"
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        for _k in (
            "session_input_tokens", "session_output_tokens", "session_cache_read_tokens",
            "session_cache_write_tokens", "session_reasoning_tokens", "session_prompt_tokens",
            "session_completion_tokens", "session_total_tokens", "session_estimated_cost_usd",
        ):
            setattr(self, _k, 0)
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []
        self.persisted_messages = None
        self.summary_calls = 0
        self._explainer = explainer
        self.diagnostics = []

    def _handle_max_iterations(self, messages, api_call_count):
        self.summary_calls += 1
        return "Recap: filed 3 triage items; 2 remain."

    def _emit_diagnostic_status(self, text, **_kwargs):
        self.diagnostics.append(text)

    def _emit_status(self, *_a, **_kw):
        pass

    def _safe_print(self, *_a, **_kw):
        pass

    def _save_trajectory(self, *_a, **_kw):
        pass

    def _cleanup_task_resources(self, *_a, **_kw):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        self.persisted_messages = list(messages)

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return self._explainer

    def _format_turn_completion_explanation(self, reason, cause=None, **_kw):
        return (
            "No reply: the turn stopped while a tool result was still pending and the "
            "model produced no follow-up text. Send `continue` to let it summarize."
        )

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_kw):
        pass


_TOOL_TAIL = [
    {"role": "user", "content": "triage the inbox"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "c1", "name": "terminal", "content": "done"},
]


def _finalize(agent, *, messages=_TOOL_TAIL, final_response=None,
              api_call_count=180, exit_reason="unknown", failed=False):
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=api_call_count,
        interrupted=False,
        failed=failed,
        messages=list(messages),
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="triage the inbox",
        original_user_message="triage the inbox",
        _should_review_memory=False,
        _turn_exit_reason=exit_reason,
    )


@pytest.mark.parametrize("exit_reason", ["unknown", "budget_exhausted"])
def test_call_cap_with_pending_tool_result_asks_for_summary(monkeypatch, exit_reason):
    """180/180 with a tool tail: the model is asked to summarise and the turn ends with
    that text — never the generic 'No reply' close."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _CapAgent()

    result = _finalize(agent, exit_reason=exit_reason)

    assert agent.summary_calls == 1, "the cap must ask the model for a final summary"
    assert result["final_response"] == "Recap: filed 3 triage items; 2 remain."
    assert result["turn_exit_reason"] == "max_iterations_reached(180/180)"
    assert result["failed"] is False
    assert "No reply" not in (result["final_response"] or "")


def test_empty_string_response_at_the_cap_still_asks_for_summary(monkeypatch):
    """The production shape: the loop left ``final_response`` as ``""`` (not ``None``), so
    the old ``is None`` gate skipped the summary and the pending-tool close minted
    "No reply". An empty string is "no answer yet" too."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _CapAgent()

    result = _finalize(agent, final_response="", exit_reason="unknown")

    assert agent.summary_calls == 1
    assert result["final_response"] == "Recap: filed 3 triage items; 2 remain."
    assert result["turn_exit_reason"] == "max_iterations_reached(180/180)"
    assert result["failed"] is False


def test_reaching_the_call_cap_delivers_instead_of_failing(monkeypatch):
    """The capped turn is a resumable handoff: cron delivers its summary as a COMPLETED
    run (``is_max_iteration_handoff``) rather than raising ``RuntimeError``."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    from agent.turn_failure_copy import is_max_iteration_handoff
    from cron.scheduler import _final_response_from_result

    agent = _CapAgent()
    result = _finalize(agent)

    assert is_max_iteration_handoff(result) is True
    assert _final_response_from_result(result, "job", "job", object) == result["final_response"]


def test_wall_clock_cap_with_pending_tool_result_asks_for_summary(monkeypatch):
    """The wall-clock cap is the same kind of stop under a different name: a run whose
    clock ran out mid-tool-round (10 of 180 calls, iteration budget untouched) ends with
    the model's summary, never the generic 'No reply' close."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _CapAgent(max_iterations=180, budget_remaining=40,
                      run_budget_seconds=1800, budget_started_ago=1810)

    result = _finalize(agent, api_call_count=10, exit_reason="unknown")

    assert agent.summary_calls == 1
    assert result["final_response"] == "Recap: filed 3 triage items; 2 remain."
    assert result["turn_exit_reason"].startswith("run_budget_exhausted(")
    assert "1800s" in result["turn_exit_reason"]
    assert result["failed"] is False


def test_pending_tool_result_still_fails_when_budget_is_not_exhausted(monkeypatch):
    """Default behaviour is unchanged: a mid-budget tool tail is still the silent stop."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _CapAgent(max_iterations=180, budget_remaining=40)

    result = _finalize(agent, api_call_count=10, exit_reason="unknown")

    assert agent.summary_calls == 0
    assert result["turn_exit_reason"] == "pending_tool_result"
    assert result["failed"] is True


def test_capped_exit_keeps_the_real_stop_reason(monkeypatch):
    """At the cap the real reason survives: a session write failure is not relabelled as the
    generic 'No reply' close just because the transcript ends on a tool row."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _CapAgent()

    result = _finalize(
        agent, api_call_count=180, exit_reason="session_persistence_failed", failed=True)

    assert result["turn_exit_reason"] == "session_persistence_failed"

"""Canonical-column dedup for reasoning: the byte-identical reasoning_content duplicate is
not stored, and every read path rehydrates it from ``reasoning``.

state.db stored each assistant turn's reasoning text twice: the streaming promotion in
``build_assistant_message`` sets ``msg["reasoning_content"] = reasoning_text`` — the same
text as ``msg["reasoning"]`` — and the write funnel bound both columns. Measured on the
live store that was 650 MB of byte-identical duplicate across ~72.8k rows.

The contract these tests pin:
  * the write funnel stores reasoning_content only when it differs from reasoning;
  * every decoder (conversation replay, repair, display projection) rebuilds the key from
    reasoning, so all four reasoning_content consumers (anthropic thinking-block
    injection, merged-turn carry, compressor budget charge, replay policy) see the same
    bytes as when the duplicate was stored;
  * rows where reasoning is NULL (reasoning_content is the only copy), rows where
    reasoning continues past reasoning_content, pads, and reasoning_details are untouched;
  * a row reclaimed by ``scripts/state_db_reasoning_dedup.py`` (reasoning_content NULLed
    where byte-identical) replays with byte-identical payloads.

Real SessionDB against a real sqlite file; no mocks of the code under test.
"""

import json
import shutil
import sqlite3

import pytest

from hermes_state import SessionDB

DETAILS = json.dumps([{"type": "thinking", "thinking": "signed text", "signature": "sig-abc"}])


def _raw_insert(db_path, session_id, rows):
    """Insert rows in the PRE-dedup stored shape (duplicate reasoning_content included)."""
    con = sqlite3.connect(db_path)
    for r in rows:
        con.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id, reasoning, "
            "reasoning_content, reasoning_details, finish_reason, timestamp, active) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,1)",
            (session_id, r.get("role", "assistant"), r.get("content", "answer"),
             json.dumps(r["tool_calls"]) if r.get("tool_calls") else None,
             r.get("tool_call_id"), r.get("reasoning", None) if "reasoning" in r else None,
             r.get("reasoning_content", None) if "reasoning_content" in r else None,
             r.get("reasoning_details"), r.get("finish_reason", "stop"),
             r.get("timestamp", 1000.0 + len(rows))))
    con.commit()
    con.close()


def _stored(db_path):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT role, reasoning, reasoning_content, reasoning_details FROM messages "
        "WHERE role = 'assistant' ORDER BY id")]
    con.close()
    return rows


def _payload(msg):
    from agent.anthropic_message_convert import _convert_assistant_message
    from agent.context_compressor import _estimate_msg_budget_tokens
    from agent.message_sanitization import _replayable_reasoning_text, apply_reasoning_content_policy

    slim = {k: v for k, v in msg.items()
            if k in ("role", "content", "tool_calls", "reasoning", "reasoning_content", "reasoning_details")}
    out = {
        "replayable_text": _replayable_reasoning_text(slim),
        "budget": _estimate_msg_budget_tokens(slim),
        "budget_no_stale": _estimate_msg_budget_tokens(slim, charge_stale_thinking=False),
        "anthropic": _convert_assistant_message(dict(slim))["content"],
    }
    for label, pad, carriers in (
        ("must_echo", True, frozenset({"reasoning_content"})),
        ("both_carriers", False, frozenset({"reasoning_content", "reasoning"})),
        ("native", False, None),
    ):
        api_msg = dict(slim)
        apply_reasoning_content_policy(slim, api_msg, pad, carriers)
        out[label] = {k: api_msg.get(k) for k in ("reasoning", "reasoning_content", "reasoning_details")}
    return out


@pytest.fixture
def db(tmp_path):
    d = SessionDB(tmp_path / "state.db")
    d.create_session("s", "cli")
    yield d, tmp_path / "state.db"
    d.close()


class TestWriteFunnel:
    def test_byte_identical_duplicate_not_stored(self, db):
        d, path = db
        d.append_message("s", "assistant", content="a", reasoning="think", reasoning_content="think")
        (row,) = _stored(path)
        assert row["reasoning"] == "think"
        assert row["reasoning_content"] is None

    def test_only_copy_kept_when_reasoning_null(self, db):
        d, path = db
        d.append_message("s", "assistant", content="a", reasoning=None, reasoning_content="only copy")
        (row,) = _stored(path)
        assert row["reasoning"] is None
        assert row["reasoning_content"] == "only copy"

    def test_kept_when_reasoning_continues_past_it(self, db):
        d, path = db
        d.append_message("s", "assistant", content="a",
                         reasoning="part1part2", reasoning_content="part1")
        (row,) = _stored(path)
        assert row["reasoning"] == "part1part2"
        assert row["reasoning_content"] == "part1"

    def test_pad_kept_when_no_reasoning(self, db):
        d, path = db
        d.append_message("s", "assistant", content="a", tool_calls=[], reasoning=None,
                         reasoning_content=" ")
        (row,) = _stored(path)
        assert row["reasoning_content"] == " "

    def test_details_stored_verbatim(self, db):
        d, path = db
        d.append_message("s", "assistant", content="a", reasoning="think",
                         reasoning_content="think", reasoning_details=json.loads(DETAILS))
        (row,) = _stored(path)
        assert row["reasoning_details"] == DETAILS


class TestReadPathRehydration:
    def test_replay_rehydrates_identical_bytes(self, db):
        d, _ = db
        d.append_message("s", "assistant", content="a", reasoning="think A", reasoning_content="think A")
        msg = next(m for m in d.get_messages_as_conversation("s") if m["role"] == "assistant")
        assert msg["reasoning"] == "think A"
        assert msg["reasoning_content"] == "think A"

    def test_display_projection_rehydrates(self, db):
        d, _ = db
        d.append_message("s", "assistant", content="a", reasoning="think B", reasoning_content="think B")
        msg = next(m for m in d.get_messages("s") if m["role"] == "assistant")
        assert msg["reasoning_content"] == "think B"

    def test_no_rehydration_without_reasoning(self, db):
        d, _ = db
        d.append_message("s", "assistant", content="a", reasoning_content="only copy")
        msg = next(m for m in d.get_messages_as_conversation("s") if m["role"] == "assistant")
        assert msg["reasoning_content"] == "only copy"
        assert "reasoning" not in msg

    def test_reflush_of_rehydrated_message_stays_deduped(self, db):
        d, path = db
        d.append_message("s", "assistant", content="a", reasoning="think C", reasoning_content="think C")
        msgs = d.get_messages_as_conversation("s")
        d.create_session("s2", "cli")
        d.append_messages_batch("s2", msgs)
        assert all(r["reasoning_content"] is None
                   for r in _stored(path) if r["reasoning"] == "think C")


SAMPLE_ROWS = [
    # (a) duplicate + signed details
    {"reasoning": "dup with details", "reasoning_content": "dup with details",
     "reasoning_details": DETAILS, "content": "a"},
    # (b) duplicate, no details
    {"reasoning": "dup no details", "reasoning_content": "dup no details", "content": "b"},
    # (c) reasoning NULL: reasoning_content is the only copy
    {"reasoning": None, "reasoning_content": "only copy", "content": "c"},
    # (d) reasoning continues past reasoning_content
    {"reasoning": "part1-part2", "reasoning_content": "part1-", "content": "d"},
    # plain, no reasoning at all
    {"reasoning": None, "reasoning_content": None, "content": "e"},
    # (e) merged tool-call turn: prev carries tool calls + duplicate, later turn carries its own
    {"reasoning": "prev think", "reasoning_content": "prev think", "content": "calling",
     "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "result", "finish_reason": None,
     "reasoning": None, "reasoning_content": None},
    {"reasoning": "later think", "reasoning_content": "later think", "content": "final"},
]


@pytest.fixture
def old_shape_db(tmp_path):
    """A store in the PRE-fix shape: every duplicate reasoning_content still stored."""
    path = tmp_path / "old.db"
    d = SessionDB(path)
    d.create_session("conv", "cli")
    d.close()
    _raw_insert(path, "conv", SAMPLE_ROWS)
    return path


class TestReclaimSafety:
    def _replay(self, path):
        d = SessionDB(path)
        msgs = d.get_messages_as_conversation("conv")
        d.close()
        return msgs

    def test_payloads_byte_identical_after_reclaim(self, old_shape_db):
        before = self._replay(old_shape_db)

        reclaimed = old_shape_db.parent / "reclaimed.db"
        shutil.copy(old_shape_db, reclaimed)
        con = sqlite3.connect(reclaimed)
        cur = con.execute(
            "UPDATE messages SET reasoning_content = NULL "
            "WHERE role = 'assistant' AND reasoning IS NOT NULL AND length(reasoning) > 0 "
            "AND reasoning_content IS reasoning")
        # the duplicate rows (a), (b), (e)x2; the only-copy, concat and pad rows survive
        assert cur.rowcount == 4
        con.commit()
        con.close()

        after = self._replay(reclaimed)
        assert len(before) == len(after)
        for old, new in zip(before, after):
            if old["role"] != "assistant":
                continue
            assert _payload(old) == _payload(new), f"payload changed for {old['content']!r}"

    def test_details_bytes_untouched_by_reclaim(self, old_shape_db):
        reclaimed = old_shape_db.parent / "reclaimed2.db"
        shutil.copy(old_shape_db, reclaimed)
        con = sqlite3.connect(reclaimed)
        con.execute(
            "UPDATE messages SET reasoning_content = NULL "
            "WHERE role = 'assistant' AND reasoning IS NOT NULL AND length(reasoning) > 0 "
            "AND reasoning_content IS reasoning")
        con.commit()
        rows = con.execute(
            "SELECT reasoning_details FROM messages WHERE reasoning_details IS NOT NULL").fetchall()
        con.close()
        assert rows == [(DETAILS,)]

    def test_budget_charge_unchanged(self, old_shape_db):
        from agent.context_compressor import _estimate_msg_budget_tokens

        def total(path, **kw):
            return sum(_estimate_msg_budget_tokens(m, **kw) for m in self._replay(path))

        reclaimed = old_shape_db.parent / "reclaimed3.db"
        shutil.copy(old_shape_db, reclaimed)
        con = sqlite3.connect(reclaimed)
        con.execute(
            "UPDATE messages SET reasoning_content = NULL "
            "WHERE role = 'assistant' AND reasoning IS NOT NULL AND length(reasoning) > 0 "
            "AND reasoning_content IS reasoning")
        con.commit()
        con.close()

        with_reasoning = [m for m in self._replay(old_shape_db)
                          if m["role"] == "assistant" and (m.get("reasoning") or m.get("reasoning_content"))]
        without = [m for m in self._replay(old_shape_db)
                   if m["role"] == "assistant" and not (m.get("reasoning") or m.get("reasoning_content"))]
        assert with_reasoning and without
        assert total(old_shape_db) == total(reclaimed)
        assert total(old_shape_db, charge_stale_thinking=False) == total(
            reclaimed, charge_stale_thinking=False)

    def test_merged_tool_call_turn_carry_identical(self, old_shape_db):
        from agent.agent_runtime_helpers import _merge_assistant_into

        def merged(path):
            assistants = [m for m in self._replay(path) if m["role"] == "assistant"]
            prev = next(a for a in assistants if a.get("tool_calls"))
            later = assistants[-1]
            prev, later = dict(prev), dict(later)
            carried = _merge_assistant_into(prev, later)
            return carried, {k: prev.get(k) for k in ("content", "reasoning", "reasoning_content")}

        reclaimed = old_shape_db.parent / "reclaimed4.db"
        shutil.copy(old_shape_db, reclaimed)
        con = sqlite3.connect(reclaimed)
        con.execute(
            "UPDATE messages SET reasoning_content = NULL "
            "WHERE role = 'assistant' AND reasoning IS NOT NULL AND length(reasoning) > 0 "
            "AND reasoning_content IS reasoning")
        con.commit()
        con.close()
        assert merged(old_shape_db) == merged(reclaimed)

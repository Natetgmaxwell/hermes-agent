"""A kanban worker spawned into a hermes-agent checkout workspace must boot the PINNED tree.

The dispatcher spawns the worker as ``sys.executable -m hermes_cli.kanban_worker_client``
with ``cwd`` = the card's workspace and the running install pinned first on the child's
PYTHONPATH (``pin_hermes_tree_on_pythonpath``). But ``python -m`` also PREPENDS the cwd
to ``sys.path``, so a workspace that is itself a hermes-agent checkout shadows the pinned
tree with its own ``hermes_cli/`` and ``hermes_bootstrap.py``: the worker derives a fresh
install key for the workspace path and dies on ``no dependency environment is committed``
before it ever reaches the owner (t_127eeb0e: three hermes-core cards, 4 dispatcher
failures each, dead since 2026-11-06).

``-P`` (argv-local, exactly ``hermes_cli/uninstall.py``'s pattern) stops the cwd prepend;
PYTHONPATH entries still apply, so the pinned tree wins. The env-var form
(``PYTHONSAFEPATH=1``) was rejected on purpose: it propagates to every interpreter the
worker later spawns, which would break a workspace-local ``python -m mymodule`` run
inside the agent's own shell.
"""

import os
import subprocess
import sys
from pathlib import Path

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd


def test_worker_spawned_in_a_hermes_checkout_boots_the_pinned_tree(tmp_path, monkeypatch):
    """Spawn the worker exactly as ``_default_spawn`` does — same interpreter, same flags,
    same env, cwd = a workspace that IS a hermes-agent checkout — and ask the child where
    ``hermes_cli`` resolves from: the pinned install root, never the workspace copy."""
    root = str(Path(kbd.__file__).resolve().parents[1])
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("HERMES_BIN", raising=False)
    monkeypatch.setattr(kbd, "_resolve_hermes_argv",
                        lambda: [sys.executable, "-m", "hermes_cli.main"])

    captured = {}

    class _FakePopen:
        def __init__(self, cmd, **kwargs):
            captured["cmd"] = list(cmd)
            captured["env"] = dict(kwargs.get("env") or {})
            captured["cwd"] = kwargs.get("cwd")
            self.pid = 4242

    real_popen = subprocess.Popen
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)

    # The workspace is a hermes-agent checkout: its own hermes_cli/ is the shadow.
    ws = tmp_path / "ws-checkout"
    (ws / "hermes_cli").mkdir(parents=True)
    (ws / "hermes_cli" / "__init__.py").write_text(
        "# workspace shadow copy; the worker must never boot this one\n", encoding="utf-8")
    (ws / "hermes_bootstrap.py").write_text(
        "raise ImportError('shadow hermes_bootstrap must not boot')\n", encoding="utf-8")

    task = kb.Task(
        id="t_pinned_tree", title="x", body=None, assignee="coder", status="ready",
        priority=0, created_by=None, created_at=0, started_at=None, completed_at=None,
        workspace_kind="worktree", workspace_path=str(ws), claim_lock=None,
        claim_expires=None, tenant=None, branch_name=None,
    )
    kbd._default_spawn(task, str(ws))
    monkeypatch.setattr(subprocess, "Popen", real_popen)

    assert captured["cwd"] == str(ws), "the worker's cwd must stay the workspace"
    cmd = captured["cmd"]
    i = cmd.index("hermes_cli.kanban_worker_client")
    assert cmd[i - 1] == "-m"
    flags = cmd[: i - 1]
    assert "-P" in flags, (
        f"worker argv {cmd!r} lacks -P: python -m prepends the workspace cwd to "
        "sys.path, so a hermes-agent checkout workspace shadows the pinned tree")
    assert captured["env"]["PYTHONPATH"].split(os.pathsep)[0] == root, (
        "the live tree must stay pinned first")

    # Re-run the spawn's interpreter with ITS flags, env and cwd; `-c` prepends the
    # cwd exactly like `-m` does, so this probe reproduces the shadowing condition.
    probe = "import importlib.util; print(importlib.util.find_spec('hermes_cli').origin)"
    r = subprocess.run(flags + ["-c", probe], capture_output=True, text=True,
                       timeout=60, env=captured["env"], cwd=str(ws), check=False)
    assert r.returncode == 0, f"probe failed: rc={r.returncode} stderr={r.stderr!r}"
    assert Path(r.stdout.strip()).resolve() == Path(root, "hermes_cli", "__init__.py"), (
        f"worker booted hermes_cli from {r.stdout.strip()!r}; "
        "the workspace copy shadowed the pinned tree")

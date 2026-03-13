"""T2 — Integration tests: SelfHealingAgent (mock Anthropic API)"""

import json
from unittest.mock import MagicMock, patch, call
from types import SimpleNamespace

import pytest

from agent.core import SelfHealingAgent, MaxTurnsExceeded
from agent.tools import ToolRegistry
from agent.memory import FailureMemory


# ── Helpers to build mock API responses ──────────────────────────────────────

def _text_block(text: str):
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _tool_use_block(tool_id: str, name: str, input_dict: dict):
    b = MagicMock()
    b.type = "tool_use"
    b.id = tool_id
    b.name = name
    b.input = input_dict
    return b


def _response(stop_reason: str, content: list):
    r = MagicMock()
    r.stop_reason = stop_reason
    r.content = content
    return r


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(
        name="echo",
        func=lambda msg: f"echo: {msg}",
        description="echo a message",
        input_schema={
            "properties": {"msg": {"type": "string"}},
            "required": ["msg"],
        },
    )
    return reg


@pytest.fixture
def agent(registry, tmp_path, monkeypatch):
    monkeypatch.setattr("agent.logger.LOG_DIR", tmp_path / "logs")
    client = MagicMock()
    mem = FailureMemory(path=tmp_path / "memory.json")
    a = SelfHealingAgent(
        client=client,
        model="claude-opus-4-6",
        tools=registry,
        max_retries=3,
        max_turns=10,
        verbose=False,
        enable_memory=True,
        enable_sandbox=False,
        print_report=False,
    )
    a.memory = mem
    return a


# ── Scenario 1: tool succeeds immediately ─────────────────────────────────────

def test_tool_success_no_reflection(agent):
    agent.client.messages.create.side_effect = [
        _response("tool_use", [_tool_use_block("t1", "echo", {"msg": "hi"})]),
        _response("end_turn", [_text_block("Done")]),
    ]

    run = agent.run("test task")
    assert run.final_answer == "Done"
    assert run.total_reflections == 0
    assert run.memory_hits == 0


# ── Scenario 2: tool fails once, reflection fixes it ─────────────────────────

def test_tool_failure_triggers_reflection(agent, tmp_path):
    failing_registry = ToolRegistry()
    call_count = {"n": 0}

    def flaky_read(path):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise FileNotFoundError(f"No such file: {path}")
        return "file contents"

    failing_registry.register(
        name="read_file",
        func=flaky_read,
        description="read a file",
        input_schema={
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )
    agent.tools = failing_registry

    # Mock reflection call
    mock_reflection = MagicMock()
    mock_reflection.root_cause = "File not found"
    mock_reflection.self_criticism = "Assumed wrong filename"
    mock_reflection.new_strategy = ["Try listing directory first"]
    mock_reflection.confidence = 80

    agent.client.messages.create.side_effect = [
        # Turn 1: try read_file → will fail
        _response("tool_use", [_tool_use_block("t1", "read_file", {"path": "/tmp/wrong.csv"})]),
        # Turn 2: after reflection, try again → will succeed
        _response("tool_use", [_tool_use_block("t2", "read_file", {"path": "/tmp/correct.csv"})]),
        # Turn 3: done
        _response("end_turn", [_text_block("All done")]),
    ]

    with patch.object(agent.reflector, "reflect", return_value=mock_reflection):
        run = agent.run("read a file")

    assert run.total_reflections == 1
    assert run.final_answer == "All done"
    assert len(run.tool_attempts) == 1


# ── Scenario 3: memory hit skips reflection ───────────────────────────────────

def test_memory_hit_skips_reflection(agent):
    # Pre-load memory with a known fix.
    # str(FileNotFoundError("No such file: /tmp/wrong.csv")) == "No such file: /tmp/wrong.csv"
    agent.memory.record(
        tool_name="read_file",
        error_msg="No such file: /tmp/wrong.csv",
        successful_strategy=["Try listing the directory first"],
    )

    fail_count = {"n": 0}

    def flaky(path):
        fail_count["n"] += 1
        if fail_count["n"] == 1:
            raise FileNotFoundError(f"No such file: {path}")
        return "contents"

    agent.tools = ToolRegistry()
    agent.tools.register(
        "read_file", flaky, "read",
        {"properties": {"path": {"type": "string"}}, "required": ["path"]},
    )

    agent.client.messages.create.side_effect = [
        _response("tool_use", [_tool_use_block("t1", "read_file", {"path": "/tmp/wrong.csv"})]),
        _response("tool_use", [_tool_use_block("t2", "read_file", {"path": "/tmp/other.csv"})]),
        _response("end_turn", [_text_block("done")]),
    ]

    with patch.object(agent.reflector, "reflect") as mock_reflect:
        run = agent.run("task")

    mock_reflect.assert_not_called()
    assert run.memory_hits == 1


# ── Scenario 4: max_retries exceeded ─────────────────────────────────────────

def test_max_retries_gives_up(agent):
    agent.tools = ToolRegistry()
    agent.tools.register(
        "broken_tool",
        lambda: (_ for _ in ()).throw(RuntimeError("always fails")),
        "always fails",
        {"properties": {}, "required": []},
    )
    agent.max_retries = 2

    tool_block = _tool_use_block("t1", "broken_tool", {})

    agent.client.messages.create.side_effect = [
        _response("tool_use", [tool_block]),   # attempt 1 → fail
        _response("tool_use", [_tool_use_block("t2", "broken_tool", {})]),  # attempt 2 → fail
        _response("tool_use", [_tool_use_block("t3", "broken_tool", {})]),  # attempt 3 → give up
        _response("end_turn", [_text_block("gave up")]),
    ]

    with patch.object(agent.reflector, "reflect") as mock_reflect:
        mock_r = MagicMock()
        mock_r.root_cause = "always fails"
        mock_r.self_criticism = "x"
        mock_r.new_strategy = ["try differently"]
        mock_r.confidence = 70
        mock_reflect.return_value = mock_r
        run = agent.run("task")

    assert run.final_answer == "gave up"


# ── Scenario 5: max_turns exceeded ───────────────────────────────────────────

def test_max_turns_exceeded(agent):
    # Agent keeps returning tool_use forever
    agent.max_turns = 3
    agent.tools = ToolRegistry()
    agent.tools.register(
        "echo",
        lambda msg: f"echo: {msg}",
        "echo",
        {"properties": {"msg": {"type": "string"}}, "required": ["msg"]},
    )

    agent.client.messages.create.return_value = _response(
        "tool_use",
        [_tool_use_block("t1", "echo", {"msg": "hi"})],
    )

    run = agent.run("infinite task")
    assert "max_turns" in run.final_answer
    assert run.run_log.success is False


# ── Scenario 6: O4 retry_counts keyed by tool_name ───────────────────────────

def test_retry_counts_by_tool_name(agent):
    """
    O4 fix: different tool_use_ids for same tool_name should share the retry counter.
    """
    attempt = {"n": 0}

    def always_fail(path):
        attempt["n"] += 1
        raise FileNotFoundError(f"No such file: {path}")

    agent.tools = ToolRegistry()
    agent.tools.register(
        "read_file", always_fail, "read",
        {"properties": {"path": {"type": "string"}}, "required": ["path"]},
    )
    agent.max_retries = 2

    # Claude uses a different tool_use_id each time (simulating real retries)
    agent.client.messages.create.side_effect = [
        _response("tool_use", [_tool_use_block("id_1", "read_file", {"path": "/a"})]),
        _response("tool_use", [_tool_use_block("id_2", "read_file", {"path": "/b"})]),
        _response("tool_use", [_tool_use_block("id_3", "read_file", {"path": "/c"})]),
        _response("end_turn", [_text_block("gave up")]),
    ]

    with patch.object(agent.reflector, "reflect") as mock_r:
        r = MagicMock()
        r.root_cause = "err"; r.self_criticism = "x"
        r.new_strategy = ["try differently"]; r.confidence = 70
        mock_r.return_value = r
        run = agent.run("task")

    # With O4 fix: attempt 3 should hit max_retries and call reflect only twice
    assert mock_r.call_count <= 2

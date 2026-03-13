"""T1 — Unit tests: RunLogger"""

import json
from pathlib import Path

import pytest

from agent.logger import RunLogger, RunLog, ToolCallLog


@pytest.fixture
def logger(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.logger.LOG_DIR", tmp_path / "logs")
    return RunLogger(run_id="test001", task="test task", model="claude-opus-4-6")


def test_tool_success_increments_count(logger):
    logger.tool_start()
    logger.tool_success("read_file", {"path": "/tmp/a.txt"})
    log = logger.finish(success=True, final_answer="done")
    assert log.total_tool_calls == 1
    assert log.tool_calls[0].success is True
    assert log.tool_calls[0].tool_name == "read_file"


def test_tool_failure_increments_reflection(logger):
    logger.tool_start()
    logger.tool_failure("read_file", {"path": "/tmp/x"}, "FileNotFoundError", reflection_depth=1)
    log = logger.finish(success=True, final_answer="done")
    assert log.total_tool_calls == 1
    assert log.total_reflections == 1
    assert log.tool_calls[0].success is False
    assert log.tool_calls[0].reflection_depth == 1


def test_memory_hit_increments(logger):
    logger.tool_start()
    logger.tool_failure("read_file", {}, "err", from_memory=True)
    log = logger.finish(success=True, final_answer="done")
    assert log.memory_hits == 1


def test_finish_writes_json(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    monkeypatch.setattr("agent.logger.LOG_DIR", log_dir)
    lg = RunLogger(run_id="abc123", task="t", model="m")
    lg.finish(success=True, final_answer="ok")
    files = list(log_dir.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["run_id"] == "abc123"
    assert data["success"] is True


def test_load_all(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    monkeypatch.setattr("agent.logger.LOG_DIR", log_dir)
    for i in range(3):
        lg = RunLogger(run_id=f"run{i}", task=f"task {i}", model="m")
        lg.finish(success=True, final_answer="ok")
    logs = RunLogger.load_all()
    assert len(logs) == 3


def test_self_heal_rate_all_success(logger):
    logger.tool_start()
    logger.tool_success("read_file", {})
    log = logger.finish(success=True, final_answer="ok")
    assert log.self_heal_rate == 100.0


def test_self_heal_rate_with_failure(logger):
    # Fail then succeed for same tool
    logger.tool_start()
    logger.tool_failure("read_file", {}, "err", reflection_depth=1)
    logger.tool_start()
    logger.tool_success("read_file", {})
    log = logger.finish(success=True, final_answer="ok")
    assert log.self_heal_rate == 100.0


def test_duration_recorded(logger):
    import time
    logger.tool_start()
    time.sleep(0.05)
    logger.tool_success("read_file", {})
    log = logger.finish(success=True, final_answer="ok")
    assert log.tool_calls[0].duration_ms >= 40

"""T1 — Unit tests: FailureMemory"""

import json
import tempfile
from pathlib import Path

import pytest

from agent.memory import FailureMemory, _normalise_error


# ── _normalise_error ──────────────────────────────────────────────────────────

def test_normalise_removes_line_numbers():
    raw = "SyntaxError: invalid syntax: line 42"
    assert ": line 42" not in _normalise_error(raw)


def test_normalise_removes_tmp_paths():
    raw = "FileNotFoundError: /tmp/tmpABC123.py not found"
    assert "/tmp/tmpABC123.py" not in _normalise_error(raw)
    assert "/tmp/<file>" in _normalise_error(raw)


def test_normalise_truncates_to_120():
    long_msg = "x" * 200
    assert len(_normalise_error(long_msg)) == 120


# ── FailureMemory ─────────────────────────────────────────────────────────────

@pytest.fixture
def mem(tmp_path):
    return FailureMemory(path=tmp_path / "memory.json")


def test_lookup_miss(mem):
    assert mem.lookup("read_file", "FileNotFoundError: no such file") is None


def test_record_and_lookup(mem):
    mem.record("read_file", "FileNotFoundError: no such file", ["Try list_directory first"])
    entry = mem.lookup("read_file", "FileNotFoundError: no such file")
    assert entry is not None
    assert entry.tool_name == "read_file"
    assert entry.successful_strategy == ["Try list_directory first"]


def test_hit_count_increments(mem):
    mem.record("read_file", "FileNotFoundError: no such file", ["step1"])
    mem.lookup("read_file", "FileNotFoundError: no such file")
    mem.lookup("read_file", "FileNotFoundError: no such file")
    entry = mem.lookup("read_file", "FileNotFoundError: no such file")
    assert entry.hit_count == 3


def test_lookup_normalises_error(mem):
    """Errors differing only by line number should match the same entry."""
    mem.record("execute_python", "SyntaxError: invalid syntax: line 5", ["fix the syntax"])
    entry = mem.lookup("execute_python", "SyntaxError: invalid syntax: line 99")
    assert entry is not None


def test_persists_to_disk(tmp_path):
    path = tmp_path / "memory.json"
    m1 = FailureMemory(path=path)
    m1.record("read_file", "FileNotFoundError", ["strategy A"])

    m2 = FailureMemory(path=path)
    entry = m2.lookup("read_file", "FileNotFoundError")
    assert entry is not None
    assert entry.successful_strategy == ["strategy A"]


def test_corrupt_file_starts_fresh(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{ invalid json {{")
    mem = FailureMemory(path=path)
    assert mem.lookup("anything", "anything") is None


def test_record_overwrites_same_key(mem):
    mem.record("read_file", "FileNotFoundError", ["old strategy"])
    mem.record("read_file", "FileNotFoundError", ["new strategy"])
    entry = mem.lookup("read_file", "FileNotFoundError")
    assert entry.successful_strategy == ["new strategy"]


def test_all_entries(mem):
    mem.record("tool_a", "error X", ["strat 1"])
    mem.record("tool_b", "error Y", ["strat 2"])
    assert len(mem.all_entries()) == 2

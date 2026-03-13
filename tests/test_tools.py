"""T1 — Unit tests: ToolRegistry + built-in tools"""

import os
import pytest

from agent.tools import (
    ToolRegistry,
    read_file,
    write_file,
    list_directory,
    execute_python,
    build_default_registry,
)


# ── ToolRegistry ──────────────────────────────────────────────────────────────

def test_register_and_execute():
    reg = ToolRegistry()
    reg.register(
        name="add",
        func=lambda a, b: a + b,
        description="add two numbers",
        input_schema={
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    )
    assert reg.execute("add", {"a": 2, "b": 3}) == 5


def test_unknown_tool_raises():
    reg = ToolRegistry()
    with pytest.raises(ValueError, match="Unknown tool"):
        reg.execute("nonexistent", {})


def test_get_definitions_returns_all():
    reg = build_default_registry()
    names = {d["name"] for d in reg.get_definitions()}
    assert {"read_file", "write_file", "list_directory", "execute_python"} <= names


# ── read_file ─────────────────────────────────────────────────────────────────

def test_read_file_success(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("world")
    assert read_file(str(f)) == "world"


def test_read_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_file("/nonexistent/path/file.txt")


# ── write_file ────────────────────────────────────────────────────────────────

def test_write_file_creates(tmp_path):
    path = str(tmp_path / "out.txt")
    result = write_file(path, "hello")
    assert "hello" in (tmp_path / "out.txt").read_text()
    assert "Successfully wrote" in result


# ── list_directory ────────────────────────────────────────────────────────────

def test_list_directory(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "subdir").mkdir()

    import json
    result = json.loads(list_directory(str(tmp_path)))
    assert "a.txt" in result["files"]
    assert "b.txt" in result["files"]
    assert "subdir" in result["directories"]


def test_list_directory_not_found():
    with pytest.raises(FileNotFoundError):
        list_directory("/nonexistent/path/")


# ── execute_python ────────────────────────────────────────────────────────────

def test_execute_python_success():
    assert execute_python("print(1+1)") == "2"


def test_execute_python_runtime_error():
    with pytest.raises(RuntimeError, match="ZeroDivisionError"):
        execute_python("print(1/0)")


def test_execute_python_syntax_error():
    with pytest.raises(RuntimeError):
        execute_python("def f(: pass")


def test_execute_python_no_output():
    assert execute_python("x = 42") == "(no output)"

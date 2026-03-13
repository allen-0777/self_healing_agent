"""T1 — Unit tests: Python Sandbox"""

import pytest

from agent.sandbox import execute_python_sandboxed, SandboxViolation, ALLOWED_MODULES


# ── AST analysis — blocked imports ────────────────────────────────────────────

@pytest.mark.parametrize("module", [
    "os", "sys", "subprocess", "shutil", "socket",
    "pickle", "ctypes", "pathlib",
])
def test_blocked_import(module):
    with pytest.raises(SandboxViolation, match=module):
        execute_python_sandboxed(f"import {module}")


def test_blocked_from_import():
    with pytest.raises(SandboxViolation):
        execute_python_sandboxed("from os import path")


def test_blocked_dunder_import():
    with pytest.raises(SandboxViolation):
        execute_python_sandboxed('__import__("os")')


def test_blocked_not_in_allowlist():
    with pytest.raises(SandboxViolation, match="not in the sandbox allow-list"):
        execute_python_sandboxed("import unknown_module_xyz")


# ── AST analysis — allowed imports ───────────────────────────────────────────

def test_allowed_math():
    out = execute_python_sandboxed("import math; print(math.floor(3.7))")
    assert out == "3"


def test_allowed_csv():
    out = execute_python_sandboxed(
        "import csv, io\n"
        "f = io.StringIO('a,b\\n1,2')\n"
        "rows = list(csv.reader(f))\n"
        "print(rows[1][0])"
    )
    assert out == "1"


def test_allowed_json():
    out = execute_python_sandboxed('import json; print(json.dumps({"x": 1}))')
    assert '"x"' in out


def test_allowed_statistics():
    out = execute_python_sandboxed(
        "import statistics; print(statistics.mean([1, 2, 3, 4, 5]))"
    )
    assert out == "3"


# ── Execution correctness ─────────────────────────────────────────────────────

def test_basic_print():
    assert execute_python_sandboxed("print('hello')") == "hello"


def test_no_output_returns_placeholder():
    assert execute_python_sandboxed("x = 1 + 1") == "(no output)"


def test_syntax_error_raises():
    with pytest.raises(SyntaxError):
        execute_python_sandboxed("def f(: pass")


def test_runtime_error_raises():
    with pytest.raises(RuntimeError, match="ZeroDivisionError"):
        execute_python_sandboxed("print(1 / 0)")


def test_timeout(monkeypatch):
    """Infinite loop should raise RuntimeError due to timeout."""
    import agent.sandbox as sb
    monkeypatch.setattr(sb, "MAX_CPU_SECONDS", 1)
    with pytest.raises(RuntimeError, match="timed out"):
        execute_python_sandboxed("while True: pass")

"""
Tool Registry & Demo Tools
---------------------------
Provides a generic ToolRegistry and a set of demo tools that mimic
real-world scenarios — including intentional failure modes so the
Self-Healing Agent has something to recover from.
"""

import os
import json
import subprocess
import tempfile
from typing import Any, Callable


# ── Registry ────────────────────────────────────────────────────────────────

class ToolRegistry:
    """Holds tool definitions (for Claude) and their Python implementations."""

    def __init__(self):
        self._tools: dict[str, dict] = {}

    def register(
        self,
        name: str,
        func: Callable,
        description: str,
        input_schema: dict,
    ) -> None:
        self._tools[name] = {
            "func": func,
            "definition": {
                "name": name,
                "description": description,
                "input_schema": {
                    "type": "object",
                    **input_schema,
                },
            },
        }

    def get_definitions(self) -> list[dict]:
        return [t["definition"] for t in self._tools.values()]

    def execute(self, name: str, inputs: dict) -> Any:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: '{name}'")
        return self._tools[name]["func"](**inputs)


# ── Demo tool implementations ────────────────────────────────────────────────

def read_file(path: str) -> str:
    """Read text content from a file."""
    with open(path, "r") as f:
        return f.read()


def write_file(path: str, content: str) -> str:
    """Write text content to a file."""
    with open(path, "w") as f:
        f.write(content)
    return f"Successfully wrote {len(content)} characters to '{path}'."


def list_directory(path: str) -> str:
    """List files in a directory."""
    entries = os.listdir(path)
    files = [e for e in entries if os.path.isfile(os.path.join(path, e))]
    dirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
    result = {"files": sorted(files), "directories": sorted(dirs)}
    return json.dumps(result, indent=2)


def execute_python(code: str) -> str:
    """
    Execute a Python snippet in a subprocess.
    Returns stdout/stderr combined. Raises on non-zero exit.
    """
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout
        if result.returncode != 0:
            raise RuntimeError(
                f"Exit code {result.returncode}\n"
                f"stderr: {result.stderr.strip()}\n"
                f"stdout: {result.stdout.strip()}"
            )
        return output.strip() or "(no output)"
    finally:
        os.unlink(tmp_path)


# ── Factory: build a pre-configured registry ────────────────────────────────

def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        name="read_file",
        func=read_file,
        description=(
            "Read the full text content of a file. "
            "Fails if the path does not exist or is not readable."
        ),
        input_schema={
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file.",
                }
            },
            "required": ["path"],
        },
    )

    registry.register(
        name="write_file",
        func=write_file,
        description="Write text content to a file, creating it if it does not exist.",
        input_schema={
            "properties": {
                "path": {"type": "string", "description": "Path to write to."},
                "content": {"type": "string", "description": "Content to write."},
            },
            "required": ["path", "content"],
        },
    )

    registry.register(
        name="list_directory",
        func=list_directory,
        description="List files and subdirectories inside a directory.",
        input_schema={
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the directory.",
                }
            },
            "required": ["path"],
        },
    )

    registry.register(
        name="execute_python",
        func=execute_python,
        description=(
            "Execute a Python code snippet. "
            "Returns stdout. Raises on syntax errors or runtime exceptions."
        ),
        input_schema={
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Valid Python 3 code to execute.",
                }
            },
            "required": ["code"],
        },
    )

    return registry

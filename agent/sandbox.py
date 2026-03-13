"""
方向五：Python 執行沙盒
------------------------
在 execute_python 前用 AST 靜態分析攔截危險 import，
並加上 timeout + 記憶體上限。
"""

import ast
import os
import resource
import subprocess
import tempfile

# ── 黑名單模組 ────────────────────────────────────────────────────────────────
BLOCKED_MODULES = {
    "os", "sys", "subprocess", "shutil", "socket",
    "http", "urllib", "requests", "httpx", "ftplib",
    "smtplib", "pickle", "ctypes", "importlib",
    "builtins", "multiprocessing", "threading",
    "signal", "pty", "atexit",
}

# 允許的標準庫白名單（其餘一律阻擋）
ALLOWED_MODULES = {
    "math", "statistics", "random", "decimal", "fractions",
    "itertools", "functools", "collections", "heapq",
    "json", "csv", "re", "string", "textwrap",
    "datetime", "time", "calendar",
    "pathlib",          # 唯讀 path 操作
    "io", "struct",
    "pprint", "copy",
    "enum", "dataclasses", "typing",
    "abc", "contextlib",
    # data science (safe)
    "numpy", "pandas", "scipy", "sklearn",
    "matplotlib", "seaborn",
}

# ── 資源上限 ─────────────────────────────────────────────────────────────────
MAX_MEMORY_MB = 256
MAX_CPU_SECONDS = 10
MAX_OUTPUT_BYTES = 64 * 1024  # 64 KB


class SandboxViolation(Exception):
    """Raised when code violates sandbox policy."""


def _check_ast(code: str) -> None:
    """
    Parse and walk the AST to block dangerous imports.
    Raises SandboxViolation on policy breach.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SyntaxError(str(e)) from e

    for node in ast.walk(tree):
        # import X  /  import X as Y
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                _assert_allowed(top)

        # from X import Y
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                _assert_allowed(top)

        # __import__("os") style
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                if node.args and isinstance(node.args[0], ast.Constant):
                    top = str(node.args[0].value).split(".")[0]
                    _assert_allowed(top)


def _assert_allowed(module_name: str) -> None:
    if module_name in BLOCKED_MODULES:
        raise SandboxViolation(
            f"Blocked: import of '{module_name}' is not allowed in sandbox."
        )
    # Modules not in the allow list are also blocked (strict whitelist)
    if module_name not in ALLOWED_MODULES:
        raise SandboxViolation(
            f"Blocked: '{module_name}' is not in the sandbox allow-list. "
            f"Allowed: {sorted(ALLOWED_MODULES)}"
        )


def _set_limits() -> None:
    """Called in subprocess before exec — sets CPU limit (memory via AS where supported)."""
    try:
        # CPU time (works on both Linux and macOS)
        resource.setrlimit(resource.RLIMIT_CPU, (MAX_CPU_SECONDS, MAX_CPU_SECONDS))
    except (ValueError, resource.error):
        pass
    try:
        # Virtual memory limit — Linux only; silently skip on macOS
        mem_bytes = MAX_MEMORY_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except (ValueError, resource.error):
        pass


def execute_python_sandboxed(code: str) -> str:
    """
    Execute Python code with:
      1. AST-based import analysis (pre-execution)
      2. Resource limits (memory + CPU via preexec_fn)
      3. Output size cap
      4. Timeout via subprocess timeout
    """
    # Step 1: static analysis
    _check_ast(code)

    # Step 2: write to temp file
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        # Step 3: run in subprocess with resource limits
        use_limits = os.name == "posix"  # resource module is POSIX-only
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=MAX_CPU_SECONDS + 2,
            preexec_fn=_set_limits if use_limits else None,
        )

        # Step 4: cap output size
        stdout = result.stdout[:MAX_OUTPUT_BYTES]
        stderr = result.stderr[:MAX_OUTPUT_BYTES]

        if result.returncode != 0:
            raise RuntimeError(
                f"Exit code {result.returncode}\n"
                f"stderr: {stderr.strip()}\n"
                f"stdout: {stdout.strip()}"
            )

        return stdout.strip() or "(no output)"

    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Execution timed out after {MAX_CPU_SECONDS} seconds."
        )
    finally:
        os.unlink(tmp_path)

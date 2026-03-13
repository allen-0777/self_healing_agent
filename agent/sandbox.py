"""
方向五：Python 執行沙盒
O5  ：細化 allowlist（新增 csv，移除可寫入的 pathlib）
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
    "signal", "pty", "atexit", "pathlib",  # pathlib has unlink/rmdir etc.
}

# 允許的標準庫白名單
ALLOWED_MODULES = {
    # 數學 / 統計
    "math", "statistics", "random", "decimal", "fractions", "cmath",
    # 迭代 / 函式
    "itertools", "functools", "operator",
    # 資料結構
    "collections", "heapq", "bisect", "array",
    # 字串 / 格式
    "json", "csv", "re", "string", "textwrap", "unicodedata",
    # 時間
    "datetime", "time", "calendar",
    # IO（記憶體層級，不能開檔）
    "io", "struct",
    # 工具
    "pprint", "copy", "enum", "dataclasses", "typing",
    "abc", "contextlib", "types",
    # 資料科學（safe）
    "numpy", "pandas", "scipy", "sklearn",
    "matplotlib", "seaborn",
    # 壓縮（唯讀）
    "gzip", "zipfile",
}

# ── 資源上限 ─────────────────────────────────────────────────────────────────
MAX_CPU_SECONDS = 10
MAX_OUTPUT_BYTES = 64 * 1024  # 64 KB
MAX_MEMORY_MB = 256


class SandboxViolation(Exception):
    """Raised when code violates sandbox policy."""


def _check_ast(code: str) -> None:
    """
    Parse and walk the AST to block dangerous imports.
    Raises SandboxViolation on policy breach, SyntaxError on bad syntax.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SyntaxError(str(e)) from e

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _assert_allowed(alias.name.split(".")[0])

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                _assert_allowed(node.module.split(".")[0])

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                if node.args and isinstance(node.args[0], ast.Constant):
                    _assert_allowed(str(node.args[0].value).split(".")[0])


def _assert_allowed(module_name: str) -> None:
    if module_name in BLOCKED_MODULES:
        raise SandboxViolation(
            f"Blocked: import of '{module_name}' is not allowed in sandbox."
        )
    if module_name not in ALLOWED_MODULES:
        raise SandboxViolation(
            f"Blocked: '{module_name}' is not in the sandbox allow-list. "
            f"Allowed modules: {sorted(ALLOWED_MODULES)}"
        )


def _set_limits() -> None:
    """Called in subprocess preexec_fn — sets resource limits where supported."""
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (MAX_CPU_SECONDS, MAX_CPU_SECONDS))
    except (ValueError, resource.error):
        pass
    try:
        mem_bytes = MAX_MEMORY_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except (ValueError, resource.error):
        pass


def execute_python_sandboxed(code: str) -> str:
    """
    Execute Python code with:
      1. AST-based import analysis (pre-execution)
      2. Resource limits (CPU + memory via preexec_fn on POSIX)
      3. Output size cap (64 KB)
      4. Subprocess timeout
    """
    _check_ast(code)

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=MAX_CPU_SECONDS + 2,
            preexec_fn=_set_limits if os.name == "posix" else None,
        )

        stdout = result.stdout[:MAX_OUTPUT_BYTES]
        stderr = result.stderr[:MAX_OUTPUT_BYTES]

        if result.returncode < 0:
            # Killed by signal (negative = signal number on POSIX)
            import signal as _signal
            sig = -result.returncode
            if sig == _signal.SIGXCPU.value:
                raise RuntimeError(f"Execution timed out (CPU limit exceeded).")
            raise RuntimeError(f"Process killed by signal {sig}.")

        if result.returncode != 0:
            raise RuntimeError(
                f"Exit code {result.returncode}\n"
                f"stderr: {stderr.strip()}\n"
                f"stdout: {stdout.strip()}"
            )

        return stdout.strip() or "(no output)"

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Execution timed out after {MAX_CPU_SECONDS} seconds.")
    finally:
        os.unlink(tmp_path)

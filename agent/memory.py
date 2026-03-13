"""
方向三：Failure Memory Store
------------------------------
把 (tool_name, error_pattern) → 成功策略 持久化到 JSON 檔。
下次遇到相似錯誤直接回傳過去的成功策略，跳過 reflection API 調用。
"""

import json
import re
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict


MEMORY_FILE = Path(__file__).parent.parent / ".failure_memory.json"

# 只保留 error 訊息的前 120 個字作為 pattern key（去除行號等易變資訊）
_PATTERN_LEN = 120


def _normalise_error(error_msg: str) -> str:
    """Strip volatile parts (line numbers, temp paths) to create a stable key."""
    # Remove line numbers like ": line 42"
    error_msg = re.sub(r": line \d+", "", error_msg)
    # Remove temp file paths
    error_msg = re.sub(r"/tmp/[^\s]+", "/tmp/<file>", error_msg)
    return error_msg[:_PATTERN_LEN].strip()


@dataclass
class MemoryEntry:
    tool_name: str
    error_pattern: str
    successful_strategy: list[str]
    hit_count: int = 0


class FailureMemory:
    """
    Persists and retrieves successful recovery strategies.

    Usage:
        memory = FailureMemory()

        # Check for a known fix before reflecting
        entry = memory.lookup(tool_name, error_msg)
        if entry:
            return entry.successful_strategy  # skip reflection

        # After a successful retry, save the winning strategy
        memory.record(tool_name, error_msg, strategy_steps)
    """

    def __init__(self, path: Path = MEMORY_FILE):
        self._path = path
        self._store: dict[str, MemoryEntry] = {}
        self._load()

    def lookup(self, tool_name: str, error_msg: str) -> MemoryEntry | None:
        """Return a stored strategy if we've seen this failure before."""
        key = self._key(tool_name, error_msg)
        entry = self._store.get(key)
        if entry:
            entry.hit_count += 1
            self._save()
        return entry

    def record(
        self,
        tool_name: str,
        error_msg: str,
        successful_strategy: list[str],
    ) -> None:
        """Save a strategy that successfully resolved this failure pattern."""
        key = self._key(tool_name, error_msg)
        self._store[key] = MemoryEntry(
            tool_name=tool_name,
            error_pattern=_normalise_error(error_msg),
            successful_strategy=successful_strategy,
        )
        self._save()

    def all_entries(self) -> list[MemoryEntry]:
        return list(self._store.values())

    # ── Private ───────────────────────────────────────────────────────────────

    def _key(self, tool_name: str, error_msg: str) -> str:
        return f"{tool_name}::{_normalise_error(error_msg)}"

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for k, v in data.items():
                self._store[k] = MemoryEntry(**v)
        except Exception:
            # Corrupt file — start fresh
            self._store = {}

    def _save(self) -> None:
        self._path.write_text(
            json.dumps(
                {k: asdict(v) for k, v in self._store.items()},
                indent=2,
            )
        )

"""
方向四：Structured Logging + 執行報告
----------------------------------------
每次 AgentRun 輸出 JSON log，支援 --report 摘要表格。
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"


@dataclass
class ToolCallLog:
    tool_name: str
    tool_input: dict
    success: bool
    error: str | None = None
    reflection_depth: int = 0        # 0 = no reflection, 1/2/3 = escalation level
    from_memory: bool = False         # True if strategy came from FailureMemory
    duration_ms: float = 0.0


@dataclass
class RunLog:
    run_id: str
    task: str
    model: str
    started_at: str
    finished_at: str = ""
    success: bool = False
    total_tool_calls: int = 0
    total_reflections: int = 0
    memory_hits: int = 0
    tool_calls: list[ToolCallLog] = field(default_factory=list)
    final_answer: str = ""

    # ── computed helpers ──────────────────────────────────────────────────────
    @property
    def self_heal_rate(self) -> float:
        """% of failed calls that were eventually recovered."""
        failures = [t for t in self.tool_calls if not t.success]
        if not failures:
            return 100.0
        recovered = sum(
            1 for t in failures
            if any(
                s.tool_name == t.tool_name and s.success
                for s in self.tool_calls
            )
        )
        return round(recovered / len(failures) * 100, 1)


class RunLogger:
    """Records structured logs for a single agent run."""

    def __init__(self, run_id: str, task: str, model: str):
        LOG_DIR.mkdir(exist_ok=True)
        self._log = RunLog(
            run_id=run_id,
            task=task,
            model=model,
            started_at=_now(),
        )
        self._tool_start: float = 0.0

    # ── event recording ───────────────────────────────────────────────────────

    def tool_start(self) -> None:
        self._tool_start = time.monotonic()

    def tool_success(self, tool_name: str, tool_input: dict) -> None:
        self._log.total_tool_calls += 1
        self._log.tool_calls.append(
            ToolCallLog(
                tool_name=tool_name,
                tool_input=tool_input,
                success=True,
                duration_ms=self._elapsed_ms(),
            )
        )

    def tool_failure(
        self,
        tool_name: str,
        tool_input: dict,
        error: str,
        reflection_depth: int = 0,
        from_memory: bool = False,
    ) -> None:
        self._log.total_tool_calls += 1
        if reflection_depth > 0:
            self._log.total_reflections += 1
        if from_memory:
            self._log.memory_hits += 1
        self._log.tool_calls.append(
            ToolCallLog(
                tool_name=tool_name,
                tool_input=tool_input,
                success=False,
                error=error,
                reflection_depth=reflection_depth,
                from_memory=from_memory,
                duration_ms=self._elapsed_ms(),
            )
        )

    def finish(self, success: bool, final_answer: str) -> RunLog:
        self._log.finished_at = _now()
        self._log.success = success
        self._log.final_answer = final_answer
        self._write()
        return self._log

    # ── report ────────────────────────────────────────────────────────────────

    @staticmethod
    def print_report(log: RunLog) -> None:
        """Print a human-readable summary table."""
        width = 60
        print(f"\n{'─'*width}")
        print(f"  📊  AGENT RUN REPORT  [{log.run_id}]")
        print(f"{'─'*width}")
        print(f"  Task:             {log.task[:50]}{'...' if len(log.task)>50 else ''}")
        print(f"  Model:            {log.model}")
        print(f"  Started:          {log.started_at}")
        print(f"  Finished:         {log.finished_at}")
        print(f"  Status:           {'✅ success' if log.success else '❌ failed'}")
        print(f"{'─'*width}")
        print(f"  Total tool calls: {log.total_tool_calls}")
        successes = sum(1 for t in log.tool_calls if t.success)
        failures  = sum(1 for t in log.tool_calls if not t.success)
        print(f"    ✅ succeeded:    {successes}")
        print(f"    ❌ failed:       {failures}")
        print(f"  Reflections:      {log.total_reflections}")
        print(f"  Memory hits:      {log.memory_hits}")
        print(f"  Self-heal rate:   {log.self_heal_rate}%")
        print(f"{'─'*width}")
        if log.tool_calls:
            print(f"  Tool call timeline:")
            for i, tc in enumerate(log.tool_calls, 1):
                status = "✅" if tc.success else "❌"
                mem    = " [MEM]" if tc.from_memory else ""
                depth  = f" [depth={tc.reflection_depth}]" if tc.reflection_depth else ""
                print(f"    {i:2}. {status} {tc.tool_name}{mem}{depth}  ({tc.duration_ms:.0f}ms)")
        print(f"{'─'*width}\n")

    @staticmethod
    def load_all() -> list[RunLog]:
        """Load all saved run logs."""
        logs = []
        for p in sorted(LOG_DIR.glob("*.json")):
            try:
                data = json.loads(p.read_text())
                # Reconstruct nested dataclasses
                data["tool_calls"] = [ToolCallLog(**tc) for tc in data.get("tool_calls", [])]
                logs.append(RunLog(**data))
            except Exception:
                pass
        return logs

    # ── private ───────────────────────────────────────────────────────────────

    def _elapsed_ms(self) -> float:
        return round((time.monotonic() - self._tool_start) * 1000, 1)

    def _write(self) -> None:
        path = LOG_DIR / f"{self._log.run_id}.json"
        data = asdict(self._log)
        path.write_text(json.dumps(data, indent=2))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

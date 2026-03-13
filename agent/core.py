"""
Self-Healing Agent Core  (v3)
==============================
O1  Memory 儲存改為反思的 new_strategy（而非只記錄 input）
O3  新增 max_turns 全局上限，防止無限迴圈
O4  retry_counts 改以 tool_name 為 key（修正每次重試 ID 變動導致計數失效的 bug）
"""

import json
import uuid
from dataclasses import dataclass, field

import anthropic

from .reflection import Reflector, LowConfidenceError, ReflectionResult
from .memory import FailureMemory
from .logger import RunLogger, RunLog
from .tools import ToolRegistry

# ── System prompt ─────────────────────────────────────────────────────────────
AGENT_SYSTEM = """\
You are a capable, self-healing AI agent.

When a tool call fails, you will receive either:
  (a) A [MEMORY STRATEGY] from past experience — follow it directly.
  (b) A [REFLECTION] with root cause, self-criticism, and new strategy.

In both cases you MUST incorporate the provided strategy before retrying.
Never repeat the exact same failing call verbatim.

If a tool fails more than {max_retries} times, acknowledge the limitation and
explain clearly what you tried and why it did not work.
"""

# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class ToolAttempt:
    tool_name: str
    tool_input: dict
    error: str
    reflection: str
    from_memory: bool = False


@dataclass
class AgentRun:
    task: str
    final_answer: str
    tool_attempts: list[ToolAttempt] = field(default_factory=list)
    total_reflections: int = 0
    memory_hits: int = 0
    run_log: RunLog | None = None


# ── Agent ─────────────────────────────────────────────────────────────────────

class MaxTurnsExceeded(Exception):
    pass


class SelfHealingAgent:

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        tools: ToolRegistry,
        max_retries: int = 3,
        max_turns: int = 30,      # O3: global turn limit
        verbose: bool = True,
        enable_memory: bool = True,
        enable_sandbox: bool = True,
        print_report: bool = True,
    ):
        self.client = client
        self.model = model
        self.tools = tools
        self.max_retries = max_retries
        self.max_turns = max_turns
        self.verbose = verbose
        self.print_report = print_report
        self.enable_sandbox = enable_sandbox
        self.reflector = Reflector(client, model)
        self.memory = FailureMemory() if enable_memory else None

    # ── Public ─────────────────────────────────────────────────────────────────

    def run(self, task: str) -> AgentRun:
        run_id = uuid.uuid4().hex[:8]
        logger = RunLogger(run_id=run_id, task=task, model=self.model)

        self._print(f"\n{'='*60}")
        self._print(f"  TASK [{run_id}]: {task}")
        self._print(f"{'='*60}\n")

        messages: list[dict] = [{"role": "user", "content": task}]
        agent_run = AgentRun(task=task, final_answer="")

        # O4: keyed by tool_name (not tool_use_id) so retries across turns are counted
        retry_counts: dict[str, int] = {}

        # O1: track last reflection strategy per tool_name for memory recording
        # {tool_name: (error_msg, new_strategy)}
        pending_reflections: dict[str, tuple[str, list[str]]] = {}

        turn = 0

        try:
            while True:
                # O3: global turn guard
                turn += 1
                if turn > self.max_turns:
                    raise MaxTurnsExceeded(
                        f"Reached max_turns={self.max_turns}. "
                        "Task did not complete in time."
                    )

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=AGENT_SYSTEM.format(max_retries=self.max_retries),
                    tools=self.tools.get_definitions(),
                    messages=messages,
                )

                messages.append({"role": "assistant", "content": response.content})

                # ── Done ──────────────────────────────────────────────────────
                if response.stop_reason == "end_turn":
                    final_text = "".join(
                        b.text for b in response.content if b.type == "text"
                    )
                    agent_run.final_answer = final_text
                    self._print(f"\n{'='*60}\n  FINAL ANSWER\n{'='*60}")
                    self._print(final_text)

                    log = logger.finish(success=True, final_answer=final_text)
                    agent_run.run_log = log
                    if self.print_report:
                        RunLogger.print_report(log)
                    return agent_run

                # ── Tool calls ────────────────────────────────────────────────
                if response.stop_reason == "tool_use":
                    tool_results = []

                    for block in response.content:
                        if block.type != "tool_use":
                            continue

                        tool_id    = block.id
                        tool_name  = block.name
                        tool_input = block.input

                        # O4: use tool_name as key instead of tool_use_id
                        attempt_no = retry_counts.get(tool_name, 0) + 1

                        self._print(f"[TOOL] {tool_name}({json.dumps(tool_input)})")
                        logger.tool_start()

                        try:
                            result = self._execute(tool_name, tool_input)

                            # O1: on success after a prior failure, record the
                            #     actual reflection strategy (not just input)
                            if tool_name in pending_reflections and self.memory:
                                prev_error, prev_strategy = pending_reflections.pop(tool_name)
                                self.memory.record(
                                    tool_name=tool_name,
                                    error_msg=prev_error,
                                    successful_strategy=prev_strategy,
                                )

                            self._print("  ✅ Success")
                            retry_counts[tool_name] = 0  # reset on success
                            logger.tool_success(tool_name, tool_input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": str(result),
                            })

                        except Exception as exc:
                            error_msg = str(exc)
                            self._print(f"  ❌ FAILED (attempt {attempt_no}): {error_msg[:120]}")

                            if attempt_no > self.max_retries:
                                self._print(f"  💀 Max retries reached — giving up.")
                                logger.tool_failure(tool_name, tool_input, error_msg)
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": (
                                        f"FATAL: Tool '{tool_name}' failed after "
                                        f"{attempt_no} attempts. Last error: {error_msg}"
                                    ),
                                    "is_error": True,
                                })
                                continue

                            # O4: increment by tool_name
                            retry_counts[tool_name] = attempt_no

                            # ── Memory lookup ─────────────────────────────────
                            mem_entry = (
                                self.memory.lookup(tool_name, error_msg)
                                if self.memory else None
                            )

                            if mem_entry:
                                self._print(
                                    f"  💾 Memory hit! "
                                    f"Reusing past strategy (seen {mem_entry.hit_count}x)"
                                )
                                strategy_text = "\n".join(
                                    f"  {i+1}. {s}"
                                    for i, s in enumerate(mem_entry.successful_strategy)
                                )
                                injection = (
                                    f"ERROR: {error_msg}\n\n"
                                    f"[MEMORY STRATEGY — from past experience]\n"
                                    f"{strategy_text}\n\n"
                                    "Apply this strategy now."
                                )
                                logger.tool_failure(
                                    tool_name, tool_input, error_msg,
                                    reflection_depth=0, from_memory=True,
                                )
                                agent_run.memory_hits += 1
                                agent_run.tool_attempts.append(ToolAttempt(
                                    tool_name=tool_name,
                                    tool_input=tool_input,
                                    error=error_msg,
                                    reflection=strategy_text,
                                    from_memory=True,
                                ))

                            else:
                                # ── Structured reflection ─────────────────────
                                self._print(f"  🔍 Reflecting (depth={attempt_no})...")
                                try:
                                    reflection: ReflectionResult = self.reflector.reflect(
                                        tool_name=tool_name,
                                        tool_input=tool_input,
                                        error_message=error_msg,
                                        task=task,
                                        attempt_number=attempt_no,
                                    )
                                    reflection_text = self.reflector.format_for_injection(reflection)
                                    self._print(f"\n  [REFLECTION]\n{reflection_text}\n")

                                    # O1: store new_strategy for memory recording on success
                                    pending_reflections[tool_name] = (
                                        error_msg,
                                        reflection.new_strategy,
                                    )

                                    injection = (
                                        f"ERROR: {error_msg}\n\n"
                                        f"[REFLECTION — read this before retrying]\n"
                                        f"{reflection_text}\n\n"
                                        "Adjust your strategy based on the reflection above."
                                    )
                                    logger.tool_failure(
                                        tool_name, tool_input, error_msg,
                                        reflection_depth=attempt_no,
                                    )
                                    agent_run.total_reflections += 1
                                    agent_run.tool_attempts.append(ToolAttempt(
                                        tool_name=tool_name,
                                        tool_input=tool_input,
                                        error=error_msg,
                                        reflection=reflection_text,
                                    ))

                                except LowConfidenceError as lce:
                                    self._print(
                                        f"  🚫 Confidence gate "
                                        f"({lce.reflection.confidence}/100 < {35}). "
                                        "Aborting."
                                    )
                                    logger.tool_failure(
                                        tool_name, tool_input, error_msg,
                                        reflection_depth=attempt_no,
                                    )
                                    injection = (
                                        f"FATAL: Tool '{tool_name}' failed and reflection "
                                        f"confidence is too low to retry "
                                        f"({lce.reflection.confidence}/100). "
                                        "Please explain the limitation to the user."
                                    )

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": injection,
                                "is_error": True,
                            })

                    messages.append({"role": "user", "content": tool_results})

        except MaxTurnsExceeded as e:
            self._print(f"\n⏹  {e}")
            agent_run.final_answer = str(e)
            log = logger.finish(success=False, final_answer=str(e))
            agent_run.run_log = log
            if self.print_report:
                RunLogger.print_report(log)
            return agent_run

        except Exception as e:
            log = logger.finish(success=False, final_answer=str(e))
            agent_run.run_log = log
            if self.print_report:
                RunLogger.print_report(log)
            raise

    # ── Private ────────────────────────────────────────────────────────────────

    def _execute(self, tool_name: str, tool_input: dict):
        if self.enable_sandbox and tool_name == "execute_python":
            from .sandbox import execute_python_sandboxed
            return execute_python_sandboxed(tool_input["code"])
        return self.tools.execute(tool_name, tool_input)

    def _print(self, msg: str) -> None:
        if self.verbose:
            print(msg)

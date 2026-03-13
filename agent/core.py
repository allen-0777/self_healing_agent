"""
Self-Healing Agent Core
------------------------
Implements the agentic loop with built-in Reflection/Self-Criticism.

Architecture
============
                ┌─────────────────────────────────┐
                │          User Task               │
                └──────────────┬──────────────────┘
                               │
                               ▼
                     ┌─────────────────┐
                     │  Claude decides │ ◄─────────────────────┐
                     │  what to do     │                       │
                     └────────┬────────┘                       │
                              │ tool_use                       │
                              ▼                                │
                     ┌─────────────────┐  ✅ success          │
                     │  Execute Tool   │ ─────────────────────►│
                     └────────┬────────┘                       │
                              │ ❌ failure                     │
                              ▼                                │
                  ┌───────────────────────┐                    │
                  │  REFLECTION CALL      │                    │
                  │  (separate API call)  │                    │
                  │  • Root Cause         │                    │
                  │  • Self-Criticism     │                    │
                  │  • New Strategy       │                    │
                  └───────────┬───────────┘                    │
                              │                                │
                              ▼                                │
                  ┌───────────────────────┐                    │
                  │ Inject reflection     │────────────────────┘
                  │ back into convo       │  (Claude reads it and
                  └───────────────────────┘   tries again)
"""

import json
from dataclasses import dataclass, field
from typing import Any

import anthropic

from .reflection import Reflector
from .tools import ToolRegistry


# ── System prompt ────────────────────────────────────────────────────────────
AGENT_SYSTEM = """\
You are a capable, self-healing AI agent.

When a tool call fails, you will receive:
  1. The error message as a tool_result with is_error=True
  2. A [REFLECTION] block containing your own self-analysis of the failure

You MUST read the reflection carefully and adjust your strategy accordingly.
Never repeat the exact same failing call — always incorporate the reflection's
"New Strategy" before retrying.

If a tool fails more than {max_retries} times, acknowledge the limitation and
explain what you tried and why it did not work.
"""

# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ToolAttempt:
    tool_name: str
    tool_input: dict
    error: str
    reflection: str


@dataclass
class AgentRun:
    task: str
    final_answer: str
    tool_attempts: list[ToolAttempt] = field(default_factory=list)
    total_reflections: int = 0


# ── Agent ────────────────────────────────────────────────────────────────────

class SelfHealingAgent:
    """
    An agentic loop that automatically reflects on tool failures
    and retries with an improved strategy — up to max_retries times.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        tools: ToolRegistry,
        max_retries: int = 3,
        verbose: bool = True,
    ):
        self.client = client
        self.model = model
        self.tools = tools
        self.max_retries = max_retries
        self.verbose = verbose
        self.reflector = Reflector(client, model)

    # ── Public ----------------------------------------------------------------

    def run(self, task: str) -> AgentRun:
        """Run the agent on a task and return a structured AgentRun result."""
        self._print(f"\n{'='*60}")
        self._print(f"  TASK: {task}")
        self._print(f"{'='*60}\n")

        messages: list[dict] = [{"role": "user", "content": task}]
        run = AgentRun(task=task, final_answer="")
        # Per-invocation retry counters: keyed by tool_use_id
        retry_counts: dict[str, int] = {}

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=AGENT_SYSTEM.format(max_retries=self.max_retries),
                tools=self.tools.get_definitions(),
                messages=messages,
            )

            # Always append the assistant turn
            messages.append({"role": "assistant", "content": response.content})

            # ── Done ────────────────────────────────────────────────────────
            if response.stop_reason == "end_turn":
                final_text = ""
                for block in response.content:
                    if block.type == "text":
                        final_text += block.text
                run.final_answer = final_text
                self._print(f"\n{'='*60}")
                self._print("  FINAL ANSWER")
                self._print(f"{'='*60}")
                self._print(final_text)
                return run

            # ── Tool calls ──────────────────────────────────────────────────
            if response.stop_reason == "tool_use":
                tool_results = []

                for block in response.content:
                    if block.type != "tool_use":
                    	continue

                    tool_id = block.id
                    tool_name = block.name
                    tool_input = block.input

                    self._print(
                        f"[TOOL] {tool_name}({json.dumps(tool_input)})"
                    )

                    attempt_no = retry_counts.get(tool_id, 0) + 1

                    try:
                        result = self.tools.execute(tool_name, tool_input)
                        self._print(f"  ✅ Success")
                        retry_counts[tool_id] = 0
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": str(result),
                            }
                        )

                    except Exception as exc:
                        error_msg = str(exc)
                        self._print(f"  ❌ FAILED (attempt {attempt_no}): {error_msg}")

                        if attempt_no > self.max_retries:
                            # Give up on this tool
                            self._print(
                                f"  💀 Max retries ({self.max_retries}) reached — giving up."
                            )
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": (
                                        f"FATAL: Tool '{tool_name}' failed after "
                                        f"{attempt_no} attempts. "
                                        f"Last error: {error_msg}"
                                    ),
                                    "is_error": True,
                                }
                            )
                            continue

                        # ── Reflection step ──────────────────────────────
                        self._print(f"\n  🔍 Triggering reflection...")
                        reflection_text = self.reflector.reflect(
                            tool_name=tool_name,
                            tool_input=tool_input,
                            error_message=error_msg,
                            task=task,
                            attempt_number=attempt_no,
                        )
                        self._print(f"\n  [REFLECTION]\n{reflection_text}\n")

                        run.tool_attempts.append(
                            ToolAttempt(
                                tool_name=tool_name,
                                tool_input=tool_input,
                                error=error_msg,
                                reflection=reflection_text,
                            )
                        )
                        run.total_reflections += 1
                        retry_counts[tool_id] = attempt_no

                        # Inject reflection into the tool result so Claude
                        # sees it on the very next turn and can self-correct.
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": (
                                    f"ERROR: {error_msg}\n\n"
                                    f"[REFLECTION — read this before retrying]\n"
                                    f"{reflection_text}\n\n"
                                    "Adjust your strategy based on the reflection above."
                                ),
                                "is_error": True,
                            }
                        )

                messages.append({"role": "user", "content": tool_results})

    # ── Helpers ---------------------------------------------------------------

    def _print(self, msg: str) -> None:
        if self.verbose:
            print(msg)

"""
Reflection / Self-Criticism Module
-----------------------------------
方向一：Structured Reflection (Pydantic) + Confidence Gate
方向二：Escalating Reflection Depth
O2  ：改用 client.messages.parse() 取代手動 JSON 解析
"""

import json
import anthropic
from pydantic import BaseModel, Field

# ── Structured output schema ─────────────────────────────────────────────────

class ReflectionResult(BaseModel):
    root_cause: str = Field(description="1-2 sentences: what specifically went wrong")
    self_criticism: str = Field(description="what assumption or mistake led to the bad approach")
    new_strategy: list[str] = Field(description="numbered list of specific, actionable next steps")
    confidence: int = Field(description="0-100, how confident the new strategy will succeed", ge=0, le=100)

# ── Prompt templates ─────────────────────────────────────────────────────────

REFLECTION_SYSTEM = """\
You are a self-critical AI assistant in reflection mode.
A tool call just failed. Analyse the failure carefully and respond with a
structured JSON object describing the root cause, self-criticism, new strategy,
and your confidence level (0-100).
"""

REFLECTION_TEMPLATE = """\
## Tool Failure Report

**Tool:** `{tool_name}`
**Input Provided:**
```json
{tool_input}
```
**Error:**
```
{error_message}
```

## Original Task
{task}

## Attempt History
{attempt_history}

Reflect on this failure and provide your structured analysis.
"""

# Confidence below this threshold → skip retry, give up immediately
CONFIDENCE_GATE = 35


class LowConfidenceError(Exception):
    """Raised when reflection confidence is below the gate threshold."""
    def __init__(self, message: str, reflection: ReflectionResult):
        super().__init__(message)
        self.reflection = reflection


class Reflector:
    """
    Calls Claude with a self-criticism prompt to analyse tool failures.

    方向二 — Escalating depth:
      attempt 1 → Haiku  (輕量、便宜)
      attempt 2 → agent model  (標準)
      attempt 3 → agent model + adaptive thinking  (深度)

    O2 — 使用 client.messages.parse() 取代手動 JSON 解析，
          讓 SDK 保證輸出格式符合 ReflectionResult schema。
    """

    HAIKU = "claude-haiku-4-5"

    def __init__(self, client: anthropic.Anthropic, agent_model: str):
        self.client = client
        self.agent_model = agent_model

    def reflect(
        self,
        tool_name: str,
        tool_input: dict,
        error_message: str,
        task: str,
        attempt_number: int,
    ) -> ReflectionResult:
        """
        Returns a parsed ReflectionResult.
        Raises LowConfidenceError if confidence < CONFIDENCE_GATE (caller should give up).
        """
        history_line = (
            f"Attempt #{attempt_number} — this tool has already failed before."
            if attempt_number > 1
            else "This is the first failure."
        )

        user_message = REFLECTION_TEMPLATE.format(
            tool_name=tool_name,
            tool_input=json.dumps(tool_input, indent=2),
            error_message=error_message,
            task=task,
            attempt_history=history_line,
        )

        model, extra_kwargs = self._pick_depth(attempt_number)

        # O2: use parse() — SDK guarantees structured output, no manual JSON stripping
        response = self.client.messages.parse(
            model=model,
            max_tokens=1024,
            system=REFLECTION_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
            output_format=ReflectionResult,
            **extra_kwargs,
        )

        result = response.parsed_output
        if result is None:
            raise ValueError("Reflection parse returned None — model may have refused.")

        if result.confidence < CONFIDENCE_GATE:
            raise LowConfidenceError(
                f"Reflection confidence {result.confidence} < {CONFIDENCE_GATE}. "
                "Giving up on this tool call.",
                result,
            )

        return result

    def _pick_depth(self, attempt_number: int) -> tuple[str, dict]:
        """Return (model, extra_kwargs) based on escalation level."""
        if attempt_number == 1:
            return self.HAIKU, {}
        elif attempt_number == 2:
            return self.agent_model, {}
        else:
            return self.agent_model, {"thinking": {"type": "adaptive"}}

    def format_for_injection(self, result: ReflectionResult) -> str:
        """Format a ReflectionResult as human-readable text to inject into conversation."""
        steps = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(result.new_strategy))
        return (
            f"**Root Cause:** {result.root_cause}\n"
            f"**Self-Criticism:** {result.self_criticism}\n"
            f"**New Strategy:**\n{steps}\n"
            f"**Confidence:** {result.confidence}/100"
        )

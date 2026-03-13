"""
Reflection / Self-Criticism Module
-----------------------------------
方向一：Structured Reflection + Confidence Gate
方向二：Escalating Reflection Depth（依失敗次數升級深度）
"""

import json
import anthropic
from pydantic import BaseModel

# ── Structured output schema ─────────────────────────────────────────────────

class ReflectionResult(BaseModel):
    root_cause: str
    self_criticism: str
    new_strategy: list[str]
    confidence: int  # 0-100

# ── Prompt templates ─────────────────────────────────────────────────────────

REFLECTION_SYSTEM = """\
You are a self-critical AI assistant in reflection mode.
A tool call just failed. Analyse the failure carefully and respond with a
structured JSON object matching this schema:

{
  "root_cause":     "<1-2 sentences: what specifically went wrong>",
  "self_criticism": "<what assumption or mistake led to the bad approach>",
  "new_strategy":   ["<step 1>", "<step 2>", ...],
  "confidence":     <integer 0-100, how confident the new strategy will succeed>
}

Respond with ONLY the JSON object — no markdown fences, no extra text.
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


class Reflector:
    """
    Calls Claude with a self-criticism prompt to analyse tool failures.

    方向二 — Escalating depth:
      attempt 1 → lightweight (Haiku, no thinking)
      attempt 2 → standard   (same model as agent)
      attempt 3 → deep       (adaptive thinking enabled)
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
        Raises ValueError if confidence < CONFIDENCE_GATE (caller should give up).
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

        response = self.client.messages.create(
            model=model,
            max_tokens=1024,
            system=REFLECTION_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
            **extra_kwargs,
        )

        raw = response.content[0].text.strip()
        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = "\n".join(raw.split("\n")[:-1])

        result = ReflectionResult.model_validate_json(raw)

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
            # Light: cheap & fast
            return self.HAIKU, {}
        elif attempt_number == 2:
            # Standard: same model as agent
            return self.agent_model, {}
        else:
            # Deep: adaptive thinking
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


class LowConfidenceError(Exception):
    """Raised when reflection confidence is below the gate threshold."""
    def __init__(self, message: str, reflection: ReflectionResult):
        super().__init__(message)
        self.reflection = reflection

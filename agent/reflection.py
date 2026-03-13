"""
Reflection / Self-Criticism Module
-----------------------------------
Implements the "Reflection" prompting architecture:
When a tool fails, this module asks Claude to step back, analyze the
failure, self-critique the original approach, and propose a concrete
alternative strategy before retrying.
"""

import json
import anthropic

# ── System prompt for the Reflector ────────────────────────────────────────
REFLECTION_SYSTEM = """\
You are a self-critical AI assistant in reflection mode.
A tool call just failed. Your job is to reason carefully about WHY it failed
and generate an improved, concrete strategy.

Structure your reflection EXACTLY as follows:

**Root Cause:** (1-2 sentences on what specifically went wrong)
**Self-Criticism:** (what assumption or mistake led to the bad approach)
**New Strategy:** (a numbered list of specific, actionable next steps)
**Confidence:** (0-100, how confident are you the new strategy will succeed?)
"""

# ── User message template ───────────────────────────────────────────────────
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

Please reflect on this failure and propose a new strategy.
"""


class Reflector:
    """
    Calls Claude with a self-criticism prompt to analyse tool failures
    and propose recovery strategies.
    """

    def __init__(self, client: anthropic.Anthropic, model: str):
        self.client = client
        self.model = model

    def reflect(
        self,
        tool_name: str,
        tool_input: dict,
        error_message: str,
        task: str,
        attempt_number: int,
    ) -> str:
        """
        Make a dedicated reflection API call.
        Returns a structured reflection string to inject back into the
        main conversation so the agent can self-correct.
        """
        history_line = (
            f"Attempt #{attempt_number} — previous attempts also failed."
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

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=REFLECTION_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
        )

        return response.content[0].text

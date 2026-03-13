"""
Self-Healing Agent — Demo Entry Point
=======================================
Demonstrates the Reflection/Self-Criticism architecture in action.

Scenario
--------
We ask the agent to read "sales.csv" from the demo directory, compute some
statistics, and save a report.  But the file is actually called
"revenue_2024.csv" — the agent will:

  1. Try  read_file("...sales.csv")          →  ❌ FileNotFoundError
  2. Reflect on the failure                  →  🔍 "I should list the dir first"
  3. Call list_directory(demo_dir)           →  ✅ sees 'revenue_2024.csv'
  4. Try  read_file("...revenue_2024.csv")   →  ✅ reads the data
  5. Write Python to parse CSV & compute     →  (may need a fix too)
  6. Save a report                           →  ✅ done

Run
---
  python examples/setup_demo.py   # create demo data (once)
  python main.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import anthropic

# Make sure local package is importable
sys.path.insert(0, str(Path(__file__).parent))

from agent import SelfHealingAgent
from agent.tools import build_default_registry

load_dotenv()

DEMO_DIR = "/tmp/self_healing_demo"
MODEL = "claude-opus-4-6"


def demo_file_recovery():
    """
    Scenario 1 — File name mismatch recovery.
    The agent asks for 'sales.csv' but the real file is 'revenue_2024.csv'.
    """
    print("\n" + "▓" * 60)
    print("  DEMO 1 — File Recovery (Wrong Filename)")
    print("▓" * 60)

    task = (
        f"Read the file '{DEMO_DIR}/sales.csv', calculate the total and "
        f"average revenue across all months, then save a summary report to "
        f"'{DEMO_DIR}/report.txt'."
    )

    client = anthropic.Anthropic()
    tools = build_default_registry()
    agent = SelfHealingAgent(client, MODEL, tools, max_retries=3, verbose=True)

    run = agent.run(task)

    print(f"\n📊 Reflection count: {run.total_reflections}")
    for i, attempt in enumerate(run.tool_attempts, 1):
        print(f"\n  Failure #{i}:")
        print(f"    Tool:  {attempt.tool_name}")
        print(f"    Error: {attempt.error[:80]}...")


def demo_code_fix():
    """
    Scenario 2 — Buggy code recovery.
    The agent produces Python code with a bug; the reflection loop fixes it.
    """
    print("\n" + "▓" * 60)
    print("  DEMO 2 — Python Code Auto-Fix")
    print("▓" * 60)

    task = (
        "Write and execute a Python script that reads the numbers "
        "[10, 0, 5, 20, 0, 8] and computes the result of 100 divided by "
        "each number, printing 'skipped' for zeros. "
        "Make sure the code handles ZeroDivisionError gracefully."
    )

    client = anthropic.Anthropic()
    tools = build_default_registry()
    agent = SelfHealingAgent(client, MODEL, tools, max_retries=3, verbose=True)

    run = agent.run(task)

    print(f"\n📊 Reflection count: {run.total_reflections}")


if __name__ == "__main__":
    # Check demo data exists
    if not os.path.isdir(DEMO_DIR):
        print(f"⚠️  Demo data not found. Run first:")
        print(f"   python examples/setup_demo.py")
        sys.exit(1)

    import argparse

    parser = argparse.ArgumentParser(description="Self-Healing Agent Demo")
    parser.add_argument(
        "--demo",
        choices=["file", "code", "all"],
        default="all",
        help="Which demo to run (default: all)",
    )
    args = parser.parse_args()

    if args.demo in ("file", "all"):
        demo_file_recovery()

    if args.demo in ("code", "all"):
        demo_code_fix()

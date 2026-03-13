"""
Self-Healing Agent — Demo Entry Point  (v2)
============================================

Demos
-----
  --demo file    File name mismatch recovery
  --demo code    Python code auto-fix with sandbox
  --demo memory  Run 'file' twice to show memory hit on 2nd run
  --report       Print aggregate stats from all saved run logs
  --demo all     Run file + code
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import anthropic

sys.path.insert(0, str(Path(__file__).parent))

from agent import SelfHealingAgent, RunLogger
from agent.tools import build_default_registry

load_dotenv()

DEMO_DIR = "/tmp/self_healing_demo"
MODEL    = "claude-opus-4-6"


def _agent() -> SelfHealingAgent:
    client = anthropic.Anthropic()
    return SelfHealingAgent(
        client=client,
        model=MODEL,
        tools=build_default_registry(),
        max_retries=3,
        verbose=True,
        enable_memory=True,
        enable_sandbox=True,
        print_report=True,
    )


def demo_file_recovery():
    """Agent asks for 'sales.csv' but only 'revenue_2024.csv' exists."""
    print("\n" + "▓"*60)
    print("  DEMO 1 — File Recovery (Wrong Filename)")
    print("▓"*60)
    _agent().run(
        f"Read the file '{DEMO_DIR}/sales.csv', calculate the total and "
        f"average revenue across all months, then save a summary report to "
        f"'{DEMO_DIR}/report.txt'."
    )


def demo_code_fix():
    """Agent writes code that might have division-by-zero; sandbox + reflection fixes it."""
    print("\n" + "▓"*60)
    print("  DEMO 2 — Python Code Auto-Fix (Sandboxed)")
    print("▓"*60)
    _agent().run(
        "Write and execute a Python script that takes the list "
        "[10, 0, 5, 20, 0, 8] and prints 100 divided by each number, "
        "printing 'skipped' for zeros. Handle ZeroDivisionError gracefully."
    )


def demo_memory():
    """Run file demo twice — second run should hit memory and skip reflection."""
    print("\n" + "▓"*60)
    print("  DEMO 3 — Memory Hit (run file demo twice)")
    print("▓"*60)
    # Remove report so second run starts fresh context
    report = Path(f"{DEMO_DIR}/report.txt")
    if report.exists():
        report.unlink()
    demo_file_recovery()

    print("\n\n🔁  Running again — expect a memory hit this time...\n")
    if report.exists():
        report.unlink()
    demo_file_recovery()


def show_aggregate_report():
    """Load all saved run logs and print aggregate stats."""
    logs = RunLogger.load_all()
    if not logs:
        print("No run logs found in ./logs/")
        return

    total_runs       = len(logs)
    total_reflections = sum(l.total_reflections for l in logs)
    total_mem_hits   = sum(l.memory_hits for l in logs)
    success_rate     = sum(1 for l in logs if l.success) / total_runs * 100

    print(f"\n{'═'*60}")
    print(f"  📈  AGGREGATE REPORT  ({total_runs} runs)")
    print(f"{'═'*60}")
    print(f"  Success rate:       {success_rate:.0f}%")
    print(f"  Total reflections:  {total_reflections}")
    print(f"  Memory hits:        {total_mem_hits}")
    print(f"{'═'*60}")
    for log in logs[-10:]:
        status = "✅" if log.success else "❌"
        print(f"  {status} [{log.run_id}] r={log.total_reflections} m={log.memory_hits}  {log.task[:45]}")
    print()


if __name__ == "__main__":
    if not os.path.isdir(DEMO_DIR):
        print(f"⚠️  Demo data not found. Run first:\n   python examples/setup_demo.py")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Self-Healing Agent Demo")
    parser.add_argument(
        "--demo",
        choices=["file", "code", "memory", "all"],
        default="all",
    )
    parser.add_argument("--report", action="store_true", help="Show aggregate log report")
    args = parser.parse_args()

    if args.report:
        show_aggregate_report()
        sys.exit(0)

    if args.demo in ("file", "all"):
        demo_file_recovery()
    if args.demo in ("code", "all"):
        demo_code_fix()
    if args.demo == "memory":
        demo_memory()

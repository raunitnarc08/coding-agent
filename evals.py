# evals.py
#
# PURPOSE: Automated evaluation framework for the coding agent.
#
# Runs a suite of tasks against the agent, scores each one,
# and produces a report showing where the agent succeeds and fails.
#
# WHY THIS EXISTS:
# Without evals, you have no way to know if a change you made
# improved or degraded the agent. You're guessing.
# With evals, every change is measurable. That's engineering.
#
# SCORING:
# Each task has a verify() function that checks the output.
# Score per task: 0 to 3 points
#   +1 task completed (status == "complete")
#   +1 output correct (verify() returns True)
#   +1 efficient (iterations <= target_iterations)
#
# Constraint tasks get an additional check:
#   +1 constraint respected (forbidden code not in written file)

import os
import json
import time
from pathlib import Path
from datetime import datetime
from agent import run_agent
from memory import AgentMemory
from router import detect_task_type, build_task_prompt
from config import SANDBOX_DIR


# ── CONSTANTS ──────────────────────────────────────────────────────────────────

MAX_ITERATIONS = 15


# ── EVAL TASK DEFINITIONS ─────────────────────────────────────────────────────
#
# Each task has:
#   name            → human readable identifier
#   input           → what you'd type to the agent
#   verify          → function that checks sandbox for correct output
#   target_iters    → how many iterations a good agent should need
#   forbidden_code  → strings that must NOT appear in written files (constraints)
#   setup           → optional function to create files before the task runs

EVAL_TASKS = [
    {
        "name": "write_fibonacci",
        "input": "Write a Python function that computes fibonacci(n) and prints fibonacci(10)",
        "verify": lambda: _check_output("fibonacci.py", "55"),
        "target_iters": 6,
        "forbidden_code": [],
    },
    {
        "name": "write_palindrome",
        "input": "Write a Python function is_palindrome(s) that returns True if s is a palindrome. Test with 'racecar' and 'hello'",
        "verify": lambda: _check_output("palindrome.py", "True") and _check_output("palindrome.py", "False"),
        "target_iters": 6,
        "forbidden_code": [],
    },
    {
        "name": "write_without_builtin_sort",
        "input": "Write a merge sort algorithm without using Python's built-in sort() or sorted(). Sort [5,2,8,1,9] and print the result.",
        "verify": lambda: _check_output("merge_sort.py", "[1, 2, 5, 8, 9]"),
        "target_iters": 8,
        "forbidden_code": [".sort(", "sorted("],
    },
    {
        "name": "debug_syntax_error",
        "input": "Debug the file buggy.py and fix all errors",
        "setup": lambda: _create_buggy_file(),
        "verify": lambda: _file_runs_successfully("buggy.py"),
        "target_iters": 7,
        "forbidden_code": [],
    },
    {
        "name": "write_without_recursion",
        "input": "Write a function that computes factorial(n) without using recursion. Test with n=5.",
        "verify": lambda: _check_output("factorial.py", "120"),
        "target_iters": 6,
        "forbidden_code": ["factorial(n-", "factorial(n -"],
    },
]


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def _check_output(filename: str, expected_substring: str) -> bool:
    """
    Run a file and check if expected_substring appears in output.
    Uses absolute path to avoid working directory issues.
    """
    import subprocess
    import sys

    sandbox = Path(SANDBOX_DIR).resolve()
    path = sandbox / filename

    if not path.exists():
        print(f"    [verify] file not found: {path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(sandbox)
        )
        output = result.stdout + result.stderr
        found = expected_substring in output
        if not found:
            print(f"    [verify] expected '{expected_substring}' in output: {repr(output[:100])}")
        return found
    except Exception as e:
        print(f"    [verify] error running {filename}: {e}")
        return False


def _file_runs_successfully(filename: str) -> bool:
    """Check that a file runs with exit code 0."""
    import subprocess
    import sys

    sandbox = Path(SANDBOX_DIR).resolve()
    path = sandbox / filename

    if not path.exists():
        print(f"    [verify] file not found: {path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(sandbox)
        )
        if result.returncode != 0:
            print(f"    [verify] exit code {result.returncode}: {result.stderr[:100]}")
        return result.returncode == 0
    except Exception as e:
        print(f"    [verify] error: {e}")
        return False


def _create_buggy_file():
    """Create a file with deliberate syntax errors for the debug eval."""
    path = Path(SANDBOX_DIR) / "buggy.py"
    # Two syntax errors:
    #   1. Missing colon after def
    #   2. Missing closing parenthesis
    buggy_code = (
        "def add_numbers(a, b)\n"
        "    result = a + b\n"
        "    return result\n\n"
        "print(add_numbers(3, 4)\n"
    )
    path.write_text(buggy_code)
    print(f"  [eval] created buggy.py with syntax errors")


def _check_forbidden_code(forbidden: list[str]) -> tuple[bool, list[str]]:
    """
    Scan all Python files in sandbox for forbidden code strings.
    Returns (all_clear, list_of_violations).

    WHY SCAN ALL FILES?
    The agent might write helper files or rename files.
    We check everything it wrote, not just the expected filename.
    """
    violations = []
    sandbox = Path(SANDBOX_DIR)

    for py_file in sandbox.glob("*.py"):
        if py_file.name == ".gitkeep":
            continue
        content = py_file.read_text()
        for forbidden_str in forbidden:
            if forbidden_str in content:
                violations.append(f"{py_file.name} contains '{forbidden_str}'")

    return len(violations) == 0, violations


def _clean_sandbox():
    """Remove all .py files from sandbox except .gitkeep between evals."""
    sandbox = Path(SANDBOX_DIR)
    for f in sandbox.glob("*.py"):
        f.unlink()
    print(f"  [eval] sandbox cleaned")


# ── SCORING ───────────────────────────────────────────────────────────────────

def score_result(task: dict, result: dict) -> dict:
    """
    Score a single task result. Returns a dict with:
      points      → total points earned (0-4 max with constraint bonus)
      max_points  → maximum possible for this task
      breakdown   → what points were earned and why
    """
    breakdown = []
    points = 0
    max_points = 3

    # Point 1: task completed without hitting max_iterations
    if result.get("status") == "complete":
        points += 1
        breakdown.append("✓ task completed")
    else:
        breakdown.append(f"✗ task failed — status: {result.get('status')}")

    # Point 2: output is correct
    verify_fn = task.get("verify")
    if verify_fn and verify_fn():
        points += 1
        breakdown.append("✓ output correct")
    else:
        breakdown.append("✗ output incorrect or missing")

    # Point 3: efficiency
    iters = result.get("iterations", MAX_ITERATIONS)
    target = task.get("target_iters", 8)
    if iters <= target:
        points += 1
        breakdown.append(f"✓ efficient ({iters} iterations, target ≤ {target})")
    else:
        breakdown.append(f"✗ too many iterations ({iters}, target ≤ {target})")

    # Bonus point: constraint respected (if task has constraints)
    forbidden = task.get("forbidden_code", [])
    if forbidden:
        max_points = 4
        all_clear, violations = _check_forbidden_code(forbidden)
        if all_clear:
            points += 1
            breakdown.append("✓ constraint respected")
        else:
            breakdown.append(f"✗ constraint violated: {violations}")

    return {
        "points": points,
        "max_points": max_points,
        "breakdown": breakdown
    }


# ── MAIN EVAL RUNNER ──────────────────────────────────────────────────────────

def run_evals(tasks: list = None, delay_between_tasks: int = 5):
    """
    Run all eval tasks and produce a scored report.

    delay_between_tasks: seconds to wait between tasks.
    Groq is generous but back-to-back agent runs can still spike usage.
    5 seconds is enough buffer.
    """
    if tasks is None:
        tasks = EVAL_TASKS

    print(f"\n{'='*60}")
    print(f"EVAL RUN — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tasks: {len(tasks)}")
    print(f"{'='*60}\n")

    results = []
    total_points = 0
    total_max = 0

    for i, task in enumerate(tasks):
        print(f"\n[EVAL {i+1}/{len(tasks)}] {task['name']}")
        print("-" * 40)

        # Clean sandbox before each task
        _clean_sandbox()

        # Run setup if task needs files pre-created
        setup_fn = task.get("setup")
        if setup_fn:
            setup_fn()

        # Run the agent on this task
        memory = AgentMemory()
        task_type = detect_task_type(task["input"])
        full_task = build_task_prompt(task["input"], task_type)

        start_time = time.time()
        result = run_agent(full_task, memory)
        elapsed = time.time() - start_time

        # Score it
        score = score_result(task, result)
        total_points += score["points"]
        total_max += score["max_points"]

        # Store for report
        results.append({
            "task": task["name"],
            "status": result.get("status"),
            "iterations": result.get("iterations"),
            "elapsed_seconds": round(elapsed, 1),
            "score": score
        })

        # Print task result
        print(f"\n  Score: {score['points']}/{score['max_points']}")
        for line in score["breakdown"]:
            print(f"    {line}")
        print(f"  Time: {elapsed:.1f}s | Iterations: {result.get('iterations')}")

        # Wait between tasks
        if i < len(tasks) - 1:
            print(f"\n  [eval] waiting {delay_between_tasks}s before next task...")
            time.sleep(delay_between_tasks)

    # ── FINAL REPORT ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"EVAL COMPLETE")
    print(f"{'='*60}")
    print(f"Total score: {total_points}/{total_max} "
          f"({100*total_points//total_max}%)")
    print(f"\nPer-task breakdown:")

    for r in results:
        score = r["score"]
        status_icon = "✓" if r["status"] == "complete" else "✗"
        print(f"  {status_icon} {r['task']:<35} "
              f"{score['points']}/{score['max_points']} pts  "
              f"{r['iterations']} iters  "
              f"{r['elapsed_seconds']}s")

    # Save results to JSON for tracking across runs
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_points": total_points,
        "total_max": total_max,
        "percentage": round(100 * total_points / total_max, 1),
        "tasks": results
    }

    report_path = Path("eval_results.json")

    # Load existing results if file exists
    if report_path.exists():
        with open(report_path) as f:
            history = json.load(f)
        if not isinstance(history, list):
            history = [history]
    else:
        history = []

    history.append(report)

    with open(report_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nResults saved to eval_results.json")
    print(f"Run count: {len(history)} "
          f"(compare runs to track improvement)")

    return report


if __name__ == "__main__":
    run_evals()
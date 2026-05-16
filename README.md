# Coding Agent

A ReAct coding agent built from scratch — no LangChain, no LangGraph.
Writes, debugs, and reviews Python code in a sandboxed environment.

## What it does

- Accepts natural language coding tasks
- Classifies intent (write / debug / review) and routes accordingly
- Writes Python code, validates syntax, executes it, verifies output
- Self-corrects on errors — reads the error, fixes the specific issue, reruns
- Enforces user constraints across sessions ("don't use recursion")
- Runs entirely on Groq's free tier (Llama 3.3 70B)

## Architecture

```
main.py      entry point, session management, persistence
agent.py     ReAct loop — think, act, observe, repeat
tools.py     tool definitions + sandboxed implementations
memory.py    conversation history + persistent constraint storage
router.py    task classifier (write/debug/review)
config.py    all settings in one place
evals.py     automated evaluation framework
sandbox/     isolated directory for agent file operations
```

## Key Design Decisions

**ReAct loop without a framework**
The agent loop is ~150 lines of Python. The model requests tool calls,
your code executes them, results go back. No magic. Every line is
readable and debuggable.

**Constraint-preserving memory compression**
Conversation history is compressed when it grows long (expensive).
But user constraints — "don't use recursion", "avoid sort()" — are
stored separately and injected into the system prompt every iteration.
Compression is lossy. Constraints are not.

**Sandboxed execution**
All file operations are confined to `sandbox/` via path traversal
prevention. All code execution uses subprocess isolation with a 15s
timeout. Imports are checked via AST parsing against an allowlist —
not a blocklist, which can always be bypassed.

**Persistent memory**
Constraints and tool call logs persist across sessions in
`agent_memory.json`. The model starts fresh each session but
remembers all rules stated in previous sessions.

## Eval Results

Automated evaluation across 5 tasks scoring correctness,
constraint adherence, and efficiency.

| Task | Score | Iterations | Time |
|---|---|---|---|
| write_fibonacci | 3/3 | 5 | 11.1s |
| write_palindrome | 3/3 | 5 | 35.6s |
| write_without_builtin_sort | 4/4 | 5 | 44.1s |
| debug_syntax_error | 3/3 | 3 | 15.8s |
| write_without_recursion | 4/4 | 2 | 9.6s |
| **Total** | **17/17 (100%)** | | |

## Setup

```bash
git clone <repo>
cd coding_agent
pip install groq
export GROQ_API_KEY="your_key_here"
python main.py
```

## Usage

```
You: Write a binary search function and test it
You: Debug the file broken.py and fix all errors
You: Review all Python files in sandbox for issues
You: Write a sorting algorithm without using sort() or sorted()
You: history
You: clear constraints
You: quit
```

## Running Evals

```bash
python evals.py
```

Results saved to `eval_results.json` for tracking across runs.

## Tech Stack

- **Model**: Llama 3.3 70B via Groq API (free tier)
- **Sandboxing**: subprocess isolation + AST import checking
- **Memory**: in-context compression + JSON persistence
- **Evals**: custom framework, no external dependencies
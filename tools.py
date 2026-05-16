# tools.py
#
# PURPOSE: Two things in one file:
#   1. TOOL DEFINITIONS — JSON schemas that tell the model what tools exist
#   2. TOOL IMPLEMENTATIONS — the actual Python that runs when a tool is called
#
# The model never sees the Python functions directly.
# It only sees the JSON schemas. Your Python runs behind the scenes.

import ast
import subprocess
import sys
from pathlib import Path
from config import (
    SANDBOX_DIR, CODE_TIMEOUT, MAX_FILE_SIZE,
    MAX_OUTPUT_SIZE, ALLOWED_MODULES
)


# ═══════════════════════════════════════════════════════════════
# PART 1: TOOL DEFINITIONS (what the model sees)
# ═══════════════════════════════════════════════════════════════
#
# Gemini's tool format is different from Anthropic's.
# Anthropic used a list of dicts with "input_schema".
# Gemini uses genai.types.Tool with FunctionDeclaration objects.
# We'll wire this up in agent.py — for now, define tools as plain dicts
# and convert them there. This keeps tools.py readable.

TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file in the sandbox directory. "
            "Use this to examine existing code before modifying it. "
            "Always read before writing — never assume file contents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename only, no path. E.g. 'main.py' not 'sandbox/main.py'"
                }
            },
            "required": ["filename"]
        }
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file in the sandbox directory. "
            "This OVERWRITES the file completely if it exists. "
            "Use read_file first if you need to preserve existing content. "
            "Only write complete, valid Python — never partial code."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename only, no path. E.g. 'solution.py'"
                },
                "content": {
                    "type": "string",
                    "description": "Complete file content to write"
                }
            },
            "required": ["filename", "content"]
        }
    },
    {
        "name": "run_python",
        "description": (
            "Execute a Python file in the sandbox and return stdout + stderr. "
            "Use this to verify your code works. Always run after writing. "
            "If you see an error, read the file, fix it, write it, run again. "
            f"Execution is limited to {CODE_TIMEOUT} seconds."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Python file to run. Must exist in sandbox."
                }
            },
            "required": ["filename"]
        }
    },
    {
        "name": "list_files",
        "description": (
            "List all files currently in the sandbox directory with their sizes. "
            "Use this at the start of a task to understand what already exists."
        ),
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "validate_python",
        "description": (
            "Check Python code for syntax errors WITHOUT running it. "
            "Use this before write_file to catch syntax errors early. "
            "Returns 'Valid' or a specific syntax error with line number."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code string to validate"
                }
            },
            "required": ["code"]
        }
    },
    {
        "name": "task_complete",
        "description": (
            "Call this when the task is fully complete and verified. "
            "Only call after running the code and confirming correct output. "
            "Provide a clear summary of what was accomplished."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "What was accomplished and how"
                },
                "files_created": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of files created or modified"
                }
            },
            "required": ["summary", "files_created"]
        }
    }
]


# ═══════════════════════════════════════════════════════════════
# PART 2: SAFETY UTILITIES
# ═══════════════════════════════════════════════════════════════

def safe_sandbox_path(filename: str) -> Path:
    """
    Converts a filename to a safe absolute path inside the sandbox.

    The attack this prevents is called PATH TRAVERSAL:
      attacker passes filename = "../../etc/passwd"
      naive code does: sandbox / "../../etc/passwd" = /etc/passwd
      we just read a system file.

    Our defense:
      Step 1: Path(filename).name strips everything before the last slash
              "../../evil.py" → "evil.py"
      Step 2: .resolve() converts to absolute path, collapses any remaining tricks
      Step 3: We verify the result actually starts with our sandbox path
    """
    sandbox = Path(SANDBOX_DIR).resolve()
    bare_filename = Path(filename).name       # strips any directory components
    target = (sandbox / bare_filename).resolve()

    if not str(target).startswith(str(sandbox)):
        raise ValueError(f"Access denied: {filename} is outside sandbox")

    return target


def check_imports(code: str) -> tuple[bool, str]:
    """
    Parses the code's Abstract Syntax Tree (AST) and checks every import.
    Rejects anything not in ALLOWED_MODULES.

    WHY AST AND NOT REGEX?
    Regex matches text patterns. It can be fooled:
      __import__('os')         — no 'import' keyword, regex misses it
      importlib.import_module  — same problem

    AST parses the actual structure of the code — what it MEANS, not what
    it looks like. You cannot fool an AST parser with clever formatting.

    We check three node types:
      ast.Import      → catches: import os
      ast.ImportFrom  → catches: from os import path
      ast.Call        → catches: __import__('os')
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True, ""  # let the syntax checker handle this separately

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split('.')[0]  # "os.path" → "os"
                if module not in ALLOWED_MODULES:
                    return False, f"Module '{module}' is not allowed"

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module = node.module.split('.')[0]
                if module not in ALLOWED_MODULES:
                    return False, f"Module '{module}' is not allowed"

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id == '__import__':
                    return False, "Dynamic imports (__import__) are not allowed"

    return True, ""


# ═══════════════════════════════════════════════════════════════
# PART 3: TOOL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════

def tool_read_file(filename: str) -> str:
    try:
        path = safe_sandbox_path(filename)

        if not path.exists():
            return f"Error: '{filename}' does not exist in sandbox"

        size = path.stat().st_size
        if size > MAX_FILE_SIZE:
            return f"Error: file too large ({size} bytes). Max is {MAX_FILE_SIZE}"

        content = path.read_text(encoding='utf-8')
        return f"Contents of {filename} ({size} bytes):\n\n{content}"

    except ValueError as e:
        return f"Security error: {e}"
    except Exception as e:
        return f"Error reading file: {e}"


def tool_write_file(filename: str, content: str) -> str:
    try:
        path = safe_sandbox_path(filename)

        if filename.endswith('.py'):
            valid, error = validate_syntax(content)
            if not valid:
                return f"Write rejected — syntax error: {error}\nFix the code and try again."

            allowed, error = check_imports(content)
            if not allowed:
                return f"Write rejected — {error}\nOnly these modules are allowed: {sorted(ALLOWED_MODULES)}"

        path.write_text(content, encoding='utf-8')
        lines = content.count('\n') + 1
        return f"Successfully wrote {filename} ({lines} lines, {len(content)} chars)"

    except ValueError as e:
        return f"Security error: {e}"
    except Exception as e:
        return f"Error writing file: {e}"


def tool_run_python(filename: str) -> str:
    """
    Runs code in a subprocess — NOT with exec().

    WHY SUBPROCESS AND NOT exec()?
    exec() runs inside your current Python process.
    If the agent's code crashes or does something bad,
    it can affect your agent process directly.

    subprocess.run() spawns a completely separate OS process.
    If it crashes, hangs, or gets killed — your agent keeps running.
    The timeout= parameter is what makes this safe:
    after CODE_TIMEOUT seconds, the OS kills the child process.
    Without timeout, an infinite loop would hang your agent forever.
    """
    try:
        path = safe_sandbox_path(filename)

        if not path.exists():
            return f"Error: '{filename}' not found. Did you write it first?"

        content = path.read_text()
        allowed, error = check_imports(content)
        if not allowed:
            return f"Execution blocked — {error}"

        result = subprocess.run(
            [sys.executable, str(path)],   # sys.executable = same Python running this agent
            capture_output=True,            # grab stdout and stderr
            text=True,                      # decode bytes → string automatically
            timeout=CODE_TIMEOUT,           # kill process if it runs too long
            cwd=Path(SANDBOX_DIR).resolve() # working directory is the sandbox
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if not output:
            output = "(no output)"

        if result.returncode != 0:
            output = f"Process exited with code {result.returncode}\n{output}"

        # Truncate huge outputs — a stack trace can be 10,000 chars
        # The model doesn't need all of it, and it wastes tokens
        if len(output) > MAX_OUTPUT_SIZE:
            output = output[:MAX_OUTPUT_SIZE] + f"\n... (truncated, {len(output)} chars total)"

        return output

    except subprocess.TimeoutExpired:
        return f"Error: execution timed out after {CODE_TIMEOUT}s. Possible infinite loop."
    except ValueError as e:
        return f"Security error: {e}"
    except Exception as e:
        return f"Error running file: {e}"


def tool_list_files() -> str:
    sandbox = Path(SANDBOX_DIR).resolve()
    files = list(sandbox.iterdir())

    if not files:
        return "Sandbox is empty."

    lines = []
    for f in sorted(files):
        if f.name == '.gitkeep':
            continue
        size = f.stat().st_size
        lines.append(f"  {f.name}  ({size} bytes)")

    return "Files in sandbox:\n" + "\n".join(lines) if lines else "Sandbox is empty."


def validate_syntax(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def tool_validate_python(code: str) -> str:
    valid, error = validate_syntax(code)
    if valid:
        allowed, import_error = check_imports(code)
        if not allowed:
            return f"Syntax OK but import blocked: {import_error}"
        return "Valid — no syntax errors, all imports allowed"
    return f"Syntax error: {error}"


# ═══════════════════════════════════════════════════════════════
# PART 4: TOOL DISPATCHER
# ═══════════════════════════════════════════════════════════════
#
# This is the bridge between the model's world and your Python.
# The model says: "call run_python with filename='solution.py'"
# This dict maps that string name to the actual function.
#
# WHY LAMBDAS?
# Each tool function has different parameters.
# The dispatcher always receives a dict of args.
# lambda args: tool_read_file(**args) unpacks that dict
# into keyword arguments automatically.
# tool_list_files takes no args, so we ignore the dict entirely.

TOOL_MAP = {
    "read_file":       lambda args: tool_read_file(**args),
    "write_file":      lambda args: tool_write_file(**args),
    "run_python":      lambda args: tool_run_python(**args),
    "list_files":      lambda args: tool_list_files(),
    "validate_python": lambda args: tool_validate_python(**args),
    "task_complete":   lambda args: None,  # None = stop signal for the agent loop
}


def execute_tool(name: str, inputs: dict) -> str | None:
    """
    Single entry point for all tool execution.
    Returns a string result, or None for task_complete.

    WHY CATCH EXCEPTIONS HERE AND RETURN STRINGS?
    If a tool crashes and raises an exception, we have two choices:
      1. Let it propagate → agent loop crashes → user sees a Python traceback
      2. Catch it → return the error as a string → model reads it and recovers

    Option 2 is always better. The model can read "Error: file not found"
    and try a different approach. It cannot recover from a crashed process.
    """
    if name not in TOOL_MAP:
        return f"Error: unknown tool '{name}'. Available: {list(TOOL_MAP.keys())}"

    try:
        return TOOL_MAP[name](inputs)
    except TypeError as e:
        return f"Error: wrong arguments for tool '{name}': {e}"
    except Exception as e:
        return f"Error executing '{name}': {e}"
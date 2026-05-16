# config.py
import os

# ── API ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL = "llama-3.3-70b-versatile"        # supports tool calling
ROUTER_MODEL = "llama-3.3-70b-versatile" # same model, groq is generous

# ── AGENT SAFETY LIMITS ───────────────────────────────────────────────────────
MAX_ITERATIONS = 12
MAX_OUTPUT_TOKENS = 2048

# ── TOOL LIMITS ───────────────────────────────────────────────────────────────
CODE_TIMEOUT = 15
MAX_FILE_SIZE = 50_000
MAX_OUTPUT_SIZE = 3_000
SANDBOX_DIR = "sandbox"

# ── MEMORY ────────────────────────────────────────────────────────────────────
WORKING_MEMORY_LIMIT = 20

# ── ALLOWED PYTHON MODULES ────────────────────────────────────────────────────
ALLOWED_MODULES = {
    "math", "random", "json", "csv", "re", "string",
    "collections", "itertools", "functools", "datetime",
    "pathlib", "typing", "dataclasses", "enum",
    "time", "copy", "pprint", "textwrap",
}
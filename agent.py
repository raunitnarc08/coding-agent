# agent.py
#
# CHANGE FROM GEMINI VERSION:
#   Uses Groq client — OpenAI-compatible spec.
#   Key differences in the loop:
#     1. Tool definitions: plain dicts with "type": "function" wrapper
#     2. Stop condition: check finish_reason == "tool_calls"
#     3. Model response: response.choices[0].message
#     4. Tool calls: message.tool_calls (list of ToolCall objects)
#     5. Tool results: role="tool" messages with tool_call_id
#     6. Memory: add_raw() for assistant messages, add_tool_result() for results

import re
import time
import json
from groq import Groq
from tools import TOOL_DEFINITIONS, execute_tool
from memory import AgentMemory
from config import (
    MODEL, MAX_ITERATIONS, MAX_OUTPUT_TOKENS, GROQ_API_KEY
)

client = Groq(api_key=GROQ_API_KEY)


# ── CONVERT TOOL DEFINITIONS TO GROQ FORMAT ───────────────────────────────────
#
# Groq/OpenAI format wraps each function in a "type": "function" object.
# Our TOOL_DEFINITIONS use "parameters" — OpenAI spec uses the same key.
# So conversion is minimal: just add the wrapper.
#
# Gemini needed types.FunctionDeclaration objects.
# Groq needs plain dicts. Much simpler.

def build_groq_tools() -> list[dict]:
    """Convert our tool definitions to Groq/OpenAI format."""
    groq_tools = []
    for tool_def in TOOL_DEFINITIONS:
        groq_tools.append({
            "type": "function",
            "function": {
                "name": tool_def["name"],
                "description": tool_def["description"],
                "parameters": tool_def.get("parameters", {})
            }
        })
    return groq_tools


GROQ_TOOLS = build_groq_tools()

SYSTEM_PROMPT = """You are an expert Python coding assistant with access to tools.

You operate in a sandbox directory where you can read, write, and run Python files.

Your workflow for ANY task:
1. THINK — reason about what you need to do (write this out explicitly)
2. ACT — use the most appropriate tool
3. OBSERVE — read the result carefully
4. REPEAT — until the task is fully working and verified

Rules:
- Always list files first to understand what exists
- Always validate syntax before writing
- Always run code after writing to verify it works
- If code fails, read the error, fix the specific issue, run again
- Only call task_complete after running the code and seeing correct output
- Never assume code works without running it
{constraint_block}"""


def call_with_retry(messages, tools, max_retries=3):
    """
    Handles three error types:
      429 rate_limit    → wait exact time from error, retry
      503 overloaded    → wait 20s, retry  
      400 tool_use_failed → model generated malformed tool call
                           → tell it to retry with proper format
    
    WHY HANDLE 400 HERE?
    400 normally means "you sent bad data" — a real bug.
    But tool_use_failed is different: the model itself generated
    something unparseable. It's a model failure, not our failure.
    We can recover by telling the model what went wrong and retrying.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_OUTPUT_TOKENS,
                tools=tools,
                messages=messages
            )
            return response

        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate_limit" in error_str.lower()
            is_overloaded = "503" in error_str
            is_tool_failed = "tool_use_failed" in error_str
            has_retries_left = attempt < max_retries - 1

            if is_tool_failed and has_retries_left:
                # Model generated a malformed tool call
                # Add a corrective message and retry
                print(f"  [agent] model generated malformed tool call — correcting...")
                messages = messages + [{
                    "role": "user",
                    "content": (
                        "Your last response had a formatting error in the tool call. "
                        "Please try again. Call exactly one tool using the proper "
                        "function calling format. Do not mix text and tool calls."
                    )
                }]
                # Small wait before retry
                time.sleep(2)

            elif (is_rate_limit or is_overloaded) and has_retries_left:
                wait_match = re.search(r'try again in ([\d.]+)s', error_str)
                wait_seconds = float(wait_match.group(1)) + 1 if wait_match else 20
                reason = "rate limited" if is_rate_limit else "server overloaded"
                print(f"  [agent] {reason} — waiting {wait_seconds:.1f}s "
                      f"(attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_seconds)

            else:
                raise


def run_agent(task: str, memory: AgentMemory) -> dict:
    print(f"\n{'='*60}")
    print(f"TASK: {task}")
    print(f"{'='*60}")

    # Extract constraints
    constraint_patterns = [
        r"don'?t\s+use\s+[^.,]+",
        r"never\s+use\s+[^.,]+",
        r"avoid\s+using\s+[^.,]+",
        r"without\s+using\s+[^.,]+",
        r"do\s+not\s+use\s+[^.,]+",
    ]
    for pattern in constraint_patterns:
        matches = re.findall(pattern, task, re.IGNORECASE)
        for match in matches:
            memory.add_constraint(match.strip())

    memory.add_message("user", task)

    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n--- Iteration {iteration}/{MAX_ITERATIONS} ---")

        memory.maybe_compress(client)

        system = SYSTEM_PROMPT.format(
            constraint_block=memory.constraint_block()
        )

        # Build full message list with system prompt prepended
        messages = [{"role": "system", "content": system}] + memory.messages

        response = call_with_retry(messages=messages, tools=GROQ_TOOLS)

        # ── PARSE RESPONSE ────────────────────────────────────────
        #
        # OpenAI/Groq spec:
        #   response.choices[0].finish_reason → why the model stopped
        #     "tool_calls" → model wants to call tools
        #     "stop"       → model finished naturally
        #     "length"     → hit max_tokens
        #
        #   response.choices[0].message → the model's response object
        #     .content    → text (may be None if only tool calls)
        #     .tool_calls → list of ToolCall objects (may be None)

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # Print reasoning text if present
        if message.content:
            print(f"  [model] {message.content.strip()[:200]}")

        # ── NO TOOL CALL ──────────────────────────────────────────
        if finish_reason == "stop" or not message.tool_calls:
            print(f"  [agent] model responded without tool call — nudging")
            # Save model response as-is, preserving exact structure
            memory.add_raw({
                "role": "assistant",
                "content": message.content or ""
            })
            memory.add_message(
                "user",
                "You stopped without completing the task. "
                "Continue working — use tools to finish and call task_complete."
            )
            continue

        # ── TOOL CALLS ────────────────────────────────────────────
        #
        # Save model message with tool_calls BEFORE executing.
        # Groq requires the assistant message with tool_calls to be
        # in history before the tool results — the IDs must match.
        # If we don't save it first, the history is invalid.
        #
        # message.tool_calls is a list of ToolCall objects:
        #   tool_call.id              → unique ID for this call
        #   tool_call.function.name   → tool name
        #   tool_call.function.arguments → JSON string of inputs

        memory.add_raw({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        })

        # Execute each tool call
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            # arguments comes as a JSON string — parse it to dict
            try:
                inputs = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                inputs = {}

            print(f"  [tool] {name}({inputs})")

            result = execute_tool(name, inputs)

            # task_complete → stop
            if result is None and name == "task_complete":
                print(f"\n{'='*60}")
                print(f"TASK COMPLETE")
                print(f"Summary: {inputs.get('summary', '')}")
                print(f"Files: {inputs.get('files_created', [])}")
                print(f"Iterations: {iteration}")
                print(f"{'='*60}")
                return {
                    "status": "complete",
                    "summary": inputs.get("summary"),
                    "files": inputs.get("files_created", []),
                    "iterations": iteration,
                }

            print(f"  [result] {str(result)[:120]}")
            memory.log_tool_call(name, inputs, result)

            # Send result back with matching tool_call_id
            # This is how the model knows which result belongs to which call
            memory.add_tool_result(
                tool_call_id=tool_call.id,
                name=name,
                result=str(result)
            )

    print(f"\n[agent] max iterations ({MAX_ITERATIONS}) reached without completion")
    return {
        "status": "max_iterations",
        "iterations": iteration,
        "result": None
    }
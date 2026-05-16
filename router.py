# router.py
#
# CHANGE FROM GEMINI VERSION:
#   Uses Groq client (OpenAI-compatible) instead of Gemini client.
#   client.chat.completions.create() instead of client.models.generate_content()
#   Response accessed via response.choices[0].message.content

import re
import time
from groq import Groq
from config import GROQ_API_KEY, ROUTER_MODEL

client = Groq(api_key=GROQ_API_KEY)


def call_with_retry(messages, max_tokens=10, max_retries=3):
    """
    Groq is generous on free tier but can still rate limit.
    Same retry pattern as before — catch 429, wait, retry.
    503 is less common on Groq but we handle it anyway.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=ROUTER_MODEL,
                max_tokens=max_tokens,
                messages=messages
            )
            return response

        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate_limit" in error_str.lower()
            is_overloaded = "503" in error_str
            has_retries_left = attempt < max_retries - 1

            if (is_rate_limit or is_overloaded) and has_retries_left:
                wait_match = re.search(r'try again in ([\d.]+)s', error_str)
                wait_seconds = float(wait_match.group(1)) + 2 if wait_match else 20
                reason = "rate limited" if is_rate_limit else "server overloaded"
                print(f"  [router] {reason} — waiting {wait_seconds:.0f}s...")
                time.sleep(wait_seconds)
            else:
                raise


def detect_task_type(user_input: str) -> str:
    """Classify request into debug / write / review."""
    response = call_with_retry(
        messages=[{
            "role": "user",
            "content": (
                f"Classify this coding request as exactly one word. "
                f"Choose exactly one of: 'debug', 'write', 'review'.\n\n"
                f"Request: {user_input}\n\n"
                f"One word only:"
            )
        }],
        max_tokens=10
    )

    classification = response.choices[0].message.content.strip().lower().strip('.,!? ')

    if classification not in ("debug", "write", "review"):
        print(f"  [router] unexpected classification '{classification}', defaulting to 'write'")
        return "write"

    return classification


def build_task_prompt(user_input: str, task_type: str) -> str:
    """Prepend workflow instructions to task."""
    prefixes = {
        "debug": (
            "DEBUG TASK: Find and fix the bug in the provided code.\n"
            "Strict order: list_files → read_file → identify the exact bug → "
            "explain what's wrong → fix it → write_file → run_python → verify output.\n"
            "Do not guess. Read the file first.\n\n"
        ),
        "write": (
            "WRITE TASK: Write Python code that solves the following problem.\n"
            "Strict order: list_files → validate_python → write_file → "
            "run_python → verify output matches requirements.\n"
            "Only call task_complete after seeing correct output.\n\n"
        ),
        "review": (
            "REVIEW TASK: Analyze the code for issues, improvements, and best practices.\n"
            "Strict order: list_files → read_file for each file → analyze thoroughly → "
            "write a review to review.txt with specific issues and line numbers.\n"
            "Be precise — vague feedback is useless.\n\n"
        )
    }
    return prefixes.get(task_type, "") + user_input
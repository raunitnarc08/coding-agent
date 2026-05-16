# memory.py
#
# PURPOSE: Conversation history + constraint preservation + persistence.
#
# PERSISTENCE STRATEGY:
#   Save: constraints, tool_call_log, session summary
#   Don't save: full message history (session-specific, confusing to restore)
#
# On startup: constraints are loaded and re-injected into system prompt
# immediately. The model starts fresh but still knows the rules.

from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
from config import WORKING_MEMORY_LIMIT

MEMORY_FILE = "agent_memory.json"


@dataclass
class AgentMemory:
    messages: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    tool_call_log: list = field(default_factory=list)
    session_count: int = 0

    def add_message(self, role: str, content):
        self.messages.append({"role": role, "content": content})

    def add_raw(self, message: dict):
        """Add pre-built message dict — for assistant messages with tool_calls."""
        self.messages.append(message)

    def add_tool_result(self, tool_call_id: str, name: str, result: str):
        """Add tool result with matching tool_call_id for Groq/OpenAI spec."""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": str(result)
        })

    def add_constraint(self, constraint: str):
        """
        Store constraint permanently.
        Checks for duplicates before adding — the same task run twice
        shouldn't double-store the same constraint.
        """
        if constraint not in self.constraints:
            self.constraints.append(constraint)
            print(f"  [memory] constraint stored: {constraint}")

    def log_tool_call(self, name: str, inputs: dict, result: str):
        self.tool_call_log.append({
            "tool": name,
            "inputs": inputs,
            "result_preview": str(result)[:200] if result else "None",
            "timestamp": datetime.now().isoformat()
        })

    def maybe_compress(self, client):
        """Compress oldest messages when history gets too long."""
        if len(self.messages) <= WORKING_MEMORY_LIMIT:
            return

        to_compress = self.messages[:6]
        self.messages = self.messages[6:]

        history_text = ""
        for m in to_compress:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            if isinstance(content, list):
                content = str(content)[:300]
            elif content is None:
                tool_calls = m.get("tool_calls", [])
                content = f"[called tools: {[tc['function']['name'] for tc in tool_calls]}]"
            history_text += f"{role}: {str(content)[:300]}\n"

        summary_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    f"Summarize what happened in these agent steps "
                    f"in 2-3 sentences. Focus on what was accomplished "
                    f"and what errors occurred:\n\n{history_text}"
                )
            }]
        )
        summary = summary_response.choices[0].message.content

        self.messages.insert(0, {
            "role": "user",
            "content": f"[Earlier steps summary]: {summary}"
        })
        self.messages.insert(1, {
            "role": "assistant",
            "content": "Understood. I'll continue from where we left off."
        })

        print(f"  [memory] compressed {len(to_compress)} messages into summary")

    def constraint_block(self) -> str:
        """Format constraints for system prompt injection."""
        if not self.constraints:
            return ""
        items = "\n".join(f"  - {c}" for c in self.constraints)
        return f"\nHARD CONSTRAINTS (never violate these):\n{items}\n"

    # ── PERSISTENCE ───────────────────────────────────────────────────────────

    def save(self):
        """
        Save persistent state to JSON.

        WHY NOT SAVE messages?
        Messages are session-specific. If you restore them in a new session,
        the model wakes up mid-conversation with no context on why it's there.
        The message history only makes sense within the session it was created.

        WHAT WE SAVE:
          constraints   → hard rules that apply across all sessions
          tool_call_log → audit trail of everything that ran
          session_count → how many sessions have been run
          last_saved    → timestamp for debugging

        This means on restart the model starts fresh BUT still knows
        all the rules the user has stated across previous sessions.
        """
        data = {
            "constraints": self.constraints,
            "tool_call_log": self.tool_call_log,
            "session_count": self.session_count,
            "last_saved": datetime.now().isoformat()
        }

        with open(MEMORY_FILE, "w") as f:
            json.dump(data, f, indent=2)

        print(f"  [memory] saved {len(self.constraints)} constraints, "
              f"{len(self.tool_call_log)} tool calls to {MEMORY_FILE}")

    def load(self):
        """
        Load persistent state from JSON.

        Safe to call even if file doesn't exist — first run starts fresh.
        Constraints are loaded and will be injected into system prompt
        on the first API call via constraint_block().
        """
        path = Path(MEMORY_FILE)

        if not path.exists():
            print(f"  [memory] no saved memory found — starting fresh")
            return

        try:
            with open(path) as f:
                data = json.load(f)

            self.constraints = data.get("constraints", [])
            self.tool_call_log = data.get("tool_call_log", [])
            self.session_count = data.get("session_count", 0)

            print(f"  [memory] loaded {len(self.constraints)} constraints, "
                  f"{len(self.tool_call_log)} tool calls from previous sessions")

            if self.constraints:
                print(f"  [memory] active constraints:")
                for c in self.constraints:
                    print(f"    - {c}")

        except json.JSONDecodeError:
            print(f"  [memory] corrupted memory file — starting fresh")
        except Exception as e:
            print(f"  [memory] error loading memory: {e} — starting fresh")

    def clear_constraints(self):
        """
        Let user explicitly clear stored constraints.
        Called from main.py when user types 'clear constraints'.
        """
        count = len(self.constraints)
        self.constraints = []
        self.save()
        print(f"  [memory] cleared {count} constraints")

    def show_history(self):
        """Print a summary of what the agent has done across sessions."""
        print(f"\n{'='*50}")
        print(f"AGENT HISTORY")
        print(f"{'='*50}")
        print(f"Sessions run: {self.session_count}")
        print(f"Total tool calls: {len(self.tool_call_log)}")
        print(f"Active constraints: {len(self.constraints)}")

        if self.constraints:
            print(f"\nConstraints:")
            for c in self.constraints:
                print(f"  - {c}")

        if self.tool_call_log:
            print(f"\nLast 5 tool calls:")
            for entry in self.tool_call_log[-5:]:
                ts = entry.get("timestamp", "unknown")[:19]
                print(f"  [{ts}] {entry['tool']}({str(entry['inputs'])[:50]})")

        print(f"{'='*50}\n")
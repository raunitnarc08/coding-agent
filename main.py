# main.py
#
# PURPOSE: Entry point. Wires router → agent → memory together.
#
# CHANGES FROM PREVIOUS VERSION:
#   - Loads memory on startup
#   - Saves memory on exit
#   - Increments session count
#   - Added commands: 'history', 'clear constraints'

from agent import run_agent
from memory import AgentMemory
from router import detect_task_type, build_task_prompt


def main():
    print("Coding Agent — type 'quit' to exit\n")
    print("Commands: 'history' | 'clear constraints' | 'quit'\n")

    memory = AgentMemory()

    # Load persistent state from previous sessions
    memory.load()
    memory.session_count += 1

    try:
        while True:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # ── COMMANDS ──────────────────────────────────────────
            if user_input.lower() in ("quit", "exit", "q"):
                memory.save()
                print(f"\nSession {memory.session_count} ended.")
                print(f"Total tool calls this session: "
                      f"{len([t for t in memory.tool_call_log])}")
                break

            if user_input.lower() == "history":
                memory.show_history()
                continue

            if user_input.lower() == "clear constraints":
                memory.clear_constraints()
                continue

            # ── NORMAL TASK ───────────────────────────────────────
            task_type = detect_task_type(user_input)
            print(f"  [router] detected: {task_type} task")

            full_task = build_task_prompt(user_input, task_type)

            result = run_agent(full_task, memory)

            # Auto-save after every task
            # WHY AUTO-SAVE?
            # If the program crashes mid-session, you don't lose everything.
            # Save incrementally, not just on clean exit.
            memory.save()

            print(f"\nResult: {result['status']}")
            if result.get('summary'):
                print(f"Summary: {result['summary']}")
            print(f"Iterations used: {result['iterations']}\n")

    except KeyboardInterrupt:
        # Ctrl+C — save before exiting
        print("\n\nInterrupted — saving memory...")
        memory.save()
        print("Memory saved. Goodbye.")


if __name__ == "__main__":
    main()
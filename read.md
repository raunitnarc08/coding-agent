main.py        receives your input
router.py      classifies it, augments with workflow
agent.py       starts the ReAct loop
  ↓ iteration 1
  gemini API   receives task + tools
  gemini API   responds with function_call: list_files
  tools.py     executes list_files
  memory.py    logs it, adds to history
  ↓ iteration 2
  gemini API   receives history + tool result
  gemini API   responds with function_call: write_file
  tools.py     validates syntax, checks imports, writes to sandbox/
  memory.py    logs it
  ↓ iteration 3
  ...and so on until task_complete
main.py        prints final result
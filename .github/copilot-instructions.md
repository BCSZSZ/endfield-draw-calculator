# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

## 5\. Pythonic Clarity & Intent

**Explicit over implicit. Type hints as documentation.**

- **Intentional Typing:** Use Type Hints (`name: str`) for all function signatures and complex variables. It's for the human (and Copilot), not just the linter.
- **Modern Idioms:** Use `f-strings` for formatting, `pathlib` for paths, and `dataclasses` for data structures. Avoid legacy `os.path` or `%` formatting.
- **Defensive Coding:** No bare `except:`. Catch specific errors. Use guard clauses (`if not x: return`) to keep nesting shallow.
- **Resource Management:** Always use `with` statements for files, locks, or network sessions. Never manually call `.close()`.
- **Clean Iteration:** Prefer list/dict comprehensions for simple transformations. If logic is complex, use a explicit `for` loop for clarity.
- **Dependency Awareness:** Keep `requirements.txt` or `pyproject.toml` updated. Use `venv` to isolate the environment.

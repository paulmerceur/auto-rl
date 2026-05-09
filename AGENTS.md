# AGENTS.md

## Working style

- Work incrementally.
- Prefer small, reviewable commits.
- Do not make large rewrites unless explicitly requested.
- Before changing code, inspect the existing repository structure and summarize the intended change.
- After each meaningful change, run the relevant tests or at least a smoke test.
- Keep the project simple. Avoid unnecessary frameworks.

## Git discipline

- Check `git status` before editing.
- Make small commits with clear messages.
- Do not commit secrets, `.env`, logs, large run artifacts, checkpoints, or model weights.
- Add generated experiment outputs to `.gitignore` unless they are tiny curated examples.
- Prefer branches for experimental changes.

## Python conventions

- Use Python 3.11+ unless the dependency requires otherwise.
- Prefer `uv` or `pip` with a simple `pyproject.toml`.
- Keep modules small and boring.
- Use type hints for public functions.
- Add minimal tests for pure logic.
- Do not over-engineer.

## Project goal

This repo is an experimental tool for LLM-guided reinforcement learning experiment management using PufferLib. The LLM should not directly optimize with arbitrary code execution. It should act as a constrained experiment assistant that proposes sweep/search-space changes based on summarized run results.

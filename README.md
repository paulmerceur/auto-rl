# Puffer LLM Sweeper

An experimental CLI tool for constrained LLM-guided reinforcement learning experiment
management on top of PufferLib.

The goal is to test whether an LLM can act as a useful experiment assistant by reading
summarized run results and proposing bounded sweep/search-space adjustments. The LLM
does not execute arbitrary code, edit source files, or override hard stopping rules.

## Current Status

This repository is being built in small phases:

1. Repository baseline and PufferLib research notes.
2. One manual PufferLib training run from the CLI.
3. Metrics extraction into normalized summaries.
4. OpenRouter client with dry-run support.
5. Strict decision schema and validation.
6. Minimal closed-loop controller.
7. Documentation polish and example outputs.

Phase 0 through Phase 2 are implemented at this point.

## Setup

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Copy `.env.example` to `.env` locally and set `OPENROUTER_API_KEY` before using real
LLM calls. `.env` is ignored by Git.

## Manual PufferLib Run

PufferLib 4.0 is not pinned from PyPI because the current workflow is source/Docker
based. Install and build PufferLib using the current upstream instructions first.

Smoke-test this tool's config handling without launching training:

```bash
PYTHONPATH=src python -m puffer_llm_sweeper run --config configs/base.yaml --dry-run
```

After PufferLib 4.0 is installed and the selected environment is built, launch one
small training run:

```bash
PYTHONPATH=src python -m puffer_llm_sweeper run --config configs/base.yaml
```

Outputs are configured under ignored `runs/` subdirectories.

## Project Boundaries

- CLI first; no web UI.
- Run outputs stay under ignored directories such as `runs/` or `outputs/`.
- LLM output must be structured JSON and validated before use.
- Hard stopping rules always override LLM recommendations.
- No secrets, checkpoints, model weights, or large artifacts should be committed.

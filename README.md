# Puffer LLM Sweeper

An experimental CLI tool for constrained LLM-guided reinforcement learning experiment
management on top of PufferLib.

The goal is to test whether an LLM can act as a useful experiment assistant by reading
summarized run results and proposing bounded sweep/search-space adjustments. The LLM
does not execute arbitrary code, edit source files, or override hard stopping rules.

## What This Is

This is a small public prototype built in phases:

1. Repository baseline and PufferLib research notes.
2. One manual PufferLib training run from the CLI.
3. Metrics extraction into normalized summaries.
4. OpenRouter client with dry-run support.
5. Strict decision schema and validation.
6. Minimal closed-loop controller.
7. Documentation polish and example outputs.

The current implementation covers the full MVP path: bounded PufferLib sweeps,
metrics summary, mock or live LLM decision, validated decision schema, and a
conservative loop controller.

## What This Is Not

- It is not an RL framework.
- It does not let an LLM execute arbitrary shell commands.
- It does not let an LLM edit source code during the experiment loop.
- It does not claim that LLMs magically solve RL.

## Setup

Use Python 3.11 or newer.

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
```

Copy `.env.example` to `.env` locally and set `OPENROUTER_API_KEY` before using real
LLM calls. `.env` is ignored by Git.

## PufferLib Setup

The default config targets PufferLib 4.0 `cartpole` sweep batches using
GPU-oriented Puffer defaults. PufferLib 4.0 currently needs the GitHub source
workflow rather than PyPI.

For GPU training, build PufferLib with CUDA support for the selected environment:

```bash
git clone --branch 4.0 https://github.com/PufferAI/PufferLib.git .deps/PufferLib
cd .deps/PufferLib
bash build.sh cartpole
cd ../..
uv pip install -e .deps/PufferLib
```

If you only need a CPU smoke test, build with `CC=gcc CXX=g++ bash build.sh
cartpole --cpu`. The default config sets `slowly: true`, which uses PufferLib's
PyTorch backend; it will use CUDA when the `_C` backend was built with GPU
support and CPU otherwise.

Smoke-test this tool's config handling without launching training:

```bash
uv run python -m puffer_llm_sweeper sweep --config configs/base.yaml --max-runs 1 --dry-run
```

After PufferLib is installed, launch one bounded sweep batch:

```bash
uv run python -m puffer_llm_sweeper sweep --config configs/base.yaml --max-runs 3
```

Outputs are configured under ignored `runs/` subdirectories.

For debugging one fixed config without sweep sampling, the lower-level command is
still available:

```bash
uv run python -m puffer_llm_sweeper run --config configs/base.yaml
```

## Metrics Summary

Parse completed PufferLib JSON logs into a normalized summary:

```bash
uv run python -m puffer_llm_sweeper summarize \
  --logs-dir runs/logs \
  --output runs/summary.json
```

## LLM Decision

Generate a validated mock decision without spending API credits:

```bash
uv run python -m puffer_llm_sweeper decide \
  --summary runs/summary.json \
  --output runs/decision.json
```

Use `--live` only when you want to spend OpenRouter credits.

## Loop MVP

Run one conservative loop iteration in mock mode:

```bash
uv run python -m puffer_llm_sweeper loop \
  --config configs/base.yaml \
  --run-dir runs/loop-smoke \
  --max-iterations 1 \
  --max-trials 10 \
  --trials-per-iteration 3
```

Use `--skip-training` to test summary and decision plumbing without launching
PufferLib. Use `--live` only when you want the loop to make OpenRouter calls.
Validated search-space updates are applied to the generated work config; the
tracked base config is not modified.

Pass `--run-dir` to isolate all loop artifacts for one experiment. When set, the
loop writes Puffer logs, checkpoints, Puffer data, summaries, decisions, reports,
and the generated work config under that directory.

After a loop, inspect:

```text
runs/loop-smoke/
  journal.jsonl                  # one compact record per loop iteration
  loop_report.json               # final answer: LLM activity, reward trend, problem flags
  performance.svg                # reward graph by iteration
  summary.json                   # cumulative parsed PufferLib run summary
  decision.json                  # most recent validated LLM/mock decision
  loop_config.yaml               # generated config used by the loop
  iterations/iteration-001.json  # per-iteration run summaries
  logs/                          # PufferLib JSON logs
```

Use `loop_report.json` to answer the v1 sanity questions: whether the LLM changed
anything, whether rewards improved over iterations, and whether the loop saw
failures, missing metrics, or training errors.

The loop is sweep-first. Each iteration launches a bounded PufferLib sweep batch.
`--trials-per-iteration` is the starting batch size. The LLM may return
`suggested_trials` from 1 to 10, but that value is only a recommendation. The loop
always clamps it by the remaining total budget, and the project defaults to a hard
total cap of 100 trials.

## Examples

Tiny mock outputs are tracked under `examples/`. Real training outputs belong
under ignored `runs/` or `outputs/`.

```bash
uv run python -m puffer_llm_sweeper decide \
  --summary examples/mock-summary.json \
  --output /tmp/mock-decision.json
```

## Tests

```bash
uv run python -m unittest discover -s tests
uv run ruff check .
```

## Limitations

- PufferLib packaging is in flux: PyPI currently provides 3.0, while current
  docs reference a 4.0 source/Docker workflow.
- The loop is intentionally conservative and small; it does not run large sweeps
  by default.
- The LLM can suggest sweep ranges and the next batch size, but hard budgets
  always win: at most 10 trials per iteration and 100 total by default.
- OpenRouter live calls are opt-in with `--live`; mock mode is the default.

## Project Boundaries

- CLI first; no web UI.
- Run outputs stay under ignored directories such as `runs/` or `outputs/`.
- LLM output must be structured JSON and validated before use.
- Hard stopping rules always override LLM recommendations.
- No secrets, checkpoints, model weights, or large artifacts should be committed.

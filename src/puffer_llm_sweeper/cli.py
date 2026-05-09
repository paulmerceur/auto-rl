"""Command-line entry point for the phased prototype."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="puffer-llm-sweeper",
        description="Constrained LLM-guided PufferLib experiment manager.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the package version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="Launch one manual PufferLib training run.",
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML run config.",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and print the local config without importing or running PufferLib.",
    )

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Parse PufferLib JSON logs into a normalized summary.",
    )
    summarize_parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("runs/logs"),
        help="Directory containing PufferLib JSON logs.",
    )
    summarize_parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/summary.json"),
        help="Path to write normalized summary JSON.",
    )

    decide_parser = subparsers.add_parser(
        "decide",
        help="Ask OpenRouter, or a mock client, for the next validated decision.",
    )
    decide_parser.add_argument(
        "--summary",
        type=Path,
        default=Path("runs/summary.json"),
        help="Path to normalized summary JSON.",
    )
    decide_parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/decision.json"),
        help="Path to write validated decision JSON.",
    )
    decide_parser.add_argument(
        "--live",
        action="store_true",
        help="Use OpenRouter instead of the default mock decision.",
    )

    loop_parser = subparsers.add_parser(
        "loop",
        help="Run the constrained experiment-management loop.",
    )
    loop_parser.add_argument("--config", type=Path, required=True, help="Path to a YAML run config.")
    loop_parser.add_argument("--logs-dir", type=Path, default=Path("runs/logs"))
    loop_parser.add_argument("--summary", type=Path, default=Path("runs/summary.json"))
    loop_parser.add_argument("--decision", type=Path, default=Path("runs/decision.json"))
    loop_parser.add_argument("--max-iterations", type=int, default=1)
    loop_parser.add_argument("--max-trials", type=int, default=1)
    loop_parser.add_argument("--max-minutes", type=float, default=10.0)
    loop_parser.add_argument("--target-reward", type=float, default=None)
    loop_parser.add_argument("--no-improvement-iterations", type=int, default=3)
    loop_parser.add_argument("--improvement-window", type=int, default=3)
    loop_parser.add_argument("--improvement-epsilon", type=float, default=0.0)
    loop_parser.add_argument("--max-failures", type=int, default=2)
    loop_parser.add_argument("--live", action="store_true", help="Use OpenRouter instead of mock mode.")
    loop_parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip PufferLib training; useful for parser/LLM smoke tests.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.version:
        from puffer_llm_sweeper import __version__

        print(__version__)
        return 0

    if args.command == "run":
        from puffer_llm_sweeper.runner import run_training

        try:
            return run_training(args.config, dry_run=args.dry_run)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    if args.command == "summarize":
        from puffer_llm_sweeper.metrics import summarize_logs, write_summary

        try:
            summaries = summarize_logs(args.logs_dir)
            write_summary(summaries, args.output)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Wrote {len(summaries)} run summaries to {args.output}")
        return 0

    if args.command == "decide":
        from puffer_llm_sweeper.decisions import decision_to_json
        from puffer_llm_sweeper.openrouter import OpenRouterClient, load_summary

        try:
            summary = load_summary(args.summary)
            decision = OpenRouterClient().propose_decision(summary, dry_run=not args.live)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(decision_to_json(decision) + "\n", encoding="utf-8")
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Wrote validated decision to {args.output}")
        return 0

    if args.command == "loop":
        from puffer_llm_sweeper.loop import StopRules, run_loop

        try:
            result = run_loop(
                config_path=args.config,
                logs_dir=args.logs_dir,
                summary_path=args.summary,
                decision_path=args.decision,
                rules=StopRules(
                    max_iterations=args.max_iterations,
                    max_trials=args.max_trials,
                    max_minutes=args.max_minutes,
                    target_reward=args.target_reward,
                    no_improvement_iterations=args.no_improvement_iterations,
                    improvement_window=args.improvement_window,
                    improvement_epsilon=args.improvement_epsilon,
                    max_failures=args.max_failures,
                ),
                live=args.live,
                skip_training=args.skip_training,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(
            "Loop stopped: "
            f"reason={result.stop_reason} "
            f"iterations={result.iterations} "
            f"trials={result.trials} "
            f"best_reward={result.best_reward}"
        )
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

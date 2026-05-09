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

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

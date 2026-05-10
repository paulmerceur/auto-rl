"""Command-line entry point for the phased prototype."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
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
    add_build_env_args(run_parser)

    build_env_parser = subparsers.add_parser(
        "build-env",
        help="Build PufferLib's native backend for one Ocean environment.",
    )
    build_env_parser.add_argument("env_name", help="PufferLib Ocean environment name, e.g. cartpole.")
    add_build_env_args(build_env_parser, include_build_flag=False, include_force_flag=False)

    check_env_parser = subparsers.add_parser(
        "check-env",
        help="Print the currently compiled PufferLib native backend.",
    )
    check_env_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config to compare against the compiled backend.",
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

    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Launch a bounded PufferLib sweep batch.",
    )
    sweep_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML run config.",
    )
    sweep_parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Override Puffer sweep max_runs.",
    )
    sweep_parser.add_argument("--dry-run", action="store_true")
    sweep_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress PufferLib dashboard output.",
    )
    add_build_env_args(sweep_parser)

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
    loop_parser.add_argument("--work-config", type=Path, default=Path("runs/loop_config.yaml"))
    loop_parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Directory for all loop artifacts. Overrides logs, summary, decision, and work config paths.",
    )
    loop_parser.add_argument("--max-iterations", type=int, default=1)
    loop_parser.add_argument("--max-trials", type=int, default=100)
    loop_parser.add_argument(
        "--trials-per-iteration",
        type=int,
        default=3,
        help="Initial sweep batch size. LLM suggestions are clamped to at most 10.",
    )
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
    add_build_env_args(loop_parser)
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
            maybe_build_backend_for_config(args)
            return run_training(args.config, dry_run=args.dry_run)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    if args.command == "build-env":
        from puffer_llm_sweeper.puffer_build import ensure_backend

        try:
            info = ensure_backend(
                args.env_name,
                source_dir=args.source_dir,
                force=True,
                cpu=args.cpu,
                arch=args.arch,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(format_backend_info(info), flush=True)
        return 0

    if args.command == "check-env":
        from puffer_llm_sweeper.puffer_build import backend_info
        from puffer_llm_sweeper.runner import load_run_config

        try:
            info = backend_info()
            if info is None:
                print("No compiled PufferLib backend could be imported.")
                return 1
            print(format_backend_info(info), flush=True)
            if args.config:
                config = load_run_config(args.config)
                if config.env_name != info.env_name:
                    print(
                        f"Config env is {config.env_name}; run build-env {config.env_name} "
                        "or pass --build-env."
                    )
                    return 1
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

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

    if args.command == "sweep":
        from puffer_llm_sweeper.runner import run_sweep

        try:
            maybe_build_backend_for_config(args)
            return run_sweep(
                args.config,
                max_runs=args.max_runs,
                dry_run=args.dry_run,
                quiet=args.quiet,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

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
            maybe_build_backend_for_config(args)
            run_dir = args.run_dir or default_loop_run_dir()
            result = run_loop(
                config_path=args.config,
                logs_dir=args.logs_dir,
                summary_path=args.summary,
                decision_path=args.decision,
                work_config_path=args.work_config,
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
                trials_per_iteration=args.trials_per_iteration,
                run_dir=run_dir,
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
        if result.report_path:
            print(f"Report: {result.report_path}")
        if result.journal_path:
            print(f"Journal: {result.journal_path}")
        if result.plot_path:
            print(f"Plot: {result.plot_path}")
        return 0

    parser.print_help()
    return 0


def default_loop_run_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"loop-{timestamp}"


def add_build_env_args(
    parser: argparse.ArgumentParser,
    include_build_flag: bool = True,
    include_force_flag: bool = True,
) -> None:
    if include_build_flag:
        parser.add_argument(
            "--build-env",
            action="store_true",
            help="Build the PufferLib native backend for the config env before running.",
        )
    if include_force_flag:
        parser.add_argument(
            "--force-build-env",
            action="store_true",
            help="Rebuild the PufferLib native backend even if it already matches.",
        )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(".deps/PufferLib"),
        help="Path to a local PufferLib source checkout.",
    )
    parser.add_argument(
        "--arch",
        default=None,
        help="Optional CUDA architecture for NVCC, e.g. sm_86. Auto-detected by default.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Build PufferLib's CPU backend instead of the CUDA backend.",
    )


def maybe_build_backend_for_config(args: argparse.Namespace) -> None:
    if not getattr(args, "build_env", False) and not getattr(args, "force_build_env", False):
        return
    from puffer_llm_sweeper.puffer_build import ensure_backend_for_config

    info = ensure_backend_for_config(
        args.config,
        source_dir=args.source_dir,
        force=args.force_build_env,
        cpu=args.cpu,
        arch=args.arch,
    )
    print(format_backend_info(info), flush=True)


def format_backend_info(info) -> str:
    return (
        f"PufferLib backend: env={info.env_name} "
        f"gpu={info.gpu} precision_bytes={info.precision_bytes}"
    )


if __name__ == "__main__":
    raise SystemExit(main())

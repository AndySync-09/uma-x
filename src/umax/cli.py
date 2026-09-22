from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .benchmark import benchmark
from .runtime import (
    build_command,
    execute,
    hardware_snapshot,
    load_profile,
    resolve_llama_binary,
    update_profile,
)


DEFAULT_PROMPT = "Explain in three short paragraphs why local AI inference is useful."


def _model_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"model not found: {value}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="umax",
        description="UMA-X transformer-aware local AI runtime",
    )
    parser.add_argument("--version", action="version", version=f"UMA-X {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="inspect the local UMA-X execution environment")
    doctor.add_argument("--llama-bin", default=None)

    run = sub.add_parser("run", help="run a GGUF model through UMA-X observe mode")
    run.add_argument("model", type=_model_path)
    run.add_argument("--llama-bin", default=None)
    run.add_argument("--prompt", default=DEFAULT_PROMPT)
    run.add_argument("--tokens", type=int, default=128)
    run.add_argument("--ctx-size", type=int, default=2048)
    run.add_argument("--threads", type=int, default=None)
    run.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="arguments after -- are passed directly to llama.cpp",
    )

    bench = sub.add_parser("bench", help="compare plain llama.cpp with UMA-X observe mode")
    bench.add_argument("model", type=_model_path)
    bench.add_argument("--llama-bin", default=None)
    bench.add_argument("--prompt", default=DEFAULT_PROMPT)
    bench.add_argument("--tokens", type=int, default=128)
    bench.add_argument("--ctx-size", type=int, default=2048)
    bench.add_argument("--threads", type=int, default=None)
    bench.add_argument("--runs", type=int, default=3)
    bench.add_argument("--out", type=Path, default=Path("benchmarks/latest"))
    bench.add_argument("extra_args", nargs=argparse.REMAINDER)

    inspect = sub.add_parser("inspect", help="show UMA-X's learned execution profile")
    inspect.add_argument("model", type=_model_path)

    return parser


def _clean_extra(args: list[str]) -> list[str]:
    return args[1:] if args and args[0] == "--" else args


def cmd_doctor(llama_bin: str | None) -> int:
    data = {
        "umax_version": __version__,
        "hardware": asdict(hardware_snapshot()),
        "llama_binary": None,
        "llama_binary_status": "not found",
        "mode": "observe",
    }
    try:
        data["llama_binary"] = resolve_llama_binary(llama_bin)
        data["llama_binary_status"] = "ready"
    except FileNotFoundError as exc:
        data["llama_error"] = str(exc)

    print(json.dumps(data, indent=2))
    return 0 if data["llama_binary_status"] == "ready" else 2


def cmd_run(args: argparse.Namespace) -> int:
    binary = resolve_llama_binary(args.llama_bin)
    command = build_command(
        binary,
        args.model,
        args.prompt,
        args.tokens,
        args.ctx_size,
        args.threads,
        _clean_extra(args.extra_args),
    )
    metrics, _ = execute(command, stream=True)
    profile = update_profile(args.model, metrics)
    print(
        f"\n[UMA-X] wall={metrics.wall_seconds:.3f}s "
        f"gen={metrics.generation_tokens_per_second or 'n/a'} tok/s "
        f"profile={profile}"
    )
    return metrics.returncode


def cmd_bench(args: argparse.Namespace) -> int:
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    path = benchmark(
        args.model,
        args.llama_bin,
        args.prompt,
        args.tokens,
        args.ctx_size,
        args.threads,
        args.runs,
        _clean_extra(args.extra_args),
        args.out,
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    summary = result["summary"]
    print("\nUMA-X benchmark complete")
    print(f"artifact: {path}")
    print(
        "baseline generation: "
        f"{summary['baseline_generation_tokens_per_second']['mean']} tok/s"
    )
    print(
        "UMA-X observe generation: "
        f"{summary['umax_generation_tokens_per_second']['mean']} tok/s"
    )
    if "generation_delta_percent" in summary:
        print(f"delta: {summary['generation_delta_percent']:+.3f}%")
    return 0


def cmd_inspect(model: Path) -> int:
    print(json.dumps(load_profile(model), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return cmd_doctor(args.llama_bin)
        if args.command == "run":
            return cmd_run(args)
        if args.command == "bench":
            return cmd_bench(args)
        if args.command == "inspect":
            return cmd_inspect(args.model)
    except (FileNotFoundError, ValueError) as exc:
        print(f"UMA-X: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from . import __version__
from .benchmark import BenchmarkError, build_report, run_llama_bench
from .hardware import resolve_binary, snapshot
from .planner import choose_plan
from .profile import ProfileStore, model_fingerprint


def _bytes_gib(value: int) -> str:
    return f"{value / (1024 ** 3):.1f} GiB" if value else "unknown"


def _require_model(path: str) -> Path:
    model = Path(path).expanduser().resolve()
    if not model.is_file():
        raise SystemExit(f"model not found: {model}")
    return model


def _doctor(args: argparse.Namespace) -> int:
    cli = resolve_binary("llama-cli", args.llama_bin_dir)
    bench = resolve_binary("llama-bench", args.llama_bin_dir)
    hw = snapshot(args.llama_bin_dir)

    print(f"UMA-X {__version__}")
    print(f"OS          {hw.system} {hw.release}")
    print(f"CPU         {hw.processor or hw.machine}")
    print(f"Logical CPU {hw.logical_cpus}")
    print(f"RAM         {_bytes_gib(hw.total_ram_bytes)}")
    print(f"llama-cli   {cli or 'not found'}")
    print(f"llama-bench {bench or 'not found'}")

    if hw.llama_devices:
        print("Devices")
        for line in hw.llama_devices:
            print(f"  {line}")
    else:
        print("Devices     none reported by llama-bench")

    return 0 if cli and bench else 2


def _run(args: argparse.Namespace) -> int:
    model = _require_model(args.model)
    llama_cli = resolve_binary("llama-cli", args.llama_bin_dir)
    if not llama_cli:
        raise SystemExit("llama-cli not found; pass --llama-bin-dir or add it to PATH")

    hw = snapshot(args.llama_bin_dir)
    plan = choose_plan(hw, model.stat().st_size)
    command = [llama_cli, "-m", str(model), *plan.llama_args()]

    passthrough = list(args.llama_args or [])
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    command.extend(passthrough)

    print("UMA-X planner path")
    for reason in plan.rationale:
        print(f"  {reason}")

    if args.dry_run:
        print("\n" + subprocess.list2cmdline(command))
        return 0

    return subprocess.call(command)


def _bench(args: argparse.Namespace) -> int:
    model = _require_model(args.model)
    llama_bench = resolve_binary("llama-bench", args.llama_bin_dir)
    if not llama_bench:
        raise SystemExit("llama-bench not found; pass --llama-bin-dir or add it to PATH")

    hw = snapshot(args.llama_bin_dir)
    plan = choose_plan(hw, model.stat().st_size)

    print("Running plain llama.cpp baseline...")
    baseline = run_llama_bench(
        llama_bench,
        model,
        args.prompt_tokens,
        args.gen_tokens,
        args.repetitions,
        extra_args=[],
    )

    print("Running llama.cpp through UMA-X planner...")
    candidate = run_llama_bench(
        llama_bench,
        model,
        args.prompt_tokens,
        args.gen_tokens,
        args.repetitions,
        extra_args=plan.llama_args(),
    )

    report = build_report(
        model=model,
        baseline=baseline,
        candidate=candidate,
        plan=plan.to_dict(),
        hardware=hw.to_dict(),
    )

    fingerprint = model_fingerprint(model)
    output = Path(args.output) if args.output else Path(
        ".umax/benchmarks"
    ) / f"{fingerprint}-comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    ProfileStore().record(
        model,
        hw,
        {
            "type": "benchmark",
            "artifact": str(output.resolve()),
            "summary": report["summary"],
            "planner": plan.to_dict(),
        },
    )

    summary = report["summary"]
    b = summary["baseline"]
    u = summary["umax"]
    d = summary["delta_percent"]

    print("\nResult")
    print(f"  Prompt       llama.cpp={b['prompt_tps']!s} t/s   UMA-X={u['prompt_tps']!s} t/s   delta={d['prompt_tps']!s}%")
    print(f"  Generation   llama.cpp={b['generation_tps']!s} t/s   UMA-X={u['generation_tps']!s} t/s   delta={d['generation_tps']!s}%")
    print(f"  Artifact     {output.resolve()}")
    print("\nNote: v0.1 is a planner/control-path comparison, not yet a transformer-aware paging benchmark.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="umax",
        description="UMA-X local AI research runtime",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="inspect this machine and llama.cpp")
    doctor.add_argument("--llama-bin-dir")
    doctor.set_defaults(func=_doctor)

    run = sub.add_parser("run", help="run a GGUF model through the UMA-X planner")
    run.add_argument("model")
    run.add_argument("--llama-bin-dir")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("llama_args", nargs=argparse.REMAINDER)
    run.set_defaults(func=_run)

    bench = sub.add_parser(
        "bench",
        help="compare plain llama.cpp with the UMA-X planner path",
    )
    bench.add_argument("model")
    bench.add_argument("--llama-bin-dir")
    bench.add_argument("--prompt-tokens", type=int, default=512)
    bench.add_argument("--gen-tokens", type=int, default=128)
    bench.add_argument("--repetitions", type=int, default=5)
    bench.add_argument("--output")
    bench.set_defaults(func=_bench)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except BenchmarkError as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

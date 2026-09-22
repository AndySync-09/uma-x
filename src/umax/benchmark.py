from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from .runtime import build_command, execute, resolve_llama_binary, update_profile


def _summary(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "median": None, "stdev": None}
    return {
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
    }


def benchmark(
    model: Path,
    llama_bin: str | None,
    prompt: str,
    tokens: int,
    ctx_size: int,
    threads: int | None,
    runs: int,
    extra_args: list[str],
    artifact_dir: Path,
) -> Path:
    binary = resolve_llama_binary(llama_bin)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "schema": 1,
        "created_unix": time.time(),
        "model": str(model.resolve()),
        "model_bytes": model.stat().st_size,
        "llama_binary": binary,
        "configuration": {
            "prompt": prompt,
            "tokens": tokens,
            "ctx_size": ctx_size,
            "threads": threads,
            "runs": runs,
            "seed": 42,
            "extra_args": extra_args,
        },
        "baseline": [],
        "umax": [],
        "note": (
            "UMA-X v0.1 observe mode invokes the same llama.cpp binary. "
            "This benchmark establishes reproducibility and wrapper overhead; "
            "it does not yet claim a tensor-placement optimization."
        ),
    }

    command = build_command(
        binary, model, prompt, tokens, ctx_size, threads, extra_args
    )

    for index in range(runs):
        print(f"\n[baseline {index + 1}/{runs}]")
        metrics, output = execute(command, stream=False)
        result["baseline"].append(asdict(metrics))
        (artifact_dir / f"baseline-{index + 1}.log").write_text(output, encoding="utf-8")

    for index in range(runs):
        print(f"\n[uma-x observe {index + 1}/{runs}]")
        metrics, output = execute(command, stream=False)
        update_profile(model, metrics)
        result["umax"].append(asdict(metrics))
        (artifact_dir / f"umax-{index + 1}.log").write_text(output, encoding="utf-8")

    baseline_tps = [
        x["generation_tokens_per_second"]
        for x in result["baseline"]
        if x["generation_tokens_per_second"] is not None
    ]
    umax_tps = [
        x["generation_tokens_per_second"]
        for x in result["umax"]
        if x["generation_tokens_per_second"] is not None
    ]
    baseline_wall = [x["wall_seconds"] for x in result["baseline"]]
    umax_wall = [x["wall_seconds"] for x in result["umax"]]

    result["summary"] = {
        "baseline_generation_tokens_per_second": _summary(baseline_tps),
        "umax_generation_tokens_per_second": _summary(umax_tps),
        "baseline_wall_seconds": _summary(baseline_wall),
        "umax_wall_seconds": _summary(umax_wall),
    }

    if baseline_tps and umax_tps:
        baseline_mean = statistics.mean(baseline_tps)
        umax_mean = statistics.mean(umax_tps)
        result["summary"]["generation_delta_percent"] = round(
            ((umax_mean / baseline_mean) - 1.0) * 100.0, 3
        )

    output_path = artifact_dir / "results.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output_path

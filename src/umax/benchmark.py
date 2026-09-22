from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any


class BenchmarkError(RuntimeError):
    pass


def _extract_json(stdout: str) -> list[dict[str, Any]]:
    text = stdout.strip()
    if not text:
        raise BenchmarkError("llama-bench produced no JSON output")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end <= start:
            raise BenchmarkError("could not locate llama-bench JSON output")
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, list):
        raise BenchmarkError("unexpected llama-bench JSON shape")
    return parsed


def run_llama_bench(
    llama_bench: str,
    model: str | Path,
    prompt_tokens: int,
    gen_tokens: int,
    repetitions: int,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    command = [
        llama_bench,
        "-m",
        str(model),
        "-p",
        str(prompt_tokens),
        "-n",
        str(gen_tokens),
        "-r",
        str(repetitions),
        "-o",
        "json",
    ]
    if extra_args:
        command.extend(extra_args)

    # Do not capture llama-bench through Python pipes. Some Windows SYCL
    # runtimes crash when stdout/stderr are attached to subprocess.PIPE.
    # Plain shell redirection is stable, so mirror that behavior with
    # temporary files and parse the files after the process exits.
    with tempfile.TemporaryDirectory(prefix="umax-bench-") as temp_dir:
        stdout_path = Path(temp_dir) / "stdout.json"
        stderr_path = Path(temp_dir) / "stderr.txt"

        with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file, \
             stderr_path.open("w", encoding="utf-8", errors="replace") as stderr_file:
            proc = subprocess.run(
                command,
                check=False,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
            )

        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")

    if proc.returncode != 0:
        raise BenchmarkError(
            f"llama-bench failed with exit code {proc.returncode}\n{stderr.strip()}"
        )

    return {
        "command": command,
        "rows": _extract_json(stdout),
        "stderr": stderr.strip(),
    }


def metric_row(rows: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    for row in rows:
        n_prompt = int(row.get("n_prompt", 0) or 0)
        n_gen = int(row.get("n_gen", 0) or 0)
        if kind == "prompt" and n_prompt > 0 and n_gen == 0:
            return row
        if kind == "generation" and n_prompt == 0 and n_gen > 0:
            return row
    return None


def metric_summary(run: dict[str, Any]) -> dict[str, float | None]:
    prompt = metric_row(run["rows"], "prompt")
    generation = metric_row(run["rows"], "generation")
    return {
        "prompt_tps": float(prompt["avg_ts"]) if prompt and "avg_ts" in prompt else None,
        "generation_tps": (
            float(generation["avg_ts"])
            if generation and "avg_ts" in generation
            else None
        ),
    }


def percent_delta(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or baseline == 0:
        return None
    return ((candidate - baseline) / baseline) * 100.0


def build_report(
    model: str | Path,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    plan: dict[str, Any],
    hardware: dict[str, Any],
) -> dict[str, Any]:
    b = metric_summary(baseline)
    c = metric_summary(candidate)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(Path(model).resolve()),
        "hardware": hardware,
        "planner": plan,
        "baseline": baseline,
        "umax": candidate,
        "summary": {
            "baseline": b,
            "umax": c,
            "delta_percent": {
                "prompt_tps": percent_delta(b["prompt_tps"], c["prompt_tps"]),
                "generation_tps": percent_delta(
                    b["generation_tps"], c["generation_tps"]
                ),
            },
        },
    }

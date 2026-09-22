from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


class BenchmarkError(RuntimeError):
    pass


_WINDOWS_ACCESS_VIOLATION_CODES = {3221225477, -1073741819}


def _extract_jsonl(stdout: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for line_number, raw_line in enumerate(stdout.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchmarkError(
                f"invalid llama-bench JSONL on line {line_number}: {exc.msg}"
            ) from exc

        if not isinstance(parsed, dict):
            raise BenchmarkError(
                f"unexpected llama-bench JSONL row shape on line {line_number}"
            )
        rows.append(parsed)

    if not rows:
        raise BenchmarkError("llama-bench produced no JSONL output")

    return rows


def _has_complete_benchmark_rows(
    rows: list[dict[str, Any]],
    prompt_tokens: int,
    gen_tokens: int,
) -> bool:
    has_prompt = prompt_tokens <= 0
    has_generation = gen_tokens <= 0

    for row in rows:
        n_prompt = int(row.get("n_prompt", 0) or 0)
        n_gen = int(row.get("n_gen", 0) or 0)
        if n_prompt > 0 and n_gen == 0 and "avg_ts" in row:
            has_prompt = True
        if n_prompt == 0 and n_gen > 0 and "avg_ts" in row:
            has_generation = True

    return has_prompt and has_generation


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
        "jsonl",
    ]
    if extra_args:
        command.extend(extra_args)

    # Keep stdout/stderr off Python pipes. On some Intel Arc/SYCL Windows
    # setups llama-bench can exit with 0xC0000005 during backend teardown.
    # JSONL is intentionally used because each completed benchmark row is
    # flushed independently, so valid measurements survive a late teardown
    # crash even when a buffered JSON array would be truncated.
    with tempfile.TemporaryDirectory(prefix="umax-bench-") as temp_dir:
        stdout_path = Path(temp_dir) / "stdout.jsonl"
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

    rows: list[dict[str, Any]] | None = None
    parse_error: BenchmarkError | None = None
    try:
        rows = _extract_jsonl(stdout)
    except BenchmarkError as exc:
        parse_error = exc

    tolerated_teardown = (
        sys.platform == "win32"
        and proc.returncode in _WINDOWS_ACCESS_VIOLATION_CODES
        and rows is not None
        and _has_complete_benchmark_rows(rows, prompt_tokens, gen_tokens)
    )

    if proc.returncode != 0 and not tolerated_teardown:
        detail = stderr.strip()
        if parse_error is not None:
            detail = f"{detail}\nJSONL: {parse_error}".strip()
        raise BenchmarkError(
            f"llama-bench failed with exit code {proc.returncode}\n{detail}"
        )

    if rows is None:
        assert parse_error is not None
        raise parse_error

    teardown_warning = None
    if tolerated_teardown:
        teardown_warning = (
            "llama-bench completed and emitted complete benchmark JSONL, then the "
            "Windows SYCL runtime exited with access violation 0xC0000005 during teardown"
        )

    return {
        "command": command,
        "rows": rows,
        "stderr": stderr.strip(),
        "process_exit_code": proc.returncode,
        "teardown_warning": teardown_warning,
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

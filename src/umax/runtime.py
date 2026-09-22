from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

TOKENS_RE = re.compile(r"eval time\s*=.*?([0-9]+(?:\.[0-9]+)?) tokens per second", re.I)
PROMPT_RE = re.compile(r"prompt eval time\s*=.*?([0-9]+(?:\.[0-9]+)?) tokens per second", re.I)


@dataclass
class RunMetrics:
    wall_seconds: float
    generation_tokens_per_second: float | None
    prompt_tokens_per_second: float | None
    returncode: int


@dataclass
class HardwareSnapshot:
    os: str
    machine: str
    processor: str
    logical_cpu_count: int | None


def hardware_snapshot() -> HardwareSnapshot:
    return HardwareSnapshot(
        os=f"{platform.system()} {platform.release()}",
        machine=platform.machine(),
        processor=platform.processor(),
        logical_cpu_count=os.cpu_count(),
    )


def resolve_llama_binary(explicit: str | None = None) -> str:
    if explicit:
        candidate = Path(explicit)
        if candidate.exists():
            return str(candidate)
        found = shutil.which(explicit)
        if found:
            return found
        raise FileNotFoundError(f"llama.cpp binary not found: {explicit}")

    for name in ("llama-cli.exe", "llama-cli"):
        found = shutil.which(name)
        if found:
            return found

    raise FileNotFoundError(
        "Could not find llama-cli. Add llama.cpp to PATH or pass --llama-bin."
    )


def model_id(model: Path) -> str:
    stat = model.stat()
    seed = f"{model.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    return hashlib.sha256(seed).hexdigest()[:16]


def profile_path(model: Path) -> Path:
    root = Path(os.environ.get("UMAX_HOME", Path.home() / ".umax"))
    path = root / "profiles"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{model_id(model)}.json"


def load_profile(model: Path) -> dict:
    path = profile_path(model)
    if not path.exists():
        return {
            "schema": 1,
            "model": str(model.resolve()),
            "model_bytes": model.stat().st_size,
            "hardware": asdict(hardware_snapshot()),
            "runs": 0,
            "ema_generation_tokens_per_second": None,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def update_profile(model: Path, metrics: RunMetrics) -> Path:
    profile = load_profile(model)
    previous = profile.get("ema_generation_tokens_per_second")
    current = metrics.generation_tokens_per_second

    if current is not None:
        profile["ema_generation_tokens_per_second"] = (
            current if previous is None else round((0.25 * current) + (0.75 * previous), 4)
        )

    profile["runs"] = int(profile.get("runs", 0)) + 1
    profile["last_run"] = asdict(metrics)
    profile["last_updated_unix"] = time.time()

    path = profile_path(model)
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return path


def parse_metrics(output: str, wall_seconds: float, returncode: int) -> RunMetrics:
    generation = TOKENS_RE.search(output)
    prompt = PROMPT_RE.search(output)
    return RunMetrics(
        wall_seconds=round(wall_seconds, 4),
        generation_tokens_per_second=float(generation.group(1)) if generation else None,
        prompt_tokens_per_second=float(prompt.group(1)) if prompt else None,
        returncode=returncode,
    )


def build_command(
    llama_bin: str,
    model: Path,
    prompt: str,
    tokens: int,
    ctx_size: int,
    threads: int | None,
    extra_args: Iterable[str] = (),
) -> list[str]:
    cmd = [
        llama_bin,
        "-m",
        str(model),
        "-p",
        prompt,
        "-n",
        str(tokens),
        "-c",
        str(ctx_size),
        "--seed",
        "42",
        "--no-display-prompt",
    ]
    if threads:
        cmd.extend(["-t", str(threads)])
    cmd.extend(extra_args)
    return cmd


def execute(command: list[str], stream: bool = True) -> tuple[RunMetrics, str]:
    started = time.perf_counter()
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        captured.append(line)
        if stream:
            print(line, end="")

    returncode = proc.wait()
    wall = time.perf_counter() - started
    output = "".join(captured)
    return parse_metrics(output, wall, returncode), output

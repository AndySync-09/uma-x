from __future__ import annotations

from dataclasses import asdict, dataclass
import ctypes
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Iterable


@dataclass(frozen=True)
class HardwareSnapshot:
    system: str
    release: str
    machine: str
    processor: str
    logical_cpus: int
    total_ram_bytes: int
    llama_devices: tuple[str, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["llama_devices"] = list(self.llama_devices)
        return data


def _windows_total_memory() -> int:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return 0
    return int(status.ullTotalPhys)


def total_memory_bytes() -> int:
    if os.name == "nt":
        try:
            return _windows_total_memory()
        except Exception:
            return 0

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (AttributeError, OSError, ValueError):
        return 0


def resolve_binary(name: str, bin_dir: str | None = None) -> str | None:
    names = [name]
    if os.name == "nt" and not name.lower().endswith(".exe"):
        names.insert(0, f"{name}.exe")

    if bin_dir:
        root = Path(bin_dir).expanduser().resolve()
        for candidate in names:
            path = root / candidate
            if path.is_file():
                return str(path)

    for candidate in names:
        found = shutil.which(candidate)
        if found:
            return found

    return None


def _run_probe(command: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return 1, ""

    combined = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
    return proc.returncode, combined.strip()


def query_llama_devices(llama_bench: str | None, bin_dir: str | None = None) -> tuple[str, ...]:
    if not llama_bench:
        return ()

    code, output = _run_probe([llama_bench, "--list-devices"])
    if code == 0 and output:
        return tuple(line.strip() for line in output.splitlines() if line.strip())

    # Older llama.cpp SYCL builds (including b5377) do not implement
    # --list-devices on llama-bench, but ship a dedicated device probe.
    sycl_probe = resolve_binary("llama-ls-sycl-device", bin_dir)
    if sycl_probe:
        code, output = _run_probe([sycl_probe])
        if code == 0 and output:
            return tuple(line.strip() for line in output.splitlines() if line.strip())

    return ()


def snapshot(bin_dir: str | None = None) -> HardwareSnapshot:
    bench = resolve_binary("llama-bench", bin_dir)
    return HardwareSnapshot(
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        processor=platform.processor(),
        logical_cpus=os.cpu_count() or 1,
        total_ram_bytes=total_memory_bytes(),
        llama_devices=query_llama_devices(bench, bin_dir),
    )


def has_accelerator(device_lines: Iterable[str]) -> bool:
    text = " ".join(device_lines).lower()
    signals = ("gpu", "igpu", "sycl", "vulkan", "cuda", "metal", "musa", "kompute")
    return any(signal in text for signal in signals)

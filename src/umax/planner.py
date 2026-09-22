from __future__ import annotations

from dataclasses import asdict, dataclass

from .hardware import HardwareSnapshot, has_accelerator


@dataclass(frozen=True)
class ExecutionPlan:
    threads: int
    gpu_layers: int
    mmap: bool
    rationale: tuple[str, ...]

    def llama_args(self) -> list[str]:
        args = [
            "-t",
            str(self.threads),
            "-ngl",
            str(self.gpu_layers),
        ]
        if not self.mmap:
            args.extend(["-mmp", "0"])
        return args

    def to_dict(self) -> dict:
        data = asdict(self)
        data["rationale"] = list(self.rationale)
        return data


def choose_plan(hw: HardwareSnapshot, model_size_bytes: int) -> ExecutionPlan:
    logical = max(1, hw.logical_cpus)
    # Conservative bootstrap heuristic. Keep v0.1 close to llama.cpp defaults
    # so the first A/B validates the control path rather than claiming tuning.
    threads = max(1, min(16, logical))

    accelerator = has_accelerator(hw.llama_devices)
    gpu_layers = 99 if accelerator else 0

    rationale = [
        f"threads={threads} from {logical} logical CPUs; v0.1 caps at llama-bench's 16-thread control default",
        "mmap enabled so the OS can provide the initial file-backed model path",
    ]
    if accelerator:
        rationale.append("llama.cpp reports an accelerator; request full supported offload")
    else:
        rationale.append("no llama.cpp accelerator detected; keep model compute on CPU")

    if hw.total_ram_bytes and model_size_bytes > hw.total_ram_bytes:
        rationale.append(
            "model file exceeds physical RAM; v0.1 records this pressure but does not yet implement UMA-X tensor paging"
        )

    return ExecutionPlan(
        threads=threads,
        gpu_layers=gpu_layers,
        mmap=True,
        rationale=tuple(rationale),
    )

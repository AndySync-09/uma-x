from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .hardware import HardwareSnapshot


_SAMPLE_BYTES = 4 * 1024 * 1024


def model_fingerprint(path: str | Path) -> str:
    file_path = Path(path)
    size = file_path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))

    with file_path.open("rb") as handle:
        digest.update(handle.read(_SAMPLE_BYTES))
        if size > _SAMPLE_BYTES:
            handle.seek(max(0, size - _SAMPLE_BYTES))
            digest.update(handle.read(_SAMPLE_BYTES))

    return digest.hexdigest()[:24]


def hardware_fingerprint(hw: HardwareSnapshot) -> str:
    payload = json.dumps(hw.to_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


@dataclass
class ProfileRecord:
    schema_version: int
    model_fingerprint: str
    hardware_fingerprint: str
    model_path: str
    created_at: str
    updated_at: str
    observations: list[dict[str, Any]]


class ProfileStore:
    def __init__(self, root: str | Path = ".umax/profiles") -> None:
        self.root = Path(root)

    def _path(self, model_id: str, hardware_id: str) -> Path:
        return self.root / f"{model_id}-{hardware_id}.json"

    def record(
        self,
        model_path: str | Path,
        hw: HardwareSnapshot,
        observation: dict[str, Any],
    ) -> Path:
        model_id = model_fingerprint(model_path)
        hardware_id = hardware_fingerprint(hw)
        target = self._path(model_id, hardware_id)
        now = datetime.now(timezone.utc).isoformat()

        if target.exists():
            data = json.loads(target.read_text(encoding="utf-8"))
            record = ProfileRecord(**data)
        else:
            record = ProfileRecord(
                schema_version=1,
                model_fingerprint=model_id,
                hardware_fingerprint=hardware_id,
                model_path=str(Path(model_path).resolve()),
                created_at=now,
                updated_at=now,
                observations=[],
            )

        record.updated_at = now
        record.observations.append(observation)
        self.root.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
        return target

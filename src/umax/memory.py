from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import inf


@dataclass(frozen=True)
class Access:
    tensor_id: str
    size_bytes: int
    seq: int


def simulate_residency(
    accesses: list[Access],
    capacity_bytes: int,
) -> dict:
    if capacity_bytes <= 0:
        raise ValueError("capacity_bytes must be positive")

    future: dict[str, deque[int]] = defaultdict(deque)
    for access in accesses:
        future[access.tensor_id].append(access.seq)

    resident: dict[str, int] = {}
    resident_bytes = 0
    peak_resident_bytes = 0
    hits = 0
    misses = 0
    evictions: list[dict[str, int | str]] = []

    for access in accesses:
        positions = future[access.tensor_id]
        if positions and positions[0] == access.seq:
            positions.popleft()

        if access.tensor_id in resident:
            hits += 1
            continue

        misses += 1

        if access.size_bytes > capacity_bytes:
            raise ValueError(
                f"tensor {access.tensor_id} exceeds residency capacity"
            )

        while resident_bytes + access.size_bytes > capacity_bytes:
            victim = max(
                resident,
                key=lambda tensor_id: (
                    future[tensor_id][0]
                    if future[tensor_id]
                    else inf
                ),
            )

            resident_bytes -= resident.pop(victim)
            evictions.append(
                {
                    "seq": access.seq,
                    "tensor_id": victim,
                }
            )

        resident[access.tensor_id] = access.size_bytes
        resident_bytes += access.size_bytes
        peak_resident_bytes = max(
            peak_resident_bytes,
            resident_bytes,
        )

    return {
        "capacity_bytes": capacity_bytes,
        "peak_resident_bytes": peak_resident_bytes,
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "resident_bytes_final": resident_bytes,
    }

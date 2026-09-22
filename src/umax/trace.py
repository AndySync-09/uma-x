from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    trace_path = Path(path)
    rows: list[dict[str, Any]] = []

    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    return rows


def _graph_chunks(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not rows:
        return []

    starts = [
        index
        for index, row in enumerate(rows)
        if row.get("split") == 0 and row.get("node") == 0
    ]

    if not starts or starts[0] != 0:
        return [rows]

    chunks: list[list[dict[str, Any]]] = []

    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(rows)
        chunks.append(rows[start:end])

    return chunks


def _next_use_distances(
    graphs: Iterable[list[dict[str, Any]]],
) -> list[int]:
    distances: list[int] = []

    for graph in graphs:
        uses: dict[str, list[int]] = defaultdict(list)

        for local_seq, row in enumerate(graph):
            for source in row.get("sources", []):
                source_id = source.get("id")
                if source_id:
                    uses[source_id].append(local_seq)

        for sequences in uses.values():
            distances.extend(
                current - previous
                for previous, current in zip(sequences, sequences[1:])
            )

    return distances


def analyze_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    backends = Counter(
        row.get("backend")
        for row in rows
        if row.get("backend") is not None
    )

    weights: dict[str, dict[str, Any]] = {}

    for row in rows:
        for source in row.get("sources", []):
            name = source.get("name", "")
            source_id = source.get("id")

            if source_id and name.endswith(".weight"):
                weights[source_id] = source

    slots: dict[tuple[str, int], set[str]] = defaultdict(set)

    for row in rows:
        buffer_name = row.get("buffer")
        buffer_offset = row.get("buffer_offset")
        tensor_id = row.get("tensor_id")

        if (
            buffer_name is not None
            and buffer_offset is not None
            and tensor_id is not None
        ):
            slots[(buffer_name, buffer_offset)].add(tensor_id)

    reused_slots = sum(1 for tensor_ids in slots.values() if len(tensor_ids) > 1)

    distances = _next_use_distances(_graph_chunks(rows))

    next_use = {
        "observations": len(distances),
        "min": min(distances) if distances else None,
        "max": max(distances) if distances else None,
        "avg": (
            round(sum(distances) / len(distances), 2)
            if distances
            else None
        ),
    }

    return {
        "records": len(rows),
        "backends": dict(backends),
        "unique_weights": len(weights),
        "weight_buffers": dict(
            Counter(
                weight.get("buffer")
                for weight in weights.values()
                if weight.get("buffer") is not None
            )
        ),
        "physical_slots": len(slots),
        "reused_slots": reused_slots,
        "next_use": next_use,
    }

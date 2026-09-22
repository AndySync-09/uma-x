from __future__ import annotations

from .memory import Access


def plan_prefetch(
    accesses: list[Access],
    lead_nodes: int,
) -> list[dict[str, int | str]]:
    if lead_nodes < 0:
        raise ValueError("lead_nodes must be non-negative")

    plan: list[dict[str, int | str]] = []

    for access in accesses:
        prefetch_seq = max(0, access.seq - lead_nodes)
        actual_lead = access.seq - prefetch_seq

        plan.append(
            {
                "tensor_id": access.tensor_id,
                "size_bytes": access.size_bytes,
                "prefetch_seq": prefetch_seq,
                "use_seq": access.seq,
                "lead_nodes": actual_lead,
            }
        )

    return plan

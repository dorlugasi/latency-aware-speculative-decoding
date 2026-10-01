"""Runtime (GPU/device) snapshots and CSV export for per-step decode records.

`snapshot_runtime` is called once per round from `SpeculativeDecoder.generate`
(see speculative.py) to attach a point-in-time GPU/device reading to each
`StepResult` -- this can only be captured live, not reconstructed after the
fact, which is why it lives inside the decode loop rather than being computed
from the finished trace.

Every device-specific call is wrapped defensively: on the actual target
hardware (a CUDA cloud/cluster box) `torch.cuda.utilization` additionally
requires `nvidia-ml-py` (aka pynvml) to be installed, and we'd rather log a
missing metric as `None` than crash a long-running experiment over it.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Iterable, List, Optional

import torch

if TYPE_CHECKING:
    from .speculative import StepResult

_BYTES_PER_MB = 1024 * 1024


@dataclass
class RuntimeSnapshot:
    device_type: str
    gpu_utilization_pct: Optional[float] = None
    gpu_mem_allocated_mb: Optional[float] = None
    gpu_mem_reserved_mb: Optional[float] = None


def snapshot_runtime(device: str) -> RuntimeSnapshot:
    """Best-effort point-in-time device snapshot; unavailable fields are None."""
    device_type = torch.device(device).type

    if device_type == "cuda":
        return _snapshot_cuda(device)
    if device_type == "mps":
        return _snapshot_mps()
    return RuntimeSnapshot(device_type=device_type)


def _snapshot_cuda(device: str) -> RuntimeSnapshot:
    try:
        allocated = torch.cuda.memory_allocated(device) / _BYTES_PER_MB
    except Exception:
        allocated = None
    try:
        reserved = torch.cuda.memory_reserved(device) / _BYTES_PER_MB
    except Exception:
        reserved = None
    try:
        # Requires nvidia-ml-py (pynvml); queries the driver directly, so it
        # doesn't force a CUDA stream sync and won't skew latency timings.
        utilization = float(torch.cuda.utilization(device))
    except Exception:
        utilization = None
    return RuntimeSnapshot(
        device_type="cuda",
        gpu_utilization_pct=utilization,
        gpu_mem_allocated_mb=allocated,
        gpu_mem_reserved_mb=reserved,
    )


def _snapshot_mps() -> RuntimeSnapshot:
    try:
        allocated = torch.mps.current_allocated_memory() / _BYTES_PER_MB
    except Exception:
        allocated = None
    try:
        reserved = torch.mps.driver_allocated_memory() / _BYTES_PER_MB
    except Exception:
        reserved = None
    # Apple doesn't expose a per-process GPU utilization % through torch.
    return RuntimeSnapshot(
        device_type="mps",
        gpu_utilization_pct=None,
        gpu_mem_allocated_mb=allocated,
        gpu_mem_reserved_mb=reserved,
    )


def step_to_dict(step: "StepResult") -> dict:
    """Flatten a StepResult (incl. derived properties) into a CSV-friendly row."""
    row = asdict(step)
    row["new_token_ids"] = " ".join(str(t) for t in step.new_token_ids)
    row["num_rejected"] = step.num_rejected
    row["tokens_per_second"] = step.tokens_per_second
    row["all_accepted"] = step.all_accepted
    row["acceptance_rate"] = step.acceptance_rate
    row["draft_ms_per_token"] = step.draft_ms_per_token
    row["verify_ms_per_token"] = step.verify_ms_per_token
    row["draft_verify_ratio"] = step.draft_verify_ratio
    return row


def steps_to_dicts(steps: Iterable["StepResult"]) -> List[dict]:
    return [step_to_dict(s) for s in steps]


def save_steps_csv(steps: Iterable["StepResult"], path: str) -> None:
    rows = steps_to_dicts(steps)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

"""Runtime harness for concurrent requests and artificial GPU load.

`num_streams` means concurrent single-request decoding streams. It is not
tensor-level batching. Keep real batch-size experiments separate in the report.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Sequence

import torch

from .speculative import GenerationResult, KSelector, SpeculativeDecoder


class GpuContention:
    """Background compute load on `device`, to simulate other processes
    sharing the same GPU. Start/stop explicitly, or use as a context manager.
    """

    def __init__(self, device: str, intensity: int = 1, matrix_size: int = 2048):
        if intensity < 0:
            raise ValueError("intensity must be >= 0")
        self.device = device
        self.intensity = intensity
        self.matrix_size = matrix_size
        self._device_type = torch.device(device).type
        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []

    def _worker(self) -> None:
        a = torch.randn(self.matrix_size, self.matrix_size, device=self.device)
        b = torch.randn(self.matrix_size, self.matrix_size, device=self.device)
        while not self._stop_event.is_set():
            a @ b
            if self._device_type == "cuda":
                torch.cuda.synchronize()
            elif self._device_type == "mps":
                torch.mps.synchronize()
            # CPU matmuls are already synchronous when the call returns.

    def start(self) -> None:
        if self._threads:
            return  # already running
        self._stop_event.clear()
        self._threads = [threading.Thread(target=self._worker, daemon=True) for _ in range(self.intensity)]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=5)
        self._threads = []

    def __enter__(self) -> "GpuContention":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()


def run_concurrent_streams(
    decoder: SpeculativeDecoder,
    prompts: Sequence[str],
    max_new_tokens: int,
    k: KSelector,
    **generate_kwargs,
) -> List[GenerationResult]:
    """Run one generate call per prompt concurrently. This measures concurrency, not batching."""
    if not prompts:
        return []

    if torch.device(decoder.device).type == "mps" and len(prompts) > 1:
        # PyTorch's MPS backend does not tolerate concurrent command-buffer
        # encoding from multiple threads onto the same device: this reliably
        # native-crashes the process (a Metal assertion failure, not a
        # catchable Python exception) rather than raising cleanly -- verified
        # directly, including with contention_intensity=0, i.e. this is not
        # about GpuContention specifically, plain concurrent generate() calls
        # trigger it. CUDA's multi-stream model is expected not to have this
        # problem but hasn't been verified on real CUDA hardware yet. CPU
        # works today and is what the test suite uses.
        raise RuntimeError(
            "Concurrent decoding streams (>1) are not supported on MPS -- PyTorch's MPS backend "
            "crashes on concurrent multi-threaded GPU submission. Use device='cpu' for local "
            "concurrency testing on a Mac, or device='cuda' on the real target hardware."
        )

    def _run(prompt: str) -> GenerationResult:
        return decoder.generate(prompt, max_new_tokens=max_new_tokens, k=k, **generate_kwargs)

    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        return list(pool.map(_run, prompts))


@dataclass
class RuntimeRegime:
    """A named runtime condition: how many concurrent streams, how much
    background GPU contention. `num_streams` is informational/for labeling
    -- the actual concurrency level used by `run_under_regime` is
    `len(prompts)`, so pass that many prompts.
    """

    name: str
    num_streams: int = 1
    contention_intensity: int = 0


PRESET_REGIMES: List[RuntimeRegime] = [
    RuntimeRegime(name="low_load", num_streams=1, contention_intensity=0),
    RuntimeRegime(name="high_concurrency", num_streams=4, contention_intensity=0),
    RuntimeRegime(name="gpu_contention", num_streams=1, contention_intensity=2),
    RuntimeRegime(name="concurrency_and_contention", num_streams=4, contention_intensity=2),
]


def run_under_regime(
    decoder: SpeculativeDecoder,
    prompts: Sequence[str],
    max_new_tokens: int,
    k: KSelector,
    regime: RuntimeRegime,
    contention_matrix_size: int = 2048,
    **generate_kwargs,
) -> List[GenerationResult]:
    """`run_concurrent_streams`, with `regime`'s background GPU contention
    (if any) running for the duration of the call.
    """
    if regime.contention_intensity <= 0:
        return run_concurrent_streams(decoder, prompts, max_new_tokens, k, **generate_kwargs)

    contention = GpuContention(
        device=decoder.device, intensity=regime.contention_intensity, matrix_size=contention_matrix_size
    )
    with contention:
        return run_concurrent_streams(decoder, prompts, max_new_tokens, k, **generate_kwargs)

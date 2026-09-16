# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
# ==============================================================================
"""Pinned-staging rotation for flashinfer-style sync-free plan calls.

flashinfer's native plan kernels (``MLAPlan`` / ``DecodePlan``) compute the
tile-scheduling metadata on the host, write it into a page-locked staging
buffer with plain CPU stores, and only then enqueue a ``cudaMemcpyAsync``
(H2D, current stream) into the device int-workspace that a captured kernel
reads at replay. The host stores complete immediately, so under overlapped
scheduling (PP microbatch slots, or ordinary consecutive-step overlap) the
NEXT plan call can overwrite the staging buffer while the PREVIOUS plan's
async copy is still queued behind an in-flight graph -- the previous graph
then runs with the next batch's schedule, which under DCP surfaces as CUDA
illegal-memory-access (and otherwise as silent wrong-page attention).

The stock ``plan()`` was accidentally immune: its blocking ``.to("cpu")``
reads drain the stream before the host stores happen. The sync-free fast
plans removed that drain, so the caller must guarantee the staging buffer is
not reused while a prior copy from it is pending. This module provides that
guarantee by rotating K pinned buffers and host-syncing each buffer's
recorded completion event before handing it out again. In steady state the
event is long complete, so the check costs ~1us and never blocks; K only
bounds how far the CPU may run ahead without syncing.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

import torch

logger = logging.getLogger(__name__)

# Matches the int-workspace size flashinfer allocates per wrapper
# (8 * 1024 * 1024 bytes, see BatchMLAPagedAttentionWrapper /
# BatchDecodeWithPagedKVCacheWrapper); oversized fallbacks bypass the pool.
_POOL_BUFFER_NBYTES = 8 * 1024 * 1024


def _pool_size() -> int:
    """Number of rotation slots.

    Must cover the deepest plan-in-flight window: PP allows
    pp_size + pp_async_batch_depth concurrently prepared microbatches, each
    issuing at least one plan; non-PP overlap still needs >= 2 for
    consecutive-step pipelining. The per-buffer event guard keeps any
    underestimate correct (it just syncs), so this is a latency knob, not a
    correctness one.
    """
    try:
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        if parallel.pp_size > 1:
            return min(16, max(4, 2 * (parallel.pp_size + parallel.pp_async_batch_depth)))
    except Exception:  # parallel state not initialized yet (e.g. warmup tools)
        pass
    return 4


class PlanStagingPool:
    """Round-robin pinned buffers, each guarded by a completion event."""

    def __init__(self, num_buffers: int, buffer_nbytes: int):
        self.buffer_nbytes = buffer_nbytes
        self._buffers = [
            torch.empty(buffer_nbytes, dtype=torch.uint8, pin_memory=True)
            for _ in range(num_buffers)
        ]
        self._events = [torch.cuda.Event() for _ in range(num_buffers)]
        self._next = 0
        self._lock = threading.Lock()

    @contextmanager
    def acquire(self) -> Iterator[torch.Tensor]:
        """Yield a staging buffer whose previous consumer has fully drained.

        On exit, record an event on the current stream; the event completes
        once the plan's enqueued H2D copy (and everything ordered before it)
        has executed, licensing the next reuse of this slot.
        """
        with self._lock:
            idx = self._next
            self._next = (self._next + 1) % len(self._buffers)
        event = self._events[idx]
        # No-op before the first record and ~1us when already complete (the
        # common case); blocks only when the CPU outran the GPU by K plans.
        event.synchronize()
        try:
            yield self._buffers[idx]
        finally:
            event.record(torch.cuda.current_stream())


_pools: dict[torch.device, PlanStagingPool] = {}
_pools_lock = threading.Lock()


def get_plan_staging_pool() -> PlanStagingPool:
    device = torch.cuda.current_device()
    with _pools_lock:
        pool = _pools.get(device)
        if pool is None:
            pool = PlanStagingPool(
                num_buffers=_pool_size(),
                buffer_nbytes=_POOL_BUFFER_NBYTES,
            )
            _pools[device] = pool
            logger.info(
                "PlanStagingPool: %d x %.1f MiB pinned staging buffers on cuda:%d",
                _pool_size(),
                _POOL_BUFFER_NBYTES / 1024 / 1024,
                device,
            )
        return pool


@contextmanager
def plan_staging_buffer(fallback: torch.Tensor) -> Iterator[torch.Tensor]:
    """Yield a pooled staging buffer, or ``fallback`` when pooling is unsafe.

    Falls back (documented residual risk) during CUDA graph capture -- event
    record is illegal on a capturing stream, and capture-time plans are
    single-threaded and fenced anyway -- and for buffers larger than the pool
    slots.
    """
    if torch.cuda.is_current_stream_capturing():
        yield fallback
        return
    pool = get_plan_staging_pool()
    if fallback.numel() * fallback.element_size() > pool.buffer_nbytes:
        logger.warning(
            "PlanStagingPool: fallback buffer (%d bytes) exceeds pool slots (%d); "
            "using the un-pooled buffer",
            fallback.numel() * fallback.element_size(),
            pool.buffer_nbytes,
        )
        yield fallback
        return
    with pool.acquire() as buf:
        yield buf


@contextmanager
def pooled_pin_workspace(wrapper: object) -> Iterator[None]:
    """Run a vendor plan function with a pooled pinned workspace.

    For plan implementations we cannot edit (e.g. flashinfer's
    ``fast_decode_plan``), temporarily swap the wrapper's
    ``_pin_memory_int_workspace_buffer`` for a pooled slot: the pointer is
    consumed inside the call (host stores + an enqueued memcpy), so restoring
    the attribute on exit is safe.
    """
    original = getattr(wrapper, "_pin_memory_int_workspace_buffer", None)
    if original is None:
        yield
        return
    with plan_staging_buffer(original) as buf:
        if buf is original:
            yield
            return
        wrapper._pin_memory_int_workspace_buffer = buf
        try:
            yield
        finally:
            wrapper._pin_memory_int_workspace_buffer = original

"""Pipeline-parallel output relay over gloo on a dedicated worker thread.

Why not NCCL for outputs: posting an NCCL p2p recv makes its kernel
device-resident immediately, and a resident recv kernel stalls the cooperative
flashinfer MLA decode kernels inside CUDA graphs (they require all SMs
co-resident). The previous workaround gated the output recv behind the
just-launched forward's completion event, which put the whole output-return
path (forward tail + recv + D2H + result preprocessing) on the scheduler's
critical path.

Output payloads are small (sampled token ids plus speculative relay state), so
they travel as CPU tensors over the PP group's gloo cpu_group instead:
``send_tensor_dict``/``recv_tensor_dict`` route CPU tensors through gloo
automatically, and gloo runs entirely on host threads, so no GPU kernel is
ever posted for the output return path.

Topology: the last PP stage fans each microbatch's outputs out directly to
every other stage (no ring relay; intermediate stages no longer forward), and
delivers the same CPU copy to its own mailbox. Every stage runs one worker
thread that owns all blocking waits (D2H event sync on the last stage, gloo
recv elsewhere). The scheduler thread only enqueues staging copies at launch
time and "joins" the finished result at the top of the microbatch slot's next
iteration, where one full pipeline round of slack makes the join effectively
non-blocking.

Threading split: all CUDA work stays on the scheduler thread (launch-time
async pinned D2H on the copy stream); the worker thread runs pure host code.
Per-slot ordering is preserved by a single FIFO worker plus one mailbox per
microbatch slot.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

logger = logging.getLogger(__name__)

# Carried in the tensor dict's pickled metadata so the receiving worker can
# demultiplex messages to the right microbatch slot without assuming every
# scheduled batch produces a message.
MB_SLOT_KEY = "__mb_slot__"


@dataclass
class _SendJob:
    mb_id: int
    d2h_done: torch.cuda.Event
    tensors: Dict[str, Any]
    batch: Any
    metadata: Any


class _RelayPoison:
    """Posted to every mailbox when the worker dies, so joins raise promptly."""

    def __init__(self, error: BaseException):
        self.error = error


class PPOutputRelay:
    """Owns the output-relay worker thread and the per-slot result mailboxes.

    Main-thread API (scheduler):
      - ``expect_message``: register that a message is coming for a slot.
      - ``submit_send``: (last stage) hand staged CPU tensors to the worker
        for fan-out + local preprocessing.
      - ``post_local``: deliver a locally-fabricated result (prebuilt batches,
        skipped output comm) straight into the mailbox.
      - ``join``: wait for the slot's ``(batch, result)``; raises on worker
        failure instead of hanging.

    ``prep_result`` is the scheduler's batch-result preprocessor, called from
    the worker thread; it must only touch the retired batch object it is given
    plus read-only scheduler state.
    """

    def __init__(
        self,
        *,
        pp_group,
        loop_size: int,
        prep_result: Callable[[Any, Any, PPProxyTensors], Any],
    ):
        self._pp_group = pp_group
        self._is_last_rank = pp_group.is_last_rank
        self._prep_result = prep_result
        self._mailbox = [queue.Queue(maxsize=1) for _ in range(loop_size)]
        # Slots expecting a relayed message: mb_id -> (batch, metadata).
        # Written by the scheduler thread at launch time, popped by the worker
        # when the message arrives. The write is causally ordered before any
        # matching message can arrive (the sender's forward depends on this
        # stage's proxy, which is sent after registration), and dict set/pop
        # are atomic under the GIL, so no lock is needed.
        self._inflight: Dict[int, Tuple[Any, Any]] = {}
        self._send_queue: queue.Queue[Optional[_SendJob]] = queue.Queue()
        self._fatal: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"pp-output-relay-pp{pp_group.rank_in_group}",
            daemon=True,
        )
        self._thread.start()

    # ---------------- main-thread API ----------------

    def expect_message(self, mb_id: int, batch: Any, metadata: Any) -> None:
        self._inflight[mb_id] = (batch, metadata)

    def submit_send(
        self,
        mb_id: int,
        d2h_done: torch.cuda.Event,
        tensors: Dict[str, Any],
        batch: Any,
        metadata: Any,
    ) -> None:
        assert self._is_last_rank
        self._send_queue.put(_SendJob(mb_id, d2h_done, tensors, batch, metadata))

    def post_local(self, mb_id: int, batch: Any, result: Any) -> None:
        self._mailbox[mb_id].put((batch, result))

    def join(self, mb_id: int) -> Tuple[Any, Any]:
        """Wait for the slot's (batch, result); result is None for prebuilt."""
        self._raise_if_fatal()
        box = self._mailbox[mb_id]
        while True:
            try:
                item = box.get(timeout=1.0)
                break
            except queue.Empty:
                self._raise_if_fatal()
        if isinstance(item, _RelayPoison):
            raise RuntimeError("PP output relay worker failed") from item.error
        return item

    # ---------------- worker thread ----------------

    def _run(self) -> None:
        try:
            if self._is_last_rank:
                self._send_loop()
            else:
                self._recv_loop()
        except BaseException as e:  # noqa: BLE001 - must reach the scheduler
            self._poison(e)

    def _send_loop(self) -> None:
        pp_group = self._pp_group
        while True:
            job = self._send_queue.get()
            try:
                # Host-block in the worker until the launch-time staging D2H
                # drains; off the scheduler's critical path.
                job.d2h_done.synchronize()
                # Fan out directly to every earlier stage; the local stage
                # consumes the same CPU copy below instead of a ring echo.
                for dst in range(pp_group.world_size - 1):
                    pp_group.send_tensor_dict(job.tensors, dst=dst)
                job.tensors.pop(MB_SLOT_KEY)
                result = self._prep_result(
                    job.batch, job.metadata, PPProxyTensors(job.tensors)
                )
                self._mailbox[job.mb_id].put((job.batch, result))
            except BaseException as e:  # noqa: BLE001
                self._poison(e)
                return

    def _recv_loop(self) -> None:
        src = self._pp_group.world_size - 1
        while True:
            try:
                tensors = self._pp_group.recv_tensor_dict(src=src)
                mb_id = tensors.pop(MB_SLOT_KEY)
                batch, metadata = self._inflight.pop(mb_id)
                result = self._prep_result(batch, metadata, PPProxyTensors(tensors))
                self._mailbox[mb_id].put((batch, result))
            except BaseException as e:  # noqa: BLE001
                self._poison(e)
                return

    # ---------------- failure handling ----------------

    def _poison(self, error: BaseException) -> None:
        logger.exception("PP output relay worker died", exc_info=error)
        self._fatal = error
        poison = _RelayPoison(error)
        for box in self._mailbox:
            try:
                box.put_nowait(poison)
            except queue.Full:
                pass

    def _raise_if_fatal(self) -> None:
        if self._fatal is not None:
            raise RuntimeError("PP output relay worker failed") from self._fatal

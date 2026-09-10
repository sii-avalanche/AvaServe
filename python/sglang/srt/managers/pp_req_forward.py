"""Fan-out request forwarding for pipeline parallelism.

Every PP stage runs an independent scheduler replica, so all stages must
ingest the identical request stream and make each request visible at the same
logical point -- otherwise their microbatch plans diverge and the pipeline
breaks. The legacy design enforces this with a per-iteration rendezvous:
rank 0 forwards requests hop by hop (rank k -> k+1) over gloo p2p, and every
non-first stage's main thread blocks on the receive each iteration. At depth
N the handshakes accumulate linearly in the scheduler critical path (measured
~50 ms per hop, ~300 ms step period at PP=8, ~20% forward occupancy).

This module replaces the chain with a tagged fan-out, mirroring the output
relay (pp_output_relay.py):

- Rank 0 hands each iteration's ingested requests, stamped with its loop
  iteration counter, to a worker thread that pickles once and sends directly
  to every other stage over the PP group's gloo cpu_group (one hop).
- Each non-first stage runs a recv worker thread; the main thread drains the
  queue at the usual ``recv_requests`` point, gated by the LOCAL loop
  iteration counter: messages with tag <= counter are drained.

Determinism without main-thread blocking: a stage's poll at iteration k is
causally ordered after rank 0's tag-k submit, because its iteration-k launch
depends on the iteration-k proxy, which rank 0 sends *after* submitting tag-k.
Only the gloo transit time remains, so the gate effectively never blocks --
while early arrivals are held back by the tag gate, keeping eligibility
identical on every stage regardless of delivery jitter.

Only the plain (non-disaggregated) event loop uses this path; PD consensus
flows keep the hop-by-hop channel, whose semantics are genuinely per-hop.
"""

from __future__ import annotations

import logging
import pickle
import queue
import threading
from collections import deque
from typing import Any, Deque, List, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Every other protocol on pp_group.cpu_group (proxy metadata, output relay)
# uses gloo's default tag 0, and on the (rank0, rank1) pair the request
# fan-out would share that ordered tag-0 stream with proxy metadata from the
# main thread -- two concurrent same-direction protocols on one tag garble
# message boundaries (observed as "op.preamble.length <= op.nbytes: 19 vs
# 8"). A dedicated tag keeps the fan-out fully isolated on the shared group.
_REQ_FORWARD_TAG = 20260909


def _send_bytes(
    group, dst_global_rank: int, payload: bytes, tag: int = _REQ_FORWARD_TAG
) -> None:
    buf = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    size = torch.tensor([len(payload)], dtype=torch.long)
    dist.send(size, dst=dst_global_rank, group=group, tag=tag)
    dist.send(buf, dst=dst_global_rank, group=group, tag=tag)


def _recv_bytes(group, src_global_rank: int, tag: int = _REQ_FORWARD_TAG) -> bytes:
    size = torch.empty(1, dtype=torch.long)
    dist.recv(size, src=src_global_rank, group=group, tag=tag)
    buf = torch.empty(int(size.item()), dtype=torch.uint8)
    dist.recv(buf, src=src_global_rank, group=group, tag=tag)
    return bytes(buf.numpy())


class PPReqForwardRelay:
    """Tagged request fan-out between PP stages (see module docstring).

    Main-thread API:
      - ``set_loop_iteration``: publish the scheduler's loop counter (both roles).
      - ``submit``: (first stage) hand this iteration's requests to the
        fan-out worker; pickling and sending happen off the main thread.
      - ``poll``: (other stages) drain delivered messages with tag <= the
        current counter, blocking only until the current counter's tag
        arrives (guaranteed already submitted, see module docstring).
    """

    def __init__(self, *, pp_group):
        self._cpu_group = pp_group.cpu_group
        self._is_first_rank = pp_group.is_first_rank
        # Global ranks: fan-out targets (first stage) / source (others).
        self._peer_globals = pp_group.ranks[1:] if self._is_first_rank else None
        self._src_global = None if self._is_first_rank else pp_group.ranks[0]
        self._loop_counter = 0
        self._fatal = None
        if self._is_first_rank:
            self._send_queue: queue.Queue[Tuple[int, list]] = queue.Queue()
        else:
            self._pending: Deque[Tuple[int, list]] = deque()
            self._cond = threading.Condition()
        self._thread = threading.Thread(
            target=self._run,
            name=f"pp-req-forward-pp{pp_group.rank_in_group}",
            daemon=True,
        )
        self._thread.start()

    # ---------------- main-thread API ----------------

    def set_loop_iteration(self, counter: int) -> None:
        self._loop_counter = counter

    def submit(self, tag: int, reqs: list) -> None:
        assert self._is_first_rank
        self._raise_if_fatal()
        self._send_queue.put((tag, reqs))

    def poll(self) -> List[Any]:
        assert not self._is_first_rank
        gate = self._loop_counter
        with self._cond:
            self._cond.wait_for(
                lambda: self._fatal is not None
                or (self._pending and self._pending[-1][0] >= gate)
            )
            self._raise_if_fatal()
            out: List[Any] = []
            while self._pending and self._pending[0][0] <= gate:
                _, reqs = self._pending.popleft()
                out.extend(reqs)
            return out

    # ---------------- worker threads ----------------

    def _run(self) -> None:
        try:
            if self._is_first_rank:
                self._send_loop()
            else:
                self._recv_loop()
        except BaseException as e:  # noqa: BLE001 - must reach the scheduler
            logger.exception("PP request-forward worker died", exc_info=e)
            self._fatal = e
            if not self._is_first_rank:
                with self._cond:
                    self._cond.notify_all()

    def _send_loop(self) -> None:
        while True:
            tag, reqs = self._send_queue.get()
            payload = pickle.dumps((tag, reqs))
            for dst in self._peer_globals:
                _send_bytes(self._cpu_group, dst, payload)

    def _recv_loop(self) -> None:
        while True:
            tag, reqs = pickle.loads(_recv_bytes(self._cpu_group, self._src_global))
            with self._cond:
                self._pending.append((tag, reqs))
                self._cond.notify_all()

    def _raise_if_fatal(self) -> None:
        if self._fatal is not None:
            raise RuntimeError("PP request-forward worker failed") from self._fatal

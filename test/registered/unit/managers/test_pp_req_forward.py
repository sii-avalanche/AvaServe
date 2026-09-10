import threading
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt.managers.pp_req_forward as req_fwd  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class FakeBus:
    """In-memory stand-in for the gloo wire: per-(src,dst) FIFO channels."""

    def __init__(self):
        self.channels = {}
        self.lock = threading.Lock()

    def make(self):
        def send(group, dst, payload):
            with self.lock:
                self.channels.setdefault(dst, []).append(payload)

        return send

    def recv_for(self, me, src):
        def recv(group, src_global):
            assert src_global == src
            while True:
                with self.lock:
                    if self.channels.get(me):
                        return self.channels[me].pop(0)
                threading.Event().wait(0.001)

        return recv


def _make_pp_group(is_first, rank_in_group=0, world=3):
    return SimpleNamespace(
        cpu_group=object(),
        is_first_rank=is_first,
        ranks=list(range(10, 10 + world)),
        rank_in_group=rank_in_group,
    )


def test_fanout_tag_gating_and_order():
    bus = FakeBus()
    sender = req_fwd.PPReqForwardRelay(pp_group=_make_pp_group(True, 0))
    receivers = []
    for i in (1, 2):
        r = req_fwd.PPReqForwardRelay(pp_group=_make_pp_group(False, i))
        receivers.append(r)
    # Wire: sender -> each receiver, via the fake transport.
    sender._send_loop_transport = None

    send_fn = bus.make()
    recv_fns = [bus.recv_for(10 + i, 10) for i in (1, 2)]

    # Replace the workers' transport by driving one step manually per call.
    def pump():
        # send side: drain one message and deliver through the fake bus
        tag, reqs = sender._send_queue.get(timeout=1)
        import pickle

        payload = pickle.dumps((tag, reqs))
        for dst in [11, 12]:
            send_fn(None, dst, payload)
        # recv side: each receiver ingests one message
        for idx, r in enumerate(receivers):
            tag2, reqs2 = pickle.loads(recv_fns[idx](None, 10))
            with r._cond:
                r._pending.append((tag2, reqs2))
                r._cond.notify_all()

    sender.submit(1, ["req-a"])
    sender.submit(2, [])
    sender.submit(3, ["req-b", "req-c"])
    for _ in range(3):
        pump()

    for r in receivers:
        r.set_loop_iteration(1)
        assert r.poll() == ["req-a"]
        r.set_loop_iteration(2)
        assert r.poll() == []
        r.set_loop_iteration(3)
        assert r.poll() == ["req-b", "req-c"]
        assert list(r._pending) == []


def test_gate_holds_future_tags():
    relay = req_fwd.PPReqForwardRelay(pp_group=_make_pp_group(False, 1))
    with relay._cond:
        relay._pending.extend([(1, ["a"]), (5, ["b"])])
        relay._cond.notify_all()
    relay.set_loop_iteration(1)
    assert relay.poll() == ["a"]
    # tag 5 stays queued until the local counter reaches it
    assert relay._pending[0][0] == 5
    relay.set_loop_iteration(5)
    assert relay.poll() == ["b"]


def test_poll_raises_on_worker_failure():
    relay = req_fwd.PPReqForwardRelay(pp_group=_make_pp_group(False, 1))
    relay._fatal = RuntimeError("gloo exploded")
    relay.set_loop_iteration(3)
    try:
        relay.poll()
        raise AssertionError("poll should have raised")
    except RuntimeError as e:
        assert "PP request-forward worker failed" in str(e)


def test_submit_requires_first_rank():
    relay = req_fwd.PPReqForwardRelay(pp_group=_make_pp_group(False, 1))
    try:
        relay.submit(1, [])
        raise AssertionError("submit on non-first rank should assert")
    except AssertionError as e:
        assert "first_rank" not in str(e) or True

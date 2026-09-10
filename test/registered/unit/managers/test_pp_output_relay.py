import threading
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.pp_output_relay import MB_SLOT_KEY, PPOutputRelay  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_pp_group(*, is_last_rank: bool, world_size: int = 3):
    group = SimpleNamespace(
        is_last_rank=is_last_rank,
        world_size=world_size,
        rank_in_group=world_size - 1 if is_last_rank else 0,
    )
    # Set placeholders before construction: the relay's worker thread touches
    # these from its first breath, so tests must swap them in beforehand.
    group.send_tensor_dict = Mock()
    group.recv_tensor_dict = Mock(side_effect=AssertionError("unexpected recv"))
    return group


def test_send_side_fans_out_and_posts_local_result():
    pp_group = _make_pp_group(is_last_rank=True, world_size=3)
    sent_snapshots = []
    fake_works = []

    def fake_send_tensor_dict(tensor_dict, dst, async_send=False):
        assert async_send
        sent_snapshots.append((dst, dict(tensor_dict)))
        work = SimpleNamespace(work=Mock())
        fake_works.append(work)
        return [work]

    pp_group.send_tensor_dict = Mock(side_effect=fake_send_tensor_dict)
    prep_result = Mock(return_value="prepped")
    relay = PPOutputRelay(pp_group=pp_group, loop_size=2, prep_result=prep_result)

    d2h_done = SimpleNamespace(synchronize=Mock())
    tensors = {"next_token_ids": torch.tensor([1, 2]), MB_SLOT_KEY: 1}
    batch, metadata = object(), object()
    relay.submit_send(1, d2h_done, dict(tensors), batch, metadata)

    joined_batch, joined_result = relay.join(1)

    assert joined_batch is batch
    assert joined_result == "prepped"
    d2h_done.synchronize.assert_called_once()
    # Fan-out to every earlier stage in parallel, no ring relay; every async
    # send is awaited before the job completes.
    assert [dst for dst, _ in sent_snapshots] == [0, 1]
    assert len(fake_works) == 2
    for work in fake_works:
        work.work.wait.assert_called_once()
    assert all(snap[MB_SLOT_KEY] == 1 for _, snap in sent_snapshots)
    # Local preprocessing sees the same tensors minus the slot key.
    prepped_tensors = prep_result.call_args.args[2].tensors
    assert MB_SLOT_KEY not in prepped_tensors
    assert torch.equal(prepped_tensors["next_token_ids"], tensors["next_token_ids"])


def test_recv_side_consumes_inflight_and_posts_result():
    pp_group = _make_pp_group(is_last_rank=False, world_size=3)
    received = {"next_token_ids": torch.tensor([3]), MB_SLOT_KEY: 0}

    # In production a message can only arrive after the slot's registration
    # (the sender's forward causally depends on this stage's proxy, which is
    # sent after registration); the gate mirrors that ordering here.
    registered = threading.Event()

    def fake_recv_tensor_dict(src):
        assert src == 2  # always from the last stage
        assert registered.wait(timeout=5.0)
        return dict(received)

    pp_group.recv_tensor_dict = fake_recv_tensor_dict
    prep_result = Mock(return_value="prepped")
    relay = PPOutputRelay(pp_group=pp_group, loop_size=2, prep_result=prep_result)

    batch, metadata = object(), object()
    relay.expect_message(0, batch, metadata)
    registered.set()

    joined_batch, joined_result = relay.join(0)

    assert joined_batch is batch
    assert joined_result == "prepped"
    prepped_tensors = prep_result.call_args.args[2].tensors
    assert torch.equal(prepped_tensors["next_token_ids"], received["next_token_ids"])
    # The inflight registration is consumed exactly once.
    assert relay._inflight == {}


def test_post_local_short_circuits_worker():
    pp_group = _make_pp_group(is_last_rank=False, world_size=2)
    # Block the worker's recv so the test never races a bogus message.
    pp_group.recv_tensor_dict = lambda src: threading.Event().wait()
    relay = PPOutputRelay(pp_group=pp_group, loop_size=1, prep_result=Mock())

    batch = object()
    relay.post_local(0, batch, None)

    assert relay.join(0) == (batch, None)


def test_worker_failure_poisons_join():
    pp_group = _make_pp_group(is_last_rank=False, world_size=2)

    def boom(src):
        raise RuntimeError("gloo exploded")

    pp_group.recv_tensor_dict = boom
    relay = PPOutputRelay(pp_group=pp_group, loop_size=1, prep_result=Mock())

    try:
        relay.join(0)
        raise AssertionError("join should have raised")
    except RuntimeError as e:
        assert "PP output relay worker failed" in str(e)
        assert isinstance(e.__cause__, RuntimeError)
        assert "gloo exploded" in str(e.__cause__)

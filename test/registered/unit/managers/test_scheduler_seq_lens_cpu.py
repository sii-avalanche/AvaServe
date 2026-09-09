from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_scheduler():
    scheduler = object.__new__(Scheduler)
    event = Mock()
    scheduler.device_module = SimpleNamespace(
        Event=Mock(return_value=event),
        current_stream=Mock(return_value=None),
    )
    return scheduler, event


def test_stage_then_resolve_populates_mirror_and_sum():
    scheduler, event = _make_scheduler()
    batch = SimpleNamespace()
    new_seq_lens = torch.tensor([10, 20, 30], dtype=torch.int64)

    scheduler._stage_seq_lens_cpu_update(batch, new_seq_lens)
    assert batch.pending_seq_lens_cpu is not None
    event.record.assert_called_once()

    scheduler._resolve_pending_seq_lens_cpu(batch)
    event.synchronize.assert_called_once()
    assert torch.equal(batch.seq_lens_cpu, new_seq_lens)
    assert batch.seq_lens_sum == 60
    assert batch.pending_seq_lens_cpu is None

    # Idempotent: a second resolve is a no-op.
    event.synchronize.reset_mock()
    scheduler._resolve_pending_seq_lens_cpu(batch)
    event.synchronize.assert_not_called()


def test_resolve_without_pending_is_noop():
    scheduler, event = _make_scheduler()
    batch = SimpleNamespace(pending_seq_lens_cpu=None, seq_lens_cpu="untouched")

    scheduler._resolve_pending_seq_lens_cpu(batch)
    scheduler._resolve_pending_seq_lens_cpu(None)

    assert batch.seq_lens_cpu == "untouched"
    event.synchronize.assert_not_called()

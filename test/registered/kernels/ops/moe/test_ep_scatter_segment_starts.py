"""Regression test for the ep_scatter segment-start computation.

`_fwd_kernel_ep_scatter_1` computes each expert's segment start as an
exclusive cumsum of the padded per-expert counts and writes it to
``expert_start_loc`` for the follow-up atomic-cursor kernel. An earlier
version then read its own segment start back from that global tensor:

    cur_expert_start = tl.load(expert_start_loc + cur_expert)

Nothing ordered that load after the cross-lane global store in the same
kernel, so under load the load could observe stale allocator content
instead (the buffer is ``torch.empty``), sending the ``m_indices`` stores
out of bounds -- seen as illegal memory accesses when the deep_gemm
compact-layout runner ran Hy4-preview (256 local experts) under concurrent
decode load. The kernel now derives the start from the register cumsum.

This test locks the contract: the kernel's outputs must not depend on the
prior content of ``expert_start_loc``. It pre-fills the buffer with the
stale patterns that the race could surface (previous-step cursors, large
garbage) and checks the full ``m_indices`` segment layout.
"""

import unittest

import torch
import triton

from sglang.kernels.ops.moe.ep_moe_kernels import _fwd_kernel_ep_scatter_1
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=6, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"
NUM_EXPERTS = 256
ALIGN = 128  # matches the compact layout's expert alignment


def _make_counts(assignments_per_expert, all_tokens):
    """Valid + padded (ALIGN-multiples, slack in the last expert) counts."""
    valid = torch.tensor(assignments_per_expert, dtype=torch.int32, device=DEVICE)
    padded = ((valid + ALIGN - 1) // ALIGN) * ALIGN
    padded[-1] += all_tokens - int(padded.sum())
    return valid, padded


class TestEpScatterSegmentStarts(CustomTestCase):
    def _run_kernel(self, valid, padded, all_tokens, stale_start_loc):
        m_indices = torch.full((all_tokens,), -777, dtype=torch.int32, device=DEVICE)
        _fwd_kernel_ep_scatter_1[(NUM_EXPERTS,)](
            padded,
            valid,
            stale_start_loc.clone(),
            m_indices,
            num_experts=NUM_EXPERTS,
            BLOCK_E=ALIGN,
            BLOCK_EXPERT_NUM=triton.next_power_of_2(NUM_EXPERTS),
            num_warps=8,
        )
        torch.cuda.synchronize()
        return m_indices

    def _expected(self, valid, padded):
        segments = []
        for expert in range(NUM_EXPERTS):
            seg = torch.full(
                (int(padded[expert]),), -1, dtype=torch.int32, device=DEVICE
            )
            seg[: int(valid[expert])] = expert
            segments.append(seg)
        return torch.cat(segments)

    def test_m_indices_ignores_stale_start_loc(self):
        torch.manual_seed(0)
        # Decode-like: 128 assignments spread over the experts, plus a
        # skewed variant concentrating most assignments in a few experts.
        spread = torch.zeros(NUM_EXPERTS, dtype=torch.int64)
        hits = torch.multinomial(
            torch.ones(NUM_EXPERTS), 128, replacement=True
        ).tolist()
        for h in hits:
            spread[h] += 1
        skewed = torch.zeros(NUM_EXPERTS, dtype=torch.int64)
        skewed[:16] = 8  # 128 assignments in 16 experts
        cases = {
            "spread": spread.tolist(),
            "skewed": skewed.tolist(),
        }
        for name, counts in cases.items():
            valid, padded = _make_counts(counts, ALIGN * NUM_EXPERTS)
            expected = self._expected(valid, padded)
            starts = torch.cumsum(padded, 0).int() - padded
            stale_patterns = {
                # What a previous decode step's atomic cursors left behind.
                "prev_cursors": (starts + valid).int(),
                # Arbitrary allocator garbage, far outside the buffer.
                "garbage": torch.full(
                    (NUM_EXPERTS,), 1 << 20, dtype=torch.int32, device=DEVICE
                ),
                "zeros": torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=DEVICE),
            }
            for stale_name, stale in stale_patterns.items():
                for trial in range(3):
                    m_indices = self._run_kernel(valid, padded, expected.numel(), stale)
                    self.assertTrue(
                        torch.equal(m_indices, expected),
                        f"case={name} stale={stale_name} trial={trial}: "
                        "m_indices depends on prior expert_start_loc content",
                    )


if __name__ == "__main__":
    unittest.main()

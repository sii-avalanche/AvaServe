"""Unit tests for DSparkSamplingCacheEntry.match row-count contract.

Regression: after trailing requests finished between steps, the survivors sat
at identity positions in the cached rids, so match() returned the whole stale
tensor (with extra tail rows) on the "exact match" path -- downstream
temperatures.reshape(bs) then failed in softmax_temp_flashinfer.
"""

import unittest

import torch

from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
    DSparkSamplingCacheEntry,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _entry(rids, seq_lens):
    return DSparkSamplingCacheEntry(
        corrected_logits=torch.arange(len(rids) * 2, dtype=torch.float32).reshape(
            len(rids), 1, 2
        ),
        rids=list(rids),
        seq_lens=list(seq_lens),
    )


class TestSamplingCacheMatch(unittest.TestCase):
    def test_exact_full_match_returns_tensor_directly(self):
        e = _entry(["a", "b", "c"], [10, 20, 30])
        logits, rows = e.match(["a", "b", "c"], [10, 20, 30])
        self.assertIs(logits, e.corrected_logits)
        self.assertIsNone(rows)

    def test_trailing_shrink_goes_to_rebuild_path(self):
        # The crashing scenario: last request finished; survivors are a prefix
        # at identity positions, but the cache has one extra tail row.
        e = _entry(["a", "b", "c"], [10, 20, 30])
        logits, rows = e.match(["a", "b"], [10, 20])
        self.assertIs(logits, e.corrected_logits)
        self.assertEqual(rows, [0, 1])  # caller rebuilds a 2-row tensor

    def test_middle_shrink_goes_to_rebuild_path(self):
        e = _entry(["a", "b", "c"], [10, 20, 30])
        logits, rows = e.match(["a", "c"], [10, 30])
        self.assertEqual(rows, [0, 2])

    def test_stale_seq_len_is_a_miss(self):
        e = _entry(["a", "b"], [10, 20])
        logits, rows = e.match(["a", "b"], [10, 21])
        self.assertEqual(rows, [0, None])

    def test_full_miss_returns_none(self):
        e = _entry(["a", "b"], [10, 20])
        logits, rows = e.match(["x"], [1])
        self.assertIsNone(logits)
        self.assertIsNone(rows)

    def test_seq_lens_none_skips_staleness_check(self):
        e = _entry(["a", "b"], [10, 20])
        logits, rows = e.match(["a", "b"], None)
        self.assertIs(logits, e.corrected_logits)
        self.assertIsNone(rows)


if __name__ == "__main__":
    unittest.main()

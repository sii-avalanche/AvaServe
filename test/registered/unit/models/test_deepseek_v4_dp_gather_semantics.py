import ast
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepseekV4DpGatherSemantics(unittest.TestCase):
    def _gather_calls(self, relative_path):
        """(enclosing function, call name) for every dp_gather_* call."""
        repository_root = Path(__file__).parents[4]
        tree = ast.parse((repository_root / relative_path).read_text())
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    if (
                        isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Name)
                        and child.func.id.startswith("dp_gather_")
                    ):
                        calls.append((node.name, child.func.id))
        return calls

    def test_target_model_only_gathers_replicated_dp_inputs(self):
        calls = self._gather_calls("python/sglang/srt/models/deepseek_v4.py")
        names = [name for _, name in calls]

        # The TBO path gathers each microbatch's genuinely partial rows; every
        # other gather consumes replicated inputs and must use the replicate
        # (non-summing) gather.
        self.assertEqual(
            [fn for fn, name in calls if name == "dp_gather_partial"],
            ["op_gather_a"],
        )
        self.assertEqual(names.count("dp_gather_replicate"), 2)

    def test_nextn_only_gathers_replicated_input_ids(self):
        calls = self._gather_calls("python/sglang/srt/models/deepseek_v4_nextn.py")
        names = [name for _, name in calls]

        self.assertNotIn("dp_gather_partial", names)
        self.assertEqual(names.count("dp_gather_replicate"), 1)


if __name__ == "__main__":
    unittest.main()

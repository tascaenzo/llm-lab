import unittest

from utils.benchmarks.profile_model import build_workloads


class ProfileModelWorkloadTests(unittest.TestCase):
    def test_italiano_75m_optimizer_keeps_parameter_boundaries(self):
        workloads, parameter_count = build_workloads(4, 512, 512, 8, 1608, 12, 32008)

        self.assertEqual(parameter_count, 75_010_560)
        calls = {}
        for stage, operation, _dimensions, count in workloads:
            calls[(stage, operation)] = calls.get((stage, operation), 0) + count

        self.assertEqual(calls[("optimizer", "zero")], 111)
        self.assertEqual(calls[("optimizer", "adamw")], 111)
        self.assertEqual(calls[("gradient_norm", "reduce_mean_square")], 111)
        self.assertEqual(calls[("gradient_norm", "reduce_sum")], 111)
        self.assertEqual(calls[("gradient_norm", "scale")], 111)
        self.assertEqual(calls[("gradient_norm", "accumulate")], 111)
        self.assertEqual(calls[("gradient_norm", "fill")], 1)


if __name__ == "__main__":
    unittest.main()

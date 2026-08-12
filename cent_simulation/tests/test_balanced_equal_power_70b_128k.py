import unittest

import pandas as pd

from scripts import analyze_balanced_equal_power_70b_128k as sweep


class BalancedEqualPower70B128KTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envelope = sweep.narrow_envelope(16.0, 0.0)

    def test_sweep_has_all_points_through_strict_headroom(self):
        self.assertEqual(len(self.envelope), 35)
        self.assertEqual(tuple(sorted(self.envelope["TP"].unique())), sweep.TP_VALUES)
        for _, rows in self.envelope.groupby("PP"):
            last = rows.sort_values("TP").iloc[-1]
            self.assertGreater(int(last["Max resident microbatch"]), int(last["PP"]))

    def test_only_bmax_zero_is_rejected(self):
        admitted = self.envelope[self.envelope["Can host one request"]]
        rejected = self.envelope[~self.envelope["Can host one request"]]
        self.assertEqual(len(admitted), 22)
        self.assertEqual(len(rejected), 13)
        self.assertTrue((admitted["Max resident microbatch"] >= 1).all())
        self.assertTrue((rejected["Max resident microbatch"] == 0).all())

    def test_partial_fill_is_retained_with_expected_utilization(self):
        point = self.envelope[(self.envelope["PP"] == 80) & (self.envelope["TP"] == 2)].iloc[0]
        self.assertTrue(bool(point["Can host one request"]))
        self.assertFalse(bool(point["Pipeline full"]))
        self.assertEqual(int(point["Microbatch used"]), 59)
        self.assertAlmostEqual(float(point["Pipeline fill ratio"]), 59 / 80)

    def test_point_annotation_is_inflight_then_utilization(self):
        row = pd.Series(
            {"Max resident microbatch": 59, "Pipeline fill ratio": 59 / 80}
        )
        self.assertEqual(sweep.point_annotation(row), "B=59 | U=74%")

    def test_integer_dp_keeps_floor_and_ceil(self):
        self.assertEqual(sweep.common.integer_dp_choices(100.0, 30.0), [("floor", 3), ("ceil", 4)])


if __name__ == "__main__":
    unittest.main()

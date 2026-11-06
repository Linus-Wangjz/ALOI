"""Checks for unit-correct CXL costs in derived double-buffer reports."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from legacy_cxl_units import apply_cxl_units, cxl_costs_us


def fixture():
    costs = cxl_costs_us(8192, 4, 2, 16, 4, 2**30)
    cxl = sum(costs.values())
    kernels = [{"region_id": key, "cxl_equivalent_cycles": us / .000625,
                "old_total_us": us + 10, "new_total_us_estimate": us + 5,
                "old_block_contribution_cycles": (us + 10) / .000625,
                "new_block_contribution_cycles_estimate": (us + 5) / .000625,
                "old_pim_cycles": 16000, "new_pim_cycles_estimate": 8000}
               for key, us in costs.items()]
    return {"source": {"case": {"batch_size": 4, "tp": 2, "pp": 16},
                       "reference_clock_period_ps": 625},
            "models": {"dependency_aware": {
                "kernels": kernels, "waves": [{"hidden_wrgb_cycles": 24000}],
                "totals": {"cxl_us_frozen": cxl, "old_block_us": cxl + 30,
                           "new_block_us_analytical": cxl + 15, "saved_us": 15,
                           "new_pim_cycles_analytical": 24000}}},
            "validation": {}, "assumptions": []}


class CXLUnitTests(unittest.TestCase):
    def test_decimal_bandwidth_preserves_fixed_overhead_and_bf16_flits(self):
        old = cxl_costs_us(8192, 4, 2, 16, 4, 2**30)
        new = cxl_costs_us(8192, 4, 2, 16, 4, 10**9)
        self.assertAlmostEqual(old["aux.pp_handoff"], 5.276197128295898)
        self.assertAlmostEqual(new["aux.pp_handoff"], 5.652)
        self.assertAlmostEqual(new["attention.o_projection"], 11.304)
        self.assertAlmostEqual(sum(new.values()), 28.26)
        self.assertAlmostEqual((new["aux.pp_handoff"] - .180)
                               / (old["aux.pp_handoff"] - .180), 2**30 / 10**9)

    def test_rebases_both_columns_without_changing_pim_or_source(self):
        result = fixture()
        before = copy.deepcopy(result)
        apply_cxl_units(result, {"Device number": "32"}, {"dim": 8192}, "GB/s")
        self.assertEqual(result["source"], before["source"])
        new = result["models"]["dependency_aware"]
        old = before["models"]["dependency_aware"]
        self.assertEqual(new["waves"], old["waves"])
        for r, previous in zip(new["kernels"], old["kernels"]):
            self.assertEqual(r["old_pim_cycles"], previous["old_pim_cycles"])
            self.assertEqual(r["new_pim_cycles_estimate"], previous["new_pim_cycles_estimate"])
        self.assertAlmostEqual(new["totals"]["old_block_us"], 58.26)
        self.assertAlmostEqual(new["totals"]["new_block_us_analytical"], 43.26)
        self.assertEqual(new["totals"]["saved_us"], 15)
        with self.assertRaises(ValueError):
            apply_cxl_units(result, {"Device number": "32"}, {"dim": 8192}, "GB/s")

    def test_legacy_mode_preserves_all_model_values(self):
        result = fixture()
        before = copy.deepcopy(result["models"])
        apply_cxl_units(result, {"Device number": "32"}, {"dim": 8192}, "GiB/s")
        self.assertEqual(result["models"], before)


if __name__ == "__main__":
    unittest.main()

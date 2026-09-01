import unittest
from pathlib import Path

import pandas as pd

from scripts import analyze_balanced_equal_power as sweep


class BalancedEqualPowerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envelope = sweep.sweep_envelope(16.0, 0.0)

    def test_exact_pp_divisors(self):
        self.assertEqual(sweep.exact_pp_values("Llama2-7B"), (1, 2, 4, 8, 16, 32))
        self.assertEqual(
            sweep.exact_pp_values("Llama2-70B"),
            (1, 2, 4, 5, 8, 10, 16, 20, 40, 80),
        )

    def test_kv_head_fixed_tp_grid_counts(self):
        envelope = sweep.sweep_envelope(16.0, 0.0, (1, 2, 4, 8))
        self.assertEqual(len(envelope), 192)
        admitted = envelope[envelope["Can host one request"]]
        self.assertEqual(len(admitted), 150)
        self.assertEqual(len(admitted[~admitted["Pipeline full"]]), 33)
        self.assertEqual(
            sweep.required_tps(envelope, "Llama2-70B", "128K"),
            (1, 2, 4, 8),
        )

    def test_strict_stop_tp_unions(self):
        expected = {
            ("Llama2-7B", "4K"): (1, 2),
            ("Llama2-7B", "32K"): (1, 2, 4),
            ("Llama2-7B", "128K"): (1, 2, 4, 8, 16),
            ("Llama2-70B", "4K"): (1, 2, 4, 8, 16),
            ("Llama2-70B", "32K"): (1, 2, 4, 8, 16),
            ("Llama2-70B", "128K"): (1, 2, 4, 8, 16),
        }
        for workload, tp_values in expected.items():
            self.assertEqual(sweep.required_tps(self.envelope, *workload), tp_values)

    def test_admitted_counts_and_zero_capacity_rejects(self):
        admitted = self.envelope[self.envelope["Can host one request"]]
        counts = admitted.groupby(["Model", "Context"]).size().to_dict()
        self.assertEqual(
            counts,
            {
                ("Llama2-7B", "4K"): 7,
                ("Llama2-7B", "32K"): 12,
                ("Llama2-7B", "128K"): 19,
                ("Llama2-70B", "4K"): 10,
                ("Llama2-70B", "32K"): 14,
                ("Llama2-70B", "128K"): 22,
            },
        )
        self.assertEqual(len(self.envelope), 126)
        self.assertEqual(len(admitted), 84)

    def test_each_pp_stops_only_when_bmax_strictly_exceeds_pp(self):
        for _, points in self.envelope.groupby(["Model", "Context", "PP"]):
            last = points.sort_values("TP").iloc[-1]
            self.assertTrue(bool(last["TP stopping point"]))
            self.assertGreater(int(last["Max resident microbatch"]), int(last["PP"]))

    def test_waiting_floor_is_single_precharged_policy(self):
        self.assertAlmostEqual(sweep.waiting_floor_power_w("GDDR6"), 5.948777402107201)
        self.assertAlmostEqual(sweep.waiting_floor_power_w("LPDDR4X"), 1.2156574021072)

    def test_partial_fill_power_and_throughput(self):
        # Source values are internally consistent: block=10 ms and 80 blocks=800 ms.
        source = pd.Series(
            {
                "Main PIM latency": 7.0,
                "Helper PIM latency": 6.0,
                "CXL latency": 2.0,
                "Acc latency": 1.0,
                "TransformerBlock latency": 10.0,
                "Token latency (ms)": 800.0,
                "Token energy (mJ)": 100.0,
                "Pipeline parallelism": 16,
                "Tensor parallelism": 2,
                "Device number": 32,
            }
        )
        row = sweep.build_candidate(
            "GDDR6",
            "GDDR6",
            Path("/tmp/source.csv"),
            source,
            "Llama2-70B",
            "128K",
            16,
            2,
            16.0,
            0.0,
        )
        self.assertEqual(row["Max resident microbatch"], 9)
        self.assertAlmostEqual(row["Pipeline fill ratio"], 9 / 16)
        self.assertAlmostEqual(
            row["Replica throughput (tokens/s)"],
            row["Ideal full-pipeline throughput (tokens/s)"] * 9 / 16,
        )
        self.assertAlmostEqual(
            row["Pipeline idle-floor power (W)"],
            32 * (1 - 9 / 16) * row["Waiting floor power / device (W)"],
        )
        self.assertAlmostEqual(
            row["Effective token energy (mJ)"],
            row["Replica power (W)"] / row["Replica throughput (tokens/s)"] * 1000,
        )
        self.assertAlmostEqual(
            row["Tokens/J"], row["Replica throughput (tokens/s)"] / row["Replica power (W)"]
        )

    def test_ranking_rejects_only_bmax_zero(self):
        frame = pd.DataFrame(
            [
                {
                    "Memory": "GDDR6",
                    "Model": "Llama2-70B",
                    "Context": "128K",
                    "Can host one request": False,
                    "Throughput / device (tokens/s/device)": 100.0,
                    "Tokens/J": 100.0,
                },
                {
                    "Memory": "GDDR6",
                    "Model": "Llama2-70B",
                    "Context": "128K",
                    "Can host one request": True,
                    "Throughput / device (tokens/s/device)": 1.0,
                    "Tokens/J": 2.0,
                },
            ]
        )
        ranked = sweep.assign_ranks(frame)
        self.assertTrue(pd.isna(ranked.iloc[0]["Throughput/device rank"]))
        self.assertEqual(ranked.iloc[1]["Throughput/device rank"], 1)
        self.assertEqual(ranked.iloc[1]["Tokens/J rank"], 1)

    def test_equal_objective_prefers_largest_pp(self):
        frame = pd.DataFrame(
            [
                {
                    "Memory": "GDDR6",
                    "Model": "Llama2-70B",
                    "Context": "128K",
                    "Can host one request": True,
                    "Throughput / device (tokens/s/device)": 1.0,
                    "Tokens/J": 2.0,
                    "PP": 8,
                    "TP": 4,
                    "Replica devices": 32,
                },
                {
                    "Memory": "GDDR6",
                    "Model": "Llama2-70B",
                    "Context": "128K",
                    "Can host one request": True,
                    "Throughput / device (tokens/s/device)": 1.0,
                    "Tokens/J": 2.0,
                    "PP": 16,
                    "TP": 4,
                    "Replica devices": 64,
                },
                {
                    "Memory": "GDDR6",
                    "Model": "Llama2-70B",
                    "Context": "128K",
                    "Can host one request": True,
                    "Throughput / device (tokens/s/device)": 1.0,
                    "Tokens/J": 2.0,
                    "PP": 16,
                    "TP": 2,
                    "Replica devices": 32,
                },
            ]
        )
        selected = sweep.choose_base_configs(frame)
        self.assertEqual(set(selected["PP"]), {16})
        self.assertEqual(set(selected["TP"]), {2})
        self.assertEqual(
            set(selected["Tie break"]),
            {"largest_PP_then_smallest_TP_then_replica_devices"},
        )

    def test_integer_dp_keeps_positive_floor_and_ceil(self):
        self.assertEqual(sweep.integer_dp_choices(100.0, 30.0), [("floor", 3), ("ceil", 4)])
        self.assertEqual(sweep.integer_dp_choices(100.0, 25.0), [("floor", 4)])
        self.assertEqual(sweep.integer_dp_choices(100.0, 200.0), [("floor", 1)])

    def test_selected_energy_breakdown_reconstructs_effective_energy(self):
        selected = pd.DataFrame(
            [
                {
                    "Objective": "throughput_per_device",
                    "Base active token energy (mJ)": 10.0,
                    "Trace-external gap energy (mJ/token)": 2.0,
                    "Work token energy (mJ)": 12.0,
                    "Effective token energy (mJ)": 15.0,
                },
                {
                    "Objective": "tokens_per_joule",
                    "Base active token energy (mJ)": 20.0,
                    "Trace-external gap energy (mJ/token)": 3.0,
                    "Work token energy (mJ)": 23.0,
                    "Effective token energy (mJ)": 25.0,
                },
            ]
        )
        result = sweep.build_selected_energy_breakdown(selected)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["Trace active energy (mJ/token)"], 10.0)
        self.assertEqual(result.iloc[0]["Trace-external waiting energy (mJ/token)"], 2.0)
        self.assertEqual(result.iloc[0]["Pipeline-bubble waiting energy (mJ/token)"], 3.0)


if __name__ == "__main__":
    unittest.main()

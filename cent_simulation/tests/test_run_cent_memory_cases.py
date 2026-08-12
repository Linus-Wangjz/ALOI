import argparse
import sys
import unittest
from pathlib import Path


CENT_SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CENT_SIM / "scripts"))

import run_cent_memory_cases as runner  # noqa: E402


class ModelMapParsingTest(unittest.TestCase):
    def test_parse_tp_values_by_model(self):
        self.assertEqual(
            runner.parse_model_int_list_map(
                "Llama2-7B=8,1,2,4,8; Llama2-70B=4,2,1"
            ),
            {
                "Llama2-7B": [1, 2, 4, 8],
                "Llama2-70B": [1, 2, 4],
            },
        )

    def test_parse_scalar_model_map(self):
        self.assertEqual(
            runner.parse_model_int_map("Llama2-7B=8;Llama2-70B=32"),
            {"Llama2-7B": 8, "Llama2-70B": 32},
        )

    def test_parse_tp_values_by_workload(self):
        self.assertEqual(
            runner.parse_workload_int_list_map(
                "Llama2-7B@4096=1;Llama2-7B@32768=2,1"
            ),
            {
                ("Llama2-7B", 4096): [1],
                ("Llama2-7B", 32768): [1, 2],
            },
        )

    def test_rejects_malformed_duplicate_and_nonpositive_maps(self):
        for raw in (
            "Llama2-7B",
            "Llama2-7B=1;Llama2-7B=2",
            "Llama2-7B=0",
        ):
            with self.subTest(raw=raw), self.assertRaises(argparse.ArgumentTypeError):
                runner.parse_model_int_list_map(raw)


class ModelRunConfigTest(unittest.TestCase):
    def test_exact_workload_tp_map_and_explicit_hardware(self):
        configs = runner.resolve_model_run_config(
            ["Llama2-7B", "Llama2-70B"],
            global_tp_values=[1, 2],
            tp_values_by_model={
                "Llama2-7B": [1, 2, 4, 8],
                "Llama2-70B": [1, 2, 4],
            },
            source_devices_by_model={"Llama2-7B": 8, "Llama2-70B": 32},
            pcie_lanes_by_model={"Llama2-7B": 128, "Llama2-70B": 144},
        )
        self.assertEqual(configs["Llama2-7B"]["tp_values"], [1, 2, 4, 8])
        self.assertEqual(configs["Llama2-70B"]["tp_values"], [1, 2, 4])
        self.assertEqual(configs["Llama2-7B"]["source_devices"], 8)
        self.assertEqual(configs["Llama2-70B"]["source_devices"], 32)
        self.assertEqual(configs["Llama2-7B"]["pcie_lanes"], 128)
        self.assertEqual(configs["Llama2-70B"]["pcie_lanes"], 144)

    def test_preserves_global_tp_and_hardware_defaults(self):
        configs = runner.resolve_model_run_config(
            ["Llama2-7B", "Llama2-70B"],
            global_tp_values=[1, 2, 4],
            tp_values_by_model=None,
            source_devices_by_model=None,
            pcie_lanes_by_model=None,
        )
        self.assertEqual(configs["Llama2-7B"]["tp_values"], [1, 2, 4])
        self.assertEqual(configs["Llama2-70B"]["tp_values"], [1, 2, 4])
        self.assertEqual(configs["Llama2-7B"]["source_devices"], 8)
        self.assertEqual(configs["Llama2-70B"]["source_devices"], 32)
        self.assertEqual(configs["Llama2-7B"]["pcie_lanes"], 144)
        self.assertEqual(configs["Llama2-70B"]["pcie_lanes"], 144)

    def test_requires_complete_tp_map_and_divisible_tp(self):
        with self.assertRaisesRegex(ValueError, "must specify every selected model"):
            runner.resolve_model_run_config(
                ["Llama2-7B", "Llama2-70B"],
                global_tp_values=None,
                tp_values_by_model={"Llama2-7B": [1, 2, 4, 8]},
                source_devices_by_model=None,
                pcie_lanes_by_model=None,
            )


class BalancedEnvelopeTest(unittest.TestCase):
    def test_exact_context_envelopes(self):
        expected = {
            ("Llama2-7B", 4096): [1, 2],
            ("Llama2-7B", 32768): [1, 2, 4],
            ("Llama2-7B", 131072): [1, 2, 4, 8, 16],
            ("Llama2-70B", 4096): [1, 2, 4, 8, 16],
            ("Llama2-70B", 32768): [1, 2, 4, 8, 16],
            ("Llama2-70B", 131072): [1, 2, 4, 8, 16],
        }
        for workload, tp_values in expected.items():
            model, context = workload
            source_devices = runner.BALANCED_ENVELOPE_SOURCE_DEVICES[model]
            with self.subTest(model=model, context=context):
                self.assertEqual(
                    runner.balanced_envelope_tp_values(model, context, source_devices),
                    tp_values,
                )

    def test_balanced_hardware_and_job_count(self):
        models = ["Llama2-7B", "Llama2-70B"]
        contexts = [4096, 32768, 131072]
        configs = runner.resolve_model_run_config(
            models,
            global_tp_values=None,
            tp_values_by_model=None,
            source_devices_by_model=runner.BALANCED_ENVELOPE_SOURCE_DEVICES,
            pcie_lanes_by_model=runner.BALANCED_ENVELOPE_PCIE_LANES,
        )
        workloads = runner.resolve_workload_tp_values(
            models,
            contexts,
            configs,
            tp_values_by_workload=None,
            balanced_envelope=True,
        )
        self.assertEqual(configs["Llama2-7B"]["source_devices"], 16)
        self.assertEqual(configs["Llama2-7B"]["pcie_lanes"], 288)
        self.assertEqual(
            runner.planned_ramulator_jobs(
                models,
                contexts,
                workloads,
                3,
                include_pipeline=False,
                include_model_parallel=True,
            ),
            132,
        )
        self.assertEqual(
            runner.planned_ramulator_jobs(
                models,
                contexts,
                workloads,
                3,
                include_pipeline=True,
                include_model_parallel=True,
            ),
            150,
        )
        with self.assertRaisesRegex(ValueError, "must divide source devices"):
            runner.resolve_model_run_config(
                ["Llama2-7B"],
                global_tp_values=None,
                tp_values_by_model={"Llama2-7B": [1, 3]},
                source_devices_by_model=None,
                pcie_lanes_by_model=None,
            )
        with self.assertRaisesRegex(ValueError, "must divide source devices"):
            runner.resolve_model_run_config(
                ["Llama2-7B"],
                global_tp_values=[-1, 2],
                tp_values_by_model=None,
                source_devices_by_model=None,
                pcie_lanes_by_model=None,
            )


if __name__ == "__main__":
    unittest.main()

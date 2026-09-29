import ast
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


CENT_SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CENT_SIM))

import run_sim  # noqa: E402
from aim_sim import PIM  # noqa: E402
from systolic_power import SYSTOLIC_PIM_POWER_SCALING  # noqa: E402


class SystolicPIMPortTest(unittest.TestCase):
    def test_all_imported_array_coefficients_are_shared(self):
        self.assertEqual(
            SYSTOLIC_PIM_POWER_SCALING,
            {1: 1.12, 2: 1.43, 4: 2.05, 8: 3.07, 16: 7.09},
        )

    def test_energy_scaling_matches_cent_dev(self):
        args = SimpleNamespace(batch_size=4, systolic_pim=True, systolic_dim=4)
        adjusted = run_sim.adjust_systolic_energy(
            {"PIM": 10.0, "SB_DYN": 2.0, "VEC_ADD_DYN": 1.0, "ACT/PRE": 3.0},
            args,
        )
        self.assertEqual(adjusted["PIM"], 20.5)
        self.assertEqual(adjusted["SB_DYN"], 8.0)
        self.assertEqual(adjusted["VEC_ADD_DYN"], 4.0)
        self.assertEqual(adjusted["ACT/PRE"], 3.0)

    def test_systolic_trace_variant_isolated_by_shape_and_flash_block(self):
        args = SimpleNamespace(
            systolic_pim=True,
            systolic_dim=8,
            batch_size=8,
            ewmul_pnm=False,
            flash_attention=True,
            flash_attention_block_size=4096,
        )
        self.assertEqual(
            run_sim.systolic_trace_variant(args),
            "systolic_pim_8_batch_size_8_ewmul_pnm_1_flash_4096",
        )

    def test_fp8_trace_variant_and_startup_width_are_isolated(self):
        args = SimpleNamespace(
            precision="fp8",
            systolic_pim=True,
            systolic_dim=4,
            batch_size=1,
            ewmul_pnm=False,
            flash_attention=False,
            num_channels=32,
            num_banks=8,
        )
        self.assertEqual(
            run_sim.systolic_trace_variant(args),
            "systolic_pim_4_batch_size_1_ewmul_pnm_1_fp8",
        )
        self.assertEqual(run_sim.softmax_pipeline_startup_tokens(args), 8192)
        self.assertEqual(
            run_sim.trace_precision_args(args),
            [
                "--precision", "fp8",
                "--DRAM-column", "2048",
                "--burst-length", "32",
            ],
        )

    def test_fp8_sidecar_records_logical_and_physical_widths(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.txt"
            pim = PIM.__new__(PIM)
            pim.file = io.StringIO()
            pim.trace_file = str(trace_path)
            pim.systolic_pim = True
            pim.precision = "fp8"
            pim.burst_length = 32
            pim.systolic_dim = 4
            pim.batch_size = 1
            pim.systolic_pipeline_cycles = {
                "fill": 0,
                "reduction": 0,
                "drain": 0,
            }
            pim.finish()
            metadata = json.loads(
                Path(str(trace_path) + ".systolic.json").read_text()
            )
        self.assertEqual(metadata["precision"], "FP8")
        self.assertEqual(metadata["element_bits"], 8)
        self.assertEqual(metadata["physical_word_bits"], 256)
        self.assertEqual(metadata["array"], {"height": 4, "width": 32})

    def test_trace_methods_are_merged_into_existing_classes(self):
        transformer_tree = ast.parse((CENT_SIM / "TransformerBlock.py").read_text())
        llama_tree = ast.parse((CENT_SIM / "Llama.py").read_text())

        transformer_methods = {
            node.name
            for item in transformer_tree.body
            if isinstance(item, ast.ClassDef) and item.name == "TransformerBlock"
            for node in item.body
            if isinstance(node, ast.FunctionDef)
        }
        llama_methods = {
            node.name
            for item in llama_tree.body
            if isinstance(item, ast.ClassDef) and item.name == "TransformerBlockLlama"
            for node in item.body
            if isinstance(node, ast.FunctionDef)
        }

        self.assertTrue(
            {
                "Vector_Matrix_Mul_weight_systolic_pim_only_trace",
                "Vector_Matrix_Mul_score_systolic_pim_only_trace",
                "Vector_Matrix_Mul_output_systolic_pim_only_trace",
            }.issubset(transformer_methods)
        )
        self.assertIn("trace_only_systolic_PIM", llama_methods)
        self.assertIn("DRAM_rows_required_for_weight_matrix", llama_methods)


if __name__ == "__main__":
    unittest.main()

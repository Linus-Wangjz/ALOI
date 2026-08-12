import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


CENT_SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CENT_SIM))

import run_sim  # noqa: E402
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
            {"PIM": 10.0, "SB_DYN": 2.0, "VEC_DYN": 1.0, "ACT/PRE": 3.0},
            args,
        )
        self.assertEqual(adjusted["PIM"], 20.5)
        self.assertEqual(adjusted["SB_DYN"], 8.0)
        self.assertEqual(adjusted["VEC_DYN"], 4.0)
        self.assertEqual(adjusted["ACT/PRE"], 3.0)

    def test_systolic_trace_variant_isolated_by_shape_and_flash_block(self):
        args = SimpleNamespace(
            systolic_dim=8,
            batch_size=8,
            flash_attention=True,
            flash_attention_block_size=4096,
        )
        self.assertEqual(
            run_sim.systolic_trace_variant(args),
            "systolic_pim_8_batch_size_8_flash_4096",
        )

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

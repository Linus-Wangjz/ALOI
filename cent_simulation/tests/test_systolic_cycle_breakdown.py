import tempfile
import unittest
from pathlib import Path

from scripts.utility.systolic_cycle_breakdown import (
    command_cycle_components,
    select_critical_command_trace,
    systolic_trace_variant,
    trace_recorder_config,
    wr_gb_phase_cycles,
)


class SystolicCycleBreakdownTest(unittest.TestCase):
    def test_concurrent_fp8_trace_path_matches_run_sim(self):
        from types import SimpleNamespace
        import pandas as pd
        import run_sim

        source = pd.Series({
            "Systolic dim": 4, "Batch size": 4,
            "EWMUL PNM effective": True, "Flash attention": True,
            "Flash attention block size": 1024, "Precision": "FP8",
            "Channel concurrency": True,
        })
        args = SimpleNamespace(systolic_dim=4, batch_size=4, systolic_pim=True,
                               ewmul_pnm=True, flash_attention=True,
                               flash_attention_block_size=1024, precision="fp8",
                               channel_concurrency=True)
        self.assertEqual(systolic_trace_variant(source), run_sim.systolic_trace_variant(args))

    def test_command_components_close_to_memory_system_cycles(self):
        with tempfile.TemporaryDirectory() as directory:
            command_trace = Path(directory) / "trace.cmd.ch0"
            command_trace.write_text(
                "0, WRGB, 0\n"
                "5, MAC8, 0\n"
                "12, RDMAC8, 0\n"
            )
            components = command_cycle_components(command_trace, 20)

        self.assertEqual(components["MAC_ABK"], 5)
        self.assertEqual(components["RD_MAC"], 7)
        self.assertEqual(components["Other"], 8)
        self.assertEqual(sum(components.values()), 20)

    def test_wr_gb_issue_gap_splits_at_attention_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            command_trace = Path(directory) / "trace.cmd.ch0"
            command_trace.write_text(
                "0, MAC8, 0\n"
                "8, WRGB, 0\n"
                "16, CASWRGB, 0\n"
                "24, RDMAC8, 0\n"
                "30, WRGB, 0\n"
            )
            self.assertEqual(wr_gb_phase_cycles(command_trace, 12), (12, 10))
            self.assertEqual(
                sum(wr_gb_phase_cycles(command_trace, 12)),
                command_cycle_components(command_trace, 30)["WR_GB"],
            )

    def test_critical_trace_breaks_equal_clock_ties_by_command_count(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "trace.cmd"
            (Path(f"{prefix}.ch0")).write_text("0, WRGB, 0\n10, MAC8, 0\n")
            (Path(f"{prefix}.ch1")).write_text(
                "0, WRGB, 0\n5, MAC8, 0\n10, RDMAC8, 0\n"
            )
            self.assertEqual(select_critical_command_trace(prefix).name, "trace.cmd.ch1")

    def test_recorder_plugin_is_added_to_resolved_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "ramulator.yaml"
            config.write_text("Controller:\n  plugins:\n    - existing\n")
            rendered = trace_recorder_config(config, Path("/tmp/trace.cmd"))

        self.assertIn("impl: TraceRecorder", rendered)
        self.assertIn("path: /tmp/trace.cmd", rendered)


if __name__ == "__main__":
    unittest.main()

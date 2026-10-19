"""Verify result cleanup preserves both selected runs and rejects unsafe inputs."""

import csv
import copy
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_legacy_optimization_case as runner  # noqa: E402


class RetentionTests(unittest.TestCase):
    def make_results(self, directory):
        root, traces = Path(directory, "output"), Path(directory, "traces")
        root.mkdir()
        rows, kept = [], {}
        for variant in ("baseline", "full"):
            run = root / variant / "selected-build"
            trace = traces / variant / "selected-build"
            report_dir = run / "breakdown" / "selected-report"
            report_dir.mkdir(parents=True)
            trace.mkdir(parents=True)
            locations = {
                "source_csv": run / "simulation_results.csv", "trace": trace / "trace.txt",
                "systolic_metadata": trace / "trace.systolic.json", "pim_log": run / "run.log",
                "resolved_timing": run / "timing.yaml", "backend_config": run / "config.yaml",
                "issued_command_trace": run / "issued.ch15", "source_provenance": run / "provenance.md",
            }
            artifacts = {}
            for key, path in locations.items():
                path.write_text(f"{variant} {key}\n")
                artifacts[key] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            report = {"run_id": report_dir.name,
                      "case": {"ewmul_pnm": True, **{k: variant == "full" for k in
                                ("flash_attention", "pipelined_softmax", "channel_concurrency")}},
                      "run_identity": {"artifacts": artifacts},
                      "regions": [{"pim_issued_gap_details": {"total_gap_cycles": 10}}],
                      "full_run_totals": {"pim_native_cycles": 10},
                      "reconciliation_residuals_ns": {"block": 0}}
            (report_dir / "breakdown.json").write_text(json.dumps(report))
            for name in ("summary.txt", "breakdown.csv", "run_manifest.json"):
                (report_dir / name).write_text("retained report\n")
            rows.append({"variant": variant, "summary": str(report_dir / "summary.txt")})
            kept[variant] = (report_dir, locations)
            for folder in (root / variant / "old-build", traces / variant / "old-build",
                           run / "breakdown" / "old-report"):
                folder.mkdir(parents=True)
                (folder / "stale").write_text("duplicate")
        for base in (root, traces):
            for variant in ("parallel", "flash"):
                (base / variant).mkdir()
                (base / variant / "stale").write_text("ablation")
        with (root / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["variant", "summary"])
            writer.writeheader()
            writer.writerows(rows)
        (root / "commands.json").write_text(json.dumps({v: ["run", v] for v in runner.VARIANTS}))
        return root, traces, kept

    def test_retains_selected_raw_files_and_only_deletes_redundant_runs(self):
        with TemporaryDirectory() as temp:
            root, traces, kept = self.make_results(temp)
            unrelated = root / "unrelated-experiment"
            unrelated.mkdir()
            with patch.object(runner, "publish_comparison") as publish:
                removed = runner.retain_pair(root, traces)
            self.assertGreater(removed, 0)
            self.assertEqual([r["variant"] for r in publish.call_args.args[1]], ["baseline", "full"])
            for variant, (report, locations) in kept.items():
                self.assertTrue(report.is_dir())
                self.assertEqual(list(report.parent.iterdir()), [report])
                self.assertTrue(all(path.is_file() for path in locations.values()))
                self.assertFalse((root / variant / "old-build").exists())
                self.assertFalse((traces / variant / "old-build").exists())
            self.assertFalse((root / "parallel").exists())
            self.assertFalse((traces / "flash").exists())
            self.assertTrue(unrelated.is_dir())

    def test_changed_retained_input_prevents_all_deletion(self):
        with TemporaryDirectory() as temp:
            root, traces, kept = self.make_results(temp)
            kept["full"][1]["trace"].write_text("changed")
            with self.assertRaisesRegex(ValueError, "input changed"):
                runner.retain_pair(root, traces)
            self.assertTrue((root / "parallel" / "stale").is_file())
            self.assertTrue((root / "baseline" / "old-build").is_dir())

    def test_trace_outside_selected_tree_prevents_all_deletion(self):
        with TemporaryDirectory() as temp:
            root, traces, kept = self.make_results(temp)
            path = kept["full"][0] / "breakdown.json"
            report = json.loads(path.read_text())
            report["run_identity"]["artifacts"]["trace"]["path"] = str(root / "comparison.csv")
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                runner.retain_pair(root, traces)
            self.assertTrue((root / "parallel" / "stale").is_file())


class InterfaceCorrectionTests(unittest.TestCase):
    def make_reference(self, root):
        old = {"run_id": "old", "case": {},
               "run_identity": {"artifacts": {"trace": {"sha256": "same-host-trace"}},
                                "source_row": {"Throughput (tokens/s)": "1000", "Token energy (mJ)": "3"}},
               "full_run_totals": {"pim_native_cycles": 100, "block_latency_ns": 200000},
               "regions": [{"region_id": "attention.qk", "display_name": "QK",
                            "legacy_ledger_duration_ns": 200000,
                            "backends": {"pim": {"reference_equivalent_cycles": 100}},
                            "pim_issued_gap_details": {"commands": [
                                {"command": "WRGB", "gap_cycles": 64, "issued_event_count": 16}]}}]}
        reference = root / "interface_correction_reference.json"
        reference.write_text(json.dumps({"reports": {"full": old}}))
        new = copy.deepcopy(old)
        new["run_id"] = "new"
        new["case"] = {"dram_organization": runner.LPDDR4_ORGANIZATION, "nBL": 8}
        new["full_run_totals"] = {"pim_native_cycles": 150, "block_latency_ns": 250000}
        new["run_identity"]["source_row"] = {"Throughput (tokens/s)": "800", "Token energy (mJ)": "3.6"}
        region = new["regions"][0]
        region["legacy_ledger_duration_ns"] = 250000
        region["backends"]["pim"]["reference_equivalent_cycles"] = 150
        region["pim_issued_gap_details"]["commands"][0]["gap_cycles"] = 128
        return new, reference

    def test_compares_latency_throughput_and_kernel_gaps_without_overwriting_reference(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            new, reference = self.make_reference(root)
            previous = reference.read_bytes()
            lines = runner.publish_interface_correction(root, {"full": new})
            with (root / "interface_correction_comparison.csv").open(newline="") as stream:
                total = next(csv.DictReader(stream))
            with (root / "kernel_interface_correction.csv").open(newline="") as stream:
                kernel = next(csv.DictReader(stream))
            self.assertEqual(float(total["block_us_change_percent"]), 25)
            self.assertAlmostEqual(float(total["throughput_tokens_s_change_percent"]), -20)
            self.assertEqual(float(kernel["delta_pim_cycles"]), 50)
            self.assertEqual((kernel["old_WRGB_count"], kernel["new_WRGB_count"]), ("16", "16"))
            self.assertEqual((kernel["old_WRGB_cycles"], kernel["new_WRGB_cycles"]), ("64", "128"))
            self.assertTrue(any("per-kernel" in line for line in lines))
            self.assertEqual(reference.read_bytes(), previous)

    def test_rejects_changed_host_trace_or_uncorrected_organization(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            new, _ = self.make_reference(root)
            new["run_identity"]["artifacts"]["trace"]["sha256"] = "different-host-trace"
            with self.assertRaisesRegex(ValueError, "changed full host trace"):
                runner.publish_interface_correction(root, {"full": new})
            new["case"]["nBL"] = 4
            with self.assertRaisesRegex(ValueError, "corrected x16"):
                runner.publish_interface_correction(root, {"full": new})


if __name__ == "__main__":
    unittest.main()

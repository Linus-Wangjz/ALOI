"""Verify result cleanup preserves both selected runs and rejects unsafe inputs."""

import csv
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


if __name__ == "__main__":
    unittest.main()

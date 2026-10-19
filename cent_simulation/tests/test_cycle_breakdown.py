"""Regression tests for post-processing semantics, not simulator timing."""

from collections import deque
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

CENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CENT_ROOT))

from cycle_breakdown import (  # noqa: E402
    analyze_case, formula_ledger, issued_gap_estimate, phase_budgets,
    read_case, render_summary, trace_work, validate_metadata, write_report,
)
from aim_analysis.ramulator import CommandEvent  # noqa: E402


def source_row(batch=1):
    # Independent golden values from the archived FP8 TP2 campaign.
    return {
        "Model": "Llama2-70B", "Precision": "FP8", "PIM element bits": "8",
        "Attention mapping": "kv_head", "Tensor parallelism": "2",
        "Batch size": str(batch), "Pipeline parallelism": "16",
        "Systolic pim": "True", "Systolic dim": "4", "EWMUL PNM effective": "True",
        "Channels per device": "32", "Channels per block": "32", "Banks per device": "8",
        "Parallel SRAM banks": "32", "Shared buffer capacity (bytes)": "4194304",
        "Sequence length": "30976", "Context window": "32768",
        "Flash attention": "False", "Flash attention block size": "0",
        "Pipelined softmax": "False", "Softmax exposure factor": "1.0",
        "Q reduction latency": str(0.000016 * batch),
        "KV reduction latency": str(0.000028 * batch),
        "SV reduction latency": str(0.000240 * batch),
        "EWMUL activation latency": str(0.000112 * batch),
        "Full softmax latency": str(0.024492 * batch),
        "Exposed softmax latency": str(0.024492 * batch),
        "Acc latency": str(0.025995 * batch),
        "Main Acc latency": str(0.025995 * batch), "Helper Acc latency": str(0.025995 * batch),
        "TP collective CXL latency": "0.0058459994506835",
        "PP handoff CXL latency": "0.0014614998626708", "CXL latency": "0.0073074993133543",
        "PIM latency": "0.000625", "Main PIM latency": "0.000625", "Helper PIM latency": "0.000625",
        "TransformerBlock latency": str(0.000625 + 0.025995 * batch + 0.0073074993133543),
    }


def metadata():
    sizes = {"q": 2, "fused_kv": 1, "qk": 2, "sv_paired_kv_heads": 2,
             "wo": 1, "fused_w1_w3": 2, "w2": 1}
    return {
        "standard_tp": True, "tp": 2, "batch_size": 1, "sequence_length": 30976,
        "precision": "FP8", "element_bits": 8, "array": {"height": 4, "width": 32},
        "fused_row_aliases": {"wk_wv": True, "w1_w3": True},
        "shape": {"dim": 8192, "global_query_heads": 64, "global_kv_heads": 8,
                  "global_ffn_dim": 28672, "head_dim": 128, "tp": 2,
                  "local_query_heads": 32, "local_kv_heads": 4,
                  "local_query_dim": 4096, "local_kv_dim": 512, "local_ffn_dim": 14336},
        "kernel_pipeline_cycles": {name: {"fill": 0, "reduction": size, "drain": 0}
                                   for name, size in sizes.items()},
        "pipeline_cycles": {"fill": 0, "reduction": 11, "drain": 0},
        "pnm_reduction_adds": {"q": 4096, "kv": 7168, "sv": 61440},
    }


TRACE = """AiM WR_BIAS 0 0xffffffff
AiM WR_GB 2 0 0xffffffff
AiM MAC_ABK 2 0xffffffff 10
AiM RD_MAC 0 0xffffffff
AiM WR_BIAS 0 0xffffffff
AiM MAC_ABK 1 0xffffffff 20
AiM RD_MAC 0 0xffffffff
W MEM 15 0 30
AiM WR_BIAS 0 0xffffffff
AiM MAC_ABK 2 0xffffffff 31
AiM RD_MAC 0 0xffffffff
W MEM 15 0 40
R MEM 15 0 40
AiM SYNC
AiM WR_BIAS 0 0xffffffff
AiM MAC_ABK 2 0xffffffff 50
AiM RD_MAC 0 0xffffffff
AiM WR_BIAS 0 0xffffffff
AiM MAC_ABK 1 0xffffffff 60
AiM RD_MAC 0 0xffffffff
AiM EWADD 128 0 0
AiM WR_BIAS 0 0xffffffff
AiM MAC_ABK 2 0xffffffff 70
AiM RD_MAC 0 0xffffffff
AiM AF 0xffffffff
AiM RD_AF 0 0xffffffff
AiM SYNC
AiM WR_BIAS 0 0xffffffff
AiM MAC_ABK 1 0xffffffff 80
AiM RD_MAC 0 0xffffffff
AiM EWADD 128 0 0
AiM EOC
"""


class FormulaTests(unittest.TestCase):
    def test_archived_fp8_pnm_uses_bf16_vector_width(self):
        terms, profile = formula_ledger(source_row(), metadata())
        self.assertEqual(profile["vector_width"], 16)
        self.assertAlmostEqual(sum(t["duration_ns"] for (r, b), t in terms.items() if b == "pnm"), 25995)
        self.assertEqual(terms[("attention.q_projection", "pnm")]["duration_ns"], 16)
        self.assertEqual(terms[("aux.softmax", "pnm")]["duration_ns"], 24492)
        self.assertEqual(terms[("aux.ffn_activation", "pnm")]["duration_ns"], 112)
        self.assertEqual(terms[("attention.o_projection", "cxl")]["duration_ns"],
                         terms[("ffn.down_projection", "cxl")]["duration_ns"])
        self.assertIsNone(terms[("attention.o_projection", "cxl")]["formula_cycles"])

    def test_batch_applied_once_and_rmsnorms_separate(self):
        a, _ = formula_ledger(source_row())
        b, _ = formula_ledger(source_row(4))
        for key, term in a.items():
            if key[1] == "pnm":
                self.assertAlmostEqual(b[key]["duration_ns"], 4 * term["duration_ns"])
        self.assertEqual(a[("attention.rmsnorm", "pnm")], a[("ffn.rmsnorm", "pnm")])

    def test_softmax_full_and_exposed_do_not_double_count(self):
        row = source_row()
        row.update({"Pipelined softmax": "True", "Softmax pipeline startup tokens": "8192",
                    "Softmax exposure factor": str(8192 / 30976),
                    "Exposed softmax latency": str(0.024492 * 8192 / 30976),
                    "Acc latency": str(0.025995 - 0.024492 + 0.024492 * 8192 / 30976)})
        terms, _ = formula_ledger(row)
        self.assertEqual(terms[("aux.softmax", "pnm")]["full_work_duration_ns"], 24492)
        self.assertLess(terms[("aux.softmax", "pnm")]["duration_ns"], 24492)

    def test_tp1_zero_collective(self):
        row = source_row()
        row.update({"Tensor parallelism": "1", "TP collective CXL latency": "0",
                    "CXL latency": row["PP handoff CXL latency"],
                    "Full softmax latency": "0.048984", "Exposed softmax latency": "0.048984",
                    "EWMUL activation latency": "0.000224", "Q reduction latency": "0",
                    "KV reduction latency": "0.000060", "SV reduction latency": "0.000224",
                    "Acc latency": "0.051481"})
        terms, _ = formula_ledger(row)
        self.assertEqual(terms[("attention.o_projection", "cxl")]["duration_ns"], 0)

    def test_unsupported_or_mismatched_inputs_fail(self):
        row = source_row()
        row["Acc latency"] = "1"
        with self.assertRaisesRegex(ValueError, "PNM total mismatch"):
            formula_ledger(row)
        data = metadata()
        data["array"]["width"] = 16
        with self.assertRaisesRegex(ValueError, "array width"):
            validate_metadata(source_row(), data)


class ProtocolTests(unittest.TestCase):
    def test_ordered_phases_decode_interleaved_flash_blocks(self):
        data = metadata()
        data["command_phases"] = [
            {"kernel": kernel, "mac_work": amount} for kernel, amount in
            (("q", 2), ("fused_kv", 1), ("qk", 1), ("sv_paired_kv_heads", 1),
             ("qk", 1), ("sv_paired_kv_heads", 1), ("wo", 1), ("fused_w1_w3", 2), ("w2", 1))]
        original = """AiM WR_BIAS 0 0xffffffff
AiM MAC_ABK 2 0xffffffff 31
AiM RD_MAC 0 0xffffffff
W MEM 15 0 40
R MEM 15 0 40
AiM SYNC
AiM WR_BIAS 0 0xffffffff
AiM MAC_ABK 2 0xffffffff 50
AiM RD_MAC 0 0xffffffff
"""
        interleaved = "".join(
            f"AiM WR_BIAS 0 0xffffffff\nAiM MAC_ABK 1 0xffffffff {row}\nAiM RD_MAC 0 0xffffffff\n"
            for row in (31, 50, 32, 51))
        with TemporaryDirectory() as temp:
            path = Path(temp, "flash.trace")
            path.write_text(TRACE.replace(original, interleaved))
            weights, owners, counts, regions = trace_work(path, data, 15)
        self.assertEqual(weights["MAC_ABK"]["attention.qk"], 2)
        self.assertEqual(weights["MAC_ABK"]["attention.sv"], 2)
        self.assertNotIn("aux.score_transfer", regions)
        self.assertEqual(counts["W_MEM"], 1)
        data["command_phases"][2]["mac_work"] += 1
        with self.assertRaisesRegex(ValueError, "reconcile"):
            phase_budgets(data)

    def test_observed_trace_work_partitions_auxiliary_without_codegen(self):
        with TemporaryDirectory() as temp:
            path = Path(temp, "trace.txt")
            path.write_text(TRACE)
            weights, owners, counts, regions = trace_work(path, metadata(), 15)
        self.assertEqual(weights["MAC_ABK"]["attention.q_projection"], 2)
        self.assertEqual(weights["W_MEM"]["aux.kv_update"], 1)
        self.assertEqual(weights["W_MEM"]["aux.score_transfer"], 1)
        self.assertEqual(weights["AF"]["aux.ffn_activation"], 1)
        self.assertNotIn("AF", regions["ffn.up_gate_projection"])
        self.assertEqual(counts["EWADD"], 2)
        self.assertEqual(sum(len(q) for q in owners.values()), 32)

    def test_incomplete_or_bad_sidecar_cannot_be_silently_normalized(self):
        with TemporaryDirectory() as temp:
            path = Path(temp, "trace.txt")
            path.write_text(TRACE.replace("AiM MAC_ABK 2 0xffffffff 10", "AiM MAC_ABK 3 0xffffffff 10"))
            with self.assertRaisesRegex(ValueError, "exceeds sidecar budget"):
                trace_work(path, metadata(), 15)
        data = metadata()
        data["pipeline_cycles"]["reduction"] += 1
        with self.assertRaisesRegex(ValueError, "reconcile"):
            phase_budgets(data)

    def test_issued_gap_includes_prerequisites_but_preserves_background_tail(self):
        events = [CommandEvent(1, "TMOD", (0, 0, -1, -1, -1, 0)),
                  CommandEvent(4, "MAC8", (0, 0, -1, -1, 10, 0)),
                  CommandEvent(6, "ACT8-1", (0, 0, -1, -1, 20, 0)),
                  CommandEvent(10, "MAC8", (0, 0, -1, -1, 20, 0)),
                  CommandEvent(12, "REFab", (0, 0, -1, -1, -1, 0))]
        owners = {"MAC_ABK": deque([("q", 10), ("kv", 20)])}
        result = issued_gap_estimate(events, owners, 20)
        self.assertEqual(dict(result), {"q": 3, "kv": 6, "aux.unattributed": 11})
        self.assertEqual(sum(result.values()), 20)
        with self.assertRaisesRegex(ValueError, "MAC rows"):
            issued_gap_estimate(events, {"MAC_ABK": deque([("q", 99), ("kv", 20)])}, 20)

    def test_gap_details_preserve_activation_wait_refresh_and_tail(self):
        def event(clock, command):
            return CommandEvent(clock, command, (0, 0, -1, -1, 10, 0))
        events = [event(1, "TMOD"), event(33, "ACT8-1"), event(34, "ACT8-2"),
                  event(82, "MAC8"), event(84, "MAC8"), event(96, "PREA"),
                  event(130, "REFab"), event(418, "ACT8-1"), event(419, "ACT8-2"),
                  event(467, "MAC8")]
        details = {}
        allocations = issued_gap_estimate(
            events, {"MAC_ABK": deque([("q", 10)] * 3)}, 500, details=details)
        self.assertEqual(dict(allocations), {"aux.unattributed": 68, "q": 432})
        for region, detail in details.items():
            self.assertEqual(sum(e["gap_cycles"] for e in detail["commands"]), allocations[region])
        commands = {e["command"]: e for e in details["q"]["commands"]}
        self.assertEqual(commands["MAC8"]["issued_event_count"], 3)
        self.assertEqual(commands["MAC8"]["gap_cycles"], 98)
        self.assertEqual(commands["MAC8"]["predecessor_gaps"], [
            {"previous_command": "ACT8-2", "gap_cycles": 48, "interval_count": 2},
            {"previous_command": "MAC8", "gap_cycles": 2, "interval_count": 1},
        ])
        background = {e["command"]: e for e in details["aux.unattributed"]["commands"]}
        self.assertEqual(background["REFab"]["gap_cycles"], 34)
        self.assertEqual(background["TAIL"]["gap_cycles"], 33)
        self.assertEqual(background["STARTUP"]["issued_event_count"], 0)
        # The detail representation must remain serializable in breakdown.json.
        self.assertEqual(json.loads(json.dumps(details)), details)


class ArtifactTests(unittest.TestCase):
    def create_inputs(self, temp):
        root = Path(temp)
        row = source_row()
        source = root / "source.csv"
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=row)
            writer.writeheader()
            writer.writerow(row)
        log = root / "trace.log"
        log.write_text("memory_system_cycles: 1000\nCH0_AiM_ISR_MAC_ABK_cycles: 500\n")
        timing = root / "timing.yaml"
        timing.write_text("impl: LPDDR4\ntiming:\n  tCK_ps: 625\n  nRAS: 68\n  nRC: 101\n  nBL: 4\n  nCCD: 2\n")
        return row, source, log, timing

    def test_report_exposes_issued_commands_for_every_pim_region(self):
        with TemporaryDirectory() as temp:
            row, source, log, timing = self.create_inputs(temp)
            trace, sidecar, issued = (Path(temp, name) for name in ("trace.txt", "trace.json", "issued.ch15"))
            trace.write_text(TRACE)
            sidecar.write_text(json.dumps(metadata()))
            # Independent issued events for every terminal in TRACE, including
            # a 49-cycle ACT8-1-to-first-MAC wait in the Q projection.
            schedule = [
                (1, "TMOD", -1), (33, "WRMAC8", -1),
                (37, "WRGB", -1), (41, "WRGB", -1),
                (42, "ACT8-1", 10), (43, "ACT8-2", 10),
                (91, "MAC8", 10), (93, "MAC8", 10), (94, "RDMAC8", -1),
            ]
            remaining = [
                ("WRMAC8", -1), ("MAC8", 20), ("RDMAC8", -1), ("WR", 30),
                ("WRMAC8", -1), ("MAC8", 31), ("MAC8", 31), ("RDMAC8", -1),
                ("WR", 40), ("RD", 40),
                ("WRMAC8", -1), ("MAC8", 50), ("MAC8", 50), ("RDMAC8", -1),
                ("WRMAC8", -1), ("MAC8", 60), ("RDMAC8", -1),
                ("WRMAC8", -1), ("MAC8", 70), ("MAC8", 70), ("RDMAC8", -1),
                ("AF8", -1), ("RDAF8", -1),
                ("WRMAC8", -1), ("MAC8", 80), ("RDMAC8", -1),
            ]
            schedule.extend((96 + 2 * i, command, address) for i, (command, address) in enumerate(remaining))
            issued.write_text("".join(f"{clock}, {command}, 15, 0, -1, -1, {address}, 0\n"
                                      for clock, command, address in schedule))
            report = analyze_case(row, log=log, timing=timing, source_csv=source,
                                  trace=trace, metadata_path=sidecar, command_trace=issued)
            run = write_report(report, Path(temp, "reports"))
            summary = (run / "summary.txt").read_text()
        detailed = [r for r in report["regions"] if r["pim_issued_gap_details"] is not None]
        self.assertEqual(sum(r["pim_issued_gap_details"]["total_gap_cycles"] for r in detailed), 1000)
        for region in detailed:
            self.assertIn(f"{region['display_name']} [{region['region_id']}]", summary)
            self.assertEqual(region["pim_issued_gap_details"]["total_gap_cycles"],
                             region["backends"]["pim"]["reference_equivalent_cycles"])
        q = next(r for r in detailed if r["region_id"] == "attention.q_projection")
        mac = next(e for e in q["pim_issued_gap_details"]["commands"] if e["command"] == "MAC8")
        self.assertEqual(mac["gap_cycles"], 50)
        self.assertIn("WRGB：写入输入 GB", summary)
        self.assertIn("ACT / PRE", summary)
        self.assertIn("MAC8 前的发射间隔分布", summary)
        self.assertIn("差额：0 cycles", summary)
        self.assertEqual(render_summary(report), summary)

    def test_missing_metadata_preserves_total_and_unavailable_status(self):
        with TemporaryDirectory() as temp:
            row, source, log, timing = self.create_inputs(temp)
            report = analyze_case(row, log=log, timing=timing, source_csv=source)
        regions = {r["region_id"]: r for r in report["regions"]}
        self.assertEqual(report["full_run_totals"]["pim_native_cycles"], 1000)
        self.assertIsNone(regions["attention.q_projection"]["backends"]["pim"]["duration_ns"])
        self.assertEqual(regions["aux.unattributed"]["backends"]["pim"]["duration_ns"], 625)
        self.assertEqual(regions["aux.attention_residual"]["backends"]["pim"]["coverage"], "unmodeled")
        self.assertIsNone(regions["attention.q_projection"]["backends"]["pim"]["completion_cycle"])

    def test_content_addressed_output_repeat_and_no_overwrite(self):
        with TemporaryDirectory() as temp:
            row, source, log, timing = self.create_inputs(temp)
            report = analyze_case(row, log=log, timing=timing, source_csv=source)
            run = write_report(report, Path(temp, "reports"))
            self.assertEqual(write_report(report, Path(temp, "reports")), run)
            self.assertEqual(json.loads((run / "breakdown.json").read_text())["run_id"], report["run_id"])
            report["diagnostics"].append("different")
            with self.assertRaisesRegex(ValueError, "differs"):
                write_report(report, Path(temp, "reports"))

    def test_tp1_has_no_helper_rank_and_no_collective(self):
        with TemporaryDirectory() as temp:
            row, source, log, timing = self.create_inputs(temp)
            row.update({"Tensor parallelism": "1", "TP collective CXL latency": "0",
                        "CXL latency": row["PP handoff CXL latency"],
                        "Full softmax latency": "0.048984", "Exposed softmax latency": "0.048984",
                        "EWMUL activation latency": "0.000224", "Q reduction latency": "0",
                        "KV reduction latency": "0.000060", "SV reduction latency": "0.000224",
                        "Acc latency": "0.051481", "Main Acc latency": "0.051481",
                        "Helper Acc latency": "0", "Helper PIM latency": "0",
                        "TransformerBlock latency": str(0.000625 + 0.051481 + float(row["PP handoff CXL latency"]))})
            report = analyze_case(row, log=log, timing=timing, source_csv=source)
        terms = next(r for r in report["regions"] if r["region_id"] == "attention.o_projection")
        self.assertEqual(terms["backends"]["cxl"]["duration_ns"], 0)

    def test_flash_attention_preserves_formulas_without_guessing_interleaved_owners(self):
        with TemporaryDirectory() as temp:
            row, source, log, timing = self.create_inputs(temp)
            row.update({"Flash attention": "True", "Flash attention block size": "1024",
                        "Acc latency": "0.02599875", "Main Acc latency": "0.02599875",
                        "Helper Acc latency": "0.02599875",
                        "TransformerBlock latency": str(0.000625 + 0.02599875 + 0.0073074993133543)})
            report = analyze_case(row, log=log, timing=timing, source_csv=source)
        self.assertEqual(report["estimator"]["id"], "unavailable")
        terms = next(r for r in report["regions"] if r["region_id"] == "aux.flash_merge")
        self.assertEqual(terms["backends"]["pnm"]["duration_ns"], 3.75)

    def test_ambiguous_csv_selection_fails(self):
        with TemporaryDirectory() as temp:
            row, source, log, timing = self.create_inputs(temp)
            with source.open("a", newline="") as stream:
                csv.DictWriter(stream, fieldnames=row).writerow(row)
            with self.assertRaisesRegex(ValueError, "found 2"):
                read_case(source, model="Llama2-70B", tp=2, batch=1, context_window=32768)


if __name__ == "__main__":
    unittest.main()

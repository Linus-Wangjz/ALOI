"""Numerical and dependency checks for the analytical double-GB estimator."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_legacy_double_buffering import (  # noqa: E402
    Wave, credited_wrgb_overlap, decode_waves, digest, issued_wrgb_gaps,
    model_report, overlap_rows, readiness, resolve_artifact,
)


def wave(index, kernel="q", phase=0, bursts=64, mac=518, barrier=False):
    return Wave(index, phase, kernel, index + 1, bursts, mac, barrier)


class DoubleBufferModelTests(unittest.TestCase):
    def test_full_overlap_cannot_subtract_more_than_WRGB_owned_gap(self):
        # First WRGB follows CAS by one clock, not the nominal eight clocks.
        gaps = [1] + [8] * 63
        self.assertEqual(credited_wrgb_overlap(512, 8, gaps), 505)

    def test_partial_overlap_cannot_borrow_later_burst_budget(self):
        # Only 44 nominal cycles overlap: first five full slots + four clocks.
        self.assertEqual(credited_wrgb_overlap(44, 8, [1] + [8] * 63), 37)
        # A long WRGB issue gap is not entirely wire time.
        self.assertEqual(credited_wrgb_overlap(16, 8, [32, 8]), 16)

    def test_raw_WRGB_gaps_exclude_CAS_setup(self):
        trace = """10, CASWRGB, 15, 0
11, WRGB, 15, 0
19, WRGB, 15, 0
21, MAC8, 15, 0
30, WRGB, 15, 0
38, WRGB, 15, 0
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "issued.ch15"
            path.write_text(trace)
            groups = issued_wrgb_gaps([wave(0, bursts=2), wave(1, bursts=2)], path, 15)
        self.assertEqual(groups, [[1, 8], [9, 8]])

    def test_KV_negative_seven_cycle_regression(self):
        def region(region_id, name, pim, wrgb, cas):
            return {"region_id": region_id, "display_name": name,
                    "legacy_ledger_duration_ns": pim * .625,
                    "backends": {"pim": {"reference_equivalent_cycles": pim},
                                 "pnm": {"reference_equivalent_cycles": 0},
                                 "cxl": {"reference_equivalent_cycles": 0}},
                    "pim_issued_gap_details": {"commands": [
                        {"command": "WRGB", "gap_cycles": wrgb},
                        {"command": "CASWRGB", "gap_cycles": cas}]}}
        source = {"case": {"nBL": 8, "nCCD": 2, "batch_size": 4},
                  "reference_clock_period_ps": 625,
                  "regions": [region("attention.q_projection", "Q", 2000, 512, 0),
                              region("attention.kv_projection", "KV", 4750, 1017, 32)],
                  "full_run_totals": {"pim_native_cycles": 6750, "pim_latency_ns": 4218.75,
                                      "block_latency_ns": 4218.75, "reference_equivalent_cycles": 6750,
                                      "pnm_exposed_latency_ns": 0, "cxl_latency_ns": 0}}
        waves = [wave(0), wave(1, "fused_kv", 1), wave(2, "fused_kv", 1)]
        result = model_report(source, waves, False, [[8] * 64, [8] * 64, [1] + [8] * 63])
        kv = result["kernels"][1]
        self.assertEqual(kv["nominal_hidden_wrgb_cycles"], 1024)
        self.assertEqual(kv["hidden_wrgb_cycles"], 1017)
        self.assertEqual(kv["exposed_wrgb_cycles"], 0)
        self.assertEqual(kv["caswrgb_gap_cycles_unchanged"], 32)
        self.assertEqual(kv["new_pim_cycles_estimate"], 3733)
        self.assertEqual(result["totals"]["new_pim_cycles_analytical"], 5733)

    def test_q_projection_pays_one_512_cycle_load(self):
        rows = overlap_rows([wave(i) for i in range(8)], 8, 2)
        self.assertEqual(sum(r["hidden_wrgb_cycles"] for r in rows), 3584)
        self.assertEqual(sum(r["exposed_wrgb_cycles"] for r in rows), 512)

    def test_bandwidth_limit_does_not_become_free_after_startup(self):
        rows = overlap_rows([wave(i, bursts=10, mac=10) for i in range(3)], 10, 2)
        self.assertEqual([r["exposed_wrgb_cycles"] for r in rows], [100, 80, 80])
        self.assertEqual(sum(r["hidden_wrgb_cycles"] for r in rows), 40)

    def test_softmax_dependency_blocks_first_sv_wave_only(self):
        waves = [wave(0, "qk", 0, 16, 134),
                 wave(1, "sv_paired_kv_heads", 1, 32, 134),
                 wave(2, "sv_paired_kv_heads", 2, 32, 134)]
        rows = overlap_rows(waves, 8, 2)
        self.assertEqual([r["hidden_wrgb_cycles"] for r in rows], [0, 0, 256])
        ideal = overlap_rows(waves, 8, 2, ideal=True)
        self.assertEqual([r["hidden_wrgb_cycles"] for r in ideal], [0, 256, 256])

    def test_short_last_sv_chunk_cannot_hide_all_of_next_query(self):
        waves = [wave(0, "sv_paired_kv_heads", 0, 4, 22),
                 wave(1, "qk", 1, 16, 134)]
        row = overlap_rows(waves, 8, 2)[1]
        self.assertEqual(row["hidden_wrgb_cycles"], 44)
        self.assertEqual(row["exposed_wrgb_cycles"], 84)

    def test_independent_q_kv_inputs_but_memory_fence_blocks_prefetch(self):
        previous, current = wave(0), wave(1, "fused_kv", 1)
        self.assertTrue(readiness(previous, current)[0])
        current.barrier_before = True
        self.assertFalse(readiness(previous, current)[0])

    def test_output_dependencies_and_same_input_tiles(self):
        self.assertFalse(readiness(wave(0, "fused_w1_w3"), wave(1, "w2", 1))[0])
        self.assertFalse(readiness(wave(0, "sv_paired_kv_heads"), wave(1, "wo", 1))[0])
        self.assertTrue(readiness(wave(0, "fused_w1_w3"), wave(1, "fused_w1_w3", 1))[0])

    def test_trace_decoder_separates_gb_chunks_from_accumulator_phases(self):
        trace = """AiM WR_BIAS 0 0xffffffff
AiM WR_BIAS 0 0xffffffff
AiM WR_GB 2 0 0xffffffff
AiM MAC_ABK 3 0xffffffff 10
AiM WR_GB 1 0 0xffffffff
AiM MAC_ABK 2 0xffffffff 11
AiM RD_MAC 0 0xffffffff
AiM RD_MAC 0 0xffffffff
AiM SYNC
AiM WR_BIAS 0 0xffffffff
AiM WR_GB 1 0 0xffffffff
AiM MAC_ABK 4 0xffffffff 20
AiM RD_MAC 0 0xffffffff
AiM EOC
"""
        metadata = {"command_phases": [{"kernel": "q", "mac_work": 5},
                                        {"kernel": "fused_kv", "mac_work": 4}]}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.txt"
            path.write_text(trace)
            waves, counts = decode_waves(path, metadata, 15)
        self.assertEqual([(w.phase, w.bursts, w.mac_commands) for w in waves],
                         [(0, 2, 3), (0, 1, 2), (1, 1, 4)])
        self.assertEqual([w.barrier_before for w in waves], [True, False, True])
        self.assertEqual(counts["q"]["WR_BIAS"], 2)
        self.assertEqual(counts["q"]["RD_MAC"], 2)

    def test_relocated_source_requires_exact_hash(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "missing" / "trace.txt"
            copy = root / "other" / "trace.txt"
            copy.parent.mkdir()
            copy.write_text("original trace")
            record = {"path": str(original), "sha256": digest(copy)}
            self.assertEqual(resolve_artifact(record, root), copy)
            copy.write_text("different trace")
            with self.assertRaises(ValueError):
                resolve_artifact(record, root)


if __name__ == "__main__":
    unittest.main()

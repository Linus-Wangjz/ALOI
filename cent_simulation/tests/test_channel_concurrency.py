"""Exercise actual AiM admission, timing, and barriers with a small trace."""

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace

import yaml

CENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CENT))
from cycle_breakdown import parse_command_trace, read_result_stats  # noqa: E402

REPO = CENT.parent
BINARY = REPO / "aim_simulator/build/ramulator2"


class FP8OptimizationTests(unittest.TestCase):
    def test_fp8_startup_uses_physical_word_width_and_retains_full_work(self):
        import run_sim
        common = dict(model="Llama2-70B", precision="fp8", num_channels=32, num_banks=8,
                      kv_head_tp=True, systolic_pim=True, systolic_dim=4,
                      max_seq_len=32768, ewmul_pnm=True, flash_attention=True,
                      flash_attention_block_size=1024)
        baseline = run_sim.calculate_acc_latency(SimpleNamespace(**common, pipelined_softmax=False), 30976, tp=2)
        pipelined = run_sim.calculate_acc_latency(SimpleNamespace(**common, pipelined_softmax=True), 30976, tp=2)
        self.assertEqual(run_sim.softmax_pipeline_startup_tokens(SimpleNamespace(**common)), 8192)
        self.assertAlmostEqual(baseline["Softmax_latency"], 0.024492)
        self.assertAlmostEqual(pipelined["Softmax_latency"], baseline["Softmax_latency"] * 8192 / 30976)
        self.assertEqual(baseline["SVReduction_latency"], pipelined["SVReduction_latency"])


@unittest.skipUnless(BINARY.exists(), "build AiM to run channel concurrency integration tests")
class ChannelConcurrencyTests(unittest.TestCase):
    def simulate(self, root, concurrent, trace_text=None):
        trace = root / "requests.trace"
        commands = []
        for direction in ("W", "R"):
            commands.extend(f"{direction} MEM {channel} 0 0\n"
                            for _ in range(64) for channel in range(32))
            commands.append("AiM SYNC\n")
        commands.extend(["AiM WR_GB 64 0 0xffffffff\n", "AiM SYNC\n", "AiM EOC\n"])
        trace.write_text(trace_text if trace_text is not None else "".join(commands))
        config = yaml.safe_load((REPO / "aim_simulator/test/example_LPDDR4.yaml").read_text())
        config["MemorySystem"]["DRAM"]["timing"]["nCCD"] = 2
        config["Frontend"]["issue_width"] = 32 if concurrent else 1
        config["MemorySystem"]["host_issue_width"] = 32 if concurrent else 1
        config["MemorySystem"]["Controller"]["plugins"] = [
            {"ControllerPlugin": {"impl": "TraceRecorder", "path": str(root / "issued")}}]
        path = root / "config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        result = subprocess.run([str(BINARY), "-f", str(path), "-t", str(trace)],
                                capture_output=True, text=True, timeout=30, check=True)
        log = root / "result.log"
        log.write_text(result.stdout)
        return read_result_stats(log), [parse_command_trace(root / f"issued.ch{ch}") for ch in range(32)]

    def test_parallel_reads_writes_multicast_gb_and_sync(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            serial, serial_events = self.simulate(root, False)
            parallel, parallel_events = self.simulate(root, True)
        self.assertLess(parallel["memory_system_cycles"], serial["memory_system_cycles"])
        self.assertEqual(parallel["max_channel_requests_per_cycle"], 1)
        for events in (serial_events, parallel_events):
            for channel in events:
                reads = [e.clock for e in channel if e.command == "RD"]
                writes = [e.clock for e in channel if e.command == "WR"]
                gb = [e.clock for e in channel if e.command == "WRGB"]
                self.assertEqual((len(reads), len(writes), len(gb)), (64, 64, 64))
                # SYNC drains real completion latency, not just issue order.
                self.assertGreaterEqual(min(reads), max(writes) + 15)
                self.assertGreaterEqual(min(gb), max(reads) + 24)
                for clocks in (reads, writes, gb):
                    self.assertTrue(all(b - a >= 4 for a, b in zip(clocks, clocks[1:])))
        for command in ("RD", "WR", "WRGB"):
            simultaneous = set.intersection(*[
                {e.clock for e in channel if e.command == command} for channel in parallel_events])
            self.assertTrue(simultaneous, f"{command} must actually issue concurrently across channels")

    def test_different_channels_read_and_write_streams_overlap(self):
        initial = "".join(f"W MEM {channel} 0 0\n" for channel in range(32)) + "AiM SYNC\n"
        mixed = "".join(f"{'R' if channel < 16 else 'W'} MEM {channel} 0 0\n"
                        for _ in range(64) for channel in range(32))
        with TemporaryDirectory() as temp:
            stats, events = self.simulate(Path(temp), True, initial + mixed + "AiM SYNC\nAiM EOC\n")
        self.assertEqual(stats["max_channel_requests_per_cycle"], 1)
        for channel, issued in enumerate(events):
            self.assertEqual(sum(e.command == "RD" for e in issued), 64 if channel < 16 else 0)
            self.assertEqual(sum(e.command == "WR" for e in issued), 1 if channel < 16 else 65)
        reads = {e.clock for e in events[0] if e.command == "RD"}
        writes = {e.clock for e in events[16] if e.command == "WR"}
        # Independent timing may put RD and WR on different clock phases;
        # both streams must progress before either one's interval ends.
        self.assertLess(max(min(reads), min(writes)), min(max(reads), max(writes)))


if __name__ == "__main__":
    unittest.main()

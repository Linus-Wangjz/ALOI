"""Exercise the real asynchronous controller, not a substitute timing formula."""
from collections import Counter
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
import yaml

CENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CENT))
from cycle_breakdown import read_result_stats
from pim_double_buffering import read_observer, validate_events, account_events

BINARY = CENT.parent / "aim_simulator/build/ramulator2"
BASE = CENT.parent / "aim_simulator/test/generated/example_LPDDR4X_nCCD2.yaml"


def workload(generations=5, mask="0xffffffff"):
    lines = [f"AiM WR_BIAS 0 {mask}"]
    for g in range(generations):
        lines += [f"AiM WR_GB 64 0 {mask}", f"AiM MAC_ABK 3 {mask} {2*g}",
                  f"AiM MAC_ABK 64 {mask} {2*g}", f"AiM MAC_ABK 64 {mask} {2*g+1}",
                  f"AiM MAC_ABK 3 {mask} {2*g+1}"]
    return "\n".join(lines + [f"AiM RD_MAC 0 {mask}", "AiM SYNC", "AiM EOC", ""])


@unittest.skipUnless(BINARY.exists(), "Build AiM first")
class DoubleBufferingTests(unittest.TestCase):
    def simulate(self, root, *, slots=2, serialized=False, nccd=2, refresh=None,
                 depth=8, trace=None, width=32, impl="AiMAsync", check=True, observe=True):
        root.mkdir(parents=True, exist_ok=True)
        config = yaml.safe_load(BASE.read_text())
        config["Frontend"]["issue_width"] = width
        memory = config["MemorySystem"]
        memory["host_issue_width"] = width
        memory["DRAM"]["timing"]["nCCD"] = nccd
        if refresh: memory["DRAM"]["timing"]["nREFI"] = refresh
        controller = memory["Controller"]
        controller.update(impl=impl, gb_buffer_count=slots, serialize_loads=serialized,
                          descriptor_depth=depth, observer_path=str(root / "tasks") if observe else "")
        controller["plugins"] = [{"ControllerPlugin": {"impl": "TraceRecorder", "path": str(root / "cmd")}},
                                 {"ControllerPlugin": {"impl": "DRAMTimingExporter", "path": str(root / "timing.yaml")}}]
        if not observe: controller["plugins"] = []
        (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        (root / "trace.txt").write_text(trace or workload())
        result = subprocess.run([str(BINARY), "-f", str(root / "config.yaml"), "-t", str(root / "trace.txt")],
                                capture_output=True, text=True, timeout=30)
        if not check: return result
        self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-2000:])
        (root / "result.log").write_text(result.stdout)
        events = [list(read_observer(root / f"tasks.ch{ch}.csv")) for ch in range(32)] if impl == "AiMAsync" and observe else []
        return read_result_stats(root / "result.log"), events

    def test_three_ablations_have_identical_work_but_only_overlap_is_faster(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            single, ev1 = self.simulate(root / "single", slots=1)
            serial, ev2 = self.simulate(root / "serial", serialized=True)
            overlap, ev3 = self.simulate(root / "overlap")
            counts = []
            for events in (ev1, ev2, ev3):
                summary = [validate_events(ch, buffers=1 if events is ev1 else 2) for ch in events]
                counts.append(summary[0]["issued_counts"])
            for command in ("MAC8", "WRGB", "WRMAC8", "RDMAC8"):
                self.assertEqual(counts[0][command], counts[1][command])
                self.assertEqual(counts[1][command], counts[2][command])
            self.assertEqual(single["memory_system_cycles"], serial["memory_system_cycles"])
            self.assertLess(overlap["memory_system_cycles"], serial["memory_system_cycles"])
            self.assertGreater(overlap["CH0_PU_GB_same_cycle_progress"], 0)
            allocations, details, commands, _ = account_events(ev3[0], {}, overlap["memory_system_cycles"])
            self.assertEqual(sum(allocations.values()), overlap["memory_system_cycles"])
            self.assertEqual(sum(sum(x.values()) for x in details.values()), sum(allocations.values()))

    def test_refresh_nccd6_backpressure_and_fences(self):
        with TemporaryDirectory() as temp:
            stats, channels = self.simulate(Path(temp), nccd=6, depth=3, refresh=1800)
            timing = yaml.safe_load((Path(temp) / "timing.yaml").read_text())["timing"]
            for channel in channels:
                last_act = last_pre = last_ref = -10**9
                for e in channel:
                    if e["event"] != "issue": continue
                    clock, cmd = e["cycle"], e["command"]
                    if cmd == "REFab": last_ref = clock
                    elif cmd == "PREA": last_pre = clock
                    elif cmd == "ACT8-1":
                        self.assertGreaterEqual(clock - last_pre, timing["nRPab"])
                        self.assertGreaterEqual(clock - last_ref, timing["nRFCab"])
                        last_act = clock
                    elif cmd == "MAC8":
                        self.assertGreaterEqual(clock - last_act, timing["nRCDRDMAC"])
                validated = validate_events(channel, nccd=6)
                self.assertGreater(validated["issued_counts"].get("REFab", 0), 0)
                self.assertEqual(validated["gb_generations"], 5)
            self.assertLessEqual(stats["CH0_max_descriptors"], 3)
            self.assertEqual(stats["max_channel_requests_per_cycle"], 1)
            self.assertLess(channels[0][-1]["cycle"], stats["memory_system_cycles"] + 1)

    def test_independent_masks_and_retry_bindings(self):
        text = workload(3, "0x80000000").replace("AiM SYNC\nAiM EOC\n", "")
        text += workload(3, "0x40000000")
        with TemporaryDirectory() as temp:
            stats, channels = self.simulate(Path(temp), trace=text, depth=2)
            for channel in channels[:2]:
                self.assertEqual(validate_events(channel)["gb_generations"], 3)
            for channel in channels[2:]:
                self.assertEqual(validate_events(channel)["gb_generations"], 0)

    def test_phase_checkpoint_retires_all_async_descriptors(self):
        import run_sim
        attention = workload(2).replace("AiM SYNC\nAiM EOC\n", "")
        marker_id = len(attention.splitlines())
        trace = attention + "# PHASE attention_end\n" + workload(2)
        for width in (1, 32):
            with self.subTest(width=width), TemporaryDirectory() as temp:
                root = Path(temp)
                stats, channels = self.simulate(root, trace=trace, width=width)
                checkpoint = run_sim.read_attention_end_cycles(root / "result.log")
                self.assertLess(checkpoint, stats["memory_system_cycles"])
                for channel in channels:
                    self.assertEqual(validate_events(channel)["gb_generations"], 4)
                    before = [e["cycle"] for e in channel if e["event"] == "complete"
                              and 0 <= e["host_id"] < marker_id]
                    after = [e["cycle"] for e in channel if e["event"] == "issue"
                             and e["host_id"] > marker_id]
                    self.assertTrue(before and after)
                    self.assertLessEqual(max(before), checkpoint)
                    self.assertGreaterEqual(min(after), checkpoint)

    def test_register_io_and_memory_preserve_work_and_burst_spacing(self):
        text = "".join(f"{direction} MEM {ch} 0 0\n" for direction in ("W", "R")
                       for _ in range(8) for ch in range(32))
        text = text.replace("R MEM 0 0 0\n", "AiM SYNC\nR MEM 0 0 0\n", 1) + "AiM SYNC\n" + workload(2)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            legacy, _ = self.simulate(root / "legacy", trace=text, impl="AiM")
            stats, channels = self.simulate(root / "async", trace=text)
            for command in ("RD", "WR", "WRGB", "MAC8", "WRMAC8", "RDMAC8"):
                self.assertEqual(legacy[f"CH0_num_{command}_commands"], stats[f"CH0_num_{command}_commands"])
            for channel in channels:
                validate_events(channel)
                for command in ("RD", "WR", "RDMAC8", "WRMAC8"):
                    clocks = [e["cycle"] for e in channel if e["event"] == "issue" and e["command"] == command]
                    self.assertTrue(all(b-a >= 8 for a, b in zip(clocks, clocks[1:])))

    def test_reject_missing_producer_and_partial_epoch(self):
        for text in ("AiM MAC_ABK 64 0xffffffff 0\nAiM EOC\n",
                     "AiM WR_GB 65 0 0xffffffff\nAiM EOC\n",
                     "AiM COPY_BKGB 64 0xffffffff 0 0\nAiM EOC\n"):
            with TemporaryDirectory() as temp:
                result = self.simulate(Path(temp), trace=text, check=False)
                self.assertNotEqual(result.returncode, 0)

    def test_short_fresh_payloads_and_last_step_completion(self):
        trace = ("AiM WR_GB 16 0 0xffffffff\nAiM MAC_ABK 16 0xffffffff 0\n"
                 "AiM WR_GB 4 0 0xffffffff\nAiM MAC_ABK 4 0xffffffff 1\n"
                 "AiM WR_GB 1 0 0xffffffff\nAiM MAC_ABK 1 0xffffffff 2\n"
                 "AiM RD_MAC 0 0xffffffff\nAiM EOC\n")
        with TemporaryDirectory() as temp:
            _, channels = self.simulate(Path(temp), trace=trace)
        for events in channels:
            validate_events(events)
            for task in {e["task_id"] for e in events if e["command"] == "MAC8"}:
                steps = [e for e in events if e["event"] == "issue" and e["command"] == "MAC8" and e["task_id"] == task]
                complete = next(e for e in events if e["event"] == "complete" and e["task_id"] == task)
                self.assertEqual(steps[-1]["cycle"] - steps[0]["cycle"], (len(steps)-1)*2)
                self.assertEqual(complete["cycle"], steps[-1]["cycle"] + 1)

    def test_cli_profile_and_multievent_clock_export(self):
        import run_sim
        with TemporaryDirectory() as temp:
            root = Path(temp)
            run_sim.write_ramulator_config(BASE, root / "config.yaml", root / "timing.yaml",
                                            root / "issued", True, double_buffering=True)
            controller = yaml.safe_load((root / "config.yaml").read_text())["MemorySystem"]["Controller"]
            self.assertEqual(controller["impl"], "AiMAsync")
            self.assertEqual(controller["gb_buffer_count"], 2)
            self.assertFalse(controller["serialize_loads"])
            _, channels = self.simulate(root)
            issued = [(e["cycle"], e["command"]) for e in channels[0] if e["event"] == "issue"]
            from cycle_breakdown import parse_command_trace
            exported = [(e.clock, e.command) for e in parse_command_trace(root / "cmd.ch0")]
            self.assertEqual(exported, issued)
            self.assertLess(len({c for c, _ in issued}), len(issued))

    def test_observer_and_plugins_do_not_change_timing(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            profiled, _ = self.simulate(root / "on")
            plain, _ = self.simulate(root / "off", observe=False)
        for key, value in profiled.items():
            if key == "memory_system_cycles" or "num_" in key:
                self.assertEqual(value, plain[key])

    def test_legacy_estimator_rejects_async_csv_without_config(self):
        from cycle_breakdown import analyze_case
        with self.assertRaisesRegex(ValueError, "completion-observer"):
            analyze_case({"PIM double buffering": True}, log=None, timing=None, source_csv=None)

    def test_global_phase_accounting_includes_other_channel_waits(self):
        from pim_double_buffering import account_global_phases
        owners = {0: "q", 1: "q", 2: "kv", 3: "kv", 4: "sv"}
        completions = {0: 5, 1: 8, 2: 12, 3: 15, 4: 20}
        timeline = [(1, 0, "MAC8"), (4, -1, "idle_or_tail"), (10, 2, "WRGB"),
                    (13, 4, "MAC8"), (20, -1, "idle_or_tail")]
        events = [dict(cycle=c, host_id=h, event="account", command=state) for c, h, state in timeline]
        allocations, details, _, _ = account_global_phases(events, owners, completions, 20)
        self.assertEqual(dict(allocations), {"q": 7, "kv": 7, "sv": 5, "aux.unattributed": 1})
        self.assertEqual(details["q"]["other_channel_or_host_wait"], 4)
        self.assertEqual(sum(sum(v.values()) for v in details.values()), 20)

    def test_af_and_readback_wait_for_accumulator_completion(self):
        trace = workload(2).replace("AiM RD_MAC 0 0xffffffff", "AiM AF 0xffffffff\nAiM RD_AF 0 0xffffffff")
        with TemporaryDirectory() as temp:
            _, channels = self.simulate(Path(temp), trace=trace)
        for events in channels:
            validate_events(events)
            mac = [e["cycle"] for e in events if e["event"] == "issue" and e["command"] == "MAC8"]
            af = [e["cycle"] for e in events if e["event"] == "issue" and e["command"] == "AF8"]
            read = [e["cycle"] for e in events if e["event"] == "issue" and e["command"] == "RDAF8"]
            self.assertEqual(len(af), 1)
            self.assertGreater(af[0], max(mac))
            self.assertGreater(read[0], af[0])

    def test_two_physical_payloads_numerical_operand_ownership(self):
        # Observer-driven tiny functional check with exactly 2 x 2048 bytes of GB
        # storage. This validates lifetime/selection, not bit-accurate FP8/SA math.
        text = workload(5)
        lines = [line.split() for line in text.splitlines()]
        expected_epoch, latest = {}, -1
        for host, fields in enumerate(lines):
            if fields[1] == "WR_GB": latest += 1
            if fields[1] == "MAC_ABK": expected_epoch[host] = latest
        with TemporaryDirectory() as temp:
            _, channels = self.simulate(Path(temp), trace=text, depth=4)
            events = channels[0]
        slots = [bytearray(2048), bytearray(2048)]
        pending, result, expected = [], [0] * 8, [0] * 8
        def payload(epoch, index): return (epoch * 17 + index * 3) % 127
        for e in events:
            now = e["cycle"]
            for due, slot, offset, data in list(pending):
                if due <= now:
                    slots[slot][offset:offset+32] = data
                    pending.remove((due, slot, offset, data))
            if e["event"] != "issue": continue
            if e["command"] == "WRGB":
                offset = e["step"] * 32
                pending.append((now + 9, e["slot"], offset,
                                bytes(payload(e["epoch"], offset+i) for i in range(32))))
            if e["command"] == "MAC8":
                self.assertEqual(e["epoch"], expected_epoch[e["host_id"]])
                row = int(lines[e["host_id"]][4])
                offset = e["step"] * 32
                for bank in range(8):
                    weights = [(row + bank + i) % 11 - 5 for i in range(32)]
                    result[bank] += sum(x*w for x, w in zip(slots[e["slot"]][offset:offset+32], weights))
                    expected[bank] += sum(payload(expected_epoch[e["host_id"]], offset+i)*weights[i] for i in range(32))
        self.assertEqual(result, expected)
        self.assertNotEqual(sum(result), 0)


if __name__ == "__main__": unittest.main()

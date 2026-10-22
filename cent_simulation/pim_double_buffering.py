"""Validate asynchronous PU/GB lifecycle events and account concurrent timelines.

The observer's `account` events attribute exclusive wall time to the oldest
unfinished host descriptor. Counts of issued commands remain separate: a PU
step and an external burst can occur in the same clock. No FIFO issued-gap
alignment is valid for this execution profile.
"""
from collections import Counter, defaultdict
import csv
from pathlib import Path


def read_observer(path):
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            for key in ("cycle", "host_id", "task_id", "epoch", "slot", "step"):
                row[key] = int(row[key])
            yield row


def validate_events(events, *, nccd=2, nbl=8, load_latency=9, buffers=2, host_completions=None):
    """Independent scoreboard: admission binding, completion, reuse and resources."""
    accepted, complete, ready, slots, readers = {}, set(), {}, {}, defaultdict(set)
    latest = -1
    last_completion = 0
    ca = set()
    last_mac = last_gb = -10**9
    overlap, mac_clocks, gb_clocks = set(), set(), set()
    counts = Counter()
    for e in events:
        kind, task, clock, epoch, slot = (e[k] for k in ("event", "task_id", "cycle", "epoch", "slot"))
        if kind == "accept":
            if task in accepted:
                raise ValueError("Duplicate descriptor admission")
            accepted[task] = e
            if epoch >= 0:
                if epoch > latest:
                    if epoch != latest + 1 or slot != epoch % buffers:
                        raise ValueError("Invalid per-channel GB generation")
                    latest = epoch
                else:
                    if epoch != latest:
                        raise ValueError("MAC bound to a mutable/stale generation")
                    readers[epoch].add(task)
        elif kind == "load_start":
            previous = slots.get(slot)
            if previous is not None and (readers[previous] - complete or previous not in ready):
                raise ValueError("Slot overwritten with pending readers/load")
            slots[slot] = epoch
        elif kind == "load_ready":
            if clock < ready.get(("last_burst", epoch), clock) + load_latency:
                raise ValueError("GB ready before last burst completion")
            ready[epoch] = clock
        elif kind == "issue":
            command, domain = e["command"], e["domain"]
            counts[command] += 1
            if domain != "pu_internal":
                if clock in ca:
                    raise ValueError("Two external CA commands in one clock")
                ca.add(clock)
            if command == "WRGB":
                if clock - last_gb < nbl:
                    raise ValueError("WRGB violates external burst interval")
                last_gb = clock
                ready[("last_burst", epoch)] = clock
                gb_clocks.add(clock)
            if command == "MAC8":
                if epoch not in ready or ready[epoch] > clock or slots.get(slot) != epoch:
                    raise ValueError("MAC read an unready/overwritten slot")
                if clock - last_mac < nccd:
                    raise ValueError("MAC violates nCCD")
                last_mac = clock
                mac_clocks.add(clock)
        elif kind == "complete":
            last_completion = max(last_completion, clock)
            if host_completions is not None:
                host_completions[e["host_id"]] = max(host_completions.get(e["host_id"], 0), clock)
            if task not in accepted or task in complete:
                raise ValueError("Duplicate/unadmitted completion")
            if task in readers[epoch] and clock <= last_mac and accepted[task]["epoch"] == epoch:
                raise ValueError("MAC completes before its last step")
            complete.add(task)
    if set(accepted) != complete:
        raise ValueError("Undrained descriptors at EOC")
    overlap = mac_clocks & gb_clocks
    return {"issued_counts": dict(counts), "simultaneous_pu_gb_cycles": len(overlap),
            "descriptors": len(accepted), "gb_generations": latest + 1, "last_completion_cycle": last_completion}


def account_events(events, host_regions, total_cycles):
    allocations, details = Counter(), defaultdict(Counter)
    commands = defaultdict(Counter)
    spans = defaultdict(lambda: {"accept": None, "first_issue": None, "last_issue": None, "complete": None})
    current = (1, "aux.unattributed", "startup")
    for e in events:
        region = host_regions.get(e["host_id"], "aux.unattributed")
        clock, kind = e["cycle"], e["event"]
        if kind == "account":
            start, prior, state = current
            if clock < start:
                raise ValueError("Observer clock regressed")
            allocations[prior] += clock - start
            details[prior][state] += clock - start
            current = (clock, region, e["command"])
        if kind == "issue":
            commands[region][e["command"]] += 1
        key = {"accept": "accept", "issue": "first_issue", "complete": "complete"}.get(kind)
        if key:
            s = spans[region]
            if key == "complete": s[key] = max(s[key] or clock, clock)
            else: s[key] = min(s[key] if s[key] is not None else clock, clock)
            if kind == "issue": s["last_issue"] = max(s["last_issue"] or clock, clock)
    start, region, state = current
    # Native memory_system_cycles includes the final host callback clock.
    if start > total_cycles + 1:
        raise ValueError("Observer exceeds full-run cycle count")
    allocations[region] += total_cycles + 1 - start
    details[region][state] += total_cycles + 1 - start
    if sum(allocations.values()) != total_cycles:
        raise ValueError("Async wall-clock accounting does not reconcile")
    return allocations, details, commands, spans


def phase_completion_intervals(host_regions, host_completions, total_cycles):
    """Global in-order phase retirement, including waits on other channels.

    Consecutive host descriptors with the same region form a phase. Its end is
    the maximum completion of its descriptors on ALL participating channels,
    clamped by earlier phase ends. Unobserved/no-op phases receive no cycles.
    This is an explicit wall-time allocation policy, not critical-path causality.
    """
    intervals = []
    previous_region, end, start = None, 1, 1
    for host, region in host_regions.items():
        if region != previous_region:
            if previous_region is not None:
                intervals.append((start, end, previous_region))
                start = end
            previous_region = region
        end = max(end, host_completions.get(host, end))
    if previous_region is not None: intervals.append((start, end, previous_region))
    if end > total_cycles + 1:
        raise ValueError("Phase completion exceeds simulation end")
    intervals.append((end, total_cycles + 1, "aux.unattributed"))
    return intervals


def account_global_phases(events, host_regions, host_completions, total_cycles):
    """Global phase wall intervals with reference-channel scheduler-state detail."""
    _, _, commands, spans = account_events(events, host_regions, total_cycles)
    phases = phase_completion_intervals(host_regions, host_completions, total_cycles)
    states = []
    previous = (1, "aux.unattributed", "startup")
    for e in events:
        if e["event"] != "account": continue
        start, region, state = previous
        if e["cycle"] > start: states.append((start, e["cycle"], region, state))
        previous = (e["cycle"], host_regions.get(e["host_id"], "aux.unattributed"), e["command"])
    start, region, state = previous
    states.append((start, total_cycles+1, region, state))
    allocations, details = Counter(), defaultdict(Counter)
    i = 0
    for first, end, region in phases:
        allocations[region] += end - first
        while i < len(states) and states[i][1] <= first: i += 1
        j = i
        while j < len(states) and states[j][0] < end:
            a, b, reference_region, state = states[j]
            duration = min(b, end) - max(a, first)
            if duration > 0:
                label = state if reference_region == region else (
                    "other_channel_or_host_wait" if state == "idle_or_tail" else "other_phase_or_channel_wait")
                details[region][label] += duration
            j += 1
    if sum(allocations.values()) != total_cycles or sum(sum(v.values()) for v in details.values()) != total_cycles:
        raise ValueError("Global phase/state wall accounting failed")
    return allocations, details, commands, spans

"""Read-only cycle accounting for the legacy systolic KV-head TP path.

The full PIM cycle count is measured. Region allocations are estimates over
existing issued-command gaps or request-service counters, never completion
timestamps. No model execution, mapping, trace generation, or simulator runs.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
AIM_SCRIPTS = REPO_ROOT / "aim_simulator" / "scripts"
if str(AIM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(AIM_SCRIPTS))

from aim_analysis.ramulator import (  # noqa: E402
    command_trace_files,
    load_resolved_timing,
    parse_command_trace,
    read_result_stats,
    result_cycles,
)
import cent_power_calculator as calibration  # noqa: E402


CATALOG = (
    ("attention.rmsnorm", "Attention RMSNorm", "main", ("rmsnorm",)),
    ("attention.q_projection", "Q Projection", "main", ("q",)),
    ("attention.kv_projection", "K/V Projection", "main", ("k", "v")),
    ("attention.qk", "QK", "main", ("qk",)),
    ("attention.sv", "SV", "main", ("sv",)),
    ("attention.o_projection", "O Projection", "main", ("o",)),
    ("ffn.rmsnorm", "FFN RMSNorm", "main", ("rmsnorm",)),
    ("ffn.up_gate_projection", "Up/Gate Projection", "main", ("up", "gate")),
    ("ffn.down_projection", "Down Projection", "main", ("down",)),
    ("aux.rope_q", "RoPE Q", "auxiliary", ("rope_q",)),
    ("aux.rope_k", "RoPE K", "auxiliary", ("rope_k",)),
    ("aux.kv_update", "KV Update", "auxiliary", ("kv_update",)),
    ("aux.softmax", "Softmax", "auxiliary", ("softmax",)),
    ("aux.flash_merge", "FlashAttention Merge", "auxiliary", ("flash_merge",)),
    ("aux.score_transfer", "Score Transfer", "auxiliary", ("score_transfer",)),
    ("aux.ffn_activation", "FFN Activation", "auxiliary", ("activation",)),
    ("aux.attention_residual", "Attention Residual", "auxiliary", ("residual",)),
    ("aux.ffn_residual", "FFN Residual", "auxiliary", ("residual",)),
    ("aux.pp_handoff", "PP Handoff", "auxiliary", ("pp_handoff",)),
    ("aux.sync_control", "SYNC / Control", "auxiliary", ("sync", "control")),
    ("aux.unattributed", "Unattributed / Background / Tail", "auxiliary", ()),
)
PIM_REGIONS = {
    "q": "attention.q_projection", "fused_kv": "attention.kv_projection",
    "qk": "attention.qk", "sv": "attention.sv", "wo": "attention.o_projection",
    "fused_w1_w3": "ffn.up_gate_projection", "w2": "ffn.down_projection",
}
TERMINALS = {
    "MAC8": "MAC_ABK", "MAC16": "MAC_ABK", "WRGB": "WR_GB",
    "WRMAC8": "WR_BIAS", "WRMAC16": "WR_BIAS",
    "RDMAC8": "RD_MAC", "RDMAC16": "RD_MAC", "AF8": "AF", "AF16": "AF",
    "RDAF8": "RD_AF", "RDAF16": "RD_AF", "RD": "R_MEM", "WR": "W_MEM",
}
PREREQUISITES = {
    "TMOD", "ACT", "ACT4", "ACT16", "ACT-1", "ACT-2", "ACT4-1", "ACT4-2",
    "ACT8-1", "ACT8-2", "PRE", "PREA", "PRE4", "CASRD", "CASWR", "CASWRGB",
    "CASWRMAC8", "CASRDMAC8", "CASWRA8",
}
SERVICE_RE = re.compile(r"CH(\d+)_(AiM_ISR_.+|R|W)_cycles$")
TOLERANCE_NS = 1e-6


def number(row, key, default=None):
    raw = row.get(key)
    if raw is None or raw == "":
        if default is None:
            raise ValueError(f"Missing required case field: {key}")
        return default
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {key}: {raw!r}")
    return value


def integer(row, key, default=None):
    value = number(row, key, default)
    if value != int(value):
        raise ValueError(f"{key} must be an integer")
    return int(value)


def flag(row, key, default=False):
    value = row.get(key, default)
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "1"):
        return True
    if str(value).lower() in ("false", "0", "", "none"):
        return False
    raise ValueError(f"Invalid boolean {key}: {value!r}")


def assert_close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=TOLERANCE_NS):
        raise ValueError(f"{label} mismatch: computed {actual:.12g} ns, source {expected:.12g} ns")


def read_case(csv_path, *, model, tp, batch, context_window, sequence_length=None):
    with Path(csv_path).open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream)
                if r.get("Model") == model and integer(r, "Tensor parallelism") == tp
                and integer(r, "Batch size", 1) == batch
                and integer(r, "Context window") == context_window
                and (sequence_length is None or integer(r, "Sequence length") == sequence_length)]
    if len(rows) != 1:
        raise ValueError(f"Expected one source CSV row, found {len(rows)}; specify an exact case")
    return rows[0]


def formula_ledger(row, metadata=None):
    """Decompose the existing CENT formulas without mutating global calibration."""
    batch = integer(row, "Batch size", 1)
    tp = integer(row, "Tensor parallelism")
    channels = integer(row, "Channels per device")
    ports = integer(row, "Parallel SRAM banks")
    sequence = integer(row, "Sequence length")
    if min(batch, tp, channels, ports, sequence) <= 0:
        raise ValueError("Batch, TP, channels, SRAM ports and sequence must be positive")
    if sequence > integer(row, "Context window"):
        raise ValueError("Active sequence cannot exceed context window")
    if row.get("Model") != "Llama2-70B" or tp not in (1, 2, 4, 8):
        raise ValueError("v0.0 supports Llama2-70B KV-head TP=1/2/4/8")
    if row.get("Attention mapping") != "kv_head" or not flag(row, "Systolic pim"):
        raise ValueError("Expected the systolic KV-head TP path")
    if not flag(row, "EWMUL PNM effective"):
        raise ValueError("Systolic legacy accounting requires effective EWMUL_PNM")
    if integer(row, "Channels per block") != channels:
        raise ValueError("Representative KV-head TP requires one block per device")

    sb_read, sb_write = calibration.SB_RD_CYCLE, calibration.SB_WR_CYCLE
    vec = channels / ports * 2 * sb_read + 1 + sb_write
    exp = channels / ports * sb_read + calibration.EXP_LANE_CYCLE + sb_write
    period_ns = 1e9 / calibration.FREQ
    # PNM and TP partial vectors retain the legacy 16-element/BF16 model,
    # even when the supplied physical PIM payload is FP8/32 elements.
    one_norm = (8192 / 256 / channels * vec + 8192 / 16 / channels * vec * 3
                + sb_read + sb_write + 1 + calibration.RV_RMSNorm_CYCLE)
    q_dim, kv_dim, ffn_dim = 8192 // tp, 1024 // tp, 28672 // tp
    terms = {}

    def add(region, cycles, full_cycles=None):
        terms[(region, "pnm")] = {
            "duration_ns": cycles * period_ns * batch,
            "full_work_duration_ns": (cycles if full_cycles is None else full_cycles) * period_ns * batch,
            "formula_cycles": cycles * batch, "clock_period_ps": period_ns * 1000,
        }

    add("attention.rmsnorm", one_norm)
    add("ffn.rmsnorm", one_norm)
    for region, dim in (("aux.rope_q", q_dim), ("aux.rope_k", kv_dim)):
        add(region, dim * calibration.RV_ROTEmbed_CYCLE_PIPELINE + dim / 16 / channels * vec)
    local_heads = 64 // tp
    softmax_full = (sequence * local_heads / 16 / channels * (exp + vec + 2 * vec)
                    + local_heads * sb_read + local_heads * calibration.RV_SFT_CYCLE_PIPELINE)
    exposure = number(row, "Softmax exposure factor", 1.0)
    if flag(row, "Pipelined softmax"):
        startup = integer(row, "Softmax pipeline startup tokens")
        assert_close(exposure, min(startup, sequence) / sequence, "Softmax exposure factor")
    elif exposure != 1:
        raise ValueError("Unpipelined Softmax requires exposure factor=1")
    add("aux.softmax", softmax_full * exposure, softmax_full)
    add("aux.ffn_activation", ffn_dim / 16 / channels * 2 * vec)
    if flag(row, "Flash attention"):
        block = integer(row, "Flash attention block size")
        if block <= 0:
            raise ValueError("FlashAttention block size must be positive")
        if block * local_heads * 2 > integer(row, "Shared buffer capacity (bytes)"):
            raise ValueError("FlashAttention score block exceeds Shared Buffer capacity")
        add("aux.flash_merge", sequence // block * local_heads / 16 / channels * vec)
    reductions = (("q", "attention.q_projection", "Q reduction latency"),
                  ("kv", "attention.kv_projection", "KV reduction latency"),
                  ("sv", "attention.sv", "SV reduction latency"))
    for key, region, column in reductions:
        ns = number(row, column) * 1e6
        if metadata is not None:
            adds = metadata["pnm_reduction_adds"][key]
            assert_close(math.ceil(adds / 16 / channels) * vec * period_ns * batch, ns, column)
        terms[(region, "pnm")] = {
            "duration_ns": ns, "full_work_duration_ns": ns,
            "formula_cycles": ns / period_ns, "clock_period_ps": period_ns * 1000,
        }
    assert_close(terms[("aux.softmax", "pnm")]["duration_ns"],
                 number(row, "Exposed softmax latency") * 1e6, "Exposed Softmax")
    assert_close(terms[("aux.softmax", "pnm")]["full_work_duration_ns"],
                 number(row, "Full softmax latency") * 1e6, "Full Softmax")
    assert_close(terms[("aux.ffn_activation", "pnm")]["duration_ns"],
                 number(row, "EWMUL activation latency") * 1e6, "FFN activation")
    pnm_ns = sum(t["duration_ns"] for (r, b), t in terms.items() if b == "pnm")
    assert_close(pnm_ns, number(row, "Acc latency") * 1e6, "PNM total")
    collective = number(row, "TP collective CXL latency") * 1e6
    handoff = number(row, "PP handoff CXL latency") * 1e6
    if tp == 1 and collective != 0:
        raise ValueError("TP=1 must have zero TP collective latency")
    if integer(row, "Pipeline parallelism") == 1 and handoff != 0:
        raise ValueError("PP=1 must have zero PP handoff latency")
    for region, ns in (("attention.o_projection", collective / 2),
                       ("ffn.down_projection", collective / 2), ("aux.pp_handoff", handoff)):
        terms[(region, "cxl")] = {
            "duration_ns": ns, "full_work_duration_ns": ns,
            "formula_cycles": None, "clock_period_ps": None,
        }
    assert_close(collective + handoff, number(row, "CXL latency") * 1e6, "CXL total")
    return terms, {"pnm_clock_period_ps": period_ns * 1000, "vector_width": 16,
                   "shared_buffer_read_cycles": sb_read, "shared_buffer_write_cycles": sb_write,
                   "vector_add_mul_cycles": vec, "exp_cycles": exp,
                   "rv_rmsnorm_cycles": calibration.RV_RMSNorm_CYCLE,
                   "rv_rope_cycles_per_element": calibration.RV_ROTEmbed_CYCLE_PIPELINE,
                   "rv_softmax_cycles_per_head": calibration.RV_SFT_CYCLE_PIPELINE}


def validate_metadata(row, metadata):
    if "flash_attention" in metadata and metadata["flash_attention"] != flag(row, "Flash attention"):
        raise ValueError("Metadata FlashAttention mode does not match source CSV")
    if flag(row, "Flash attention") and "flash_attention_block_size" in metadata:
        if metadata["flash_attention_block_size"] != integer(row, "Flash attention block size"):
            raise ValueError("Metadata FlashAttention block size does not match source CSV")
    if not metadata.get("standard_tp"):
        raise ValueError("Systolic metadata must declare standard_tp=true")
    for key, column in (("tp", "Tensor parallelism"), ("batch_size", "Batch size"),
                        ("sequence_length", "Sequence length")):
        if metadata[key] != integer(row, column):
            raise ValueError(f"Metadata {key} does not match source CSV")
    shape = metadata["shape"]
    tp = integer(row, "Tensor parallelism")
    expected = {"dim": 8192, "global_query_heads": 64, "global_kv_heads": 8,
                "global_ffn_dim": 28672, "head_dim": 128, "tp": tp,
                "local_query_heads": 64 // tp, "local_kv_heads": 8 // tp,
                "local_query_dim": 8192 // tp, "local_kv_dim": 1024 // tp,
                "local_ffn_dim": 28672 // tp}
    if any(shape.get(k) != v for k, v in expected.items()):
        raise ValueError("Metadata tensor shapes do not match Llama2-70B TP")
    if metadata["array"]["height"] != integer(row, "Systolic dim"):
        raise ValueError("Metadata array height does not match source CSV")
    bits = integer(row, "PIM element bits", 16)
    if bits not in (8, 16):
        raise ValueError("Supported physical PIM element sizes are 8 and 16 bits")
    if metadata["array"]["width"] != 256 // bits:
        raise ValueError("Metadata PIM array width does not match 256-bit element packing")
    if metadata.get("element_bits", bits) != bits:
        raise ValueError("Metadata element bits do not match source CSV")
    if metadata.get("precision", row.get("Precision")) != row.get("Precision"):
        raise ValueError("Metadata precision does not match source CSV")
    if not all(metadata.get("fused_row_aliases", {}).get(k) for k in ("wk_wv", "w1_w3")):
        raise ValueError("Expected fused K/V and Up/Gate physical aliases")


def phase_budgets(metadata):
    work = metadata["kernel_pipeline_cycles"]
    batch = metadata["batch_size"]

    def total(name):
        values = work[name]
        if set(values) != {"fill", "reduction", "drain"}:
            raise ValueError(f"Invalid systolic stage fields for {name}")
        if any(not isinstance(v, int) or v < 0 for v in values.values()):
            raise ValueError(f"Invalid stage work for {name}")
        return sum(values.values())

    if metadata.get("command_phases"):
        phases, observed = [], Counter()
        for phase in metadata["command_phases"]:
            name, amount = phase["kernel"], phase["mac_work"]
            if name not in work or not isinstance(amount, int) or amount <= 0:
                raise ValueError("Invalid ordered systolic command phase")
            observed[name] += amount
            phases.append(("sv" if name.startswith("sv_") else name, amount))
        if dict(observed) != {name: total(name) for name in work}:
            raise ValueError("Ordered command phases do not reconcile with kernel work")
        if sum(observed.values()) != sum(metadata["pipeline_cycles"].values()):
            raise ValueError("Ordered command phases do not reconcile with metadata total")
        return phases

    sv_names = [name for name in work if name.startswith("sv_")]
    if len(sv_names) != 1:
        raise ValueError("Expected one SV packing mode")
    qk, sv = total("qk"), total(sv_names[0])
    if qk % batch or sv % batch:
        raise ValueError("Attention stage work must divide by batch")
    phases = [("q", total("q")), ("fused_kv", total("fused_kv"))]
    for _ in range(batch):
        phases.extend((("qk", qk // batch), ("sv", sv // batch)))
    phases.extend((name, total(name)) for name in ("wo", "fused_w1_w3", "w2"))
    if any(amount <= 0 for _, amount in phases):
        raise ValueError("Expected complete, positive projection/attention stage work")
    if sum(v for _, v in phases) != sum(metadata["pipeline_cycles"].values()):
        raise ValueError("Per-kernel stage work does not reconcile with metadata total")
    return phases


def trace_work(trace_path, metadata, channel):
    """Decode observed commands against sidecar budgets, not a mapping algorithm.

    This supports the non-Flash standard-TP emission order. Ownership remains
    a protocol-based estimate; no dependency or completion is reconstructed.
    """
    phases = phase_budgets(metadata)
    index, remaining = 0, phases[0][1]
    weights = defaultdict(Counter)
    owners = defaultdict(deque)
    counts = Counter()
    region_counts = defaultdict(Counter)
    last_opcode = None
    with Path(trace_path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields or fields[0] == "#":
                continue
            prefix = fields[:2]
            if prefix in (["R", "MEM"], ["W", "MEM"]):
                family = fields[0] + "_MEM"
                if phases[index][0] == "fused_kv" and remaining == 0:
                    region = "aux.kv_update"
                elif phases[index][0] == "qk" and remaining == 0:
                    region = "aux.score_transfer"
                else:
                    raise ValueError(f"Unrecognized MEM phase at trace line {line_number}")
                counts[family] += 1
                region_counts[region][family] += 1
                if int(fields[2]) == channel:
                    weights[family][region] += 1
                    owners[family].append((region, int(fields[4])))
                last_opcode = family
                continue
            if fields[0] != "AiM":
                raise ValueError(f"Unsupported host command at trace line {line_number}")
            opcode = fields[1].removeprefix("ISR_")
            counts[opcode] += 1
            last_opcode = opcode
            if opcode == "WR_BIAS" and remaining == 0:
                index += 1
                if index >= len(phases):
                    raise ValueError("Trace has more compute phases than sidecar")
                remaining = phases[index][1]
            region = PIM_REGIONS[phases[index][0]]
            if opcode in ("AF", "RD_AF"):
                region = "aux.ffn_activation"
            if opcode in ("SYNC", "EOC"):
                region_counts["aux.sync_control"][opcode] += 1
                continue
            if opcode == "EWADD":
                residual = "aux.attention_residual" if phases[index][0] == "wo" else "aux.ffn_residual"
                region_counts[residual][opcode] += 1
                continue
            if opcode not in ("MAC_ABK", "WR_GB", "WR_BIAS", "RD_MAC", "AF", "RD_AF"):
                raise ValueError(f"Unsupported standard-TP opcode {opcode}")
            size = int(fields[2]) if opcode in ("MAC_ABK", "WR_GB") else 1
            if size <= 0:
                raise ValueError("PIM work size must be positive")
            if opcode == "MAC_ABK":
                remaining -= size
                if remaining < 0:
                    raise ValueError(f"Trace MAC work exceeds sidecar budget at line {line_number}")
            mask = int(fields[4] if opcode == "WR_GB" else fields[2] if opcode == "AF" else fields[3], 0)
            # The existing AiM implementation has a fixed, reversed 32-bit mask.
            if mask < 0 or mask >= 1 << 32:
                raise ValueError("Invalid 32-bit channel mask")
            region_counts[region][opcode] += 1
            if mask & (1 << (31 - channel)):
                weights[opcode][region] += size
                row = int(fields[4]) if opcode == "MAC_ABK" else None
                owners[opcode].extend((region, row) for _ in range(size))
    if index != len(phases) - 1 or remaining != 0 or last_opcode != "EOC" or counts["EOC"] != 1:
        raise ValueError("Trace is incomplete or does not match the full standard-TP protocol")
    return weights, owners, counts, region_counts


def verify_trace_counts(counts, stats):
    for opcode, expected in counts.items():
        key = f"total_num_{opcode}_requests" if opcode.endswith("_MEM") else f"total_num_AiM_ISR_{opcode}_requests"
        if key in stats and stats[key] != expected:
            raise ValueError(f"Trace/log request count mismatch for {opcode}: {expected} vs {stats[key]}")


def issued_gap_estimate(events, owners, total_cycles, *, details=None):
    """Account issued gaps by the next aligned terminal's inferred region."""
    allocations = Counter()
    command_details = defaultdict(lambda: defaultdict(
        lambda: {"issued_event_count": 0, "gap_cycles": 0, "predecessors": Counter()}
    ))

    def account(region, command, predecessor, gap, issued=True):
        allocations[region] += gap
        if details is not None:
            entry = command_details[region][command]
            entry["issued_event_count"] += int(issued)
            entry["gap_cycles"] += gap
            entry["predecessors"][(predecessor, gap)] += 1

    labels = []
    last = -1
    for event in events:
        if event.clock < last or event.clock > total_cycles:
            raise ValueError("Issued command clocks are not monotonic/in the full-run interval")
        last = event.clock
        family = TERMINALS.get(event.command)
        owner = None
        if family:
            if not owners[family]:
                raise ValueError(f"Unexpected issued terminal {event.command}")
            owner, expected_row = owners[family].popleft()
            # AiM compute stays ordered. Conventional accesses can reorder
            # within one auxiliary region, so only MAC row order is checked.
            if family == "MAC_ABK" and (len(event.address) < 2 or event.address[-2] != expected_row):
                raise ValueError("Issued MAC rows do not align with the source trace")
        labels.append(owner)
    if any(owners.values()):
        raise ValueError("Issued terminals do not exhaust source trace work on this channel")
    future = None
    for index in range(len(events) - 1, -1, -1):
        if labels[index] is not None:
            future = labels[index]
        elif events[index].command in PREREQUISITES:
            labels[index] = future
    previous = 0
    for index, event in enumerate(events):
        region = labels[index] if index else None
        account(region or "aux.unattributed", event.command if index else "STARTUP",
                events[index - 1].command if index else "TRACE_START",
                event.clock - previous, issued=index > 0)
        previous = event.clock
    account("aux.unattributed", "TAIL", events[-1].command if events else "TRACE_START",
            total_cycles - previous, issued=False)
    if details is not None:
        details.clear()
        for region, commands in command_details.items():
            details[region] = {
                "total_gap_cycles": allocations[region],
                "commands": [
                    {"command": command, "issued_event_count": entry["issued_event_count"],
                     "gap_cycles": entry["gap_cycles"],
                     "predecessor_gaps": [
                         {"previous_command": predecessor, "gap_cycles": gap, "interval_count": count}
                         for (predecessor, gap), count in sorted(entry["predecessors"].items())
                     ]}
                    for command, entry in sorted(commands.items())
                ],
            }
    return allocations


def service_estimate(stats, weights, channel, total_cycles):
    allocations = Counter()
    pool_details = {}
    for key, cycles in stats.items():
        match = SERVICE_RE.fullmatch(key)
        if not match or int(match[1]) != channel or cycles == 0:
            continue
        family = match[2].removeprefix("AiM_ISR_")
        if family in ("R", "W"):
            family += "_MEM"
        shares = weights.get(family, {})
        denominator = sum(shares.values())
        if denominator:
            pool_details[family] = cycles
            for region, work in shares.items():
                allocations[region] += cycles * work / denominator
    if sum(allocations.values()) > total_cycles + 1e-9:
        raise ValueError("Overlapping service pools exceed elapsed cycles; cannot form an additive estimate")
    allocations["aux.unattributed"] = total_cycles - sum(allocations.values())
    return allocations, pool_details


def choose_command_trace(prefix):
    candidates = []
    for path in command_trace_files(prefix):
        events = parse_command_trace(path)
        candidates.append((events[-1].clock, len(events), int(path.suffix[3:]), path))
    return max(candidates)[3] if candidates else None


def path_record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    try:
        display = str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        display = str(path.resolve())
    return {"path": display, "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def git_identity():
    def value(*args):
        result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": value("rev-parse", "HEAD"), "branch": value("branch", "--show-current"),
            "aim_checkout_commit": value("-C", "aim_simulator", "rev-parse", "HEAD")}


def analyze_case(row, *, log, timing, source_csv, trace=None, metadata_path=None,
                 config=None, provenance=None, command_trace=None, estimator="auto"):
    if estimator not in ("auto", "issued-gap", "service-work"):
        raise ValueError(f"Unknown estimator: {estimator}")
    stats = read_result_stats(Path(log))
    total_cycles = result_cycles(stats)
    impl, resolved = load_resolved_timing(Path(timing))
    organization = yaml.safe_load(Path(timing).read_text()).get("organization")
    if organization is not None:
        if (organization["channels"] != integer(row, "Channels per device")
                or organization["prefetch"] * organization["channel_width_bits"]
                != organization["transaction_bytes"] * 8):
            raise ValueError("Resolved DRAM organization does not match channels/transaction size")
    period_ps = resolved["tCK_ps"]
    if total_cycles < 0 or period_ps <= 0:
        raise ValueError("Invalid full-run clock/cycles")
    if integer(row, "Channels per device") != 32:
        raise ValueError("Initial v0.0 protocol estimator supports the existing 32-channel target")
    if config is not None:
        cfg = yaml.safe_load(Path(config).read_text())
        memory = cfg["MemorySystem"]
        if memory["DRAM"]["impl"] != impl:
            raise ValueError("Resolved/config DRAM implementation mismatch")
        for node in (cfg["Frontend"], memory, memory["Controller"]):
            if node.get("clock_ratio", 1) != 1:
                raise ValueError("Initial v0.0 estimates require clock_ratio=1")
        configured_nccd = memory["DRAM"]["timing"].get("nCCD")
        if configured_nccd is not None and configured_nccd != resolved.get("nCCD"):
            raise ValueError("Configured/resolved nCCD mismatch")
        expected_width = 32 if flag(row, "Channel concurrency") else 1
        if (cfg["Frontend"].get("issue_width", 1) != expected_width
                or memory.get("host_issue_width", 1) != expected_width):
            raise ValueError("Configured channel concurrency does not match source CSV")
    metadata = json.loads(Path(metadata_path).read_text()) if metadata_path is not None else None
    if metadata is not None:
        validate_metadata(row, metadata)
    formula, profile = formula_ledger(row, metadata)
    pim_ns = total_cycles * period_ps / 1000
    assert_close(pim_ns, number(row, "PIM latency") * 1e6, "PIM log/CSV latency")
    assert_close(pim_ns, number(row, "Main PIM latency") * 1e6, "Main PIM latency")
    helper_present = integer(row, "Tensor parallelism") > 1
    assert_close(pim_ns if helper_present else 0.0,
                 number(row, "Helper PIM latency") * 1e6, "Symmetric helper PIM latency")
    assert_close(number(row, "Main Acc latency") * 1e6, number(row, "Acc latency") * 1e6, "Main PNM latency")
    assert_close(number(row, "Helper Acc latency") * 1e6,
                 number(row, "Acc latency") * 1e6 if helper_present else 0.0,
                 "Symmetric helper PNM latency")
    diagnostics = []
    if config is None:
        diagnostics.append("Backend config unavailable; cycle conversion follows the legacy log/CSV tCK convention")
    pools = Counter()
    for key, value in stats.items():
        match = SERVICE_RE.fullmatch(key)
        if match:
            pools[int(match[1])] += value
    channel = max(pools, key=lambda c: (pools[c], -c)) if pools else 0
    can_decode = metadata is not None and trace is not None and (
        not flag(row, "Flash attention") or bool(metadata.get("command_phases")))
    if command_trace is None and estimator in ("auto", "issued-gap") and can_decode:
        prefix = str(log).removesuffix(".log") + ".cmd"
        command_trace = choose_command_trace(prefix)
    if estimator == "issued-gap" and command_trace is None:
        raise ValueError("issued-gap estimator requires an existing TraceRecorder artifact")
    allocations = Counter({"aux.unattributed": total_cycles})
    method = "unavailable"
    counts, region_counts, pool_details, gap_details = {}, {}, {}, {}
    decoded = False
    if can_decode:
        if command_trace is not None:
            channel = int(Path(command_trace).suffix[3:])
            if not 0 <= channel < 32:
                raise ValueError("Issued command trace channel must be in [0, 31]")
        weights, owners, counts, region_counts = trace_work(trace, metadata, channel)
        verify_trace_counts(counts, stats)
        decoded = True
        if command_trace is not None and estimator in ("auto", "issued-gap"):
            allocations = issued_gap_estimate(parse_command_trace(Path(command_trace)), owners,
                                             total_cycles, details=gap_details)
            method = "legacy_tp_issued_gap_v1"
        else:
            allocations, pool_details = service_estimate(stats, weights, channel, total_cycles)
            method = "legacy_tp_channel_service_work_v1"
    else:
        diagnostics.append("PIM region ownership unavailable: requires sidecar + full standard-TP trace; Flash requires ordered command phases; total preserved")
        if estimator == "issued-gap":
            raise ValueError(diagnostics[-1])
    diagnostics.extend([
        "PIM region allocations are estimates; no host/kernel completion timestamps exist",
        "Issued-gap ownership uses a validated emission protocol and one reference channel; shared stalls are not a unique causal attribution",
        "PNM/CXL retain legacy formulas; residual EWADD arithmetic is unmodeled",
        "FP8 PIM does not change the legacy 16-element PNM/BF16 partial-vector formulas",
    ])
    regions = []
    for region_id, name, classification, members in CATALOG:
        backend_terms = {}
        for backend in ("pim", "pnm", "cxl"):
            value = formula.get((region_id, backend))
            is_pim_work = region_id in PIM_REGIONS.values() or region_id in ("aux.kv_update", "aux.score_transfer", "aux.ffn_activation")
            unmodeled = region_id.endswith("residual") and backend == "pim"
            unavailable = backend == "pim" and (
                unmodeled or region_id == "aux.sync_control" or (is_pim_work and not decoded))
            duration = value["duration_ns"] if value else allocations[region_id] * period_ps / 1000 if backend == "pim" else 0.0
            full = value["full_work_duration_ns"] if value else duration
            backend_terms[backend] = {
                "duration_ns": None if unavailable else duration,
                "legacy_ledger_duration_ns": duration,
                "full_work_duration_ns": None if unavailable else full,
                "reference_equivalent_cycles": None if unavailable else duration * 1000 / period_ps,
                "native_cycles": None, "formula_cycles": value["formula_cycles"] if value else None,
                "clock_period_ps": value["clock_period_ps"] if value else period_ps if backend == "pim" else None,
                "measurement_kind": "unavailable" if unavailable else "estimated_attribution" if backend == "pim" else "legacy_formula",
                "coverage": "unmodeled" if unmodeled else "unavailable" if unavailable else "estimated" if backend == "pim" else "modeled",
                "observed_span_cycles": None, "accounted_cycles": None,
                "accept_cycle": None, "first_issue_cycle": None, "last_issue_cycle": None, "completion_cycle": None,
            }
        duration = sum(term["legacy_ledger_duration_ns"] for term in backend_terms.values())
        regions.append({"region_id": region_id, "display_name": name,
                        "main_or_auxiliary": classification, "logical_members": list(members),
                        "fusion": len(members) > 1, "segment_id": None, "instance_id": None,
                        "command_counts": dict(region_counts.get(region_id, {})), "backends": backend_terms,
                        "pim_issued_gap_details": gap_details.get(region_id),
                        "legacy_ledger_duration_ns": duration,
                        "reference_equivalent_cycles": duration * 1000 / period_ps})
        if region_id in gap_details:
            detailed_cycles = sum(entry["gap_cycles"] for entry in gap_details[region_id]["commands"])
            assert_close(detailed_cycles * period_ps / 1000,
                         backend_terms["pim"]["legacy_ledger_duration_ns"], f"{name} issued-gap details")
    pnm_ns = sum(v["duration_ns"] for (r, b), v in formula.items() if b == "pnm")
    cxl_ns = sum(v["duration_ns"] for (r, b), v in formula.items() if b == "cxl")
    block_ns = pim_ns + pnm_ns + cxl_ns
    assert_close(block_ns, number(row, "TransformerBlock latency") * 1e6, "Block total")
    ledger_ns = sum(r["legacy_ledger_duration_ns"] for r in regions)
    assert_close(ledger_ns, block_ns, "Region additive ledger")
    assert_close(sum(allocations.values()) * period_ps / 1000, pim_ns, "PIM additive ledger")
    paths = {"source_csv": source_csv, "pim_log": log, "resolved_timing": timing,
             "trace": trace, "systolic_metadata": metadata_path, "backend_config": config,
             "source_provenance": provenance, "issued_command_trace": command_trace,
             "postprocessor": Path(__file__), "calibration_adapter": Path(calibration.__file__),
             "report_cli": REPO_ROOT / "cent_simulation" / "scripts" / "analyze_cycle_breakdown.py",
             "calibration_model": Path(calibration.CELLAR_POWER_CALCULATOR.__file__),
             "legacy_formula_reference": REPO_ROOT / "cent_simulation" / "run_sim.py",
             "stats_parser": AIM_SCRIPTS / "aim_analysis" / "ramulator.py"}
    artifacts = {key: path_record(path) for key, path in paths.items() if path is not None}
    identity = {"artifacts": artifacts, "source_row": dict(row), "formula_profile": profile,
                "estimator": method, "source": git_identity()}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    report = {
        "profile_schema_version": "0.1", "legacy_framework_version": "0.0",
        "method": "postprocess_estimate", "scope": "representative_block_batch_group",
        "run_id": run_id, "run_identity": identity,
        "case": {"model": row["Model"], "tp": integer(row, "Tensor parallelism"),
                 "batch_size": integer(row, "Batch size"), "pp": integer(row, "Pipeline parallelism"),
                 "context_window": integer(row, "Context window"), "sequence_length": integer(row, "Sequence length"),
                 "precision": row.get("Precision", "legacy_16bit"), "pim_element_bits": integer(row, "PIM element bits", 16),
                 "array_height": integer(row, "Systolic dim"), "array_width": metadata["array"]["width"] if metadata else None,
                 "dram_timing_impl": impl, "dram_power_label": Path(source_csv).parent.name,
                 "dram_organization": organization, "nBL": int(resolved["nBL"]),
                 "nCCD": int(resolved["nCCD"]) if "nCCD" in resolved else None,
                 "flash_attention": flag(row, "Flash attention"),
                 "flash_attention_block_size": integer(row, "Flash attention block size", 0),
                 "ewmul_pnm": flag(row, "EWMUL PNM effective"),
                 "pipelined_softmax": flag(row, "Pipelined softmax"),
                 "channel_concurrency": flag(row, "Channel concurrency")},
        "reference_clock_period_ps": period_ps, "estimator": {"id": method, "reference_channel": channel,
            "service_pools_cycles": dict(pool_details), "completion_observed": False,
            "timing_cycles": {key: resolved[key] for key in
                              ("nCCD", "nRCDRDMAC", "nRPab", "nRTP", "nRFCab", "nMODCH") if key in resolved},
            "policy": "following issued-event/terminal gap ownership; prerequisites look ahead; background/startup/tail separate"
                      if method == "legacy_tp_issued_gap_v1" else "one channel's service pools split by traced work; elapsed remainder unassigned"},
        "regions": regions, "trace_command_counts": dict(counts), "diagnostics": diagnostics,
        "full_run_totals": {"pim_native_cycles": total_cycles, "pim_latency_ns": pim_ns,
                            "pnm_exposed_latency_ns": pnm_ns, "cxl_latency_ns": cxl_ns,
                            "block_latency_ns": block_ns, "reference_equivalent_cycles": block_ns * 1000 / period_ps,
                            "per_output_token_amortized_ns": block_ns / integer(row, "Batch size")},
        "reconciliation_residuals_ns": {"pim": sum(allocations.values()) * period_ps / 1000 - pim_ns,
                                        "pnm": pnm_ns - number(row, "Acc latency") * 1e6,
                                        "cxl": cxl_ns - number(row, "CXL latency") * 1e6,
                                        "block": ledger_ns - number(row, "TransformerBlock latency") * 1e6},
    }
    return report


def render_issued_gap_details(report):
    """Expand existing PIM allocations without adding another latency charge."""
    if report["estimator"]["id"] != "legacy_tp_issued_gap_v1":
        return ["", "PIM 发射事件间隔明细：unavailable；本次没有使用 issued-command gap 归因。"]
    if not any(r.get("pim_issued_gap_details") is not None for r in report["regions"]):
        return ["", "PIM 发射事件间隔明细：unavailable；请用新版后处理器重新生成报表。"]
    lines = [
        "", "PIM 发射事件间隔明细",
        f"参考 channel={report['estimator']['reference_channel']}；单位为 {report['reference_clock_period_ps']:g} ps 的参考周期。",
        "下列明细展开上表的 PIM 分项，已经计入 block 总计，不再额外相加。",
        "间隔归给后一个发射事件；前置命令关联后续 terminal 所属部分。包含等待，不等同命令执行或 completion latency。",
        "STARTUP/TAIL 是区间记录，不是发射命令；刷新等未归属事件保留在 Unattributed 中。",
    ]
    timing = report["estimator"].get("timing_cycles", {})
    if timing:
        lines.append("Resolved timing（cycles）：" + ", ".join(f"{key}={value:g}" for key, value in timing.items()) + "。")

    def group(command):
        descriptions = {
            "WRGB": "WRGB：写入输入 GB",
            "WRMAC8": "WRMAC8：初始化累加器，对应 WR_BIAS",
            "WRMAC16": "WRMAC16：初始化累加器，对应 WR_BIAS",
            "RDMAC8": "RDMAC8：读取结果",
            "RDMAC16": "RDMAC16：读取结果",
            "RD": "RD：DRAM 读取", "WR": "WR：DRAM 写入",
            "AF8": "AF8：阵列内 activation", "RDAF8": "RDAF8：读取 activation 结果",
            "STARTUP": "STARTUP：首次发射前", "TAIL": "TAIL：末次发射后",
        }
        if command.startswith(("ACT", "PRE")):
            return "ACT / PRE"
        if command == "TMOD" or command.startswith("CAS"):
            return "TMOD / CAS 等控制命令"
        return descriptions.get(command, command)

    preferred = ("MAC8", "MAC16", "WRGB", "WRMAC8", "WRMAC16", "RDMAC8", "RDMAC16")
    for region in report["regions"]:
        detail = region.get("pim_issued_gap_details")
        if detail is None:
            continue
        commands = detail["commands"]
        grouped = Counter()
        for entry in sorted(commands, key=lambda e: (
                preferred.index(e["command"]) if e["command"] in preferred else len(preferred), e["command"])):
            grouped[group(entry["command"])] += entry["gap_cycles"]
        lines.extend(["", f"{region['display_name']} [{region['region_id']}]",
                      "  归到这些发射事件的时间间隔                         周期"])
        for label, cycles in grouped.items():
            lines.append(f"  {label:<43} {cycles:>12,}")
        lines.append(f"  {'PIM 合计':<43} {detail['total_gap_cycles']:>12,}")
        lines.extend(["  原始命令明细：", "    命令                  发射事件数         归属周期"])
        for entry in commands:
            lines.append(f"    {entry['command']:<20} {entry['issued_event_count']:>10,} {entry['gap_cycles']:>16,}")
        for entry in commands:
            if entry["command"] not in ("MAC8", "MAC16"):
                continue
            lines.extend([f"  {entry['command']} 前的发射间隔分布：",
                          "    前一发射命令           单次间隔        区间数         累计周期"])
            for bucket in entry["predecessor_gaps"]:
                gap, count = bucket["gap_cycles"], bucket["interval_count"]
                lines.append(f"    {bucket['previous_command']:<20} {gap:>10,} {count:>12,} {gap * count:>16,}")
            if "nCCD" in timing:
                baseline = entry["issued_event_count"] * timing["nCCD"]
                lines.append(f"    以每条 nCCD={timing['nCCD']:g} 为参考预算："
                             f"{entry['issued_event_count']:,} × {timing['nCCD']:g} = {baseline:,.0f}；"
                             f"实际归属 {entry['gap_cycles']:,}；差额 {entry['gap_cycles'] - baseline:,.0f}。")
        lines.append("  Backend 合计（等效周期）：" + " + ".join(
            f"{backend.upper()}={region['backends'][backend]['reference_equivalent_cycles']:,.3f}"
            for backend in ("pim", "pnm", "cxl")
        ) + f" = {region['reference_equivalent_cycles']:,.3f}。")
    detailed_total = sum(r["pim_issued_gap_details"]["total_gap_cycles"] for r in report["regions"]
                         if r.get("pim_issued_gap_details") is not None)
    lines.extend(["", f"全部 PIM 事件间隔合计：{detailed_total:,}；"
                  f"整条 trace：{report['full_run_totals']['pim_native_cycles']:,}；"
                  f"差额：{detailed_total - report['full_run_totals']['pim_native_cycles']:,} cycles。"])
    return lines


def render_backend_details(report):
    """Show all regions, including kernels represented only by PNM formulas."""
    period_ns = report["reference_clock_period_ps"] / 1000
    lines = ["", "逐 kernel 的 PIM / PNM / CXL 周期组成",
             "等效周期统一按 PIM 参考时钟换算；PNM formula cycles 使用其独立时钟。",
             "以下是已有总计的展开，不重复计时；未建模项不能解释为硬件耗时为零。"]
    for region in report["regions"]:
        lines.extend(["", f"{region['display_name']} [{region['region_id']}]"])
        for backend in ("pim", "pnm", "cxl"):
            term = region["backends"][backend]
            ns = term["duration_ns"]
            if ns is None:
                lines.append(f"  {backend.upper()}: {term['coverage']}；独立执行时间 unavailable。")
                continue
            equivalent = term["reference_equivalent_cycles"]
            if backend == "pnm" and term["formula_cycles"] is not None:
                native_period = term["clock_period_ps"] / 1000
                lines.append(f"  PNM: {term['formula_cycles']:,.6f} formula cycles × {native_period:g} ns"
                             f" = {ns:,.6f} ns = {equivalent:,.6f} 等效周期。")
                if term["full_work_duration_ns"] != ns:
                    full = term["full_work_duration_ns"]
                    lines.append(f"    完整 Softmax 工作：{full:,.6f} ns；exposed：{ns:,.6f} ns；"
                                 f"分析模型中的 overlap：{full - ns:,.6f} ns。")
            elif backend == "pim":
                lines.append(f"  PIM: {equivalent:,.6f} 归属周期 × {period_ns:g} ns = {ns:,.6f} ns。")
            else:
                lines.append(f"  {backend.upper()}: {ns:,.6f} ns / {period_ns:g} ns"
                             f" = {equivalent:,.6f} 等效周期（legacy 公式）。")
        lines.append(f"  Legacy 合计：{region['legacy_ledger_duration_ns']:,.6f} ns"
                     f" = {region['reference_equivalent_cycles']:,.6f} 等效周期。")
    return lines


def render_summary(report):
    """Generate the user's summary from the same numbers as JSON/CSV."""
    case, totals = report["case"], report["full_run_totals"]
    lines = [
        f"Llama2-70B legacy v0.0 cycle breakdown — {case['dram_timing_impl']} nCCD={case['nCCD']}, TP={case['tp']}, batch={case['batch_size']}",
        f"精度：{case['precision']}；PIM 阵列：{case['array_height']}×{case['array_width']}。",
        f"Context window：{case['context_window']:,}；实际 decode 长度：{case['sequence_length']:,}。",
        f"优化开关：EWMUL_PNM={case.get('ewmul_pnm', True)}；FlashAttention={case['flash_attention']}"
        f"（block={case.get('flash_attention_block_size', 0)}）；pipelined_softmax={case['pipelined_softmax']}；"
        f"channel_concurrency={case.get('channel_concurrency', False)}。",
        f"范围：一个代表 rank 上一个 block 的 batch group；PP={case['pp']}，未乘 80 层或 TP。",
        f"参考周期：{report['reference_clock_period_ps']:g} ps；下列周期均为该时钟的等效周期。",
        f"整条 trace 的 PIM 实测周期：{totals['pim_native_cycles']:,}。",
        f"Legacy block 合计：{totals['block_latency_ns']/1000:.6f} µs；等效周期 {totals['reference_equivalent_cycles']:,.3f}。",
        f"其中 PIM={totals['pim_latency_ns']/1000:.6f} µs，PNM={totals['pnm_exposed_latency_ns']/1000:.6f} µs，CXL={totals['cxl_latency_ns']/1000:.6f} µs。",
        f"每个输出 token 的摊销 block 时间：{totals['per_output_token_amortized_ns']/1000:.6f} µs（不是单请求端到端 token latency）。",
        f"PIM 分项方法：{report['estimator']['id']}，参考 channel={report['estimator']['reference_channel']}。",
        "PIM kernel 分项是估算归因，未观测 completion；PNM/CXL 是 legacy 公式。K/V、Up/Gate 保持融合。",
        "",
        "部分                                      PIM等效周期     PNM等效周期     CXL等效周期      合计(µs)    占block",
    ]
    organization = case.get("dram_organization")
    if organization:
        lines.insert(2, f"DRAM interface: {organization['channels']} channels × x{organization['channel_width_bits']}; "
                     f"device DQ={organization['dq_bits']}; prefetch={organization['prefetch']}; "
                     f"transaction={organization['transaction_bytes']} B; nBL={case['nBL']} clocks.")
    for region in report["regions"]:
        if region["region_id"].endswith("residual"):
            lines.append(f"{region['display_name']}: arithmetic 未建模；旧账本没有单独计时。")
            continue
        if region["legacy_ledger_duration_ns"] == 0:
            if region["region_id"] == "aux.score_transfer" and case["flash_attention"]:
                lines.append("Score Transfer: 0；FlashAttention 未物化 score DRAM workspace，无对应读写命令。")
            continue
        terms = region["backends"]
        values = ["unavailable" if terms[b]["reference_equivalent_cycles"] is None
                  else f"{terms[b]['reference_equivalent_cycles']:,.3f}" for b in ("pim", "pnm", "cxl")]
        us = region["legacy_ledger_duration_ns"] / 1000
        percent = 100 * region["legacy_ledger_duration_ns"] / totals["block_latency_ns"]
        lines.append(f"{region['display_name']:<39} {values[0]:>13} {values[1]:>15} {values[2]:>15} {us:>14.6f} {percent:>9.2f}%")
    lines.extend(render_backend_details(report))
    lines.extend(render_issued_gap_details(report))
    lines.extend(["", "对账差额（ns）：" + json.dumps(report["reconciliation_residuals_ns"], sort_keys=True),
                  "Source revisions/provenance 与输入 SHA256 见 run_manifest.json。",
                  "SYNC/control 的独立完成时间 unavailable；其共享等待可能包含在 issued gaps 中。"])
    return "\n".join(lines) + "\n"


def csv_rows(report):
    for region in report["regions"]:
        row = {"Region": region["region_id"], "Name": region["display_name"],
               "Classification": region["main_or_auxiliary"], "Logical members": "+".join(region["logical_members"]),
               "Legacy exposed latency (us)": region["legacy_ledger_duration_ns"] / 1000,
               "Reference equivalent cycles": region["reference_equivalent_cycles"]}
        for backend, term in region["backends"].items():
            row[f"{backend.upper()} latency (us)"] = None if term["duration_ns"] is None else term["duration_ns"] / 1000
            row[f"{backend.upper()} equivalent cycles"] = term["reference_equivalent_cycles"]
            row[f"{backend.upper()} measurement"] = term["measurement_kind"]
            row[f"{backend.upper()} coverage"] = term["coverage"]
        row["Full Softmax latency (us)"] = region["backends"]["pnm"]["full_work_duration_ns"] / 1000 if region["region_id"] == "aux.softmax" else None
        yield row


def write_report(report, output_dir):
    """Publish deterministic artifacts; an existing differing run is never overwritten."""
    import io
    from tempfile import TemporaryDirectory
    output_dir = Path(output_dir)
    run_dir = output_dir / report["run_id"]
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    rows = list(csv_rows(report))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    files = {"breakdown.json": serialized, "breakdown.csv": stream.getvalue(),
             "summary.txt": render_summary(report),
             "run_manifest.json": json.dumps({"run_id": report["run_id"], **report["run_identity"]}, indent=2, sort_keys=True) + "\n"}
    if run_dir.exists():
        for name, content in files.items():
            if not (run_dir / name).exists() or (run_dir / name).read_bytes() != content.encode():
                raise ValueError(f"Existing run differs or is incomplete: {run_dir}")
        return run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".cycle_breakdown_", dir=output_dir) as temp:
        for name, content in files.items():
            Path(temp, name).write_bytes(content.encode())
        # Rename only to a nonexistent content-addressed directory.
        Path(temp).rename(run_dir)
    return run_dir

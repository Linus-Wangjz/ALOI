# Standard TP on systolic PIM

This path keeps the existing KV-head TP tensor shapes and collective semantics,
then replaces the old logical-only placement with an explicit
Device–Channel-group–Bank layout.  The evaluated devices have 32 channels, a
1024-element GB, BF16 burst length 16, and an `H x 16` systolic array. GDDR6
has 16 banks/channel; LPDDR4X has 8 banks/channel and therefore joins two
channels into each 128-dimension SV physical group. Wo/W2, and Q when TP=1,
use two output tiles on LPDDR4X because its 256 banks expose 4096 output
columns per pass.

## Mapping

| Kernel | TP semantics | Physical layout | Reduction |
| --- | --- | --- | --- |
| Q | output-column parallel | enough channels to hold the local Q outputs; remaining channel groups split K | PNM across channel groups |
| K/V | output-column parallel, fused | K and V columns share one physical row region; all channel groups split K | PNM across channel groups; cache writes remain explicit `W MEM` commands |
| KQ | local attention | K is sequence-striped across the KV head's banks | none across devices |
| SV | local attention | V uses both independent 8-lane halves of `MAC_output_systolic_pim` | PNM across context groups |
| Wo | row parallel | all 512 banks produce 8192 partial outputs | one hidden-vector TP all-reduce |
| W1/W3 | output-column parallel, fused | combined output columns are tiled over all banks | no reduction split |
| W2 | row parallel | all 512 banks produce 8192 partial outputs | one hidden-vector TP all-reduce |

For Llama-70B at TP=8, Q uses 4 channels/group and 8 reduction
groups (`Kslice=1024`); fused K/V uses 1 channel/group and 32 reduction
groups (`Kslice=256`); fused W1/W3 uses 28 active channels in one tile.

The only TP CXL collectives are Wo and W2.  Their total link payload per block
is `4 * (TP-1) * B * D` BF16 values when gather and broadcast traffic are both
counted.  TP=1 therefore has zero TP-collective CXL traffic.  A pipeline-stage
handoff still moves one hidden vector when PP is greater than one; it is
independent of this mapping and identical on both sides of the comparison.

## Why SV need not stay at 50% width

The existing hardware model already exposes independent left and right
8-lane operands in `MAC_output_systolic_pim`.  The 50% utilization is therefore
caused by the old V layout, not by the 4x16/8x16 array itself.

- With an even number of local KV heads, the two halves carry two KV heads.
- At TP=8 with SA=4x16, the two halves carry two groups of four query heads and
  duplicate the corresponding V lanes.
- At TP=8 with SA=8x16, the two halves carry two context shards whose partials
  are combined by PNM.

All three modes use 16/16 lanes.  Under the specified sequence-striped K
layout, QK at TP=8 and 4K context is still 8/16 wide because each bank owns only
eight sequence positions.  That QK limitation is separate from SV.

## Trace and power accounting

Every new systolic kernel emits the migrated command protocol:
`WR_BIAS`, `WR_GB`, fill `MAC_ABK`, reduction `MAC_ABK`, drain `MAC_ABK`, and
`RD_MAC`.  A `<trace>.systolic.json` sidecar records fill/reduction/drain cycles
per kernel, utilization, physical rows, fused aliases, and PNM reduction work.

The end-to-end and microbenchmark power paths share the fitted PIM coefficients
for SA heights 1/2/4/8/16: 1.12/1.43/2.05/3.07/7.09.  Q/KV/SV reductions are
charged analytically to the shared-buffer and
instruction-buffer units.  K/V cache updates remain explicit `W MEM` commands.

The device-level shared buffer matches the cent_dev hardware point: 4 MiB
split into 32 independently accessible 128-bit banks/ports, for 512 B/cycle
aggregate I/O. `--parallel-sram` defaults to 32. Thus 32 PIM channel streams
are serviced in one SRAM-I/O round (`EXP=13`, `VEC_ADD=4`, `VEC_MUL=4` cycles at 32 channels),
instead of being serialized through the legacy 16 B/cycle path. The cent_dev
4 MiB/32-bank SRAM static and dynamic power coefficients are used as well.
FlashAttention score blocks are checked against the 4 MiB capacity.

`--pipelined-softmax` enables the separate cent_dev producer/consumer overlap
model. The full Softmax work and energy are retained, but its exposed latency
is multiplied by
`min(16 * channels * banks, sequence_length) / sequence_length` for systolic
PIM. Consequently the factor is 1 at 4K for both GDDR6 and LPDDR4X; at
32K/128K it is 1/4 and 1/16 for GDDR6, and 1/8 and 1/32 for LPDDR4X. This
option is analytical only and does not change the QK/SV Ramulator trace.

Softmax additionally includes two 16-lane `VEC_MUL` operations per
`sequence_length * local_heads / 16` element group. They model the two
normalization multiplies and are charged in both accelerator latency and
dynamic energy, using the same banked-SRAM `VEC_MUL` cycle/coefficient as the
cent_dev-compatible PNM vector path.

Element-wise multiplication follows cent_dev's `--EWMUL_PNM` behavior.
The flag moves RMSNorm, Q/K RoPE, and the multiply in `SiLU(W1) * W3` to the
PNM `VEC_MUL` path.  Systolic PIM forces this behavior on even when the flag is
not written explicitly; the CSV records both the requested and effective
values.  The imported in-array `AF`/`RD_AF` commands on W1 output tiles remain
in the trace: they form the SiLU result, while PNM performs the subsequent
element-wise multiplication with W3.  The trace variant records the effective
EWMUL mode, so incompatible legacy `--activation` results are not reused.

## PP=80, TP=1, context=4K comparison

The comparison uses SA=4x16, batch=1, the same GDDR6 Ramulator configuration,
and the same Cellar power tables.  A single block trace is simulated and then
scaled to 80 pipeline blocks.  Both imported and mapped systolic paths use the
effective EWMUL_PNM behavior while retaining W1's in-array AF/RD_AF commands.
The identical one-hidden-vector PP handoff is included on
both sides using one PCIe lane/device (144 lanes divided across 80 devices with
the existing integer allocation policy).

The report is generated rather than hard-coded: it is written as
`comparison_metrics.csv`, `comparison_metrics.png`, and
`comparison_metrics.pdf`.  This avoids mixing incompatible legacy activation
provenance or old KV-repack accounting in a single table.  The new trace is faster mainly because
fused K/V and W1/W3 remove fill, drain, GB-write, and result-drain commands.
Channel-group reductions and element-wise multiplies remain explicit analytical
PNM work, while AF/RD_AF remains a PIM trace command.  These structural
differences are why numerical agreement with the imported implementation is
neither expected nor required.

Reproduce the report with `scripts/compare_systolic_tp_traces.py`, optionally
passing both Ramulator logs to include the latency and energy comparison.
The end-to-end wrapper is `scripts/run_systolic_mapping_comparison.py`.

## SA4 all-context campaign

`scripts/run_cent_memory_cases.py --kv-head-tp-systolic` runs only SA=4x16,
batch 1/2/3/4, TP=1/2/4/8 for 7B and 70B at the 4K/32K/128K midpoint samples.
It produces 288 Ramulator jobs across GDDR6, LPDDR4X nCCD2, and LPDDR4X
nCCD6; the LPDDR timing variants share functional traces, leaving 192 unique
functional traces. No Vector job is launched.

Projection, Wo, and FFN traces use the active batch as the first SA dimension.
QK and SV retain the existing query-head row mapping and stream batches
serially. Each request owns one contiguous `max_seq_len` K block and one
contiguous V block; batches are not interleaved across banks.

`scripts/analyze_kv_head_tp_systolic_all_context.py` reads those sources and
reuses the Vector candidates from
`balanced_equal_power_all_contexts/analysis/all_candidates.csv`. It selects
PP/TP/batch/DP independently within each architecture and memory under the DGX
H100 device-side power cap. Capacity keeps the balanced campaign's logical
weight/embedding/KV admission accounting, while the TP trace, local attention,
collectives, and Device–Channel-group–Bank layout remain the systolic KV-head
implementation. `floor(resident_requests / batch)` determines resident batch
groups, and `min(groups, PP) / PP` is applied as pipeline utilization. The
equal-power figures contain six CENT bars per context (two architectures by
three memories), with DGX H100 shown as a line.

### Re-running and overwriting this campaign

The commands below rerun the same 7B/70B, GDDR6/LPDDR4X, 4K/32K/128K SA=4x16
campaign into the existing
`output/kv_head_tp_systolic_all_context/raw/systolic_4x16` directory. They
reuse valid Ramulator traces/logs and refresh CSV post-processing; this is
sufficient for an accelerator-only accounting change such as Softmax
`VEC_MUL`.

```bash
cd /home/linuswang/Documents/CENT/cent_simulation
MPLCONFIGDIR=/tmp/cent_mpl /home/linuswang/miniforge3/envs/cent/bin/python \
  scripts/run_cent_memory_cases.py --kv-head-tp-systolic \
  --models Llama2-7B,Llama2-70B \
  --cases GDDR6,LPDDR4X_nCCD2,LPDDR4X_nCCD6 \
  --EWMUL_PNM --flash-attention --flash-attention-block-size 1024

MPLCONFIGDIR=/tmp/cent_mpl /home/linuswang/miniforge3/envs/cent/bin/python \
  scripts/analyze_kv_head_tp_systolic_all_context.py \
  --flash-attention --flash-attention-block-size 1024
```

`update_csv` retains rows with explicitly incompatible legacy `--activation`
provenance for audit. The analyzer excludes them, so the above refresh
overwrites the effective campaign and analysis. To physically replace the
three raw result CSVs too, remove exactly the following files before the first
command; traces and Ramulator logs remain intact and are reused.

```bash
rm -f \
  output/kv_head_tp_systolic_all_context/raw/systolic_4x16/GDDR6/simulation_results_decode_only_long_context_midpoint.csv \
  output/kv_head_tp_systolic_all_context/raw/systolic_4x16/LPDDR4X/simulation_results_decode_only_long_context_midpoint_nCCD2.csv \
  output/kv_head_tp_systolic_all_context/raw/systolic_4x16/LPDDR4X/simulation_results_decode_only_long_context_midpoint_nCCD6.csv
```

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
| K/V | output-column parallel, fused | K and V columns share one physical row region; all channel groups split K | PNM across channel groups, then cache repack |
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
for SA heights 1/2/4/8/16: 1.12/1.43/2.05/3.07/7.09.  PNM cache repack,
Q/KV/SV reductions, and fused `SiLU(W1) * W3` are charged analytically to the
shared-buffer, instruction-buffer, EXP, and vector units.

## PP=80, TP=1, context=4K comparison

The comparison uses SA=4x16, batch=1, the same GDDR6 Ramulator configuration,
and the same Cellar power tables.  A single block trace is simulated and then
scaled to 80 pipeline blocks.  The identical one-hidden-vector PP handoff is
included on both sides using one PCIe lane/device (144 lanes divided across 80
devices with the existing integer allocation policy).

| Metric | Imported implementation | Standard TP mapping | Delta |
| --- | ---: | ---: | ---: |
| trace commands/block | 37,773 | 36,006 | -4.68% |
| `MAC_ABK` cycles/block | 118,472 | 109,040 | -7.96% |
| latency/block including PP handoff | 0.303731 ms | 0.290219 ms | -4.45% |
| serial 80-block token latency | 24.2985 ms | 23.2176 ms | -4.45% |
| energy/block including PP handoff | 16.2841 mJ | 15.9401 mJ | -2.11% |
| 80-block energy/token | 1302.73 mJ | 1275.21 mJ | -2.11% |
| steady 80-stage power | 4289.08 W | 4393.93 W | +2.44% |

The new trace is faster mainly because fused K/V and W1/W3 remove fill, drain,
GB-write, and result-drain commands.  Its accelerator overhead is higher
(0.039593 ms versus 0.030567 ms/block) because channel-group reductions, cache
repack, and fused activation are now explicit PNM work.  Energy falls less than
latency, so the fully occupied 80-stage steady-state power rises slightly even
though energy per token falls.  These structural differences are why numerical
agreement with the imported implementation is neither expected nor required.

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

# CENT PP/TP/DP sweep

This directory contains a capacity-constrained PP/TP/DP sweep for BF16
Llama2-7B and Llama2-70B at 4K, 32K, and 128K context.

Assumptions:

- 16 GiB per PIM card, zero additional reserve.
- At most 32 pipeline stages.
- `DP` means independent model replicas, so total cards = `PP * TP * DP`.
- Existing per-TP Ramulator traces supply full-model serial token latency and
  energy. New PP values are reconstructed analytically; they are not new
  Ramulator runs.
- The primary mapping, `k_sharded_v_local`, matches the current no-inter-device
  code: K/QK is striped over TP devices, while V/PV remains on the master.
- Discrete layer imbalance is included. For 70B PP=32, effective saturated PP
  is `80 / ceil(80/32) = 26.67`, not 32.

## Fixed-card winners: current K-sharded/V-local mapping

| Memory/cards | Model | Context | PP | TP | DP | Throughput (tok/s) | Power (W) |
|---|---|---:|---:|---:|---:|---:|---:|
| GDDR6/128 | 7B | 4K | 1 | 1 | 128 | 42006.55 | 4080.27 |
| GDDR6/128 | 7B | 32K | 16 | 1 | 8 | 7394.94 | 2289.52 |
| GDDR6/128 | 7B | 128K | 32 | 1 | 4 | 455.90 | 473.26 |
| GDDR6/128 | 70B | 4K | 16 | 1 | 8 | 5172.29 | 4753.29 |
| GDDR6/128 | 70B | 32K | 32 | 1 | 4 | 1015.81 | 2628.52 |
| GDDR6/128 | 70B | 128K | 32 | 1 | 4 | 73.65 | 611.92 |
| LPDDR4X nCCD2/256 | 7B | 4K | 1 | 1 | 256 | 40233.18 | 3508.35 |
| LPDDR4X nCCD2/256 | 7B | 32K | 16 | 1 | 16 | 8235.77 | 2301.45 |
| LPDDR4X nCCD2/256 | 7B | 128K | 32 | 1 | 8 | 522.61 | 490.29 |
| LPDDR4X nCCD2/256 | 70B | 4K | 16 | 1 | 16 | 4489.52 | 3818.49 |
| LPDDR4X nCCD2/256 | 70B | 32K | 32 | 1 | 8 | 1017.31 | 2439.97 |
| LPDDR4X nCCD2/256 | 70B | 128K | 16 | 2 | 8 | 78.71 | 514.94 |

For 7B/4K, multiple TP1 PP/DP factorizations tie in this model. For 7B/32K,
PP16/TP1 and PP32/TP1 also tie. The table uses one representative winner.

## GDDR6 128-card, 70B/128K PP32 comparison

| PP | TP | DP | Master resident requests | Aggregate throughput (tok/s) |
|---:|---:|---:|---:|---:|
| 32 | 1 | 4 | 7 | **73.65** |
| 32 | 2 | 2 | 11 | 64.78 |
| 32 | 4 | 1 | 15 | 46.96 |

TP only weakly accelerates the master-attention critical path, while every TP
doubling halves DP. It also reduces master KV from `S` only to
`S * (1 + 1/TP) / 2`, because the complete V cache remains on the master.

## Mapping sensitivity at 70B/128K

- If K and V are both genuinely TP-sharded, GDDR6/128 changes to
  PP16/TP2/DP4 at 106.01 tok/s; LPDDR4X/256 changes to PP16/TP2/DP8 at
  118.06 tok/s.
- If K and V are both master-local, GDDR6/128 stays PP32/TP1/DP4 at
  73.65 tok/s; LPDDR4X/256 becomes PP32/TP1/DP8 at 76.45 tok/s.

The non-primary mappings in the CSV are capacity sensitivities only. Their
latency and energy still come from the current master trace.

## Equal-power caveat

DGX H100 budgets are 4652 W for 7B and 4450.4 W for 70B. GDDR6/128 at
70B/4K exceeds the latter by 6.8%; the best layout under both a 128-card cap
and the DGX power limit is PP12/TP1/DP10 (120 cards), 4618.11 tok/s at
4244.01 W. Other fixed-card winners are below their DGX power target.

If card count is unbounded and the current zero-idle-power model is used
literally, the equal-power optimizer selects thousands of cards at long
context. Those rows are retained as diagnostics, not hardware recommendations.
They require an idle/background/controller/CXL/leakage power model before they
can define a credible equal-power card count.

TP>1 rows have two further optimistic omissions in the current simulator:

- helper-device K/QK energy is not included in the FC-only worker energy;
- distributed QK scores are not gathered back to the master in CXL latency.

Therefore the TP1 fixed-card winners are robust, while the narrow LPDDR4X
70B/128K TP2 lead should be treated as provisional.

Files:

- `parallelism_sweep_best.csv`: best layout for every memory/model/context,
  mapping, and deployment mode.
- `parallelism_sweep_all_candidates.csv`: all feasible candidates.
- `simulator_ideal/`: sensitivity using the original ideal PP multiplier.

Regenerate with:

```bash
python3 cent_simulation/scripts/sweep_equal_power_parallelism.py
```

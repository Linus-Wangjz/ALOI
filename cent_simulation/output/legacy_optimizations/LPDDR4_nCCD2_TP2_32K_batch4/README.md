# Retained legacy single-buffer results

This archive contains the corrected LPDDR4 baseline/full comparison for
Llama2-70B: FP8, SA4×32, TP2, batch4, a 32,768-token context window, and a
30,976-token active decode sample. Timing uses 32 LPDDR4 channels, each x16,
prefetch16, a 32-byte burst, nBL=8, nCCD=2, and a 625 ps DRAM clock.
The power-model label in the historical files is LPDDR4X.

These results use the legacy **single-buffer `AiM` backend**. PIM totals come
from simulation; per-kernel issued-gap attribution includes waiting. PNM, CXL,
and pipelined Softmax exposure use analytical formulas. Residual/EWADD
arithmetic remains unmodeled. Totals cover one Transformer block and one
batch group on a representative TP rank, without multiplying by TP or layers.

| Variant | EWMUL_PNM | FlashAttention | Pipelined Softmax | Channel concurrency | PIM cycles | Block (µs) |
| --- | --- | --- | --- | --- | ---: | ---: |
| baseline | ON | OFF | OFF | OFF | 713,214 | 576.119736 |
| full | ON | ON, block1024 | ON | ON | 564,405 | 411.070003 |

Systolic execution forces EWMUL_PNM on, so baseline is not an all-four-off run.
Full optimizations reduce block latency by 28.65% against this baseline.

- [summary.txt](summary.txt) compares both runs and embeds their complete
  per-kernel cycle details. [comparison.csv](comparison.csv) contains totals.
- [baseline/summary.txt](baseline/summary.txt) and
  [full/summary.txt](full/summary.txt) are the stable individual reports.
  Their `breakdown.json`, `breakdown.csv`, and `run_manifest.json` links resolve
  to the retained content-addressed report directories inside this archive.
- Each variant's fingerprinted directory contains the source CSV, simulator
  log, all 32 channel command records, backend configuration, resolved timing,
  provenance, and source/binary hashes. `BUILD_MANIFEST.json` and
  `RUN_PROVENANCE.md` describe the original run, not a new simulation performed
  when this archive was committed.
- `interface_correction_*` and `kernel_interface_correction.csv` preserve the
  before/after comparison with the superseded width32/prefetch8/nBL4 interface.
  Those older metrics are historical, not the current reference totals.

All existing output files are retained, including the raw channel records
(about 279 MiB uncompressed in total). Reports can be read directly after
cloning. Original commands/configurations preserve absolute paths from the
run machine; the generated input traces and simulator binaries are outside
this output archive. To regenerate them and run the comparison from a fresh
checkout, build AiM using the root README, then run from the repository root:

```bash
cd cent_simulation
python scripts/run_legacy_optimization_case.py --variants baseline full
```

The runner generates paths for the current checkout and fingerprints the
current sources and binaries. A rerun may therefore create different build
directories. The separate double-buffer experiment is documented in
[the committed double-buffer comparison](../../../results/double_buffering_fp8_32k_batch4/summary.txt)
on the `double-buffering` branch.

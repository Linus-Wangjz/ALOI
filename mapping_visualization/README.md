# Llama 3.1 70B Systolic-PIM Mapping Explorer

This is a dependency-free interactive visualization of the current KV-head TP
mapping. It focuses on one representative decoder block, fixed batch 4,
TP 1/2/4/8, 32K/128K contexts, and two precision geometries: BF16 with a
4x16 systolic array, or FP8 with a 4x32 array. BF16 supports GDDR6 and
LPDDR4X; FP8 is restricted to the simulator-validated LPDDR4X mapping.

The **Channel mapping** control compares two intra-device policies without
changing TP ownership:

- **Channel Group Mapping** assigns output-column tiles to channel groups and
  uses additional groups for reduction slices when required.
- **Reduction-Split Mapping** gives every channel the complete output-column
  range, divides K across C0-C31, and locally reduces the 32 channel partials.
  Wo and W2 still perform their existing TP all-reduce after this local step.

Open `index.html` directly, or serve this directory locally:

```bash
cd mapping_visualization
python3 -m http.server 8000
```

Then visit `http://localhost:8000`.

The interaction has three levels:

1. Select a kernel in the Transformer Block.
2. Inspect a KV-aware 3D logical tensor (or a 2D FFN matrix) and how its
   KV-head/group depth is partitioned across TP devices.
3. Select a device and inspect the thin `[4 x K]` runtime matrix against the
   native `[K rows x N columns]` weight layout. QK and SV instead show four
   independent batch-cache cubes with Batch 0 expanded. Matching colors connect
   tensor slices, channel groups and physical bank rows. The selected mapping
   reports both the full selected tile and its per-channel C0 tile. Every kernel
   expands C0 into configuration-dependent N waves and K chunks, then divides
   each wave across the channel banks. In Reduction-Split Mapping the other 31
   channels run the same schedule in lockstep and their partials are combined
   locally. The fourth interactive phase is attached to this wave panel and
   animates the schedule through the selected systolic array. Click the
   highlighted matrix tile to play all four mapping stages, or click a stage to
   replay it independently.

The third level also reports a critical-path cycle estimate. GB fill cycles use
a 32-bit/cycle write path and count only K-chunk changes; N waves reuse the
existing GB payload. SA cycles use the reduction length directly and omit
pipeline startup/drain overhead. Dense projections place the four batches in
the four SA rows together. QK/SV instead keep batch caches independent: one SA
launch packs four GQA heads from one batch, so eight GQA heads execute as two
waves and batch 4 contributes four temporal rounds. Local KV heads are spatial
in Channel Group Mapping and temporal in the current Reduction-Split Mapping.
Inter-channel reduction cards count each channel-output set separately and
then count pairwise channel merges after local K-chunk accumulation; TP
all-reduce remains a separate operation.

For SV, one physical wave may contain multiple parallel K-parts. GDDR6 BF16
uses `2 x 128` K values and LPDDR4X FP8 uses `2 x 256`; therefore a selected
`[1024 x 128]` C0 tile takes four and two physical K waves respectively. The
complete thin GEMM operand contains all eight GQA heads, while GB and the SA
show one four-head GQA wave at a time. For FP8, the `4 x 32` array is drawn as
two `4 x 16` halves. Each 16-element V slice fetched by a DRAM bank broadcasts
across the four GQA rows; the two halves can consume distinct K chunks only
when they cover the same output columns. When the halves instead cover two N
slices, they share one K stream and do not double reduction throughput.

The active second- or third-level SVG can be exported using **Export SVG**.
Configuration and navigation state are encoded in the URL query string when
the browser permits history updates.

# Llama 3.1 70B Systolic-PIM Mapping Explorer

This is a dependency-free interactive visualization of the current KV-head TP
mapping. It focuses on one representative decoder block, TP 1/2/4/8, 32K/128K
contexts, and GDDR6/LPDDR4X devices with a 4x16 systolic array.

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
3. Select a device and inspect the horizontal GEMV vector against the displayed
   transposed matrix. Matching colors connect tensor slices, channel groups and
   physical bank rows. Click the highlighted matrix tile to play all four
   mapping stages, or click a stage at the bottom to replay bank placement,
   vector-to-GB loading, or the diagonal SA4x16 stream independently.

The active second- or third-level SVG can be exported using **Export SVG**.
Configuration and navigation state are encoded in the URL query string when
the browser permits history updates.

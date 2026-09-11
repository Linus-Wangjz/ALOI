# Llama 3.1 70B Systolic-PIM Mapping Explorer

This is a dependency-free interactive visualization of the current KV-head TP
mapping. It focuses on one representative decoder block, TP 1/2/4/8, 32K/128K
contexts, and GDDR6/LPDDR4X devices with a 4x16 systolic array.

Open `index.html` directly, or serve this directory locally:

```bash
cd mapping_explorer
python3 -m http.server 8000
```

Then visit `http://localhost:8000`.

The interaction has three levels:

1. Select a kernel in the Transformer Block.
2. Inspect how its global matrix/cache is partitioned across TP devices.
3. Select a device, click the highlighted local tile, and inspect the
   channel/bank/row and runtime GB/SA mapping.

The active second- or third-level SVG can be exported using **Export SVG**.
Configuration and navigation state are encoded in the URL query string when
the browser permits history updates.

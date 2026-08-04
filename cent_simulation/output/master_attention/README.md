# Master-attention CENT results

These decode-only results reproduce the paper's tensor-parallel mapping:

- fully connected layers are tensor-sharded across the TP group;
- KV-cache attention, normalization, and residual operations remain on the
  TP-group master device;
- `--inter-device-attention` is deliberately not supplied.

The run covers Llama2-7B and Llama2-70B at 4K, 32K, and 128K context
windows.  Each CSV has 36 rows: 15 Llama2-7B configurations and 21
Llama2-70B configurations.

Reproduction command:

```bash
python3 cent_simulation/scripts/run_cent_memory_cases.py \
  --paper-long-context --master-attention \
  --output-root /home/linuswang/Documents/CENT/cent_simulation/output/master_attention \
  --trace-root /home/linuswang/Documents/CENT/cent_simulation/trace/master_attention \
  --trace-workers 1 --run-workers 8
```

The matching generated traces are under `cent_simulation/trace/master_attention/`.

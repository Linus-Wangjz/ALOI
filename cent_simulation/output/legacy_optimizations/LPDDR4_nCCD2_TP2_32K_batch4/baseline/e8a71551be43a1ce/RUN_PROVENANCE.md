# FP8 LPDDR4 nCCD2 TP2 batch4 optimization comparison

Window=32768; active decode length=30976; array=4x32; 32 channels; EWMUL_PNM=true.
Corrected LPDDR4 interface: x16/channel; prefetch16; 32 B/burst; nBL8; 625 ps/clock; internal MAC nCCD2.
Supersedes width32/prefetch8/nBL4 results. QK query reloads and all host trace work are unchanged.
FP8 geometry and legacy analytical formulas restored from ALOI 637537c.
PNM/CXL and streamed Softmax remain legacy analytical models; FlashAttention is a timing-trace abstraction.
Concurrent mode accepts at most one request/channel/cycle; DRAM timing and SYNC/EOC remain active.
WR_GB retains multicast semantics. Native full-run PIM cycles are simulated; kernel attribution uses issued gaps.
Reproduce: python scripts/run_legacy_optimization_case.py
BUILD_MANIFEST.json SHA256: b26edb985e3a9c89d1a20f682c930f77df833af4480e09dbc3850488139a9d47

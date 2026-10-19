"""Shared measured scaling coefficients for the imported systolic PIM arrays."""


# Relative PIM command energy for an H-row, 256-bit-wide systolic array.  The
# available measurements are BF16; FP8 functional runs deliberately reuse
# the fixed-width coefficient until an FP8 circuit-level table is available.
SYSTOLIC_PIM_POWER_ASSUMPTION = "iso_256b_bf16_calibration"


SYSTOLIC_PIM_POWER_SCALING = {
    1: 1.12,
    2: 1.43,
    4: 2.05,
    8: 3.07,
    16: 7.09,
}

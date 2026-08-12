"""Shared measured scaling coefficients for the imported systolic PIM arrays."""


# Relative PIM command energy for an H x 16 systolic array.  Keep this table
# shared by the end-to-end driver and the microbenchmark power reporter so a
# simulated array shape always uses the same fitted coefficient.
SYSTOLIC_PIM_POWER_SCALING = {
    1: 1.12,
    2: 1.43,
    4: 2.05,
    8: 3.07,
    16: 7.09,
}

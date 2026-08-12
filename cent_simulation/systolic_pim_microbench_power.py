import argparse

from cent_power_calculator import command_processor, power_calculator
from systolic_power import SYSTOLIC_PIM_POWER_SCALING


def parse_args():
    parser = argparse.ArgumentParser(description="Report systolic-PIM microbenchmark power")
    parser.add_argument("mlog", help="Ramulator log")
    parser.add_argument("sa_size", type=int, choices=sorted(SYSTOLIC_PIM_POWER_SCALING))
    parser.add_argument("--timing", help="Matching DRAMTimingExporter YAML")
    return parser.parse_args()


def main():
    args = parse_args()
    stat = command_processor(args.mlog, args.timing)
    energy, _ = power_calculator(stat, 0, 0, 0, 0, 1)
    energy["PIM"] *= SYSTOLIC_PIM_POWER_SCALING[args.sa_size]
    total_power = sum(energy.values()) / stat["latency"]
    print("Device Power", total_power, "W", "Latency", stat["latency"], "ms")


if __name__ == "__main__":
    main()

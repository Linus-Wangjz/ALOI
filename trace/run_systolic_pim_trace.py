#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CENT_SIM = REPO_ROOT / "cent_simulation"
sys.path.insert(0, str(CENT_SIM))

from run_sim import write_ramulator_config  # noqa: E402


def run_trace(trace_path, config_template):
    trace_path = trace_path.resolve()
    log_path = Path(f"{trace_path}.log")
    base_path = Path(str(log_path)[:-4])
    config_path = Path(f"{base_path}.ramulator.yaml")
    timing_path = Path(f"{base_path}.timing.yaml")
    command_prefix = Path(f"{base_path}.cmd")

    write_ramulator_config(
        config_template,
        config_path,
        timing_path,
        command_prefix,
        include_command_trace=False,
    )
    with log_path.open("w") as log_file:
        subprocess.run(
            [
                str(REPO_ROOT / "aim_simulator" / "build" / "ramulator2"),
                "-f",
                str(config_path),
                "-t",
                str(trace_path),
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Run systolic-PIM microbenchmark traces")
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "aim_simulator" / "test" / "example_GDDR6.yaml",
    )
    args = parser.parse_args()
    for trace_path in args.traces:
        run_trace(trace_path, args.config.resolve())


if __name__ == "__main__":
    main()

import sys


with open(sys.argv[1], "r") as file:
    lines = file.readlines()

log = ""
total_cycles = -1
total_idle_cycles = -1
total_active_cycles = -1
total_precharged_cycles = -1
commands = [
    "ACT", "PREA", "PRE", "RD", "WR", "RDA", "WRA", "REFab", "REFpb",
    "ACT4", "ACT16", "PRE4", "MAC", "MAC16", "AF16", "EWMUL16", "RDCP",
    "WRCP", "WRGB", "RDMAC16", "RDAF16", "WRMAC16", "WRA16", "TMOD",
    "SYNC", "EOC", "ACT-1", "ACT-2", "CASRD", "CASWR", "CASWRGB",
    "CASWRMAC8", "CASRDMAC8", "CASWRA8", "RFMab", "RFMpb", "ACT4-1",
    "ACT8-1", "ACT4-2", "ACT8-2", "MAC8", "AF8", "EWMUL8", "RDMAC8",
    "RDAF8", "WRMAC8", "WRA8",
]
command_count = {command: 0 for command in commands}


def print_result():
    print(
        f"{log}\t{total_cycles / 2000000.0}\t"
        f"{total_active_cycles / 32.0 / 2000000.0}\t"
        f"{total_precharged_cycles / 32.0 / 2000000.0}\t"
        f"{100.0 - (total_idle_cycles / 32.0 / total_cycles) * 100.0}\t",
        end="",
    )
    for command in commands:
        print(f"{command_count[command]}\t", end="")
    print()


for line in lines:
    if "Processing" in line:
        if total_cycles != -1:
            print_result()
            command_count = {command: 0 for command in commands}
            total_cycles = -1
            total_idle_cycles = -1
            total_active_cycles = -1
            total_precharged_cycles = -1
        log = line.split()[1]

    if "memory_system_cycles" in line:
        total_cycles = int(line.split()[1])
    if "idle_cycles" in line:
        total_idle_cycles = int(line.split()[1]) if total_idle_cycles == -1 else total_idle_cycles + int(line.split()[1])
    if "active_cycles" in line:
        total_active_cycles = int(line.split()[1]) if total_active_cycles == -1 else total_active_cycles + int(line.split()[1])
    if "precharged_cycles" in line:
        total_precharged_cycles = int(line.split()[1]) if total_precharged_cycles == -1 else total_precharged_cycles + int(line.split()[1])
    for command in commands:
        if "num_" + command + "_commands" in line:
            command_count[command] += int(line.split()[1])

if total_cycles != -1:
    print_result()

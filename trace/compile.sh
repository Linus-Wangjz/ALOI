commands=(
    ACT PREA PRE RD WR RDA WRA REFab REFpb ACT4 ACT16 PRE4 MAC MAC16 AF16
    EWMUL16 RDCP WRCP WRGB RDMAC16 RDAF16 WRMAC16 WRA16 TMOD SYNC EOC
    ACT-1 ACT-2 CASRD CASWR CASWRGB CASWRMAC8 CASRDMAC8 CASWRA8 RFMab RFMpb
    ACT4-1 ACT8-1 ACT4-2 ACT8-2 MAC8 AF8 EWMUL8 RDMAC8 RDAF8 WRMAC8 WRA8
)

for f in trace*.log
do
    [ -e "$f" ] || continue
    echo "Processing $f file..."
    tail -n 10000 "$f" > logs.txt
    grep "memory_system_cycles" logs.txt
    grep "idle_cycles" logs.txt
    grep "precharged_cycles" logs.txt
    grep "active_cycles" logs.txt
    for command in "${commands[@]}"
    do
        grep "num_${command}_commands" logs.txt
    done
done

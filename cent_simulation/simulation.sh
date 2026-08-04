# Pipeline Parallel
threads=${1:-8}
seqlen_gap=${2:-4096}
num_banks=${3:-16}
num_channels=32
decode_only="--decode-only"
if [ "$num_banks" -eq 8 ]; then
    experiment="LPDDR4X"
    ramulator_config="../aim_simulator/test/example_LPDDR4.yaml"
    dram_power_impl="LPDDR4X"
else
    experiment="GDDR6"
    ramulator_config="../aim_simulator/test/example_GDDR6.yaml"
    dram_power_impl="GDDR6"
fi
output_dir="output/${experiment}"
trace_root="trace/${experiment}"
log_root="${output_dir}/ramulator"
simulation_result_path="${output_dir}/simulation_results_decode_only.csv"

python3 run_sim.py --num_channels $num_channels --num_banks $num_banks --experiment $experiment --trace-root $trace_root --log-root $log_root --ramulator-config $ramulator_config --dram-power-impl $dram_power_impl --simulation_result_path $simulation_result_path --model Llama2-7B --generate_trace --simulate_trace --update_csv --num_devices 8 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap $seqlen_gap $decode_only
python3 run_sim.py --num_channels $num_channels --num_banks $num_banks --experiment $experiment --trace-root $trace_root --log-root $log_root --ramulator-config $ramulator_config --dram-power-impl $dram_power_impl --simulation_result_path $simulation_result_path --model Llama2-70B --generate_trace --simulate_trace --update_csv --num_devices 32 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap $seqlen_gap $decode_only

# Model Parallel
python3 run_sim.py --num_channels $num_channels --num_banks $num_banks --experiment $experiment --trace-root $trace_root --log-root $log_root --ramulator-config $ramulator_config --dram-power-impl $dram_power_impl --simulation_result_path $simulation_result_path --model Llama2-7B --model_parallel --generate_trace --simulate_trace --update_csv --num_devices 8 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap $seqlen_gap $decode_only
python3 run_sim.py --num_channels $num_channels --num_banks $num_banks --experiment $experiment --trace-root $trace_root --log-root $log_root --ramulator-config $ramulator_config --dram-power-impl $dram_power_impl --simulation_result_path $simulation_result_path --model Llama2-70B --model_parallel --generate_trace --simulate_trace --update_csv --num_devices 32 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap $seqlen_gap $decode_only

# Long Context
python3 run_sim.py --num_channels $num_channels --num_banks $num_banks --experiment $experiment --trace-root $trace_root --log-root $log_root --ramulator-config $ramulator_config --dram-power-impl $dram_power_impl --simulation_result_path $simulation_result_path --model Llama2-70B --generate_trace --simulate_trace --update_csv --num_devices 32 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen 4096 32768 131072 $decode_only
python3 run_sim.py --num_channels $num_channels --num_banks $num_banks --experiment $experiment --trace-root $trace_root --log-root $log_root --ramulator-config $ramulator_config --dram-power-impl $dram_power_impl --simulation_result_path $simulation_result_path --model Llama2-70B --model_parallel --inter-device-attention --generate_trace --simulate_trace --update_csv --num_devices 32 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen 4096 32768 131072 $decode_only

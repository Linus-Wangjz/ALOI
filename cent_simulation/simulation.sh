# Pipeline Parallel
threads=$1
seqlen_gap=$2
num_banks=${3:-16}
num_channels=32
result_suffix="${num_channels}_channels_${num_banks}_banks_per_device"
simulation_result_path="simulation_results_${result_suffix}.csv"
long_context_result_path="simulation_results_long_context_${result_suffix}.csv"

python3 run_sim.py --num_channels $num_channels --num_banks $num_banks --simulation_result_path $simulation_result_path --model Llama2-7B --generate_trace --simulate_trace --process_results --update_csv --num_devices 8 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap $seqlen_gap
python3 run_sim.py --num_channels $num_channels --num_banks $num_banks --simulation_result_path $simulation_result_path --model Llama2-13B --generate_trace --simulate_trace --process_results --update_csv --num_devices 20 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap $seqlen_gap
python3 run_sim.py --num_channels $num_channels --num_banks $num_banks --simulation_result_path $simulation_result_path --model Llama2-70B --generate_trace --simulate_trace --process_results --update_csv --num_devices 32 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap $seqlen_gap

# Model Parallel
python3 run_sim.py --num_channels $num_channels --num_banks $num_banks --simulation_result_path $simulation_result_path --model Llama2-7B --model_parallel --generate_trace --simulate_trace --process_results --update_csv --num_devices 8 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap $seqlen_gap
python3 run_sim.py --num_channels $num_channels --num_banks $num_banks --simulation_result_path $simulation_result_path --model Llama2-13B --model_parallel --generate_trace --simulate_trace --process_results --update_csv --num_devices 20 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap $seqlen_gap
python3 run_sim.py --num_channels $num_channels --num_banks $num_banks --simulation_result_path $simulation_result_path --model Llama2-70B --model_parallel --generate_trace --simulate_trace --process_results --update_csv --num_devices 32 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap $seqlen_gap

# Long Context
python3 run_sim.py --num_channels $num_channels --num_banks $num_banks --simulation_result_path $long_context_result_path --model Llama2-70B --generate_trace --simulate_trace --process_results --update_csv --num_devices 32 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen 2304 6400 14592 30976
python3 run_sim.py --num_channels $num_channels --num_banks $num_banks --simulation_result_path $long_context_result_path --model Llama2-70B --model_parallel --inter-device-attention --generate_trace --simulate_trace --process_results --update_csv --num_devices 32 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen 2304 6400 14592 30976

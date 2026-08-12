threads=${1:-8}
skip_to_update=${2:-False}

# seqlen="512 1024 2048 4096"
# seqlen="256 2304"
# seqlen="256 1024 4096"
seqlen="2048"

# Mixtral-8x7B

# MoE model explores various channels_per_block configurations, thereby no need to simulate split_decoder_across_devices and not split_decoder_across_devices scenarios separately, no need to explore data parallel in DREAM.

if [ "$skip_to_update" != "True" ]; then

# CENT Baseline

# 64 Devices

for data_parallel in {1..16}
do
for devices in 48 56 64 72 80 88 96 104 108 112 120 128
do
python3 run_sim.py --model Mixtral-8x7B --generate_trace --simulate_trace --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices $devices --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads
python3 run_sim.py --model Mixtral-8x7B --generate_trace --simulate_trace --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices $devices --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --split_decoder_across_devices
done
done

python3 run_sim.py --model Mixtral-8x7B --process_results --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices 120 --data_parallel 3 --seqlen $seqlen --run_simulation_max_workers 1 --generate_trace_max_workers 1 --density 8Gb --split_decoder_across_devices


# Systolic PIM

# 64 Devices

for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
python3 run_sim.py --model Mixtral-8x7B --MoE_minimal_channels --generate_trace --simulate_trace --process_results --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices 128 --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density 32Gb --split_decoder_across_devices
done
done

fi

# 5000 W Power

# CENT

for density in 8Gb 16Gb 24Gb
do
for data_parallel in {1..16}
do
for devices in 48 56 64 72 80 88 96 104 108 112 120 128
do
python3 run_sim.py --model Mixtral-8x7B --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_${density}.csv --density ${density} --power_limit 5600
done
done
done

for density in 8Gb 16Gb 24Gb
do
for data_parallel in {1..16}
do
for devices in 48 56 64 72 80 88 96 104 108 112 120 128
do
python3 run_sim.py --model Mixtral-8x7B --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_${density}.csv --density ${density} --split_decoder_across_devices --power_limit 5600
done
done
done

# DREAM

for density in 8Gb 16Gb 24Gb
do
for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
for devices in 48 56 64 72 80 88 96 104 108 112 120 128
do
python3 run_sim.py --model Mixtral-8x7B --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_${density}.csv --density ${density} --power_limit 5600
done
done
done
done

for density in 8Gb 16Gb 24Gb
do
for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
for devices in 48 56 64 72 80 88 96 104 108 112 120 128
do
python3 run_sim.py --model Mixtral-8x7B --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_${density}.csv --density ${density} --split_decoder_across_devices --power_limit 5600
done
done
done
done

python3 run_sim.py --model Mixtral-8x7B --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim 1 --num_devices 128 --data_parallel 1 --seqlen 2048 --run_simulation_max_workers 8 --generate_trace_max_workers 8 --simulation_result_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_8Gb.csv --density 8Gb --split_decoder_across_devices --power_limit 5600


# 2500 W Power (DP=2)

seqlen="2048"
threads=8

for density in 8Gb 16Gb 24Gb
do
for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
for devices in 32 48
do
python3 run_sim.py --model Mixtral-8x7B --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_${density}_2800W.csv --density ${density} --split_decoder_across_devices --power_limit 2500
done
done
done
done

for density in 8Gb 16Gb 24Gb
do
for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
for devices in 32 48
do
python3 run_sim.py --model Mixtral-8x7B --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_${density}_2800W.csv --density ${density} --power_limit 2500
done
done
done
done

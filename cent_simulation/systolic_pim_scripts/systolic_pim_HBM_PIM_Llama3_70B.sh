threads=${1:-8}

# 16Gb chip

for seqlen in 256 1024 4096
do

for devices in 64 96 128
do

for data_parallel in {1..8}
do
python3 run_sim.py --model Llama3-70B --generate_trace --simulate_trace --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density 16Gb --split_decoder_across_devices
done

for data_parallel in {1..8}
do
for systolic_dim in 1 2 4 8
do
python3 run_sim.py --model Llama3-70B --generate_trace --simulate_trace --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density 16Gb --split_decoder_across_devices
done
done

done
done

python3 run_sim.py --model Llama3-70B --process_results --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim 1 --num_devices 64 --data_parallel 1 --seqlen 256 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density 16Gb --split_decoder_across_devices


# 16Gb chip

# Split decoder across devices

for seqlen in 256 1024 4096
do

for density in 8Gb 16Gb
do

for devices in 64 96 128
do

for data_parallel in {1..8}
do
python3 run_sim.py --model Llama3-70B --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path DREAM_simulation_results/simulation_results_HBM_PIM_comparison_Llama3_70B_${seqlen}.csv --split_decoder_across_devices --power_limit 5600
done

for data_parallel in {1..8}
do
for systolic_dim in 1 2 4 8
do
python3 run_sim.py --model Llama3-70B --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path DREAM_simulation_results/simulation_results_HBM_PIM_comparison_Llama3_70B_${seqlen}.csv  --split_decoder_across_devices --power_limit 5600
done
done


# Not split decoder across devices

for data_parallel in {1..8}
do
python3 run_sim.py --model Llama3-70B --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path DREAM_simulation_results/simulation_results_HBM_PIM_comparison_Llama3_70B_${seqlen}.csv --power_limit 5600
done

for data_parallel in {1..8}
do
for systolic_dim in 1 2 4 8
do
python3 run_sim.py --model Llama3-70B --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path DREAM_simulation_results/simulation_results_HBM_PIM_comparison_Llama3_70B_${seqlen}.csv --power_limit 5600
done
done

done

done

done








# 16Gb chip

for seqlen in 256 1024 4096
do

for devices in 64
do

for data_parallel in 1
do
python3 run_sim.py --model Mixtral-8x7B --MoE_minimal_channels --generate_trace --simulate_trace --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density 16Gb --split_decoder_across_devices
done

for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
python3 run_sim.py --model Mixtral-8x7B --MoE_minimal_channels --generate_trace --simulate_trace --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density 16Gb --split_decoder_across_devices
done
done

done
done

python3 run_sim.py --model Mixtral-8x7B --MoE_minimal_channels --process_results --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim 1 --num_devices 64 --data_parallel 1 --seqlen 256 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density 16Gb --split_decoder_across_devices


# 16Gb chip

# Split decoder across devices

for seqlen in 256 1024 4096
do

for density in 16Gb
do

for devices in 64
do

for data_parallel in 1
do
python3 run_sim.py --model Mixtral-8x7B --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path DREAM_simulation_results/simulation_results_HBM_PIM_comparison_Mixtral_8x7B_${seqlen}.csv --split_decoder_across_devices
done

for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
python3 run_sim.py --model Mixtral-8x7B --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path DREAM_simulation_results/simulation_results_HBM_PIM_comparison_Mixtral_8x7B_${seqlen}.csv  --split_decoder_across_devices
done
done


# Not split decoder across devices

for data_parallel in 1
do
python3 run_sim.py --model Mixtral-8x7B --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path DREAM_simulation_results/simulation_results_HBM_PIM_comparison_Mixtral_8x7B_${seqlen}.csv
done

for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
python3 run_sim.py --model Mixtral-8x7B --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path DREAM_simulation_results/simulation_results_HBM_PIM_comparison_Mixtral_8x7B_${seqlen}.csv
done
done

done

done

done

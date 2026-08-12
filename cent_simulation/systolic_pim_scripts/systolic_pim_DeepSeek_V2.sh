threads=${1:-8}
skip_to_update=${2:-False}


# seqlen="512 1024 2048 4096"
# seqlen="256 2304"
seqlen="2048"


# DeepSeek-V2

if [ "$skip_to_update" != "True" ]; then

# CENT Baseline

# 64 Devices

for data_parallel in {1..8}
do
for devices in 48 56 64 72 80 88 96 104 108 112 120 128
do
python3 run_sim.py --model DeepSeek-V2 --generate_trace --simulate_trace --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices $devices --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads

python3 run_sim.py --model DeepSeek-V2 --generate_trace --simulate_trace --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices $devices --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --split_decoder_across_devices
done
done

# Systolic PIM

# 64 Devicess

for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
for flash_attention_block_size in 1024 2048 4096 8192
do
python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --generate_trace --simulate_trace --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices 128 --data_parallel $data_parallel --seqlen $seqlen --flash-attention-block-size $flash_attention_block_size --run_simulation_max_workers $threads --generate_trace_max_workers $threads --split_decoder_across_devices --density 32Gb

python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --generate_trace --simulate_trace --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices 128 --data_parallel $data_parallel --seqlen $seqlen --flash-attention-block-size $flash_attention_block_size --run_simulation_max_workers $threads --generate_trace_max_workers $threads --split_decoder_across_devices --density 32Gb --MoE_multiple_experts_per_channel
done
done
done

python3 run_sim.py --model DeepSeek-V2 --process_results --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices 120 --data_parallel 3 --seqlen $seqlen --run_simulation_max_workers 1 --generate_trace_max_workers 1 --density 8Gb --split_decoder_across_devices


for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
for flash_attention_block_size in 1024 2048 4096 8192
do
python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --generate_trace --simulate_trace --process_results --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices 128 --data_parallel $data_parallel --seqlen $seqlen --flash-attention-block-size $flash_attention_block_size --run_simulation_max_workers $threads --generate_trace_max_workers $threads --split_decoder_across_devices --density 32Gb
done
done
done


for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
for flash_attention_block_size in 1024 2048 4096 8192
do
python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --generate_trace --simulate_trace --process_results --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices 256 --data_parallel $data_parallel --seqlen $seqlen --flash-attention-block-size $flash_attention_block_size --run_simulation_max_workers $threads --generate_trace_max_workers $threads --split_decoder_across_devices --density 32Gb
done
done
done


for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
for flash_attention_block_size in 1024 2048 4096 8192
do
python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --generate_trace --simulate_trace --process_results --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices 512 --data_parallel $data_parallel --seqlen $seqlen --flash-attention-block-size $flash_attention_block_size --run_simulation_max_workers $threads --generate_trace_max_workers $threads --split_decoder_across_devices --density 32Gb
done
done
done



fi






# 5000 W Power

# CENT

for density in 8Gb 16Gb 24Gb
do
for data_parallel in {1..8}
do
for devices in 48 56 64 72 80 88 96 104 108 112 120 128
do
for flash_attention_block_size in 1024 2048 4096 8192
do
python3 run_sim.py --model DeepSeek-V2 --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_DeepSeek_V2_${density}.csv --density ${density} --power_limit 5600
python3 run_sim.py --model DeepSeek-V2 --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_DeepSeek_V2_${density}.csv --density ${density} --flash-attention-block-size $flash_attention_block_size --split_decoder_across_devices --power_limit 5600
done
done
done
done

# DREAM

for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
for devices in 48 56 64 72 80 88 96 104 108 112 120 128
do
for flash_attention_block_size in 1024 2048 4096 8192
do

for density in 16Gb 24Gb
do
python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_DeepSeek_V2_${density}.csv --density ${density} --flash-attention-block-size $flash_attention_block_size  --split_decoder_across_devices --power_limit 5600
done

for density in 8Gb
do
python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_DeepSeek_V2_${density}.csv --density ${density} --flash-attention-block-size $flash_attention_block_size --split_decoder_across_devices --power_limit 5600 --MoE_multiple_experts_per_channel
done

for density in 16Gb 24Gb
do
python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_DeepSeek_V2_${density}.csv --density ${density} --flash-attention-block-size $flash_attention_block_size --power_limit 5600
done

for density in 8Gb
do
python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_DeepSeek_V2_${density}.csv --density ${density} --flash-attention-block-size $flash_attention_block_size --power_limit 5600 --MoE_multiple_experts_per_channel
done

done
done
done
done

# 10000 W Power

for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
for devices in 120 128
do
for flash_attention_block_size in 1024 2048 4096 8192
do

for density in 8Gb 16Gb 24Gb
do
python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_DeepSeek_V2_${density}_11200W.csv --density ${density} --flash-attention-block-size $flash_attention_block_size  --split_decoder_across_devices --power_limit 11200 --PCIE_lanes 288
done

# for density in 8Gb
# do
# python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_DeepSeek_V2_${density}_11200W.csv --density ${density} --flash-attention-block-size $flash_attention_block_size --split_decoder_across_devices --power_limit 11200 --PCIE_lanes 288 --MoE_multiple_experts_per_channel
# done

for density in 8Gb 16Gb 24Gb
do
python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_DeepSeek_V2_${density}_11200W.csv --density ${density} --flash-attention-block-size $flash_attention_block_size --power_limit 11200 --PCIE_lanes 288
done

# for density in 8Gb
# do
# python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_DeepSeek_V2_${density}_11200W.csv --density ${density} --flash-attention-block-size $flash_attention_block_size --power_limit 11200 --PCIE_lanes 288 --MoE_multiple_experts_per_channel
# done

done
done
done
done



# 20000 W Power


for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
for devices in 240 248 256
do
for flash_attention_block_size in 1024 2048 4096 8192
do

for density in 8Gb 16Gb 24Gb
do
python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_DeepSeek_V2_${density}_22400W.csv --density ${density} --flash-attention-block-size $flash_attention_block_size  --split_decoder_across_devices --power_limit 22400 --PCIE_lanes 576
done

# for density in 8Gb
# do
# python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_DeepSeek_V2_${density}_22400W.csv --density ${density} --flash-attention-block-size $flash_attention_block_size --split_decoder_across_devices --power_limit 22400 --PCIE_lanes 576 --MoE_multiple_experts_per_channel
# done

for density in 8Gb 16Gb 24Gb
do
python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_DeepSeek_V2_${density}_22400W.csv --density ${density} --flash-attention-block-size $flash_attention_block_size --power_limit 22400 --PCIE_lanes 576
done

# for density in 8Gb
# do
# python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_DeepSeek_V2_${density}_22400W.csv --density ${density} --flash-attention-block-size $flash_attention_block_size --power_limit 22400 --PCIE_lanes 576 --MoE_multiple_experts_per_channel
# done

done
done
done
done



# 40000 W Power


for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
for devices in 480 496 512
do
for flash_attention_block_size in 1024 2048 4096 8192
do

for density in 8Gb 16Gb 24Gb
do
python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_DeepSeek_V2_${density}_44800W.csv --density ${density} --flash-attention-block-size $flash_attention_block_size  --split_decoder_across_devices --power_limit 44800 --PCIE_lanes 1152
done

# for density in 8Gb
# do
# python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_DeepSeek_V2_${density}_44800W.csv --density ${density} --flash-attention-block-size $flash_attention_block_size --split_decoder_across_devices --power_limit 44800 --PCIE_lanes 1152 --MoE_multiple_experts_per_channel
# done

for density in 8Gb 16Gb 24Gb
do
python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_DeepSeek_V2_${density}_44800W.csv --density ${density} --flash-attention-block-size $flash_attention_block_size --power_limit 44800 --PCIE_lanes 1152
done

# for density in 8Gb
# do
# python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_DeepSeek_V2_${density}_44800W.csv --density ${density} --flash-attention-block-size $flash_attention_block_size --power_limit 44800 --PCIE_lanes 1152 --MoE_multiple_experts_per_channel
# done

done
done
done
done








# # 96 devices


# 10000 W Power

seqlen="2048"
threads=32

for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
for flash_attention_block_size in 8192
do
python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --generate_trace --simulate_trace --process_results --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices 192 --data_parallel $data_parallel --seqlen $seqlen --flash-attention-block-size $flash_attention_block_size --run_simulation_max_workers $threads --generate_trace_max_workers $threads --split_decoder_across_devices --density 32Gb
done
done
done


for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
for devices in 192
do
for flash_attention_block_size in 8192
do

for density in 8Gb 16Gb 24Gb
do
python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_DeepSeek_V2_${density}_10000W.csv --density ${density} --flash-attention-block-size $flash_attention_block_size  --split_decoder_across_devices --power_limit 11200 --PCIE_lanes 288
done

for density in 8Gb 16Gb 24Gb
do
python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_DeepSeek_V2_${density}_10000W.csv --density ${density} --flash-attention-block-size $flash_attention_block_size --power_limit 11200 --PCIE_lanes 288
done

done
done
done
done



# 20000 W Power

seqlen="2048"
threads=32

for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
for flash_attention_block_size in 8192
do
python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --generate_trace --simulate_trace --process_results --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices 384 --data_parallel $data_parallel --seqlen $seqlen --flash-attention-block-size $flash_attention_block_size --run_simulation_max_workers $threads --generate_trace_max_workers $threads --split_decoder_across_devices --density 32Gb
done
done
done

for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
for devices in 384
do
for flash_attention_block_size in 8192
do

for density in 8Gb 16Gb 24Gb
do
python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_DeepSeek_V2_${density}_20000W.csv --density ${density} --flash-attention-block-size $flash_attention_block_size  --split_decoder_across_devices --power_limit 22400 --PCIE_lanes 576
done


for density in 8Gb 16Gb 24Gb
do
python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_DeepSeek_V2_${density}_20000W.csv --density ${density} --flash-attention-block-size $flash_attention_block_size --power_limit 22400 --PCIE_lanes 576
done


done
done
done
done



# 40000 W Power

seqlen="2048"
threads=32

for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
for flash_attention_block_size in 8192
do
python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --generate_trace --simulate_trace --process_results --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices 768 --data_parallel $data_parallel --seqlen $seqlen --flash-attention-block-size $flash_attention_block_size --run_simulation_max_workers $threads --generate_trace_max_workers $threads --split_decoder_across_devices --density 32Gb
done
done
done

for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
for devices in 768
do
for flash_attention_block_size in 8192
do

for density in 8Gb 16Gb 24Gb
do
python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_DeepSeek_V2_${density}_40000W.csv --density ${density} --flash-attention-block-size $flash_attention_block_size  --split_decoder_across_devices --power_limit 44800 --PCIE_lanes 1152
done

for density in 8Gb 16Gb 24Gb
do
python3 run_sim.py --model DeepSeek-V2 --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --simulation_result_path DREAM_simulation_results/simulation_results_DeepSeek_V2_${density}_40000W.csv --density ${density} --flash-attention-block-size $flash_attention_block_size --power_limit 44800 --PCIE_lanes 1152
done

done
done
done
done

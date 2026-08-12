threads=${1:-8}
skip_to_update=${2:-False}


# seqlen="576 640 768 1024 1536 2560 4608 8704 16896"
# seqlen="1024 2048 4096 8192 16384 32768 65536 131072"

if [ "$skip_to_update" != "True" ]; then

for seqlen in 1024 2048 4096 8192 16384 32768 65536 131072
do

for density in 8Gb
do
for devices in 64 96 128
do
for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
for pipeline_parallelism in 56 28 14 8
do
python3 run_sim.py --model Mixtral-8x22B --MoE_minimal_channels --generate_trace --simulate_trace --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --split_decoder_across_devices --pipeline_stages $pipeline_parallelism
done
done
done
done
done

for density in 16Gb
do
for devices in 64 96 128
do
for data_parallel in 1
do
for systolic_dim in 1 2 4
do
for pipeline_parallelism in 56 28 14 8
do
python3 run_sim.py --model Mixtral-8x22B --MoE_minimal_channels --generate_trace --simulate_trace --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --split_decoder_across_devices --pipeline_stages $pipeline_parallelism
done
done
done
done
done


for density in 24Gb
do
for devices in 64 96 128
do
for data_parallel in 1
do
for systolic_dim in 1
do
for pipeline_parallelism in 56 28 14 8
do
python3 run_sim.py --model Mixtral-8x22B --MoE_minimal_channels --generate_trace --simulate_trace --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --split_decoder_across_devices --pipeline_stages $pipeline_parallelism
done
done
done
done
done

done

python3 run_sim.py --model Mixtral-8x22B --MoE_minimal_channels --process_results --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim 1 --num_devices 64 --data_parallel 1 --seqlen 256 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density 16Gb --split_decoder_across_devices

fi

# Split decoder across devices

for seqlen in 1024 2048 4096 8192 16384 32768 65536 131072
do

for density in 8Gb
do
for devices in 64 96 128
do
for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
for pipeline_parallelism in 56 28 14 8
do
python3 run_sim.py --model Mixtral-8x22B --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_${density}_seqlen_${seqlen}.csv  --split_decoder_across_devices --pipeline_stages $pipeline_parallelism --power_limit 5600
python3 run_sim.py --model Mixtral-8x22B --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_${density}_seqlen_${seqlen}.csv --pipeline_stages $pipeline_parallelism --power_limit 5600
done
done
done
done
done


for density in 16Gb
do
for devices in 64 96 128
do
for data_parallel in 1
do
for systolic_dim in 1 2 4
do
for pipeline_parallelism in 56 28 14 8
do
python3 run_sim.py --model Mixtral-8x22B --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_${density}_seqlen_${seqlen}.csv  --split_decoder_across_devices --pipeline_stages $pipeline_parallelism --power_limit 5600
python3 run_sim.py --model Mixtral-8x22B --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_${density}_seqlen_${seqlen}.csv --pipeline_stages $pipeline_parallelism --power_limit 5600
done
done
done
done
done


for density in 24Gb
do
for devices in 64 96 128
do
for data_parallel in 1
do
for systolic_dim in 1
do
for pipeline_parallelism in 56 28 14 8
do
python3 run_sim.py --model Mixtral-8x22B --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_${density}_seqlen_${seqlen}.csv  --split_decoder_across_devices --pipeline_stages $pipeline_parallelism --power_limit 5600
python3 run_sim.py --model Mixtral-8x22B --MoE_minimal_channels --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_${density}_seqlen_${seqlen}.csv --pipeline_stages $pipeline_parallelism --power_limit 5600
done
done
done
done
done

done

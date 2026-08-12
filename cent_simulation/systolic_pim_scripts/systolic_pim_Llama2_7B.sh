threads=${1:-8}
skip_to_update=${2:-False}

# seqlen="512 1024 2048 4096"
seqlen="2048"


# Llama2-7B

# Design Space Exploration After Micro Submission

# Dense model has to explore both split_decoder_across_devices and not split_decoder_across_devices scenarios.

if [ "$skip_to_update" != "True" ]; then

# Split decoder across devices

for devices in 48 56 64 72 80 88 96 104 108 112 120 128
do

for data_parallel in {4..18}
do
python3 run_sim.py --model Llama2-7B --generate_trace --simulate_trace --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density 8Gb --simulation_result_path DREAM_simulation_results/simulation_results_Llama2_7B_8Gb.csv --split_decoder_across_devices
done

for data_parallel in {4..18}
do
for systolic_dim in 1 2 4 8
do
python3 run_sim.py --model Llama2-7B --generate_trace --simulate_trace --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density 8Gb --simulation_result_path DREAM_simulation_results/simulation_results_Llama2_7B_8Gb.csv  --split_decoder_across_devices
done
done


# Not split decoder across devices

for data_parallel in {4..18}
do
python3 run_sim.py --model Llama2-7B --generate_trace --simulate_trace --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density 8Gb --simulation_result_path DREAM_simulation_results/simulation_results_Llama2_7B_8Gb.csv
done

for data_parallel in {4..18}
do
for systolic_dim in 1 2 4 8
do
python3 run_sim.py --model Llama2-7B --generate_trace --simulate_trace --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density 8Gb --simulation_result_path DREAM_simulation_results/simulation_results_Llama2_7B_8Gb.csv
done
done

done


python3 run_sim.py --model Llama2-7B --process_results --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices 64 --data_parallel 16 --seqlen $seqlen --run_simulation_max_workers 1 --generate_trace_max_workers 1 --density 8Gb --split_decoder_across_devices

fi


# Split decoder across devices

for density in 8Gb 16Gb 24Gb
do

for devices in 48 56 64 72 80 88 96 104 108 112 120 128
do

for data_parallel in {4..18}
do
python3 run_sim.py --model Llama2-7B --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path DREAM_simulation_results/simulation_results_Llama2_7B_${density}.csv --split_decoder_across_devices --power_limit 5600
done

for data_parallel in {4..18}
do
for systolic_dim in 1 2 4 8
do
python3 run_sim.py --model Llama2-7B --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path DREAM_simulation_results/simulation_results_Llama2_7B_${density}.csv  --split_decoder_across_devices --power_limit 5600
done
done


# Not split decoder across devices

for data_parallel in {4..18}
do
python3 run_sim.py --model Llama2-7B --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path DREAM_simulation_results/simulation_results_Llama2_7B_${density}.csv --power_limit 5600
done

for data_parallel in {4..18}
do
for systolic_dim in 1 2 4 8
do
python3 run_sim.py --model Llama2-7B --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path DREAM_simulation_results/simulation_results_Llama2_7B_${density}.csv --power_limit 5600
done
done

done

done

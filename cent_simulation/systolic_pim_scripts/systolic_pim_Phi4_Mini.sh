threads=${1:-8}
skip_to_update=${2:-False}


# seqlen="512 1024 2048 4096"
seqlen="256"


# Design Space Exploration After Micro Submission

# Dense model has to explore both split_decoder_across_devices and not split_decoder_across_devices scenarios.

if [ "$skip_to_update" != "True" ]; then


for devices in 1 #2 3 4
do

# Not split decoder across devices

# Pipeline Parallel

for data_parallel in 1
do
python3 run_sim.py --model Phi4-Mini --generate_trace --simulate_trace --DRAM_type LPDDR5 --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density 8Gb --simulation_result_path simulation_results_Phi4_Mini_8Gb.csv
done

for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
python3 run_sim.py --model Phi4-Mini --generate_trace --simulate_trace --DRAM_type LPDDR5 --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density 8Gb --simulation_result_path simulation_results_Phi4_Mini_8Gb.csv
done
done


# Model Parallel

for data_parallel in 1
do
python3 run_sim.py --model Phi4-Mini --model_parallel --FC_devices 1 --generate_trace --simulate_trace --DRAM_type LPDDR5 --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density 8Gb --simulation_result_path simulation_results_Phi4_Mini_8Gb.csv
done

for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
python3 run_sim.py --model Phi4-Mini --model_parallel --FC_devices 1 --generate_trace --simulate_trace --DRAM_type LPDDR5 --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density 8Gb --simulation_result_path simulation_results_Phi4_Mini_8Gb.csv
done
done


done




python3 run_sim.py --model Phi4-Mini --process_results --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices 64 --data_parallel 16 --seqlen 2048 --run_simulation_max_workers 1 --generate_trace_max_workers 1 --density 8Gb --split_decoder_across_devices

fi




for density in 8Gb #16Gb 32Gb
do

for devices in 1 #2 3 4
do

# Not split decoder across devices

# Pipeline Parallel

for data_parallel in 1
do
python3 run_sim.py --model Phi4-Mini --update_csv --DRAM_type LPDDR5 --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path simulation_results_Phi4_Mini_${density}.csv --power_limit 5600
done

for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
python3 run_sim.py --model Phi4-Mini --update_csv --DRAM_type LPDDR5 --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path simulation_results_Phi4_Mini_${density}.csv --power_limit 5600
done
done


# Model Parallel

for data_parallel in 1
do
python3 run_sim.py --model Phi4-Mini --model_parallel --DRAM_type LPDDR5 --FC_devices 1 --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path simulation_results_Phi4_Mini_${density}.csv --power_limit 5600
done

for data_parallel in 1
do
for systolic_dim in 1 2 4 8
do
python3 run_sim.py --model Phi4-Mini --model_parallel --DRAM_type LPDDR5 --FC_devices 1 --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path simulation_results_Phi4_Mini_${density}.csv --power_limit 5600
done
done

done

done

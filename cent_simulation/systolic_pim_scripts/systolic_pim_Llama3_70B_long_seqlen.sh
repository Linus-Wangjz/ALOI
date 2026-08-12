threads=${1:-8}
skip_to_update=${2:-False}


seqlen="576 640 768 1024 1536 2560 4608 8704 16896"
# seqlen="4160 4224 4352 4608 5120 6144 8192 12288 20480"
# seqlen="256 1024 4096"
# seqlen="2048"

if [ "$skip_to_update" != "True" ]; then

# for seqlen in 576 640 768 1024 1536 2560 4608 8704 16896
# do
for density in 8Gb 16Gb 24Gb
do
for devices in 64
do

for data_parallel in {1..8}
do
python3 run_sim.py --model Llama3-70B --generate_trace --simulate_trace --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --split_decoder_across_devices
done

for data_parallel in {1..8}
do
for systolic_dim in 1 2 4 8
do
python3 run_sim.py --model Llama3-70B --generate_trace --simulate_trace --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --split_decoder_across_devices
done
done

done
done
# done

python3 run_sim.py --model Llama3-70B --process_results --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim 1 --num_devices 64 --data_parallel 1 --seqlen 256 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density 16Gb --split_decoder_across_devices

fi

# Split decoder across devices

for seqlen in 576 640 768 1024 1536 2560 4608 8704 16896
do

for density in 8Gb 16Gb 24Gb
do

for devices in 64 #72 80 88 96 104 108 112 120 128
do

for data_parallel in {1..8}
do
python3 run_sim.py --model Llama3-70B --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path DREAM_simulation_results/simulation_results_Llama3_70B_${density}_seqlen_${seqlen}.csv --split_decoder_across_devices --power_limit 5600
done

for data_parallel in {1..8}
do
for systolic_dim in 1 2 4 8
do
python3 run_sim.py --model Llama3-70B --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path DREAM_simulation_results/simulation_results_Llama3_70B_${density}_seqlen_${seqlen}.csv  --split_decoder_across_devices --power_limit 5600
done
done


# Not split decoder across devices

for data_parallel in {1..8}
do
python3 run_sim.py --model Llama3-70B --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --EWMUL_PNM --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path DREAM_simulation_results/simulation_results_Llama3_70B_${density}_seqlen_${seqlen}.csv --power_limit 5600
done

for data_parallel in {1..8}
do
for systolic_dim in 1 2 4 8
do
python3 run_sim.py --model Llama3-70B --update_csv --full_accelerator_softmax --pipelined_softmax --flash_attention --systolic_pim --systolic_dim $systolic_dim --num_devices ${devices} --data_parallel $data_parallel --seqlen $seqlen --run_simulation_max_workers $threads --generate_trace_max_workers $threads --density ${density} --simulation_result_path DREAM_simulation_results/simulation_results_Llama3_70B_${density}_seqlen_${seqlen}.csv --power_limit 5600
done
done

done
done
done

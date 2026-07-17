num_banks=${1:-16}
num_channels=32

for phase in prefill decoding end2end
do
python3 run_sim.py --num_channels $num_channels --num_banks $num_banks --model Llama2-7B --process_throughputs --num_devices 8 --phase $phase
python3 run_sim.py --num_channels $num_channels --num_banks $num_banks --model Llama2-13B --process_throughputs --num_devices 20 --phase $phase
python3 run_sim.py --num_channels $num_channels --num_banks $num_banks --model Llama2-70B --process_throughputs --num_devices 32 --phase $phase

python3 run_sim.py --num_channels $num_channels --num_banks $num_banks --model Llama2-7B --model_parallel --process_throughputs --num_devices 8 --phase $phase
python3 run_sim.py --num_channels $num_channels --num_banks $num_banks --model Llama2-13B --model_parallel --process_throughputs --num_devices 20 --phase $phase
python3 run_sim.py --num_channels $num_channels --num_banks $num_banks --model Llama2-70B --model_parallel --process_throughputs --num_devices 32 --phase $phase
done

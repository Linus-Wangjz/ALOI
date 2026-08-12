#!/usr/bin/env bash

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
mkdir -p log

# generate/simulate trace and update csv
bash systolic_pim_scripts/systolic_pim_Mixtral_8x7B.sh 8 > log/Mixtral_8x7B.log 2>&1
bash systolic_pim_scripts/systolic_pim_Mixtral_8x22B.sh 8 > log/Mixtral_8x22B.log 2>&1
bash systolic_pim_scripts/systolic_pim_DeepSeek_V2.sh 8 > log/DeepSeek_V2.log 2>&1
bash systolic_pim_scripts/systolic_pim_DeepSeek_V2_Lite.sh 8 > log/DeepSeek_V2_Lite.log 2>&1

bash systolic_pim_scripts/systolic_pim_Llama2_7B.sh 8 > log/Llama2_7B.log 2>&1
bash systolic_pim_scripts/systolic_pim_Llama3_8B.sh 8 > log/Llama3_8B.log 2>&1
bash systolic_pim_scripts/systolic_pim_Llama3_70B.sh 8 > log/Llama3_70B.log 2>&1

bash systolic_pim_scripts/systolic_pim_Llama3_70B_long_seqlen.sh 8 > log/Llama3_70B_long.log 2>&1
bash systolic_pim_scripts/systolic_pim_HBM_PIM_Llama3_70B.sh > log/HBM_PIM_Llama3_70B.log 2>&1
bash systolic_pim_scripts/systolic_pim_HBM_PIM_Mixtral_8x7B.sh > log/HBM_PIM_Mixtral_8x7B.log 2>&1

# DREAM DSE 8GB+8x16SA vs 16GB+4x16SA vs 24GB+1x16SA
bash systolic_pim_scripts/systolic_pim_DSE_Llama3_70B.sh 8 > log/Llama3_70B_DSE.log 2>&1
bash systolic_pim_scripts/systolic_pim_DSE_Mixtral_8x22B.sh 8 > log/Mixtral_8x22B_DSE.log 2>&1
bash systolic_pim_scripts/systolic_pim_DSE_DeepSeek_V2.sh 8 > log/DeepSeek_V2_DSE.log 2>&1

# only update csv
bash systolic_pim_scripts/systolic_pim_Mixtral_8x7B.sh 8 True > log/Mixtral_8x7B.log 2>&1
bash systolic_pim_scripts/systolic_pim_Mixtral_8x22B.sh 8 True > log/Mixtral_8x22B.log 2>&1
bash systolic_pim_scripts/systolic_pim_DeepSeek_V2.sh 8 True > log/DeepSeek_V2.log 2>&1
bash systolic_pim_scripts/systolic_pim_DeepSeek_V2_Lite.sh 8 True > log/DeepSeek_V2_Lite.log 2>&1

bash systolic_pim_scripts/systolic_pim_Llama2_7B.sh 8 True > log/Llama2_7B.log 2>&1
bash systolic_pim_scripts/systolic_pim_Llama3_8B.sh 8 True > log/Llama3_8B.log 2>&1
bash systolic_pim_scripts/systolic_pim_Llama3_70B.sh 8 True > log/Llama3_70B.log 2>&1

bash systolic_pim_scripts/systolic_pim_Llama3_70B_long_seqlen.sh 8 True > log/Llama3_70B_long.log 2>&1
bash systolic_pim_scripts/systolic_pim_HBM_PIM_Llama3_70B.sh > log/HBM_PIM_Llama3_70B.log 2>&1
bash systolic_pim_scripts/systolic_pim_HBM_PIM_Mixtral_8x7B.sh > log/HBM_PIM_Mixtral_8x7B.log 2>&1

# DREAM DSE 8GB+8x16SA vs 16GB+4x16SA vs 24GB+1x16SA
bash systolic_pim_scripts/systolic_pim_DSE_Llama3_70B.sh 8 True > log/Llama3_70B_DSE.log 2>&1
bash systolic_pim_scripts/systolic_pim_DSE_Mixtral_8x22B.sh 8 True > log/Mixtral_8x22B_DSE.log 2>&1
bash systolic_pim_scripts/systolic_pim_DSE_DeepSeek_V2.sh 8 True > log/DeepSeek_V2_DSE.log 2>&1

# Systolic size evaluation

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_16Gb.csv --device 96
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_16Gb.csv --systolic_pim --systolic_dim 1 --device 96
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_16Gb.csv --systolic_pim --systolic_dim 2 --device 96
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_16Gb.csv --systolic_pim --systolic_dim 4 --device 96
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_16Gb.csv --systolic_pim --systolic_dim 8 --device 96

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_8Gb.csv --device 64
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_8Gb.csv --systolic_pim --systolic_dim 1 --device 64
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_8Gb.csv --systolic_pim --systolic_dim 2 --device 64
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_8Gb.csv --systolic_pim --systolic_dim 4 --device 64
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_8Gb.csv --systolic_pim --systolic_dim 8 --device 64


# CENT baselines


python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama2_7B_8Gb.csv --device 128
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_8B_8Gb.csv --device 128
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_8Gb.csv --device 128
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_8Gb.csv --device 128
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_8Gb.csv --device 128
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_8Gb.csv --device 128
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_8Gb.csv --device 128

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama2_7B_8Gb.csv --device 96
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_8B_8Gb.csv --device 96
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_8Gb.csv --device 96
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_8Gb.csv --device 96
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_8Gb.csv --device 96
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_8Gb.csv --device 96
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_8Gb.csv --device 96

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama2_7B_8Gb.csv --device 64
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_8B_8Gb.csv --device 64
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_8Gb.csv --device 64
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_8Gb.csv --device 64
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_8Gb.csv --device 64
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_8Gb.csv --device 64
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_8Gb.csv --device 64

# DREAM results

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama2_7B_16Gb.csv --systolic_pim --systolic_dim 4 --device 96
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_8B_16Gb.csv --systolic_pim --systolic_dim 4 --device 96
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_16Gb.csv --systolic_pim --systolic_dim 4 --device 96
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_16Gb.csv --systolic_pim --systolic_dim 4 --device 96
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_16Gb.csv --systolic_pim --systolic_dim 4 --device 96
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_16Gb.csv --systolic_pim --systolic_dim 4 --device 96
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_16Gb.csv --systolic_pim --systolic_dim 4 --device 96

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_16Gb_2800W.csv --systolic_pim --systolic_dim 4 --device 48 --power_limit 2800
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_16Gb_2800W.csv --systolic_pim --systolic_dim 4 --device 48 --power_limit 2800
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_16Gb_11200W.csv --systolic_pim --systolic_dim 4 --device 192 --power_limit 11200
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_16Gb_11200W.csv --systolic_pim --systolic_dim 4 --device 192 --power_limit 11200
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_16Gb_22400W.csv --systolic_pim --systolic_dim 4 --device 384 --power_limit 22400
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_16Gb_44800W.csv --systolic_pim --systolic_dim 4 --device 768 --power_limit 44800

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama2_7B_16Gb.csv --systolic_pim --systolic_dim 4 --device 128
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_8B_16Gb.csv --systolic_pim --systolic_dim 4 --device 128
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_16Gb.csv --systolic_pim --systolic_dim 4 --device 128
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_16Gb.csv --systolic_pim --systolic_dim 4 --device 128
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_16Gb.csv --systolic_pim --systolic_dim 4 --device 128
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_16Gb.csv --systolic_pim --systolic_dim 4 --device 128
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_16Gb.csv --systolic_pim --systolic_dim 4 --device 128

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_16Gb_2800W.csv --systolic_pim --systolic_dim 4 --device 64 --power_limit 2800
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_16Gb_2800W.csv --systolic_pim --systolic_dim 4 --device 64 --power_limit 2800
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_16Gb_22400W.csv --systolic_pim --systolic_dim 4 --device 256 --power_limit 11200
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_16Gb_44800W.csv --systolic_pim --systolic_dim 4 --device 512 --power_limit 22400


python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama2_7B_8Gb.csv --systolic_pim --systolic_dim 8 --device 64
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_8B_8Gb.csv --systolic_pim --systolic_dim 8 --device 64
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_8Gb.csv --systolic_pim --systolic_dim 8 --device 64
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_8Gb.csv --systolic_pim --systolic_dim 8 --device 64
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_8Gb.csv --systolic_pim --systolic_dim 8 --device 64
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_8Gb.csv --systolic_pim --systolic_dim 8 --device 64
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_8Gb.csv --systolic_pim --systolic_dim 8 --device 64

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_8Gb_2800W.csv --systolic_pim --systolic_dim 8 --device 32 --power_limit 2800
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_8Gb_2800W.csv --systolic_pim --systolic_dim 8 --device 32 --power_limit 2800
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_8Gb_11200W.csv --systolic_pim --systolic_dim 8 --device 128 --power_limit 11200
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_8Gb_11200W.csv --systolic_pim --systolic_dim 8 --device 128 --power_limit 11200
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_8Gb_22400W.csv --systolic_pim --systolic_dim 8 --device 256 --power_limit 22400
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_8Gb_44800W.csv --systolic_pim --systolic_dim 8 --device 512 --power_limit 44800

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama2_7B_8Gb.csv --systolic_pim --systolic_dim 8 --device 96
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_8B_8Gb.csv --systolic_pim --systolic_dim 8 --device 96
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_8Gb.csv --systolic_pim --systolic_dim 8 --device 96
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_8Gb.csv --systolic_pim --systolic_dim 8 --device 96
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_8Gb.csv --systolic_pim --systolic_dim 8 --device 96
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_8Gb.csv --systolic_pim --systolic_dim 8 --device 96
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_8Gb.csv --systolic_pim --systolic_dim 8 --device 96

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_8Gb_2800W.csv --systolic_pim --systolic_dim 8 --device 48 --power_limit 2800
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_8Gb_2800W.csv --systolic_pim --systolic_dim 8 --device 48 --power_limit 2800
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_8Gb_11200W.csv --systolic_pim --systolic_dim 8 --device 192 --power_limit 11200
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_8Gb_11200W.csv --systolic_pim --systolic_dim 8 --device 192 --power_limit 11200
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_8Gb_22400W.csv --systolic_pim --systolic_dim 8 --device 384 --power_limit 22400
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_8Gb_44800W.csv --systolic_pim --systolic_dim 8 --device 768 --power_limit 44800

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama2_7B_8Gb.csv --systolic_pim --systolic_dim 8 --device 128
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_8B_8Gb.csv --systolic_pim --systolic_dim 8 --device 128
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_8Gb.csv --systolic_pim --systolic_dim 8 --device 128
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_8Gb.csv --systolic_pim --systolic_dim 8 --device 128
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_8Gb.csv --systolic_pim --systolic_dim 8 --device 128
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_8Gb.csv --systolic_pim --systolic_dim 8 --device 128
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_8Gb.csv --systolic_pim --systolic_dim 8 --device 128

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_8Gb_2800W.csv --systolic_pim --systolic_dim 8 --device 64 --power_limit 2800
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_8Gb_2800W.csv --systolic_pim --systolic_dim 8 --device 64 --power_limit 2800
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_8Gb_22400W.csv --systolic_pim --systolic_dim 8 --device 256 --power_limit 11200
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_8Gb_44800W.csv --systolic_pim --systolic_dim 8 --device 512 --power_limit 22400








# CENT resutls

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama2_7B_8Gb.csv
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama2_7B_16Gb.csv
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama2_7B_24Gb.csv

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_8B_8Gb.csv
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_8B_16Gb.csv
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_8B_24Gb.csv

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_8Gb.csv
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_16Gb.csv
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_24Gb.csv

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_8Gb.csv
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_16Gb.csv
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_24Gb.csv

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_8Gb.csv
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_16Gb.csv
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_24Gb.csv

python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_8Gb.csv
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_16Gb.csv
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_24Gb.csv

python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_8Gb.csv
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_16Gb.csv
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_24Gb.csv


# DREAM results

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama2_7B_8Gb.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama2_7B_16Gb.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama2_7B_24Gb.csv --systolic_pim

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_8B_8Gb.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_8B_16Gb.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_8B_24Gb.csv --systolic_pim

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_8Gb.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_16Gb.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_24Gb.csv --systolic_pim

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_8Gb.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_16Gb.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_24Gb.csv --systolic_pim

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_8Gb.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_16Gb.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_24Gb.csv --systolic_pim

python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_8Gb.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_16Gb.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_24Gb.csv --systolic_pim

python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_8Gb.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_16Gb.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_24Gb.csv --systolic_pim

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_8Gb_11200W.csv --systolic_pim --device 128 --power_limit 10000
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_16Gb_11200W.csv --systolic_pim --device 128 --power_limit 10000
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_24Gb_11200W.csv --systolic_pim --device 128 --power_limit 10000

python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_8Gb_2800W.csv --systolic_pim --device 48 --power_limit 2500
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_16Gb_2800W.csv --systolic_pim --device 48 --power_limit 2500
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_24Gb_2800W.csv --systolic_pim --device 48 --power_limit 2500

python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_8Gb_2800W.csv --systolic_pim --device 32 --power_limit 1866
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_16Gb_2800W.csv --systolic_pim --device 32 --power_limit 1866
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_24Gb_2800W.csv --systolic_pim --device 32 --power_limit 1866

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_8Gb_2800W.csv --systolic_pim --device 48 --power_limit 2500
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_16Gb_2800W.csv --systolic_pim --device 48 --power_limit 2500
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_24Gb_2800W.csv --systolic_pim --device 48 --power_limit 2500

python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_8Gb_11200W.csv --systolic_pim --device 128 --power_limit 10000
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_16Gb_11200W.csv --systolic_pim --device 128 --power_limit 10000
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_24Gb_11200W.csv --systolic_pim --device 128 --power_limit 10000

python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_8Gb_22400W.csv --systolic_pim --device 256 --power_limit 20000
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_16Gb_22400W.csv --systolic_pim --device 256 --power_limit 20000
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_24Gb_22400W.csv --systolic_pim --device 256 --power_limit 20000

python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_8Gb_44800W.csv --systolic_pim --device 512 --power_limit 40000
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_16Gb_44800W.csv --systolic_pim --device 512 --power_limit 40000
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_24Gb_44800W.csv --systolic_pim --device 512 --power_limit 40000


python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_8Gb_11200W.csv --systolic_pim --device 192 --power_limit 10000
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_16Gb_11200W.csv --systolic_pim --device 192 --power_limit 10000
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_24Gb_11200W.csv --systolic_pim --device 192 --power_limit 10000

python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_8Gb_22400W.csv --systolic_pim --device 384 --power_limit 20000
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_16Gb_22400W.csv --systolic_pim --device 384 --power_limit 20000
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_24Gb_22400W.csv --systolic_pim --device 384 --power_limit 20000


python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_16Gb_seqlen_576.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_16Gb_seqlen_640.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_16Gb_seqlen_768.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_16Gb_seqlen_1024.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_16Gb_seqlen_1536.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_16Gb_seqlen_2560.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_16Gb_seqlen_4608.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_16Gb_seqlen_8704.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_16Gb_seqlen_16896.csv --systolic_pim

python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama2_7B_16Gb.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_8B_16Gb.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_16Gb.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x7B_16Gb.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_16Gb.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_Lite_16Gb.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_DeepSeek_V2_16Gb.csv --systolic_pim

python filter_results.py --file_path DREAM_simulation_results/simulation_results_HBM_PIM_comparison_Llama3_70B_256.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_HBM_PIM_comparison_Llama3_70B_1024.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_HBM_PIM_comparison_Llama3_70B_4096.csv --systolic_pim

python filter_results.py --file_path DREAM_simulation_results/simulation_results_HBM_PIM_comparison_Mixtral_8x7B_256.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_HBM_PIM_comparison_Mixtral_8x7B_1024.csv --systolic_pim
python filter_results.py --file_path DREAM_simulation_results/simulation_results_HBM_PIM_comparison_Mixtral_8x7B_4096.csv --systolic_pim

# seqlen here should double

for seqlen in 1024 2048 4096 8192 16384 32768 65536 131072; do
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_8Gb_seqlen_${seqlen}.csv --systolic_pim --systolic_dim 8
done

for seqlen in 1024 2048 4096 8192 16384 32768 65536 131072; do
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_16Gb_seqlen_${seqlen}.csv --systolic_pim --systolic_dim 4
done

for seqlen in 1024 2048 4096 8192 16384 32768 65536 131072; do
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Llama3_70B_24Gb_seqlen_${seqlen}.csv --systolic_pim --systolic_dim 1
done

for seqlen in 1024 2048 4096 8192 16384 32768 65536 131072; do
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_8Gb_seqlen_${seqlen}.csv --systolic_pim --systolic_dim 8
done

for seqlen in 1024 2048 4096 8192 16384 32768 65536 131072; do
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_16Gb_seqlen_${seqlen}.csv --systolic_pim --systolic_dim 4
done


for seqlen in 1024 2048 4096 8192 16384 32768 65536 131072; do
python filter_results.py --file_path DREAM_simulation_results/simulation_results_Mixtral_8x22B_24Gb_seqlen_${seqlen}.csv --systolic_pim --systolic_dim 1
done

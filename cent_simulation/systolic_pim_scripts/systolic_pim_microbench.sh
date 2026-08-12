
mkdir -p ../trace/systolic_pim_microbench

option=Vector
python3 function_sim.py --n_heads 64 --ffn_dim 14336 --systolic-pim-microbench --microbench $option --trace-file ../trace/systolic_pim_microbench/trace_8_8K_8K_${option}_reuse_bank.txt --only-trace --GEMV reuse-bank --reuse-size 16 --channels-per-block 8 --Llama-GQA
python3 function_sim.py --n_heads 64 --ffn_dim 14336 --systolic-pim-microbench --microbench $option --trace-file ../trace/systolic_pim_microbench/trace_8_8K_8K_${option}_reuse_GB.txt --only-trace --GEMV reuse-GB --reuse-size 16 --channels-per-block 8 --Llama-GQA

# Use for Systolic Array Design Space Exploration
option=Vector
channels=8
for option in SA1x16 SA2x16 SA4x16 SA8x16 SA16x16
do
python3 function_sim.py --n_heads 64 --ffn_dim 14336 --systolic-pim-microbench --microbench $option --trace-file ../trace/systolic_pim_microbench/trace_8_8K_8K_${channels}_channels_${option}.txt --only-trace --reuse-size 16 --channels-per-block ${channels} --Llama-GQA
done

python systolic_pim_microbench_power.py ../trace/systolic_pim_microbench/trace_8_8K_8K_Vector_reuse_GB.txt.log 1
python systolic_pim_microbench_power.py ../trace/systolic_pim_microbench/trace_8_8K_8K_8_channels_SA1x16.txt.log 1
python systolic_pim_microbench_power.py ../trace/systolic_pim_microbench/trace_8_8K_8K_8_channels_SA2x16.txt.log 2
python systolic_pim_microbench_power.py ../trace/systolic_pim_microbench/trace_8_8K_8K_8_channels_SA4x16.txt.log 4
python systolic_pim_microbench_power.py ../trace/systolic_pim_microbench/trace_8_8K_8K_8_channels_SA8x16.txt.log 8
python systolic_pim_microbench_power.py ../trace/systolic_pim_microbench/trace_8_8K_8K_8_channels_SA16x16.txt.log 16


option=Vector
channels=32
for option in SA1x16 SA2x16 SA4x16 SA8x16
do
python3 function_sim.py --n_heads 64 --ffn_dim 14336 --systolic-pim-microbench --microbench $option --trace-file ../trace/systolic_pim_microbench/trace_8_8K_8K_${channels}_channels_${option}.txt --only-trace --reuse-size 16 --channels-per-block ${channels} --Llama-GQA
done

heads=64
channels=8
seqlen=2048
for option in SA1x16_score SA2x16_score SA4x16_score SA8x16_score
do
python3 function_sim.py --n_heads $heads --ffn_dim 14336 --systolic-pim-microbench --microbench $option --trace-file ../trace/systolic_pim_microbench/trace_${heads}_heads_2K_seqlen_${channels}_channels_${option}.txt --seqlen $seqlen --only-trace --reuse-size 16 --channels-per-block ${channels} --Llama-GQA
done


heads=64
channels=32
seqlen=8192
for option in SA1x16_score SA2x16_score SA4x16_score SA8x16_score
do
python3 function_sim.py --n_heads $heads --ffn_dim 14336 --systolic-pim-microbench --microbench $option --trace-file ../trace/systolic_pim_microbench/trace_${heads}_heads_8K_seqlen_${channels}_channels_${option}.txt --seqlen $seqlen --only-trace --reuse-size 16 --channels-per-block ${channels} --Llama-GQA
done




for heads in 32 64
do

for option in Vector_output Vector_score
do
python3 function_sim.py --n_heads $heads --ffn_dim 14336 --systolic-pim-microbench --microbench $option --trace-file ../trace/systolic_pim_microbench/trace_${heads}_heads_4K_seqlen_${option}_reuse_GB.txt --only-trace --GEMV reuse-GB --reuse-size 16 --channels-per-block 8 --Llama-GQA
done

for option in SA1x16_output SA2x16_output SA4x16_output SA8x16_output SA1x16_score SA2x16_score SA4x16_score SA8x16_score
do
python3 function_sim.py --n_heads $heads --ffn_dim 14336 --systolic-pim-microbench --microbench $option --trace-file ../trace/systolic_pim_microbench/trace_${heads}_heads_4K_seqlen_${option}.txt --only-trace --reuse-size 16 --channels-per-block 8 --Llama-GQA
done

done


cd ../trace/systolic_pim_microbench
bash run.sh
bash parse_results.sh
cd ../../cent_simulation

mkdir -p ../trace/32_channels_per_device/microbench/Llama3-70B


# full trace
python3 function_sim.py --n_heads 64 --ffn_dim 28672 --channels-per-block 32 --pipeline-parallel --seqlen 2048 --GEMV reuse-GB --reuse-size 32 --trace-file ../trace/32_channels_per_device/microbench/Llama3-70B/trace_32_channels_per_block_seqlen_2048_full.txt --only-trace --Llama-GQA --full-accelerator-softmax --op-trace --flash-attention --flash-attention-block-size 1024 --systolic-pim --systolic-dim 8 --batch-size 8
# norm trace, which can be offloaded to Accelerator
python3 function_sim.py --n_heads 64 --ffn_dim 28672 --channels-per-block 32 --pipeline-parallel --seqlen 2048 --GEMV reuse-GB --reuse-size 32 --trace-file ../trace/32_channels_per_device/microbench/Llama3-70B/trace_32_channels_per_block_seqlen_2048_norm.txt --only-trace --Llama-GQA --full-accelerator-softmax --trace-norm --flash-attention --flash-attention-block-size 1024  --systolic-pim --systolic-dim 8 --batch-size 8
# FC and Activation trace
python3 function_sim.py --n_heads 64 --ffn_dim 28672 --channels-per-block 32 --pipeline-parallel --seqlen 2048 --GEMV reuse-GB --reuse-size 32 --trace-file ../trace/32_channels_per_device/microbench/Llama3-70B/trace_32_channels_per_block_seqlen_2048_FC.txt --only-trace --Llama-GQA --full-accelerator-softmax --trace-fc-kqvo --trace-fc-ffn --trace-activation --flash-attention --flash-attention-block-size 1024  --systolic-pim --systolic-dim 8 --batch-size 8
# Attention trace
python3 function_sim.py --n_heads 64 --ffn_dim 28672 --channels-per-block 32 --pipeline-parallel --seqlen 2048 --GEMV reuse-GB --reuse-size 32 --trace-file ../trace/32_channels_per_device/microbench/Llama3-70B/trace_32_channels_per_block_seqlen_2048_attention.txt --only-trace --Llama-GQA --full-accelerator-softmax --trace-attention --flash-attention --flash-attention-block-size 1024  --systolic-pim --systolic-dim 8 --batch-size 8



channels=32
seqlen=2048
# full trace
python3 function_sim.py --n_heads 64 --ffn_dim 28672 --channels-per-block ${channels} --pipeline-parallel --seqlen ${seqlen} --GEMV reuse-GB --reuse-size 32 --trace-file ../trace/32_channels_per_device/microbench/Llama3-70B/trace_${channels}_channels_per_block_seqlen_${seqlen}_full.txt --only-trace --Llama-GQA --full-accelerator-softmax --op-trace --flash-attention --flash-attention-block-size 4096 --systolic-pim --systolic-dim 8 --batch-size 8
# norm trace, which can be offloaded to Accelerator
python3 function_sim.py --n_heads 64 --ffn_dim 28672 --channels-per-block ${channels} --pipeline-parallel --seqlen ${seqlen} --GEMV reuse-GB --reuse-size 32 --trace-file ../trace/32_channels_per_device/microbench/Llama3-70B/trace_${channels}_channels_per_block_seqlen_${seqlen}_norm.txt --only-trace --Llama-GQA --full-accelerator-softmax --trace-norm --flash-attention --flash-attention-block-size 4096  --systolic-pim --systolic-dim 8 --batch-size 8
# FC and Activation trace
python3 function_sim.py --n_heads 64 --ffn_dim 28672 --channels-per-block ${channels} --pipeline-parallel --seqlen ${seqlen} --GEMV reuse-GB --reuse-size 32 --trace-file ../trace/32_channels_per_device/microbench/Llama3-70B/trace_${channels}_channels_per_block_seqlen_${seqlen}_FC.txt --only-trace --Llama-GQA --full-accelerator-softmax --trace-fc-kqvo --trace-fc-ffn --trace-activation --flash-attention --flash-attention-block-size 4096  --systolic-pim --systolic-dim 8 --batch-size 8
# Attention trace
python3 function_sim.py --n_heads 64 --ffn_dim 28672 --channels-per-block ${channels} --pipeline-parallel --seqlen ${seqlen} --GEMV reuse-GB --reuse-size 32 --trace-file ../trace/32_channels_per_device/microbench/Llama3-70B/trace_${channels}_channels_per_block_seqlen_${seqlen}_attention.txt --only-trace --Llama-GQA --full-accelerator-softmax --trace-attention --flash-attention --flash-attention-block-size 4096  --systolic-pim --systolic-dim 8 --batch-size 8

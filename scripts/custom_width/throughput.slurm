#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/EffViT/outs/profile/throughput/b1_b3_l3_width_res_fp16.out
#SBATCH --nodes=1   # Get one node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1           # And two GPU
#SBATCH --cpus-per-task=8            # Two cores per task
#SBATCH --job-name=TPAll
#SBATCH --constraint=gmem12

source /home/aaryang/anaconda3/bin/activate
conda activate evit

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

models=("b1_custom" "b3_custom" "l3_custom")
width_multipliers=("0.25" "0.50" "0.75" "1.0")
resolutions=("128" "160" "192" "224")

for model in "${models[@]}"; do 
    for width in "${width_multipliers[@]}"; do
        for res in  "${resolutions[@]}"; do

        echo "$model || w : $width || res : $res"
        python3 ../../find_max_throughput.py \
            --image_size $res \
            --width_multiplier $width \
            --student_model $model \
            --fp16 true
    
        done
    done
done 

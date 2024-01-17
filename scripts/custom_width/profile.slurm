#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/EffViT/outs/profile/macs/b1_b3_l3_depths_multires.out
#SBATCH --nodes=1   # Get one node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1           # And two GPU
#SBATCH --cpus-per-task=8            # Two cores per task
#SBATCH --job-name=B1Profile
#SBATCH --constraint=gmem12

source /home/aaryang/anaconda3/bin/activate
conda activate evit

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

models=("b1_custom" "b3_custom" "l3_custom")
width_multipliers=("0.25" "0.50" "0.75" "1.0")
resolutions=("128" "160" "192" "224")

for model in "${models[@]}"; do
    for width_multiplier in "${width_multipliers[@]}"; do
        for resolution in "${resolutions[@]}"; do
            echo "$model || depth : $width_multiplier x || res : $resolution"
            python3 ../../model_profile.py --image_size $resolution \
                --depth_multiplier $width_multiplier \
                --student_model $model \
                --find_macs true \
                --profile false
        done
    done
done
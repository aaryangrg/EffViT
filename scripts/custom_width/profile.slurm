#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/EffViT/outs/profile/throughput/b1_0.75x_224.out
#SBATCH --nodes=1   # Get one node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1           # And two GPU
#SBATCH --cpus-per-task=8            # Two cores per task
#SBATCH --job-name=Throughput
#SBATCH --constraint=gmem12

source /home/aaryang/anaconda3/bin/activate
conda activate evit

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

models=("b1_custom" "b3_custom" "l3_custom")
width_multipliers=("0.25" "0.50" "0.75" "1.0")
resolutions=("128" "160" "192" "224")

# for model in "${models[@]}"; do
#     for width_multiplier in "${width_multipliers[@]}"; do
#         for resolution in "${resolutions[@]}"; do
#             echo "$model || depth : $width_multiplier x || res : $resolution"
#             python3 ../../model_profile.py --image_size $resolution \
#                 --width_multiplier 1.0 \
#                 --student_model $model \
#         done
#     done
# done

# for model in "${models[@]}"; do 
#     for width in "${width_multipliers[@]}"; do
#         echo "$model - $width x"
#         python3 ../../model_profile.py --image_size 224 \
#                 --width_multiplier $width \
#                 --student_model $model
#     done
# done

python3 ../../model_profile.py --image_size 224 \
        --width_multiplier 0.75 \
        --student_model b1_custom
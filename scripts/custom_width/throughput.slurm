#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/EffViT/outs/profile/throughput/b1_0.75x.out
#SBATCH --nodes=1   # Get one node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1           # And two GPU
#SBATCH --cpus-per-task=8            # Two cores per task
#SBATCH --job-name=B1xTP
#SBATCH --constraint=gmem12

source /home/aaryang/anaconda3/bin/activate
conda activate evit

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

models=("b1_custom" "b3_custom" "l3_custom")
width_multipliers=("0.25" "0.50" "0.75" "1.0")

python3 ../../find_max_throughput.py \
    --image_size 224 \
    --width_multiplier 0.75 \
    --student_model b1_custom \

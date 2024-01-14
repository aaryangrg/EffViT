#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/EffViT/outs/profile/b1_1x.out
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

python3 ../../model_profile.py --model b1_custom --image_size 224 \
    --weight_url ../../exp/custom_width/distillation/b3_b1_0.75w_r224/checkpoint/model_best.pt \
    --batch_size 128 \
    --path /home/c3-0/datasets/ImageNet/validation \
    --reduced_width true \
    --student_model b1_custom
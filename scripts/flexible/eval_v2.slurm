#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/EffViT/outs/flexible/valv2/b3_pretrained_4widths_224.out
#SBATCH --nodes=1   # Get one node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2            # And two GPUs
#SBATCH --cpus-per-task=8            # Eight cores per task
#SBATCH --job-name=B1Val2
#SBATCH --constraint=gmem12

source /home/aaryang/anaconda3/bin/activate
conda activate evit

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
nvidia-smi
width_multipliers=("0.25" "0.50" "0.75" "1.0")

for width in "${width_multipliers[@]}"; do
    echo "$width x"
    python3 ../../eval_cls_v2.py --model b3 --image_size 224 \
        --weight_url ../../b3-224.pt \
        --batch_size 128 \
        --path /home/c3-0/datasets/ImageNet/validation \
        --width_multiplier $width \
        --flexible_width true
done
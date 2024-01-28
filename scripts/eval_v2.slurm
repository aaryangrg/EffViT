#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/EffViT/outs/valv2/b1_finetuned_224.out
#SBATCH --nodes=1   # Get one node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2            # And two GPU
#SBATCH --cpus-per-task=8            # Two cores per task
#SBATCH --job-name=B1Val2
#SBATCH --constraint=gmem12

source /home/aaryang/anaconda3/bin/activate
conda activate evit

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

width_multipliers=("0.25" "0.50" "0.75")
resolutions=("128" "160" "192" "224")


for width in "${width_multipliers[@]}"; do
    for res in "${resolutions[@]}"; do
        echo "WIDTH : $width x || RES : $res"
        python3 ../eval_cls_v1.py --model b1 --image_size $res \
            --weight_url ../exp/flexible/distillation/miniimgnet/b1_mutualv1_224/checkpoint/model_best.pt \
            --batch_size 128 \
            --path /home/c3-0/datasets/ImageNet/train \
            --flexible_width true \
            --width_multiplier $width \
    done 
done

#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/EffViT/outs/custom_width/valv2/dist_b1_width_res.out
#SBATCH --nodes=1   # Get one node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2            # And two GPU
#SBATCH --cpus-per-task=8            # Two cores per task
#SBATCH --job-name=DisthalfVal
#SBATCH --constraint=gmem12

source /home/aaryang/anaconda3/bin/activate
conda activate evit

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

width_multipliers=("0.25" "0.5" "0.75")
resolutions=("128" "160" "192" "224")


for width in "${width_multipliers[@]}"; do
    for res in "${resolutions[@]}"; do

        python3 ../../eval_cls_v2.py --model b1_custom --image_size $res \
            --weight_url ../../exp/custom_width/train/b1_r224_train_${width}w/checkpoint/model_best.pt \
            --batch_size 128 \
            --path /home/c3-0/datasets/ImageNet/validation \
            --reduced_width true \
            --width_multiplier $width \

    done
done
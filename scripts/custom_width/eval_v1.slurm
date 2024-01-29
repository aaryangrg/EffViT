#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/EffViT/outs/custom_width/valv1/dist_b1_res_width.out
#SBATCH --nodes=1   # Get one node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2            # And two GPU
#SBATCH --cpus-per-task=8            # Two cores per task
#SBATCH --job-name=Valv1
#SBATCH --constraint=gmem12

source /home/aaryang/anaconda3/bin/activate
conda activate evit

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

# width_multipliers=("0.25" "0.5" "0.75")
resolutions=("128" "160" "192" "224")

# for width in "${width_multipliers[@]}"; do
#     for res in "${resolutions[@]}"; do

#         python3 ../../eval_cls_v1.py --model b1_custom --image_size $res \
#             --weight_url ../../exp/custom_width/train/b1_r224_train_${width}w/checkpoint/model_best.pt \
#             --batch_size 128 \
#             --path /home/c3-0/datasets/ImageNet/train \
#             --reduced_width true \
#             --width_multiplier $width \

#     done
# done

# Distillation - subwidths
combinations=("0.25 b3_b1_0.25w_450epochs_r224" "0.50 b3_b1_0.5w_450epochs_r224" "0.75 b3_b1_0.75w_r224")
for combination in "${combinations[@]}"; do
    IFS=' ' read -ra params <<< "$combination"
    width=${params[0]}
    path=${params[1]}
    for res in "${resolutions[@]}"; do
        echo "Dist || width : $width || res : $res "
        python3 ../../eval_cls_v1.py --model b1_custom --image_size $res \
            --weight_url ../../exp/custom_width/distillation/$path/checkpoint/model_best.pt \
            --batch_size 128 \
            --path /home/c3-0/datasets/ImageNet/train \
            --reduced_width true \
            --width_multiplier $width \

    done
done

#Distillation - 1x
for res in "${resolutions[@]}"; do
    echo "Dist || width : 1x || res : $res "
    python3 ../../eval_cls_v1.py --model b1_custom --image_size $res \
        --weight_url ../../exp/distillation/miniimgnet/b3_b1_r224/checkpoint/model_best.pt \
        --batch_size 128 \
        --path /home/c3-0/datasets/ImageNet/train \
        --reduced_width true \
        --width_multiplier 1.0 \

done

#Training - 1x
for res in "${resolutions[@]}"; do
    echo "Train || width : 1x || res : $res "
    python3 ../../eval_cls_v1.py --model b1_custom --image_size $res \
        --weight_url ../../exp/train/miniimgnet/b1_r224_train/checkpoint/model_best.pt \
        --batch_size 128 \
        --path /home/c3-0/datasets/ImageNet/train \
        --reduced_width true \
        --width_multiplier 1.0 \

done

#Pre-trained - Only resolution change
for res in "${resolutions[@]}"; do
    echo "Pretrained || width : 1x || res : $res "
    python3 ../../eval_cls_v1.py --model b1_custom --image_size $res \
        --weight_url ../../pretrained/b1-r224.pt \
        --batch_size 128 \
        --path /home/c3-0/datasets/ImageNet/train \
        --reduced_width true \
        --width_multiplier 1.0 \

done
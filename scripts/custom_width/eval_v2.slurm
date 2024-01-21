#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/EffViT/outs/custom_width/valv2/b3_b1_0.75w_r224.out
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

python3 ../../eval_cls_v2.py --model b1_custom --image_size 224 \
    --weight_url ../../exp/custom_width/train/b1_r224_train_0.25w/checkpoint/model_best.pt \
    --batch_size 128 \
    --path /home/c3-0/datasets/ImageNet/validation \
    --reduced_width true \
    --width_multiplier 0.25 \

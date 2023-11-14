#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/EffViT/outs/valv2/b1_miniimgnet_training.out
#SBATCH --nodes=1   # Get one node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2            # And two GPU
#SBATCH --cpus-per-task=8            # Two cores per task
#SBATCH --job-name=B1ScratchVal
#SBATCH --constraint=gmem12

source /home/aaryang/anaconda3/bin/activate
conda activate evit

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

python3 ../eval_cls_v2.py --model b1-r224 --image_size 224 \
    --weight_url  ../train/miniimgnet/b1_r224_train/checkpoint/model_best.pt \
    --batch_size 128 \
    --path /home/c3-0/datasets/ImageNet/validation

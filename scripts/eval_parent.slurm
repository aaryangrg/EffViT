#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/EffViT/outs/parent/b3_subset_cls.out
#SBATCH --nodes=1   # Get one node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2            # And two GPU
#SBATCH --cpus-per-task=8            # Two cores per task
#SBATCH --job-name=CLSParent
#SBATCH --constraint=gmem12

source /home/aaryang/anaconda3/bin/activate
conda activate evit

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

python3 ../eval_subset_cls.py --model b3-r224 --image_size 224 \
    --weight_url /home/aaryang/experiments/EffViT/pretrained/b3-r224.pt \
    --subset_classes 200 \
    --batch_size 128 \
    --path /home/c3-0/datasets/ImageNet/validation

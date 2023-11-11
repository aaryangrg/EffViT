#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/EffViT/outs/train/b1_mini_ce.out
#SBATCH --nodes=1   # Get one node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=TrainCE
#SBATCH --constraint=gmem32

source /home/aaryang/anaconda3/bin/activate
conda activate evit

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

torchpack dist-run -np 1 \
python ../train_cls_mini_imagenet.py ../configs/cls/imagenet/b1.yaml  --fp16 \
    --path ../exp/train/miniimgnet/b1_r224_train_ce/ \
    --data_provider.train_batch_size 768 \
#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/EffViT/outs/b1_224.out
#SBATCH --nodes=1   # Get one node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2            # And two GPU
#SBATCH --cpus-per-task=8            # Two cores per task
#SBATCH --job-name=CLSKD224
#SBATCH --constraint=gmem16

source /home/aaryang/anaconda3/bin/activate
conda activate evit

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

torchpack dist-run -np 1 \
python3 ../train_kd.py ../configs/cls/imagenet/b1.yaml --fp16  \
    --path ../exp/cls/imagenet/b1_r224/ \
    --parent_model b3 \
    --parent_weights_url ../pretrained/b3-224.pt \
    --use_subset false \
    --gpu 0,1

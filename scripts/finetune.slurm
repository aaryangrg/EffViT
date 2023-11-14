##!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/EffViT/outs/train/b1_pretrained_mini.out
#SBATCH --nodes=1   # Get one node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --job-name=MiniTrain
#SBATCH --constraint=gmem16

source /home/aaryang/anaconda3/bin/activate
conda activate evit

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

torchpack dist-run -np 1 \
python ../train_cls_mini_imagenet.py ../configs/cls/imagenet/b1.yaml  --fp16 \
    --path ../exp/train/miniimgnet/b1_pretrained_r224_train/ \
    --data_provider.train_batch_size 512 \
    --run_config.bce true \
    --student_weights ../pretrained/b1-r224.pt \

#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/EffViT/outs/custom_width/train/b1_w_0.75_miniimgnet.out
#SBATCH --nodes=1   # Get one node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=0.75WidthTrain
#SBATCH --constraint=gmem24

source /home/aaryang/anaconda3/bin/activate
conda activate evit

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

torchpack dist-run -np 1 \
python ../../train_cls_mini_imagenet.py ../../configs/cls/imagenet/b1.yaml  --fp16 \
    --path ../../exp/custom_width/train/b1_r224_train_0.75w/ \
    --data_provider.train_batch_size 1024 \
    --run_config.bce true \
    --reduced_width true \
    --width_multiplier 0.75 \
    --student_model b1_custom

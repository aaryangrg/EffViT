#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/EffViT/outs/dist/b3_b1_224_distillation.out
#SBATCH --nodes=1   # Get one node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1         # And two GPU
#SBATCH --cpus-per-task=10            # Two cores per task
#SBATCH --job-name=MiniDist
#SBATCH --constraint=gmem32

source /home/aaryang/anaconda3/bin/activate
conda activate evit

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

torchpack dist-run -np 1 \
python3 ../train_kd.py ../configs/cls/imagenet/b1.yaml  --fp16 \
    --path ../exp/distillation/miniimgnet/b3_b1_r224/ \
    --parent_model b3 \
    --parent_weights_url /home/aaryang/experiments/EffViT/pretrained/b3-r224.pt \
    --use_subset true \
    --data_provider.train_batch_size 512 \
    --student_weights /home/aaryang/experiments/EffViT/pretrained/b1-r224.pt
 

#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/EffViT/outs/flexible/dist/b1_mutualv1_224_batchchange.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2      # And two GPU
#SBATCH --cpus-per-task=10            # Two cores per task
#SBATCH --job-name=Distributed
#SBATCH --constraint=gmem24

source /home/aaryang/anaconda3/bin/activate
conda activate evit

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

torchpack dist-run -np 1 \
python3 ../../mutual_train.py ../../configs/cls/imagenet/b1.yaml \
    --path ../exp/flexible/distillation/miniimgnet/b1_mutualv1_224_batchchange/ \
    --data_provider.base_batch_size 384 \
    --data_provider.n_worker 12 \
    --student_model b1 \
    --run_config.base_lr 0.00025
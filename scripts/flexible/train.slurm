#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/EffViT/outs/flexible/dist/distributed_test.out
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
python3 ../../mutual_train.py ../../configs/cls/imagenet/b1.yaml --fp16 \
    --path ../exp/flexible/distillation/miniimgnet/distributed_test/ \
    --data_provider.train_batch_size 1024 \
    --data_provider.n_worker 10 \
    --student_model b1 \
    --run_config.base_lr 0.00025
#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/EffViT/outs/flexible/dist/b1_4ths_mutliway_training_fp32.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1       # And two GPU
#SBATCH --cpus-per-task=10            # Two cores per task
#SBATCH --job-name=DistFlex
#SBATCH --constraint=gmem32

source /home/aaryang/anaconda3/bin/activate
conda activate evit

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

torchpack dist-run -np 1 \
python3 ../../mutual_train.py ../../configs/cls/imagenet/b1.yaml \
    --path ../exp/flexible/distillation/miniimgnet/b1_4ths_mutliway_training_fp32/ \
    --parent_model b3 \
    --parent_weights_url /home/aaryang/experiments/EffViT/pretrained/b3-r224.pt \
    --data_provider.train_batch_size 256 \
    --data_provider.n_worker 10 \
    --student_model b1 \
    --run_config.base_lr 0.00025
#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/EffViT/outs/gdino/backbone/test.out
#SBATCH --nodes=1   # Get one node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=FDinoTest
#SBATCH --constraint=gmem11

source /home/aaryang/anaconda3/bin/activate
conda activate evit

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

torchpack dist-run -np 1 \
python ../../gdino_back_distill.py ../../configs/cls/imagenet/b1.yaml \
    --path ../../exp/gdino/backbone/test/ \
    --config_file /home/aaryang/experiments/Open-GDINO/config/cfg_coco.py\
    --datasets /home/aaryang/experiments/Open-GDINO/config/coco_od_datasets.json\
    --pretrain_model_path /home/aaryang/experiments/Open-GDINO/weights/groundingdino_swint_ogc.pth \

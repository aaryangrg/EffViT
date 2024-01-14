#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/EffViT/outs/model_params/b1_b3_l3.out  # Get one node
#SBATCH --nodes=1   # Get one node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=ParamTot
#SBATCH --constraint=gmem24

source /home/aaryang/anaconda3/bin/activate
conda activate evit

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
nvidia-smi

models=("b1_custom" "b3_custom" "l3_custom")
width_multipliers=("0.25" "0.50" "0.75" "1.0")

for model in "${models[@]}"; do
    for width_multiplier in "${width_multipliers[@]}"; do
        echo "$model - $width_multiplier x"

        torchpack dist-run -np 1 \
        python ../../model_params.py ../../configs/cls/imagenet/b1.yaml  --fp16 \
            --path ../../exp/custom_width/train/b1_r224_train_0.5w/ \
            --data_provider.train_batch_size 128 \
            --run_config.bce true \
            --reduced_width true \
            --width_multiplier $width_multiplier \
            --student_model $model
    done
done

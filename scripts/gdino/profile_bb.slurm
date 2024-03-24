#!/bin/bash
#SBATCH --account=group3
#SBATCH --output=/home/aaryang/experiments/Open-GDINO/effvit_expts/profile/backbonev1.out
#SBATCH --nodes=1   # Get one node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1           # And two GPU
#SBATCH --cpus-per-task=8            # Two cores per task
#SBATCH --job-name=Profile
#SBATCH --constraint=gmem11

source /home/aaryang/anaconda3/bin/activate
conda activate opdino

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

nvidia-smi

resolutions=("320" "480" "1333")

for resolution in "${resolutions[@]}"; do
    echo "Resolution : $resolution"
    python3 ../../model_profile.py --image_size $resolution 

done

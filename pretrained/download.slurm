#!/bin/bash
#SBATCH --output=/home/aaryang/experiments/CIRKD/model_weights/checkpoints.out
#SBATCH --account=group3
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --job-name=checkpoints
source /home/aaryang/anaconda3/bin/activate
conda activate cirkd

# 224 x 224
URL_EFFICIENT_B1_IMAGE_NET="https://drive.google.com/u/0/uc?id=1hKN_hvLG4nmRzbfzKY7GlqwpR5uKpOOk&export=download"
URL_EFFICIENT_B2_IMAGE_NET="https://drive.google.com/u/0/uc?id=1DiM-iqVGTrq4te8mefHl3e1c12u4qR7d&export=download"
URL_EFFICIENT_B3_IMAGE_NET="https://drive.google.com/u/0/uc?id=18RZDGLiY8KsyJ7LGic4mg1JHwd-a_ky6&export=download"

gdown $URL_EFFICIENT_B2_IMAGE_NET
gdown $URL_EFFICIENT_B1_IMAGE_NET
gdown $URL_EFFICIENT_B3_IMAGE_NET
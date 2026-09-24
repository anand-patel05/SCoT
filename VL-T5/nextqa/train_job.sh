#!/bin/bash
#SBATCH --job-name=nextqa_tv_G5                  
#SBATCH --partition=a40                                 
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=logs14/%x_%j.out
#SBATCH --error=logs14/%x_%j.err

conda activate /home/vis-comp/24m2119/miniconda3/envs/vqacl310
cd /home/vis-comp/24m2119/HypVQACL

# bash VL-T5/nextqa/train.sh 1
bash VL-T5/nextqa/train_tv.sh 1

# for run: sbatch ./VL-T5/nextqa/train_job.sh

# #SBATCH --nodelist=cn39-a40

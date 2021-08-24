#!/bin/bash

#SBATCH --partition debug
#SBATCH --time 00-00:01:00
#SBATCH --qos gpu
#SBATCH --gres gpu:1

set -e

module load gcc/8.4.0-cuda
module load cuda/10.2.89
module list

echo date = `date`
echo
pwd
echo
sinfo
echo
hostname
echo
nvidia-smi

echo _DONE_

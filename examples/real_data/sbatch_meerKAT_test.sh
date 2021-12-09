#!/bin/bash

#SBATCH --time 00-02:00:00
#SBATCH --partition gpu
#SBATCH --qos gpu_free
#SBATCH --gres gpu:4
#SBATCH --mem 16G

set -e

module load gcc
module load cuda
module list


python meerKAT_test.py

#!/bin/bash

#SBATCH --partition build
#SBATCH --time 00-00:15:00
#SBATCH --qos gpu
#SBATCH --gres gpu:1
#SBATCH --mem 40G

set -e

module load gcc/8.4.0-cuda
module load cuda/10.2.89
module list

source pypeline.sh --no_shell
which python
python -V
hostname

python "./examples/simulation/lofar_toothbrush_ps.py"

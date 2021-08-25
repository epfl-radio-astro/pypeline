#!/bin/bash

#SBATCH --partition build
#SBATCH --time 00-00:15:00
#SBATCH --qos gpu
#SBATCH --gres gpu:1

set -e

module load gcc/8.4.0-cuda
module load cuda/10.2.89
module list

source pypeline.sh --no_shell
which python
python -V

time python ./examples/simulation/lofar_toothbrush_ps.py

echo _DONE_

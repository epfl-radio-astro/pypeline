#!/bin/bash
set -e

module load gcc/8.4.0-cuda
module load cuda/10.2.89
module list

# Asssumes miniconda is installed and available in $PATH
# (~/.bashrc modified by installer)
conda update -n base -c defaults conda --yes

# Delete and recreate conda environment;
# (needs to be activated if conda requirements are updated)
if [ 1 -eq 0 ]; then
    conda remove --name pypeline --all --yes
    conda create --name=pypeline \
                 --channel=defaults \
                 --channel=conda-forge \
                 --file=conda_requirements.txt \
                 --yes
fi

pwd
conda env list

source ./pypeline.sh --no_shell
conda env list

python -V
pip install --upgrade setuptools
pip install cupy-cuda102

git clone https://github.com/imagingofthings/pyFFS.git
cd pyFFS/
git checkout v1.0
python3 setup.py develop
cd ..


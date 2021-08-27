#!/bin/bash
set -e

module load gcc/8.4.0-cuda
module load cuda/10.2.89
module list

# Asssumes conda is installed and available in $PATH
export PATH=$HOME/miniconda3/bin/:$PATH
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
conda list

python -V

#pip install -e . #EO: dead slow (from dev branch)
#pip uninstall .

#exit
#pip install --upgrade setuptools

# to wipe up 
#pip freeze > req_delete_all.txt
#pip uninstall -r req_delete_all.txt -y

pip install cupy-cuda102

if [ 1 -eq 1 ]; then

    if [ -d pyFFS ]; then
        cd pyFFS
        python3 setup.py develop --user --uninstall
        cd ..
        rm -rf pyFFS/
    fi
    git clone https://github.com/imagingofthings/pyFFS.git
    cd pyFFS/
    git checkout v1.0
    python3 setup.py develop --user
    cd ..
    
    # ImoT_tools: check out 'dev' branch, otherwise no fits.py under io/
    if [ -d ImoT_tools ]; then
        cd ImoT_tools
        python3 setup.py develop --user --uninstall
        cd ..
        rm -rf ImoT_tools
    fi
    git clone https://github.com/imagingofthings/ImoT_tools.git
    cd ImoT_tools/
    #git checkout v1.0
    git checkout dev
    python3 setup.py develop --user
    cd ..
    
    if [ -d pycsou ]; then
        pip uninstall pycsou --yes
        rm -rf pycsou
    fi
    git clone https://github.com/matthieumeo/pycsou.git
    cd pycsou
    #git checkout tags/v1.0.5 #EO: lofar_toothbrush_ps.py uses head version (eps in call to mappeddistancematrix)
    cat requirements.txt
    #pip install -e .
    pip install .
    cd ..
fi

echo;
pwd
ls -l
echo

#python3 setup.py develop --user --uninstall
#python3 setup.py develop --user
pip uninstall pypeline
pip install -e .


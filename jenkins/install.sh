#!/bin/bash

set -e

pwd

# Install Miniconda in batch mode the first time
bash ./Miniconda3-latest-Linux-x86_64.sh -b


ENV_NAME=pype-111

conda env create -f ./conda_environments/pype-111.yml

conda env list

eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

which pip
pip --version

which python
python -V

# Install non-conda packages
pip install cupy-cuda111
pip install pycsou --no-deps
pip install pyFFS --no-deps

# Install dev branch of ImoT_tools
IMOT_TOOLS=ImoT_tools
if [ -d $IMOT_TOOLS ]; then
    cd $IMOT_TOOLS
    if [ `git symbolic-ref --short HEAD` != 'dev' ]; then
        echo "Fatal: $IMOT_TOOLS already existing but not on dev branch. Exit."
        exit 1
    fi
else
    git clone https://github.com/imagingofthings/ImoT_tools.git
    cd $IMOT_TOOLS
    git checkout dev
fi
pip install --no-deps .
cd ..


# Install FINUFTT (CPU) from source
# !!! GCC 8 not recommended !!! but fftw not available for GCC 9...
module load gcc fftw
if [ -d finufft ]; then
    echo "A finufft directory already exits. Will not do anything."
    #git pull + recomp?
    #rm -rf?
else
    git clone https://github.com/flatironinstitute/finufft.git
    cd finufft
    # Only if you want to have debug symbol/info included in bin
    echo "CXXFLAGS += -g -DFFTW_PLAN_SAFE" > make.inc
    make test -j
    ###make perftest
    make python
    cd ..
fi


# Install pypeline locally in editable mode
pip install --no-deps -e .


# Export newly created environment (commenting out imot-tools)
#printf -v date '%(%Y%m%d_%H%M%S)T' -1
#ENV_YML=${ENV_NAME}_environment_${date}.yml
#conda env export > $ENV_YML
#sed -e '/imot-tools/ s/^#*/#/' -i $ENV_YML
#echo "Exported newly created environment $ENV_NAME to $ENV_YML."


conda deactivate

echo "Conda installation of environment $ENV_NAME complete."


# To remove the environment
# conda remove --name $ENV_NAME --all #--yes



exit 0
### OLD STUFF BELOW

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
    #EO: use head, otherwise periodic analysis is broken
    #git checkout v1.0
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
pip uninstall pypeline --yes
pip install .


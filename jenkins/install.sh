#!/bin/bash

set -e


#echo "Warning: early exit. Re-enable if a new installation is required."
#exit 0


# Install Miniconda in batch mode the first time
# (step only required once)
if [ 1 == 0 ]; then
    rm -rf ~/miniconda3
    bash ./Miniconda3-latest-Linux-x86_64.sh -b
    #source ~/miniconda3/bin/activate
    #conda init
    #sed -i.bak '/~\/miniconda3\/bin:/d' ~/.bashrc
    #cat ~/.bashrc
fi

conda config --set auto_activate_base false

which conda -a
conda env list


ENV_NAME=pype-111
# Create conda environment
# (step only required to create the environment)
#conda remove --name $ENV_NAME --all --yes
#conda env create -f ./conda_environments/pype-111.yml

#eval "$(conda shell.bash hook)"
source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda env list
conda activate $ENV_NAME
conda env list

which pip
pip --version

which python
python -V




# Install non-conda packages
pip install cupy-cuda110
exit 0

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
    echo "A finufft directory already exits. Will clean, pull, and recompile."
    cd finufft
    make clean
    git pull
else
    git clone https://github.com/flatironinstitute/finufft.git
    cd finufft
fi
# Only if you want to have debug symbol/info included in bin
echo "CXXFLAGS += -g -DFFTW_PLAN_SAFE" > make.inc
make test -j
###make perftest
make python
cd ..

# Install pypeline locally in editable mode
#pip install --no-deps -e .
pwd
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

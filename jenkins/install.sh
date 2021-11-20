#!/bin/bash

set -e

# Set symbolic links to a solution that serves as a reference
#
if [ 1 == 0 ]; then
    REF_SOL=${WORK_DIR}/${GIT_BRANCH}/2021-11-18T15-32-16Z_160
    [ -d $REF_SOL ] || (echo "Error: reference directory $REF_SOL not found." && exit 1)
    echo REF_SOL = $REF_SOL
    
    echo REF_DIR = $REF_DIR
    [ -d $REF_DIR ] || (echo "Error: reference directory $REF_DIR not found." && exit 1)
    
    rm -r $REF_DIR/test_standard_cpu
    rm -r $REF_DIR/test_standard_gpu
    rm -r $REF_DIR/lofar_bootes_nufft3
    rm -r $REF_DIR/lofar_bootes_nufft_small_fov
    
    ln -s $REF_SOL/test_standard_cpu            $REF_DIR/test_standard_cpu
    ln -s $REF_SOL/test_standard_gpu            $REF_DIR/test_standard_gpu
    ln -s $REF_SOL/lofar_bootes_nufft3          $REF_DIR/lofar_bootes_nufft3
    ln -s $REF_SOL/lofar_bootes_nufft_small_fov $REF_DIR/lofar_bootes_nufft_small_fov
fi


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

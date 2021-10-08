#!/bin/bash

# Assumes that Miniconda was installed, and conda added to $PATH

set -e

ENV_NAME=pype102_intel

conda create --name=$ENV_NAME --channel=intel --channel=defaults --channel=conda-forge --file=new_conda_requirements_intel.txt --yes

conda env list 

eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

which pip
pip --version

which python
python -V

# Install non-conda packages
pip install cupy-cuda102
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

# Install pypeline locally in editable mode
pip install --no-deps -e .


# Export newly created environment (commenting out imot-tools)
printf -v date '%(%Y%m%d_%H%M%S)T' -1
ENV_YML=${ENV_NAME}_environment_${date}.yml
conda env export > $ENV_YML
sed -e '/imot-tools/ s/^#*/#/' -i $ENV_YML
echo "Exported newly created environment $ENV_NAME to $ENV_YML."


conda deactivate

echo "Conda installation of environment $ENV_NAME complete."


# To remove the environment
# conda remove --name $ENV_NAME --all #--yes


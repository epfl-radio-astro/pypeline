#!/bin/bash

set -e

module load gcc
module load fftw
module load cuda/11.0.2;

CONDA_ENV=pype-111
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV


# Output directory must be defined and existing
if [[ -z "${DOC_DIR}" ]]; then
    echo "Error: DOC_DIR unset. Must point to an existing directory."
    exit 1
else 
    if [[ ! -d "${DOC_DIR}" ]]; then
        echo "Error: DOC_DIR must point to an existing directory."
        exit 1
    fi
fi

# Build the doc
python setup.py egg_info
python setup.py build_sphinx

# Copy the doc to /work build's directory
echo WORKSPACE = ${WORKSPACE}
cp -rv ${WORKSPACE}/build/html ${DOC_DIR}

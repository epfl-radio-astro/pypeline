#!/bin/bash

set -e

# $1: absolute path to bluebild Bash library
# ------------------------------------------
source "$1"

bb_load_stack gcc
bb_activate_venv

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


# To test on the command line
# $ WORKSPACE="." DOC_DIR="/tmp" sh jenkins/build_documentation.sh

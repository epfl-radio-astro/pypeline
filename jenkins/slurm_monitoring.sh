#!/bin/bash

set -e

module load gcc/8.4.0-cuda
module load cuda/11.1.1
CONDA_ENV=pype111
module list

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV
conda env list

echo "WORK_DIR   = ${WORK_DIR}"
echo "GIT_BRANCH = ${GIT_BRANCH}"
echo "BUILD_ID   = ${BUILD_ID}"
echo "TEST_DIR   = ${TEST_DIR}"
echo "TEST_FSTAT = ${TEST_FSTAT}" 

OUTPUT_DIR=${TEST_DIR:-.}     # default to cwd when ENV[TEST_DIR] not set
echo OUTPUT_DIR = $OUTPUT_DIR

# fail fast with set -e, outdir must exist
ls $OUTPUT_DIR 

python ./jenkins/tts.py -i ${WORK_DIR}/${GIT_BRANCH} -o $OUTPUT_DIR -f $TEST_FSTAT -b $BUILD_ID


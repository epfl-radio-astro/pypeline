#!/bin/bash

set -e

CONDA_ENV=pype-111
module list

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV
conda env list

echo
echo "WORK_DIR         = ${WORK_DIR}"
echo "REF_DIR          = ${REF_DIR}"
echo "GIT_BRANCH       = ${GIT_BRANCH}"
echo "BUILD_ID         = ${BUILD_ID}"
echo "TEST_DIR         = ${TEST_DIR}"
echo "TEST_IGNORE_UPTO = ${TEST_IGNORE_UPTO}"
echo "TEST_FSTAT_RT    = ${TEST_FSTAT_RT}" 
echo "TEST_FSTAT_IMG   = ${TEST_FSTAT_IMG}" 
OUTPUT_DIR=${TEST_DIR:-.}     # default to cwd when ENV[TEST_DIR] not set
echo OUTPUT_DIR = $OUTPUT_DIR
echo 

# fail fast with set -e, outdir must exist
ls $OUTPUT_DIR

python ./jenkins/tts.py  --input_directory ${WORK_DIR}/${GIT_BRANCH} --output_directory $OUTPUT_DIR --stat_file $TEST_FSTAT_RT --last_build $BUILD_ID  --ignore_up_to $TEST_IGNORE_UPTO

python ./jenkins/imap.py --input_directory ${WORK_DIR}/${GIT_BRANCH} --output_directory $OUTPUT_DIR --stat_file $TEST_FSTAT_IMG --last_build $BUILD_ID  --ignore_up_to $TEST_IGNORE_UPTO --reference_directory $REF_DIR


# To test locally
# ---------------
#cd to pypeline
#export BUILD_ID=21 GIT_BRANCH=ci-master OUTPUT_DIR=/tmp/ TEST_FSTAT_RT=/tmp/file_rt.tst TEST_FSTAT_IMG=/tmp/file_img.tst TEST_IGNORE_UPTO=0 WORK_DIR=/work/backup/ska/ci-jenkins/izar-ska/ REF_DIR=/work/backup/ska/ci-jenkins/references/ TEST_DIR=.
#sh ./jenkins/slurm_monitoring.sh

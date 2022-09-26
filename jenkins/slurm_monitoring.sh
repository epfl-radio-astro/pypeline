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

python ./jenkins/tts.py -i ${WORK_DIR}/${GIT_BRANCH}  -o $OUTPUT_DIR -f $TEST_FSTAT_RT  -b $BUILD_ID -s $TEST_IGNORE_UPTO

python ./jenkins/imap.py -i ${WORK_DIR}/${GIT_BRANCH} -o $OUTPUT_DIR -f $TEST_FSTAT_IMG -b $BUILD_ID -s $TEST_IGNORE_UPTO -r $REF_DIR

#EO: to test locally
#conda activate pype-111
#BUILD_ID=1 GIT_BRANCH=eo_jenkins OUT_DIR=/tmp/ TEST_FSTAT_IMG=/tmp/file.tst TEST_IGNORE_UPTO=0 WORK_DIR=/work/backup/ska/ci-jenkins/izar-ska/ REF_DIR=/work/backup/ska/ci-jenkins/references/ bash jenkins/slurm_monitoring.sh

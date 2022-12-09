#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
    echo "-E- Exactly 1 argument is expected: the path to the Bluebild Bash library"
fi

# $1: absolute path to bluebild Bash library
# ------------------------------------------
echo "\$1 = \"$1\" (\$1 must point to Bluebild Bash library)"
bb_bash_lib=$(realpath "$1")
source "${bb_bash_lib}"

bb_activate_venv

bb_bash_dir=$(dirname "${bb_bash_lib}")
echo "-I- bb_bash_dir = $bb_bash_dir"

echo
echo "WORK_DIR         = ${WORK_DIR}"
echo "REF_DIR          = ${REF_DIR}"
echo "GIT_BRANCH       = ${GIT_BRANCH}"
echo "BUILD_ID         = ${BUILD_ID}"
echo "UTC_TAG          = ${UTC_TAG}"
echo "TEST_DIR         = ${TEST_DIR}"
echo "TEST_IGNORE_UPTO = ${TEST_IGNORE_UPTO}"
echo "TEST_FSTAT_RT    = ${TEST_FSTAT_RT}" 
echo "TEST_FSTAT_IMG   = ${TEST_FSTAT_IMG}" 
OUTPUT_DIR=${TEST_DIR:-.}     # defaults to cwd when ENV[TEST_DIR] not set
echo OUTPUT_DIR = $OUTPUT_DIR
echo 

# fail fast with set -e, outdir must exist
ls $OUTPUT_DIR

[ -z "${UTC_TAG}" ] && (echo "-E- Environment variable UTC_TAG not defined!"; exit 1)

LAST_BUILD=$(echo "${UTC_TAG}" | cut -d "Z" -f 1 | sed "s/[-T]//g")
IGNORE_UP_TO=$(echo "${TEST_IGNORE_UPTO}" | cut -d "Z" -f 1 | sed "s/[-T]//g")

python $bb_bash_dir/tts.py  --input_directory ${WORK_DIR}/${GIT_BRANCH} --output_directory $OUTPUT_DIR --stat_file $TEST_FSTAT_RT --last_build $LAST_BUILD --ignore_up_to $IGNORE_UP_TO

python $bb_bash_dir/imap.py --input_directory ${WORK_DIR}/${GIT_BRANCH} --output_directory $OUTPUT_DIR --stat_file $TEST_FSTAT_IMG --last_build $LAST_BUILD --ignore_up_to $IGNORE_UP_TO --reference_directory $REF_DIR


# To test locally
# ---------------
# cd to pypeline/jenkins
# UTC_TAG=2022-09-26T06-00-38Z BUILD_ID=139 GIT_BRANCH=ci-tmp01 OUTPUT_DIR=/tmp/ TEST_FSTAT_RT=/tmp/file_rt.tst TEST_FSTAT_IMG=/tmp/file_img.tst TEST_IGNORE_UPTO=0 WORK_DIR=/work/ska/ci-jenkins/izar-ska/ REF_DIR=/work/backup/ska/ci-jenkins/references/ TEST_DIR=. sh ./slurm_monitoring.sh ./install.sh

#!/bin/bash

set -e

echo REF_DIR = $REF_DIR
[ -d $REF_DIR ] || (echo "Error: reference directory $REF_DIR not found." && exit 1)


# 06/04/2022: New reference for LBSSi/t solutions as grid was changed
if [ 1 == 0 ]; then
    REF_SOL=${WORK_DIR}/${GIT_BRANCH}/2022-03-30T07-03-54Z_34/
    [ -d $REF_SOL ] || (echo "Error: reference directory $REF_SOL not found." && exit 1)
    echo REF_SOL = $REF_SOL

    [ -L $REF_DIR/lofar_bootes_ss ] && rm $REF_DIR/lofar_bootes_ss

    ln -s $REF_SOL/lofar_bootes_ss $REF_DIR/lofar_bootes_ss
fi



# 29/11/2021: Jenkins history was lost after renaming of the workspace
#             so it started again from build no 1
# => renaming the current solution directory
#mv -v /work/backup/ska/ci-jenkins/izar-ska/eo_jenkins/  /work/backup/ska/ci-jenkins/izar-ska/eo_jenkins_old

# 29/11/2021: Set symbolic links to an earlier solution that serves as a reference
#
if [ 1 == 0 ]; then

    REF_SOL=${WORK_DIR}/${GIT_BRANCH}/2021-11-29T17-36-15Z_3/
    [ -d $REF_SOL ] || (echo "Error: reference directory $REF_SOL not found." && exit 1)
    echo REF_SOL = $REF_SOL
        
    [ -L $REF_DIR/test_standard_cpu ]            && rm $REF_DIR/test_standard_cpu
    [ -L $REF_DIR/test_standard_gpu ]            && rm $REF_DIR/test_standard_gpu 
    [ -L $REF_DIR/lofar_bootes_ss ]              && rm $REF_DIR/lofar_bootes_ss
    [ -L $REF_DIR/lofar_bootes_nufft3 ]          && rm $REF_DIR/lofar_bootes_nufft3
    [ -: $REF_DIR/lofar_bootes_nufft_small_fov ] && rm $REF_DIR/lofar_bootes_nufft_small_fov
    
    ln -s $REF_SOL/test_standard_cpu            $REF_DIR/test_standard_cpu
    ln -s $REF_SOL/test_standard_gpu            $REF_DIR/test_standard_gpu
    ln -s $REF_SOL/lofar_bootes_ss              $REF_DIR/lofar_bootes_ss
    ln -s $REF_SOL/lofar_bootes_nufft3          $REF_DIR/lofar_bootes_nufft3
    ln -s $REF_SOL/lofar_bootes_nufft_small_fov $REF_DIR/lofar_bootes_nufft_small_fov
fi



exit 0

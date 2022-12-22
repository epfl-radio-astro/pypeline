#!/bin/bash

### Example
### source bluebild.sh && time bb_{name_of_function}

set -x
ROOT=`pwd` # Jenkins branch's workspace (checkout of pypeline Git repository)
PYPELINE_ROOT=$ROOT
if [ $USER == "orliac" ]; then    # for local dev
    ROOT=~/SKA/epfl-radio-astro
    PYPELINE_ROOT=$ROOT/pypeline
fi
export PYPELINE_ROOT=${PYPELINE_ROOT}
NINJA_DIR=$ROOT/ninja
umpire_DIR=$ROOT/Umpire/inst/usr/local
export FINUFFT_ROOT=$ROOT/finufft
export CUFINUFFT_ROOT=$ROOT/cufinufft
export MARLA_ROOT=$ROOT/marla
IMOT_TOOLS_ROOT=$ROOT/ImoT_tools
export PATH=$NINJA_DIR:$FINUFFT_ROOT:$CUFINUFFT_ROOT/bin:$umpire_DIR/bin:$PATH
export LD_LIBRARY_PATH=$FINUFFT_ROOT/lib:$CUFINUFFT_ROOT/lib:$umpire_DIR/lib:$LD_LIBRARY_PATH
#export ICCCFG="/work/ska/soft/intel/icc.cfg"
#export ICPCCFG="/work/ska/soft/intel/icpc.cfg"
#unset ICCCFG
#unset ICPCCFG
set +x

# Set Python virtual environment name
VENV_NAME=PYPE1102


function bb_activate_venv() {
    source $ROOT/$VENV_NAME/bin/activate
    export PYTHONPATH=${PYPELINE_ROOT}/jenkins:${PYTHON_PATH}
}

function bb_load_stack() {
    COMPILER=$1
    case $COMPILER in
        intel)
            bb_load_intel_stack
            ;;
        gcc)
            bb_load_gcc_stack
            ;;
        *)
            echo "-E- Fatal. Unknown software stack. Use intel or gcc."
            return 1
            ;;
    esac
}

function bb_load_gcc_stack() {
    module purge
    module load gcc/9.3.0-cuda
    module load python/3.7.7
    module load cuda/11.0.2
    module load openblas/0.3.10-openmp
    module load mvapich2/2.3.4
    module load fftw/3.3.8-mpi-openmp
    module load cmake
}

function bb_load_intel_stack() {
    module purge
    module load intel/19.0.5
    module load python/3.7.7
    module load intel-mpi/2019.5.281
    module load intel-mkl/2019.5.281
    module load fftw/3.3.8-mpi-openmp
    module load cuda/11.0.2
    module load cmake
    export LD_PRELOAD=${LD_PRELOAD}:$MKLROOT/lib/intel64/libmkl_def.so:$MKLROOT/lib/intel64/libmkl_avx2.so:$MKLROOT/lib/intel64/libmkl_core.so:$MKLROOT/lib/intel64/libmkl_intel_lp64.so:$MKLROOT/lib/intel64/libmkl_intel_thread.so
    export LD_PRELOAD=${LD_PRELOAD}:$INTEL_MKL_ROOT/lib/intel64/libiomp5.so
}

function bb_create_python_venv() {
    ENTRY=`pwd`
    bb_load_stack $1 || return 1
    cd $ROOT
    [ -d $VENV_NAME ] && rm -r ./$VENV_NAME
    python -m venv $VENV_NAME
    source $VENV_NAME/bin/activate
    pip install --upgrade pip
    pip install \
        numpy   astropy healpy \
        numexpr pandas  pybind11 \
        scipy   pbr     pyproj \
        plotly  nvtx \
        python-casacore cupy-cuda110 \
        bluebild_tools  tqdm \
        tk sphinx sphinx_rtd_theme \
        scikit-learn
    pip install --no-deps \
        pycsou  pyFFS
    deactivate
    module purge
    cd $ENTRY
}

function bb_pip_install() {
    if [[ "$VIRTUAL_ENV" == "${ROOT}/${VENV_NAME}" ]]; then
        pip install $@
        pip list
    else
        echo "Warning: not installing $@ because expected venv not activated"
    fi
}

# Umpire
# https://umpire.readthedocs.io/en/develop/sphinx/getting_started.html#installation
#
function bb_install_umpire() {
    ENTRY=`pwd`
    bb_load_stack $1 || return 1
    cd $ROOT
    [ -d Umpire ] && rm -rf Umpire
    git clone --recursive https://github.com/LLNL/Umpire.git
    cd Umpire
    mkdir build && cd build
    cmake -DENABLE_CUDA=On ../
    make -j
    make DESTDIR=../inst install
    module purge
    cd $ENTRY
}

function bb_install_finufft {
    ENTRY=`pwd`
    bb_load_stack $1 || return 1
    bb_activate_venv
    cd $ROOT
    [ -d finufft ] && rm -rf finufft
    git clone https://github.com/flatironinstitute/finufft.git
    cd finufft
    if [ `pwd` != ${FINUFFT_ROOT} ]; then
        echo "Error: check FINUFFT setup. Mismatch between $PWD and expected $FINUFFT_ROOT"
        cd $ENTRY
        return 1
    fi
    # Only if you want to have debug symbol/info included in bin
    echo "CXXFLAGS += -g -DFFTW_PLAN_SAFE" > make.inc
    make test -j
    ###make perftest
    make python
    module purge
    deactivate
    cd $ENTRY
}

# Function to install CUFINUFFT from Simon's fork
#
function bb_install_cufinufft {
    ENTRY=`pwd`
    bb_load_stack $1 || return 1
    cd $ROOT
    [ -d cufinufft ] && rm -rf ./cufinufft
    git clone https://github.com/AdhocMan/cufinufft.git
    cd cufinufft
    if [ `pwd` != ${CUFINUFFT_ROOT} ]; then
        echo "Error: check CUFINUFFT setup. Mismatch between $PWD and expected $CUFINUFFT_ROOT"
        cd $ENTRY
        return 1
    fi
    git branch
    git fetch --all
    git checkout t3_d3
    git branch
    echo "CXXFLAGS  += -g" > make.inc
    echo "NVCCFLAGS += -g" >> make.inc
    cat make.inc
    make all -j
    module purge
    cd $ENTRY
}

function bb_install_ninja {
    ENTRY=`pwd`
    [ -d $NINJA_DIR ] || mkdir -pv $NINJA_DIR
    cd $NINJA_DIR
    rm -f *
    wget https://github.com/ninja-build/ninja/releases/download/v1.11.0/ninja-linux.zip
    unzip ninja-linux.zip
    cd $ENTRY
}

function bb_set_compilation_frame() {
    COMPILER=$1
    case $COMPILER in
        intel)
            bb_load_intel_stack            
            BLUEBILD_CMAKE_ARGS="-DMARLA_ROOT=$MARLA_ROOT \
                                 -DBLUEBILD_BUILD_TYPE=DEBUG \
                                 -DCMAKE_CXX_FLAGS_DEBUG=\"-g -DNDEBUG -fPIC -m64 -Ofast -qopenmp -qopt-report=2 -qopt-report-phase=vec -xCORE-AVX512 -qopt-zmm-usage=high\" \
                                 -DCMAKE_CUDA_FLAGS_DEBUG=\"-g -lineinfo\""
            ;;
        gcc)
            bb_load_gcc_stack
            BLUEBILD_CMAKE_ARGS="-DMARLA_ROOT=$MARLA_ROOT \
                                 -DBLUEBILD_BUILD_TYPE=DEBUG \
                                 -DCMAKE_CXX_FLAGS_DEBUG=\"-g -Ofast -march=skylake-avx512 -mprefer-vector-width=512 -ftree-vectorize\" \
                                 -DCMAKE_CUDA_FLAGS_DEBUG=\"-g -lineinfo\""
            ;;
        *)
            echo "-E- Fatal. Unknown software stack. Use intel or gcc."
            return 1
            ;;
    esac
    echo "${BLUEBILD_CMAKE_ARGS}"
}

# Argument 1: 'intel' or 'gcc' to choose the software stack
#
function bb_install_bluebild() {
    BLUEBILD_CMAKE_ARGS=$(bb_set_compilation_frame $1)
    ENTRY=`pwd`
    bb_load_stack $1 || return 1
    module list
    bb_activate_venv
    cd $PYPELINE_ROOT/src/bluebild
    [ -d _skbuild ] && rm -r _skbuild
    echo ${BLUEBILD_CMAKE_ARGS}
    BLUEBILD_CMAKE_ARGS=${BLUEBILD_CMAKE_ARGS} pip install -v --no-deps .
    deactivate
    module purge
    cd $ENTRY
}

function bb_compile_bluebild() {
    BLUEBILD_CMAKE_ARGS=$(bb_set_compilation_frame $1)
    ENTRY=`pwd`
    bb_load_stack $1 || return 1
    module list
    bb_activate_venv
    cd $PYPELINE_ROOT/src/bluebild
    BLUEBILD_CMAKE_ARGS=${BLUEBILD_CMAKE_ARGS} pip install -v --no-deps .
    deactivate
    module purge
    cd $ENTRY
}

function bb_install_pypeline() {
    ENTRY=`pwd`
    bb_load_stack $1 || return 1
    bb_activate_venv
    cd $PYPELINE_ROOT
    pip install -v --no-deps .
    deactivate
    module purge
    cd $ENTRY
}

function bb_install_marla() {
    ENTRY=`pwd`
    cd $ROOT
    pwd
    ls -l
    [ -d marla ] && rm -rf marla
    git clone https://gitlab.com/ursache/marla.git
    cd marla
    if [ `pwd` != ${MARLA_ROOT} ]; then
        echo "Error: check MARLA setup. Mismatch between $PWD and expected $MARLA_ROOT"
        cd $ENTRY
        return 1
    fi
    #EO: use dev branch as it contains fixes for the floor anad floorh
    git checkout dev
    cd $ENTRY
}

# Installs dev branch of ImoT_tools
#
function bb_install_imot_tools() {
    ENTRY=`pwd`
    bb_activate_venv
    cd $ROOT
    pwd
    ls -l
    [ -d ImoT_tools ] && rm -rf ImoT_tools
    git clone https://github.com/imagingofthings/ImoT_tools.git
    cd ImoT_tools
    if [ `pwd` != ${IMOT_TOOLS_ROOT} ]; then
        echo "Error: check IMOT_TOOLS setup. Mismatch between $PWD and expected $IMOT_TOOLS_ROOT"
        cd $ENTRY
        return 1
    fi
    git checkout dev
    pip install --no-deps -e .
    deactivate
    cd $ENTRY
}

function bb_print_env() {
    # || true to avoid failure when grep returns nothing under set -e
    echo; echo
    env | grep THREADS || true
    echo
    env | grep SLURM || true
    echo; echo
}

function bb_print_jenkins_env() {
    echo
    echo "------------------------------ Jenkins environment variables"
    echo "TEST_ALGO        = ${TEST_ALGO}"
    echo "TEST_BENCH       = ${TEST_BENCH}" 
    echo "TEST_DIR         = ${TEST_DIR}"
    echo "TEST_SEFF        = ${TEST_SEFF}"
    echo "TEST_TRANGE      = ${TEST_TRANGE}"
    echo "PROFILE_CPROFILE = ${PROFILE_CPROFILE}"
    echo "PROFILE_NSIGHT   = ${PROFILE_NSIGHT}"
    echo "PROFILE_VTUNE    = ${PROFILE_VTUNE}"
    echo "PROFILE_ADVISOR  = ${PROFILE_ADVISOR}"
    echo "------------------------------------------------------------"
    echo
}

# Output directory must be defined and existing
function bb_check_output_dir() {
    local TEST_DIR=$1
    if [[ -z "${TEST_DIR}" ]]; then
        echo "Error: TEST_DIR unset. Must point to an existing directory."
        return 1
    else 
        if [[ ! -d "${TEST_DIR}" ]]; then
            echo "Error: TEST_DIR must point to an existing directory."
            return 1
        fi
    fi
}

function bb_source_vtune() {
    source /work/scitas-ge/richart/test_stacks/syrah/v1/opt/spack/linux-rhel7-skylake_avx512/gcc-8.4.0/intel-oneapi-vtune-2021.6.0-34ym22fgautykbgmg5hhgkiwrvbwfvko/setvars.sh --force
}

function bb_source_advisor() {
    source /work/scitas-ge/richart/test_stacks/syrah/v1/opt/spack/linux-rhel7-skylake_avx512/gcc-8.4.0/intel-oneapi-advisor-2021.4.0-any7cfov5s4ujprr7plf7ks7xzoyqljz/setvars.sh --force
}

# For bb_vtune_{functions}:
#     $1: output directory
#     $2: absolute path to Python interpreter
#     $3: absolute path to Python script
#    $4+: arguments to Python script

function bb_vtune_hotspots() {
    echo; echo "### Running Intel VTune Amplifier hotspots analysis"
    echo args = $@
    bb_source_vtune
    #vtune -collect hotspots -run-pass-thru=--no-altstack -strategy ldconfig:notrace:notrace -source-search-dir=. -search-dir=. -result-dir="$1"/vtune_hs  -- "$2" "$3" "${flags}"
    vtune -collect hotspots -run-pass-thru=--no-altstack -strategy ldconfig:notrace:notrace -result-dir="$1"/vtune_hs  -- "$2" "$3" ${@:4}
    echo; echo
}

function bb_vtune_hpc_performance() {
    echo; echo "### Running Intel VTune Amplifier hpc-performance analysis"
    echo args = $@
    bb_source_vtune
    #vtune -collect hpc-performance -run-pass-thru=--no-altstack -strategy ldconfig:notrace:notrace -source-search-dir=. -search-dir=. -result-dir="$1"/vtune_hpc  -- "$2" "$3" "${flags}"
    vtune -collect hpc-performance -run-pass-thru=--no-altstack -strategy ldconfig:notrace:notrace -result-dir="$1"/vtune_hpc  -- "$2" "$3" ${@:4}
    echo; echo
}

function bb_vtune_memory_consumption() {
    echo; echo "### Running Intel VTune Amplifier memory-consumption analysis"
    echo args = $@
    bb_source_vtune
    #vtune -collect memory-consumption -run-pass-thru=--no-altstack -strategy ldconfig:notrace:notrace -source-search-dir=. -search-dir=. -result-dir="$1"/vtune_mem  -- "$2" "$3" "${flags}"
    vtune -collect memory-consumption -run-pass-thru=--no-altstack -strategy ldconfig:notrace:notrace -result-dir="$1"/vtune_mem  -- "$2" "$3" ${@:4}
    echo; echo
}

function bb_advisor_roofline() {
    echo; echo "### Running Intel Advisor roofline analysis"
    echo args = $@
    bb_source_advisor
    ADVIXE_RUNTOOL_OPTIONS=--no-altstack advixe-cl -collect roofline --enable-cache-simulation --profile-python -project-dir "$1"/advisor -search-dir src:=. -- "$2" "$3" ${@:4}
    echo; echo
}

function bb_check_input_file() {
    if [ -z "$1" ]; then
        echo "Error: argument holding path to Python script to run is expected"
        exit 1
    fi
    if [ ! -f "$1" ]; then
        echo "Error: input file not found! >>$1<<"
        exit 1
    fi
}

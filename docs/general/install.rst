.. ############################################################################
.. install.rst
.. ===========
.. Author : Sepand KASHANI [kashani.sepand@gmail.com]
.. ############################################################################


Installation
============

The following describes how `pipeline` and `bluebild` (with its C++ CPU and
GPU ports) can be installed on the EPFL's GPU cluster 
`izar <https://www.epfl.ch/research/facilities/scitas/hardware/izar/>`_ using
the available GCC 9.3.0 software stack.

* Load the following modules::

    module purge
    module load gcc/9.3.0-cuda
    module load python/3.7.7
    module load cuda/11.0.2
    module load openblas/0.3.10-openmp
    module load mvapich2/2.3.4
    module load fftw/3.3.8-mpi-openmp
    module load cmake

* Create a Python virtual environment $VENV_NAME::

    python -m venv $VENV_NAME
    source $VENV_NAME/bin/activate
    pip install --upgrade pip
    pip install \
        numpy   astropy healpy \
        numexpr pandas  pybind11 \
        scipy   pbr     pyproj \
        plotly  sklearn nvtx \
        python-casacore cupy-cuda110 \
        bluebild_tools  tqdm \
        tk sphinx sphinx_rtd_theme
    pip install --no-deps \
        pycsou  pyFFS

* Install `FINUFFT <https://finufft.readthedocs.io/en/latest/index.html>`_:

  Official installation instructions can be found 
  `here <https://finufft.readthedocs.io/en/latest/install.html>`_ ::

    git clone https://github.com/flatironinstitute/finufft.git
    cd finufft
    # Only if you want to have debugging information
    echo "CXXFLAGS += -g -DFFTW_PLAN_SAFE" > make.inc
    make test -j
    ###make perftest
    make python

* Install Simon Frasch's fork of `cuFINUFFT <https://github.com/AdhocMan/cufinufft>`_:
  
  Simon's fork contains an implementation for 3D FFT of type 3 which is not
  available from the `official cuFINUFFT <https://github.com/flatironinstitute/cufinufft>`_. ::

    git clone -b t3_d3 --single-branch https://github.com/AdhocMan/cufinufft.git
    cd cufinufft
    # Only if you want to have debugging information
    echo "CXXFLAGS  += -g" > make.inc
    echo "NVCCFLAGS += -g" >> make.inc
    cat make.inc
    make all -j

* Install `Ninja <https://ninja-build.org/>`_:

  Ninja will be installed in ``$NINJA_DIR``, a location of your choice. ::

    mkdir -pv $NINJA_DIR
    cd $NINJA_DIR
    rm -f *
    wget https://github.com/ninja-build/ninja/releases/download/v1.11.0/ninja-linux.zip
    unzip ninja-linux.zip
  

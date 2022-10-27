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

* Create a Python virtual environment `$VENV_NAME`::

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



Remarks
-------

Pypeline is developed and tested on x86_64 systems running Linux.
It should also run correctly on macOS, but we provide no support for this.

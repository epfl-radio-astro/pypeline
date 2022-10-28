.. ############################################################################
.. install.rst
.. ===========
.. Author : Sepand KASHANI [kashani.sepand@gmail.com]
.. ############################################################################


Installation
============

The following describes how ``pypeline`` and ``bluebild`` (with its C++ CPU and
GPU ports) can be installed on EPFL's GPU cluster 
`izar <https://www.epfl.ch/research/facilities/scitas/hardware/izar/>`_ using
the available GCC 9.3.0 software stack.

It is advised to choose a root location where to install ``pypeline`` (and ``bluebild``)
and all their dependencies. This location is referred to as ``$ROOT`` hereinafter.


.. _modules:

* Load the following modules::

    module purge
    module load gcc/9.3.0-cuda
    module load python/3.7.7
    module load cuda/11.0.2
    module load openblas/0.3.10-openmp
    module load mvapich2/2.3.4
    module load fftw/3.3.8-mpi-openmp
    module load cmake


* Create a Python virtual environment ``$VENV_NAME``::

    cd $ROOT
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

    cd $ROOT
    git clone https://github.com/flatironinstitute/finufft.git
    cd finufft
    # Only if you want to have debugging information
    echo "CXXFLAGS += -g -DFFTW_PLAN_SAFE" > make.inc
    make test -j
    ###make perftest
    make python


* Install S. Frasch's fork of `cuFINUFFT <https://github.com/AdhocMan/cufinufft>`_:
  
  Simon's fork contains an implementation for 3D FFT of type 3 which is not
  available from the `official cuFINUFFT <https://github.com/flatironinstitute/cufinufft>`_. ::

    cd $ROOT
    git clone -b t3_d3 --single-branch https://github.com/AdhocMan/cufinufft.git
    cd cufinufft
    # Only if you want to have debugging information
    echo "CXXFLAGS  += -g" > make.inc
    echo "NVCCFLAGS += -g" >> make.inc
    cat make.inc
    make all -j


* Install `Ninja <https://ninja-build.org/>`_::

    cd $ROOT
    wget https://github.com/ninja-build/ninja/releases/download/v1.11.0/ninja-linux.zip
    unzip ninja-linux.zip


* Install G. Fourestey's `Marla <https://gitlab.com/ursache/marla>`_ library:

  Marla will be installed in ``$ROOT``. We use branch ``dev`` as it contains 
  some bug fixes for functions ``floor`` and ``floorh``. ::

    cd $ROOT
    git clone https://gitlab.com/ursache/marla.git
    cd marla
    git checkout dev


* Install `ImoT_tools <https://github.com/imagingofthings/ImoT_tools.git>`_:

  We install the ``dev`` branch of ImoT_tools. ::

    cd $ROOT
    source $VENV_NAME/bin/activate
    git clone -b dev --single-branch https://github.com/imagingofthings/ImoT_tools.git
    cd ImoT_tools
    pip install --no-deps .
    deactivate


* Intall `pypeline <https://github.com/epfl-radio-astro/pypeline>`_ from epfl-radio-astro's fork:

  Assumptions:
  
  1. Required `modules`_ are loaded
  2. $VENV_NAME Python virtual environment is activated

  .. code-block:: shell

     cd $ROOT
     git clone https://github.com/epfl-radio-astro/pypeline.git
     cd pypeline
     pip install -v --no-deps -e .



.. ############################################################################
.. manual.rst
.. ===========
.. Author : E. Orliac @EPFL
.. ############################################################################


Manual installation
###################

This pages illustrates how to install ``bluebild++`` in a standard HPC environment
where we assume that standard software and libraries are available via modules.
It is based on the environment that is currently available on EPFL's GPU cluster 
`izar <https://www.epfl.ch/research/facilities/scitas/hardware/izar/>`_.

.. note:: For the sake of simplicity, every installation is made in a root location
          referred to as ``$PROJ_ROOT`` hereinafter.


.. _modules:

* Load the following modules:

  .. code-block:: shell

    module purge
    module load gcc/9.3.0-cuda
    module load python/3.7.7
    module load cuda/11.0.2
    module load openblas/0.3.10-openmp
    module load mvapich2/2.3.4
    module load fftw/3.3.8-mpi-openmp
    module load cmake


* Create a Python virtual environment ``$VENV_NAME``:

  .. code-block:: shell

    cd $PROJ_ROOT
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

.. warning::

   In the following steps it is assumed that:
  
   1. Required `modules`_ are loaded
   2. $VENV_NAME Python virtual environment is activated



* Install `FINUFFT <https://finufft.readthedocs.io/en/latest/index.html>`_:

  Official installation instructions can be found 
  `here <https://finufft.readthedocs.io/en/latest/install.html>`_.

  .. code-block:: shell

    cd $PROJ_ROOT
    git clone https://github.com/flatironinstitute/finufft.git
    cd finufft
    # Only if you want to have debugging information
    echo "CXXFLAGS += -g -DFFTW_PLAN_SAFE" > make.inc
    make test -j
    ###make perftest
    make python


* Install S. Frasch's fork of `cuFINUFFT <https://github.com/AdhocMan/cufinufft>`_:
  
  Simon's fork contains an implementation for 3D FFT of type 3 which is not
  available from the `official cuFINUFFT <https://github.com/flatironinstitute/cufinufft>`_.

  .. code-block:: shell

    cd $PROJ_ROOT
    git clone -b t3_d3 --single-branch https://github.com/AdhocMan/cufinufft.git
    cd cufinufft
    # Only if you want to have debugging information
    echo "CXXFLAGS  += -g" > make.inc
    echo "NVCCFLAGS += -g" >> make.inc
    cat make.inc
    make all -j


* Install `Ninja <https://ninja-build.org/>`_:

  .. code-block:: shell

    cd $PROJ_ROOT
    wget https://github.com/ninja-build/ninja/releases/download/v1.11.0/ninja-linux.zip
    unzip ninja-linux.zip


* Install G. Fourestey's `Marla <https://gitlab.com/ursache/marla>`_ library:

  We use branch ``dev`` as it contains some bug fixes for functions ``floor`` and ``floorh``.

  .. code-block:: shell

    cd $PROJ_ROOT
    git clone https://gitlab.com/ursache/marla.git
    cd marla
    git checkout dev


* Install `ImoT_tools <https://github.com/imagingofthings/ImoT_tools.git>`_:

  We install the ``dev`` branch of ImoT_tools.

  .. code-block:: shell

    cd $PROJ_ROOT
    source $VENV_NAME/bin/activate
    git clone -b dev --single-branch https://github.com/imagingofthings/ImoT_tools.git
    cd ImoT_tools
    pip install --no-deps .


* Intall `pypeline <https://github.com/epfl-radio-astro/pypeline>`_ from epfl-radio-astro's fork:

  .. code-block:: shell

     cd $PROJ_ROOT
     git clone https://github.com/epfl-radio-astro/pypeline.git
     cd pypeline
     #EO: until PR to merge ci-master into master is done, use ci-master
     git checkout ci-master
     pip install -v --no-deps -e .


* Compile CPU/GPU C++ ports of ``bluebild``

  .. code-block:: shell

     cd $PROJ_ROOT/pypeline/src/bluebild
     BLUEBILD_CMAKE_ARGS="-DMARLA_ROOT=$PROJ_ROOT/marla" pip install -v --no-deps .

* Edit your ``.bashrc`` file with:

  .. code-block:: shell

     PROJ_ROOT=/path/to/your/project

     export PATH=$PROJ_ROOT/ninja:$PROJ_ROOT/cufinufft/bin:$PROJ_ROOT/Umpire/inst/usr/local/bin:$PATH
     export LD_LIBRARY_PATH=$PROJ_ROOT/finufft/lib:$PROJ_ROOT/cufinufft/lib:$PROJ_ROOT/Umpire/inst/usr/local/lib:$LD_LIBRARY_PATH

.. warning::

   1. Log out and log in again, or
   2. Resource your ``~/.bashrc`` file


Testing the installation
------------------------

Now you should be able to run example simulation pypelines such as `lofar_bootes_nufft3_cpp_data_proc.py <https://github.com/epfl-radio-astro/pypeline/blob/ci-master/examples/simulation/lofar_bootes_nufft3_cpp_data_proc.py>`_ or `lofar_bootes_ss_cpp.py <https://github.com/epfl-radio-astro/pypeline/blob/ci-master/examples/simulation/lofar_bootes_ss_cpp.py>`_.

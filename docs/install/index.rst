.. ############################################################################
.. index.rst
.. =========
.. Author : E. Orliac @EPFL
.. ############################################################################

####################
  Installation
####################

``bluebild++`` requires a certain number of dependencies to be installed before it
can be run. However, if you are eager to test ``bluebild++``, use the containers
we make available for both Docker and Singularity.


Required dependencies
=====================

Here are the dependencies needed to install ``bluebild++`` with its default
configuration:

* C++ compiler supporting C++17
* `CMake <https://cmake.org/>`_ (3.11 or newer)
* `CUDA <https://developer.nvidia.com/cuda-downloads>`_ (11.2 or newer)
* `FINUFFT <https://finufft.readthedocs.io/en/latest/index.html>`_
* `cuFINUFFT <https://github.com/AdhocMan/cufinufft>`_ (from Simon Frasch's fork, containining type 3 for dimension 3)
* `Python <https://www.python.org/downloads>`_ (3.9 or newer)
* `ImoT_tools <https://github.com/imagingofthings/ImoT_tools.git>`_ (``dev`` branch)


Optional dependencies
=====================

When compiling ``bluebild++`` there are certain options you can turn on, to
activate alternate libraries to the default ones. Those are:

* VC
* HIP

Python packages
===============

The following Python packages need to be installed. With respect to a manual installation,
we advise the user to create a Python virtual environment to encompass them (see 
:ref:`ref-manual-installation`).

.. hlist::
   :columns: 3
      
   * numpy
   * astropy
   * matplotlib
   * tqdm
   * pyproj
   * healpy
   * scikit-learn
   * pandas


.. important::
   See pages on :ref:`ref-manual-installation` and setting up a :ref:`ref-spack-environment` to get
   the dependencies installed.


Building ``bluebild++``
=======================

First, clone ``bluebild++`` Git repository and enter it::

  git clone -b cpp_new_imager https://github.com/AdhocMan/pypeline.git
  cd pypeline 


Default configuration
*********************

The default configuration corresponds to the CMake ``Release`` build type.
To install, run the following commands::

  pip install --verbose --user --no-deps --no-build-isolation ./src/bluebild
  pip install --user --no-deps --no-build-isolation -e .


Advanced Configuration
**********************
In addition to the normal options provided by CMake, ``bluebild++`` uses some
additional configuration arguments to control optional features and behavior.


.. toctree::
   :maxdepth: 2
   :hidden:
    
   manual
   spack_environment

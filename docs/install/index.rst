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

The following Python packages need to be installed. Wrt to a manual installation,
we advise the user to create a Python virtual environment for those.

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

Default configuration
*********************

Advanced Configuration
**********************
In addition to the normal options provided by CMake, ``bluebild++`` uses some
additional configuration arguments to control optional features and behavior.


.. toctree::
   :maxdepth: 2
   :hidden:
    
   manual
   spack_environment

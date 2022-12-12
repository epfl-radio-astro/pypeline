.. ############################################################################
.. index.rst
.. =========
.. Author : E. Orliac @EPFL
.. ############################################################################

####################
  Installation
####################

`bluebild++` requires a certain number of dependencies to be installed before it
can be run. However, if you are eager to test `bluebild++`, use the containers
we make available for both Docker and Singularity.


Required dependencies
=====================
Here are the dependencies needed to install `bluebild++` with its default
configuration:
* C++ compiler supporting C++17
* CMake (3.11 or newer)
* CUDA (11.2 or newer)
* FINUFFT
* cuFINUFFT (from Simon Frasch's fork, containining type 3 for dimension 3)
* Python (3.9 or newer)
* ImoT_tools


Optional dependencies
=====================
When compiling `bluebild++` there are certain options you can turn on, to
activate alternate libraries to the default ones. Those are:
* VC
* HIP


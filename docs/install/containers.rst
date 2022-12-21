.. ############################################################################
.. containers.rst
.. ===========
.. Author : E. Orliac @EPFL
.. ############################################################################

.. _ref-containers:

Containers
##########

If you are just eager to test ``bipp`` without willing to go through the
whole installation procedure, use the containers we make available for both
Docker and Singularity. They are respectively built upon this
`Dockefile <https://gist.github.com/AdhocMan/3a96dccdca6ecac9f6779b93747869f0>`_ and this
Singularity definition file `bipp.def <https://github.com/epfl-radio-astro/ska-containers/blob/main/bipp/singularity/bipp.def>`_.

To build the container, download and edit the definition file and run, for Docker::

  docker build -t bipp:v1 .

and for Singularity::

  singularity build --fakeroot bipp.sif bipp.def

Images we produced are made available through the `Docker Hub <https://hub.docker.com/>`_ 
and the `Sylabs Cloud Library <https://cloud.sylabs.io/library>`_.

To run the images please refer to :ref:`ref-running-from-containers`.

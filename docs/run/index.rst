.. ############################################################################
.. index.rst
.. =========
.. Author : E. Orliac @EPFL
.. ############################################################################

################
Running ``bipp``
################

Running from standard installation
==================================
If you either went through a :ref:`ref-manual-installation` or that you use
a :ref:`ref-spack-environment`, you are ready to run the example pipelines
provided with the source code. They are located in the ``examples`` directory
of your ``bipp`` installation and are separated between ``examples/real_data``
and ``examples/simulation``::

  python ./examples/simulation/lofar_bootes_nufft3_cpp

or::

  python ../examples/real_data/lofar_bootes_ss.py

.. _ref-running-from-containers:

Running from containers
=======================

Docker
------

Singularity
-----------

Pull the image::
  
  singularity pull --arch amd64 library://orliac/bipp/bipp:latest

This will download an Singularity Image Format (SIF) file called ``bipp_latest.sif``.

List example imaging pipelines which run over simulated data::

  singularity run bipp_latest.sif "ls -l pypeline/examples/simulation/*_cpp.py"
  -rw-rw-r-- 1 root root 6018 Nov 23 18:21 pypeline/examples/simulation/lofar_bootes_nufft3_cpp.py
  -rw-rw-r-- 1 root root 4664 Nov 23 18:21 pypeline/examples/simulation/lofar_bootes_ss_cpp.py

To run an example pipeline, proceed as follows. First, copy an example pipeline to your local filesystem::

  singularity exec --bind $PWD bipp_latest.sif cp -iv /project/pypeline/examples/simulation/lofar_bootes_ss_cpp.py .

Edit the file if you wish so. Then run it with::

  singularity run --nv --bind $PWD:/work bipp_latest.sif "cd /work && python lofar_bootes_ss_cpp.py"

It will produce a image ``test.png`` that should look like:

  .. image:: images/wrong_test.png
             
     Incorrect test image generated with the wrong ``STD`` filter (should be ``INV_SQ``).


Switching between GPU and CPU
=============================

``bipp`` will always use a GPU device for computing, if available. But
you can easily force it to compute on CPU by switching from "AUTO"matic mode
to "CPU" mode with::

  # Create context with selected processing unit.
  # Options are "AUTO", "CPU" and "GPU".
  ctx = bluebild.Context("CPU")

instead of the default::

  ctx = bluebild.Context("AUTO")

in the context creation (usually the first step of a ``bipp`` pipeline).

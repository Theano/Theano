Requirements
============

.. note::

    We only support the installation of the requirements through conda.

.. _BLAS: http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
.. _Python: http://www.python.org/

    Python_ >= 2.7 or >= 3.3 The development package (python-dev or
        python-devel on most Linux distributions) is recommended (see
        just below). Python 2.4 was supported up to and including the
        release 0.6. Python 2.6 was supported up to and including the
        release 0.8.2. Python 3 is supported past the 3.3 release.

    `NumPy <http://numpy.scipy.org/>`_ >= 1.9.1 < 1.11.1
        Earlier versions could work, but we don’t test it.

    `SciPy <http://scipy.org>`_ >= 0.14 < 0.17.1
        Only currently required for sparse matrix and special functions support, but highly recommended. SciPy >=0.8 could work, but earlier versions have known bugs with sparse matrices.

    `BLAS`_ installation (with Level 3 functionality)
        * **Recommended**: MKL, which is free through Conda. 
        * Alternatively, we suggest to install OpenBLAS, with the development headers (``-dev``, ``-devel``, depending on your Linux distribution).

**Optional requirements**

    |PlatformCompiler|
        **Highly recommended.** Theano can fall back on a NumPy-based Python execution model, but a C compiler allows for vastly faster execution.

    `nose <http://nose.readthedocs.io/en/latest/>`_ >= 1.3.0
        Recommended, to run Theano's test-suite.

    `Sphinx <http://sphinx.pocoo.org/>`_ >= 0.5.1, `pygments <http://pygments.org/>`_
        For building the documentation. LaTeX_ and dvipng_ are also necessary for math to show up as images.

    `pydot-ng <https://github.com/pydot/pydot-ng>`_
        To handle large picture for gif/images.

    `NVIDIA CUDA drivers and SDK`_
        **Highly recommended** Required for GPU code generation/execution on NVIDIA gpus. See instruction below.

    `libgpuarray`_
        Required for GPU/CPU code generation on CUDA and OpenCL devices (see: :ref:`gpuarray`.)

Requirements installation through Conda (recommended)
-----------------------------------------------------

Install Miniconda
^^^^^^^^^^^^^^^^^

Follow this `link <http://conda.pydata.org/miniconda.html>`__ to install Miniconda.

.. note::

    If you want fast compiled code (recommended), make sure you have g++ (Windows/Linux) or Clang (OS X) installed.

Install requirements and optional packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    conda install numpy scipy mkl <nose> <sphinx> <pydot-ng>

* Arguments between <...> are optional.


Install and configure the GPU drivers (recommended)
---------------------------------------------------

.. warning::

    OpenCL support is still minimal for now.

1. Install CUDA drivers

    * Follow `this link <https://developer.nvidia.com/cuda-downloads>`__
      to install the CUDA driver and the CUDA Toolkit.
    * You must reboot the computer after the driver installation.
    * Test that it was loaded correctly after the reboot, executing the
      command `nvidia-smi` from the command line.

    .. note::

        Sanity check: The *bin* subfolder should contain an *nvcc*
        program. This folder is called the *cuda root* directory.

2. Fix 'lib' path
    * Add the 'lib' subdirectory (and/or 'lib64' subdirectory if you have a
      64-bit OS) to your ``$LD_LIBRARY_PATH`` environment 
      variable.

3. Set Theano's config flags

    To use the GPU you need to define the *cuda root*. You can do it in one 
    of the following ways:

    * Define a $CUDA_ROOT environment variable to equal the cuda root directory, as in ``CUDA_ROOT=/path/to/cuda/root``, or
    * add a ``cuda.root`` flag to :envvar:`THEANO_FLAGS`, as in ``THEANO_FLAGS='cuda.root=/path/to/cuda/root'``, or
    * add a [cuda] section to your .theanorc file containing the option ``root = /path/to/cuda/root``.

.. _LaTeX: http://www.latex-project.org/
.. _dvipng: http://savannah.nongnu.org/projects/dvipng/
.. _NVIDIA CUDA drivers and SDK: http://developer.nvidia.com/object/gpucomputing.html
.. _libgpuarray: http://deeplearning.net/software/libgpuarray/installation.html

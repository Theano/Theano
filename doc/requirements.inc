Requirements
============

.. note::

    We only support the installation of the requirements through conda.

.. _BLAS: http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
.. _Python: http://www.python.org/
.. _LaTeX: http://www.latex-project.org/
.. _dvipng: http://savannah.nongnu.org/projects/dvipng/
.. _NVIDIA CUDA drivers and SDK: http://developer.nvidia.com/object/gpucomputing.html
.. _libgpuarray: http://deeplearning.net/software/libgpuarray/installation.html
.. _pycuda: https://mathema.tician.de/software/pycuda/
.. _skcuda: http://scikit-cuda.readthedocs.io/en/latest/
.. _warp-ctc: https://github.com/baidu-research/warp-ctc

    Python_ == 2.7* or ( >= 3.4 and < 3.6 )
        |PythonDistRecommended|. Python 2.4 was supported up to and including the
        release 0.6. Python 2.6 was supported up to and including the
        release 0.8.2. Python 3.3 was supported up to and including release 0.9.

    `NumPy <http://numpy.scipy.org/>`_ >= 1.9.1 <= 1.12
        Earlier versions could work, but we don’t test it.

    `SciPy <http://scipy.org>`_ >= 0.14 < 0.17.1
        Only currently required for sparse matrix and special functions support, but highly recommended. SciPy >=0.8 could work, but earlier versions have known bugs with sparse matrices.

    `BLAS`_ installation (with Level 3 functionality)
        * **Recommended**: MKL, which is free through Conda with ``mkl-service`` package.
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
        Required for GPU/CPU code generation on CUDA and OpenCL devices (see: :ref:`gpuarray`).

    `pycuda`_ and `skcuda`_
        Required for some extra operations on the GPU like fft and
        solvers. We use them to wrap cufft and cusolver. Quick install
        ``pip install pycuda scikit-cuda``. For cuda 8, the dev
        version of skcuda (will be released as 0.5.2) is needed for
        cusolver: ``pip install pycuda; pip install
        git+https://github.com/lebedov/scikit-cuda.git#egg=scikit-cuda``.

    `warp-ctc`_
        Required for :ref:`Theano CTC implementation
        <libdoc_tensor_nnet_ctc>`. It is faster then using an
        equivalent graph of Theano ops.


Requirements installation through Conda (recommended)
-----------------------------------------------------

Install Miniconda
^^^^^^^^^^^^^^^^^

Follow this `link <http://conda.pydata.org/miniconda.html>`__ to install Miniconda.

.. note::

    If you want fast compiled code (recommended), make sure you have |CompilerName| installed.

.. install_requirements_and_optional_packages

Install requirements and optional packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    conda install numpy scipy mkl <nose> <sphinx> <pydot-ng>

* Arguments between <...> are optional.

Package ``parameterized`` is also optional but may be required for unit testing. It is available via ``pip``.

.. code-block:: bash

    pip install parameterized

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
    * Add the CUDA 'lib' subdirectory (and/or 'lib64' subdirectory if you have a
      64-bit OS) to your ``$LD_LIBRARY_PATH`` environment 
      variable. Example: ``/usr/local/cuda/lib64``

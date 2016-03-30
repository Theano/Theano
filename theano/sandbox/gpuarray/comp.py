from __future__ import absolute_import, print_function, division
import os

import numpy

import theano
from theano import config

# This is a big hack to avoid creating a second context on the card.
from theano.sandbox.cuda.nvcc_compiler import (NVCC_compiler as NVCC_base,
                                               hash_from_file)


class NVCC_compiler(NVCC_base):
    @staticmethod
    def compile_args():
        """
        Re-implementation of compile_args that does not create an
        additionnal context on the GPU.

        """
        flags = [flag for flag in config.nvcc.flags.split(' ') if flag]
        if config.nvcc.fastmath:
            flags.append('-use_fast_math')
        cuda_ndarray_cuh_hash = hash_from_file(
            os.path.join(os.path.split(theano.sandbox.cuda.__file__)[0],
                         'cuda_ndarray.cuh'))
        flags.append('-DCUDA_NDARRAY_CUH=' + cuda_ndarray_cuh_hash)

        # numpy 1.7 deprecated the following macros but they didn't
        # exist in the past
        numpy_ver = [int(n) for n in numpy.__version__.split('.')[:2]]
        if bool(numpy_ver < [1, 7]):
            flags.append("-DNPY_ARRAY_ENSURECOPY=NPY_ENSURECOPY")
            flags.append("-DNPY_ARRAY_ALIGNED=NPY_ALIGNED")
            flags.append("-DNPY_ARRAY_WRITEABLE=NPY_WRITEABLE")
            flags.append("-DNPY_ARRAY_UPDATE_ALL=NPY_UPDATE_ALL")
            flags.append("-DNPY_ARRAY_C_CONTIGUOUS=NPY_C_CONTIGUOUS")
            flags.append("-DNPY_ARRAY_F_CONTIGUOUS=NPY_F_CONTIGUOUS")

        # If the user didn't specify architecture flags add them
        if not any(['-arch=sm_' in f for f in flags]):
            dev = theano.sandbox.gpuarray.init_dev.device
            if dev is None:
                raise Exception("Trying to compile GPU code without a context")
            if dev.startswith("opencl"):
                raise Exception("Trying to call nvcc with an OpenCL context")
            assert dev.startswith('cuda')
            if dev == 'cuda':
                n = theano.sandbox.cuda.use.device_number
            else:
                n = int(dev[4:])
            p = theano.sandbox.cuda.device_properties(n)
            flags.append('-arch=sm_' + str(p['major']) + str(p['minor']))

        return flags

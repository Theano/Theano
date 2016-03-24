from __future__ import absolute_import, print_function, division
import numpy

import theano
from theano import scalar, gof
from theano.tests.unittest_tools import SkipTest, assert_allclose

from theano.tensor.tests import test_elemwise

from .config import mode_with_gpu, test_ctx_name
from .test_basic_ops import rand_gpuarray
from ..elemwise import (GpuElemwise, GpuDimShuffle,
                        GpuCAReduceCuda, GpuCAReduceCPY)
from ..type import GpuArrayType, get_context

from pygpu import ndgpuarray as gpuarray


# This is acutally a test for GpuElemwise
class test_gpu_Broadcast(test_elemwise.test_Broadcast):
    op = GpuElemwise
    type = GpuArrayType
    cop = GpuElemwise
    ctype = GpuArrayType
    # The order is important
    linkers = [gof.PerformLinker, gof.CLinker]

    def setUp(self):
        if get_context(test_ctx_name).kind != 'cuda':
            self.linkers = [gof.PerformLinker]

    def rand_val(self, shp):
        return rand_gpuarray(*shp, **dict(cls=gpuarray))

    def rand_cval(self, shp):
        return rand_gpuarray(*shp, **dict(cls=gpuarray))

    def test_c(self):
        if get_context(test_ctx_name).kind != 'cuda':
            raise SkipTest("Cuda specific tests")
        super(test_gpu_Broadcast, self).test_c()

    def test_c_inplace(self):
        if get_context(test_ctx_name).kind != 'cuda':
            raise SkipTest("Cuda specific tests")
        super(test_gpu_Broadcast, self).test_c_inplace()


def test_elemwise_pow():
    # Test that GpuElemwise(pow) can compile with any combination of integer
    # or float input dtype.
    if get_context(test_ctx_name).kind != 'cuda':
        raise SkipTest("Cuda specific tests")

    dtypes = ["uint8", "uint16", "uint32", "uint64",
              "int8", "int16", "int32", "int64",
              "float16", "float32", "float64"]

    for dtype_base in dtypes:
        for dtype_exp in dtypes:

            # Compile a gpu function with the specified dtypes
            base = theano.tensor.vector(dtype=dtype_base)
            exp = theano.tensor.vector(dtype=dtype_exp)
            output = base ** exp
            f = theano.function([base, exp], output)

            # Call the function to make sure the output is valid
            base_val = numpy.random.randint(0, 5, size=10).astype(dtype_base)
            exp_val = numpy.random.randint(0, 3, size=10).astype(dtype_exp)

            out = f(base_val, exp_val)
            expected_out = base_val ** exp_val
            assert_allclose(out, expected_out)


class test_GpuDimShuffle(test_elemwise.test_DimShuffle):
    op = GpuDimShuffle


class test_GpuCAReduceCPY(test_elemwise.test_CAReduce):
    dtypes = ["float32"]
    bin_dtypes = ["uint8", "int8"]
    op = GpuCAReduceCPY
    reds = [scalar.add, scalar.mul]
    pre_scalar_op = None
    mode = mode_with_gpu

    def test_perform(self):
        for dtype in self.dtypes + self.bin_dtypes:
            for op in self.reds:
                self.with_linker(gof.PerformLinker(), op, dtype=dtype,
                                 pre_scalar_op=self.pre_scalar_op)

    def test_perform_nan(self):
        for dtype in self.dtypes:
            if not dtype.startswith('float'):
                continue
            for op in self.reds:
                self.with_linker(gof.PerformLinker(), op, dtype=dtype,
                                 test_nan=True,
                                 pre_scalar_op=self.pre_scalar_op)

    def test_c(self):
        for dtype in self.dtypes + self.bin_dtypes:
            for op in self.reds:
                self.with_linker(gof.CLinker(), op, dtype=dtype,
                                 pre_scalar_op=self.pre_scalar_op)

    def test_c_nan(self):
        for dtype in self.dtypes:
            if not dtype.startswith('float'):
                continue
            for op in self.reds:
                self.with_linker(gof.CLinker(), op, dtype=dtype,
                                 test_nan=True,
                                 pre_scalar_op=self.pre_scalar_op)

    def test_infer_shape(self):
        for dtype in self.dtypes:
            super(test_GpuCAReduceCPY, self).test_infer_shape(dtype)


class test_GpuCAReduceCuda(test_GpuCAReduceCPY):
    dtypes = ["float32", "int64"]
    bin_dtypes = ["uint8", "int8"]

    cases = [((5, 6), None),
             ((5, 6), (0, 1)),
             ((5, 6), (0, )),
             ((5, 6), (1, )),
             ((5, 6), (-1, )),
             ((5, 6), (-2, )),
             # ((5, 6), ()),  #reduce on no axis(copy) isn't implemented
             # ((2, 3, 4, 5), (0, 1, 3)), mask 1101 isn't implemented
             # ((2, 3, 4, 5), (-2, -3)), mask 0110 isn't implemented
             ((5, 0), None),
             ((5, 0), (0, )),
             ((5, 0), (1, )),
             # ((5, 0), ()), reduce on no axis isn't implemented
             # ((), None), reduce on no axis isn't implemented
             # ((), ()) reduce on no axis isn't implemented

             # Test all GPU cases implemented
             ((1, 0), (1,)),
             ((0, 1), (1,)),
             ((0, 0), (1,)),
             ((0, 0, 0), (1, 2)),
             ((0, 0, 0, 0), (1, 2, 3)),
             ((2, 1), (1,)),
             ((1, 2), (1,)),
             ((100, 3, 1300), [1]),
             ((0,), [0]), ((5,), [0]),
             ((0, 0), [0, 1]), ((1, 0), [0, 1]), ((5, 4), [0, 1]), ((33, 31), [0, 1]), ((5, 4), [1]), ((5, 4), [0]),  # need something bigger then 32 for some opt test.
             ((5, 4, 3), [0]), ((5, 4, 3), [1]), ((5, 4, 3), [0, 1]), ((5, 4, 3), [2]), ((5, 4, 3), [1, 2]), ((5, 4, 3), [0, 1, 2]),
             ((0, 0, 0, 0), [0, 1, 2, 3]),
             ((5, 4, 3, 20), [2, 3]), ((5, 4, 3, 2), [0, 1, 2, 3]), ((5, 4, 3, 2), [0, 2, 3]), ((5, 4, 3, 2), [1, 2, 3]),

             # test shape bigger then 4096 on each dimension to make sure that we work correctly when we don't have enough thread/block in each dimensions
             ((4100, 3), [0]), ((3, 4101), [0]),  # 10
             ((1024, 33), [0]), ((33, 1024), [0]),  # 10
             ((1025, 33), [0]), ((33, 1025), [0]),  # 10

             ((4100, 3), [1]), ((3, 4101), [1]),  # 01
             ((1024, 33), [1]), ((33, 1024), [1]),  # 01
             ((1025, 33), [1]), ((33, 1025), [1]),  # 01

             ((4100, 3), [0, 1]), ((3, 4101), [0, 1]),  # 11
             ((1024, 33), [0, 1]), ((33, 1024), [0, 1]),  # 01
             ((1025, 33), [0, 1]), ((33, 1025), [0, 1]),  # 01

             ((4100, 4, 3), [0]), ((5, 4100, 3), [0]), ((5, 4, 4100), [0]), ((3, 65536, 1), [0]),  # 100
             ((4100, 4, 3), [1]), ((5, 4100, 3), [1]), ((5, 4, 4100), [1]),  # 010
             ((4100, 4, 3), [2]), ((5, 4100, 3), [2]), ((5, 4, 4100), [2]),  # 001
             ((4100, 4, 3), [0, 1]), ((5, 4100, 3), [0, 1]), ((5, 4, 4100), [0, 1]),  # 110
             ((4100, 4, 3), [1, 2]), ((5, 4100, 3), [1, 2]), ((5, 4, 4100), [1, 2]),  # 011
             ((4100, 4, 3), [0, 2]), ((5, 4100, 3), [0, 2]), ((5, 4, 4100), [0, 2]),  # 101
             ((4100, 4, 3), [0, 1, 2]), ((5, 4100, 3), [0, 1, 2]), ((5, 4, 4100), [0, 1, 2]),  # 111
             ((65, 4, 3), [0, 1, 2]), ((5, 65, 3), [0, 1, 2]), ((5, 4, 65), [0, 1, 2]),  # 111

             # reduce over 2d
             ((4100, 4, 3, 2), [2, 3]), ((4, 4100, 3, 2), [2, 3]), ((4, 3, 4100, 2), [2, 3]), ((4, 3, 2, 4100), [2, 3]),  # 0011
             ((4100, 4, 3, 2), [1, 3]), ((4, 4100, 3, 2), [1, 3]), ((4, 3, 4100, 2), [1, 3]), ((4, 3, 2, 4100), [1, 3]),  # 0101
             # ((4100, 4, 3, 2), [1, 2]), ((4, 4100, 3, 2), [1, 2]), ((4, 3, 4100, 2), [1, 2]), ((4, 3, 2, 4100), [1, 2]),  # 0110 by reshape
             # ((4100,4,3,2),[0,3]),((4,4100,3,2),[0,3]),((4,3,4100,2),[0,3]),((4,3,2,4100),[0,3]),  # 1001 by reshape
             # ((4100,4,3,2),[0,2]),((4,4100,3,2),[0,2]),((4,3,4100,2),[0,2]),((4,3,2,4100),[0,2]),  # 1010 not implemented
             # ((4100, 4, 3, 2), [0, 1]), ((4, 4100, 3, 2), [0, 1]), ((4, 3, 4100, 2), [0, 1]), ((4, 3, 2, 4100), [0, 1]),  # 1100 by reshape

             # reduce over 3d
             # 3d not tested: 1101, 1110, 1111
             # ((4100,4,3,2),[0,1,3]),((4,4100,3,2),[0,1,3]),((4,3,4100,2),[0,1,3]),((4,3,2,4100),[0,1,3]),  # 1101 by reshape
             # ((4100, 4, 3, 2), [0, 1, 2]), ((4, 4100, 3, 2), [0, 1, 2]), ((4, 3, 4100, 2), [0, 1, 2]), ((4, 3, 2, 4100), [0, 1, 2]),  # 1110 by reshape
             ((4100, 4, 3, 2), [0, 2, 3]), ((4, 4100, 3, 2), [0, 2, 3]), ((4, 3, 4100, 2), [0, 2, 3]),  # ((4,3,2,4100),[0,2,3]),  # 1011
             ((4100, 4, 3, 2), [1, 2, 3]), ((4, 4100, 3, 2), [1, 2, 3]), ((4, 3, 4100, 2), [1, 2, 3]), ((4, 3, 2, 4100), [1, 2, 3]),  # 0111
             ((65, 4, 3, 2), [1, 2, 3]), ((4, 65, 3, 2), [1, 2, 3]), ((4, 3, 65, 2), [1, 2, 3]), ((4, 3, 2, 65), [1, 2, 3]),  # 0111

             # reduce over 4d
             ((4100, 2, 3, 4), [0, 1, 2, 3]), ((2, 4100, 3, 4), [0, 1, 2, 3]), ((2, 3, 4100, 4), [0, 1, 2, 3]), ((2, 3, 4, 4100), [0, 1, 2, 3]), ((128, 1, 3, 3), [0, 1, 2, 3]),  # 1111

             # test pattern implemented by reshape
             # Skip them as this test the op directly, not the optimization with reshape
             # ((4100,4,3,2),[0]),((4,4100,3,2),[0]),((4,3,4100,2),[0]),((4,3,2,4100),[0]),#1000
             # ((4100,4,3,2),[1]),((4,4100,3,2),[1]),((4,3,4100,2),[1]),((4,3,2,4100),[1]),#0100
             # ((4100,4,3,2),[2]),((4,4100,3,2),[2]),((4,3,4100,2),[2]),((4,3,2,4100),[2]),#0010
             # ((4100,4,3,2),[3]),((4,4100,3,2),[3]),((4,3,4100,2),[3]),((4,3,2,4100),[3]),#0001
             # ((1100,2,3,4,5),[0,1,2,3,4]),((2,1100,3,4,5),[0,1,2,3,4]),((2,3,1100,4,5),[0,1,2,3,4]),((2,3,4,1100,5),[0,1,2,3,4]),((2,3,4,5,1100),[0,1,2,3,4]),#11111
             # ((5,4,3,10,11),[1,2]),
             ]
    op = GpuCAReduceCuda
    reds = [scalar.add, scalar.mul,
            scalar.maximum, scalar.minimum]
    pre_scalar_op = None

    def test_perform(self):
        return

    def test_perform_nan(self):
        return

    def setUp(self):
        super(test_GpuCAReduceCuda, self).setUp()
        if get_context(test_ctx_name).kind != 'cuda':
            raise SkipTest("Cuda specific tests")


class T_gpureduce_dtype(test_elemwise.T_reduce_dtype):
    mode = mode_with_gpu.excluding('local_cut_useless_reduce')
    op = GpuCAReduceCuda
    # Currently we don't support reduction on 0 axis
    axes = [None, 0, 1, 1, [0], [1], [0, 1]]
    # We don't support complex dtype
    dtypes = ['int8', 'int16', 'int32', 'int64',
              'uint8', 'uint16', 'uint32', 'uint64',
              'float32', 'float64']

    def setUp(self):
        if get_context(test_ctx_name).kind != 'cuda':
            raise SkipTest("Cuda specific tests")


def speed_reduce10():
    import numpy
    import theano
    data = numpy.random.rand(1000, 1000).astype("float32")
    m = theano.tensor.fmatrix()
    f = theano.function([m], [m.sum(axis=0), m.T.sum(axis=0)],
                        mode=mode_with_gpu)
    f(data)

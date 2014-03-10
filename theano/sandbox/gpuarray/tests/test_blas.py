from unittest import TestCase

import theano
from theano.tensor.blas import gemv_inplace, gemm_inplace, _dot22

from theano.sandbox.gpuarray.tests.test_basic_ops import makeTester, rand

from theano.sandbox.gpuarray.blas import (gpugemv_inplace,
                                          gpugemm_inplace, gpu_dot22)


GpuGemvTester = makeTester('GpuGemvTester',
                           op=gemv_inplace, gpu_op=gpugemv_inplace,
                           cases=dict(
        dot_vv=[rand(1), 1, rand(1, 2), rand(2), 0],
        dot_vm=[rand(3), 1, rand(3, 2), rand(2), 0],
#        test_02=[rand(0), 1, rand(0, 2), rand(2), 0],
#        test_30=[rand(3), 1, rand(3, 0), rand(0), 0],
#        test_00=[rand(0), 1, rand(0, 0), rand(0), 0],
        test_stride=[rand(3)[::-1], 1, rand(3, 2)[::-1], rand(2)[::-1], 0],
        )
)

GpuGemmTester = makeTester('GpuGemmTester',
                           op=gemm_inplace, gpu_op=gpugemm_inplace,
                           cases=dict(
        test1=[rand(3, 4), 1.0, rand(3, 5), rand(5, 4), 0.0],
        test2=[rand(3, 4), 1.0, rand(3, 5), rand(5, 4), 1.0],
        test3=[rand(3, 4), 1.0, rand(3, 5), rand(5, 4), -1.0],
        test4=[rand(3, 4), 0.0, rand(3, 5), rand(5, 4), 0.0],
        test5=[rand(3, 4), 0.0, rand(3, 5), rand(5, 4), 0.6],
        test6=[rand(3, 4), 0.0, rand(3, 5), rand(5, 4), -1.0],
        test7=[rand(3, 4), -1.0, rand(3, 5), rand(5, 4), 0.0],
        test8=[rand(3, 4), -1.0, rand(3, 5), rand(5, 4), 1.1],
        test9=[rand(3, 4), -1.0, rand(3, 5), rand(5, 4), -1.1],
 #       test10=[rand(0, 4), -1.0, rand(0, 5), rand(5, 4), 0.0],
 #       test11=[rand(3, 0), -1.0, rand(3, 5), rand(5, 0), 1.1],
 #       test12=[rand(3, 4), -1.0, rand(3, 0), rand(0, 4), -1.1],
 #       test13=[rand(0, 0), -1.0, rand(0, 0), rand(0, 0), -1.1],
    )
)


GpuDot22Tester = makeTester(
    'GpuGemmTester',
    op=_dot22, gpu_op=gpu_dot22,
    cases=dict(
        test1=[rand(3, 4), rand(4, 5)],
        test2=[rand(1, 4), rand(4, 5)],
        test3=[rand(3, 1), rand(1, 5)],
        test4=[rand(3, 4), rand(4, 1)],
#        test5=[rand(0, 4), rand(4, 5)],
#        test6=[rand(3, 0), rand(0, 5)],
#        test7=[rand(3, 4), rand(4, 0)],
#        test8=[rand(0, 4), rand(4, 0)],
#        test9=[rand(0, 0), rand(0, 0)],
    )
)

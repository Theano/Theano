#!/usr/bin/env python

# print info to check we link with witch version of blas
# test the speed of the blas gemm fct:
# C=a*C+dot(A,B)*b
# A,B,C matrix
# a,b scalar
from __future__ import absolute_import, print_function, division

import os
import sys
import time
from optparse import OptionParser
import subprocess

import numpy as np
import theano
import theano.tensor as T


def execute(execute=True, verbose=True, M=2000, N=2000, K=2000,
            iters=10, order='C'):
    """
    :param execute: If True, execute a Theano function that should call gemm.
    :param verbose: If True, will print some Theano flags and env variables.
    :param M,N,K: The M,N,K size used by gemm.
    :param iters: The number of calls to gemm to do.

    :return: a tuple (execution time,
                      str that represents the implementation used)
    """

    if verbose:
        print('Some Theano flags:')
        print('    blas.ldflags=', theano.config.blas.ldflags)
        print('    compiledir=', theano.config.compiledir)
        print('    floatX=', theano.config.floatX)
        print('    device=', theano.config.device)
        print('Some OS information:')
        print('    sys.platform=', sys.platform)
        print('    sys.version=', sys.version)
        print('    sys.prefix=', sys.prefix)
        print('Some environment variables:')
        print('    MKL_NUM_THREADS=', os.getenv('MKL_NUM_THREADS'))
        print('    OMP_NUM_THREADS=', os.getenv('OMP_NUM_THREADS'))
        print('    GOTO_NUM_THREADS=', os.getenv('GOTO_NUM_THREADS'))
        print()
        print('Numpy config: (used when the Theano flag'
              ' "blas.ldflags" is empty)')
        np.show_config()
        print('Numpy dot module:', np.dot.__module__)
        print('Numpy location:', np.__file__)
        print('Numpy version:', np.__version__)
        if (theano.config.device.startswith("gpu") or
                theano.config.init_gpu_device.startswith("gpu")):
            print('nvcc version:')
            subprocess.call((theano.sandbox.cuda.nvcc_compiler.nvcc_path,
                             "--version"))
            print()

    a = theano.shared(np.ones((M, N), dtype=theano.config.floatX,
                              order=order))
    b = theano.shared(np.ones((N, K), dtype=theano.config.floatX,
                              order=order))
    c = theano.shared(np.ones((M, K), dtype=theano.config.floatX,
                              order=order))
    f = theano.function([], updates=[(c, 0.4 * c + .8 * T.dot(a, b))])

    if any([x.op.__class__.__name__ == 'Gemm' for x in
            f.maker.fgraph.toposort()]):
        c_impl = [hasattr(thunk, 'cthunk')
                  for node, thunk in zip(f.fn.nodes, f.fn.thunks)
                  if node.op.__class__.__name__ == "Gemm"]
        assert len(c_impl) == 1
        if c_impl[0]:
            impl = 'CPU (with direct Theano binding to blas)'
        else:
            impl = 'CPU (without direct Theano binding to blas but with numpy/scipy binding to blas)'
    elif any([x.op.__class__.__name__ == 'GpuGemm' for x in
              f.maker.fgraph.toposort()]):
        impl = 'GPU'
    else:
        impl = 'ERROR, unable to tell if Theano used the cpu or the gpu:\n'
        impl += str(f.maker.fgraph.toposort())

    t0 = 0
    t1 = -1

    f()  # Ignore first function call to get representative time.
    if execute:
        sync = (hasattr(theano, "sandbox") and
                hasattr(theano.sandbox, "cuda") and
                isinstance(c, theano.sandbox.cuda.CudaNdarraySharedVariable))
        sync2 = (hasattr(theano, "gpuarray") and
                 isinstance(c, theano.gpuarray.GpuArraySharedVariable))
        t0 = time.time()
        for i in range(iters):
            f()
        if sync:
            theano.sandbox.cuda.synchronize()
        if sync2:
            c.get_value(borrow=True, return_internal_type=True).sync()
        t1 = time.time()
    return t1 - t0, impl


def jobman_job(state, channel):
    execute()
    return channel.COMPLETE


def test():
    return execute()


parser = OptionParser(
    usage='%prog <options>\nCompute time needed to perform BLAS gemm '
    'computations between matrices of size (M, N) and (N, K).')

parser.add_option('-q', '--quiet', action='store_true', dest='quiet',
                  default=False,
                  help="If true, do not print the comparison table and config "
                       "options")
parser.add_option('--print_only', action='store_true', dest='print_only',
                  default=False,
                  help="If true, do not perform gemm computations")
parser.add_option('-M', '--M', action='store', dest='M',
                  default=0, type="int",
                  help="The M size to gemm")
parser.add_option('-N', '--N', action='store', dest='N',
                  default=0, type="int",
                  help="The N size to gemm")
parser.add_option('-K', '--K', action='store', dest='K',
                  default=0, type="int",
                  help="The K size to gemm")
parser.add_option('--iter', action='store', dest='iter',
                  default=10, type="int",
                  help="The number of calls to gemm")
parser.add_option('--order', action='store', dest='order',
                  default="C",
                  help="The numpy memory layout parameter used when creating"
                  " the numpy.ndarray objects. It accepts 'C' for C memory"
                  " order and 'F' for Fortran order (for all matrices).")
parser.add_option('-B', '--B', action='store', dest='B',
                  default=5000, type="int",
                  help="The M, N, and K for big gemm")


if __name__ == "__main__":
    options, arguments = parser.parse_args(sys.argv)

    if hasattr(options, "help"):
        print(options.help)
        sys.exit(0)

    if not options.quiet:
        print("""
        Some results that you can compare against. They were 10 executions
        of gemm in float64 with matrices of shape 2000x2000 (M=N=K=2000).
        All memory layout was in C order.

        CPU tested: Xeon E5345(2.33Ghz, 8M L2 cache, 1333Mhz FSB),
                    Xeon E5430(2.66Ghz, 12M L2 cache, 1333Mhz FSB),
                    Xeon E5450(3Ghz, 12M L2 cache, 1333Mhz FSB),
                    Xeon X5560(2.8Ghz, 12M L2 cache, hyper-threads?)
                    Core 2 E8500, Core i7 930(2.8Ghz, hyper-threads enabled),
                    Core i7 950(3.07GHz, hyper-threads enabled)
                    Xeon X5550(2.67GHz, 8M l2 cache?, hyper-threads enabled)


        Libraries tested:
            * numpy with ATLAS from distribution (FC9) package (1 thread)
            * manually compiled numpy and ATLAS with 2 threads
            * goto 1.26 with 1, 2, 4 and 8 threads
            * goto2 1.13 compiled with multiple threads enabled

                          Xeon   Xeon   Xeon  Core2 i7    i7     Xeon   Xeon
        lib/nb threads    E5345  E5430  E5450 E8500 930   950    X5560  X5550

        numpy 1.3.0 blas                                                775.92s
        numpy_FC9_atlas/1 39.2s  35.0s  30.7s 29.6s 21.5s 19.60s
        goto/1            18.7s  16.1s  14.2s 13.7s 16.1s 14.67s
        numpy_MAN_atlas/2 12.0s  11.6s  10.2s  9.2s  9.0s
        goto/2             9.5s   8.1s   7.1s  7.3s  8.1s  7.4s
        goto/4             4.9s   4.4s   3.7s  -     4.1s  3.8s
        goto/8             2.7s   2.4s   2.0s  -     4.1s  3.8s
        openblas/1                                        14.04s
        openblas/2                                         7.16s
        openblas/4                                         3.71s
        openblas/8                                         3.70s
        mkl 11.0.083/1            7.97s
        mkl 10.2.2.025/1                                         13.7s
        mkl 10.2.2.025/2                                          7.6s
        mkl 10.2.2.025/4                                          4.0s
        mkl 10.2.2.025/8                                          2.0s
        goto2 1.13/1                                                     14.37s
        goto2 1.13/2                                                      7.26s
        goto2 1.13/4                                                      3.70s
        goto2 1.13/8                                                      1.94s
        goto2 1.13/16                                                     3.16s

        Test time in float32

        cuda version      6.5    6.0    5.5    5.0    4.2    4.1    4.0    3.2    3.0   # note
        gpu
        K6000/NOECC       0.06s         0.06s
        K40                             0.07s
        K20m/ECC          0.08s 0.08s          0.07s
        K20/NOECC                              0.07s
        M2090                           0.19s
        C2075                                         0.25s
        M2075                                  0.25s
        M2070                                  0.25s         0.27s         0.32s
        M2070-Q                                0.48s         0.27s         0.32s
        M2050(Amazon)                          0.25s
        C1060                                                              0.46s
        K600                            1.04s

        GTX Titan Black                 0.05s
        GTX Titan(D15U-50)              0.06s  0.06s  don't work
        GTX 780                         0.06s
        GTX 980           0.06s
        GTX 970           0.08s
        GTX 680                         0.11s  0.12s  0.154s               0.218s
        GRID K520         0.14s
        GTX 580                         0.16s  0.16s  0.164s               0.203s
        GTX 480                         0.19s  0.19s  0.192s               0.237s 0.27s
        GTX 750 Ti        0.20s
        GTX 470                         0.23s  0.23s  0.238s               0.297s 0.34s
        GTX 660                         0.18s  0.20s  0.23s
        GTX 560                                       0.30s
        GTX 650 Ti                             0.27s
        GTX 765M                 0.27s
        GTX 460                                0.37s                0.45s
        GTX 285                         0.42s         0.452s        0.452s        0.40s # cuda 3.0 seems faster? driver version?
        750M                                   0.49s
        GT 610            2.38s
        GTX 550 Ti                                                  0.57s
        GT 520                                        2.68s                3.06s
        GT 520M                                2.44s                       3.19s        # with bumblebee on Ubuntu 12.04
        GT 220                                                             3.80s
        GT 210                                                      6.35s
        8500 GT                                                                   10.68s

        Results for larger matrices.
        There were 10 executions of gemm in float32
        with matrices of shape 5000x5000 (M=N=K=5000).
        All memory layout was in C order.

        cuda version      7.5    7.0    6.5
        gpu
        M40               0.47s
        k80               0.96s
        K6000/NOECC              0.69s
        K40                             0.88s
        K20m/ECC
        K20/NOECC
        M2090
        C2075
        M2075
        M2070
        M2070-Q
        M2050(Amazon)
        C1060
        K600

        GTX Titan X       0.45s  0.47s
        GTX Titan Black   0.64s  0.64s
        GTX Titan(D15U-50)
        GTX 780
        GTX 980 Ti        0.41s
        GTX 980
        GTX 970           0.66s
        GTX 680                  1.57s
        GRID K520
        GTX 750 Ti        2.01s  2.01s
        GTX 750           2.46s  2.37s
        GTX 660           2.32s  2.32s
        GTX 580           2.42s         2.47s
        GTX 480           2.87s         2.88s
        TX1                      7.6s (float32 storage and computation)
        GT 610                   33.5s
        """)

    if options.M == 0:
        M = options.B
    else:
        M = options.M
    if options.N == 0:
        N = options.B
    else:
        N = options.N
    if options.K == 0:
        K = options.B
    else:
        K = options.K

    t, impl = execute(not options.print_only, not options.quiet,
                      M=M, N=N, K=K, iters=options.iter,
                      order=options.order)

    if options.print_only:
        pass
    elif options.quiet:
        print(t)
    else:
        print()
        print("We executed", options.iter, end=' ')
        print("calls to gemm with a and b matrices of shapes", end=' ')
        print("(%d, %d) and (%d, %d)." % (M, N, N, K))

        print()
        print('Total execution time: %.2fs on %s.' % (t, impl))
        print()
        print('Try to run this script a few times. Experience shows that'
              ' the first time is not as fast as followings calls. The'
              ' difference is not big, but consistent.')

#!/usr/bin/env python

#print info to check we link with witch version of blas
#test the speed of the blas gemm fct:
#C=a*C+dot(A,B)*b
#A,B,C matrix
#a,b scalar

s = """
result for shapes=(2000,2000) and iters=100
GTX 470 7.22s
GTX 285, 6.84s
GTX 480 5.83s
"""

import os
import sys
import time
from optparse import OptionParser
import subprocess

import numpy
import theano
import theano.tensor as T

from theano.gof.python25 import any


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
        print 'Some Theano flags:'
        print '    blas.ldflags=', theano.config.blas.ldflags
        print '    compiledir=', theano.config.compiledir
        print '    floatX=', theano.config.floatX
        print 'Some environment variables:'
        print '    MKL_NUM_THREADS=', os.getenv('MKL_NUM_THREADS')
        print '    OMP_NUM_THREADS=', os.getenv('OMP_NUM_THREADS')
        print '    GOTO_NUM_THREADS=', os.getenv('GOTO_NUM_THREADS')
        print
        print ('Numpy config: (used when the Theano flag'
               ' "blas.ldflags" is empty)')
        numpy.show_config()
        print 'Numpy dot module:', numpy.dot.__module__
        print 'Numpy location:', numpy.__file__
        print 'Numpy version:', numpy.__version__
        if (theano.config.device.startswith("gpu") or
            theano.config.init_gpu_device.startswith("gpu")):
            print 'nvcc version:'
            subprocess.call((theano.sandbox.cuda.nvcc_compiler.nvcc_path,
                             "--version"))
            print

    a = theano.shared(numpy.ones((M, N), dtype=theano.config.floatX,
                                 order=order))
    b = theano.shared(numpy.ones((N, K), dtype=theano.config.floatX,
                                 order=order))
    c = theano.shared(numpy.ones((M, K), dtype=theano.config.floatX,
                                 order=order))
    f = theano.function([], updates=[(c, 0.4 * c + .8 * T.dot(a, b))],
                        mode=theano.compile.ProfileMode())

    if any([x.op.__class__.__name__ == 'Gemm' for x in
            f.maker.fgraph.toposort()]):
        c_impl = f.profile.apply_cimpl.values()
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

    if execute:
        sync = (hasattr(theano, "sandbox") and
                hasattr(theano.sandbox, "cuda") and
                theano.sandbox.cuda.cuda_available)
        t0 = time.time()
        for i in range(iters):
            f()
        if sync:
            theano.sandbox.cuda.synchronize()
        t1 = time.time()
    return t1 - t0, impl


def jobman_job(state, channel):
    execute()
    return channel.COMPLETE


def test():
    execute()


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
                  default=2000, type="int",
                  help="The M size to gemm")
parser.add_option('-N', '--N', action='store', dest='N',
                  default=2000, type="int",
                  help="The N size to gemm")
parser.add_option('-K', '--K', action='store', dest='K',
                  default=2000, type="int",
                  help="The K size to gemm")
parser.add_option('--iter', action='store', dest='iter',
                  default=10, type="int",
                  help="The number of calls to gemm")
parser.add_option('--order', action='store', dest='order',
                  default="C",
                  help="The numpy memory layout parameter used when creating"
                  " the numpy.ndarray objects. It accepts 'C' for C memory"
                  " order and 'F' for Fortran order (for all matrices).")


if __name__ == "__main__":
    options, arguments = parser.parse_args(sys.argv)

    if hasattr(options, "help"):
        print options.help
        sys.exit(0)

    if not options.quiet:
        print """
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
        (cuda version 3.2RC and up have a faster gemm on the Fermi/GTX[45]??)

        gpu/cuda version
        M2050(Amazon)/5.0 0.25s

        GTX680/4.2        0.154s
        GTX580/4.2        0.164s
        GTX480/4.2        0.192s
        GTX470/4.2        0.238s
        C2075/4.2         0.25s
        GTX285/4.2        0.452s #cuda 3.0 seam faster? driver version?
        GT520/4.2         2.68s
        GTX560/4.2        0.30s

        GTX460/4.0        0.45s

        GTX580/3.2        0.203s
        GTX680/3.2        0.218s
        GTX480/3.2        0.237s
        GTX470/3.2        0.297s
        GTX285/3.2        0.452s #cuda 3.0 seam faster? driver version?

        GTX480/3.0        0.27s
        M2070/4.1         0.27s
        GTX470/3.2        0.29s
        M2070/3.2         0.32s
        GTX470/3.0        0.34s
        GTX285/3.0        0.40s
        C1060/3.2         0.46s
        GTX550Ti/4.0      0.57s
        520/3.2           3.06s
        520M/3.2          3.19s with bumblebee on Ubuntu 12.04
        GT220/3.2RC       3.80s
        GT210/4.0         6.35s
        8500GT/3.0       10.68s
        """

    t, impl = execute(not options.print_only, not options.quiet,
                      M=options.M, N=options.N, K=options.K,
                      iters=options.iter, order=options.order)

    if options.print_only:
        pass
    elif options.quiet:
        print t
    else:
        print
        print "We executed", options.iter,
        print "calls to gemm with a and b matrices of shapes",
        print "(%d, %d) and (%d, %d)." % (options.M, options.N,
                                          options.N, options.K)

        print
        print 'Total execution time: %.2fs on %s.' % (t, impl)
        print
        print ('Try to run this script a few times. Experience shows that'
               ' the first time is not as fast as followings calls. The'
               ' difference is not big, but consistent.')

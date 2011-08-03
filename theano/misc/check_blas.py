#print info to check we link with witch version of blas
#test the speed of the blas gemm fct:
#C=a*C+dot(A,B)*b
#A,B,C matrix
#a,b scalar

s="""
result for shapes=(2000,2000) and iters=100
GTX 470 7.22s
GTX 285, 6.84s
GTX 480 5.83s
"""

import os, sys, time

import numpy
import theano
import theano.tensor as T

from theano.gof.python25 import any


shapes=(2000,2000)
iters = 10


def execute(execute=True, verbose=True):

    a=theano.shared(numpy.ones(shapes, dtype=theano.config.floatX))
    b=theano.shared(numpy.ones(shapes, dtype=theano.config.floatX))
    c=theano.shared(numpy.ones(shapes, dtype=theano.config.floatX))

    f=theano.function([],updates={c:0.4*c+.8*T.dot(a,b)})

    if verbose:
        print 'Some theano flags:'
        print '    blas.ldflags=',theano.config.blas.ldflags
        print '    compiledir=',theano.config.compiledir
        print '    floatX=',theano.config.floatX
        print 'Some env flags:'
        print '    MKL_NUM_THREADS=',os.getenv('MKL_NUM_THREADS')
        print '    OMP_NUM_THREADS=',os.getenv('OMP_NUM_THREADS')
        print '    GOTO_NUM_THREADS=',os.getenv('GOTO_NUM_THREADS')
        print
        print 'Numpy config:(used when the theano flags "blas.ldflags" is empty)'
        numpy.show_config();
        print 'Numpy dot module:',numpy.dot.__module__;
        print 'Numpy file location that was loaded:',numpy.__file__;
        print 'Numpy version:',numpy.__version__
        print
        if any( [x.op.__class__.__name__=='Gemm' for x in f.maker.env.toposort()]):
            print 'Used the cpu'
        elif any( [x.op.__class__.__name__=='GpuGemm' for x in f.maker.env.toposort()]):
            print 'Used the gpu'
        else:
            print 'ERROR, not able to tell if theano used the cpu or the gpu'
            print f.maker.env.toposort()
    t0=0
    t1=-1

    if execute:
        t0=time.time()
        for i in range(iters):
            f()
        t1=time.time()
    if verbose and execute:
        print
        print 'This execution time took %.2fs'%(t1-t0)
        print
        print 'Try to run this script a few times. Experience show that the first time is not as fast as followings call. The difference is not big, but consistent.'
    return t1-t0


def jobman_job(state, channel):
    execute()
    return channel.COMPLETE

def test():
    execute()


if __name__ == "__main__":
    verbose = True
    print_only = False

    if '--quiet' in sys.argv:
        verbose = False
    if '--print_only' in sys.argv:
        print_only = True

    t = execute(not print_only, verbose)

    if verbose:
        print """
        Some result that you can compare again. They where 10 executions of gemm in float64 with matrix of shape 2000x2000 on FC9.

        Cpu tested: Xeon E5345(2.33Ghz, 8M L2 cache, 1333Mhz FSB), Xeon E5430(2.66Ghz, 12M L2 cache, 1333Mhz FSB),
                    Xeon E5450(3Ghz, 12M L2 cache, 1333Mhz FSB), Xeon X5560(2.8Ghz, 12M L2 cache, 6.4GT/s QPI, hyper-threads enabled?)
                    Core 2 E8500, Core i7 930(2.8Ghz, hyper-threads enabled), Core i7 950(3.07GHz, hyper-threads enabled)
                    Xeon X5550(2.67GHz, 8M l2 cache?, hyper-threads enabled)


        Lib tested:
            * numpy with ATLAS from distribution(FC9) package (1 thread)
            * manually compiled numpy and ATLAS with 2 threads
            * goto 1.26 with 1, 2, 4 and 8 threads.
            * goto2 1.13 compiled with multiple thread enabled.

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

        Test time in float32 with cuda 3.0.14
        (cuda version 3.2RC and up are supposed to have faster gemm on the GTX4?? card)
        gpu/cuda version
        GTX580/3.2        0.20s
        GTX480/3.2        0.24s
        GTX480/3.0        0.27s
        GTX470/3.2        0.29s
        M2070/3.2         0.32s
        GTX470/3.0        0.34s
        GTX285/3.0        0.40s
        C1060/3.2         0.46s
        GTX550Ti/4.0      0.57s
        GT220/3.2RC       3.80s
        8500GT/3.0       10.68s
        """

        print
        print "We timed",iters,"executions of gemm with matrix of shapes",shapes
    else:
        print t

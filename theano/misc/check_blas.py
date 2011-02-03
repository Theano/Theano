#print info to check we link with witch version of blas
#test the speed of the blas gemm fct:
#C=a*C+dot(A,B)*b
#A,B,C matrix
#a,b scalar
import os

s="""
result for shapes=(2000,2000) and iters=100
GTX 470 7.22s
GTX 285, 6.84s
GTX 480 5.83s
"""
import sys

import theano,numpy,time
import theano.tensor as T

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
        print 'this execution time took %.2fs'%(t1-t0)
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

        Cpu tested: Xeon E5345, Xeon E5430, Xeon E5450(3Ghz), Xeon X5560(2.8Ghz, hyper-threads enabled?)
                    Core 2 E8500, Core i7 930(2.8Ghz, hyper-threads enabled)


        Lib tested:
            * numpy with ATLAS from distribution(FC9) package (1 thread)
            * manually compiled numpy and ATLAS with 2 threads
            * goto with 1, 2, 4 and 8 threads.
                          Xeon   Xeon   Xeon  Core2 i7    Xeon
        lib/nb threads    E5345  E5430  E5450 E8500 930   X5560

        numpy_FC9_atlas/1 39.2s  35.0s  30.7s 29.6s 21.5s
        goto/1            18.7s  16.1s  14.2s 13.7s 16.1s
        numpy_MAN_atlas/2 12.0s  11.6s  10.2s 9.2s  9.0s
        goto/2            9.5s   8.1s   7.1s  7.3s  8.1s
        goto/4            4.9s   4.4s   3.7s  -     4.1s
        goto/8            2.7s   2.4s   2.0s  -     4.1s
        mkl 10.2.2.025/1                                 13.7s
        mkl 10.2.2.025/2                                 7.6s
        mkl 10.2.2.025/4                                 4.0s
        mkl 10.2.2.025/8                                 2.0s

        Test time in float32 with cuda 3.0.14
        (cuda version 3.2RC and up are supposed to have faster gemm on the GTX4?? card)
        cpu/cuda version
        GTX480/3.2        0.24s
        GTX480/3.0        0.27s
        GTX470/3.2        0.29s
        GTX470/3.0        0.34s
        GTX285/3.0        0.40s
        GT220/3.2RC       5.15s
        8500GT/3.0       10.68s
        """

        print
        print "We timed",iters,"executions of gemm with matrix of shapes",shapes
    else:
        print t

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

import theano,numpy,time
import theano.tensor as T
shapes=(2000,2000)
iters = 10


def execute(verbose=True):

    a=theano.shared(numpy.ones(shapes, dtype=theano.config.floatX))
    b=theano.shared(numpy.ones(shapes, dtype=theano.config.floatX))
    c=theano.shared(numpy.ones(shapes, dtype=theano.config.floatX))

    f=theano.function([],updates={c:0.4*c+.8*T.dot(a,b)})
    if verbose:
        print 'Some theano flags:'
        print '    blas.ldflags=',theano.config.blas.ldflags
        print '    compiledir=',theano.config.compiledir
        print '    floatX=',theano.config.floatX
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

    t0=time.time()
    for i in range(iters):
        f()
    t1=time.time()
    if verbose:
        print
        print 'this execution time took %.2fs'%(t1-t0)
    return t1-t0


def jobman_job(state, channel):
    execute()
    return channel.COMPLETE

def test():
    execute()
    

if __name__ == "__main__":
    execute()
    print """
 	Some result that you can compare again. They where 10 executions of gemm in float64 with matrix of shape 2000x2000 on FC9.
 	
        We tested 3 cpus: Xeon E5345, Xeon E5430 and Xeon E5450

        Lib tested:
            * numpy with ATLAS from distribution(FC9) package (1 thread)
            * manually compiled numpy and ATLAS with 2 threads
            * goto with 1, 2, 4 and 8 threads.

 	lib/nb threads    E5345(s)  E5430(s)  E5450(s)

        numpy_FC9_atlas/1 39.2s     35.0s     30.7s
 	goto/1            18.7s     16.1s     14.2s
        numpy_MAN_atlas/2 12.0s     11.6s     10.2s
 	goto/2            9.5s      8.1s      7.1s
 	goto/4            4.9s      4.4s      3.7s
 	goto/8            2.7s      2.4s      2.0s
 	"""

    print 
    print "We timed",iters,"executions of gemm with matrix of shapes",shapes

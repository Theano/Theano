#print info to check we link with witch version of blas
#test the speed of the blas gemm fct:
#C=a*C+dot(A,B)*b
#A,B,C matrix
#a,b scalar

import theano,numpy,time
import theano.tensor as T

shapes=(2000,2000)
iters = 10


a=T.matrix()
b=T.matrix()

c=theano.shared(numpy.ones(shapes))
f=theano.function([a,b],updates={c:0.4*c+.8*T.dot(a,b)})
print 'blas.ldflags=',theano.config.blas.ldflags
print 'compiledir=',theano.config.compiledir
print f.maker.env.toposort()

av=numpy.ones(shapes)
bv=numpy.ones(shapes)
t0=time.time()
for i in range(iters):
    f(av,bv)

print 'times=%.2fs'%(time.time()-t0)

print """
Some result that you can compare again:

Lib: goto
computer: Intel(R) Xeon(R) CPU E5430  @ 2.66GHz

OMP_NUM_THREADS     time
1                   16.05s
2                   8.22s
4                   4.13s
8                   2.32s
"""

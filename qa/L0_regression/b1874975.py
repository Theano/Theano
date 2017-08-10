# bug 1874975
import theano
import theano.tensor as T
import numpy as np
import sys

d = T.vector('d', dtype=theano.config.floatX)
d_np = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=theano.config.floatX)
f = theano.function([d], d.sum())
# 9.0 with gpu, but used to be 5.0 with cuda
res = f(d_np)
print ("Result should be 9.0:")
print(res)
sys.exit(0 if 9==res else 1) 

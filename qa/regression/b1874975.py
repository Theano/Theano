import theano
import theano.tensor as T
import numpy as np

d = T.vector('d', dtype=floatX)
d_np = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=floatX)
f = theano.function([d], d.sum())
# 9.0 with gpu, but 5.0 with cuda
print ("9.0 with gpu, but 5.0 with cuda")
print(f(d_np))


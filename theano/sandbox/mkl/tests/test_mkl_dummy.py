import theano
from theano.sandbox.mkl.mkl_dummy import dummyOP

x = theano.tensor.matrix('x', dtype='float32')
f = dummyOP()(x)

print("=====================================================")
z = theano.function([x], f)

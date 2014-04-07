import numpy

import theano

x, y, z = theano.tensor.vectors('xyz')
f = theano.function([x, y, z], [(x + y + z) * 2])
xv = numpy.random.rand(10).astype(theano.config.floatX)
yv = numpy.random.rand(10).astype(theano.config.floatX)
zv = numpy.random.rand(10).astype(theano.config.floatX)
f(xv, yv, zv)

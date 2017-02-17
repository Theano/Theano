from __future__ import absolute_import, print_function, division
import time

import numpy

import theano
from theano import tensor as tt
from six.moves import xrange
from theano.ifelse import ifelse

a, b = tt.scalars('a', 'b')
x, y = tt.matrices('x', 'y')

z_switch = tt.switch(tt.lt(a, b), tt.mean(x), tt.mean(y))
z_lazy = ifelse(tt.lt(a, b), tt.mean(x), tt.mean(y))

f_switch = theano.function([a, b, x, y], z_switch)
f_lazyifelse = theano.function([a, b, x, y], z_lazy)

val1 = 0.
val2 = 1.
big_mat1 = numpy.ones((10000, 1000))
big_mat2 = numpy.ones((10000, 1000))

n_times = 10

tic = time.clock()
for i in xrange(n_times):
    f_switch(val1, val2, big_mat1, big_mat2)
print('time spent evaluating both values %f sec' % (time.clock() - tic))

tic = time.clock()
for i in xrange(n_times):
    f_lazyifelse(val1, val2, big_mat1, big_mat2)
print('time spent evaluating one value %f sec' % (time.clock() - tic))

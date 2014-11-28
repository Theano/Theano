import numpy
import theano
import theano.tensor as T
from theano.printing import debugprint

X = T.matrix('X')
W = T.matrix('W')
Y = T.dot(X, W)
Z = X - W

f = theano.function(inputs=[X, W], outputs=[Y, Z], profile=False)
for i in xrange(10):
    f(numpy.random.uniform(size=(2, 2)),
      numpy.random.uniform(size=(2, 2)))

#print f.profile.compute_total_times()
#print f.profile.apply_time
debugprint(f)

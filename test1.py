import numpy
import theano
import theano.tensor as T
from theano.printing import debugprint

X = T.matrix('X')
W = T.matrix('W')
Y = T.dot(X, W)
R = X - W
Z = Y + R

f = theano.function(inputs=[X, W], outputs=[Z, R], profile=True)
for i in xrange(10):
    f(numpy.random.uniform(size=(2, 2)),
      numpy.random.uniform(size=(2, 2)))

#print f.profile.compute_total_times()
#print f.profile.apply_time
debugprint(f)

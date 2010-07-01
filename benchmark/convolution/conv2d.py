import sys, timeit, time
import numpy
import theano, theano.tensor.signal.conv

try:
    img_shape =  int(sys.argv[1]), int(sys.argv[2])
    ker_shape =  int(sys.argv[3]), int(sys.argv[4])
    dtype = sys.argv[5]
except:
    print >> sys.stderr, "Usage: %s <img rows> <img cols> <ker rows> <ker cols> <dtype>" % sys.argv[0]
    sys.exit(-1)

img = theano.shared(numpy.ones(img_shape, dtype=dtype))
ker = theano.shared(numpy.ones(ker_shape, dtype=dtype))
out = theano.shared(numpy.ones((2,2,2), dtype=dtype))

f = theano.function([], theano.tensor.signal.conv.conv2d(img, ker))
T = timeit.Timer(f)
print min(T.repeat(repeat=3, number=1)), 'without shape'

f = theano.function([], [], updates={out:theano.tensor.signal.conv.conv2d(img,
    ker,image_shape=img_shape,filter_shape=ker_shape)})
T = timeit.Timer(f)
print min(T.repeat(repeat=3, number=1)), 'with shape'

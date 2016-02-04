from __future__ import absolute_import, print_function, division
import sys, timeit, time
import numpy
import theano, theano.tensor.signal.conv

try:
    img_shape =  int(sys.argv[1]), int(sys.argv[2])
    ker_shape =  int(sys.argv[3]), int(sys.argv[4])
    dtype = sys.argv[5]
except:
    print("Usage: %s <img rows> <img cols> <ker rows> <ker cols> <dtype> [nb_call]" % sys.argv[0], file=sys.stderr)
    sys.exit(-1)

nb_call = 1
if len(sys.argv)>6:
    nb_call=int(sys.argv[6])

setup="""
import sys, timeit, time
import numpy
import theano, theano.tensor.signal.conv

img_shape =  int(sys.argv[1]), int(sys.argv[2])
ker_shape =  int(sys.argv[3]), int(sys.argv[4])
dtype = sys.argv[5]

img = theano.shared(numpy.ones(img_shape, dtype=dtype))
ker = theano.shared(numpy.ones(ker_shape, dtype=dtype))
out = theano.shared(numpy.ones((2,2,2), dtype=dtype))
"""

T = timeit.Timer("f()", 
                 setup+"f = theano.function([], theano.tensor.signal.conv.conv2d(img, ker))")
time_without_shape = T.repeat(repeat=3, number=nb_call)
print(min(time_without_shape), 'theano without shape')

T = timeit.Timer("f()", setup+"""f = theano.function([], [], 
updates={out:theano.tensor.signal.conv.conv2d(img,
    ker,image_shape=img_shape,filter_shape=ker_shape)})""")
time_with_shape = T.repeat(repeat=3, number=nb_call)

print(min(time_with_shape), 'theano with shape')

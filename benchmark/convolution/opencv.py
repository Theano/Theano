import sys, timeit, time
import numpy
import scikits.image.opencv

try:
    img_shape =  int(sys.argv[1]), int(sys.argv[2])
    ker_shape =  int(sys.argv[3]), int(sys.argv[4])
    dtype = sys.argv[5]
except:
    print >> sys.stderr, "Usage: %s <img rows> <img cols> <ker rows> <ker cols> <dtype>" % sys.argv[0]
    sys.exit(-1)

img = numpy.ones(img_shape, dtype=dtype)
ker = numpy.ones(ker_shape, dtype=dtype)

def f():
    scikits.image.opencv.cvFilter2D(img, ker)

T = timeit.Timer(f)
print min(T.repeat(repeat=3, number=1))


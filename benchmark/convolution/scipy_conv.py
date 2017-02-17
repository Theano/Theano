from __future__ import absolute_import, print_function, division
import sys, timeit
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

T = timeit.Timer("f()","""
from scipy.signal import convolve2d
import numpy

img_shape =  int(sys.argv[1]), int(sys.argv[2])
ker_shape =  int(sys.argv[3]), int(sys.argv[4])
dtype = sys.argv[5]

img = numpy.ones(img_shape, dtype=dtype)
ker = numpy.ones(ker_shape, dtype=dtype)

def f():
    convolve2d(img, ker, mode="valid")
""")
time = T.repeat(repeat=3, number=nb_call)
print(min(time), "scipy")


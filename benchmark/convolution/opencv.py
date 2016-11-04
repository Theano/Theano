from __future__ import absolute_import, print_function, division
import sys, timeit
import numpy as np
import scikits.image.opencv

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
import scikits.image.opencv, sys, numpy as np
img_shape =  int(sys.argv[1]), int(sys.argv[2])
ker_shape =  int(sys.argv[3]), int(sys.argv[4])
dtype = sys.argv[5]

img = np.ones(img_shape, dtype=dtype)
ker = np.ones(ker_shape, dtype=dtype)

def f():
    scikits.image.opencv.cvFilter2D(img, ker)
""")
time = T.repeat(repeat=3, number=nb_call)
print(min(time), "opencv")


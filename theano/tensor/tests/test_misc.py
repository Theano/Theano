import sys
import numpy, theano
from theano import tensor

def test_bug_2009_06_02_trac_387():

    y = tensor.lvector('y')
    #f = theano.function([y], tensor.stack(y[0] / 2))
    #f = theano.function([y], tensor.join(0,tensor.shape_padleft(y[0] / 2,1)))
    f = theano.function([y], tensor.int_div(tensor.DimShuffle(y[0].broadcastable, ['x'])(y[0]), 2))
    sys.stdout.flush()
    print f(numpy.ones(1) * 3)
    #z = tensor.lscalar('z')
    #f = theano.function([z], tensor.DimShuffle([], ['x'])(z) / 2)


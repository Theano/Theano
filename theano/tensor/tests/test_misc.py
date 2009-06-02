import numpy, theano
from theano import tensor

def test_bug_2009_06_02():

    y = tensor.lvector()
    f = theano.function([y], tensor.stack(y[0] / 2))

    print f(numpy.ones(1) * 2)

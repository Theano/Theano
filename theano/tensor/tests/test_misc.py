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

def test_bug_2009_07_17_borrowed_output():
    """Regression test for a bug where output was borrowed by mistake."""
    a = theano.tensor.dmatrix()
    b = theano.tensor.dmatrix()
    # The output should *NOT* be borrowed.
    g = theano.function([a, b],
            theano.Out(theano.tensor.dot(a, b), borrow=False))
    
    x = numpy.zeros((1, 2))
    y = numpy.ones((2, 5))
    
    z = g(x, y)
    print z         # Should be zero.
    x.fill(1)
    print g(x, y)   # Should be non-zero.
    print z         # Should still be zero.
    assert numpy.linalg.norm(z) == 0


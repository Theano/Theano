# This bug is detailed in ticket #387. Please close this ticket once the test
# passes (http://pylearn.org/theano/trac/ticket/387).

import numpy, theano
from theano import tensor

def test_bug_2009_06_02_trac_387():

    y = tensor.lvector()
    f = theano.function([y], tensor.stack(y[0] / 2))

    print f(numpy.ones(1) * 2)

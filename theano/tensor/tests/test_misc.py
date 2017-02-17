from __future__ import absolute_import, print_function, division
import copy
import sys
import numpy
import theano
from theano import tensor
from theano.tensor.nnet import crossentropy_softmax_argmax_1hot_with_bias


def test_bug_2009_06_02_trac_387():
    y = tensor.lvector('y')
    f = theano.function([y],
            tensor.int_div(
                tensor.DimShuffle(y[0].broadcastable, ['x'])(y[0]), 2))
    print(f(numpy.ones(1, dtype='int64') * 3))
    # XXX: there is no assert, nor comment that DEBUGMODE is to do the
    #      checking. What was the bug, and how is it being tested?


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
    print(z)         # Should be zero.
    x.fill(1)
    print(g(x, y))   # Should be non-zero.
    print(z)         # Should still be zero.
    assert numpy.linalg.norm(z) == 0

    # The code above was supposed to fail when it was written (or, more
    # accurately, on the next revision, i.e. when it was merged with the
    # rest of the code, i.e. on revision cac9c9e9f08e).
    # However, for some reason, it does not fail anymore when at this revision.
    # Thus, a new test (below) was added that exhibits the same issue. Note
    # that it may better be moved into the test_nnet.py test file if it turns
    # out the bug was caused by 'crossentropy_softmax_argmax_1hot_with_bias',
    # and was not a more general issue.
    test_output_activation_no_bias = theano.tensor.dmatrix()
    test_b2 = theano.tensor.dvector()
    test_target = theano.tensor.ivector()
    nll_softmax_argmax = (
            crossentropy_softmax_argmax_1hot_with_bias(
                test_output_activation_no_bias,
                test_b2,
                test_target))
    output = nll_softmax_argmax[1]
    g = theano.function([test_output_activation_no_bias, test_b2, test_target],
            theano.Out(output, borrow=False))

    a = numpy.zeros((1, 5))
    b = numpy.ones(5)
    c = numpy.zeros(1, dtype=numpy.int32)

    z = g(a, b, c)
    z_backup = copy.copy(z)
    id_z = id(z)
    print(('Output z after first call: %s' % (z, )))
    a[0, 0] = 1
    id_other = id(g(a, b, c))
    print(('Output z after second call: %s' % (z, )))
    # Ensure that calling the function again returns a pointer towards a new
    # array.
    assert id_z != id_other
    # Just to be 100% sure, ensure that z was not altered.
    assert (z == z_backup).all()


def test_deepcopied_type_filter():
    a = copy.deepcopy(tensor.matrix())

    # The following should run cleanly.
    # As of commit 731e2d2fa68487733320d341d08b454a50c90d12
    # it was failing.
    a.type.filter(
            numpy.ones((2, 2), dtype=a.dtype),
            strict=True)

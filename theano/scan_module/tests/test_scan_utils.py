import theano
from theano.scan_module.scan_utils import equal_computations
from theano.tensor.type_other import NoneConst


def test_equal_compuations():
    # This was a bug report by a Theano user.
    c = NoneConst
    assert equal_computations([c], [c])
    m = theano.tensor.matrix()
    max_argmax1 = theano.tensor.max_and_argmax(m)
    max_argmax2 = theano.tensor.max_and_argmax(m)
    assert equal_computations(max_argmax1, max_argmax2)

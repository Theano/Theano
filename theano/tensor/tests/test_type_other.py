""" This file don't test everything. It only test one past crash error."""
import theano
from theano.gof import Constant
from theano.tensor.type_other import MakeSlice, make_slice, NoneTypeT, NoneConst


def test_make_slice_merge():
    # In the past, this was crahsing during compilation.
    i = theano.tensor.iscalar()
    s1 = make_slice(0, i)
    s2 = make_slice(0, i)
    f = theano.function([i], [s1, s2])
    nodes = f.maker.fgraph.nodes
    assert len([n for n in nodes if isinstance(n.op, MakeSlice)]) == 1
    theano.printing.debugprint(f)


def test_none_Constant():
    """ Tests equals

    We had an error in the past with unpickling
    """
    o1 = Constant(NoneTypeT(), None, name='NoneConst')
    o2 = Constant(NoneTypeT(), None, name='NoneConst')
    assert o1.equals(o2)
    assert NoneConst.equals(o1)
    assert o1.equals(NoneConst)
    assert NoneConst.equals(o2)
    assert o2.equals(NoneConst)

    # This trigger equals that returned the wrong answer in the past.
    import cPickle
    import theano
    from theano import tensor

    x = tensor.vector('x')
    y = tensor.argmax(x)
    f = theano.function([x], [y])
    cPickle.loads(cPickle.dumps(f))

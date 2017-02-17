""" This file don't test everything. It only test one past crash error."""
from __future__ import absolute_import, print_function, division
import theano
from theano.gof import Constant
from theano.tensor.type_other import MakeSlice, make_slice, NoneTypeT, NoneConst


def test_make_slice_merge():
    # In the past, this was crahsing during compilation.
    i = theano.tensor.iscalar()
    s1 = make_slice(0, i)
    s2 = make_slice(0, i)
    f = theano.function([i], [s1, s2])
    nodes = f.maker.fgraph.apply_nodes
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
    import six.moves.cPickle as pickle
    import theano
    from theano import tensor

    x = tensor.vector('x')
    y = tensor.argmax(x)
    kwargs = {}
    # We can't pickle DebugMode
    if theano.config.mode in ["DebugMode", "DEBUG_MODE"]:
        kwargs = {'mode': 'FAST_RUN'}
    f = theano.function([x], [y], **kwargs)
    pickle.loads(pickle.dumps(f))

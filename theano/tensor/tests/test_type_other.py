""" This file don't test everything. It only test one past crash error."""
import theano
from theano.tensor.type_other import MakeSlice, make_slice


def test_make_slice_merge():
    # In the past, this was crahsing during compilation.
    i = theano.tensor.iscalar()
    s1 = make_slice(0, i)
    s2 = make_slice(0, i)
    f = theano.function([i], [s1, s2])
    nodes = f.maker.fgraph.nodes
    assert len([n for n in nodes if isinstance(n.op, MakeSlice)]) == 1
    theano.printing.debugprint(f)
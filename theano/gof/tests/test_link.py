from __future__ import absolute_import, print_function, division
from copy import deepcopy
import unittest

import numpy

import theano
from theano.gof import graph
from theano.gof.graph import Variable, Apply, Constant
from theano.gof.type import Type
from theano.gof.op import Op
from theano.gof import fg

from theano.gof.link import *  # noqa
from theano.compat import cmp


def as_variable(x):
    assert isinstance(x, Variable)
    return x


class TDouble(Type):
    def filter(self, data):
        return float(data)

tdouble = TDouble()


def double(name):
    return Variable(tdouble, None, None, name=name)


class MyOp(Op):

    __props__ = ("nin", "name", "impl")

    def __init__(self, nin, name, impl=None):
        self.nin = nin
        self.name = name
        if impl:
            self.impl = impl

    def make_node(self, *inputs):
        assert len(inputs) == self.nin
        inputs = [as_variable(i) for i in inputs]
        for input in inputs:
            if input.type is not tdouble:
                raise Exception("Error 1")
        outputs = [double(self.name + "_R")]
        return Apply(self, inputs, outputs)

    def __str__(self):
        return self.name

    def perform(self, node, inputs, out_):
        out, = out_
        out[0] = self.impl(*inputs)

add = MyOp(2, 'Add', lambda x, y: x + y)
sub = MyOp(2, 'Sub', lambda x, y: x - y)
mul = MyOp(2, 'Mul', lambda x, y: x * y)
div = MyOp(2, 'Div', lambda x, y: x / y)


def notimpl(self, x):
    raise NotImplementedError()


raise_err = MyOp(1, 'RaiseErr', notimpl)


def inputs():
    x = double('x')
    y = double('y')
    z = double('z')
    return x, y, z


def perform_linker(fgraph):
    lnk = PerformLinker().accept(fgraph)
    return lnk


def FunctionGraph(inputs, outputs):
    e = fg.FunctionGraph(inputs, outputs)
    return e


class TestPerformLinker(unittest.TestCase):
    def test_thunk(self):
        x, y, z = inputs()
        e = mul(add(x, y), div(x, y))
        fn, i, o = perform_linker(FunctionGraph([x, y, z], [e])).make_thunk()
        i[0].data = 1
        i[1].data = 2
        fn()
        assert o[0].data == 1.5

    def test_function(self):
        x, y, z = inputs()
        e = mul(add(x, y), div(x, y))
        fn = perform_linker(FunctionGraph([x, y, z], [e])).make_function()
        assert fn(1.0, 2.0, 3.0) == 1.5

    def test_constant(self):
        x, y, z = inputs()
        y = Constant(tdouble, 2.0)
        e = mul(add(x, y), div(x, y))
        fn = perform_linker(FunctionGraph([x], [e])).make_function()
        assert fn(1.0) == 1.5

    def test_input_output_same(self):
        x, y, z = inputs()
        fn = perform_linker(FunctionGraph([x], [x])).make_function()
        assert 1.0 is fn(1.0)

    def test_input_dependency0(self):
        x, y, z = inputs()
        a, d = add(x, y), div(x, y)
        e = mul(a, d)
        fn = perform_linker(FunctionGraph(*graph.clone([x, y, a],
                                                       [e]))).make_function()
        assert fn(1.0, 2.0, 9.0) == 4.5

    def test_skiphole(self):
        x, y, z = inputs()
        a = add(x, y)
        r = raise_err(a)
        e = add(r, a)
        fn = perform_linker(FunctionGraph(*graph.clone([x, y, r],
                                                       [e]))).make_function()
        assert fn(1.0, 2.0, 4.5) == 7.5


def wrap_linker(fgraph, linkers, wrapper):
    lnk = WrapLinker(linkers, wrapper).accept(fgraph)
    return lnk


class TestWrapLinker(unittest.TestCase):
    def test_0(self):
        nodes = []

        def wrap(i, node, th):
            nodes.append(node.op)

        x, y, z = inputs()
        e = mul(add(x, y), div(x, y))
        fn, i, o = wrap_linker(
            FunctionGraph([x, y, z], [e]),
            [PerformLinker(allow_gc=False)], wrap).make_thunk()
        i[0].data = 1
        i[1].data = 2
        fn()
        assert nodes == [div, add, mul]
        assert o[0].data is None

    def test_1(self):
        nodes = []

        def wrap(i, node, th):
            nodes.append(node.op)
            th()

        x, y, z = inputs()
        e = mul(add(x, y), div(x, y))
        fn, i, o = wrap_linker(
            FunctionGraph([x, y, z], [e]),
            [PerformLinker(allow_gc=False)], wrap).make_thunk()
        i[0].data = 1
        i[1].data = 2
        fn()
        assert nodes == [div, add, mul]
        assert o[0].data == 1.5


def test_sort_schedule_fn():
    import theano
    from theano.gof.sched import sort_schedule_fn, make_depends
    x = theano.tensor.matrix('x')
    y = theano.tensor.dot(x[:5] * 2, x.T + 1).T

    def str_cmp(a, b):
        return cmp(str(a), str(b))  # lexicographical sort

    linker = theano.OpWiseCLinker(schedule=sort_schedule_fn(str_cmp))
    mode = theano.Mode(linker=linker)
    f = theano.function((x,), (y,), mode=mode)

    nodes = f.maker.linker.make_all()[-1]
    depends = make_depends()
    for a, b in zip(nodes[:-1], nodes[1:]):
        if not depends((b, a)):
            assert str(a) < str(b)


def test_container_deepcopy():
    """
    This is a test to a work around a NumPy bug.
    """
    t = theano.tensor.scalar()
    # It seam that numpy.asarray(0.).astype(floatX) can return a numpy
    # scalar with some NumPy Version. So we call numpy.asarray with
    # the dtype parameter.
    v = numpy.asarray(0., dtype=theano.config.floatX)
    assert isinstance(v, numpy.ndarray), type(v)
    for readonly in [True, False]:
        c = Container(t, [v], readonly=readonly)
        assert isinstance(c.storage[0], numpy.ndarray), (c.storage[0],
                                                         type(c.storage[0]))
        assert c.storage[0].dtype == v.dtype, (c.storage[0].dtype, v.dtype)
        assert c.storage[0].dtype == c.type.dtype, (c.storage[0].dtype,
                                                    c.type.dtype)
        d = deepcopy(c)
        assert isinstance(d.storage[0], numpy.ndarray), (d.storage[0],
                                                         type(d.storage[0]))
        assert d.storage[0].dtype == v.dtype, (d.storage[0].dtype, v.dtype)
        assert d.storage[0].dtype == c.type.dtype, (d.storage[0].dtype,
                                                    c.type.dtype)

from copy import copy

import theano

from theano.gof.op import *
from theano.gof.type import Type, Generic
from theano.gof.graph import Apply, Variable

def as_variable(x):
    assert isinstance(x, Variable)
    return x


class MyType(Type):

    def __init__(self, thingy):
        self.thingy = thingy

    def __eq__(self, other):
        return type(other) == type(self) and other.thingy == self.thingy

    def __str__(self):
        return str(self.thingy)

    def __repr__(self):
        return str(self.thingy)

    def filter(self, x, strict=False, allow_downcast=None):
        # Dummy filter: we want this type to represent strings that
        # start with `self.thingy`.
        assert isinstance(x, str) and x.startswith(self.thingy)
        return x

class MyOp(Op):

    def make_node(self, *inputs):
        inputs = map(as_variable, inputs)
        for input in inputs:
            if not isinstance(input.type, MyType):
                raise Exception("Error 1")
            outputs = [MyType(sum([input.type.thingy for input in inputs]))()]
            return Apply(self, inputs, outputs)

MyOp = MyOp()


class NoInputOp(Op):

    """An Op to test the corner-case of an Op with no input."""

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self):
        return Apply(self, [], [MyType('test')()])

    def perform(self, node, inputs, output_storage):
        output_storage[0][0] = 'test Op no input'


class TestOp:

    # Sanity tests
    def test_sanity_0(self):
        r1, r2 = MyType(1)(), MyType(2)()
        node = MyOp.make_node(r1, r2)
        assert [x for x in node.inputs] == [r1, r2] # Are the inputs what I provided?
        assert [x.type for x in node.outputs] == [MyType(3)] # Are the outputs what I expect?
        assert node.outputs[0].owner is node and node.outputs[0].index == 0

    # validate
    def test_validate(self):
        try:
            MyOp(Generic()(), MyType(1)()) # MyOp requires MyType instances
            raise Exception("Expected an exception")
        except Exception, e:
            if str(e) != "Error 1":
                raise

    def test_op_no_input(self):
        x = NoInputOp()()
        f = theano.function([], x)
        rval = f()
        assert rval == 'test Op no input'


if __name__ == '__main__':
    unittest.main()

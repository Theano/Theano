
import unittest
from copy import copy
from op import *
from type import Type, Generic
from graph import Apply, Result

def as_result(x):
    assert isinstance(x, Result)
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


class MyOp(Op):

    def make_node(self, *inputs):
        inputs = map(as_result, inputs)
        for input in inputs:
            if not isinstance(input.type, MyType):
                raise Exception("Error 1")
            outputs = [MyType(sum([input.type.thingy for input in inputs]))()]
            return Apply(self, inputs, outputs)

MyOp = MyOp()


class _test_Op(unittest.TestCase):

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



if __name__ == '__main__':
    unittest.main()

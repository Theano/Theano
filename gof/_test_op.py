
import unittest
from copy import copy
from op import *
from result import ResultBase


class MyResult(ResultBase):

    def __init__(self, thingy):
        self.thingy = thingy
        ResultBase.__init__(self, role = None)
        self.data = [self.thingy]

    def __eq__(self, other):
        return self.same_properties(other)

    def same_properties(self, other):
        return isinstance(other, MyResult) and other.thingy == self.thingy

    def __str__(self):
        return str(self.thingy)

    def __repr__(self):
        return str(self.thingy)


class MyOp(Op):

    def __init__(self, *inputs):
        for input in inputs:
            if not isinstance(input, MyResult):
                raise Exception("Error 1")
        self.inputs = inputs
        self.outputs = [MyResult(sum([input.thingy for input in inputs]))]


class _test_Op(unittest.TestCase):

    # Sanity tests
    def test_sanity_0(self):
        r1, r2 = MyResult(1), MyResult(2)
        op = MyOp(r1, r2)
        assert op.inputs == [r1, r2] # Are the inputs what I provided?
        assert op.outputs == [MyResult(3)] # Are the outputs what I expect?
        assert op.outputs[0].owner is op and op.outputs[0].index == 0

    # validate_update
    def test_validate_update(self):
        try:
            MyOp(ResultBase(), MyResult(1)) # MyOp requires MyResult instances
        except Exception, e:
            assert str(e) == "Error 1"
        else:
            raise Exception("Expected an exception")



if __name__ == '__main__':
    unittest.main()

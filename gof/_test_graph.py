

import unittest
from graph import *

from op import Op
from result import ResultBase


class MyResult(ResultBase):

    def __init__(self, thingy):
        self.thingy = thingy
        ResultBase.__init__(self, role = None )
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


class _test_inputs(unittest.TestCase):

    def test_0(self):
        r1, r2 = MyResult(1), MyResult(2)
        op = MyOp(r1, r2)
        assert inputs(op.outputs) == set([r1, r2])

    def test_1(self):
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        op = MyOp(r1, r2)
        op2 = MyOp(op.outputs[0], r5)
        assert inputs(op2.outputs) == set([r1, r2, r5])


class _test_orphans(unittest.TestCase):

    def test_0(self):
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        op = MyOp(r1, r2)
        op2 = MyOp(op.outputs[0], r5)
        assert orphans([r1, r2], op2.outputs) == set([r5])

    def test_1(self):
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        op = MyOp(r1, r2)
        op2 = MyOp(op.outputs[0], r5)
        try:
            ro = results_and_orphans([r1, r2, op2.outputs[0]], op.outputs, True)
            self.fail()
        except Exception, e:
            if e[0] is results_and_orphans.E_unreached:
                return
            raise
    

class _test_as_string(unittest.TestCase):

    leaf_formatter = str
    node_formatter = lambda op, argstrings: "%s(%s)" % (op.__class__.__name__,
                                                        ", ".join(argstrings))

    def test_0(self):
        r1, r2 = MyResult(1), MyResult(2)
        op = MyOp(r1, r2)
        assert as_string([r1, r2], op.outputs) == ["MyOp(1, 2)"]

    def test_1(self):
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        op = MyOp(r1, r2)
        op2 = MyOp(op.outputs[0], r5)
        assert as_string([r1, r2, r5], op2.outputs) == ["MyOp(MyOp(1, 2), 5)"]

    def test_2(self):
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        op = MyOp(r1, r2)
        op2 = MyOp(op.outputs[0], op.outputs[0])
        assert as_string([r1, r2, r5], op2.outputs) == ["MyOp(*1 -> MyOp(1, 2), *1)"]

    def test_3(self):
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        op = MyOp(r1, r2)
        op2 = MyOp(op.outputs[0], op.outputs[0])
        assert as_string(op.outputs, op2.outputs) == ["MyOp(3, 3)"]
        assert as_string(op2.inputs, op2.outputs) == ["MyOp(3, 3)"]


class _test_clone(unittest.TestCase):

    def test_0(self):
        r1, r2 = MyResult(1), MyResult(2)
        op = MyOp(r1, r2)
        new = clone([r1, r2], op.outputs)
        assert as_string([r1, r2], new) == ["MyOp(1, 2)"]

    def test_1(self):
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        op = MyOp(r1, r2)
        op2 = MyOp(op.outputs[0], r5)
        new = clone([r1, r2, r5], op2.outputs)
        assert op2.outputs[0] == new[0] and op2.outputs[0] is not new[0]
        assert op2 is not new[0].owner
        assert new[0].owner.inputs[1] is r5
        assert new[0].owner.inputs[0] == op.outputs[0] and new[0].owner.inputs[0] is not op.outputs[0]

    def test_2(self):
        "Checks that manipulating a cloned graph leaves the original unchanged."
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        op = MyOp(MyOp(r1, r2).outputs[0], r5)
        new = clone([r1, r2, r5], op.outputs)
        new_op = new[0].owner
        new_op.inputs = MyResult(7), MyResult(8)

        assert as_string(inputs(new_op.outputs), new_op.outputs) == ["MyOp(7, 8)"]
        assert as_string(inputs(op.outputs), op.outputs) == ["MyOp(MyOp(1, 2), 5)"]



if __name__ == '__main__':
    unittest.main()




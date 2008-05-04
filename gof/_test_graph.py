

import unittest
from graph import *

from op import Op
from result import Result

if 1:
    testcase = unittest.TestCase
else:
    testcase = object
    realtestcase = unittest.TestCase


class MyResult(Result):

    def __init__(self, thingy):
        self.thingy = thingy
        Result.__init__(self, role = None )
        self.data = [self.thingy]

    def __eq__(self, other):
        return self.same_properties(other)

    def same_properties(self, other):
        return isinstance(other, MyResult) and other.thingy == self.thingy

    def __str__(self):
        return 'R%s' % str(self.thingy)

    def __repr__(self):
        return 'R%s' % str(self.thingy)


class MyOp(Op):

    def __init__(self, *inputs):
        for input in inputs:
            if not isinstance(input, MyResult):
                raise Exception("Error 1")
        self.inputs = inputs
        self.outputs = [MyResult(sum([input.thingy for input in inputs]))]


class _test_inputs(testcase):

    def test_straightforward(self):
        r1, r2 = MyResult(1), MyResult(2)
        op = MyOp(r1, r2)
        assert inputs(op.outputs) == [r1, r2]

    def test_deep(self):
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        op = MyOp(r1, r2)
        op2 = MyOp(op.outputs[0], r5)
        i = inputs(op2.outputs)
        self.failUnless(i == [r1, r2, r5], i)

    def test_unreached_inputs(self):
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        op = MyOp(r1, r2)
        op2 = MyOp(op.outputs[0], r5)
        try:
            # function doesn't raise if we put False instead of True
            ro = results_and_orphans([r1, r2, op2.outputs[0]], op.outputs, True)
        except Exception, e:
            if e[0] is results_and_orphans.E_unreached:
                return
        self.fail()

    def test_uz(self):
        pass


class _test_orphans(testcase):

    def test_straightforward(self):
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        op = MyOp(r1, r2)
        op2 = MyOp(op.outputs[0], r5)
        orph = orphans([r1, r2], op2.outputs)
        self.failUnless(orph == [r5], orph)
    

class _test_as_string(testcase):

    leaf_formatter = str
    node_formatter = lambda op, argstrings: "%s(%s)" % (op.__class__.__name__,
                                                        ", ".join(argstrings))

    def test_straightforward(self):
        r1, r2 = MyResult(1), MyResult(2)
        op = MyOp(r1, r2)
        s = as_string([r1, r2], op.outputs)
        self.failUnless(s == ["MyOp(R1, R2)"], s)

    def test_deep(self):
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        op = MyOp(r1, r2)
        op2 = MyOp(op.outputs[0], r5)
        s = as_string([r1, r2, r5], op2.outputs)
        self.failUnless(s == ["MyOp(MyOp(R1, R2), R5)"], s)

    def test_multiple_references(self):
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        op = MyOp(r1, r2)
        op2 = MyOp(op.outputs[0], op.outputs[0])
        assert as_string([r1, r2, r5], op2.outputs) == ["MyOp(*1 -> MyOp(R1, R2), *1)"]

    def test_cutoff(self):
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        op = MyOp(r1, r2)
        op2 = MyOp(op.outputs[0], op.outputs[0])
        assert as_string(op.outputs, op2.outputs) == ["MyOp(R3, R3)"]
        assert as_string(op2.inputs, op2.outputs) == ["MyOp(R3, R3)"]


class _test_clone(testcase):

    def test_accurate(self):
        r1, r2 = MyResult(1), MyResult(2)
        op = MyOp(r1, r2)
        new = clone([r1, r2], op.outputs)
        assert as_string([r1, r2], new) == ["MyOp(R1, R2)"]

    def test_copy(self):
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        op = MyOp(r1, r2)
        op2 = MyOp(op.outputs[0], r5)
        new = clone([r1, r2, r5], op2.outputs)
        assert op2.outputs[0] == new[0] and op2.outputs[0] is not new[0] # the new output is like the old one but not the same object
        assert op2 is not new[0].owner # the new output has a new owner
        assert new[0].owner.inputs[1] is r5 # the inputs are not copied
        assert new[0].owner.inputs[0] == op.outputs[0] and new[0].owner.inputs[0] is not op.outputs[0] # check that we copied deeper too

    def test_not_destructive(self):
        # Checks that manipulating a cloned graph leaves the original unchanged.
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        op = MyOp(MyOp(r1, r2).outputs[0], r5)
        new = clone([r1, r2, r5], op.outputs)
        new_op = new[0].owner
        new_op.inputs = MyResult(7), MyResult(8)

        s = as_string(inputs(new_op.outputs), new_op.outputs)
        self.failUnless( s == ["MyOp(R7, R8)"], s)
        assert as_string(inputs(op.outputs), op.outputs) == ["MyOp(MyOp(R1, R2), R5)"]



if __name__ == '__main__':
    unittest.main()
    #_test_inputs('test_unreached_inputs').debug()


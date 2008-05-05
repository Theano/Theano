
from collections import deque
import unittest
from graph import *

from op import Op
from result import Result

def inputs(result_list):
    """
    @type result_list: list of L{Result}
    @param result_list: output L{Result}s (from which to search backward through owners)
    @returns: the list of L{Result}s with no owner, in the order found by a
    left-recursive depth-first search started at the L{Result}s in result_list.

    """
    def expand(r):
        if r.owner:
            l = list(r.owner.inputs)
            l.reverse()
            return l
    dfs_results = stack_search(deque(result_list), expand, 'dfs')
    rval = [r for r in dfs_results if r.owner is None]
    #print rval, _orig_inputs(o)
    return rval

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

def prenode(obj):
    if isinstance(obj, Result): 
        if obj.owner:
            return [obj.owner]
    if isinstance(obj, Op):
        return obj.inputs

class _test_toposort(testcase):
    def test0(self):
        """Test a simple graph"""
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        o = MyOp(r1, r2)
        o2 = MyOp(o.outputs[0], r5)

        all = general_toposort(o2.outputs, prenode)
        self.failUnless(all == [r5, r2, r1, o, o.outputs[0], o2, o2.outputs[0]], all)

        all = io_toposort([r5], o2.outputs)
        self.failUnless(all == [o, o2], all)

    def test1(self):
        """Test a graph with double dependencies"""
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        o = MyOp(r1, r1)
        o2 = MyOp(o.outputs[0], r5)
        all = general_toposort(o2.outputs, prenode)
        self.failUnless(all == [r5, r1, o, o.outputs[0], o2, o2.outputs[0]], all)

    def test2(self):
        """Test a graph where the inputs have owners"""
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        o = MyOp(r1, r1)
        r2b = o.outputs[0]
        o2 = MyOp(r2b, r2b)
        all = io_toposort([r2b], o2.outputs)
        self.failUnless(all == [o2], all)

        o2 = MyOp(r2b, r5)
        all = io_toposort([r2b], o2.outputs)
        self.failUnless(all == [o2], all)

    def test3(self):
        """Test a graph which is not connected"""
        r1, r2, r3, r4 = MyResult(1), MyResult(2), MyResult(3), MyResult(4)
        o0 = MyOp(r1, r2)
        o1 = MyOp(r3, r4)
        all = io_toposort([r1, r2, r3, r4], o0.outputs + o1.outputs)
        self.failUnless(all == [o1,o0], all)

    def test4(self):
        """Test inputs and outputs mixed together in a chain graph"""
        r1, r2, r3, r4 = MyResult(1), MyResult(2), MyResult(3), MyResult(4)
        o0 = MyOp(r1, r2)
        o1 = MyOp(o0.outputs[0], r1)
        all = io_toposort([r1, o0.outputs[0]], [o0.outputs[0], o1.outputs[0]])
        self.failUnless(all == [o1], all)

    def test5(self):
        """Test when outputs have clients"""
        r1, r2, r3, r4 = MyResult(1), MyResult(2), MyResult(3), MyResult(4)
        o0 = MyOp(r1, r2)
        o1 = MyOp(o0.outputs[0], r4)
        all = io_toposort([], o0.outputs)
        self.failUnless(all == [o0], all)




if __name__ == '__main__':
    if 1:
        #run all tests
        unittest.main()
    elif 1:
        #load some TestCase classes
        suite = unittest.TestLoader()
        suite = suite.loadTestsFromTestCase(_test_toposort)

        #run just some of them
        unittest.TextTestRunner(verbosity=2).run(suite)

    else:
        #run just a single test
        _test_toposort('test0').debug()


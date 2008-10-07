
from collections import deque
import unittest
from theano.gof.graph import *

from theano.gof.op import Op
from theano.gof.type import Type
from theano.gof.graph import Result


if 1:
    testcase = unittest.TestCase
else:
    testcase = object
    realtestcase = unittest.TestCase


def as_result(x):
    assert isinstance(x, Result)
    return x


class MyType(Type):

    def __init__(self, thingy):
        self.thingy = thingy

    def __eq__(self, other):
        return isinstance(other, MyType) and other.thingy == self.thingy

    def __str__(self):
        return 'R%s' % str(self.thingy)

    def __repr__(self):
        return 'R%s' % str(self.thingy)

def MyResult(thingy):
    return Result(MyType(thingy), None, None)


class MyOp(Op):

    def make_node(self, *inputs):
        inputs = map(as_result, inputs)
        for input in inputs:
            if not isinstance(input.type, MyType):
                print input, input.type, type(input), type(input.type)
                raise Exception("Error 1")
        outputs = [MyResult(sum([input.type.thingy for input in inputs]))]
        return Apply(self, inputs, outputs)

    def __str__(self):
        return self.__class__.__name__

MyOp = MyOp()



# class MyResult(Result):

#     def __init__(self, thingy):
#         self.thingy = thingy
#         Result.__init__(self, role = None )
#         self.data = [self.thingy]

#     def __eq__(self, other):
#         return self.same_properties(other)

#     def same_properties(self, other):
#         return isinstance(other, MyResult) and other.thingy == self.thingy

#     def __str__(self):
#         return str(self.thingy)

#     def __repr__(self):
#         return str(self.thingy)


# class MyOp(Op):

#     def __init__(self, *inputs):
#         for input in inputs:
#             if not isinstance(input, MyResult):
#                 raise Exception("Error 1")
#         self.inputs = inputs
#         self.outputs = [MyResult(sum([input.thingy for input in inputs]))]


class _test_inputs(testcase):

    def test_straightforward(self):
        r1, r2 = MyResult(1), MyResult(2)
        node = MyOp.make_node(r1, r2)
        assert inputs(node.outputs) == [r1, r2]

    def test_deep(self):
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        node = MyOp.make_node(r1, r2)
        node2 = MyOp.make_node(node.outputs[0], r5)
        i = inputs(node2.outputs)
        self.failUnless(i == [r1, r2, r5], i)

#     def test_unreached_inputs(self):
#         r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
#         op = MyOp(r1, r2)
#         op2 = MyOp(op.outputs[0], r5)
#         try:
#             # function doesn't raise if we put False instead of True
#             ro = results_and_orphans([r1, r2, op2.outputs[0]], op.outputs, True)
#         except Exception, e:
#             if e[0] is results_and_orphans.E_unreached:
#                 return
#         self.fail()


# class _test_orphans(testcase):

#     def test_straightforward(self):
#         r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
#         node = MyOp.make_node(r1, r2)
#         node2 = MyOp.make_node(node.outputs[0], r5)
#         orph = orphans([r1, r2], node2.outputs)
#         self.failUnless(orph == [r5], orph)
    

class _test_as_string(testcase):

    leaf_formatter = lambda self, leaf: str(leaf.type)
    node_formatter = lambda self, node, argstrings: "%s(%s)" % (node.op,
                                                                ", ".join(argstrings))

    def str(self, inputs, outputs):
        return as_string(inputs, outputs,
                         leaf_formatter = self.leaf_formatter,
                         node_formatter = self.node_formatter)

    def test_straightforward(self):
        r1, r2 = MyResult(1), MyResult(2)
        node = MyOp.make_node(r1, r2)
        s = self.str([r1, r2], node.outputs)
        self.failUnless(s == ["MyOp(R1, R2)"], s)

    def test_deep(self):
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        node = MyOp.make_node(r1, r2)
        node2 = MyOp.make_node(node.outputs[0], r5)
        s = self.str([r1, r2, r5], node2.outputs)
        self.failUnless(s == ["MyOp(MyOp(R1, R2), R5)"], s)

    def test_multiple_references(self):
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        node = MyOp.make_node(r1, r2)
        node2 = MyOp.make_node(node.outputs[0], node.outputs[0])
        assert self.str([r1, r2, r5], node2.outputs) == ["MyOp(*1 -> MyOp(R1, R2), *1)"]

    def test_cutoff(self):
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        node = MyOp.make_node(r1, r2)
        node2 = MyOp.make_node(node.outputs[0], node.outputs[0])
        assert self.str(node.outputs, node2.outputs) == ["MyOp(R3, R3)"]
        assert self.str(node2.inputs, node2.outputs) == ["MyOp(R3, R3)"]


class _test_clone(testcase):

    leaf_formatter = lambda self, leaf: str(leaf.type)
    node_formatter = lambda self, node, argstrings: "%s(%s)" % (node.op,
                                                                ", ".join(argstrings))

    def str(self, inputs, outputs):
        return as_string(inputs, outputs,
                         leaf_formatter = self.leaf_formatter,
                         node_formatter = self.node_formatter)

    def test_accurate(self):
        r1, r2 = MyResult(1), MyResult(2)
        node = MyOp.make_node(r1, r2)
        _, new = clone([r1, r2], node.outputs, False)
        assert self.str([r1, r2], new) == ["MyOp(R1, R2)"]

    def test_copy(self):
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        node = MyOp.make_node(r1, r2)
        node2 = MyOp.make_node(node.outputs[0], r5)
        _, new = clone([r1, r2, r5], node2.outputs, False)
        assert node2.outputs[0].type == new[0].type and node2.outputs[0] is not new[0] # the new output is like the old one but not the same object
        assert node2 is not new[0].owner # the new output has a new owner
        assert new[0].owner.inputs[1] is r5 # the inputs are not copied
        assert new[0].owner.inputs[0].type == node.outputs[0].type and new[0].owner.inputs[0] is not node.outputs[0] # check that we copied deeper too

    def test_not_destructive(self):
        # Checks that manipulating a cloned graph leaves the original unchanged.
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        node = MyOp.make_node(MyOp.make_node(r1, r2).outputs[0], r5)
        _, new = clone([r1, r2, r5], node.outputs, False)
        new_node = new[0].owner
        new_node.inputs = MyResult(7), MyResult(8)
        assert self.str(inputs(new_node.outputs), new_node.outputs) == ["MyOp(R7, R8)"]
        assert self.str(inputs(node.outputs), node.outputs) == ["MyOp(MyOp(R1, R2), R5)"]

def prenode(obj):
    if isinstance(obj, Result): 
        if obj.owner:
            return [obj.owner]
    if isinstance(obj, Apply):
        return obj.inputs

class _test_toposort(testcase):
    def test0(self):
        """Test a simple graph"""
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        o = MyOp.make_node(r1, r2)
        o2 = MyOp.make_node(o.outputs[0], r5)

        all = general_toposort(o2.outputs, prenode)
        self.failUnless(all == [r5, r2, r1, o, o.outputs[0], o2, o2.outputs[0]], all)

        all = io_toposort([r5], o2.outputs)
        self.failUnless(all == [o, o2], all)

    def test1(self):
        """Test a graph with double dependencies"""
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        o = MyOp.make_node(r1, r1)
        o2 = MyOp.make_node(o.outputs[0], r5)
        all = general_toposort(o2.outputs, prenode)
        self.failUnless(all == [r5, r1, o, o.outputs[0], o2, o2.outputs[0]], all)

    def test2(self):
        """Test a graph where the inputs have owners"""
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        o = MyOp.make_node(r1, r1)
        r2b = o.outputs[0]
        o2 = MyOp.make_node(r2b, r2b)
        all = io_toposort([r2b], o2.outputs)
        self.failUnless(all == [o2], all)

        o2 = MyOp.make_node(r2b, r5)
        all = io_toposort([r2b], o2.outputs)
        self.failUnless(all == [o2], all)

    def test3(self):
        """Test a graph which is not connected"""
        r1, r2, r3, r4 = MyResult(1), MyResult(2), MyResult(3), MyResult(4)
        o0 = MyOp.make_node(r1, r2)
        o1 = MyOp.make_node(r3, r4)
        all = io_toposort([r1, r2, r3, r4], o0.outputs + o1.outputs)
        self.failUnless(all == [o1,o0], all)

    def test4(self):
        """Test inputs and outputs mixed together in a chain graph"""
        r1, r2, r3, r4 = MyResult(1), MyResult(2), MyResult(3), MyResult(4)
        o0 = MyOp.make_node(r1, r2)
        o1 = MyOp.make_node(o0.outputs[0], r1)
        all = io_toposort([r1, o0.outputs[0]], [o0.outputs[0], o1.outputs[0]])
        self.failUnless(all == [o1], all)

    def test5(self):
        """Test when outputs have clients"""
        r1, r2, r3, r4 = MyResult(1), MyResult(2), MyResult(3), MyResult(4)
        o0 = MyOp.make_node(r1, r2)
        o1 = MyOp.make_node(o0.outputs[0], r4)
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


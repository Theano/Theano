

import unittest
from graph import *

from op import Op
from type import Type
from graph import Result



class MyType(Type):

    def __init__(self, thingy):
        self.thingy = thingy

    def __eq__(self, other):
        return isinstance(other, MyType) and other.thingy == self.thingy

    def __str__(self):
        return str(self.thingy)

    def __repr__(self):
        return str(self.thingy)

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


class _test_inputs(unittest.TestCase):

    def test_straightforward(self):
        r1, r2 = MyResult(1), MyResult(2)
        node = MyOp.make_node(r1, r2)
        assert inputs(node.outputs) == set([r1, r2])

    def test_deep(self):
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        node = MyOp.make_node(r1, r2)
        node2 = MyOp.make_node(node.outputs[0], r5)
        assert inputs(node2.outputs) == set([r1, r2, r5])

#     def test_unreached_inputs(self):
#         r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
#         node = MyOp.make_node(r1, r2)
#         node2 = MyOp.make_node(node.outputs[0], r5)
#         try:
#             # function doesn't raise if we put False instead of True
#             ro = results_and_orphans([r1, r2, node2.outputs[0]], node.outputs, True)
#             self.fail()
#         except Exception, e:
#             if e[0] is results_and_orphans.E_unreached:
#                 return
#             raise


class _test_orphans(unittest.TestCase):

    def test_straightforward(self):
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        node = MyOp.make_node(r1, r2)
        node2 = MyOp.make_node(node.outputs[0], r5)
        assert orphans([r1, r2], node2.outputs) == set([r5])
    

class _test_as_string(unittest.TestCase):

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
        assert self.str([r1, r2], node.outputs) == ["MyOp(1, 2)"]

    def test_deep(self):
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        node = MyOp.make_node(r1, r2)
        node2 = MyOp.make_node(node.outputs[0], r5)
        assert self.str([r1, r2, r5], node2.outputs) == ["MyOp(MyOp(1, 2), 5)"]

    def test_multiple_references(self):
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        node = MyOp.make_node(r1, r2)
        node2 = MyOp.make_node(node.outputs[0], node.outputs[0])
        assert self.str([r1, r2, r5], node2.outputs) == ["MyOp(*1 -> MyOp(1, 2), *1)"]

    def test_cutoff(self):
        r1, r2, r5 = MyResult(1), MyResult(2), MyResult(5)
        node = MyOp.make_node(r1, r2)
        node2 = MyOp.make_node(node.outputs[0], node.outputs[0])
        assert self.str(node.outputs, node2.outputs) == ["MyOp(3, 3)"]
        assert self.str(node2.inputs, node2.outputs) == ["MyOp(3, 3)"]


class _test_clone(unittest.TestCase):

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
        assert self.str([r1, r2], new) == ["MyOp(1, 2)"]

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

        assert self.str(inputs(new_node.outputs), new_node.outputs) == ["MyOp(7, 8)"]
        assert self.str(inputs(node.outputs), node.outputs) == ["MyOp(MyOp(1, 2), 5)"]



if __name__ == '__main__':
    unittest.main()




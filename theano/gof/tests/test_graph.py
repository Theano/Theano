import unittest

from theano import tensor
from theano.gof.graph import (
        Apply, as_string, clone, general_toposort, inputs, io_toposort,
        is_same_graph, Variable)
from theano.gof.op import Op
from theano.gof.type import Type


def as_variable(x):
    assert isinstance(x, Variable)
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

def MyVariable(thingy):
    return Variable(MyType(thingy), None, None)


class MyOp(Op):

    def make_node(self, *inputs):
        inputs = map(as_variable, inputs)
        for input in inputs:
            if not isinstance(input.type, MyType):
                print input, input.type, type(input), type(input.type)
                raise Exception("Error 1")
        outputs = [MyVariable(sum([input.type.thingy for input in inputs]))]
        return Apply(self, inputs, outputs)

    def __str__(self):
        return self.__class__.__name__

MyOp = MyOp()


##########
# inputs #
##########

class TestInputs:

    def test_inputs(self):
        r1, r2 = MyVariable(1), MyVariable(2)
        node = MyOp.make_node(r1, r2)
        assert inputs(node.outputs) == [r1, r2]

    def test_inputs_deep(self):
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        node = MyOp.make_node(r1, r2)
        node2 = MyOp.make_node(node.outputs[0], r5)
        i = inputs(node2.outputs)
        assert i == [r1, r2, r5], i


#############
# as_string #
#############


class X:

    leaf_formatter = lambda self, leaf: str(leaf.type)
    node_formatter = lambda self, node, argstrings: "%s(%s)" % (node.op,
                                                                ", ".join(argstrings))

    def str(self, inputs, outputs):
        return as_string(inputs, outputs,
                         leaf_formatter = self.leaf_formatter,
                         node_formatter = self.node_formatter)
    

class TestStr(X):

    def test_as_string(self):
        r1, r2 = MyVariable(1), MyVariable(2)
        node = MyOp.make_node(r1, r2)
        s = self.str([r1, r2], node.outputs)
        assert s == ["MyOp(R1, R2)"]

    def test_as_string_deep(self):
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        node = MyOp.make_node(r1, r2)
        node2 = MyOp.make_node(node.outputs[0], r5)
        s = self.str([r1, r2, r5], node2.outputs)
        assert s == ["MyOp(MyOp(R1, R2), R5)"]

    def test_multiple_references(self):
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        node = MyOp.make_node(r1, r2)
        node2 = MyOp.make_node(node.outputs[0], node.outputs[0])
        assert self.str([r1, r2, r5], node2.outputs) == ["MyOp(*1 -> MyOp(R1, R2), *1)"]

    def test_cutoff(self):
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        node = MyOp.make_node(r1, r2)
        node2 = MyOp.make_node(node.outputs[0], node.outputs[0])
        assert self.str(node.outputs, node2.outputs) == ["MyOp(R3, R3)"]
        assert self.str(node2.inputs, node2.outputs) == ["MyOp(R3, R3)"]


#########
# clone #
#########

class TestClone(X):

    def test_accurate(self):
        r1, r2 = MyVariable(1), MyVariable(2)
        node = MyOp.make_node(r1, r2)
        _, new = clone([r1, r2], node.outputs, False)
        assert self.str([r1, r2], new) == ["MyOp(R1, R2)"]

    def test_copy(self):
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        node = MyOp.make_node(r1, r2)
        node2 = MyOp.make_node(node.outputs[0], r5)
        _, new = clone([r1, r2, r5], node2.outputs, False)
        assert node2.outputs[0].type == new[0].type and node2.outputs[0] is not new[0] # the new output is like the old one but not the same object
        assert node2 is not new[0].owner # the new output has a new owner
        assert new[0].owner.inputs[1] is r5 # the inputs are not copied
        assert new[0].owner.inputs[0].type == node.outputs[0].type and new[0].owner.inputs[0] is not node.outputs[0] # check that we copied deeper too

    def test_not_destructive(self):
        # Checks that manipulating a cloned graph leaves the original unchanged.
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        node = MyOp.make_node(MyOp.make_node(r1, r2).outputs[0], r5)
        _, new = clone([r1, r2, r5], node.outputs, False)
        new_node = new[0].owner
        new_node.inputs = MyVariable(7), MyVariable(8)
        assert self.str(inputs(new_node.outputs), new_node.outputs) == ["MyOp(R7, R8)"]
        assert self.str(inputs(node.outputs), node.outputs) == ["MyOp(MyOp(R1, R2), R5)"]


############
# toposort #
############

def prenode(obj):
    if isinstance(obj, Variable): 
        if obj.owner:
            return [obj.owner]
    if isinstance(obj, Apply):
        return obj.inputs

class TestToposort:

    def test_0(self):
        """Test a simple graph"""
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        o = MyOp.make_node(r1, r2)
        o2 = MyOp.make_node(o.outputs[0], r5)

        all = general_toposort(o2.outputs, prenode)
        assert all == [r5, r2, r1, o, o.outputs[0], o2, o2.outputs[0]]

        all = io_toposort([r5], o2.outputs)
        assert all == [o, o2]

    def test_1(self):
        """Test a graph with double dependencies"""
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        o = MyOp.make_node(r1, r1)
        o2 = MyOp.make_node(o.outputs[0], r5)
        all = general_toposort(o2.outputs, prenode)
        assert all == [r5, r1, o, o.outputs[0], o2, o2.outputs[0]]

    def test_2(self):
        """Test a graph where the inputs have owners"""
        r1, r2, r5 = MyVariable(1), MyVariable(2), MyVariable(5)
        o = MyOp.make_node(r1, r1)
        r2b = o.outputs[0]
        o2 = MyOp.make_node(r2b, r2b)
        all = io_toposort([r2b], o2.outputs)
        assert all == [o2]

        o2 = MyOp.make_node(r2b, r5)
        all = io_toposort([r2b], o2.outputs)
        assert all == [o2]

    def test_3(self):
        """Test a graph which is not connected"""
        r1, r2, r3, r4 = MyVariable(1), MyVariable(2), MyVariable(3), MyVariable(4)
        o0 = MyOp.make_node(r1, r2)
        o1 = MyOp.make_node(r3, r4)
        all = io_toposort([r1, r2, r3, r4], o0.outputs + o1.outputs)
        assert all == [o1,o0]

    def test_4(self):
        """Test inputs and outputs mixed together in a chain graph"""
        r1, r2, r3, r4 = MyVariable(1), MyVariable(2), MyVariable(3), MyVariable(4)
        o0 = MyOp.make_node(r1, r2)
        o1 = MyOp.make_node(o0.outputs[0], r1)
        all = io_toposort([r1, o0.outputs[0]], [o0.outputs[0], o1.outputs[0]])
        assert all == [o1]

    def test_5(self):
        """Test when outputs have clients"""
        r1, r2, r3, r4 = MyVariable(1), MyVariable(2), MyVariable(3), MyVariable(4)
        o0 = MyOp.make_node(r1, r2)
        o1 = MyOp.make_node(o0.outputs[0], r4)
        all = io_toposort([], o0.outputs)
        assert all == [o0]


#################
# is_same_graph #
#################

class TestIsSameGraph(unittest.TestCase):

    def check(self, expected, debug=True):
        """
        Core function to perform comparison.

        :param expected: A list of tuples (v1, v2, ((g1, o1), ..., (gN, oN)))
        with:
            - `v1` and `v2` two Variables (the graphs to be compared)
            - `gj` a `givens` dictionary to give as input to `is_same_graph`
            - `oj` the expected output of `is_same_graph(v1, v2, givens=gj)`

        :param debug: If True, then we make sure we are testing both
        implementations of `is_same_graph`.

        This function also tries to call `is_same_graph` by inverting `v1` and
        `v2`, and ensures the output remains the same.
        """
        for v1, v2, go in expected:
            for gj, oj in go:
                r1 = is_same_graph(v1, v2, givens=gj, debug=debug)
                assert r1 == oj
                r2 = is_same_graph(v2, v1, givens=gj, debug=debug)
                assert r2 == oj

    def test_single_var(self):
        """
        Test `is_same_graph` with some trivial graphs (one Variable).
        """
        x, y, z = tensor.vectors('x', 'y', 'z')
        self.check([
            (x, x, (({}, True), )),
            (x, y, (({}, False), ({y: x}, True), )),
            (x, tensor.neg(x), (({}, False), )),
            (x, tensor.neg(y), (({}, False), )),
            ])

    def test_full_graph(self):
        """
        Test `is_same_graph` with more complex graphs.
        """
        x, y, z = tensor.vectors('x', 'y', 'z')
        t = x * y
        self.check([
            (x * 2, x * 2, (({}, True), )),
            (x * 2, y * 2, (({}, False), ({y: x}, True), )),
            (x * 2, y * 2, (({}, False), ({x: y}, True), )),
            (x * 2, y * 3, (({}, False), ({y: x}, False), )),
            (t * 2, z * 2, (({}, False), ({t: z}, True), )),
            (t * 2, z * 2, (({}, False), ({z: t}, True), )),
            (x * (y * z), (x * y) * z, (({}, False), )),
            ])

    def test_merge_only(self):
        """
        Test `is_same_graph` when `equal_computations` cannot be used.
        """
        x, y, z = tensor.vectors('x', 'y', 'z')
        t = x * y
        self.check([
            (x, t, (({}, False), ({t: x}, True))),
            (t * 2, x * 2, (({}, False), ({t: x}, True), )),
            (x * x, x * y, (({}, False), ({y: x}, True), )),
            (x * x, x * y, (({}, False), ({y: x}, True), )),
            (x * x + z, x * y + t, (({}, False),
                                    ({y: x}, False),
                                    ({y: x, t: z}, True))),
            ],
            debug=False)

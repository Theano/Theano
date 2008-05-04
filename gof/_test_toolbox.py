
import unittest

from graph import Result, as_result, Apply
from type import Type
from op import Op
#from opt import PatternOptimizer, OpSubOptimizer

from env import Env, InconsistencyError
from toolbox import *


class MyType(Type):

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, MyType)


def MyResult(name):
    return Result(MyType(name), None, None)


class MyOp(Op):

    def __init__(self, nin, name):
        self.nin = nin
        self.name = name

    def make_node(self, *inputs):
        assert len(inputs) == self.nin
        inputs = map(as_result, inputs)
        for input in inputs:
            if not isinstance(input.type, MyType):
                raise Exception("Error 1")
        outputs = [MyType(self.name + "_R")()]
        return Apply(self, inputs, outputs)

    def __str__(self):
        return self.name

sigmoid = MyOp(1, 'Sigmoid')
add = MyOp(2, 'Add')
dot = MyOp(2, 'Dot')


def inputs():
    x = MyResult('x')
    y = MyResult('y')
    z = MyResult('z')
    return x, y, z


# class _test_EquivTool(unittest.TestCase):

#     def test_straightforward(self):
#         x, y, z = inputs()
#         sx = sigmoid(x)
#         e = add(sx, sigmoid(y))
#         g = Env([x, y, z], [e])
#         g.extend(EquivTool(g))
#         assert hasattr(g, 'equiv')
#         assert g.equiv(sx) is sx
#         g.replace(sx, dot(x, z))
#         assert g.equiv(sx) is not sx
#         assert g.equiv(sx).owner.op is dot


class _test_NodeFinder(unittest.TestCase):

    def test_straightforward(self):
        x, y, z = inputs()
        e0 = dot(y, z)
        e = add(add(sigmoid(x), sigmoid(sigmoid(z))), dot(add(x, y), e0))
        g = Env([x, y, z], [e])
        g.extend(NodeFinder())
        assert hasattr(g, 'get_nodes')
        for type, num in ((add, 3), (sigmoid, 3), (dot, 2)):
            if not len([x for x in g.get_nodes(type)]) == num:
                self.fail((type, num))
        new_e0 = add(y, z)
        assert e0.owner in g.get_nodes(dot)
        assert new_e0.owner not in g.get_nodes(add)
        g.replace(e0, new_e0)
        assert e0.owner not in g.get_nodes(dot)
        assert new_e0.owner in g.get_nodes(add)
        for type, num in ((add, 4), (sigmoid, 3), (dot, 1)):
            if not len([x for x in g.get_nodes(type)]) == num:
                self.fail((type, num))

    def test_robustness(self):
        x, y, z = inputs()
        e = add(add(sigmoid(x), sigmoid(sigmoid(z))), dot(add(x, y), dot(y, z)))
        g = Env([x, y, z], [e])
        g.extend(NodeFinder())
        gen = g.get_nodes(sigmoid) # I want to get Sigmoid instances
        g.replace(e, add(x, y)) # but here I prune them all
        assert len([x for x in gen]) == 0 # the generator should not yield them



if __name__ == '__main__':
    unittest.main()

    

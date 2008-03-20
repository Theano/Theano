
import unittest

from result import ResultBase
from op import Op
from opt import PatternOptimizer, OpSubOptimizer

from env import Env, InconsistencyError
from toolbox import *



class MyResult(ResultBase):

    def __init__(self, name):
        ResultBase.__init__(self, role = None, name = name)
        self.data = [1000]

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class MyOp(Op):
    nin = -1

    def __init__(self, *inputs):
        assert len(inputs) == self.nin
        for input in inputs:
            if not isinstance(input, MyResult):
                raise Exception("Error 1")
        self.inputs = inputs
        self.outputs = [MyResult(self.__class__.__name__ + "_R")]

class Sigmoid(MyOp):
    nin = 1

class Add(MyOp):
    nin = 2

class Dot(MyOp):
    nin = 2


import modes
modes.make_constructors(globals())

def inputs():
    x = modes.build(MyResult('x'))
    y = modes.build(MyResult('y'))
    z = modes.build(MyResult('z'))
    return x, y, z

def env(inputs, outputs, validate = True, features = []):
    return Env(inputs, outputs, features = features, consistency_check = validate)


class _test_EquivTool(unittest.TestCase):

    def test_straightforward(self):
        x, y, z = inputs()
        sx = sigmoid(x)
        e = add(sx, sigmoid(y))
        g = env([x, y, z], [e], features = [EquivTool])
        assert g.equiv(sx) is sx
        g.replace(sx, dot(x, z))
        assert g.equiv(sx) is not sx
        assert isinstance(g.equiv(sx).owner, Dot)


class _test_InstanceFinder(unittest.TestCase):

    def test_straightforward(self):
        x, y, z = inputs()
        e0 = dot(y, z)
        e = add(add(sigmoid(x), sigmoid(sigmoid(z))), dot(add(x, y), e0))
        g = env([x, y, z], [e], features = [InstanceFinder])
        for type, num in ((Add, 3), (Sigmoid, 3), (Dot, 2)):
            if not len([x for x in g.get_instances_of(type)]) == num:
                self.fail((type, num))
        new_e0 = add(y, z)
        assert e0.owner in g.get_instances_of(Dot)
        assert new_e0.owner not in g.get_instances_of(Add)
        g.replace(e0, new_e0)
        assert e0.owner not in g.get_instances_of(Dot)
        assert new_e0.owner in g.get_instances_of(Add)
        for type, num in ((Add, 4), (Sigmoid, 3), (Dot, 1)):
            if not len([x for x in g.get_instances_of(type)]) == num:
                self.fail((type, num))

    def test_robustness(self):
        x, y, z = inputs()
        e = add(add(sigmoid(x), sigmoid(sigmoid(z))), dot(add(x, y), dot(y, z)))
        g = env([x, y, z], [e], features = [InstanceFinder])
        gen = g.get_instances_of(Sigmoid) # I want to get Sigmoid instances
        g.replace(e, add(x, y)) # but here I prune them all
        assert len([x for x in gen]) == 0 # the generator should not yield them



if __name__ == '__main__':
    unittest.main()

    

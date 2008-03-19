
import unittest

from result import ResultBase
from op import Op
from ext import Destroyer
from opt import *
from env import Env
from toolbox import *


class MyResult(ResultBase):

    def __init__(self, name):
        ResultBase.__init__(self, role = None, name = name)
        self.data = [1000]

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def hash(self):
        return self.data



class MyOp(Op):
        
    def __init__(self, *inputs):
        for input in inputs:
            if not isinstance(input, MyResult):
                raise Exception("Error 1")
        self.inputs = inputs
        self.outputs = [MyResult(self.__class__.__name__ + "_R")]


class Op1(MyOp):
    pass

class Op2(MyOp):
    pass

class Op3(MyOp):
    pass

class Op4(MyOp):
    pass

class OpD(MyOp, Destroyer):
    def destroyed_inputs(self):
        return [self.inputs[0]]


import modes
modes.make_constructors(globals())


def inputs():
    x = modes.build(MyResult('x'))
    y = modes.build(MyResult('y'))
    z = modes.build(MyResult('z'))
    return x, y, z

def env(inputs, outputs, validate = True):
#     inputs = [input.r for input in inputs]
#     outputs = [output.r for output in outputs]
    return Env(inputs, outputs, features = [EquivTool], consistency_check = validate)


class _test_PatternOptimizer(unittest.TestCase):
    
    def test_0(self):
        x, y, z = inputs()
        e = op1(op2(x, y), z)
        g = env([x, y, z], [e])
        PatternOptimizer((Op1, (Op2, '1', '2'), '3'),
                         (Op4, '3', '2')).optimize(g)
        assert str(g) == "[Op4(z, y)]"

    def test_1(self):
        x, y, z = inputs()
        e = op1(op2(x, y), z)
        g = env([x, y, z], [e])
        PatternOptimizer((Op1, (Op2, '1', '1'), '2'),
                         (Op4, '2', '1')).optimize(g)
        assert str(g) != "[Op4(z, y)]"

    def test_2(self):
        x, y, z = inputs()
        e = op1(op2(x, y), z)
        g = env([x, y, z], [e])
        PatternOptimizer((Op2, '1', '2'),
                         (Op1, '2', '1')).optimize(g)
        assert str(g) == "[Op1(Op1(y, x), z)]"

    def test_3(self):
        x, y, z = inputs()
        e = op1(op2(x, y), op2(x, y), op2(y, z))
        g = env([x, y, z], [e])
        PatternOptimizer((Op2, '1', '2'),
                         (Op4, '1')).optimize(g)
        assert str(g) == "[Op1(Op4(x), Op4(x), Op4(y))]"

    def test_4(self):
        x, y, z = inputs()
        e = op1(op1(op1(op1(x))))
        g = env([x, y, z], [e])
        PatternOptimizer((Op1, (Op1, '1')),
                         '1').optimize(g)
        assert str(g) == "[x]"

    def test_5(self):
        x, y, z = inputs()
        e = op1(op1(op1(op1(op1(x)))))
        g = env([x, y, z], [e])
        PatternOptimizer((Op1, (Op1, '1')),
                         '1').optimize(g)
        assert str(g) == "[Op1(x)]"

    def test_6(self):
        x, y, z = inputs()
        x.constant = True
        x.value = 2
        z.constant = True
        z.value = 2
        e = op1(op1(x, y), y)
        g = env([y], [e])
        PatternOptimizer((Op1, z, '1'),
                         (Op2, '1', z)).optimize(g)
        assert str(g) == "[Op1(Op2(y, z), y)]"


class _test_OpSubOptimizer(unittest.TestCase):
    
    def test_0(self):
        x, y, z = inputs()
        e = op1(op1(op1(op1(op1(x)))))
        g = env([x, y, z], [e])
        OpSubOptimizer(Op1, Op2).optimize(g)
        assert str(g) == "[Op2(Op2(Op2(Op2(Op2(x)))))]"
    
    def test_1(self):
        x, y, z = inputs()
        e = op1(op2(x), op3(y), op4(z))
        g = env([x, y, z], [e])
        OpSubOptimizer(Op3, Op4).optimize(g)
        assert str(g) == "[Op1(Op2(x), Op4(y), Op4(z))]"


class _test_MergeOptimizer(unittest.TestCase):

    def test_0(self):
        x, y, z = inputs()
        e = op1(op2(x, y), op2(x, y), op2(x, z))
        g = env([x, y, z], [e])
        MergeOptimizer().optimize(g)
        assert str(g) == "[Op1(*1 -> Op2(x, y), *1, Op2(x, z))]"

    def test_1(self):
        x, y, z = inputs()
        y.data = 2
        y.constant = True
        z.data = 2.0
        z.constant = True
        e = op1(op2(x, y), op2(x, y), op2(x, z))
        g = env([x, y, z], [e])
        MergeOptimizer().optimize(g)
        assert str(g) == "[Op1(*1 -> Op2(x, y), *1, *1)]" \
            or str(g) == "[Op1(*1 -> Op2(x, z), *1, *1)]"


class _test_ConstantFinder(unittest.TestCase):

    def test_0(self):
        x, y, z = inputs()
        y.data = 2
        z.data = 2
        e = op1(x, y, z)
        g = env([x], [e])
        ConstantFinder().optimize(g)
        assert y.constant and z.constant
        MergeOptimizer().optimize(g)
        assert str(g) == "[Op1(x, y, y)]" \
            or str(g) == "[Op1(x, z, z)]"

    def test_1(self):
        x, y, z = inputs()
        y.data = 2
        z.data = 2
        e = op1(op2(x, y), op2(x, y), op2(x, z))
        g = env([x], [e])
        ConstantFinder().optimize(g)
        assert y.constant and z.constant
        MergeOptimizer().optimize(g)
        assert str(g) == "[Op1(*1 -> Op2(x, y), *1, *1)]" \
            or str(g) == "[Op1(*1 -> Op2(x, z), *1, *1)]"

    def test_2(self):
        x, y, z = inputs()
        y.data = 2
        z.data = 2
        e = op_d(x, op2(y, z))
        g = env([y], [e])
        ConstantFinder().optimize(g)
        assert not getattr(x, 'constant', False) and z.constant
        MergeOptimizer().optimize(g)



if __name__ == '__main__':
    unittest.main()




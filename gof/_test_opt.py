
import unittest

from result import ResultBase
from op import Op
from opt import *
from env import Env
from toolbox import *


class MyResult(ResultBase):

    def __init__(self, name):
        ResultBase.__init__(self, role = None, data = [1000], name = name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name



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


from constructor import Constructor
from allocators import BuildAllocator
c = Constructor(BuildAllocator)
c.update(globals())
for k, v in c.items():
    globals()[k.lower()] = v


def inputs():
    x = MyResult('x')
    y = MyResult('y')
    z = MyResult('z')
    return x, y, z

def env(inputs, outputs, validate = True):
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


if __name__ == '__main__':
    unittest.main()




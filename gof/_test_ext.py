
import unittest

from result import ResultBase
from op import Op
from opt import PatternOptimizer, OpSubOptimizer

from ext import *
from env import Env, InconsistencyError
from toolbox import EquivTool

from _test_result import MyResult

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

class TransposeView(MyOp, Viewer):
    nin = 1
    def view_map(self):
        return {self.outputs[0]: [self.inputs[0]]}

class Add(MyOp):
    nin = 2

class AddInPlace(MyOp, Destroyer):
    nin = 2
    def destroyed_inputs(self):
        return self.inputs[:1]

class Dot(MyOp):
    nin = 2


dtv_elim = PatternOptimizer((TransposeView, (TransposeView, 'x')), 'x')

a2i = OpSubOptimizer(Add, AddInPlace)
i2a = OpSubOptimizer(AddInPlace, Add)

t2s = OpSubOptimizer(TransposeView, Sigmoid)
s2t = OpSubOptimizer(Sigmoid, TransposeView)


# from constructor import Constructor
# from allocators import BuildAllocator
# c = Constructor(BuildAllocator)
# c.update(globals())
# globals().update(c)

# def inputs():
#     x = (MyResult('x'))
#     y = (MyResult('y'))
#     z = (MyResult('z'))
#     return x, y, z

# def env(inputs, outputs, validate = True):
#     return Env(inputs, outputs, features = [EquivTool], consistency_check = validate)


import modes
modes.make_constructors(globals(), name_filter = lambda x:x)

# def inputs():
#     x = modes.BuildMode(MyResult('x'))
#     y = modes.BuildMode(MyResult('y'))
#     z = modes.BuildMode(MyResult('z'))
#     return x, y, z

def inputs():
    x = modes.build(MyResult('x'))
    y = modes.build(MyResult('y'))
    z = modes.build(MyResult('z'))
    return x, y, z

def env(inputs, outputs, validate = True):
    inputs = [input for input in inputs]
    outputs = [output for output in outputs]
    return Env(inputs, outputs, features = [EquivTool], consistency_check = validate)


class _test_all(unittest.TestCase):

    def test_0(self):
        x, y, z = inputs()
        e = Add(AddInPlace(x, y), AddInPlace(x, y))
        try:
            g = env([x,y,z], [e])
        except InconsistencyError, e:
            pass
        else:
            raise Exception("Expected an InconsistencyError")

    def test_1(self):
        # the loop is needed because a2i will optimize in a random order and sometimes
        # only one of them fails
        for i in xrange(100):
            x, y, z = inputs()
            e = Add(Add(x, y), Add(y, x))
            g = env([x,y,z], [e])
            assert g.consistent()
            a2i.optimize(g)
            assert g.consistent()
            assert str(g) != "[AddInPlace(AddInPlace(x, y), AddInPlace(y, x))]"

    def test_2(self):
        x, y, z = inputs()
        g = env([x,y,z], [Dot(AddInPlace(x, z), x)], False)
        assert not g.consistent()
        i2a.optimize(g)
        assert g.consistent()
            
    def test_3(self):
        for i in xrange(100):
            x, y, z = inputs()
            e = Dot(Add(TransposeView(z), y), Add(z, x))
            g = env([x,y,z], [e])
            assert g.consistent()
            a2i.optimize(g)
            assert g.consistent()
            assert str(g) != "[Dot(AddInPlace(TransposeView(z), y), AddInPlace(z, x))]"

    def test_4(self):
        x, y, z = inputs()
        e = Dot(AddInPlace(x,y), TransposeView(x))
        g = env([x,y,z], [e], False)
        assert not g.consistent()
        g.replace(e.owner.inputs[1], Add(x,z))
        assert g.consistent()

    def test_5(self):
        x, y, z = inputs()
        e = Dot(AddInPlace(x,y), TransposeView(TransposeView(TransposeView(TransposeView(Sigmoid(x))))))
        g = env([x,y,z], [e])
        assert g.consistent()
        g.replace(e.owner.inputs[1].owner.inputs[0], x, False)
        assert not g.consistent()

    def test_6(self):
        for i in xrange(100):
            x, y, z = inputs()
            e = Dot(AddInPlace(x,Sigmoid(y)), Sigmoid(Sigmoid(Sigmoid(Sigmoid(Sigmoid(x))))))
            g = env([x,y,z], [e])
            assert g.consistent()
            s2t.optimize(g)
            assert g.consistent()
            assert str(g) != "[Dot(AddInPlace(x,TransposeView(y)), TransposeView(TransposeView(TransposeView(TransposeView(TransposeView(x))))))]"

    def test_7(self):
        x, y, z = inputs()
        e = TransposeView(TransposeView(TransposeView(TransposeView(x))))
        g = env([x,y,z], [e])
        assert g.consistent()
        chk = g.checkpoint()
        dtv_elim.optimize(g)
        assert str(g) == "[x]"
        g.replace(g.equiv(e), Add(x,y))
        assert str(g) == "[Add(x, y)]"
        g.replace(g.equiv(e), Dot(AddInPlace(x,y), TransposeView(x)), False)
        assert str(g) == "[Dot(AddInPlace(x, y), TransposeView(x))]"
        assert not g.consistent()
        g.revert(chk)
        assert g.consistent()
        assert str(g) == "[TransposeView(TransposeView(TransposeView(TransposeView(x))))]"

    def test_8(self):
        x, y, z = inputs()
        e = Dot(Dot(AddInPlace(x,y), AddInPlace(y,z)), Add(z,x))
        g = env([x,y,z], [e])
        assert g.consistent()
        a2i.optimize(g)
        assert g.consistent()
        assert str(g) != "[Dot(Dot(AddInPlace(x, y), AddInPlace(y, z)), AddInPlace(z, x))]" # we don't want to see that!

    def test_9(self):
        x, y, z = inputs()
        x.indestructible = True
        e = AddInPlace(x, y)
        g = env([x,y,z], [e], False)
        assert not g.consistent()
        g.replace(e, Add(x, y))
        assert g.consistent()

    def test_10(self):
        x, y, z = inputs()
        x.indestructible = True
        tv = TransposeView(x)
        e = AddInPlace(tv, y)
        g = env([x,y,z], [e], False)
        assert not g.consistent()
        g.replace(tv, Sigmoid(x))
        assert g.consistent()

    def test_11(self):
        x, y, z = inputs()
        e1 = TransposeView(TransposeView(x))
        e2 = TransposeView(TransposeView(e1))
        e3 = AddInPlace(e2, y)
        e4 = AddInPlace(e1, z)
        g = env([x,y,z], [e3, e4], False)
        assert not g.consistent()
        g.replace(e2, TransposeView(x), False)
        assert not g.consistent()

    def test_12(self):
        x, y, z = inputs()
        e0 = AddInPlace(x, y)
        e = Dot(Sigmoid(e0), TransposeView(x))
        g = env([x,y,z], [e], False)
        assert not g.consistent()
        new_e0 = Add(x, y)
        g.replace(e0, new_e0, False)
        assert g.consistent()
        g.replace(new_e0, AddInPlace(x, y), False)
        assert not g.consistent()

    def test_13(self):
        x, y, z = inputs()
        e0 = TransposeView(x)
        e = Dot(Sigmoid(AddInPlace(x, y)), e0)
        g = env([x,y,z], [e], False)
        assert not g.consistent()
        new_e0 = Add(e0, y)
        g.replace(e0, new_e0, False)
        assert g.consistent()


if __name__ == '__main__':
    unittest.main()




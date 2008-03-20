
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


# dtv_elim = PatternOptimizer((TransposeView, (TransposeView, 'x')), 'x')

# AddCls = Add
# AddInPlaceCls = AddInPlace

# a2i = OpSubOptimizer(Add, AddInPlace)
# i2a = OpSubOptimizer(AddInPlace, Add)

# t2s = OpSubOptimizer(TransposeView, Sigmoid)
# s2t = OpSubOptimizer(Sigmoid, TransposeView)


import modes
modes.make_constructors(globals()) #, name_filter = lambda x:x)

def inputs():
    x = modes.build(MyResult('x'))
    y = modes.build(MyResult('y'))
    z = modes.build(MyResult('z'))
    return x, y, z

def env(inputs, outputs, validate = True):
    return Env(inputs, outputs, features = [EquivTool], consistency_check = validate)


class FailureWatch:
    # when passed to OpSubOptimizer or PatternOptimizer, counts the number of failures
    def __init__(self):
        self.failures = 0
    def __call__(self, op1, op2, exception):
        assert isinstance(exception, InconsistencyError)
        self.failures += 1


class _test_all(unittest.TestCase):

    def test_multi_destroyers(self):
        x, y, z = inputs()
        e = add(add_in_place(x, y), add_in_place(x, y))
        try:
            g = env([x,y,z], [e])
            self.fail()
        except InconsistencyError, e:
            pass

    def test_multi_destroyers_through_views(self):
        x, y, z = inputs()
        e = dot(add(transpose_view(z), y), add(z, x))
        g = env([x,y,z], [e])
        assert g.consistent()
        fail = FailureWatch()
        OpSubOptimizer(Add, AddInPlace, fail).optimize(g)
        assert g.consistent()
        assert fail.failures == 1 # should have succeeded once and failed once

    def test_destroyers_loop(self):
        # AddInPlace(x, y) and AddInPlace(y, x) should not coexist
        x, y, z = inputs()
        e1 = add(x, y)
        e2 = add(y, x)
        g = env([x,y,z], [e1, e2])
        chk = g.checkpoint()
        assert g.consistent()
        g.replace(e1, add_in_place(x, y))
        assert g.consistent()
        try:
            g.replace(e2, add_in_place(y, x))
            self.fail()
        except InconsistencyError:
            pass
        assert g.consistent()
        g.revert(chk)
        g.replace(e2, add_in_place(y, x))
        assert g.consistent()
        try:
            g.replace(e1, add_in_place(x, y))
            self.fail()
        except InconsistencyError:
            pass
        assert g.consistent()

    def test_long_destroyers_loop(self):
        x, y, z = inputs()
        e = dot(dot(add_in_place(x,y), add_in_place(y,z)), add(z,x))
        g = env([x,y,z], [e])
        assert g.consistent()
        OpSubOptimizer(Add, AddInPlace).optimize(g)
        assert g.consistent()
        assert str(g) != "[Dot(Dot(AddInPlace(x, y), AddInPlace(y, z)), AddInPlace(z, x))]" # we don't want to see that!
        e2 = dot(dot(add_in_place(x,y), add_in_place(y,z)), add_in_place(z,x))
        try:
            g2 = env([x,y,z], [e2])
            self.fail()
        except InconsistencyError:
            pass

    def test_usage_loop(self):
        x, y, z = inputs()
        g = env([x,y,z], [dot(add_in_place(x, z), x)], False)
        assert not g.consistent()
        OpSubOptimizer(AddInPlace, Add).optimize(g) # replace AddInPlace with Add
        assert g.consistent()

    def test_usage_loop_through_views(self):
        x, y, z = inputs()
        aip = add_in_place(x, y)
        e = dot(aip, transpose_view(x))
        g = env([x,y,z], [e], False)
        assert not g.consistent()
        g.replace(aip, add(x, z))
        assert g.consistent()

    def test_usage_loop_through_views_2(self):
        x, y, z = inputs()
        e0 = transpose_view(transpose_view(transpose_view(sigmoid(x))))
        e = dot(add_in_place(x,y), transpose_view(e0))
        g = env([x,y,z], [e])
        assert g.consistent() # because sigmoid can do the copy
        g.replace(e0, x, False)
        assert not g.consistent() # we cut off the path to the sigmoid

    def test_usage_loop_insert_views(self):
        x, y, z = inputs()
        e = dot(add_in_place(x, add(y, z)), sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(x))))))
        g = env([x,y,z], [e])
        assert g.consistent()
        fail = FailureWatch()
        OpSubOptimizer(Sigmoid, TransposeView, fail).optimize(g)
        assert g.consistent()
        assert fail.failures == 1 # it must keep one sigmoid in the long sigmoid chain

    def test_misc(self):
        x, y, z = inputs()
        e = transpose_view(transpose_view(transpose_view(transpose_view(x))))
        g = env([x,y,z], [e])
        assert g.consistent()
        chk = g.checkpoint()
        PatternOptimizer((TransposeView, (TransposeView, 'x')), 'x').optimize(g)
        assert str(g) == "[x]"
        g.replace(g.equiv(e), add(x,y))
        assert str(g) == "[Add(x, y)]"
        g.replace(g.equiv(e), dot(add_in_place(x,y), transpose_view(x)), False)
        assert str(g) == "[Dot(AddInPlace(x, y), TransposeView(x))]"
        assert not g.consistent()
        g.revert(chk)
        assert g.consistent()
        assert str(g) == "[TransposeView(TransposeView(TransposeView(TransposeView(x))))]"

    def test_indestructible(self):
        x, y, z = inputs()
        x.indestructible = True
        e = add_in_place(x, y)
        g = env([x,y,z], [e], False)
        assert not g.consistent()
        g.replace(e, add(x, y))
        assert g.consistent()

    def test_indestructible_through_views(self):
        x, y, z = inputs()
        x.indestructible = True
        tv = transpose_view(x)
        e = add_in_place(tv, y)
        g = env([x,y,z], [e], False)
        assert not g.consistent()
        g.replace(tv, sigmoid(x))
        assert g.consistent()

    def test_repair_destroy_path(self):
        x, y, z = inputs()
        e1 = transpose_view(transpose_view(x))
        e2 = transpose_view(transpose_view(e1))
        e3 = add_in_place(e2, y)
        e4 = add_in_place(e1, z)
        g = env([x,y,z], [e3, e4], False)
        assert not g.consistent()
        g.replace(e2, transpose_view(x), False)
        assert not g.consistent()

    def test_indirect(self):
        x, y, z = inputs()
        e0 = add_in_place(x, y)
        e = dot(sigmoid(e0), transpose_view(x))
        g = env([x,y,z], [e], False)
        assert not g.consistent()
        new_e0 = add(x, y)
        g.replace(e0, new_e0, False)
        assert g.consistent()
        g.replace(new_e0, add_in_place(x, y), False)
        assert not g.consistent()

    def test_indirect_2(self):
        x, y, z = inputs()
        e0 = transpose_view(x)
        e = dot(sigmoid(add_in_place(x, y)), e0)
        g = env([x,y,z], [e], False)
        assert not g.consistent()
        new_e0 = add(e0, y)
        g.replace(e0, new_e0, False)
        assert g.consistent()


if __name__ == '__main__':
    unittest.main()




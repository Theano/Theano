
import unittest

from theano.gof.type import Type
from theano.gof import graph
from theano.gof.graph import Result, Apply
from theano.gof.op import Op
from theano.gof.opt import *

from theano.gof import destroyhandler
from theano.gof.env import Env, InconsistencyError
from theano.gof.toolbox import ReplaceValidate

from copy import copy

PatternOptimizer = lambda p1, p2, ign=True: OpKeyOptimizer(PatternSub(p1, p2), ignore_newtrees=ign)
OpSubOptimizer = lambda op1, op2, fail=keep_going, ign=True: TopoOptimizer(OpSub(op1, op2), ignore_newtrees=ign, failure_callback = fail)


def as_result(x):
    assert isinstance(x, Result)
    return x


class MyType(Type):

    def filter(self, data):
        return data

    def __eq__(self, other):
        return isinstance(other, MyType)


def MyResult(name):
    return Result(MyType(), None, None, name = name)

def MyValue(data):
    return graph.Value(MyType(), data = data)


class MyOp(Op):

    def __init__(self, nin, name, vmap = {}, dmap = {}, nout = 1, tolerate_same = []):
        self.nin = nin
        self.nout = nout
        self.name = name
        self.destroy_map = dmap
        self.view_map = vmap
        self.tolerate_same = tolerate_same
    
    def make_node(self, *inputs):
        assert len(inputs) == self.nin
        inputs = map(as_result, inputs)
        for input in inputs:
            if not isinstance(input.type, MyType):
                raise Exception("Error 1")
        outputs = [MyResult(self.name + "_R") for i in xrange(self.nout)]
        return Apply(self, inputs, outputs)

    def __str__(self):
        return self.name


sigmoid = MyOp(1, 'Sigmoid')
transpose_view = MyOp(1, 'TransposeView', vmap = {0: [0]})
add = MyOp(2, 'Add')
add_in_place = MyOp(2, 'AddInPlace', dmap = {0: [0]})
add_in_place_2 = MyOp(2, 'AddInPlace', dmap = {0: [0]}, tolerate_same = [(0, 1)])
dot = MyOp(2, 'Dot')


def inputs():
    x = MyResult('x')
    y = MyResult('y')
    z = MyResult('z')
    return x, y, z

_Env = Env
def Env(inputs, outputs, validate = True):
    e = _Env(inputs, outputs)
    ##e.extend(EquivTool(e))

    e.extend(destroyhandler.DestroyHandler())

    e.extend(ReplaceValidate())
    if validate:
        e.validate()
    return e


class FailureWatch:
    # when passed to OpSubOptimizer or PatternOptimizer, counts the number of failures
    def __init__(self):
        self.failures = 0
    def __call__(self, exc, nav, pairs):
        assert isinstance(exc, InconsistencyError)
        self.failures += 1


class _test_protocol(unittest.TestCase):
    def test_misc(self):
        x, y, z = inputs()
        e = transpose_view(transpose_view(transpose_view(transpose_view(x))))
        g = Env([x,y,z], [e])
        assert g.consistent()
        chk = g.checkpoint()
        PatternOptimizer((transpose_view, (transpose_view, 'x')), 'x').optimize(g)
        assert str(g) == "[x]"
        new_e = add(x,y)
        g.replace_validate(x, new_e)
        assert str(g) == "[Add(x, y)]"
        g.replace(new_e, dot(add_in_place(x,y), transpose_view(x)))
        assert str(g) == "[Dot(AddInPlace(x, y), TransposeView(x))]"
        assert not g.consistent()
        g.revert(chk)
        assert g.consistent()
        assert str(g) == "[TransposeView(TransposeView(TransposeView(TransposeView(x))))]"


class _test_protocol_skip(unittest.TestCase):


    def test_aliased_inputs_replacement(self):
        x, y, z = inputs()
        tv = transpose_view(x)
        tvv = transpose_view(tv)
        sx = sigmoid(x)
        e = add_in_place(x, tv)
        g = Env([x, y], [e], False)
        assert not g.consistent()
        g.replace(tv, sx)
        assert g.consistent()
        g.replace(sx, tv)
        assert not g.consistent()
        g.replace(tv, tvv)
        assert not g.consistent()
        g.replace(tv, sx)
        assert g.consistent()
    def test_indestructible(self):
        x, y, z = inputs()
        x.tag.indestructible = True
        x = copy(x)
        assert x.tag.indestructible  # checking if indestructible survives the copy!
        e = add_in_place(x, y)
        g = Env([x,y,z], [e], False)
        assert not g.consistent()
        g.replace_validate(e, add(x, y))
        assert g.consistent()
    def test_usage_loop_through_views_2(self):
        x, y, z = inputs()
        e0 = transpose_view(transpose_view(sigmoid(x)))
        e = dot(add_in_place(x,y), transpose_view(e0))
        g = Env([x,y,z], [e])
        assert g.consistent() # because sigmoid can do the copy
#         print g
#         print g.destroy_handler.children
        g.replace(e0, x)
        assert not g.consistent() # we cut off the path to the sigmoid

    def test_destroyers_loop(self):
        # AddInPlace(x, y) and AddInPlace(y, x) should not coexist
        x, y, z = inputs()
        e1 = add(x, y)
        e2 = add(y, x)
        g = Env([x,y,z], [e1, e2])
        chk = g.checkpoint()
        assert g.consistent()
        g.replace_validate(e1, add_in_place(x, y))
        assert g.consistent()
        try:
            g.replace_validate(e2, add_in_place(y, x))
            self.fail()
        except InconsistencyError:
            pass
        assert g.consistent()
        g.revert(chk)
        g.replace_validate(e2, add_in_place(y, x))
        assert g.consistent()
        try:
            g.replace_validate(e1, add_in_place(x, y))
            self.fail()
        except InconsistencyError:
            pass
        assert g.consistent()


class _test_0(unittest.TestCase):

    def test_aliased_inputs(self):
        x, y, z = inputs()
        e = add_in_place(x, x)
        g = Env([x], [e], False)
        assert not g.consistent()

    def test_aliased_inputs2(self):
        x, y, z = inputs()
        e = add_in_place(x, transpose_view(x))
        g = Env([x], [e], False)
        assert not g.consistent()

    def test_aliased_inputs_tolerate(self):
        x, y, z = inputs()
        e = add_in_place_2(x, x)
        g = Env([x], [e], False)
        assert g.consistent()

    def test_aliased_inputs_tolerate2(self):
        x, y, z = inputs()
        e = add_in_place_2(x, transpose_view(x))
        g = Env([x], [e], False)
        assert not g.consistent()

    def test_indestructible_through_views(self):
        x, y, z = inputs()
        x.tag.indestructible = True
        tv = transpose_view(x)
        e = add_in_place(tv, y)
        g = Env([x,y,z], [e], False)
        assert not g.consistent()
        g.replace_validate(tv, sigmoid(x))
        assert g.consistent()

    def test_indirect(self):
        x, y, z = inputs()
        e0 = add_in_place(x, y)
        e = dot(sigmoid(e0), transpose_view(x))
        g = Env([x,y,z], [e], False)
        assert not g.consistent()
        new_e0 = add(x, y)
        g.replace(e0, new_e0)
        assert g.consistent()
        g.replace(new_e0, add_in_place(x, y))
        assert not g.consistent()

    def test_indirect_2(self):
        x, y, z = inputs()
        e0 = transpose_view(x)
        e = dot(sigmoid(add_in_place(x, y)), e0)
        g = Env([x,y,z], [e], False)
        assert not g.consistent()
        new_e0 = add(e0, y)
        g.replace(e0, new_e0)
        assert g.consistent()

    def test_long_destroyers_loop(self):
        x, y, z = inputs()
        e = dot(dot(add_in_place(x,y), add_in_place(y,z)), add(z,x))
        g = Env([x,y,z], [e])
        assert g.consistent()
        OpSubOptimizer(add, add_in_place).optimize(g)
        assert g.consistent()
        assert str(g) != "[Dot(Dot(AddInPlace(x, y), AddInPlace(y, z)), AddInPlace(z, x))]" # we don't want to see that!
        e2 = dot(dot(add_in_place(x,y), add_in_place(y,z)), add_in_place(z,x))
        try:
            g2 = Env(*graph.clone([x,y,z], [e2]))
            self.fail()
        except InconsistencyError:
            pass

    def test_misc_2(self):
        x, y, z = inputs()
        tv = transpose_view(x)
        e = add_in_place(x, tv)
        g = Env([x,y], [e], False)
        assert not g.consistent()
        g.replace(tv, x)
        assert not g.consistent()

    def test_multi_destroyers(self):
        x, y, z = inputs()
        e = add(add_in_place(x, y), add_in_place(x, y))
        try:
            g = Env([x,y,z], [e])
            self.fail()
        except InconsistencyError, e:
            pass

    def test_multi_destroyers_through_views(self):
        x, y, z = inputs()
        e = dot(add(transpose_view(z), y), add(z, x))
        g = Env([x,y,z], [e])
        assert g.consistent()
        fail = FailureWatch()
        OpSubOptimizer(add, add_in_place, fail).optimize(g)
        assert g.consistent()
        assert fail.failures == 1 # should have succeeded once and failed once


    def test_repair_destroy_path(self):
        x, y, z = inputs()
        e1 = transpose_view(transpose_view(x))
        e2 = transpose_view(transpose_view(e1))
        e3 = add_in_place(e2, y)
        e4 = add_in_place(e1, z)
        g = Env([x,y,z], [e3, e4], False)
        assert not g.consistent()
        g.replace(e2, transpose_view(x))
        assert not g.consistent()

    def test_usage_loop(self):
        x, y, z = inputs()
        g = Env([x,y,z], [dot(add_in_place(x, z), x)], False)
        assert not g.consistent()
        OpSubOptimizer(add_in_place, add).optimize(g) # replace add_in_place with add
        assert g.consistent()

    def test_usage_loop_through_views(self):
        x, y, z = inputs()
        aip = add_in_place(x, y)
        e = dot(aip, transpose_view(x))
        g = Env([x,y,z], [e], False)
        assert not g.consistent()
        g.replace_validate(aip, add(x, z))
        assert g.consistent()

    def test_usage_loop_insert_views(self):
        x, y, z = inputs()
        e = dot(add_in_place(x, add(y, z)), sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(x))))))
        g = Env([x,y,z], [e])
        assert g.consistent()
        fail = FailureWatch()
        OpSubOptimizer(sigmoid, transpose_view, fail).optimize(g)
        assert g.consistent()
        assert fail.failures == 1 # it must keep one sigmoid in the long sigmoid chain

    def test_value_repl(self):
        x, y, z = inputs()
        sy = sigmoid(y)
        e = add_in_place(x, sy)
        g = Env([x,y], [e], False)
        assert g.consistent()
        g.replace(sy, MyValue("abc"))
        assert g.consistent()

    def test_value_repl_2(self):
        x, y, z = inputs()
        sy = sigmoid(y)
        e = add_in_place(x, sy)
        g = Env([x,y], [e], False)
        assert g.consistent()
        g.replace(sy, transpose_view(MyValue("abc")))
        assert g.consistent()

    pass

if __name__ == '__main__':
    if 1:
        unittest.main()
    else:
        testcase = _test_protocol

        suite = unittest.TestLoader()
        suite = suite.loadTestsFromTestCase(testcase)
        unittest.TextTestRunner(verbosity=2).run(suite)




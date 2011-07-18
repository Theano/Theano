
import unittest

from theano.gof.type import Type
from theano.gof import graph
from theano.gof.graph import Variable, Apply
from theano.gof.op import Op
from theano.gof.opt import *

from theano.gof import destroyhandler
from theano.gof.env import Env, InconsistencyError
from theano.gof.toolbox import ReplaceValidate

from copy import copy

PatternOptimizer = lambda p1, p2, ign=True: OpKeyOptimizer(PatternSub(p1, p2), ignore_newtrees=ign)
OpSubOptimizer = lambda op1, op2, fail=NavigatorOptimizer.warn_ignore, ign=True: TopoOptimizer(OpSub(op1, op2), ignore_newtrees=ign, failure_callback = fail)


def as_variable(x):
    assert isinstance(x, Variable)
    return x


class MyType(Type):

    def filter(self, data):
        return data

    def __eq__(self, other):
        return isinstance(other, MyType)


def MyVariable(name):
    return Variable(MyType(), None, None, name = name)

def MyValue(data):
    return graph.Value(MyType(), data = data)


class MyOp(Op):

    def __init__(self, nin, name, vmap = {}, dmap = {}, nout = 1,
            destroyhandler_tolerate_same = [],
            destroyhandler_tolerate_aliased = []):
        self.nin = nin
        self.nout = nout
        self.name = name
        self.destroy_map = dmap
        self.view_map = vmap
        self.destroyhandler_tolerate_same = destroyhandler_tolerate_same
        self.destroyhandler_tolerate_aliased = destroyhandler_tolerate_aliased
    
    def make_node(self, *inputs):
        assert len(inputs) == self.nin
        inputs = map(as_variable, inputs)
        for input in inputs:
            if not isinstance(input.type, MyType):
                raise Exception("Error 1")
        outputs = [MyVariable(self.name + "_R") for i in xrange(self.nout)]
        return Apply(self, inputs, outputs)

    def __str__(self):
        return self.name


sigmoid = MyOp(1, 'Sigmoid')
transpose_view = MyOp(1, 'TransposeView', vmap = {0: [0]})
add = MyOp(2, 'Add')
add_in_place = MyOp(2, 'AddInPlace', dmap = {0: [0]})
add_in_place_2 = MyOp(2, 'AddInPlace', dmap = {0: [0]},
        destroyhandler_tolerate_same = [(0, 1)])
add_in_place_3 = MyOp(2, 'AddInPlace', dmap = {0: [0]},
        destroyhandler_tolerate_aliased = [(0, 1)])
dot = MyOp(2, 'Dot')


def inputs():
    x = MyVariable('x')
    y = MyVariable('y')
    z = MyVariable('z')
    return x, y, z

_Env = Env
def Env(inputs, outputs, validate = True):
    e = _Env(inputs, outputs)
    e.extend(destroyhandler.DestroyHandler())
    e.extend(ReplaceValidate())
    if validate:
        e.validate()
    return e


class FailureWatch:
    # when passed to OpSubOptimizer or PatternOptimizer, counts the number of failures
    def __init__(self):
        self.failures = 0
    def __call__(self, exc, nav, pairs, lopt):
        assert isinstance(exc, InconsistencyError)
        self.failures += 1


def consistent(g):
    print "Testing consistent:", g
    try:
        assert g.consistent()
    except AssertionError:
        print "Test failed! The graph was marked as NOT consistent."
        raise
    print "Test OK"

def inconsistent(g):
    print "Testing NOT consistent:", g
    try:
        assert not g.consistent()
    except AssertionError:
        print "Test failed! The graph was marked as consistent."
        raise
    print "Test OK"



#################
# Test protocol #
#################

def test_misc():
    x, y, z = inputs()
    e = transpose_view(transpose_view(transpose_view(transpose_view(x))))
    g = Env([x,y,z], [e])
    consistent(g)
    chk = g.checkpoint()
    PatternOptimizer((transpose_view, (transpose_view, 'x')), 'x').optimize(g)
    assert str(g) == "[x]"
    new_e = add(x,y)
    g.replace_validate(x, new_e)
    assert str(g) == "[Add(x, y)]"
    g.replace(new_e, dot(add_in_place(x,y), transpose_view(x)))
    assert str(g) == "[Dot(AddInPlace(x, y), TransposeView(x))]"
    inconsistent(g)
    g.revert(chk)
    consistent(g)
    assert str(g) == "[TransposeView(TransposeView(TransposeView(TransposeView(x))))]"




######################
# Test protocol skip #
######################

def test_aliased_inputs_replacement():
    x, y, z = inputs()
    tv = transpose_view(x)
    tvv = transpose_view(tv)
    sx = sigmoid(x)
    e = add_in_place(x, tv)
    g = Env([x, y], [e], False)
    inconsistent(g)
    g.replace(tv, sx)
    consistent(g)
    g.replace(sx, tv)
    inconsistent(g)
    g.replace(tv, tvv)
    inconsistent(g)
    g.replace(tv, sx)
    consistent(g)

def test_indestructible():
    x, y, z = inputs()
    x.tag.indestructible = True
    x = copy(x)
    assert x.tag.indestructible  # checking if indestructible survives the copy!
    e = add_in_place(x, y)
    g = Env([x,y,z], [e], False)
    inconsistent(g)
    g.replace_validate(e, add(x, y))
    consistent(g)

def test_usage_loop_through_views_2():
    x, y, z = inputs()
    e0 = transpose_view(transpose_view(sigmoid(x)))
    e = dot(add_in_place(x,y), transpose_view(e0))
    g = Env([x,y,z], [e])
    consistent(g) # because sigmoid can do the copy
    g.replace(e0, x)
    inconsistent(g) # we cut off the path to the sigmoid

def test_destroyers_loop():
    # AddInPlace(x, y) and AddInPlace(y, x) should not coexist
    x, y, z = inputs()
    e1 = add(x, y)
    e2 = add(y, x)
    g = Env([x,y,z], [e1, e2])
    chk = g.checkpoint()
    consistent(g)
    g.replace_validate(e1, add_in_place(x, y))
    consistent(g)
    try:
        g.replace_validate(e2, add_in_place(y, x))
        raise Exception("Shouldn't have reached this point.")
    except InconsistencyError:
        pass
    consistent(g)
    g.revert(chk)
    g.replace_validate(e2, add_in_place(y, x))
    consistent(g)
    try:
        g.replace_validate(e1, add_in_place(x, y))
        raise Exception("Shouldn't have reached this point.")
    except InconsistencyError:
        pass
    consistent(g)


########
# Misc #
########

def test_aliased_inputs():
    x, y, z = inputs()
    e = add_in_place(x, x)
    g = Env([x], [e], False)
    inconsistent(g)

def test_aliased_inputs2():
    x, y, z = inputs()
    e = add_in_place(x, transpose_view(x))
    g = Env([x], [e], False)
    inconsistent(g)

def test_aliased_inputs_tolerate():
    x, y, z = inputs()
    e = add_in_place_2(x, x)
    g = Env([x], [e], False)
    consistent(g)

def test_aliased_inputs_tolerate2():
    x, y, z = inputs()
    e = add_in_place_2(x, transpose_view(x))
    g = Env([x], [e], False)
    inconsistent(g)

def test_same_aliased_inputs_ignored():
    x, y, z = inputs()
    e = add_in_place_3(x, x)
    g = Env([x], [e], False)
    consistent(g)

def test_different_aliased_inputs_ignored():
    x, y, z = inputs()
    e = add_in_place_3(x, transpose_view(x))
    g = Env([x], [e], False)
    consistent(g)
    # warning - don't run this because it would produce the wrong answer
    # add_in_place_3 is actually not correct when aliasing of inputs
    # is ignored.


def test_indestructible_through_views():
    x, y, z = inputs()
    x.tag.indestructible = True
    tv = transpose_view(x)
    e = add_in_place(tv, y)
    g = Env([x,y,z], [e], False)
    inconsistent(g)
    g.replace_validate(tv, sigmoid(x))
    consistent(g)

def test_indirect():
    x, y, z = inputs()
    e0 = add_in_place(x, y)
    e = dot(sigmoid(e0), transpose_view(x))
    g = Env([x,y,z], [e], False)
    inconsistent(g)
    new_e0 = add(x, y)
    g.replace(e0, new_e0)
    consistent(g)
    g.replace(new_e0, add_in_place(x, y))
    inconsistent(g)

def test_indirect_2():
    x, y, z = inputs()
    e0 = transpose_view(x)
    e = dot(sigmoid(add_in_place(x, y)), e0)
    g = Env([x,y,z], [e], False)
    inconsistent(g)
    new_e0 = add(e0, y)
    g.replace(e0, new_e0)
    consistent(g)

def test_long_destroyers_loop():
    x, y, z = inputs()
    e = dot(dot(add_in_place(x,y), add_in_place(y,z)), add(z,x))
    g = Env([x,y,z], [e])
    consistent(g)
    OpSubOptimizer(add, add_in_place).optimize(g)
    consistent(g)
    assert str(g) != "[Dot(Dot(AddInPlace(x, y), AddInPlace(y, z)), AddInPlace(z, x))]" # we don't want to see that!
    e2 = dot(dot(add_in_place(x,y), add_in_place(y,z)), add_in_place(z,x))
    try:
        g2 = Env(*graph.clone([x,y,z], [e2]))
        raise Exception("Shouldn't have reached this point.")
    except InconsistencyError:
        pass

def test_misc_2():
    x, y, z = inputs()
    tv = transpose_view(x)
    e = add_in_place(x, tv)
    g = Env([x,y], [e], False)
    inconsistent(g)
    g.replace(tv, x)
    inconsistent(g)

def test_multi_destroyers():
    x, y, z = inputs()
    e = add(add_in_place(x, y), add_in_place(x, y))
    try:
        g = Env([x,y,z], [e])
        raise Exception("Shouldn't have reached this point.")
    except InconsistencyError, e:
        pass

def test_multi_destroyers_through_views():
    x, y, z = inputs()
    e = dot(add(transpose_view(z), y), add(z, x))
    g = Env([x,y,z], [e])
    consistent(g)
    fail = FailureWatch()
    OpSubOptimizer(add, add_in_place, fail).optimize(g)
    consistent(g)
    assert fail.failures == 1 # should have succeeded once and failed once


def test_repair_destroy_path():
    x, y, z = inputs()
    e1 = transpose_view(transpose_view(x))
    e2 = transpose_view(transpose_view(e1))
    e3 = add_in_place(e2, y)
    e4 = add_in_place(e1, z)
    g = Env([x,y,z], [e3, e4], False)
    inconsistent(g)
    g.replace(e2, transpose_view(x))
    inconsistent(g)

def test_usage_loop():
    x, y, z = inputs()
    g = Env([x,y,z], [dot(add_in_place(x, z), x)], False)
    inconsistent(g)
    OpSubOptimizer(add_in_place, add).optimize(g) # replace add_in_place with add
    consistent(g)

def test_usage_loop_through_views():
    x, y, z = inputs()
    aip = add_in_place(x, y)
    e = dot(aip, transpose_view(x))
    g = Env([x,y,z], [e], False)
    inconsistent(g)
    g.replace_validate(aip, add(x, z))
    consistent(g)

def test_usage_loop_insert_views():
    x, y, z = inputs()
    e = dot(add_in_place(x, add(y, z)), sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(x))))))
    g = Env([x,y,z], [e])
    consistent(g)
    fail = FailureWatch()
    OpSubOptimizer(sigmoid, transpose_view, fail).optimize(g)
    consistent(g)
    assert fail.failures == 1 # it must keep one sigmoid in the long sigmoid chain

def test_value_repl():
    x, y, z = inputs()
    sy = sigmoid(y)
    e = add_in_place(x, sy)
    g = Env([x,y], [e], False)
    consistent(g)
    g.replace(sy, MyValue("abc"))
    consistent(g)

def test_value_repl_2():
    x, y, z = inputs()
    sy = sigmoid(y)
    e = add_in_place(x, sy)
    g = Env([x,y], [e], False)
    consistent(g)
    g.replace(sy, transpose_view(MyValue("abc")))
    consistent(g)




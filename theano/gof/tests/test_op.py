from copy import copy
import unittest

import numpy

import theano

import theano.gof.op as op
from theano.gof.type import Type, Generic
from theano.gof.graph import Apply, Variable
import theano.tensor as T
from theano import scalar
from theano import shared

config = theano.config
Op = op.Op
utils = op.utils


def as_variable(x):
    assert isinstance(x, Variable)
    return x


class MyType(Type):

    def __init__(self, thingy):
        self.thingy = thingy

    def __eq__(self, other):
        return type(other) == type(self) and other.thingy == self.thingy

    def __str__(self):
        return str(self.thingy)

    def __repr__(self):
        return str(self.thingy)

    def filter(self, x, strict=False, allow_downcast=None):
        # Dummy filter: we want this type to represent strings that
        # start with `self.thingy`.
        if not isinstance(x, basestring):
            raise TypeError("Invalid type")
        if not x.startswith(self.thingy):
            raise ValueError("Invalid value")
        return x

class MyOp(Op):

    def make_node(self, *inputs):
        inputs = map(as_variable, inputs)
        for input in inputs:
            if not isinstance(input.type, MyType):
                raise Exception("Error 1")
            outputs = [MyType(sum([input.type.thingy for input in inputs]))()]
            return Apply(self, inputs, outputs)

MyOp = MyOp()


class NoInputOp(Op):

    """An Op to test the corner-case of an Op with no input."""

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self):
        return Apply(self, [], [MyType('test')()])

    def perform(self, node, inputs, output_storage):
        output_storage[0][0] = 'test Op no input'


class TestOp:

    # Sanity tests
    def test_sanity_0(self):
        r1, r2 = MyType(1)(), MyType(2)()
        node = MyOp.make_node(r1, r2)
        assert [x for x in node.inputs] == [r1, r2] # Are the inputs what I provided?
        assert [x.type for x in node.outputs] == [MyType(3)] # Are the outputs what I expect?
        assert node.outputs[0].owner is node and node.outputs[0].index == 0

    # validate
    def test_validate(self):
        try:
            MyOp(Generic()(), MyType(1)()) # MyOp requires MyType instances
            raise Exception("Expected an exception")
        except Exception, e:
            if str(e) != "Error 1":
                raise

    def test_op_no_input(self):
        x = NoInputOp()()
        f = theano.function([], x)
        rval = f()
        assert rval == 'test Op no input'

class TestMakeThunk(unittest.TestCase):
    def test_no_c_code(self):
        class IncOnePython(Op):
            """An Op with only a Python (perform) implementation"""

            def __eq__(self, other):
                return type(self) == type(other)

            def __hash__(self):
                return hash(type(self))

            def make_node(self, input):
                input = scalar.as_scalar(input)
                output = input.type()
                return Apply(self, [input], [output])

            def perform(self, node, inputs, outputs):
                input, = inputs
                output, = outputs
                output[0] = input + 1


        i = scalar.int32('i')
        o = IncOnePython()(i)

        # Check that the c_code function is not implemented
        self.assertRaises((NotImplementedError, utils.MethodNotDefined),
                o.owner.op.c_code,
                o.owner, 'o', ['x'], 'z', {'fail': ''})

        storage_map = {
                i: [numpy.int32(3)],
                o: [None]}
        compute_map = {
                i: [True],
                o: [False]}

        thunk = o.owner.op.make_thunk(o.owner, storage_map, compute_map,
                no_recycling=[])

        required = thunk()
        # Check everything went OK
        assert not required # We provided all inputs
        assert compute_map[o][0]
        assert storage_map[o][0] == 4

    def test_no_perform(self):
        class IncOneC(Op):
            """An Op with only a C (c_code) implementation"""

            def __eq__(self, other):
                return type(self) == type(other)

            def __hash__(self):
                return hash(type(self))

            def make_node(self, input):
                input = scalar.as_scalar(input)
                output = input.type()
                return Apply(self, [input], [output])

            def c_code(self, node, name, inputs, outputs, sub):
                x, = inputs
                z, = outputs
                return "%(z)s = %(x)s + 1;" % locals()


        i = scalar.int32('i')
        o = IncOneC()(i)

        # Check that the perform function is not implemented
        self.assertRaises((NotImplementedError, utils.MethodNotDefined),
                o.owner.op.perform,
                o.owner, 0, [None])

        storage_map = {
                i: [numpy.int32(3)],
                o: [None]}
        compute_map = {
                i: [True],
                o: [False]}

        thunk = o.owner.op.make_thunk(o.owner, storage_map, compute_map,
                no_recycling=[])

        required = thunk()
        # Check everything went OK
        assert not required # We provided all inputs
        assert compute_map[o][0]
        assert storage_map[o][0] == 4


def test_test_value_python_objects():
    for x in (range(3), 0, 0.5, 1):
        assert (op.get_test_value(x) == x).all()


def test_test_value_ndarray():
    x = numpy.zeros((5,5))
    v = op.get_test_value(x)
    assert (v == x).all()

def test_test_value_constant():
    x = T.as_tensor_variable(numpy.zeros((5,5)))
    v = op.get_test_value(x)

    assert numpy.all(v == numpy.zeros((5,5)))

def test_test_value_shared():
    x = shared(numpy.zeros((5,5)))
    v = op.get_test_value(x)

    assert numpy.all(v == numpy.zeros((5,5)))

def test_test_value_op():
    try:
        prev_value = config.compute_test_value
        config.compute_test_value = 'raise'
        x = T.log(numpy.ones((5,5)))
        v = op.get_test_value(x)

        assert numpy.allclose(v, numpy.zeros((5,5)))
    finally:
        config.compute_test_value = prev_value


def test_get_debug_values_no_debugger():
    'get_debug_values should return [] when debugger is off'

    prev_value = config.compute_test_value
    try:
        config.compute_test_value = 'off'

        x = T.vector()

        for x_val in op.get_debug_values(x):
            assert False

    finally:
        config.compute_test_value = prev_value

def test_get_det_debug_values_ignore():
    """get_debug_values should return [] when debugger is ignore
        and some values are missing """


    prev_value = config.compute_test_value
    try:
        config.compute_test_value = 'ignore'

        x = T.vector()

        for x_val in op.get_debug_values(x):
            assert False

    finally:
        config.compute_test_value = prev_value


def test_get_debug_values_success():
    """tests that get_debug_value returns values when available
    (and the debugger is on)"""

    prev_value = config.compute_test_value
    for mode in [ 'ignore', 'warn', 'raise' ]:

        try:
            config.compute_test_value = mode

            x = T.vector()
            x.tag.test_value = numpy.zeros((4,), dtype=config.floatX)
            y = numpy.zeros((5,5))

            iters = 0

            for x_val, y_val in op.get_debug_values(x, y):

                assert x_val.shape == (4,)
                assert y_val.shape == (5,5)

                iters += 1

            assert iters == 1

        finally:
            config.compute_test_value = prev_value

def test_get_debug_values_exc():
    """tests that get_debug_value raises an exception when
        debugger is set to raise and a value is missing """

    prev_value = config.compute_test_value
    try:
        config.compute_test_value = 'raise'

        x = T.vector()

        try:
            for x_val in op.get_debug_values(x):
                #this assert catches the case where we
                #erroneously get a value returned
                assert False
            raised = False
        except AttributeError:
            raised = True

        #this assert catches the case where we got []
        #returned, and possibly issued a warning,
        #rather than raising an exception
        assert raised

    finally:
        config.compute_test_value = prev_value

def test_debug_error_message():
    """tests that debug_error_message raises an
    exception when it should."""

    prev_value = config.compute_test_value

    for mode in [ 'ignore', 'raise' ]:

        try:
            config.compute_test_value = mode

            try:
                op.debug_error_message('msg')
                raised = False
            except ValueError:
                raised = True
            assert raised
        finally:
            config.compute_test_value = prev_value

if __name__ == '__main__':
    unittest.main()

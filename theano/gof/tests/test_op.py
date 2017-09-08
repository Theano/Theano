from __future__ import absolute_import, print_function, division
import unittest

from nose.plugins.skip import SkipTest
import numpy as np

import theano
import theano.gof.op as op
from six import string_types
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
        if not isinstance(x, string_types):
            raise TypeError("Invalid type")
        if not x.startswith(self.thingy):
            raise ValueError("Invalid value")
        return x

    # Added to make those tests pass in DebugMode
    @staticmethod
    def may_share_memory(a, b):
        # As this represent a string and string are immutable, they
        # never share memory in the DebugMode sence. This is needed as
        # Python reuse string internally.
        return False


class MyOp(Op):

    __props__ = ()

    def make_node(self, *inputs):
        inputs = list(map(as_variable, inputs))
        for input in inputs:
            if not isinstance(input.type, MyType):
                raise Exception("Error 1")
            outputs = [MyType(sum([input.type.thingy for input in inputs]))()]
            return Apply(self, inputs, outputs)

MyOp = MyOp()


class NoInputOp(Op):
    """An Op to test the corner-case of an Op with no input."""
    __props__ = ()

    def make_node(self):
        return Apply(self, [], [MyType('test')()])

    def perform(self, node, inputs, output_storage):
        output_storage[0][0] = 'test Op no input'


class StructOp(Op):
    __props__ = ()

    def do_constant_folding(self, node):
        # we are not constant
        return False

    # The input only serves to distinguish thunks
    def make_node(self, i):
        return Apply(self, [i], [scalar.uint64()])

    def c_support_code_struct(self, node, name):
        return "npy_uint64 counter%s;" % (name,)

    def c_init_code_struct(self, node, name, sub):
        return "counter%s = 0;" % (name,)

    def c_code(self, node, name, input_names, outputs_names, sub):
        return """
%(out)s = counter%(name)s;
counter%(name)s++;
""" % dict(out=outputs_names[0], name=name)

    def c_code_cache_version(self):
        return (1,)


class TestOp:

    # Sanity tests
    def test_sanity_0(self):
        r1, r2 = MyType(1)(), MyType(2)()
        node = MyOp.make_node(r1, r2)
        # Are the inputs what I provided?
        assert [x for x in node.inputs] == [r1, r2]
        # Are the outputs what I expect?
        assert [x.type for x in node.outputs] == [MyType(3)]
        assert node.outputs[0].owner is node and node.outputs[0].index == 0

    # validate
    def test_validate(self):
        try:
            MyOp(Generic()(), MyType(1)())  # MyOp requires MyType instances
            raise Exception("Expected an exception")
        except Exception as e:
            if str(e) != "Error 1":
                raise

    def test_op_no_input(self):
        x = NoInputOp()()
        f = theano.function([], x)
        rval = f()
        assert rval == 'test Op no input'

    def test_op_struct(self):
        if not theano.config.cxx:
            raise SkipTest("G++ not available, so we need to skip this test.")
        sop = StructOp()
        c = sop(theano.tensor.constant(0))
        mode = None
        if theano.config.mode == 'FAST_COMPILE':
            mode = 'FAST_RUN'
        f = theano.function([], c, mode=mode)
        rval = f()
        assert rval == 0
        rval = f()
        assert rval == 1

        c2 = sop(theano.tensor.constant(1))
        f2 = theano.function([], [c, c2], mode=mode)
        rval = f2()
        assert rval == [0, 0]


class TestMakeThunk(unittest.TestCase):

    def test_no_c_code(self):
        class IncOnePython(Op):
            """An Op with only a Python (perform) implementation"""
            __props__ = ()

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

        storage_map = {i: [np.int32(3)],
                       o: [None]}
        compute_map = {i: [True],
                       o: [False]}

        thunk = o.owner.op.make_thunk(o.owner, storage_map, compute_map,
                                      no_recycling=[])

        required = thunk()
        # Check everything went OK
        assert not required  # We provided all inputs
        assert compute_map[o][0]
        assert storage_map[o][0] == 4

    def test_no_perform(self):
        class IncOneC(Op):
            """An Op with only a C (c_code) implementation"""
            __props__ = ()

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

        storage_map = {i: [np.int32(3)],
                       o: [None]}
        compute_map = {i: [True],
                       o: [False]}

        thunk = o.owner.op.make_thunk(o.owner, storage_map, compute_map,
                                      no_recycling=[])
        if theano.config.cxx:
            required = thunk()
            # Check everything went OK
            assert not required  # We provided all inputs
            assert compute_map[o][0]
            assert storage_map[o][0] == 4
        else:
            self.assertRaises((NotImplementedError, utils.MethodNotDefined),
                              thunk)

    def test_no_make_node(self):
        class DoubleOp(Op):
            """An Op without make_node"""
            __props__ = ()

            itypes = [T.dmatrix]
            otypes = [T.dmatrix]

            def perform(self, node, inputs, outputs):
                inp = inputs[0]
                output = outputs[0]
                output[0] = inp * 2

        x_input = T.dmatrix('x_input')
        f = theano.function([x_input], DoubleOp()(x_input))
        inp = np.random.rand(5, 4)
        out = f(inp)
        assert np.allclose(inp * 2, out)


def test_test_value_python_objects():
    for x in ([0, 1, 2], 0, 0.5, 1):
        assert (op.get_test_value(x) == x).all()


def test_test_value_ndarray():
    x = np.zeros((5, 5))
    v = op.get_test_value(x)
    assert (v == x).all()


def test_test_value_constant():
    x = T.as_tensor_variable(np.zeros((5, 5)))
    v = op.get_test_value(x)

    assert np.all(v == np.zeros((5, 5)))


def test_test_value_shared():
    x = shared(np.zeros((5, 5)))
    v = op.get_test_value(x)

    assert np.all(v == np.zeros((5, 5)))


def test_test_value_op():
    try:
        prev_value = config.compute_test_value
        config.compute_test_value = 'raise'
        x = T.log(np.ones((5, 5)))
        v = op.get_test_value(x)

        assert np.allclose(v, np.zeros((5, 5)))
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
    # get_debug_values should return [] when debugger is ignore
    # and some values are missing

    prev_value = config.compute_test_value
    try:
        config.compute_test_value = 'ignore'

        x = T.vector()

        for x_val in op.get_debug_values(x):
            assert False

    finally:
        config.compute_test_value = prev_value


def test_get_debug_values_success():
    # tests that get_debug_value returns values when available
    # (and the debugger is on)

    prev_value = config.compute_test_value
    for mode in ['ignore', 'warn', 'raise']:

        try:
            config.compute_test_value = mode

            x = T.vector()
            x.tag.test_value = np.zeros((4,), dtype=config.floatX)
            y = np.zeros((5, 5))

            iters = 0

            for x_val, y_val in op.get_debug_values(x, y):

                assert x_val.shape == (4,)
                assert y_val.shape == (5, 5)

                iters += 1

            assert iters == 1

        finally:
            config.compute_test_value = prev_value


def test_get_debug_values_exc():
    # tests that get_debug_value raises an exception when
    # debugger is set to raise and a value is missing

    prev_value = config.compute_test_value
    try:
        config.compute_test_value = 'raise'

        x = T.vector()

        try:
            for x_val in op.get_debug_values(x):
                # this assert catches the case where we
                # erroneously get a value returned
                assert False
            raised = False
        except AttributeError:
            raised = True

        # this assert catches the case where we got []
        # returned, and possibly issued a warning,
        # rather than raising an exception
        assert raised

    finally:
        config.compute_test_value = prev_value


def test_debug_error_message():
    # tests that debug_error_message raises an
    # exception when it should.

    prev_value = config.compute_test_value

    for mode in ['ignore', 'raise']:

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

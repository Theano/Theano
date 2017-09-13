from __future__ import absolute_import, print_function, division
import os
import numpy as np

import theano
from theano import Op, Apply, scalar
from theano.tensor import TensorType
from theano.gof.type import CDataType, EnumType, EnumList, CEnumType
from unittest import TestCase

from nose.plugins.skip import SkipTest

# todo: test generic


class ProdOp(Op):
    __props__ = ()

    def make_node(self, i):
        return Apply(self, [i], [CDataType('void *', 'py_decref')()])

    def c_support_code(self):
        return """
void py_decref(void *p) {
  Py_XDECREF((PyObject *)p);
}
"""

    def c_code(self, node, name, inps, outs, sub):
        return """
Py_XDECREF(%(out)s);
%(out)s = (void *)%(inp)s;
Py_INCREF(%(inp)s);
""" % dict(out=outs[0], inp=inps[0])

    def c_code_cache_version(self):
        return (0,)


class GetOp(Op):
    __props__ = ()

    def make_node(self, c):
        return Apply(self, [c], [TensorType('float32', (False,))()])

    def c_support_code(self):
        return """
void py_decref(void *p) {
  Py_XDECREF((PyObject *)p);
}
"""

    def c_code(self, node, name, inps, outs, sub):
        return """
Py_XDECREF(%(out)s);
%(out)s = (PyArrayObject *)%(inp)s;
Py_INCREF(%(out)s);
""" % dict(out=outs[0], inp=inps[0])

    def c_code_cache_version(self):
        return (0,)


def test_cdata():
    if not theano.config.cxx:
        raise SkipTest("G++ not available, so we need to skip this test.")
    i = TensorType('float32', (False,))()
    c = ProdOp()(i)
    i2 = GetOp()(c)
    mode = None
    if theano.config.mode == "FAST_COMPILE":
        mode = "FAST_RUN"

    # This should be a passthrough function for vectors
    f = theano.function([i], i2, mode=mode)

    v = np.random.randn(9).astype('float32')

    v2 = f(v)
    assert (v2 == v).all()


class MyOpEnumList(Op):
    __props__ = ('op_chosen',)
    params_type = EnumList(('ADD', '+'), ('SUB', '-'), ('MULTIPLY', '*'), ('DIVIDE', '/'), ctype='unsigned long long')

    def __init__(self, choose_op):
        assert self.params_type.ADD == 0
        assert self.params_type.SUB == 1
        assert self.params_type.MULTIPLY == 2
        assert self.params_type.DIVIDE == 3
        assert self.params_type.fromalias('+') == self.params_type.ADD
        assert self.params_type.fromalias('-') == self.params_type.SUB
        assert self.params_type.fromalias('*') == self.params_type.MULTIPLY
        assert self.params_type.fromalias('/') == self.params_type.DIVIDE
        assert self.params_type.has_alias(choose_op)
        self.op_chosen = choose_op

    def get_params(self, node):
        return self.op_chosen

    def make_node(self, a, b):
        return Apply(self, [scalar.as_scalar(a), scalar.as_scalar(b)], [scalar.float64()])

    def perform(self, node, inputs, outputs, op):
        a, b = inputs
        o, = outputs
        if op == self.params_type.ADD:
            o[0] = a + b
        elif op == self.params_type.SUB:
            o[0] = a - b
        elif op == self.params_type.MULTIPLY:
            o[0] = a * b
        elif op == self.params_type.DIVIDE:
            if any(dtype in theano.tensor.continuous_dtypes for dtype in (a.dtype, b.dtype)):
                o[0] = a / b
            else:
                o[0] = a // b
        else:
            raise NotImplementedError('Unknown op id ' + str(op))
        o[0] = np.float64(o[0])

    def c_code_cache_version(self):
        return (1,)

    def c_code(self, node, name, inputs, outputs, sub):
        return """
        switch(%(op)s) {
            case ADD:
                %(o)s = %(a)s + %(b)s;
                break;
            case SUB:
                %(o)s = %(a)s - %(b)s;
                break;
            case MULTIPLY:
                %(o)s = %(a)s * %(b)s;
                break;
            case DIVIDE:
                %(o)s = %(a)s / %(b)s;
                break;
            default:
                {%(fail)s}
                break;
        }
        """ % dict(op=sub['params'], o=outputs[0], a=inputs[0], b=inputs[1], fail=sub['fail'])


class MyOpCEnumType(Op):
    __props__ = ('python_value',)
    params_type = CEnumType(('MILLION', 'million'), ('BILLION', 'billion'), ('TWO_BILLIONS', 'two_billions'),
                            ctype='size_t')

    def c_header_dirs(self):
        return [os.path.join(os.path.dirname(__file__), 'c_code')]

    def c_headers(self):
        return ['test_cenum.h']

    def __init__(self, value_name):
        # As we see, Python values of constants are not related to real C values.
        assert self.params_type.MILLION == 0
        assert self.params_type.BILLION == 1
        assert self.params_type.TWO_BILLIONS == 2
        assert self.params_type.has_alias(value_name)
        self.python_value = self.params_type.fromalias(value_name)

    def get_params(self, node):
        return self.python_value

    def make_node(self):
        return Apply(self, [], [scalar.uint32()])

    def c_code_cache_version(self):
        return (3,)

    def c_code(self, node, name, inputs, outputs, sub):
        return """
        %(o)s = %(val)s;
        """ % dict(o=outputs[0],
                   # params in C code will already contains expected C constant value.
                   val=sub['params'])


class TestEnumTypes(TestCase):

    def test_enum_class(self):
        # Check that invalid enum name raises exception.
        for invalid_name in ('a', '_A', '0'):
            try:
                EnumList(invalid_name)
            except AttributeError:
                pass
            else:
                raise Exception('EnumList with invalid name should faild.')

            try:
                EnumType(**{invalid_name: 0})
            except AttributeError:
                pass
            else:
                raise Exception('EnumType with invalid name should fail.')

        # Check that invalid enum value raises exception.
        try:
            EnumType(INVALID_VALUE='string is not allowed.')
        except TypeError:
            pass
        else:
            raise Exception('EnumType with invalid value should fail.')

        # Check EnumType.
        e1 = EnumType(C1=True, C2=12, C3=True, C4=-1, C5=False, C6=0.0)
        e2 = EnumType(C1=1, C2=12, C3=1, C4=-1.0, C5=0.0, C6=0)
        assert e1 == e2
        assert not (e1 != e2)
        assert hash(e1) == hash(e2)
        # Check access to attributes.
        assert len((e1.ctype, e1.C1, e1.C2, e1.C3, e1.C4, e1.C5, e1.C6)) == 7

        # Check enum with aliases.
        e1 = EnumType(A=('alpha', 0), B=('beta', 1), C=2)
        e2 = EnumType(A=('alpha', 0), B=('beta', 1), C=2)
        e3 = EnumType(A=('a', 0), B=('beta', 1), C=2)
        assert e1 == e2
        assert e1 != e3
        assert e1.filter('beta') == e1.fromalias('beta') == e1.B == 1
        assert e1.filter('C') == e1.fromalias('C') == e1.C == 2

        # Check that invalid alias (same as a constant) raises exception.
        try:
            EnumList(('A', 'a'), ('B', 'B'))
        except TypeError:
            EnumList(('A', 'a'), ('B', 'b'))
        else:
            raise Exception('Enum with an alias name equal to a constant name should fail.')

    def test_op_with_enumlist(self):
        a = scalar.int32()
        b = scalar.int32()
        c_add = MyOpEnumList('+')(a, b)
        c_sub = MyOpEnumList('-')(a, b)
        c_multiply = MyOpEnumList('*')(a, b)
        c_divide = MyOpEnumList('/')(a, b)
        f = theano.function([a, b], [c_add, c_sub, c_multiply, c_divide])
        va = 12
        vb = 15
        ref = [va + vb, va - vb, va * vb, va // vb]
        out = f(va, vb)
        assert ref == out, (ref, out)

    def test_op_with_cenumtype(self):
        if theano.config.cxx == '':
            raise SkipTest('need c++')
        million = MyOpCEnumType('million')()
        billion = MyOpCEnumType('billion')()
        two_billions = MyOpCEnumType('two_billions')()
        f = theano.function([], [million, billion, two_billions])
        val_million, val_billion, val_two_billions = f()
        assert val_million == 1000000
        assert val_billion == val_million * 1000
        assert val_two_billions == val_billion * 2

    @theano.change_flags(**{'cmodule.debug': True})
    def test_op_with_cenumtype_debug(self):
        self.test_op_with_cenumtype()

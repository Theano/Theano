from __future__ import absolute_import, print_function, division
import numpy as np

import theano
from theano import Op, Apply
from theano.tensor import TensorType
from theano.gof.type import CDataType

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

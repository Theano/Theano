from __future__ import absolute_import, print_function, division
from nose.plugins.skip import SkipTest
import unittest

import numpy as np

from theano import config
from theano import gof
import theano
import theano.tensor
from theano.compat import exc_message
from theano.compile import debugmode
import theano.compile
from theano.tests import unittest_tools as utt


def test0():
    x = theano.tensor.dvector()
    f = theano.function([x], ((2. * x) + 7) / 2., mode=debugmode.DebugMode())
    f([1, 2])


class BROKEN_ON_PURPOSE_Add(gof.Op):

    __props__ = ("py_offset",)

    def __init__(self, py_offset):
        gof.Op.__init__(self)
        self.py_offset = py_offset

    def make_node(self, a, b):
        a = theano.tensor.as_tensor_variable(a)
        b = theano.tensor.as_tensor_variable(b)
        assert a.type.dtype == 'float64'
        assert a.type.dtype == b.type.dtype
        assert a.type.ndim == 1
        r = gof.Apply(self, [a, b], [a.type()])
        return r

    def perform(self, node, inp, out_):
        a, b = inp
        out, = out_
        z = a + b
        # ERROR TO ADD THIS CRAPPY OFFSET
        if self.py_offset:
            out[0] = z + 0.5
        else:
            out[0] = z

    def c_code_cache_version(self):
        return (1,)

    def c_code(self, node, name, inp, out, sub):
        a, b = inp
        z, = out
        return """
        if (PyArray_NDIM(%(a)s) != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(a) != 1"); %(fail)s;}
        if (PyArray_NDIM(%(b)s) != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 1"); %(fail)s;}

        if (PyArray_DESCR(%(a)s)->type_num != NPY_DOUBLE)
        {PyErr_SetString(PyExc_NotImplementedError, "a dtype not NPY_DOUBLE"); %(fail)s;}

        if (PyArray_DESCR(%(b)s)->type_num != NPY_DOUBLE)
        {PyErr_SetString(PyExc_NotImplementedError, "b's dtype not NPY_DOUBLE"); %(fail)s;}

        if (PyArray_DIMS(%(a)s)[0] != PyArray_DIMS(%(b)s)[0])
        {PyErr_SetString(PyExc_NotImplementedError, "a and b have different lengths"); %(fail)s;}

        if ((!%(z)s)
            || (PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(b)s)[0])
            )
        {
            {Py_XDECREF(%(z)s);}
            npy_intp dims[] = {0};
            dims[0] = PyArray_DIMS(%(b)s)[0];
            %(z)s = (PyArrayObject*) PyArray_SimpleNew(1, dims, PyArray_DESCR(%(b)s)->type_num);
        }

        {
            for (npy_intp m = 0; m < PyArray_DIMS(%(z)s)[0]; ++m)
            {
                ((double*)PyArray_GETPTR1(%(z)s, m))[0]
                = 0.5
                + ((double*)PyArray_GETPTR1(%(a)s, m))[0]
                + ((double*)PyArray_GETPTR1(%(b)s, m))[0] ;
            }
        }
        """ % dict(locals(), **sub)

# inconsistent is a invalid op, whose perform and c_code do not match
inconsistent = BROKEN_ON_PURPOSE_Add(False)

# off_by_half is a good op, that is different from theano.sparse.sd_csc
off_by_half = BROKEN_ON_PURPOSE_Add(True)


class WeirdBrokenOp(gof.Op):
    """
    This op can be inplace if behaviour is 'times1_inplace'
    This op can be destructive if behaviour is 'times2_inplace'

    In both cases, it does not set the destroy_map or view_map correctly so
    it should raise an error in DebugMode.
    """
    __props__ = ("behaviour", )

    def __init__(self, behaviour):
        gof.Op.__init__(self)
        self.behaviour = behaviour

    def make_node(self, a):
        a_ = theano.tensor.as_tensor_variable(a)
        r = gof.Apply(self, [a_], [a_.type()])
        return r

    def dontuse_perform(self, node, inp, out_):
        a, = inp
        out, = out_
        if self.behaviour == 'times2':
            out[0] = a * 2
        elif self.behaviour == 'times2_inplace':
            out[0] = a
            out[0] *= 2
        elif self.behaviour == 'times1':
            out[0] = a * 1
        elif self.behaviour == 'times1_inplace':
            out[0] = a
        else:
            raise ValueError(self.behaviour)

    def c_code_cache_version(self):
        return (1,)

    def c_code(self, node, name, inp, out, sub):
        a, = inp
        z, = out
        if "inplace" in self.behaviour:
            z_code = """
            {Py_XDECREF(%(z)s);}
            Py_INCREF(%(a)s);
            %(z)s = %(a)s;
            """
        else:
            z_code = """
            {Py_XDECREF(%(z)s);}
            %(z)s = (PyArrayObject*) PyArray_SimpleNew(1, PyArray_DIMS(%(a)s), PyArray_DESCR(%(a)s)->type_num);
            """
        prep_vars = """
            //the output array has size M x N
            npy_intp M = PyArray_DIMS(%(a)s)[0];
            npy_intp Sa = PyArray_STRIDES(%(a)s)[0] / PyArray_DESCR(%(a)s)->elsize;
            npy_intp Sz = PyArray_STRIDES(%(z)s)[0] / PyArray_DESCR(%(z)s)->elsize;

            npy_double * Da = (npy_double*)PyArray_BYTES(%(a)s);
            npy_double * Dz = (npy_double*)PyArray_BYTES(%(z)s);

            //clear the output array
            for (npy_intp m = 0; m < M; ++m)
            {
        """

        if self.behaviour == 'times2':
            behaviour = "     Dz[m * Sz] = 2 * Da[m * Sa]; "
            # out[0] = a * 2
        elif self.behaviour == 'times2_inplace':
            # out[0] = a
            # out[0] *= 2
            behaviour = "     Dz[m * Sz] = 2 * Da[m * Sa]; "
        elif self.behaviour == 'times1':
            # out[0] = a * 1
            behaviour = "     Dz[m * Sz] = Da[m * Sa]; "
        elif self.behaviour == 'times1_inplace':
            # out[0] = a
            behaviour = ""
        else:
            raise ValueError(self.behaviour)

        prep_vars2 = """
            }
        """

        total = ((z_code + prep_vars + behaviour + prep_vars2)
                 % dict(locals(), **sub))
        return total

wb2i = WeirdBrokenOp('times2_inplace')
wb2 = WeirdBrokenOp('times2')
wb1i = WeirdBrokenOp('times1_inplace')
wb1 = WeirdBrokenOp('times1')


def test_badthunkoutput():
    # Check if the c and python code is consistent.
    a = theano.tensor.dvector()
    b = theano.tensor.dvector()

    f_good = theano.function([a, b],
                             off_by_half(a, b),
                             mode=debugmode.DebugMode(check_c_code=theano.config.cxx))
    f_inconsistent = theano.function([a, b],
                                     inconsistent(a, b),
                                     mode=debugmode.DebugMode(check_c_code=theano.config.cxx))

    # this should evaluate with no error
    f_good([1.0, 2.0, 3.0], [2, 3, 4])
    if not theano.config.cxx:
        raise SkipTest("G++ not available, so we need to skip this test.")

    try:
        f_inconsistent([1.0, 2.0, 3.0], [2, 3, 4])
    except debugmode.BadThunkOutput as e:
        # print repr(e)
        assert e.r.owner.op is inconsistent
        return  # TEST PASS

    assert False  # an error should have been detected


def test_badoptimization():
    @gof.local_optimizer([theano.tensor.add])
    def insert_broken_add(node):
        if node.op == theano.tensor.add:
            return [off_by_half(*node.inputs)]
        return False
    edb = gof.EquilibriumDB()
    edb.register('insert_broken_add', insert_broken_add, 'all')
    opt = edb.query('+all')

    a = theano.tensor.dvector()
    b = theano.tensor.dvector()

    f = theano.function([a, b], a + b,
                        mode=debugmode.DebugMode(optimizer=opt))

    try:
        f([1.0, 2.0, 3.0], [2, 3, 4],)
    except debugmode.BadOptimization as e:
        assert str(e.reason) == 'insert_broken_add'
        return  # TEST PASS

    assert False


def test_badoptimization_opt_err():
    """This variant of test_badoptimization() replace the working code
    with a new apply node that will raise an error.

    """
    @gof.local_optimizer([theano.tensor.add])
    def insert_bigger_b_add(node):
        if node.op == theano.tensor.add:
            inputs = list(node.inputs)
            if inputs[-1].owner is None:
                inputs[-1] = theano.tensor.concatenate((inputs[-1],
                                                        inputs[-1]))
                return [node.op(*inputs)]
        return False
    edb = gof.EquilibriumDB()
    edb.register('insert_bigger_b_add', insert_bigger_b_add, 'all')
    opt = edb.query('+all')

    a = theano.tensor.dvector()
    b = theano.tensor.dvector()

    f = theano.function([a, b], a + b,
                        mode=debugmode.DebugMode(optimizer=opt))

    try:
        f([1.0, 2.0, 3.0], [2, 3, 4],)
    except Exception as e:
        assert 'insert_bigger_b_add' in exc_message(e)
        return  # TEST PASS

    assert False


def test_stochasticoptimization():

    # this optimization alternates between triggering and not triggering.

    last_time_replaced = [False]

    @gof.local_optimizer([theano.tensor.add])
    def insert_broken_add_sometimes(node):
        if node.op == theano.tensor.add:
            last_time_replaced[0] = not last_time_replaced[0]
            if last_time_replaced[0]:
                return [off_by_half(*node.inputs)]
        return False

    edb = gof.EquilibriumDB()
    edb.register(
        'insert_broken_add_sometimes',
        insert_broken_add_sometimes,
        'all')
    opt = edb.query('+all')

    a = theano.tensor.dvector()
    b = theano.tensor.dvector()

    try:
        theano.function([a, b],
                        theano.tensor.add(a, b),
                        mode=debugmode.DebugMode(
                            optimizer=opt,
                            check_c_code=True,
                            stability_patience=max(2, config.DebugMode.patience)))
    except debugmode.StochasticOrder:
        return  # TEST PASS
    assert False


def test_just_c_code():
    if not theano.config.cxx:
        raise SkipTest("G++ not available, so we need to skip this test.")
    x = theano.tensor.dvector()
    f = theano.function([x], wb2(x),
                        mode=debugmode.DebugMode(check_py_code=False))
    assert np.all(f([1, 2]) == [2, 4])


def test_baddestroymap():
    class BadAdd(gof.Op):
        def make_node(self, a, b):
            c = a.type()
            return gof.Apply(self, [a, b], [c])

        def perform(self, node, inp, out):
            a, b = inp
            c, = out
            c[0] = a
            c[0] += b

    x = theano.tensor.dvector()
    y = theano.tensor.dvector()
    f = theano.function([x, y], BadAdd()(x, y), mode='DEBUG_MODE')

    try:
        f([1, 2], [3, 4])
        assert False  # failed to raise error
    except debugmode.BadDestroyMap:
        pass


def test_baddestroymap_c():
    if not theano.config.cxx:
        raise SkipTest("G++ not available, so we need to skip this test.")
    x = theano.tensor.dvector()
    f = theano.function([x], wb2i(x),
                        mode=debugmode.DebugMode(check_py_code=False))
    try:
        assert np.all(f([1, 2]) == [2, 4])
        assert False  # failed to raise error
    except debugmode.BadDestroyMap:
        pass


class Test_ViewMap(unittest.TestCase):

    class BadAddRef(gof.Op):
        def make_node(self, a, b):
            c = b.type()
            return gof.Apply(self, [a, b], [c])

        def perform(self, node, inp, out):
            a, b = inp
            c, = out
            c[0] = b

    class BadAddSlice(gof.Op):
        def make_node(self, a, b):
            c = b.type()
            return gof.Apply(self, [a, b], [c])

        def perform(self, node, inp, out):
            a, b = inp
            c, = out
            c[0] = b[1:3]

    def test_badviewmap_ref(self):
        x = theano.tensor.dvector()
        y = theano.tensor.dvector()
        f = theano.function([x, y], self.BadAddRef()(x, y), mode='DEBUG_MODE')
        try:
            f([1, 2], [3, 4])
            assert False  # failed to raise error
        except debugmode.BadViewMap:
            return

    def test_badviewmap_slice(self):
        x = theano.tensor.dvector()
        y = theano.tensor.dvector()
        f = theano.function([x, y], self.BadAddSlice()(x, y),
                            mode='DEBUG_MODE')
        try:
            f([1, 2], [3, 4])
            assert False  # failed to raise error
        except debugmode.BadViewMap:
            return

    def test_goodviewmap(self):
        goodop = self.BadAddRef()
        goodop.view_map = {0: [1]}
        x = theano.tensor.dvector()
        y = theano.tensor.dvector()
        f = theano.function([x, y], goodop(x, y), mode='DEBUG_MODE')
        try:
            f([1, 5, 1], [3, 4, 2, 1, 4])
            return
        except debugmode.BadViewMap:
            assert False  # failed to raise error

    def test_badviewmap_c(self):
        if not theano.config.cxx:
            raise SkipTest("G++ not available, so we need to skip this test.")
        x = theano.tensor.dvector()
        f = theano.function([x], wb1i(x),
                            mode=debugmode.DebugMode(check_py_code=False))
        try:
            f([1, 2])
            assert False  # failed to raise error
        except debugmode.BadViewMap:
            pass

    def test_aliased_outputs_ok(self):
        # here aliased outputs is ok because they are both aliased to an input
        # as well
        class CustomOp(gof.Op):
            view_map = {0: [0], 1: [0]}

            def make_node(self, a, b):
                c = a.type()
                d = a.type()
                return gof.Apply(self, [a, b], [c, d])

            def perform(self, node, inp, out):
                a, b = inp
                c, d = out
                c[0] = a
                d[0] = a[1:]

        x = theano.tensor.dvector('x')
        y = theano.tensor.dvector('y')
        f = theano.function([x, y], CustomOp()(x, y), mode='DEBUG_MODE')

        r0, r1 = f([1, 2, 3, 4], [5, 6, 7, 8])

        assert np.all(r0 == [1, 2, 3, 4])
        assert np.all(r1 == [2, 3, 4])

    def test_aliased_outputs_ok_output(self):
        # here aliased outputs is ok because they are both outputs of the
        # function as a whole and thus not destroy-able
        class CustomOp(gof.Op):
            def make_node(self, a, b):
                c = a.type()
                d = a.type()
                return gof.Apply(self, [a, b], [c, d])

            def perform(self, node, inp, out):
                a, b = inp
                c, d = out
                r = a * 2
                c[0] = r
                d[0] = r[1:]

        x = theano.tensor.dvector()
        y = theano.tensor.dvector()
        f = theano.function([x, y], CustomOp()(x, y), mode='DEBUG_MODE')

        r0, r1 = f([1, 2, 3, 4], [5, 6, 7, 8])

        assert np.all(r0 == [2, 4, 6, 8])
        assert np.all(r1 == [4, 6, 8])

    def test_aliased_outputs_ok_shadow(self):
        # here the alias between outputs is ok because one of them is not used
        # for subsequent computation.  This is like the case where we use one
        # output as a memory buffer to serve another output.
        class CustomOp(gof.Op):
            def make_node(self, a, b):
                c = a.type()
                d = a.type()
                return gof.Apply(self, [a, b], [c, d])

            def perform(self, node, inp, out):
                a, b = inp
                c, d = out
                r = a * 1
                c[0] = r
                d[0] = r[1:]

        x = theano.tensor.dvector('x')
        y = theano.tensor.dvector('y')
        f = theano.function([x, y], CustomOp()(x, y)[0] * 2, mode='DEBUG_MODE')

        r0 = f([1, 2, 3, 4], [5, 6, 7, 8])

        assert np.all(r0 == [2, 4, 6, 8])

    def test_aliased_outputs_bad(self):
        # here the alias between outputs is not ok because destroying one
        # destroys the other, but there's no way to warn theano about it
        # through the view_map mechanism.
        class CustomOp(gof.Op):
            def make_node(self, a, b):
                c = a.type()
                d = a.type()
                return gof.Apply(self, [a, b], [c, d])

            def perform(self, node, inp, out):
                a, b = inp
                c, d = out
                r = a * 1
                c[0] = r[:-1]
                d[0] = r[1:]

        custom_op = CustomOp()

        x = theano.tensor.dvector()
        y = theano.tensor.dvector()
        bad_xy0, bad_xy1 = custom_op(x, y)
        out = bad_xy0 * 2 + bad_xy1 * 2
        f = theano.function([x, y], out, mode='DEBUG_MODE')

        try:
            f([1, 2, 3, 4], [5, 6, 7, 8])
            assert False  # DebugMode should have caught the error
        except debugmode.BadViewMap:
            # print e
            pass

        # the situation can be rescued by picking one of the inputs and
        # pretending that it is aliased to both the outputs.
        # This unfairly disables any destructive operations on the
        # input, but guarantees correctness.
        # custom_op.view_map = {0:[0], 1:[1]}
        # f([1,2,3,4],[5,6,7,8])


class Test_check_isfinite(unittest.TestCase):
    def setUp(self):
        self.old_ts = theano.tensor.TensorType.filter_checks_isfinite
        self.old_dm = theano.compile.mode.predefined_modes[
            'DEBUG_MODE'].check_isfinite

    def tearDown(self):
        theano.tensor.TensorType.filter_checks_isfinite = self.old_ts
        theano.compile.mode.predefined_modes[
            'DEBUG_MODE'].check_isfinite = self.old_dm

    def test_check_isfinite(self):
        x = theano.tensor.vector()
        f = theano.function([x], (x + 2) * 5, mode='DEBUG_MODE')
        g = theano.function([x], theano.tensor.log(x), mode='DEBUG_MODE')

        # this should work
        f(np.log([3, 4, 5]).astype(config.floatX))

        # if TensorType.filter_checks_isfinite were true, these would raise
        # ValueError
        # if not, DebugMode will check internally, and raise InvalidValueError
        # passing an invalid value as an input should trigger ValueError
        self.assertRaises(debugmode.InvalidValueError, f,
                          np.log([3, -4, 5]).astype(config.floatX))
        self.assertRaises(debugmode.InvalidValueError, f,
                          (np.asarray([0, 1.0, 0]) / 0).astype(config.floatX))
        self.assertRaises(debugmode.InvalidValueError, f,
                          (np.asarray([1.0, 1.0, 1.0]) / 0).astype(config.floatX))

        # generating an invalid value internally should trigger
        # InvalidValueError
        self.assertRaises(debugmode.InvalidValueError, g,
                          np.asarray([3, -4, 5], dtype=config.floatX))

        # this should disable the exception
        theano.tensor.TensorType.filter_checks_isfinite = False
        theano.compile.mode.predefined_modes[
            'DEBUG_MODE'].check_isfinite = False
        # insert several Inf
        f(np.asarray(np.asarray([1.0, 1.0, 1.0]) / 0,
                     dtype=config.floatX))

    def test_check_isfinite_disabled(self):
        x = theano.tensor.dvector()
        f = theano.function([x], (x + 2) * 5,
                            mode=debugmode.DebugMode(check_isfinite=False))

        # nan should go through
        f(np.log([3, -4, 5]))

        # inf should go through
        infs = np.asarray([1.0, 1., 1.]) / 0
        # print infs
        f(infs)
        return


class BrokenCImplementationAdd(gof.Op):

    __props__ = ()

    def make_node(self, a, b):
        a = theano.tensor.as_tensor_variable(a)
        b = theano.tensor.as_tensor_variable(b)
        assert a.type.dtype == 'float32'
        assert a.type.dtype == b.type.dtype
        assert a.type.ndim == 2
        r = gof.Apply(self, [a, b], [a.type()])
        return r

    def perform(self, node, inp, out_):
        # print 'executing python perform'
        a, b = inp
        out, = out_
        z = a + b
        # print 'out[0] was:', out[0]
        out[0] = z

    def c_code_cache_version(self):
        return (1,)

    def c_code(self, node, name, inp, out, sub):
        a, b = inp
        z, = out
        debug = 0
        return """
        //printf("executing c_code\\n");
        if (PyArray_NDIM(%(a)s) != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(a) != 2"); %(fail)s;}
        if (PyArray_NDIM(%(b)s) != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 2"); %(fail)s;}

        if (PyArray_DESCR(%(a)s)->type_num != NPY_FLOAT)
        {PyErr_SetString(PyExc_NotImplementedError, "a dtype not NPY_FLOAT"); %(fail)s;}

        if (PyArray_DESCR(%(b)s)->type_num != NPY_FLOAT)
        {PyErr_SetString(PyExc_NotImplementedError, "b's dtype not NPY_FLOAT"); %(fail)s;}

        if (PyArray_DIMS(%(a)s)[0] != PyArray_DIMS(%(a)s)[1])
        {PyErr_SetString(PyExc_NotImplementedError, "a is not square"); %(fail)s;}

        if (PyArray_DIMS(%(b)s)[0] != PyArray_DIMS(%(b)s)[1])
        {PyErr_SetString(PyExc_NotImplementedError, "b is not square"); %(fail)s;}

        if (PyArray_DIMS(%(a)s)[0] != PyArray_DIMS(%(b)s)[0])
        {PyErr_SetString(PyExc_NotImplementedError, "a and b have different dimensions"); %(fail)s;}

        // We do not check for c_contiguous property here
        if (%(debug)s)
        {
            if (!%(z)s)
                printf("%(z)s is not there, %%p \\n", %(z)s);
            else if (PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(b)s)[0])
                printf("Dimension 0 mismatch for %(z)s and %(b)s\\n");
            else if (PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(b)s)[1])
                printf("Dimension 1 mismatch for %(z)s and %(b)s\\n");
            else
                printf("Reusing %(z)s\\n");
        }

        if ((!%(z)s)
            || (PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(b)s)[0])
            || (PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(b)s)[1])
            )
        {
            Py_XDECREF(%(z)s);
            npy_intp dims[] = {0, 0};
            dims[0] = PyArray_DIMS(%(b)s)[0];
            dims[1] = PyArray_DIMS(%(b)s)[1];
            %(z)s = (PyArrayObject*) PyArray_SimpleNew(2, dims, PyArray_DESCR(%(b)s)->type_num);
        }

        // Let us assume that %(z)s is c_contiguous
        {
            dtype_%(z)s * z = ((dtype_%(z)s*)(PyArray_GETPTR2(%(z)s,0,0)));
            for (int i=0; i<PyArray_DIMS(%(b)s)[0]; i++)
            {
                for (int j=0; j<PyArray_DIMS(%(b)s)[1]; j++)
                {
                    *z = ((float*)PyArray_GETPTR2(%(a)s, i, j))[0] +
                         ((float*)PyArray_GETPTR2(%(b)s, i, j))[0] ;
                    z++;
                }
            }
        }
        """ % dict(locals(), **sub)


class VecAsRowAndCol(gof.Op):
    """
    Transforms a vector into a row and a column.

    This Op exists to check everything is correct when an Op has
    two outputs with different broadcasting patterns.
    """
    __props__ = ()

    def make_node(self, v):
        if not isinstance(v, gof.Variable):
            v = theano.tensor.as_tensor_variable(v)
        assert v.type.ndim == 1
        type_class = type(v.type)
        out_r_type = type_class(dtype=v.dtype, broadcastable=(True, False))
        out_c_type = type_class(dtype=v.dtype, broadcastable=(False, True))
        return gof.Apply(self, [v], [out_r_type(), out_c_type()])

    def perform(self, node, inp, out):
        v, = inp
        r, c = out
        lv = v.shape[0]
        if (r[0] is None) or (r[0].shape != (1, lv)):
            r[0] = node.outputs[0].type.value_zeros((1, lv))

        if (c[0] is None) or (c[0].shape != (lv, 1)):
            c[0] = node.outputs[1].type.value_zeros((lv, 1))

        # Python loop because CudaNdarrays do not support newaxis
        for i in range(lv):
            r[0][0, i] = v[i]
            c[0][i, 0] = v[i]


class Test_preallocated_output(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(seed=utt.fetch_seed())

    def test_f_contiguous(self):
        a = theano.tensor.fmatrix('a')
        b = theano.tensor.fmatrix('b')
        z = BrokenCImplementationAdd()(a, b)
        # In this test, we do not want z to be an output of the graph.
        out = theano.tensor.dot(z, np.eye(7))

        a_val = self.rng.randn(7, 7).astype('float32')
        b_val = self.rng.randn(7, 7).astype('float32')

        # Should work
        mode = debugmode.DebugMode(
            check_preallocated_output=['c_contiguous'])

        f = theano.function([a, b], out, mode=mode)
        f(a_val, b_val)
        # print 'out_val =', out_val
        # print out_val.strides

        # Should raise an Exception, since the output buffer is
        # used incorrectly.
        mode = debugmode.DebugMode(
            check_preallocated_output=['f_contiguous'])

        f = theano.function([a, b], out, mode=mode)

        if theano.config.cxx:
            self.assertRaises(debugmode.BadThunkOutput, f, a_val, b_val)
        else:
            # The python code of this op is good.
            f(a_val, b_val)

    def test_f_contiguous_out(self):
        # Same test as test_f_contiguous, but check that it works
        # even if z _is_ the output of the graph
        a = theano.tensor.fmatrix('a')
        b = theano.tensor.fmatrix('b')
        out = BrokenCImplementationAdd()(a, b)

        a_val = self.rng.randn(7, 7).astype('float32')
        b_val = self.rng.randn(7, 7).astype('float32')

        # Should work
        mode = debugmode.DebugMode(
            check_preallocated_output=['c_contiguous'])

        f = theano.function([a, b], out, mode=mode)
        f(a_val, b_val)
        # print 'out_val =', out_val
        # print out_val.strides

        # Should raise an Exception, since the output buffer is
        # used incorrectly.
        mode = debugmode.DebugMode(
            check_preallocated_output=['f_contiguous'])

        f = theano.function([a, b], out, mode=mode)

        if theano.config.cxx:
            self.assertRaises(debugmode.BadThunkOutput, f, a_val, b_val)
        else:
            # The python code of this op is good.
            f(a_val, b_val)

    def test_output_broadcast_tensor(self):
        v = theano.tensor.fvector('v')
        c, r = VecAsRowAndCol()(v)
        f = theano.function([v], [c, r])

        v_val = self.rng.randn(5).astype('float32')
        f(v_val)

    def test_output_broadcast_cuda(self):
        from theano.sandbox import cuda
        if not cuda.cuda_available:
            raise SkipTest("Optional package Cuda disabled")
        if cuda.use.device_number is None:
            # We should normally set VecAsRowAndCol as a GPUOp But we
            # don't want to do this here as this will disable others
            # tests in this file.  So we manually init the GPU if
            # needed to remove warning.
            cuda.use("gpu",
                     force=True,
                     default_to_move_computation_to_gpu=False,
                     move_shared_float32_to_gpu=False,
                     enable_cuda=False)
        v = cuda.fvector('v')
        c, r = VecAsRowAndCol()(v)
        f = theano.function([v], [c, r])

        v_val = cuda.CudaNdarray(self.rng.randn(5).astype('float32'))
        f(v_val)

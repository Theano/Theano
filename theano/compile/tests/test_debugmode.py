import sys
import numpy

from theano import config
from theano import gof
import theano
import theano.tensor
from theano.compile import debugmode
import theano.compile
import unittest

def test0():
    x = theano.tensor.dvector()
    f = theano.function([x], (2.*x + 7) / 2., mode=debugmode.DebugMode())
    print f([1,2])

class BROKEN_ON_PURPOSE_Add(gof.Op):
    def __init__(self, py_offset):
        gof.Op.__init__(self)
        self.py_offset = py_offset
    def __eq__(self, other):
        return type(self) == type(other) and (self.py_offset == other.py_offset)
    def __hash__(self):
        return 29834 ^ hash(type(self)) ^ hash(self.py_offset)
    def make_node(self, a, b):
        a = theano.tensor.as_tensor_variable(a)
        b = theano.tensor.as_tensor_variable(b)
        assert a.type.dtype == 'float64'
        assert a.type.dtype == b.type.dtype
        assert a.type.ndim==1
        r = gof.Apply(self, [a, b], [a.type()])
        return r

    def perform(self, node, (a, b), (out,)):
        z = a+b
        #ERROR TO ADD THIS CRAPPY OFFSET
        if self.py_offset:
            out[0] = z+0.5
        else: out[0] = z 

    def c_code(self, node, name, (a, b), (z,), sub):
        return """
        if (%(a)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(a) != 1"); %(fail)s;}
        if (%(b)s->nd != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(b) != 1"); %(fail)s;}

        if (%(a)s->descr->type_num != PyArray_DOUBLE)
        {PyErr_SetString(PyExc_NotImplementedError, "a dtype not NPY_DOUBLE"); %(fail)s;}

        if (%(b)s->descr->type_num != PyArray_DOUBLE)
        {PyErr_SetString(PyExc_NotImplementedError, "b's dtype not NPY_DOUBLE"); %(fail)s;}

        if (%(a)s->dimensions[0] != %(b)s->dimensions[0])
        {PyErr_SetString(PyExc_NotImplementedError, "a and b have different lengths"); %(fail)s;}

        if ((!%(z)s)
            || (%(z)s->dimensions[0] != %(b)s->dimensions[0])
            )
        {
            {Py_XDECREF(%(z)s);}
            npy_intp dims[] = {0};
            dims[0] = %(b)s->dimensions[0];
            %(z)s = (PyArrayObject*) PyArray_SimpleNew(1, dims, %(b)s->descr->type_num);
        }

        {
            for (npy_intp m = 0; m < %(z)s->dimensions[0]; ++m)
            {
                ((double*)PyArray_GETPTR1(%(z)s, m))[0]
                = 0.5
                + ((double*)PyArray_GETPTR1(%(a)s, m))[0]
                + ((double*)PyArray_GETPTR1(%(b)s, m))[0] ;
            }
        }
        """% dict(locals(), **sub)
# inconsistent is a invalid op, whose perform and c_code do not match
inconsistent = BROKEN_ON_PURPOSE_Add(False)
# off_by_half is a good op, that is different from theano.sparse.sd_csc
off_by_half = BROKEN_ON_PURPOSE_Add(True) 

class WeirdBrokenOp(gof.Op):
    """
    This op can be inplace if behaviour is 'times1_inplace'
    This op can be destructive if behaviour is 'times2_inplace'

    In both cases, it does not set  the destroy_map or view_map correctly so it should raise an
    error in DebugMode.
    """
    def __init__(self, behaviour):
        gof.Op.__init__(self)
        self.behaviour = behaviour

    def __eq__(self, other):
        return type(self) == type(other) and (self.behaviour == other.behaviour)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.behaviour)

    def make_node(self, a):
        a_ = theano.tensor.as_tensor_variable(a)
        r = gof.Apply(self, [a_], [a_.type()])
        return r

    def dontuse_perform(self, node, (a,), (out,)):
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

    def c_code(self, node, name, (a,), (z,), sub):
        if "inplace" in self.behaviour:
            z_code = """
            {Py_XDECREF(%(z)s);}
            Py_INCREF(%(a)s);
            %(z)s = %(a)s;
            """
        else:
            z_code = """
            {Py_XDECREF(%(z)s);}
            %(z)s = (PyArrayObject*) PyArray_SimpleNew(1, %(a)s->dimensions, %(a)s->descr->type_num);
            """
        prep_vars = """
            //the output array has size M x N
            npy_intp M = %(a)s->dimensions[0];
            npy_intp Sa = %(a)s->strides[0] / %(a)s->descr->elsize;
            npy_intp Sz = %(z)s->strides[0] / %(z)s->descr->elsize;

            npy_double * Da = (npy_double*)%(a)s->data;
            npy_double * Dz = (npy_double*)%(z)s->data;

            //clear the output array
            for (npy_intp m = 0; m < M; ++m)
            {
        """

        if self.behaviour == 'times2':
            behaviour = "     Dz[m * Sz] = 2 * Da[m * Sa]; "
            #out[0] = a * 2
        elif self.behaviour == 'times2_inplace':
            #out[0] = a
            #out[0] *= 2
            behaviour = "     Dz[m * Sz] = 2 * Da[m * Sa]; "
        elif self.behaviour == 'times1':
            #out[0] = a * 1
            behaviour = "     Dz[m * Sz] = Da[m * Sa]; "
        elif self.behaviour == 'times1_inplace':
            #out[0] = a
            behaviour = ""
        else:
            raise ValueError(self.behaviour)

        prep_vars2 = """
            }
        """

        total = (z_code + prep_vars + behaviour + prep_vars2)% dict(locals(), **sub)
        return total

wb2i = WeirdBrokenOp('times2_inplace')
wb2 = WeirdBrokenOp('times2')
wb1i = WeirdBrokenOp('times1_inplace')
wb1 = WeirdBrokenOp('times1')

def test_badclinkeroutput():

    a = theano.tensor.dvector()
    b = theano.tensor.dvector()

    f_good = theano.function([a, b], 
            off_by_half(a, b), 
            mode=debugmode.DebugMode(check_c_code=True))
    f_inconsistent = theano.function([a,b], 
            inconsistent(a, b), 
            mode=debugmode.DebugMode(check_c_code=True))

    #this should evaluate with no error
    f_good([1.0, 2.0, 3.0], [2,3,4])
    try:
        f_inconsistent([1.0, 2.0, 3.0], [2,3,4])
    except debugmode.BadCLinkerOutput, e:
        print repr(e)
        assert e.r.owner.op is inconsistent
        return #TEST PASS

    assert False  #an error should have been detected
        

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

    f = theano.function([a, b], a+b, 
            mode=debugmode.DebugMode(optimizer=opt, check_c_code=True))

    try:
        rval = f([1.0, 2.0, 3.0], [2,3,4],)
    except debugmode.BadOptimization, e:
        assert str(e.reason) == 'insert_broken_add'
        return #TEST PASS

    assert False

def test_stochasticoptimization():

    # this optimization alternates between triggering and not triggering.

    last_time_replaced=[False]
    @gof.local_optimizer([theano.tensor.add])
    def insert_broken_add_sometimes(node):
        if node.op == theano.tensor.add:
            last_time_replaced[0] = not last_time_replaced[0]
            if last_time_replaced[0]:
                return [off_by_half(*node.inputs)]
        return False
    edb = gof.EquilibriumDB()
    edb.register('insert_broken_add_sometimes', insert_broken_add_sometimes, 'all')
    opt = edb.query('+all')

    a = theano.tensor.dvector()
    b = theano.tensor.dvector()

    try:
        f = theano.function([a, b], 
                theano.tensor.add(a, b), 
                mode=debugmode.DebugMode(optimizer=opt, check_c_code=True))
    except debugmode.StochasticOrder:
        return #TEST PASS
    assert False


def test_just_c_code():
    x = theano.tensor.dvector()
    f = theano.function([x], wb2(x), mode=debugmode.DebugMode(check_py_code=False))
    assert numpy.all(f([1,2]) == [2, 4])

def test_baddestroymap():
    class BadAdd(gof.Op):
        def make_node(self, a, b):
            c = a.type()
            return gof.Apply(self, [a,b], [c])
        def perform(self, node, (a,b), (c,)):
            c[0] = a
            c[0] += b

    x = theano.tensor.dvector()
    y = theano.tensor.dvector()
    f = theano.function([x, y], BadAdd()(x,y), mode='DEBUG_MODE')

    try:
        f([1,2], [3,4])
        assert False #failed to raise error
    except debugmode.BadDestroyMap:
        pass

def test_baddestroymap_c():
    x = theano.tensor.dvector()
    f = theano.function([x], wb2i(x), mode=debugmode.DebugMode(check_py_code=False))
    try:
        assert numpy.all(f([1,2]) == [2, 4])
        assert False #failed to raise error
    except debugmode.BadDestroyMap:
        pass


class Test_ViewMap(unittest.TestCase):

    class BadAddRef(gof.Op):
        def make_node(self, a, b):
            c = b.type()
            return gof.Apply(self, [a,b], [c])
        def perform(self, node, (a,b), (c,)):
            c[0] = b

    class BadAddSlice(gof.Op):
        def make_node(self, a, b):
            c = b.type()
            return gof.Apply(self, [a,b], [c])
        def perform(self, node, (a,b), (c,)):
            c[0] = b[1:3]

    def test_badviewmap_ref(self):
        x = theano.tensor.dvector()
        y = theano.tensor.dvector()
        f = theano.function([x, y], self.BadAddRef()(x,y), mode='DEBUG_MODE')
        try:
            f([1,2], [3,4])
            assert False #failed to raise error
        except debugmode.BadViewMap:
            return

    def test_badviewmap_slice(self):
        x = theano.tensor.dvector()
        y = theano.tensor.dvector()
        f = theano.function([x, y], self.BadAddSlice()(x,y), mode='DEBUG_MODE')
        try:
            f([1,2], [3,4])
            assert False #failed to raise error
        except debugmode.BadViewMap:
            return

    def test_goodviewmap(self):
        goodop = self.BadAddRef()
        goodop.view_map = {0: [1]}
        x = theano.tensor.dvector()
        y = theano.tensor.dvector()
        f = theano.function([x, y], goodop(x,y), mode='DEBUG_MODE')
        try:
            f([1,5,1], [3,4,2,1,4])
            return
        except debugmode.BadViewMap:
            assert False #failed to raise error


    def test_badviewmap_c(self):
        x = theano.tensor.dvector()
        f = theano.function([x], wb1i(x), mode=debugmode.DebugMode(check_py_code=False))
        try:
            f([1,2])
            assert False #failed to raise error
        except debugmode.BadViewMap:
            pass

    def test_aliased_outputs_ok(self):
        #here aliased outputs is ok because they are both aliased to an input as well
        class CustomOp(gof.Op):
            view_map = {0:[0], 1:[0]}
            def make_node(self, a, b):
                c = a.type()
                d = a.type()
                return gof.Apply(self, [a,b], [c,d])
            def perform(self, node, (a,b), (c,d)):
                c[0] = a
                d[0] = a[1:]

        x = theano.tensor.dvector('x')
        y = theano.tensor.dvector('y')
        f = theano.function([x, y], CustomOp()(x,y), mode='DEBUG_MODE')

        r0, r1 = f([1,2,3,4],[5,6,7,8])

        assert numpy.all(r0 == [1,2,3,4])
        assert numpy.all(r1 == [2,3,4])

    def test_aliased_outputs_ok_output(self):
        # here aliased outputs is ok because they are both outputs of the function as a whole and
        # thus not destroy-able
        class CustomOp(gof.Op):
            def make_node(self, a, b):
                c = a.type()
                d = a.type()
                return gof.Apply(self, [a,b], [c,d])
            def perform(self, node, (a,b), (c,d)):
                r = a * 2
                c[0] = r
                d[0] = r[1:]

        x = theano.tensor.dvector()
        y = theano.tensor.dvector()
        f = theano.function([x, y], CustomOp()(x,y), mode='DEBUG_MODE')

        r0, r1 = f([1,2,3,4],[5,6,7,8])

        assert numpy.all(r0 == [2,4,6,8])
        assert numpy.all(r1 == [4,6,8])

    def test_aliased_outputs_ok_shadow(self):
        # here the alias between outputs is ok because one of them is not used for subsequent
        # computation.  This is like the case where we use one output as a memory buffer to serve
        # another output.
        class CustomOp(gof.Op):
            def make_node(self, a, b):
                c = a.type()
                d = a.type()
                return gof.Apply(self, [a,b], [c,d])
            def perform(self, node, (a,b), (c,d)):
                r = a * 1
                c[0] = r
                d[0] = r[1:]

        x = theano.tensor.dvector('x')
        y = theano.tensor.dvector('y')
        f = theano.function([x, y], CustomOp()(x,y)[0] * 2, mode='DEBUG_MODE')

        r0 = f([1,2,3,4],[5,6,7,8])

        assert numpy.all(r0 == [2,4,6,8])


    def test_aliased_outputs_bad(self):
        # here the alias between outputs is not ok because destroying one destroys the other, but
        # there's no way to warn theano about it through the view_map mechanism.
        class CustomOp(gof.Op):
            def make_node(self, a, b):
                c = a.type()
                d = a.type()
                return gof.Apply(self, [a,b], [c,d])
            def perform(self, node, (a,b), (c,d)):
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
            r0 = f([1,2,3,4],[5,6,7,8])
            assert False # DebugMode should have caught the error
        except debugmode.BadViewMap, e:
            print e
            pass

        # the situation can be rescued by picking one of the inputs and pretending that it is
        # aliased to both the outputs.  This unfairly disables any destructive operations on the
        # input, but guarantees correctness.
        #custom_op.view_map = {0:[0], 1:[1]}
        #f([1,2,3,4],[5,6,7,8])

class Test_check_isfinite(unittest.TestCase):
    def setUp(self):
        self.old_ts = theano.tensor.TensorType.filter_checks_isfinite
        self.old_dm = theano.compile.mode.predefined_modes['DEBUG_MODE'].check_isfinite
    def tearDown(self):
        theano.tensor.TensorType.filter_checks_isfinite = self.old_ts
        theano.compile.mode.predefined_modes['DEBUG_MODE'].check_isfinite = self.old_dm

    def test_check_isfinite(self):
        x = theano.tensor.vector()
        f = theano.function([x], (x+2) * 5, mode='DEBUG_MODE')
        g = theano.function([x], theano.tensor.log(x), mode='DEBUG_MODE')

        # this should work
        f(numpy.log([3, 4, 5]).astype(config.floatX))

        # if TensorType.filter_checks_isfinite were true, these would raise ValueError
        # if not, DebugMode will check internally, and raise InvalidValueError
        # passing an invalid value as an input should trigger ValueError
        self.failUnlessRaises(debugmode.InvalidValueError, f,
                numpy.log([3, -4, 5]).astype(config.floatX))
        self.failUnlessRaises(debugmode.InvalidValueError, f,
                (numpy.asarray([0, 1.0, 0])/0).astype(config.floatX))
        self.failUnlessRaises(debugmode.InvalidValueError, f,
                (numpy.asarray([1.0, 1.0, 1.0])/0).astype(config.floatX))

        # generating an invalid value internally should trigger InvalidValueError
        self.failUnlessRaises(debugmode.InvalidValueError, g,
                numpy.asarray([3,-4,5], dtype=config.floatX))

        # this should disable the exception
        theano.tensor.TensorType.filter_checks_isfinite = False
        theano.compile.mode.predefined_modes['DEBUG_MODE'].check_isfinite = False
        # insert several Inf
        f(numpy.asarray(numpy.asarray([1.0, 1.0, 1.0])/0, dtype=config.floatX))


    def test_check_isfinite_disabled(self):
        x = theano.tensor.dvector()
        f = theano.function([x], (x+2) * 5, mode=debugmode.DebugMode(check_isfinite=False))

        #nan should go through
        f(numpy.log([3, -4, 5]))

        #inf should go through
        infs = numpy.asarray([1.0,1.,1.])/0
        print infs
        f(infs)
        return


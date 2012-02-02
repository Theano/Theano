import cPickle, time, unittest
from itertools import imap

from numpy.testing import dec

from theano.gof import Variable, Op
from theano import gof

from theano.scalar import *

from theano import tensor
from theano.compile.mode import get_default_mode
from theano.tensor.elemwise import *
from theano.tests import unittest_tools


def Env(i, o):
    e = gof.Env(i, o)
    return e

class test_DimShuffle(unittest.TestCase):

    def with_linker(self, linker):
        for xsh, shuffle, zsh in [((2, 3), (1, 'x', 0), (3, 1, 2)),
                                  ((1, 2, 3), (1, 2), (2, 3)),
                                  ((1, 2, 1, 3), (1, 3), (2, 3)),
                                  ((2, 3, 4), (2, 1, 0), (4, 3, 2)),
                                  ((2, 3, 4), ('x', 2, 1, 0, 'x'), (1, 4, 3, 2, 1)),
                                  ((1, 4, 3, 2, 1), (3, 2, 1), (2, 3, 4)),
                                  ((1, 1, 4), (1, 2), (1, 4)),
                                  ((1, 1, 1), (), ()),
                                  ((1,), ('x', 'x'), (1, 1)),]:
            ib = [(entry == 1) for entry in xsh]
            x = TensorType('float64', ib)('x')
            e = DimShuffle(ib, shuffle)(x)
            f = copy(linker).accept(Env([x], [e])).make_function()
            assert f(numpy.ones(xsh)).shape == zsh
            #test that DimShuffle.infer_shape work correctly
            x = TensorType('float64', ib)('x')
            e = DimShuffle(ib, shuffle)(x)
            f = copy(linker).accept(Env([x], [e.shape])).make_function()
            assert all(f(numpy.ones(xsh))) == all(zsh)

        # Test when we drop a axis that is not broadcastable
        ib = [False, True, False]
        x = TensorType('float64', ib)('x')
        self.assertRaises(ValueError, DimShuffle, ib, shuffle)

        # Test when we drop a axis that don't have shape 1
        ib = [True, True, False]
        x = TensorType('float64', ib)('x')
        e = DimShuffle(ib, (1, 2))(x)
        f = copy(linker).accept(Env([x], [e.shape])).make_function()
        self.assertRaises(TypeError, f, numpy.ones((2, 1, 4)))

        # Test that we can't take a dimensions multiple time
        xsh, shuffle, zsh = ((1, 1, 4), (0, 1, 2, 0), (1, 4))
        ib = [False, True, False]
        x = TensorType('float64', ib)('x')
        self.assertRaises(ValueError, DimShuffle, ib, shuffle)

    def test_perform(self):
        self.with_linker(gof.PerformLinker())

    def test_c_or_py(self):
        # Shape op don't have C code.
        # But This will test DimShuffle c code
        self.with_linker(gof.OpWiseCLinker())

class test_Broadcast(unittest.TestCase):
    def setUp(self):
        unittest_tools.seed_rng()

    def with_linker(self, linker):
        for xsh, ysh in [((3, 5), (3, 5)),
                         ((3, 5), (1, 5)),
                         ((3, 5), (3, 1)),
                         ((1, 5), (5, 1)),
                         ((1, 1), (1, 1)),
                         ((2, 3, 4, 5), (2, 3, 4, 5)),
                         ((2, 3, 4, 5), (1, 3, 1, 5)),
                         ((2, 3, 4, 5), (1, 1, 1, 1)),
                         ((), ())]:
            x = TensorType('float64', [(entry == 1) for entry in xsh])('x')
            y = TensorType('float64', [(entry == 1) for entry in ysh])('y')
            e = Elemwise(add)(x, y)
            f = copy(linker).accept(Env([x, y], [e])).make_function()
            xv = numpy.asarray(numpy.random.rand(*xsh))
            yv = numpy.asarray(numpy.random.rand(*ysh))
            zv = xv + yv

            self.assertTrue((f(xv, yv) == zv).all())

            #test Elemwise.infer_shape
            #the Shape op don't implement c_code!
            if isinstance(linker,gof.PerformLinker):
                x = TensorType('float64', [(entry == 1) for entry in xsh])('x')
                y = TensorType('float64', [(entry == 1) for entry in ysh])('y')
                e = Elemwise(add)(x, y)
                f = copy(linker).accept(Env([x, y], [e.shape])).make_function()
                assert tuple(f(xv, yv))==tuple(zv.shape)

    def with_linker_inplace(self, linker):
        for xsh, ysh in [((5, 5), (5, 5)),
                         ((5, 5), (1, 5)),
                         ((5, 5), (5, 1)),
                         ((1, 1), (1, 1)),
                         ((2, 3, 4, 5), (2, 3, 4, 5)),
                         ((2, 3, 4, 5), (1, 3, 1, 5)),
                         ((2, 3, 4, 5), (1, 1, 1, 1)),
                         ((), ())]:
            x = TensorType('float64', [(entry == 1) for entry in xsh])('x')
            y = TensorType('float64', [(entry == 1) for entry in ysh])('y')
            e = Elemwise(Add(transfer_type(0)), {0:0})(x, y)
            f = copy(linker).accept(Env([x, y], [e])).make_function()
            xv = numpy.asarray(numpy.random.rand(*xsh))
            yv = numpy.asarray(numpy.random.rand(*ysh))
            zv = xv + yv

            f(xv, yv)

            self.assertTrue((xv == zv).all())
            #test Elemwise.infer_shape
            #the Shape op don't implement c_code!
            if isinstance(linker,gof.PerformLinker):
                x = TensorType('float64', [(entry == 1) for entry in xsh])('x')
                y = TensorType('float64', [(entry == 1) for entry in ysh])('y')
                e = Elemwise(Add(transfer_type(0)), {0:0})(x, y)
                f = copy(linker).accept(Env([x, y], [e.shape])).make_function()
                xv = numpy.asarray(numpy.random.rand(*xsh))
                yv = numpy.asarray(numpy.random.rand(*ysh))
                zv = xv + yv

                f(xv, yv)

                assert xv.shape==zv.shape

    def test_perform(self):
        self.with_linker(gof.PerformLinker())

    def test_c(self):
        self.with_linker(gof.CLinker())

    def test_perform_inplace(self):
        self.with_linker_inplace(gof.PerformLinker())

    def test_c_inplace(self):
        self.with_linker_inplace(gof.CLinker())

    def test_fill(self):
        x = TensorType('float64', [0, 0])('x')
        y = TensorType('float64', [1, 1])('y')
        e = Elemwise(Second(transfer_type(0)), {0:0})(x, y)
        f = gof.CLinker().accept(Env([x, y], [e])).make_function()
        xv = numpy.ones((5, 5))
        yv = numpy.random.rand(1, 1)
        f(xv, yv)
        assert (xv == yv).all()

    def test_weird_strides(self):
        x = TensorType('float64', [0, 0, 0, 0, 0])('x')
        y = TensorType('float64', [0, 0, 0, 0, 0])('y')
        e = Elemwise(add)(x, y)
        f = gof.CLinker().accept(Env([x, y], [e])).make_function()
        xv = numpy.random.rand(2, 2, 2, 2, 2)
        yv = numpy.random.rand(2, 2, 2, 2, 2).transpose(4, 0, 3, 1, 2)
        zv = xv + yv
        assert (f(xv, yv) == zv).all()

    def test_same_inputs(self):
        x = TensorType('float64', [0, 0])('x')
        e = Elemwise(add)(x, x)
        f = gof.CLinker().accept(Env([x], [e])).make_function()
        xv = numpy.random.rand(2, 2)
        zv = xv + xv
        assert (f(xv) == zv).all()


class test_CAReduce(unittest.TestCase):
    def setUp(self):
        unittest_tools.seed_rng()

    def with_linker(self, linker, scalar_op = add, dtype="floatX",
                    test_nan=False):
        for xsh, tosum in [((5, 6), None),
                           ((5, 6), (0, 1)),
                           ((5, 6), (0, )),
                           ((5, 6), (1, )),
                           ((5, 6), (-1, )),
                           ((5, 6), (-2, )),
                           ((5, 6), ()),
                           ((2, 3, 4, 5), (0, 1, 3)),
                           ((2, 3, 4, 5), (-2, -3)),
                           ((5, 0), None),
                           ((5, 0), (0, )),
                           ((5, 0), (1, )),
                           ((5, 0), ()),
                           ((), None),
                           ((), ())]:
            if dtype == "floatX":
                dtype = theano.config.floatX
            x = TensorType(dtype, [(entry == 1) for entry in xsh])('x')
            e = CAReduce(scalar_op, axis = tosum)(x)
            if tosum is None: tosum = range(len(xsh))
            f = copy(linker).accept(Env([x], [e])).make_function()
            xv = numpy.asarray(numpy.random.rand(*xsh))

            if not "int" in dtype:
                xv = numpy.asarray(xv,dtype=dtype)
            else:
                xv = numpy.asarray(xv<0.5,dtype=dtype)

            if test_nan and xv.size > 0:
                if len(xsh)>0:
                    xv = xv.flatten()
                    xv[0] = numpy.nan
                    xv = xv.reshape(*xsh)
                else:
                    xv = numpy.asarray(numpy.nan, dtype=dtype)
            zv = xv
            numpy_raised = False
            if len(tosum)>1 and any([a<0 for a in tosum]):
                #In that case, we need to use the good order of axis in the reduction.
                axis2 = []
                for a in tosum:
                    if a<0: axis2.append(a+len(xsh))
                    else: axis2.append(a)
                assert len(axis2)==len(tosum)
                tosum = tuple(axis2)

            if scalar_op == add:
                for axis in reversed(sorted(tosum)):
                    zv = numpy.add.reduce(zv, axis)
            elif scalar_op == mul:
                for axis in reversed(sorted(tosum)):
                    zv = numpy.multiply.reduce(zv, axis)
            elif scalar_op == maximum:
                try:
                    for axis in reversed(sorted(tosum)):
                        zv = numpy.maximum.reduce(zv, axis)
                except ValueError:
                    numpy_raised=True
            elif scalar_op == minimum:
                try:
                    for axis in reversed(sorted(tosum)):
                        zv = numpy.minimum.reduce(zv, axis)
                except ValueError:
                    numpy_raised=True
            elif scalar_op == or_:
                for axis in reversed(sorted(tosum)):
                    zv = numpy.bitwise_or.reduce(zv, axis)
            elif scalar_op == and_:
                for axis in reversed(sorted(tosum)):
                    zv = numpy.bitwise_and.reduce(zv, axis)
            elif scalar_op == xor:
                # There is no identity value for the xor function
                # So we can't support shape of dimensions 0.
                if numpy.prod(zv.shape)==0:
                    continue
                for axis in reversed(sorted(tosum)):
                    zv = numpy.bitwise_xor.reduce(zv, axis)
            else:
                raise Exception("Test for CAReduce with scalar_op %s not implemented"%str(scalar_op))
            if scalar_op in [maximum,minimum] and numpy_raised:
                try:
                    out = f(xv)
                    assert out.dtype == dtype
                except ValueError:
                    pass
                else:
                    self.fail()
            else:
                #numpy.{all,any} return bool type.
                if scalar_op in [and_, or_]:
                    zv = numpy.asarray(zv, dtype=dtype)
                if test_nan:
                    self.assertTrue(theano.tensor.TensorType.values_eq(f(xv), zv), (f(xv), zv))
                else:
                    self.assertTrue(numpy.allclose(f(xv), zv), (f(xv), zv))


            #test CAReduce.infer_shape
            #the Shape op don't implement c_code!
            if isinstance(linker,gof.PerformLinker):
                x = TensorType(dtype, [(entry == 1) for entry in xsh])('x')
                e = CAReduce(scalar_op, axis = tosum)(x)
                if tosum is None: tosum = range(len(xsh))
                f = copy(linker).accept(Env([x], [e.shape])).make_function()
                if not(scalar_op in [maximum,minimum] and ((xsh==() or numpy.prod(xsh)==0))):
                    assert all(f(xv)== zv.shape)

    def test_perform(self):
        for dtype in ["floatX", "complex64", "complex128", "int8", "uint8"]:
            self.with_linker(gof.PerformLinker(), add, dtype=dtype)
            self.with_linker(gof.PerformLinker(), mul, dtype=dtype)
            self.with_linker(gof.PerformLinker(), maximum, dtype=dtype)
            self.with_linker(gof.PerformLinker(), minimum, dtype=dtype)
        for dtype in ["int8", "uint8"]:
            self.with_linker(gof.PerformLinker(), or_, dtype=dtype)
            self.with_linker(gof.PerformLinker(), and_, dtype=dtype)
            self.with_linker(gof.PerformLinker(), xor, dtype=dtype)

    @dec.knownfailureif(
        True,
        ("When there is nan in the input of CAReduce, we don't have a good output. "))
    def test_perform_nan(self):
        for dtype in ["floatX", "complex64", "complex128"]:
            self.with_linker(gof.PerformLinker(), add, dtype=dtype,
                             test_nan=True)
            self.with_linker(gof.PerformLinker(), mul, dtype=dtype,
                             test_nan=True)
            self.with_linker(gof.PerformLinker(), maximum, dtype=dtype,
                             test_nan=True)
            self.with_linker(gof.PerformLinker(), minimum, dtype=dtype,
                             test_nan=True)
            self.with_linker(gof.PerformLinker(), or_, dtype=dtype,
                             test_nan=True)
            self.with_linker(gof.PerformLinker(), and_, dtype=dtype,
                             test_nan=True)

    def test_c(self):
        for dtype in ["floatX", "complex64", "complex128", "int8", "uint8"]:
            self.with_linker(gof.CLinker(), add, dtype=dtype)
            self.with_linker(gof.CLinker(), mul, dtype=dtype)
        for dtype in ["floatX", "int8", "uint8"]:
            self.with_linker(gof.CLinker(), minimum, dtype=dtype)
            self.with_linker(gof.CLinker(), maximum, dtype=dtype)
        for dtype in ["int8", "uint8"]:
            self.with_linker(gof.CLinker(), or_, dtype=dtype)
            self.with_linker(gof.CLinker(), and_, dtype=dtype)
            self.with_linker(gof.CLinker(), xor, dtype=dtype)

    @dec.knownfailureif(
        True,
        ("When there is nan in the input of CAReduce, we don't have a good output. "))
    def test_c_nan(self):
        for dtype in ["floatX", "complex64", "complex128"]:
            self.with_linker(gof.CLinker(), add, dtype=dtype,
                             test_nan=True)
            self.with_linker(gof.CLinker(), mul, dtype=dtype,
                             test_nan=True)
        for dtype in ["floatX"]:
            self.with_linker(gof.CLinker(), minimum, dtype=dtype,
                             test_nan=True)
            self.with_linker(gof.CLinker(), maximum, dtype=dtype,
                             test_nan=True)


class test_Prod(unittest.TestCase):
    def setUp(self):
        unittest_tools.seed_rng()

        # we want to allow nans in the matrices, so we disable this DEBUG_MODE check
        mode = theano.compile.mode.get_default_mode()
        mode = copy(mode)
        mode.check_isfinite = False

        self.mode = mode

    def test_verify_grad(self):

        # including zeros, as the case with zeros is important
        # (and special cases: 1 zero in the row, more than 1 zero in the row)
        x_val = numpy.asarray([[1,2,3],[4,5,6],[7,8,9]], dtype='float32')
        x = theano.tensor.dmatrix()
        # now with verify_grad
        unittest_tools.verify_grad(Prod(axis=1), [x_val], mode=self.mode)

        # second time, with some added complexity
        # verify_grad takes the sum of the matrices anyway
        def fn(x2):
            return theano.tensor.sqr(Prod(axis=1)(x2))

        unittest_tools.verify_grad(fn, [x_val], mode=self.mode)


    def test_verify_grad_with_zeros(self):
        # including zeros, as the case with zeros is important
        # (and special cases: 1 zero in the row, more than 1 zero in the row)
        x_val = numpy.asarray([[1.,2.,3.],[0.,5.,6.],[0.,0.,9.]], dtype='float32')
        x = theano.tensor.dmatrix()

        # sanity check
        x2 = theano.tensor.dmatrix()
        p = Prod(axis=1)(x)
        p2 = Prod(axis=1)(x2)
        fn = theano.function([x,x2],[p-p2], mode=self.mode)
        #print "hand computed diff for each row"
        x2_val = numpy.asarray([[1., 2., 3.003], [0.003,5.,6], [0.,0.,9.01]])
        #print fn(x_val, x2_val)
        fn2 = theano.function([x],[theano.tensor.grad(p.sum(),x)], mode=self.mode)
        #print "real grad"
        #print fn2(x_val)
        fn3 = theano.function([x],[p], mode=self.mode)
        assert numpy.allclose(fn3(x_val), [6.,0.,0.])

        # now with verify_grad
        unittest_tools.verify_grad(Prod(axis=1), [x_val], mode=self.mode)

        # second time, with some added complexity
        # verify_grad takes the sum of the matrices anyway
        #def fn5(x5):
        #    return theano.tensor.sqr(Prod(axis=1)(x5))

        #x4 = theano.tensor.dmatrix()
        #p4 = theano.tensor.sqr(Prod(axis=1)(x4))
        #fn4 = theano.function([x4], p4)
        #print "with sqr"
        #print fn4(x_val)
        #print fn4(x2_val)

        #unittest_tools.verify_grad(fn5, [x_val])

    def test_prod_without_zeros(self):
        x = theano.tensor.dmatrix()
        x_val = numpy.array([[1,2,3],[0,5,6],[0,0,9]], dtype='float32')
        pwz = ProdWithoutZeros(axis=1)(x)
        fn = theano.function([x], pwz, mode=self.mode)
        assert numpy.allclose(fn(x_val), [6,30,9])

        pwz_a0 = ProdWithoutZeros(axis=0)(x)
        fn_a0 = theano.function([x], pwz_a0, mode=self.mode)
        assert numpy.allclose(fn_a0(x_val), [1, 10, 162])

    def test_other_grad_tests(self):
        x = theano.tensor.dmatrix()
        x_val1 = numpy.array([[1,2,3],[0,5,6],[0,0,9]], dtype='float32')
        x_val2 = numpy.array([[1,2,0],[0,5,6],[7,8,9],[9,10,0]], dtype='float32')
        rng = rng = numpy.random.RandomState(43)

        p = Prod(axis=1)
        grad_p = theano.tensor.grad(p(x).sum(), x)
        grad_fn = theano.function([x], grad_p, mode=self.mode)
        assert numpy.allclose(grad_fn(x_val1), [[6.,3.,2.],[30.,0.,0.],[0.,0.,0.]])
        assert numpy.allclose(grad_fn(x_val2), [[0., 0., 2.], [30., 0., 0.], [72., 63., 56.], [0., 0., 90.]])

        p_axis0 = Prod(axis=0)
        grad_p_axis0 = theano.tensor.grad(p_axis0(x).sum(), x)
        grad_fn_axis0 = theano.function([x], grad_p_axis0, mode=self.mode)
        assert numpy.allclose(grad_fn_axis0(x_val2), [[0., 400., 0.],[63., 160., 0.], [0., 100., 0.], [0., 80., 0.]])

        tensor.verify_grad(p, [x_val1], rng=rng, mode=self.mode)

    def test_mul_without_zeros_zeros(self):
        a = numpy.zeros((3,3))

        x = theano.tensor.dmatrix()

        mul1 = ProdWithoutZeros(axis=0)(x)

        fn_debug = theano.function([x], mul1, mode=self.mode)

        fn_debug(a)

    def test_pickle_bug(self):
        # Regression test for bug fixed in 24d4fd291054.
        o = Prod()
        s = cPickle.dumps(o, protocol=-1)
        o = cPickle.loads(s)
        cPickle.dumps(o)


class test_IsInf_IsNan(unittest.TestCase):

    def setUp(self):
        self.test_vals = [numpy.array(x, dtype=config.floatX) for x in [
            0,
            1,
            numpy.nan,
            numpy.inf,
            -numpy.inf,
            [numpy.nan, numpy.inf, -numpy.inf, 0, 1, -1],
            ]]
        self.scalar = tensor.scalar()
        self.vector = tensor.vector()
        self.mode = get_default_mode()
        if isinstance(self.mode, theano.compile.debugmode.DebugMode):
            # Disable the check preventing usage of NaN / Inf values.
            self.mode = copy(self.mode)
            self.mode.check_isfinite = False

    def run_isfunc(self, isfunc):
        for input in (self.scalar, self.vector):
            theano_isfunc = theano.function([input],
                                            getattr(tensor, isfunc)(input),
                                            mode=self.mode)
            numpy_isfunc = getattr(numpy, isfunc)
            for x in self.test_vals:
                if ((x.ndim == 0 and input is not self.scalar) or
                    (x.ndim == 1 and input is not self.vector)):
                    # We only test with the appropriate input type.
                    continue
                assert (theano_isfunc(x) == numpy_isfunc(x)).all()

    def test_isinf(self):
        return self.run_isfunc('isinf')

    def test_isnan(self):
        return self.run_isfunc('isnan')


class T_sum_dtype(unittest.TestCase):
    def test_sum_default_dtype(self):
        """
        Test the default dtype of a sum().
        """
        # We try multiple axis combinations even though axis should not matter.
        axes = [None, 0, 1, [0], [1], [0, 1]]
        for idx, dtype in enumerate(imap(str, theano.scalar.all_types)):
            axis = axes[idx % len(axes)]
            x = tensor.matrix(dtype=dtype).sum(axis=axis)
            assert x.dtype == dict(
                    int8='int64',
                    int16='int64',
                    int32='int64',
                    uint8='uint64',
                    uint16='uint64',
                    uint32='uint64',
                    ).get(dtype, dtype)

    def test_sum_custom_dtype(self):
        """
        Test the ability to provide your own output dtype for a sum.
        """
        # We try multiple axis combinations even though axis should not matter.
        axes = [None, 0, 1, [0], [1], [0, 1]]
        idx = 0
        for input_dtype in imap(str, theano.scalar.all_types):
            x = tensor.matrix(dtype=input_dtype)
            for output_dtype in imap(str, theano.scalar.all_types):
                axis = axes[idx % len(axes)]
                # If output_dtype would force a downcast, we expect a TypeError
                # We always allow int/uint inputs with float/complex outputs.
                upcasted_dtype = scalar.upcast(input_dtype, output_dtype)
                if (output_dtype == upcasted_dtype or
                        (input_dtype in tensor.discrete_dtypes and
                            output_dtype in tensor.continuous_dtypes)
                        ):
                    sum_var = x.sum(dtype=output_dtype, axis=axis)
                    assert sum_var.dtype == output_dtype

                    # Check that we can take the gradient
                    grad_var = tensor.grad(sum_var.sum(), x,
                            disconnected_inputs='ignore')
                else:
                    self.assertRaises(TypeError,
                            x.sum, dtype=output_dtype, axis=axis)

                idx += 1

class T_mean_dtype(unittest.TestCase):
    def test_mean_default_dtype(self):
        """
        Test the default dtype of a mean().
        """
        # We try multiple axis combinations even though axis should not matter.
        axes = [None, 0, 1, [0], [1], [0, 1]]
        for idx, dtype in enumerate(imap(str, theano.scalar.all_types)):
            axis = axes[idx % len(axes)]
            x = tensor.matrix(dtype=dtype).mean(axis=axis)
            if dtype in tensor.discrete_dtypes:
                assert x.dtype == 'float64'
            else:
                assert x.dtype == dtype, (x, x.dtype, dtype)

    def test_mean_custom_dtype(self):
        """
        Test the ability to provide your own output dtype for a mean.
        """
        # We try multiple axis combinations even though axis should not matter.
        axes = [None, 0, 1, [0], [1], [0, 1]]
        idx = 0
        for input_dtype in imap(str, theano.scalar.all_types):
            x = tensor.matrix(dtype=input_dtype)
            for sum_dtype in imap(str, theano.scalar.all_types):
                axis = axes[idx % len(axes)]
                # If the inner sum cannot be created, it will raise a TypeError.
                try:
                    mean_var = x.mean(dtype=sum_dtype, axis=axis)
                except TypeError:
                    pass
                else:
                    # Executed if no TypeError was raised
                    if sum_dtype in tensor.discrete_dtypes:
                        assert mean_var.dtype == 'float64', (mean_var.dtype, sum_dtype)
                    else:
                        assert mean_var.dtype == sum_dtype, (mean_var.dtype, output_dtype)

                    # Check that we can take the gradient, when implemented
                    try:
                        grad_var = tensor.grad(mean_var.sum(), x,
                                disconnected_inputs='ignore')
                    except NotImplementedError:
                        # TrueDiv does not seem to have a gradient when
                        # the numerator is complex.
                        if mean_var.dtype in tensor.complex_dtypes:
                            pass
                        else:
                            raise

                idx += 1

class T_prod_dtype(unittest.TestCase):
    def test_prod_default_dtype(self):
        """
        Test the default dtype of a prod().
        """
        # We try multiple axis combinations even though axis should not matter.
        axes = [None, 0, 1, [0], [1], [0, 1]]
        for idx, dtype in enumerate(imap(str, theano.scalar.all_types)):
            axis = axes[idx % len(axes)]
            x = tensor.matrix(dtype=dtype).prod(axis=axis)
            assert x.dtype == dict(
                    int8='int64',
                    int16='int64',
                    int32='int64',
                    uint8='uint64',
                    uint16='uint64',
                    uint32='uint64',
                    ).get(dtype, dtype)

    def test_prod_custom_dtype(self):
        """
        Test the ability to provide your own output dtype for a prod.
        """
        # We try multiple axis combinations even though axis should not matter.
        axes = [None, 0, 1, [0], [1], [0, 1]]
        idx = 0
        for input_dtype in imap(str, theano.scalar.all_types):
            x = tensor.matrix(dtype=input_dtype)
            for output_dtype in imap(str, theano.scalar.all_types):
                axis = axes[idx % len(axes)]
                # If output_dtype would force a downcast, we expect a TypeError
                # We always allow int/uint inputs with float/complex outputs.
                upcasted_dtype = scalar.upcast(input_dtype, output_dtype)
                if (output_dtype == upcasted_dtype or
                        (input_dtype in tensor.discrete_dtypes and
                            output_dtype in tensor.continuous_dtypes)
                        ):
                    prod_var = x.prod(dtype=output_dtype, axis=axis)
                    assert prod_var.dtype == output_dtype

                    # Check that we can take the gradient
                    grad_var = tensor.grad(prod_var.sum(), x,
                            disconnected_inputs='ignore')
                else:
                    self.assertRaises(TypeError,
                            x.prod, dtype=output_dtype, axis=axis)

                idx += 1

class T_prod_without_zeros_dtype(unittest.TestCase):
    def test_prod_without_zeros_default_dtype(self):
        """
        Test the default dtype of a ProdWithoutZeros().
        """
        # We try multiple axis combinations even though axis should not matter.
        axes = [None, 0, 1, [0], [1], [0, 1]]
        for idx, dtype in enumerate(imap(str, theano.scalar.all_types)):
            axis = axes[idx % len(axes)]
            x = ProdWithoutZeros(axis=axis)(tensor.matrix(dtype=dtype))
            assert x.dtype == dict(
                    int8='int64',
                    int16='int64',
                    int32='int64',
                    uint8='uint64',
                    uint16='uint64',
                    uint32='uint64',
                    ).get(dtype, dtype)

    def test_prod_without_zeros_custom_dtype(self):
        """
        Test the ability to provide your own output dtype for a ProdWithoutZeros().
        """
        # We try multiple axis combinations even though axis should not matter.
        axes = [None, 0, 1, [0], [1], [0, 1]]
        idx = 0
        for input_dtype in imap(str, theano.scalar.all_types):
            x = tensor.matrix(dtype=input_dtype)
            for output_dtype in imap(str, theano.scalar.all_types):
                axis = axes[idx % len(axes)]
                # If output_dtype would force a downcast, we expect a TypeError
                # We always allow int/uint inputs with float/complex outputs.
                upcasted_dtype = scalar.upcast(input_dtype, output_dtype)
                if (output_dtype == upcasted_dtype or
                        (input_dtype in tensor.discrete_dtypes and
                            output_dtype in tensor.continuous_dtypes)
                        ):
                    prod_woz_var = ProdWithoutZeros(
                            axis=axis, dtype=output_dtype)(x)
                    assert prod_woz_var.dtype == output_dtype
                else:
                    self.assertRaises(TypeError,
                            ProdWithoutZeros(axis=axis, dtype=output_dtype),
                            x)

                idx += 1

if __name__ == '__main__':
    #unittest.main()
    suite = unittest.TestSuite([test_Prod('test_mul_without_zeros_zeros')])
    #suite.addTest(test_Prod('test_verify_grad_with_zeros'))
    #suite.addTest(test_Prod('test_prod_without_zeros'))
    #suite.addTest(test_Prod('test_other_grad_tests'))
    unittest.TextTestRunner().run(suite)

"""
WRITE ME

Tests for the R operator / L operator

For the list of op with r op defined, with or without missing test
see this file: doc/library/tensor/basic.txt

For function to automatically test your Rop implementation, look at
the docstring of the functions: check_mat_rop_lop, check_rop_lop,
check_nondiff_rop,
"""
from __future__ import absolute_import, print_function, division
import unittest
from theano.tests import unittest_tools as utt
from theano import function
import theano
from theano import tensor
import itertools
import numpy as np
from theano.gof import Op, Apply
from theano.gradient import grad_undefined
from theano.tests.unittest_tools import SkipTest
from theano.tensor.signal.pool import Pool
from theano.tensor.nnet import conv, conv2d

'''
Special Op created to test what happens when you have one op that is not
differentiable in the computational graph
'''


class BreakRop(Op):
    """
    @note: Non-differentiable.
    """
    __props__ = ()

    def make_node(self, x):
        return Apply(self, [x], [x.type()])

    def perform(self, node, inp, out_):
        x, = inp
        out, = out_
        out[0] = x

    def grad(self, inp, grads):
        return [grad_undefined(self, 0, inp[0])]

    def R_op(self, inputs, eval_points):
        return [None]

break_op = BreakRop()


class RopLop_checker(unittest.TestCase):
    """
    Don't peform any test, but provide the function to test the
    Rop to class that inherit from it.
    """
    def setUp(self):
        utt.seed_rng()
        # Using vectors make things a lot simpler for generating the same
        # computations using scan
        self.x = tensor.vector('x')
        self.v = tensor.vector('v')
        self.rng = np.random.RandomState(utt.fetch_seed())
        self.in_shape = (5 + self.rng.randint(3),)
        self.mx = tensor.matrix('mx')
        self.mv = tensor.matrix('mv')
        self.mat_in_shape = (5 + self.rng.randint(3),
                             5 + self.rng.randint(3))

    def check_nondiff_rop(self, y):
        """
        If your op is not differentiable(so you can't define Rop)
        test that an error is raised.
        """
        raised = False
        try:
            tensor.Rop(y, self.x, self.v)
        except ValueError:
            raised = True
        if not raised:
            self.fail((
                'Op did not raise an error even though the function'
                ' is not differentiable'))

    def check_mat_rop_lop(self, y, out_shape):
        """
        Test the Rop/Lop when input is a matrix and the output is a vector

        :param y: the output variable of the op applied to self.mx
        :param out_shape: Used to generate a random tensor
                          corresponding to the evaluation point of the Rop
                          (i.e. the tensor with which you multiply the
                          Jacobian). It should be a tuple of ints.

        If the Op has more than 1 input, one of them must be mx, while
        others must be shared variables / constants. We will test only
        against the input self.mx, so you must call
        check_mat_rop_lop/check_rop_lop for the other inputs.

        We expect all inputs/outputs have dtype floatX.

        If you want to test an Op with an output matrix, add a sum
        after the Op you want to test.
        """
        vx = np.asarray(self.rng.uniform(size=self.mat_in_shape),
                        theano.config.floatX)
        vv = np.asarray(self.rng.uniform(size=self.mat_in_shape),
                        theano.config.floatX)
        yv = tensor.Rop(y, self.mx, self.mv)
        rop_f = function([self.mx, self.mv], yv, on_unused_input='ignore')
        sy, _ = theano.scan(lambda i, y, x, v:
                            (tensor.grad(y[i], x) * v).sum(),
                            sequences=tensor.arange(y.shape[0]),
                            non_sequences=[y, self.mx, self.mv])
        scan_f = function([self.mx, self.mv], sy, on_unused_input='ignore')

        v1 = rop_f(vx, vv)
        v2 = scan_f(vx, vv)

        assert np.allclose(v1, v2), ('ROP mismatch: %s %s' % (v1, v2))

        self.check_nondiff_rop(theano.clone(y, replace={self.mx: break_op(self.mx)}))

        vv = np.asarray(self.rng.uniform(size=out_shape), theano.config.floatX)
        yv = tensor.Lop(y, self.mx, self.v)
        lop_f = function([self.mx, self.v], yv)

        sy = tensor.grad((self.v * y).sum(), self.mx)
        scan_f = function([self.mx, self.v], sy)

        v1 = lop_f(vx, vv)
        v2 = scan_f(vx, vv)
        assert np.allclose(v1, v2), ('LOP mismatch: %s %s' % (v1, v2))

    def check_rop_lop(self, y, out_shape):
        """
        As check_mat_rop_lop, except the input is self.x which is a
        vector. The output is still a vector.
        """
        # TEST ROP
        vx = np.asarray(self.rng.uniform(size=self.in_shape),
                        theano.config.floatX)
        vv = np.asarray(self.rng.uniform(size=self.in_shape),
                        theano.config.floatX)

        yv = tensor.Rop(y, self.x, self.v)
        rop_f = function([self.x, self.v], yv, on_unused_input='ignore')
        J, _ = theano.scan(lambda i, y, x: tensor.grad(y[i], x),
                           sequences=tensor.arange(y.shape[0]),
                           non_sequences=[y, self.x])
        sy = tensor.dot(J, self.v)

        scan_f = function([self.x, self.v], sy, on_unused_input='ignore')

        v1 = rop_f(vx, vv)
        v2 = scan_f(vx, vv)
        assert np.allclose(v1, v2), ('ROP mismatch: %s %s' % (v1, v2))
        known_fail = False
        try:
            self.check_nondiff_rop(theano.clone(y, replace={self.x: break_op(self.x)}))
        except AssertionError:
            known_fail = True

        # TEST LOP

        vx = np.asarray(self.rng.uniform(size=self.in_shape),
                        theano.config.floatX)
        vv = np.asarray(self.rng.uniform(size=out_shape),
                        theano.config.floatX)

        yv = tensor.Lop(y, self.x, self.v)
        lop_f = function([self.x, self.v], yv, on_unused_input='ignore')
        J, _ = theano.scan(lambda i, y, x: tensor.grad(y[i], x),
                           sequences=tensor.arange(y.shape[0]),
                           non_sequences=[y, self.x])
        sy = tensor.dot(self.v, J)

        scan_f = function([self.x, self.v], sy)

        v1 = lop_f(vx, vv)
        v2 = scan_f(vx, vv)
        assert np.allclose(v1, v2), ('LOP mismatch: %s %s' % (v1, v2))

        if known_fail:
            raise SkipTest('Rop does not handle non-differentiable inputs '
                           'correctly. Bug exposed by fixing Add.grad method.')


class test_RopLop(RopLop_checker):
    def test_shape(self):
        self.check_nondiff_rop(self.x.shape[0])

    def test_specifyshape(self):
        self.check_rop_lop(tensor.specify_shape(self.x, self.in_shape),
                           self.in_shape)

    def test_max(self):
        # If we call max directly, we will return an CAReduce object
        # which doesn't have R_op implemented!
        # self.check_mat_rop_lop(tensor.max(self.mx, axis=[0,1])[0], ())
        self.check_mat_rop_lop(tensor.max(self.mx, axis=0),
                               (self.mat_in_shape[1],))
        self.check_mat_rop_lop(tensor.max(self.mx, axis=1),
                               (self.mat_in_shape[0],))

    def test_argmax(self):
        self.check_nondiff_rop(tensor.argmax(self.mx, axis=1))

    def test_subtensor(self):
        self.check_rop_lop(self.x[:4], (4,))

    def test_incsubtensor1(self):
        tv = np.asarray(self.rng.uniform(size=(3,)),
                        theano.config.floatX)
        t = theano.shared(tv)
        out = tensor.inc_subtensor(self.x[:3], t)
        self.check_rop_lop(out, self.in_shape)

    def test_incsubtensor2(self):
        tv = np.asarray(self.rng.uniform(size=(10,)),
                        theano.config.floatX)
        t = theano.shared(tv)
        out = tensor.inc_subtensor(t[:4], self.x[:4])
        self.check_rop_lop(out, (10,))

    def test_setsubtensor1(self):
        tv = np.asarray(self.rng.uniform(size=(3,)),
                        theano.config.floatX)
        t = theano.shared(tv)
        out = tensor.set_subtensor(self.x[:3], t)
        self.check_rop_lop(out, self.in_shape)

    def test_print(self):
        out = theano.printing.Print('x', attrs=('shape',))(self.x)
        self.check_rop_lop(out, self.in_shape)

    def test_setsubtensor2(self):
        tv = np.asarray(self.rng.uniform(size=(10,)),
                        theano.config.floatX)
        t = theano.shared(tv)
        out = tensor.set_subtensor(t[:4], self.x[:4])
        self.check_rop_lop(out, (10,))

    def test_dimshuffle(self):
        # I need the sum, because the setup expects the output to be a
        # vector
        self.check_rop_lop(self.x[:4].dimshuffle('x', 0).sum(axis=0),
                           (4,))

    def test_rebroadcast(self):
        # I need the sum, because the setup expects the output to be a
        # vector
        self.check_rop_lop(tensor.unbroadcast(
            self.x[:4].dimshuffle('x', 0), 0).sum(axis=1),
            (1,))

    def test_downsample(self):
        rng = np.random.RandomState(utt.fetch_seed())
        # ws, shp
        examples = (
            ((2,), (16,)),
            ((2,), (4, 16,)),
            ((2,), (4, 2, 16,)),
            ((1, 1), (4, 2, 16, 16)),
            ((2, 2), (4, 2, 16, 16)),
            ((3, 3), (4, 2, 16, 16)),
            ((3, 2), (4, 2, 16, 16)),
            ((3, 2, 2), (3, 2, 16, 16, 16)),
            ((2, 3, 2), (3, 2, 16, 16, 16)),
            ((2, 2, 3), (3, 2, 16, 16, 16)),
            ((2, 2, 3, 2), (3, 2, 6, 6, 6, 5)),
        )

        for example, ignore_border in itertools.product(examples, [True, False]):
            (ws, shp) = example
            vx = rng.rand(*shp)
            vex = rng.rand(*shp)

            x = theano.shared(vx)
            ex = theano.shared(vex)

            maxpool_op = Pool(ignore_border, ndim=len(ws))
            a_pooled = maxpool_op(x, ws).flatten()
            yv = tensor.Rop(a_pooled, x, ex)
            mode = None
            if theano.config.mode == "FAST_COMPILE":
                mode = "FAST_RUN"
            rop_f = function([], yv, on_unused_input='ignore', mode=mode)
            sy, _ = theano.scan(lambda i, y, x, v:
                                (tensor.grad(y[i], x) * v).sum(),
                                sequences=tensor.arange(a_pooled.shape[0]),
                                non_sequences=[a_pooled, x, ex],
                                mode=mode)
            scan_f = function([], sy, on_unused_input='ignore', mode=mode)
            v1 = rop_f()
            v2 = scan_f()
            assert np.allclose(v1, v2), ("Rop mismatch: %s %s" % (v1, v2))

    def test_conv(self):
        for conv_op in [conv.conv2d, conv2d]:
            for border_mode in ['valid', 'full']:
                image_shape = (2, 2, 4, 5)
                filter_shape = (2, 2, 2, 3)
                image_dim = len(image_shape)
                filter_dim = len(filter_shape)
                input = tensor.TensorType(
                    theano.config.floatX,
                    [False] * image_dim)(name='input')
                filters = tensor.TensorType(
                    theano.config.floatX,
                    [False] * filter_dim)(name='filter')
                ev_input = tensor.TensorType(
                    theano.config.floatX,
                    [False] * image_dim)(name='ev_input')
                ev_filters = tensor.TensorType(
                    theano.config.floatX,
                    [False] * filter_dim)(name='ev_filters')

                def sym_conv2d(input, filters):
                    return conv_op(input, filters, border_mode=border_mode)
                output = sym_conv2d(input, filters).flatten()
                yv = tensor.Rop(output, [input, filters], [ev_input, ev_filters])
                mode = None
                if theano.config.mode == "FAST_COMPILE":
                    mode = "FAST_RUN"
                rop_f = function([input, filters, ev_input, ev_filters],
                                 yv, on_unused_input='ignore', mode=mode)
                sy, _ = theano.scan(lambda i, y, x1, x2, v1, v2:
                                    (tensor.grad(y[i], x1) * v1).sum() +
                                    (tensor.grad(y[i], x2) * v2).sum(),
                                    sequences=tensor.arange(output.shape[0]),
                                    non_sequences=[output, input, filters,
                                                   ev_input, ev_filters],
                                    mode=mode)
                scan_f = function([input, filters, ev_input, ev_filters], sy,
                                  on_unused_input='ignore', mode=mode)
                dtype = theano.config.floatX
                image_data = np.random.random(image_shape).astype(dtype)
                filter_data = np.random.random(filter_shape).astype(dtype)
                ev_image_data = np.random.random(image_shape).astype(dtype)
                ev_filter_data = np.random.random(filter_shape).astype(dtype)
                v1 = rop_f(image_data, filter_data, ev_image_data, ev_filter_data)
                v2 = scan_f(image_data, filter_data, ev_image_data, ev_filter_data)
                assert np.allclose(v1, v2), ("Rop mismatch: %s %s" % (v1, v2))

    def test_join(self):
        tv = np.asarray(self.rng.uniform(size=(10,)),
                        theano.config.floatX)
        t = theano.shared(tv)
        out = tensor.join(0, self.x, t)
        self.check_rop_lop(out, (self.in_shape[0] + 10,))

    def test_dot(self):
        insh = self.in_shape[0]
        vW = np.asarray(self.rng.uniform(size=(insh, insh)),
                        theano.config.floatX)
        W = theano.shared(vW)
        self.check_rop_lop(tensor.dot(self.x, W), self.in_shape)

    def test_elemwise0(self):
        self.check_rop_lop((self.x + 1) ** 2, self.in_shape)

    def test_elemwise1(self):
        self.check_rop_lop(self.x + tensor.cast(self.x, 'int32'),
                           self.in_shape)

    def test_reshape(self):
        new_shape = tensor.constant(np.asarray([
            self.mat_in_shape[0] * self.mat_in_shape[1]],
            dtype='int64'))

        self.check_mat_rop_lop(self.mx.reshape(new_shape),
                               (self.mat_in_shape[0] * self.mat_in_shape[1],))

    def test_flatten(self):
        self.check_mat_rop_lop(self.mx.flatten(),
                               (self.mat_in_shape[0] * self.mat_in_shape[1],))

    def test_sum(self):
        self.check_mat_rop_lop(self.mx.sum(axis=1), (self.mat_in_shape[0],))

    def test_softmax(self):
        # Softmax adds an extra dimnesion !
        self.check_rop_lop(tensor.nnet.softmax(self.x)[0], self.in_shape[0])

    def test_alloc(self):
        # Alloc of the sum of x into a vector
        out1d = tensor.alloc(self.x.sum(), self.in_shape[0])
        self.check_rop_lop(out1d, self.in_shape[0])

        # Alloc of x into a 3-D tensor, flattened
        out3d = tensor.alloc(self.x, self.mat_in_shape[0], self.mat_in_shape[1], self.in_shape[0])
        self.check_rop_lop(out3d.flatten(), self.mat_in_shape[0] * self.mat_in_shape[1] * self.in_shape[0])

    def test_invalid_input(self):
        success = False

        try:
            tensor.Rop(0., [tensor.matrix()], [tensor.vector()])
            success = True
        except ValueError:
            pass

        assert not success

    def test_multiple_outputs(self):
        m = tensor.matrix('m')
        v = tensor.vector('v')
        m_ = tensor.matrix('m_')
        v_ = tensor.vector('v_')

        mval = self.rng.uniform(size=(3, 7)).astype(theano.config.floatX)
        vval = self.rng.uniform(size=(7,)).astype(theano.config.floatX)
        m_val = self.rng.uniform(size=(3, 7)).astype(theano.config.floatX)
        v_val = self.rng.uniform(size=(7,)).astype(theano.config.floatX)

        rop_out1 = tensor.Rop([m, v, m + v], [m, v], [m_, v_])
        assert isinstance(rop_out1, list)
        assert len(rop_out1) == 3
        rop_out2 = tensor.Rop((m, v, m + v), [m, v], [m_, v_])
        assert isinstance(rop_out2, tuple)
        assert len(rop_out2) == 3

        all_outs = []
        for o in rop_out1, rop_out2:
            all_outs.extend(o)
        f = theano.function([m, v, m_, v_], all_outs)
        f(mval, vval, m_val, v_val)

    def test_Rop_dot_bug_18Oct2013_Jeremiah(self):
        # This test refers to a bug reported by Jeremiah Lowin on 18th Oct
        # 2013. The bug consists when through a dot operation there is only
        # one differentiable path (i.e. there is no gradient wrt to one of
        # the inputs).
        x = tensor.arange(20.0).reshape([1, 20])
        v = theano.shared(np.ones([20]))
        d = tensor.dot(x, v).sum()
        tensor.Rop(tensor.grad(d, v), v, v)

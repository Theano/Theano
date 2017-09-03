"""
 Tests fof the lazy conditiona
"""

from __future__ import absolute_import, print_function, division
import unittest
import numpy as np
from nose.plugins.skip import SkipTest
from six.moves import reduce

import theano
from theano import tensor
import theano.ifelse
from theano.ifelse import IfElse, ifelse
from theano.tests import unittest_tools as utt


__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu ")
__copyright__ = "(c) 2010, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"


class test_ifelse(unittest.TestCase, utt.TestOptimizationMixin):
    mode = None
    dtype = theano.config.floatX
    cast_output = staticmethod(tensor.as_tensor_variable)
    shared = staticmethod(theano.shared)

    def get_ifelse(self, n):
        if theano.config.mode == "FAST_COMPILE":
            return IfElse(n)
        else:
            return IfElse(n, as_view=True)

    def test_lazy_if(self):
        # Tests that lazy if works .. even if the two results have different
        # shapes but the same type (i.e. both vectors, or matrices or
        # whatnot of same dtype)
        x = tensor.vector('x', dtype=self.dtype)
        y = tensor.vector('y', dtype=self.dtype)
        c = tensor.iscalar('c')
        f = theano.function([c, x, y], ifelse(c, x, y), mode=self.mode)
        self.assertFunctionContains1(f, self.get_ifelse(1))
        rng = np.random.RandomState(utt.fetch_seed())

        xlen = rng.randint(200)
        ylen = rng.randint(200)

        vx = np.asarray(rng.uniform(size=(xlen,)), self.dtype)
        vy = np.asarray(rng.uniform(size=(ylen,)), self.dtype)

        assert np.allclose(vx, f(1, vx, vy))
        assert np.allclose(vy, f(0, vx, vy))

    def test_not_lazy_if_inplace(self):
        # Tests that if the outputs are scalars and the graph is big,
        # we disable the inplace opt to speed up optimization
        x = tensor.vector('x', dtype=self.dtype)
        y = tensor.vector('y', dtype=self.dtype)
        c = tensor.iscalar('c')
        mode = theano.compile.get_mode(self.mode).excluding(
            # Disable many opt to keep the graph big enough to disable
            # the opt.
            'fusion', 'local_add_canonizer',
            'inplace', 'constant_folding', 'constant_folding')
        y2 = reduce(lambda x, y: x + y, [y] + list(range(200)))
        f = theano.function([c, x, y], ifelse(c, x, y2), mode=mode)
        # For not inplace ifelse
        ifnode = [n for n in f.maker.fgraph.toposort()
                  if isinstance(n.op, IfElse)]
        assert len(ifnode) == 1
        assert not ifnode[0].op.as_view
        rng = np.random.RandomState(utt.fetch_seed())

        xlen = rng.randint(200)
        ylen = rng.randint(200)

        vx = np.asarray(rng.uniform(size=(xlen,)), self.dtype)
        vy = np.asarray(rng.uniform(size=(ylen,)), self.dtype)

        assert np.allclose(vx, f(1, vx, vy))
        assert np.allclose(vy + sum(range(200)), f(0, vx, vy))

    def test_mixed_dtype(self):
        x1 = tensor.vector('x1', dtype='int32')
        x2 = tensor.vector('x2', dtype=self.dtype)
        y1 = tensor.vector('y1', dtype='int32')
        y2 = tensor.vector('y2', dtype=self.dtype)
        c = tensor.iscalar('c')
        f = theano.function([c, x1, x2, y1, y2],
                            ifelse(c, (x1, x2), (y1, y2)), mode=self.mode)
        self.assertFunctionContains1(f, self.get_ifelse(2))
        rng = np.random.RandomState(utt.fetch_seed())

        xlen = rng.randint(200)
        ylen = rng.randint(200)

        vx1 = np.asarray(rng.uniform(size=(xlen,)) * 3, 'int32')
        vx2 = np.asarray(rng.uniform(size=(xlen,)), self.dtype)
        vy1 = np.asarray(rng.uniform(size=(ylen,)) * 3, 'int32')
        vy2 = np.asarray(rng.uniform(size=(ylen,)), self.dtype)

        o1, o2 = f(1, vx1, vx2, vy1, vy2)
        assert np.allclose(vx1, o1)
        assert np.allclose(vx2, o2)

        o1, o2 = f(0, vx1, vx2, vy1, vy2)
        assert np.allclose(vy1, o1)
        assert np.allclose(vy2, o2)

    def test_lazy_if_on_generics(self):
        x = theano.generic()
        y = theano.generic()
        c = tensor.iscalar('c')
        f = theano.function([c, x, y], ifelse(c, x, y))

        vx = ['testX']
        vy = ['testY']
        assert f(1, vx, vy) == vx
        assert f(0, vx, vy) == vy

    def test_grad_lazy_if(self):
        # Tests that we can compute the gradients through lazy if
        x = tensor.vector('x', dtype=self.dtype)
        y = tensor.vector('y', dtype=self.dtype)
        c = tensor.iscalar('c')
        z = ifelse(c, x, y)
        gx, gy = tensor.grad(z.sum(), [x, y])

        f = theano.function([c, x, y], [self.cast_output(gx),
                                        self.cast_output(gy)],
                            mode=self.mode)
        # There is only 2 of the 3 ifelse that are moved on the GPU.
        # The one that stay on the CPU is for the shape.
        self.assertFunctionContains(f, self.get_ifelse(1), min=2, max=3)
        rng = np.random.RandomState(utt.fetch_seed())

        xlen = rng.randint(200)
        ylen = rng.randint(200)

        vx = np.asarray(rng.uniform(size=(xlen,)), self.dtype)
        vy = np.asarray(rng.uniform(size=(ylen,)), self.dtype)
        gx0, gy0 = f(1, vx, vy)
        assert np.allclose(gx0.shape, vx.shape)
        assert np.allclose(gy0.shape, vy.shape)
        assert np.all(np.asarray(gx0) == 1.)
        assert np.all(np.asarray(gy0) == 0.)

        gx0, gy0 = f(0, vx, vy)
        assert np.allclose(gx0.shape, vx.shape)
        assert np.allclose(gy0.shape, vy.shape)
        assert np.all(np.asarray(gx0) == 0.)
        assert np.all(np.asarray(gy0) == 1.)

    def test_grad_cast_input(self):
        # Tests the gradient when both inputs are on the GPU.
        x = tensor.vector('x', dtype=self.dtype)
        y = tensor.vector('y', dtype=self.dtype)
        c = tensor.iscalar('c')
        z = ifelse(c, self.cast_output(x), self.cast_output(y))
        gx, gy = tensor.grad(z.sum(), [x, y])

        theano.function([c, x, y], [gx, gy],
                        mode=self.mode)

    def test_multiple_out(self):
        x1 = tensor.vector('x1', dtype=self.dtype)
        x2 = tensor.vector('x2', dtype=self.dtype)
        y1 = tensor.vector('y1', dtype=self.dtype)
        y2 = tensor.vector('y2', dtype=self.dtype)
        c = tensor.iscalar('c')
        z = ifelse(c, (x1, x2), (y1, y2))
        f = theano.function([c, x1, x2, y1, y2], z, mode=self.mode)
        self.assertFunctionContains1(f, self.get_ifelse(2))

        ifnode = [x for x in f.maker.fgraph.toposort()
                  if isinstance(x.op, IfElse)][0]
        assert len(ifnode.outputs) == 2

        rng = np.random.RandomState(utt.fetch_seed())

        x1len = rng.randint(200)
        x2len = rng.randint(200)
        y1len = rng.randint(200)
        y2len = rng.randint(200)

        vx1 = np.asarray(rng.uniform(size=(x1len,)), self.dtype)
        vx2 = np.asarray(rng.uniform(size=(x2len,)), self.dtype)
        vy1 = np.asarray(rng.uniform(size=(y1len,)), self.dtype)
        vy2 = np.asarray(rng.uniform(size=(y2len,)), self.dtype)

        ovx1, ovx2 = f(1, vx1, vx2, vy1, vy2)
        ovy1, ovy2 = f(0, vx1, vx2, vy1, vy2)
        assert np.allclose(vx1, ovx1)
        assert np.allclose(vy1, ovy1)
        assert np.allclose(vx2, ovx2)
        assert np.allclose(vy2, ovy2)

    def test_multiple_out_grad(self):
        # Tests that we can compute the gradients through lazy if
        x1 = tensor.vector('x1')
        x2 = tensor.vector('x2')
        y1 = tensor.vector('y1')
        y2 = tensor.vector('y2')
        c = tensor.iscalar('c')
        z = ifelse(c, (x1, x2), (y1, y2))
        grads = tensor.grad(z[0].sum() + z[1].sum(),
                            [x1, x2, y1, y2])

        f = theano.function([c, x1, x2, y1, y2], grads)
        rng = np.random.RandomState(utt.fetch_seed())

        lens = [rng.randint(200) for i in range(4)]
        values = [np.asarray(rng.uniform(size=(l,)), theano.config.floatX)
                  for l in lens]
        outs_1 = f(1, *values)
        assert all([x.shape[0] == y for x, y in zip(outs_1, lens)])
        assert np.all(outs_1[0] == 1.)
        assert np.all(outs_1[1] == 1.)
        assert np.all(outs_1[2] == 0.)
        assert np.all(outs_1[3] == 0.)

        outs_0 = f(0, *values)
        assert all([x.shape[0] == y for x, y in zip(outs_1, lens)])
        assert np.all(outs_0[0] == 0.)
        assert np.all(outs_0[1] == 0.)
        assert np.all(outs_0[2] == 1.)
        assert np.all(outs_0[3] == 1.)

    def test_multiple_out_crash(self):
        # This test failed up to commit 2faeb62c38
        p0 = self.shared(np.asarray(np.random.random([4, 8]),
                                    dtype=self.dtype))
        p1 = self.shared(np.asarray(np.random.random(8),
                                    dtype=self.dtype))
        p2 = self.shared(np.asarray(np.random.random([8, 3]),
                                    dtype=self.dtype))
        p3 = self.shared(np.asarray(np.random.random(3),
                                    dtype=self.dtype))
        p = [p0, p1, p2, p3]

        # in my code these vars are the result of applying scan
        ften0 = tensor.tensor3('ft0', dtype=self.dtype)
        fmat1 = tensor.matrix('fm1', dtype=self.dtype)
        ften2 = tensor.tensor3('ft2', dtype=self.dtype)
        fmat3 = tensor.matrix('fm3', dtype=self.dtype)

        # then I keep only the last iteration
        fsub0 = ften0[-1]
        fsub1 = fmat1[-1]
        fsub2 = ften2[-1]
        fsub3 = fmat3[-1]

        fsub = [fsub0, fsub1, fsub2, fsub3]

        acc = theano.tensor.constant(1, 'int8') >= 0

        new_positions = theano.ifelse.ifelse(acc, fsub, p)

        new_updates = [(p[0], new_positions[0])]

        f = theano.function([ften0, fmat1, ften2, fmat3], [],
                            updates=new_updates, mode=self.mode)
        self.assertFunctionContains1(f, self.get_ifelse(4))

        i1 = np.asarray(np.random.random([19, 4, 8]), dtype=self.dtype)
        i2 = np.asarray(np.random.random([19, 8]), dtype=self.dtype)
        i3 = np.asarray(np.random.random([19, 8, 3]), dtype=self.dtype)
        i4 = np.asarray(np.random.random([19, 3]), dtype=self.dtype)

        f(i1, i2, i3, i4)

    def test_dtype_mismatch(self):
        rng = np.random.RandomState(utt.fetch_seed())
        data = rng.rand(5).astype(self.dtype)
        x = self.shared(data)
        y = tensor.cast(x * 10, 'int8')
        cond = theano.tensor.iscalar('cond')

        self.assertRaises(TypeError, ifelse, cond, x, y)
        self.assertRaises(TypeError, ifelse, cond, y, x)

    def test_ndim_mismatch(self):
        rng = np.random.RandomState(utt.fetch_seed())
        data = rng.rand(5).astype(self.dtype)
        x = self.shared(data)
        y = tensor.col('y', self.dtype)
        cond = theano.tensor.iscalar('cond')

        self.assertRaises(TypeError, ifelse, cond, x, y)
        self.assertRaises(TypeError, ifelse, cond, y, x)

    def test_broadcast_mismatch(self):
        rng = np.random.RandomState(utt.fetch_seed())
        data = rng.rand(5).astype(self.dtype)
        x = self.shared(data)
        # print x.broadcastable
        y = tensor.row('y', self.dtype)
        # print y.broadcastable
        cond = theano.tensor.iscalar('cond')

        self.assertRaises(TypeError, ifelse, cond, x, y)
        self.assertRaises(TypeError, ifelse, cond, y, x)

    def test_sparse_tensor_error(self):
        import theano.sparse
        if not theano.sparse.enable_sparse:
            raise SkipTest("Optimization temporarily disabled")
        rng = np.random.RandomState(utt.fetch_seed())
        data = rng.rand(2, 3).astype(self.dtype)
        x = self.shared(data)
        y = theano.sparse.matrix('csc', dtype=self.dtype, name='y')
        z = theano.sparse.matrix('csr', dtype=self.dtype, name='z')
        cond = theano.tensor.iscalar('cond')

        self.assertRaises(TypeError, ifelse, cond, x, y)
        self.assertRaises(TypeError, ifelse, cond, y, x)
        self.assertRaises(TypeError, ifelse, cond, x, z)
        self.assertRaises(TypeError, ifelse, cond, z, x)
        self.assertRaises(TypeError, ifelse, cond, y, z)
        self.assertRaises(TypeError, ifelse, cond, z, y)

    def test_merge(self):
        raise SkipTest("Optimization temporarily disabled")
        x = tensor.vector('x')
        y = tensor.vector('y')
        c = tensor.iscalar('c')
        z1 = ifelse(c, x + 1, y + 1)
        z2 = ifelse(c, x + 2, y + 2)
        z = z1 + z2
        f = theano.function([c, x, y], z)
        assert len([n for n in f.maker.fgraph.toposort()
                    if isinstance(n.op, IfElse)]) == 1

    def test_remove_useless_inputs1(self):
        raise SkipTest("Optimization temporarily disabled")
        x = tensor.vector('x')
        y = tensor.vector('y')
        c = tensor.iscalar('c')
        z = ifelse(c, (x, x), (y, y))
        f = theano.function([c, x, y], z)

        ifnode = [n for n in f.maker.fgraph.toposort()
                  if isinstance(n.op, IfElse)][0]
        assert len(ifnode.inputs) == 3

    def test_remove_useless_inputs2(self):
        raise SkipTest("Optimization temporarily disabled")
        x1 = tensor.vector('x1')
        x2 = tensor.vector('x2')
        y1 = tensor.vector('y1')
        y2 = tensor.vector('y2')
        c = tensor.iscalar('c')
        z = ifelse(c, (x1, x1, x1, x2, x2), (y1, y1, y2, y2, y2))
        f = theano.function([c, x1, x2, y1, y2], z)

        ifnode = [x for x in f.maker.fgraph.toposort()
                  if isinstance(x.op, IfElse)][0]
        assert len(ifnode.outputs) == 3

    def test_pushout1(self):
        raise SkipTest("Optimization temporarily disabled")
        x1 = tensor.scalar('x1')
        x2 = tensor.scalar('x2')
        y1 = tensor.scalar('y1')
        y2 = tensor.scalar('y2')
        w1 = tensor.scalar('w1')
        w2 = tensor.scalar('w2')
        c = tensor.iscalar('c')
        x, y = ifelse(c, (x1, y1), (x2, y2), name='f1')
        z = ifelse(c, w1, w2, name='f2')
        out = x * z * y

        f = theano.function([x1, x2, y1, y2, w1, w2, c], out,
                            allow_input_downcast=True)
        assert isinstance(f.maker.fgraph.toposort()[-1].op, IfElse)
        rng = np.random.RandomState(utt.fetch_seed())
        vx1 = rng.uniform()
        vx2 = rng.uniform()
        vy1 = rng.uniform()
        vy2 = rng.uniform()
        vw1 = rng.uniform()
        vw2 = rng.uniform()

        assert np.allclose(f(vx1, vx2, vy1, vy2, vw1, vw2, 1),
                           vx1 * vy1 * vw1)
        assert np.allclose(f(vx1, vx2, vy1, vy2, vw1, vw2, 0),
                           vx2 * vy2 * vw2)

    def test_pushout3(self):
        raise SkipTest("Optimization temporarily disabled")
        x1 = tensor.scalar('x1')
        y1 = tensor.scalar('x2')
        y2 = tensor.scalar('y2')
        c = tensor.iscalar('c')
        two = np.asarray(2, dtype=theano.config.floatX)
        x, y = ifelse(c, (x1, y1), (two, y2), name='f1')
        o3 = np.asarray(0.3, dtype=theano.config.floatX)
        o2 = np.asarray(0.2, dtype=theano.config.floatX)
        z = ifelse(c, o3, o2, name='f2')
        out = x * z * y

        f = theano.function([x1, y1, y2, c], out,
                            allow_input_downcast=True)
        assert isinstance(f.maker.fgraph.toposort()[-1].op, IfElse)
        rng = np.random.RandomState(utt.fetch_seed())
        vx1 = rng.uniform()
        vy1 = rng.uniform()
        vy2 = rng.uniform()

        assert np.allclose(f(vx1, vy1, vy2, 1), vx1 * vy1 * 0.3)
        assert np.allclose(f(vx1, vy1, vy2, 0), 2 * vy2 * 0.2)

    def test_pushout2(self):
        raise SkipTest("Optimization temporarily disabled")
        x1 = tensor.scalar('x1')
        x2 = tensor.scalar('x2')
        y1 = tensor.scalar('y1')
        y2 = tensor.scalar('y2')
        w1 = tensor.scalar('w1')
        w2 = tensor.scalar('w2')
        c = tensor.iscalar('c')
        x, y = ifelse(c, (x1, y1), (x2, y2), name='f1')
        z = ifelse(x > y, w1, w2, name='f2')
        out = x * z * y

        f = theano.function([x1, x2, y1, y2, w1, w2, c], out,
                            allow_input_downcast=True)
        assert isinstance(f.maker.fgraph.toposort()[-1].op, IfElse)
        rng = np.random.RandomState(utt.fetch_seed())
        vx1 = rng.uniform()
        vx2 = rng.uniform()
        vy1 = rng.uniform()
        vy2 = rng.uniform()
        vw1 = rng.uniform()
        vw2 = rng.uniform()
        if vx1 > vy1:
            vw = vw1
        else:
            vw = vw2
        assert np.allclose(f(vx1, vx2, vy1, vy2, vw1, vw2, 1),
                           vx1 * vy1 * vw)

        if vx2 > vy2:
            vw = vw1
        else:
            vw = vw2
        assert np.allclose(f(vx1, vx2, vy1, vy2, vw1, vw2, 0),
                           vx2 * vy2 * vw)

    def test_merge_ifs_true_false(self):
        raise SkipTest("Optimization temporarily disabled")
        x1 = tensor.scalar('x1')
        x2 = tensor.scalar('x2')
        y1 = tensor.scalar('y1')
        y2 = tensor.scalar('y2')
        w1 = tensor.scalar('w1')
        w2 = tensor.scalar('w2')
        c = tensor.iscalar('c')

        out = ifelse(c,
                     ifelse(c, x1, x2) + ifelse(c, y1, y2) + w1,
                     ifelse(c, x1, x2) + ifelse(c, y1, y2) + w2)
        f = theano.function([x1, x2, y1, y2, w1, w2, c], out,
                            allow_input_downcast=True)
        assert len([x for x in f.maker.fgraph.toposort()
                    if isinstance(x.op, IfElse)]) == 1

        rng = np.random.RandomState(utt.fetch_seed())
        vx1 = rng.uniform()
        vx2 = rng.uniform()
        vy1 = rng.uniform()
        vy2 = rng.uniform()
        vw1 = rng.uniform()
        vw2 = rng.uniform()
        assert np.allclose(f(vx1, vx2, vy1, vy2, vw1, vw2, 1),
                           vx1 + vy1 + vw1)
        assert np.allclose(f(vx1, vx2, vy1, vy2, vw1, vw2, 0),
                           vx2 + vy2 + vw2)

    def test_grad_test_values(self):
        # Regression test for test values of `ifelse` gradient.

        backup = theano.config.compute_test_value
        theano.config.compute_test_value = 'raise'
        try:
            x = tensor.scalar('x')
            x.tag.test_value = 1
            # Used to crash due to undefined test value.
            tensor.grad(ifelse(0, x, x), x)
        finally:
            theano.config.compute_test_value = backup

    def test_grad_int_value(self):
        w = theano.shared(np.random.rand(10))
        b = theano.shared(np.random.rand())
        params = [w, b]

        x = tensor.vector()
        y = tensor.scalar()

        score = w.dot(x) + b
        correct = (score * y > 0)

        loss = ifelse(correct, 0, 1)
        [(param, param - 0.5 * tensor.grad(cost=loss, wrt=param))
         for param in params]


if __name__ == '__main__':
    print(' Use nosetests to run these tests ')

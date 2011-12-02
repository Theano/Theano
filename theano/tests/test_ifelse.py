"""
 Tests fof the lazy conditiona
"""

__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu ")
__copyright__ = "(c) 2010, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

import unittest
import numpy
from nose.plugins.skip import SkipTest

import theano
from theano import tensor
from theano.ifelse import IfElse, ifelse
from theano.tests  import unittest_tools as utt


class test_ifelse(unittest.TestCase):
    def test_lazy_if(self):
        # Tests that lazy if works .. even if the two results have different
        # shapes but the same type (i.e. both vectors, or matrices or
        # whatnot of same dtype)
        x = tensor.vector('x')
        y = tensor.vector('y')
        c = tensor.iscalar('c')
        f = theano.function([c, x, y], ifelse(c, x, y))
        rng = numpy.random.RandomState(utt.fetch_seed())

        xlen = rng.randint(200)
        ylen = rng.randint(200)

        vx = numpy.asarray(rng.uniform(size=(xlen,)), theano.config.floatX)
        vy = numpy.asarray(rng.uniform(size=(ylen,)), theano.config.floatX)

        assert numpy.allclose(vx, f(1, vx, vy))
        assert numpy.allclose(vy, f(0, vx, vy))

    def test_lazy_if_inplace(self):
        # Tests that lazy if works inplace
        x = tensor.vector('x')
        y = tensor.vector('y')
        c = tensor.iscalar('c')
        f = theano.function([c, x, y], ifelse(c, x, y))
        rng = numpy.random.RandomState(utt.fetch_seed())

        xlen = rng.randint(200)
        ylen = rng.randint(200)

        vx = numpy.asarray(rng.uniform(size=(xlen,)), theano.config.floatX)
        vy = numpy.asarray(rng.uniform(size=(ylen,)), theano.config.floatX)
        if theano.config.mode != "FAST_COMPILE":
            assert numpy.all([x.op.as_view for x in f.maker.env.toposort() if
                              isinstance(x.op, IfElse)])
        assert len([x.op for x in f.maker.env.toposort()
                   if isinstance(x.op, IfElse)]) > 0
        assert numpy.allclose(vx, f(1, vx, vy))
        assert numpy.allclose(vy, f(0, vx, vy))

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
        x = tensor.vector('x')
        y = tensor.vector('y')
        c = tensor.iscalar('c')
        z = ifelse(c, x, y)
        gx, gy = tensor.grad(z.sum(), [x, y])

        f = theano.function([c, x, y], [gx, gy])
        rng = numpy.random.RandomState(utt.fetch_seed())

        xlen = rng.randint(200)
        ylen = rng.randint(200)

        vx = numpy.asarray(rng.uniform(size=(xlen,)), theano.config.floatX)
        vy = numpy.asarray(rng.uniform(size=(ylen,)), theano.config.floatX)
        gx0, gy0 = f(1, vx, vy)
        assert numpy.allclose(gx0.shape, vx.shape)
        assert numpy.allclose(gy0.shape, vy.shape)
        assert numpy.all(gx0 == 1.)
        assert numpy.all(gy0 == 0.)

        gx0, gy0 = f(0, vx, vy)
        assert numpy.allclose(gx0.shape, vx.shape)
        assert numpy.allclose(gy0.shape, vy.shape)
        assert numpy.all(gx0 == 0.)
        assert numpy.all(gy0 == 1.)

    def test_merge(self):
        raise SkipTest("Optimization temporarily disabled")
        x = tensor.vector('x')
        y = tensor.vector('y')
        c = tensor.iscalar('c')
        z1 = ifelse(c, x + 1, y + 1)
        z2 = ifelse(c, x + 2, y + 2)
        z = z1 + z2
        f = theano.function([c, x, y], z)
        assert len([x for x in f.maker.env.toposort()
                    if isinstance(x.op, IfElse)]) == 1

    def test_remove_useless_inputs1(self):
        raise SkipTest("Optimization temporarily disabled")
        x = tensor.vector('x')
        y = tensor.vector('y')
        c = tensor.iscalar('c')
        z = ifelse(c, (x, x), (y, y))
        f = theano.function([c, x, y], z)

        ifnode = [x for x in f.maker.env.toposort()
                  if isinstance(x.op, IfElse)][0]
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

        ifnode = [x for x in f.maker.env.toposort()
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
        assert isinstance(f.maker.env.toposort()[-1].op, IfElse)
        rng = numpy.random.RandomState(utt.fetch_seed())
        vx1 = rng.uniform()
        vx2 = rng.uniform()
        vy1 = rng.uniform()
        vy2 = rng.uniform()
        vw1 = rng.uniform()
        vw2 = rng.uniform()

        assert numpy.allclose(f(vx1, vx2, vy1, vy2, vw1, vw2, 1),
                              vx1 * vy1 * vw1)
        assert numpy.allclose(f(vx1, vx2, vy1, vy2, vw1, vw2, 0),
                              vx2 * vy2 * vw2)

    def test_pushout3(self):
        raise SkipTest("Optimization temporarily disabled")
        x1 = tensor.scalar('x1')
        y1 = tensor.scalar('x2')
        y2 = tensor.scalar('y2')
        c = tensor.iscalar('c')
        two = numpy.asarray(2, dtype=theano.config.floatX)
        x, y = ifelse(c, (x1, y1), (two, y2), name='f1')
        o3 = numpy.asarray(0.3, dtype=theano.config.floatX)
        o2 = numpy.asarray(0.2, dtype=theano.config.floatX)
        z = ifelse(c, o3, o2, name='f2')
        out = x * z * y

        f = theano.function([x1, y1, y2, c], out,
                            allow_input_downcast=True)
        assert isinstance(f.maker.env.toposort()[-1].op, IfElse)
        rng = numpy.random.RandomState(utt.fetch_seed())
        vx1 = rng.uniform()
        vy1 = rng.uniform()
        vy2 = rng.uniform()

        assert numpy.allclose(f(vx1, vy1, vy2, 1), vx1 * vy1 * 0.3)
        assert numpy.allclose(f(vx1, vy1, vy2, 0), 2 * vy2 * 0.2)

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
        assert isinstance(f.maker.env.toposort()[-1].op, IfElse)
        rng = numpy.random.RandomState(utt.fetch_seed())
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
        assert numpy.allclose(f(vx1, vx2, vy1, vy2, vw1, vw2, 1),
                              vx1 * vy1 * vw)

        if vx2 > vy2:
            vw = vw1
        else:
            vw = vw2
        assert numpy.allclose(f(vx1, vx2, vy1, vy2, vw1, vw2, 0),
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
        assert len([x for x in f.maker.env.toposort()
                if isinstance(x.op, IfElse)]) == 1

        rng = numpy.random.RandomState(utt.fetch_seed())
        vx1 = rng.uniform()
        vx2 = rng.uniform()
        vy1 = rng.uniform()
        vy2 = rng.uniform()
        vw1 = rng.uniform()
        vw2 = rng.uniform()
        assert numpy.allclose(f(vx1, vx2, vy1, vy2, vw1, vw2, 1),
                              vx1 + vy1 + vw1)
        assert numpy.allclose(f(vx1, vx2, vy1, vy2, vw1, vw2, 0),
                              vx2 + vy2 + vw2)


if __name__ == '__main__':
    print ' Use nosetests to run these tests '

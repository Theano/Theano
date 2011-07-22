"""
 WRITE ME

 Tests for the R operator / L operator

ops without:

    PermuteRowElements
    Tile
    AdvancedSubtensor
    TensorDot
    Outer
    Prod
    MulwithoutZeros
    ProdWithoutZeros
    CAReduce(for max,... done for MaxAndArgmax op)

list of ops that support R-op:
 * with test
    * SpecifyShape
    * MaxAndArgmax
    * Subtensor
    * IncSubtensor set_subtensor too
    * Alloc
    * Dot
    * Elemwise
    * Sum
    * Softmax
    * Shape
    * Join

 * without test
    * Split
    * ARange
    * ScalarFromTensor
    * Rebroadcast
    * Reshape
    * Flatten
    * AdvancedSubtensor1
    * AdvancedIncSubtensor1
    * AdvancedIncSubtensor
    * DimShuffle
    * Scan [ RP: scan has a test in scan_module/tests/test_scan.test_rop ]






"""

import unittest
from theano.tests  import unittest_tools as utt
from theano import function
import theano
import theano.tensor as TT
import numpy
from theano.gof import Op, Apply

'''
Special Op created to test what happens when you have one op that is not
differentiable in the computational graph
'''
class BreakRop(Op):
    """
    @note: Non-differentiable.
    """
    def __hash__(self):
        return hash(type(self))
    def __eq__(self, other):
        return type(self) == type(other)
    def make_node(self, x):
        return Apply(self, [x], [x.type()])
    def perform(self, node, inp, out_):
        x, = inp
        out, = out_
        out[0] = x
    def grad(self, inp, grads):
        return [None]
    def R_op(self, inputs, eval_points):
        return [None]

break_op = BreakRop()


class test_RopLop(unittest.TestCase):

    def setUp(self):
        # Using vectors make things a lot simpler for generating the same
        # computations using scan
        self.x = TT.vector('x')
        self.v = TT.vector('v')
        self.rng  = numpy.random.RandomState(utt.fetch_seed())
        self.in_shape = ( 5+self.rng.randint(30),)
        self.mx = TT.matrix('mx')
        self.mv = TT.matrix('mv')
        self.mat_in_shape = ( 5 + self.rng.randint(30),
                             5+self.rng.randint(30))

    def check_nondiff_rop(self, y):
        raised = False
        try:
            tmp = TT.Rop(y, self.x, self.v)
        except ValueError:
            raised = True
        if not raised:
            self.fail((
                'Op did not raised an error even though the function'
                ' is not differentiable'))

    def check_mat_rop_lop(self, y, out_shape):
        vx = numpy.asarray(self.rng.uniform(size=self.mat_in_shape), theano.config.floatX)
        vv = numpy.asarray(self.rng.uniform(size=self.mat_in_shape), theano.config.floatX)
        yv = TT.Rop(y, self.mx, self.mv)
        rop_f = function([self.mx, self.mv], yv)
        sy, _ = theano.scan( lambda i,y,x,v: (TT.grad(y[i],x)*v).sum(),
                           sequences = TT.arange(y.shape[0]),
                           non_sequences = [y,self.mx,self.mv])
        scan_f = function([self.mx,self.mv], sy)


        v1 = rop_f(vx,vv)
        v2 = scan_f(vx,vv)

        assert numpy.allclose(v1,v2), ('ROP mismatch: %s %s' % (v1, v2))

        self.check_nondiff_rop( theano.clone(y,
                                             replace={self.mx:break_op(self.mx)}))

        vv = numpy.asarray(self.rng.uniform(size=out_shape), theano.config.floatX)
        yv = TT.Lop(y, self.mx, self.v)
        lop_f = function([self.mx, self.v], yv)

        sy = TT.grad((self.v*y).sum(), self.mx)
        scan_f = function([self.mx, self.v], sy)


        v1 = lop_f(vx,vv)
        v2 = scan_f(vx,vv)
        assert numpy.allclose(v1,v2), ('LOP mismatch: %s %s' % (v1, v2))



    def check_rop_lop(self, y, out_shape):
        # TEST ROP
        vx = numpy.asarray(self.rng.uniform(size=self.in_shape), theano.config.floatX)
        vv = numpy.asarray(self.rng.uniform(size=self.in_shape), theano.config.floatX)

        yv = TT.Rop(y,self.x,self.v)
        rop_f = function([self.x,self.v], yv)
        J, _ = theano.scan( lambda i,y,x: TT.grad(y[i],x),
                           sequences = TT.arange(y.shape[0]),
                           non_sequences = [y,self.x])
        sy = TT.dot(J, self.v)

        scan_f = function([self.x,self.v], sy)

        v1 = rop_f(vx,vv)
        v2 = scan_f(vx,vv)
        assert numpy.allclose(v1,v2), ('ROP mismatch: %s %s' % (v1, v2))
        self.check_nondiff_rop( theano.clone(y,
                                             replace={self.x:break_op(self.x)}))

        # TEST LOP

        vx = numpy.asarray(self.rng.uniform(size=self.in_shape), theano.config.floatX)
        vv = numpy.asarray(self.rng.uniform(size=out_shape), theano.config.floatX)

        yv = TT.Lop(y,self.x,self.v)
        lop_f = function([self.x,self.v], yv)
        J, _ = theano.scan( lambda i,y,x: TT.grad(y[i],x),
                           sequences = TT.arange(y.shape[0]),
                           non_sequences = [y,self.x])
        sy = TT.dot(self.v, J)

        scan_f = function([self.x,self.v], sy)

        v1 = lop_f(vx,vv)
        v2 = scan_f(vx,vv)
        assert numpy.allclose(v1,v2), ('LOP mismatch: %s %s' % (v1, v2))


    def test_shape(self):
        self.check_nondiff_rop( self.x.shape[0])

    def test_specifyshape(self):
        self.check_rop_lop(TT.specify_shape(self.x, self.in_shape),
                           self.in_shape)


    def test_max(self):
        ## If we call max directly, we will return an CAReduce object
        ## and he don't have R_op implemented!
        #self.check_mat_rop_lop(TT.max(self.mx, axis=[0,1])[0],
        #                       ())
        self.check_mat_rop_lop(TT.max(self.mx, axis=0),
                               (self.mat_in_shape[1],))
        self.check_mat_rop_lop(TT.max(self.mx, axis=1),
                               (self.mat_in_shape[0],))

    def test_argmax(self):
        self.check_nondiff_rop(TT.argmax(self.mx,axis=1))

    def test_subtensor(self):
        self.check_rop_lop(self.x[:4], (4,))

    def test_incsubtensor1(self):
        tv = numpy.asarray( self.rng.uniform(size=(3,)),
                           theano.config.floatX)
        t = theano.shared(tv)
        out = TT.inc_subtensor(self.x[:3], t)
        self.check_rop_lop(out, self.in_shape)


    def test_incsubtensor2(self):
        tv = numpy.asarray( self.rng.uniform(size=(10,)),
                           theano.config.floatX)
        t = theano.shared(tv)
        out = TT.inc_subtensor(t[:4], self.x[:4])
        self.check_rop_lop(out, (10,))


    def test_setsubtensor1(self):
        tv = numpy.asarray( self.rng.uniform(size=(3,)),
                           theano.config.floatX)
        t = theano.shared(tv)
        out = TT.set_subtensor(self.x[:3], t)
        self.check_rop_lop(out, self.in_shape)


    def test_setsubtensor2(self):
        tv = numpy.asarray( self.rng.uniform(size=(10,)),
                           theano.config.floatX)
        t = theano.shared(tv)
        out = TT.set_subtensor(t[:4], self.x[:4])
        self.check_rop_lop(out, (10,))


    def test_join(self):
        tv = numpy.asarray( self.rng.uniform(size=(10,)),
                           theano.config.floatX)
        t = theano.shared(tv)
        out = TT.join(0, self.x, t)
        self.check_rop_lop(out, (self.in_shape[0]+10,))

    def test_dot(self):
        insh = self.in_shape[0]
        vW   = numpy.asarray(self.rng.uniform(size=(insh,insh)),
                           theano.config.floatX)
        W = theano.shared(vW)
        self.check_rop_lop( TT.dot(self.x, W), self.in_shape)


    def test_elemwise0(self):
        self.check_rop_lop( (self.x+1)**2, self.in_shape)

    def test_elemwise1(self):
        self.check_rop_lop( self.x+TT.cast(self.x, 'int32'),
                           self.in_shape)

    def test_sum(self):
        self.check_mat_rop_lop(self.mx.sum(axis=1), (self.mat_in_shape[0],))


    def test_softmax(self):
        # Softmax adds an extra dimnesion !
        self.check_rop_lop( TT.nnet.softmax(self.x)[0], self.in_shape[0])

    def test_alloc(self):
        # Alloc of the sum of x into a vector
        out1d = TT.alloc(self.x.sum(), self.in_shape[0])
        self.check_rop_lop(out1d, self.in_shape[0])

        # Alloc of x into a 3-D tensor, flattened
        out3d = TT.alloc(self.x,
                self.mat_in_shape[0], self.mat_in_shape[1], self.in_shape[0])
        self.check_rop_lop(out3d.flatten(),
                self.mat_in_shape[0] * self.mat_in_shape[1] * self.in_shape[0])

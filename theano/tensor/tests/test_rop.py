"""
 WRITE ME

 Tests for the R operator / L operator

 For the list of op with r op defined, with or without missing test see this file: defined see this file

"""

import unittest
from theano.tests  import unittest_tools as utt
from theano import function
import theano
from theano import tensor
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
        self.x = tensor.vector('x')
        self.v = tensor.vector('v')
        self.rng  = numpy.random.RandomState(utt.fetch_seed())
        self.in_shape = ( 5+self.rng.randint(30),)
        self.mx = tensor.matrix('mx')
        self.mv = tensor.matrix('mv')
        self.mat_in_shape = ( 5 + self.rng.randint(30),
                             5+self.rng.randint(30))

    def check_nondiff_rop(self, y):
        raised = False
        try:
            tmp = tensor.Rop(y, self.x, self.v)
        except ValueError:
            raised = True
        if not raised:
            self.fail((
                'Op did not raised an error even though the function'
                ' is not differentiable'))

    def check_mat_rop_lop(self, y, out_shape):
        vx = numpy.asarray(self.rng.uniform(size=self.mat_in_shape), theano.config.floatX)
        vv = numpy.asarray(self.rng.uniform(size=self.mat_in_shape), theano.config.floatX)
        yv = tensor.Rop(y, self.mx, self.mv)
        rop_f = function([self.mx, self.mv], yv)
        sy, _ = theano.scan( lambda i,y,x,v: (tensor.grad(y[i],x)*v).sum(),
                           sequences = tensor.arange(y.shape[0]),
                           non_sequences = [y,self.mx,self.mv])
        scan_f = function([self.mx,self.mv], sy)


        v1 = rop_f(vx,vv)
        v2 = scan_f(vx,vv)

        assert numpy.allclose(v1,v2), ('ROP mismatch: %s %s' % (v1, v2))

        self.check_nondiff_rop( theano.clone(y,
                                             replace={self.mx:break_op(self.mx)}))

        vv = numpy.asarray(self.rng.uniform(size=out_shape), theano.config.floatX)
        yv = tensor.Lop(y, self.mx, self.v)
        lop_f = function([self.mx, self.v], yv)

        sy = tensor.grad((self.v*y).sum(), self.mx)
        scan_f = function([self.mx, self.v], sy)


        v1 = lop_f(vx,vv)
        v2 = scan_f(vx,vv)
        assert numpy.allclose(v1,v2), ('LOP mismatch: %s %s' % (v1, v2))



    def check_rop_lop(self, y, out_shape):
        # TEST ROP
        vx = numpy.asarray(self.rng.uniform(size=self.in_shape), theano.config.floatX)
        vv = numpy.asarray(self.rng.uniform(size=self.in_shape), theano.config.floatX)

        yv = tensor.Rop(y,self.x,self.v)
        rop_f = function([self.x,self.v], yv)
        J, _ = theano.scan( lambda i,y,x: tensor.grad(y[i],x),
                           sequences = tensor.arange(y.shape[0]),
                           non_sequences = [y,self.x])
        sy = tensor.dot(J, self.v)

        scan_f = function([self.x,self.v], sy)

        v1 = rop_f(vx,vv)
        v2 = scan_f(vx,vv)
        assert numpy.allclose(v1,v2), ('ROP mismatch: %s %s' % (v1, v2))
        self.check_nondiff_rop( theano.clone(y,
                                             replace={self.x:break_op(self.x)}))

        # TEST LOP

        vx = numpy.asarray(self.rng.uniform(size=self.in_shape), theano.config.floatX)
        vv = numpy.asarray(self.rng.uniform(size=out_shape), theano.config.floatX)

        yv = tensor.Lop(y,self.x,self.v)
        lop_f = function([self.x,self.v], yv)
        J, _ = theano.scan( lambda i,y,x: tensor.grad(y[i],x),
                           sequences = tensor.arange(y.shape[0]),
                           non_sequences = [y,self.x])
        sy = tensor.dot(self.v, J)

        scan_f = function([self.x,self.v], sy)

        v1 = lop_f(vx,vv)
        v2 = scan_f(vx,vv)
        assert numpy.allclose(v1,v2), ('LOP mismatch: %s %s' % (v1, v2))


    def test_shape(self):
        self.check_nondiff_rop( self.x.shape[0])

    def test_specifyshape(self):
        self.check_rop_lop(tensor.specify_shape(self.x, self.in_shape),
                           self.in_shape)


    def test_max(self):
        ## If we call max directly, we will return an CAReduce object
        ## and he don't have R_op implemented!
        #self.check_mat_rop_lop(tensor.max(self.mx, axis=[0,1])[0],
        #                       ())
        self.check_mat_rop_lop(tensor.max(self.mx, axis=0),
                               (self.mat_in_shape[1],))
        self.check_mat_rop_lop(tensor.max(self.mx, axis=1),
                               (self.mat_in_shape[0],))

    def test_argmax(self):
        self.check_nondiff_rop(tensor.argmax(self.mx,axis=1))

    def test_subtensor(self):
        self.check_rop_lop(self.x[:4], (4,))

    def test_incsubtensor1(self):
        tv = numpy.asarray( self.rng.uniform(size=(3,)),
                           theano.config.floatX)
        t = theano.shared(tv)
        out = tensor.inc_subtensor(self.x[:3], t)
        self.check_rop_lop(out, self.in_shape)


    def test_incsubtensor2(self):
        tv = numpy.asarray( self.rng.uniform(size=(10,)),
                           theano.config.floatX)
        t = theano.shared(tv)
        out = tensor.inc_subtensor(t[:4], self.x[:4])
        self.check_rop_lop(out, (10,))


    def test_setsubtensor1(self):
        tv = numpy.asarray( self.rng.uniform(size=(3,)),
                           theano.config.floatX)
        t = theano.shared(tv)
        out = tensor.set_subtensor(self.x[:3], t)
        self.check_rop_lop(out, self.in_shape)


    def test_setsubtensor2(self):
        tv = numpy.asarray( self.rng.uniform(size=(10,)),
                           theano.config.floatX)
        t = theano.shared(tv)
        out = tensor.set_subtensor(t[:4], self.x[:4])
        self.check_rop_lop(out, (10,))

    def test_dimshuffle(self):
        # I need the sum, because the setup expects the output to be a
        # vector
        self.check_rop_lop(self.x[:4].dimshuffle('x',0).sum(axis=0),
                           (4,))

    def test_rebroadcast(self):
        # I need the sum, because the setup expects the output to be a
        # vector
        self.check_rop_lop(tensor.unbroadcast(self.x[:4].dimshuffle('x',0),0).sum(axis=1),
                           (1,))


    def test_join(self):
        tv = numpy.asarray( self.rng.uniform(size=(10,)),
                           theano.config.floatX)
        t = theano.shared(tv)
        out = tensor.join(0, self.x, t)
        self.check_rop_lop(out, (self.in_shape[0]+10,))

    def test_dot(self):
        insh = self.in_shape[0]
        vW   = numpy.asarray(self.rng.uniform(size=(insh,insh)),
                           theano.config.floatX)
        W = theano.shared(vW)
        self.check_rop_lop( tensor.dot(self.x, W), self.in_shape)


    def test_elemwise0(self):
        self.check_rop_lop( (self.x+1)**2, self.in_shape)

    def test_elemwise1(self):
        self.check_rop_lop( self.x+tensor.cast(self.x, 'int32'),
                           self.in_shape)




    def test_reshape(self):
        new_shape = tensor.constant( numpy.asarray([
            self.mat_in_shape[0]*self.mat_in_shape[1]],
            dtype = 'int64'))

        self.check_mat_rop_lop(self.mx.reshape(new_shape),
                               (self.mat_in_shape[0]*self.mat_in_shape[1],))

    def test_flatten(self):
        self.check_mat_rop_lop(self.mx.flatten(),
                               (self.mat_in_shape[0]*self.mat_in_shape[1],))

    def test_sum(self):
        self.check_mat_rop_lop(self.mx.sum(axis=1), (self.mat_in_shape[0],))


    def test_softmax(self):
        # Softmax adds an extra dimnesion !
        self.check_rop_lop( tensor.nnet.softmax(self.x)[0], self.in_shape[0])

    def test_alloc(self):
        # Alloc of the sum of x into a vector
        out1d = tensor.alloc(self.x.sum(), self.in_shape[0])
        self.check_rop_lop(out1d, self.in_shape[0])

        # Alloc of x into a 3-D tensor, flattened
        out3d = tensor.alloc(self.x,
                self.mat_in_shape[0], self.mat_in_shape[1], self.in_shape[0])
        self.check_rop_lop(out3d.flatten(),
                self.mat_in_shape[0] * self.mat_in_shape[1] * self.in_shape[0])

    def test_invalid_input(self):
        success = False

        try:
            tensor.Rop(0., [ tensor.matrix() ], [ tensor.vector() ] )
            success = True
        except ValueError:
            pass

        assert not success

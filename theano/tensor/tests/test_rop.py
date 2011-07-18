"""
 WRITE ME

 Tests for the R operator / L operator
"""

import unittest
from theano.tests  import unittest_tools as utt
from theano import function
import theano
import theano.tensor as TT
import numpy

class test_rop(unittest.TestCase):

    def test_specifyshape(self):
        rng  = numpy.random.RandomState(utt.fetch_seed())
        vx = numpy.asarray(rng.uniform(size=(5,)), theano.config.floatX)
        vv = numpy.asarray(rng.uniform(size=(5,)), theano.config.floatX)

        x  = TT.vector('x')
        v  = TT.vector('v')
        y  = TT.specify_shape(x, (5,))
        yv = TT.Rop(y,x,v)
        rop_f = function([x,v], yv)
        J, _ = theano.scan( lambda i,y,x: TT.grad(y[i],x),
                           sequences = TT.arange(x.shape[0]),
                           non_sequences = [y,x])
        sy = TT.dot(J, v)

        scan_f = function([x,v], sy)

        v1 = rop_f(vx,vv)
        v2 = scan_f(vx,vv)
        assert numpy.allclose(v1,v2)


class test_lop(unittest.TestCase):

    def test_specifyshape(self):
        rng  = numpy.random.RandomState(utt.fetch_seed())
        vx = numpy.asarray(rng.uniform(size=(5,)), theano.config.floatX)
        vv = numpy.asarray(rng.uniform(size=(5,)), theano.config.floatX)

        x  = TT.vector('x')
        v  = TT.vector('v')
        y  = TT.specify_shape(x, (5,))
        yv = TT.Lop(y,x,v)
        rop_f = function([x,v], yv)
        J, _ = theano.scan( lambda i,y,x: TT.grad(y[i],x),
                           sequences = TT.arange(x.shape[0]),
                           non_sequences = [y,x])
        sy = TT.dot(v, J)

        scan_f = function([x,v], sy)

        v1 = rop_f(vx,vv)
        v2 = scan_f(vx,vv)
        assert numpy.allclose(v1,v2)



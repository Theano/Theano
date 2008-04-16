from base_tensor import *

import unittest
from copy import copy
from compile import Function
import gof

def _tensor(data, broadcastable=None, name=None):
    """Return a BaseTensor containing given data"""
    data = numpy.asarray(data)
    if broadcastable is None:
        broadcastable = [s==1 for s in data.shape]
    elif broadcastable in [0, 1]:
        broadcastable = [broadcastable] *  len(data.shape)
    rval = BaseTensor(data.dtype, broadcastable, name)
    rval.data = data # will raise if broadcastable was mis-specified
    return rval



class T_tensor(unittest.TestCase):
    def test0(self): # allocate from a scalar float
        t = _tensor(1.0)
        self.failUnless(isinstance(t, BaseTensor))
        self.failUnless(t.dtype == 'float64')
        self.failUnless(t.broadcastable == ())
        self.failUnless(t.role == None)
        self.failUnless(isinstance(t.data, numpy.ndarray))
        self.failUnless(str(t.data.dtype) == 'float64')
        self.failUnless(t.data == 1.0)
    def test0_int(self): # allocate from a scalar float
        t = _tensor(1)
        self.failUnless(isinstance(t, BaseTensor))
        self.failUnless(t.dtype == 'int64' or t.dtype == 'int32')
    def test1(self): # allocate from a vector of ints, not broadcastable
        t = _tensor(numpy.ones(5,dtype='int32'))
        self.failUnless(isinstance(t, BaseTensor))
        self.failUnless(t.dtype == 'int32')
        self.failUnless(t.broadcastable == (0,))
        self.failUnless(isinstance(t.data, numpy.ndarray))
        self.failUnless(str(t.data.dtype) == 'int32')
    def test2(self): # allocate from a column matrix of complex with name
        t = _tensor(numpy.ones((5,1),dtype='complex64'),name='bart')
        self.failUnless(isinstance(t, BaseTensor))
        self.failUnless(t.dtype == 'complex64')
        self.failUnless(t.broadcastable == (0,1))
        self.failUnless(isinstance(t.data, numpy.ndarray))
        self.failUnless(t.name == 'bart')
    def test2b(self): # allocate from a column matrix, not broadcastable
        t = _tensor(numpy.ones((5,1),dtype='complex64'),broadcastable=0)
        self.failUnless(isinstance(t, BaseTensor))
        self.failUnless(t.dtype == 'complex64')
        self.failUnless(t.broadcastable == (0,0))
        self.failUnless(isinstance(t.data, numpy.ndarray))
        f = Function([t], [t], linker_cls=gof.CLinker)
        self.failUnless(numpy.all(t.data == f(t.data)))
    def test_data_normal(self): #test that assigning to .data works when it should
        t = _tensor(numpy.ones((5,1),dtype='complex64'), broadcastable=0)
        o27 = numpy.ones((2,7), dtype='complex64')
        t.data = o27
        lst = t._data
        self.failUnless(t.data.shape == (2,7))
        self.failUnless(t.data is o27)
        self.failUnless(t._data is lst)
    def test_data_badrank0(self):
        t = _tensor(numpy.ones((5,1),dtype='complex64'), broadcastable=0)
        try:
            t.data = numpy.ones((2,7,1))
            self.fail()
        except ValueError, e:
            self.failUnless(e[0] is BaseTensor.filter.E_rank)
        try:
            t.data = numpy.ones(1)
            self.fail()
        except ValueError, e:
            self.failUnless(e[0] is BaseTensor.filter.E_rank)
    def test_data_badrank1(self):
        t = _tensor(numpy.ones((1,1),dtype='complex64'), broadcastable=1)
        try:
            t.data = numpy.ones((1,1,1))
            self.fail()
        except ValueError, e:
            self.failUnless(e[0] is BaseTensor.filter.E_rank)
        try:
            t.data = numpy.ones(1)
            self.fail()
        except ValueError, e:
            self.failUnless(e[0] is BaseTensor.filter.E_rank)
    def test_data_badshape0(self):
        t = _tensor(numpy.ones((1,1),dtype='complex64'), broadcastable=1)
        try:
            t.data = numpy.ones((1,2))
            self.fail()
        except ValueError, e:
            self.failUnless(e[0] is BaseTensor.filter.E_shape)
        try:
            t.data = numpy.ones((0,1))
            self.fail()
        except ValueError, e:
            self.failUnless(e[0] is BaseTensor.filter.E_shape)

    def test_cast0(self):
        t = BaseTensor('float32', [0])
        t.data = numpy.random.rand(4) > 0.5
        self.failUnless(str(t.data.dtype) == t.dtype)

class T_stdlib(unittest.TestCase):
    def test0(self):
        t = _tensor(1.0)
        tt = t.clone(False)
        self.failUnless(t.dtype == tt.dtype)
        self.failUnless(t.broadcastable is tt.broadcastable)
        self.failUnless(tt.data is None)
        self.failUnless(t.data == 1.0)
    def test0b(self):
        t = _tensor(1.0)
        tt = t.clone()
        self.failUnless(t.dtype == tt.dtype)
        self.failUnless(t.broadcastable is tt.broadcastable)
        self.failUnless(tt.data is None)
        self.failUnless(t.data == 1.0)

    def test1(self):
        t = _tensor(1.0)
        tt = t.clone(True)
        self.failUnless(t.dtype == tt.dtype)
        self.failUnless(t.broadcastable is tt.broadcastable)
        self.failUnless(tt.data == 1.0)
        self.failUnless(t.data == 1.0)
        self.failUnless(t.data is not tt.data)
    def test1b(self):
        t = _tensor(1.0)
        tt = copy(t)
        self.failUnless(t.dtype == tt.dtype)
        self.failUnless(t.broadcastable is tt.broadcastable)
        self.failUnless(tt.data == 1.0)
        self.failUnless(t.data == 1.0)
        self.failUnless(t.data is not tt.data)

if __name__ == '__main__':
    unittest.main()


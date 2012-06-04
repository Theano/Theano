import unittest
from theano import tensor, function, Variable, Generic
import numpy

class T_load_tensor(unittest.TestCase):
    def test0(self):
        data = numpy.arange(5, dtype=numpy.int32)
        filename = "_load_tensor_test_1.npy"
        numpy.save(filename, data)
        path = Variable(Generic())
        x = tensor.load(path, 'int32', (False,))
        y = x*2
        fn = function([path], y)
        assert (fn(filename) == data*2).all()
    def test_memmap(self):
        data = numpy.arange(5, dtype=numpy.int32)
        filename = "_load_tensor_test_1.npy"
        numpy.save(filename, data)
        path = Variable(Generic())
        x = tensor.load(path, 'int32', (False,), mmap_mode='r+')
        fn = function([path], x)
        assert type(fn(filename)) == numpy.core.memmap

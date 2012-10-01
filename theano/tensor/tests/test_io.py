import unittest
import theano
from theano import tensor, function, Variable, Generic
import numpy
import os


class T_load_tensor(unittest.TestCase):
    def test0(self):
        data = numpy.arange(5, dtype=numpy.int32)
        filename = os.path.join(
            theano.config.base_compiledir,
            "_test.npy")
        numpy.save(filename, data)
        path = Variable(Generic())
        x = tensor.load(path, 'int32', (False,))
        y = x*2
        fn = function([path], y)
        assert (fn(filename) == data*2).all()

    def test_memmap(self):
        data = numpy.arange(5, dtype=numpy.int32)
        filename = os.path.join(
            theano.config.base_compiledir,
            "_test.npy")
        numpy.save(filename, data)
        path = Variable(Generic())
        x = tensor.load(path, 'int32', (False,), mmap_mode='r+')
        fn = function([path], x)
        assert type(fn(filename)) == numpy.core.memmap

    def tearDown(self):
        os.remove(os.path.join(
            theano.config.base_compiledir,
            "_test.npy"))

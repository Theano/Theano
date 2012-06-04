import unittest
from theano import tensor, function, Variable, Generic
import numpy

class T_load_tensor(unittest.TestCase):
    def test0(self):
        data = numpy.arange(5)
        filename = "_load_tensor_test_1.npz"
        numpy.savez(filename, data)
        path = Variable(Generic())
        x = tensor.load(path, 'int64', (False,))
        y = x*2
        fn = function([path], [y])
        assert (fn(filename) == data*2).all()


from __future__ import absolute_import, print_function, division
import unittest
import theano
from theano import tensor, function, Variable, Generic
import numpy as np
import os


class T_load_tensor(unittest.TestCase):
    def setUp(self):
        self.data = np.arange(5, dtype=np.int32)
        self.filename = os.path.join(
            theano.config.compiledir,
            "_test.npy")
        np.save(self.filename, self.data)

    def test0(self):
        path = Variable(Generic())
        # Not specifying mmap_mode defaults to None, and the data is
        # copied into main memory
        x = tensor.load(path, 'int32', (False,))
        y = x * 2
        fn = function([path], y)
        assert (fn(self.filename) == (self.data * 2)).all()

    def test_invalid_modes(self):
        # Modes 'r+', 'r', and 'w+' cannot work with Theano, becausei
        # the output array may be modified inplace, and that should not
        # modify the original file.
        path = Variable(Generic())
        for mmap_mode in ('r+', 'r', 'w+', 'toto'):
            self.assertRaises(ValueError,
                    tensor.load, path, 'int32', (False,), mmap_mode)

    def test1(self):
        path = Variable(Generic())
        # 'c' means "copy-on-write", which allow the array to be overwritten
        # by an inplace Op in the graph, without modifying the underlying
        # file.
        x = tensor.load(path, 'int32', (False,), 'c')
        # x ** 2 has been chosen because it will work inplace.
        y = (x ** 2).sum()
        fn = function([path], y)
        # Call fn() twice, to check that inplace ops do not cause trouble
        assert (fn(self.filename) == (self.data ** 2).sum()).all()
        assert (fn(self.filename) == (self.data ** 2).sum()).all()

    def test_memmap(self):
        path = Variable(Generic())
        x = tensor.load(path, 'int32', (False,), mmap_mode='c')
        fn = function([path], x)
        assert type(fn(self.filename)) == np.core.memmap

    def tearDown(self):
        os.remove(os.path.join(
            theano.config.compiledir,
            "_test.npy"))

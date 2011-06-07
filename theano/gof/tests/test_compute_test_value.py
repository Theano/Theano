import numpy
import unittest

import theano
from theano import tensor as T


class TestComputeTestValue(unittest.TestCase):

    def test_variable_only(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = True

            x = T.matrix('x')
            x.tag.test_value = numpy.random.rand(3,4)
            y = T.matrix('y')
            y.tag.test_value = numpy.random.rand(4,5)

            # should work
            z = T.dot(x,y)

            # this test should fail
            y.tag.test_value = numpy.random.rand(6,5)
            self.assertRaises(ValueError, T.dot, x, y)
        finally:
            theano.config.compute_test_value = orig_compute_test_value


    def test_compute_flag(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            x = T.matrix('x')
            y = T.matrix('y')
            y.tag.test_value = numpy.random.rand(4,5)

            # should skip computation of test value
            theano.config.compute_test_value = False
            z = T.dot(x,y)

            # should fail one or another when flag is set
            theano.config.compute_test_value = 'warn'
            self.assertRaises(Warning, T.dot, x, y)
            theano.config.compute_test_value = 'err'
            self.assertRaises(ValueError, T.dot, x, y)
        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_string_var(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = True

            x = T.matrix('x')
            x.tag.test_value = numpy.random.rand(3,4)
            y = T.matrix('y')
            y.tag.test_value = numpy.random.rand(4,5)

            z = theano.shared(numpy.random.rand(5,6))

            # should work
            out = T.dot(T.dot(x,y), z)
            def f(x,y,z):
                return T.dot(T.dot(x,y),z)

            # this test should fail
            z.set_value(numpy.random.rand(7,6))
            self.assertRaises(ValueError, f, x, y, z)
        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_shared(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = True

            x = T.matrix('x')
            x.tag.test_value = numpy.random.rand(3,4)
            y = theano.shared(numpy.random.rand(4,6), 'y')

            # should work
            z = T.dot(x,y)

            # this test should fail
            y.set_value(numpy.random.rand(5,6))
            self.assertRaises(ValueError, T.dot, x, y)
        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_ndarray(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = True

            x = numpy.random.rand(2,3)
            y = theano.shared(numpy.random.rand(3,6), 'y')

            # should work
            z = T.dot(x,y)

            # this test should fail
            x = numpy.random.rand(2,4)
            self.assertRaises(ValueError, T.dot, x, y)
        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_constant(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = True

            x = T.constant(numpy.random.rand(2,3))
            y = theano.shared(numpy.random.rand(3,6), 'y')

            # should work
            z = T.dot(x,y)

            # this test should fail
            x = T.constant(numpy.random.rand(2,4))
            self.assertRaises(ValueError, T.dot, x, y)
        finally:
            theano.config.compute_test_value = orig_compute_test_value

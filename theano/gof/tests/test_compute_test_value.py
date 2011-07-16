import os, sys, traceback, warnings

import numpy
import unittest

import theano
from theano import config
from theano import tensor as T
from theano.tensor.basic import _allclose
from theano.scan_module import scan


class TestComputeTestValue(unittest.TestCase):

    def test_variable_only(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = 'raise'

            x = T.matrix('x')
            x.tag.test_value = numpy.random.rand(3,4).astype(config.floatX)
            y = T.matrix('y')
            y.tag.test_value = numpy.random.rand(4,5).astype(config.floatX)

            # should work
            z = T.dot(x,y)
            assert hasattr(z.tag, 'test_value')
            f = theano.function([x,y], z)
            assert _allclose(f(x.tag.test_value, y.tag.test_value),
                             z.tag.test_value)

            # this test should fail
            y.tag.test_value = numpy.random.rand(6,5).astype(config.floatX)
            self.assertRaises(ValueError, T.dot, x, y)
        finally:
            theano.config.compute_test_value = orig_compute_test_value


    def test_compute_flag(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            x = T.matrix('x')
            y = T.matrix('y')
            y.tag.test_value = numpy.random.rand(4,5).astype(config.floatX)

            # should skip computation of test value
            theano.config.compute_test_value = 'off'
            z = T.dot(x,y)
            assert not hasattr(z.tag, 'test_value')

            # should fail when asked by user
            theano.config.compute_test_value = 'raise'
            self.assertRaises(ValueError, T.dot, x, y)

            # test that a warning is raised if required
            theano.config.compute_test_value = 'warn'
            warnings.simplefilter('error', UserWarning)
            try:
                self.assertRaises(UserWarning, T.dot, x, y)
            finally:
                # Restore the default behavior.
                # TODO There is a cleaner way to do this in Python 2.6, once
                # Theano drops support of Python 2.4 and 2.5.
                warnings.simplefilter('default', UserWarning)
        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_string_var(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = 'raise'

            x = T.matrix('x')
            x.tag.test_value = numpy.random.rand(3,4).astype(config.floatX)
            y = T.matrix('y')
            y.tag.test_value = numpy.random.rand(4,5).astype(config.floatX)

            z = theano.shared(numpy.random.rand(5,6).astype(config.floatX))

            # should work
            out = T.dot(T.dot(x,y), z)
            assert hasattr(out.tag, 'test_value')
            tf = theano.function([x,y], out)
            assert _allclose(
                    tf(x.tag.test_value, y.tag.test_value),
                    out.tag.test_value)

            def f(x,y,z):
                return T.dot(T.dot(x,y),z)

            # this test should fail
            z.set_value(numpy.random.rand(7,6).astype(config.floatX))
            self.assertRaises(ValueError, f, x, y, z)
        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_shared(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = 'raise'

            x = T.matrix('x')
            x.tag.test_value = numpy.random.rand(3,4).astype(config.floatX)
            y = theano.shared(numpy.random.rand(4,6).astype(config.floatX), 'y')

            # should work
            z = T.dot(x,y)
            assert hasattr(z.tag, 'test_value')
            f = theano.function([x], z)
            assert _allclose(f(x.tag.test_value), z.tag.test_value)

            # this test should fail
            y.set_value(numpy.random.rand(5,6).astype(config.floatX))
            self.assertRaises(ValueError, T.dot, x, y)
        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_ndarray(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = 'raise'

            x = numpy.random.rand(2,3).astype(config.floatX)
            y = theano.shared(numpy.random.rand(3,6).astype(config.floatX), 'y')

            # should work
            z = T.dot(x,y)
            assert hasattr(z.tag, 'test_value')
            f = theano.function([], z)
            assert _allclose(f(), z.tag.test_value)

            # this test should fail
            x = numpy.random.rand(2,4).astype(config.floatX)
            self.assertRaises(ValueError, T.dot, x, y)
        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_constant(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = 'raise'

            x = T.constant(numpy.random.rand(2,3), dtype=config.floatX)
            y = theano.shared(numpy.random.rand(3,6).astype(config.floatX), 'y')

            # should work
            z = T.dot(x,y)
            assert hasattr(z.tag, 'test_value')
            f = theano.function([], z)
            assert _allclose(f(), z.tag.test_value)

            # this test should fail
            x = T.constant(numpy.random.rand(2,4), dtype=config.floatX)
            self.assertRaises(ValueError, T.dot, x, y)
        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_incorrect_type(self):
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = 'raise'

            x = T.fmatrix('x')
            # Incorrect dtype (float64) for test_value
            x.tag.test_value = numpy.random.rand(3,4)
            y = T.dmatrix('y')
            y.tag.test_value = numpy.random.rand(4,5)

            self.assertRaises(TypeError, T.dot, x, y)
        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_scan(self):
        """
        Test the compute_test_value mechanism Scan.
        """
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = 'raise'
            #theano.config.compute_test_value = 'warn'
            k = T.iscalar("k")
            A = T.vector("A")
            k.tag.test_value = 3
            A.tag.test_value = numpy.random.rand(5).astype(config.floatX)

            def fx(prior_result, A):
                return prior_result * A
            # Symbolic description of the result
            result, updates = theano.scan(fn=fx,
                                          outputs_info=T.ones_like(A),
                                          non_sequences=A,
                                          n_steps=k)

            # We only care about A**k, but scan has provided us with A**1 through A**k.
            # Discard the values that we don't care about. Scan is smart enough to
            # notice this and not waste memory saving them.
            final_result = result[-1]
            assert hasattr(final_result.tag, 'test_value')
        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_scan_err1(self):
        # This test should fail when building fx for the first time
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = 'raise'

            k = T.iscalar("k")
            A = T.matrix("A")
            k.tag.test_value = 3
            A.tag.test_value = numpy.random.rand(5,3).astype(config.floatX)

            def fx(prior_result, A):
                return T.dot(prior_result, A)

            # Since we have to inspect the traceback,
            # we cannot simply use self.assertRaises()
            try:
                theano.scan(
                        fn=fx,
                        outputs_info=T.ones_like(A),
                        non_sequences=A,
                        n_steps=k)
                assert False
            except ValueError, e:
                # Get traceback
                tb = sys.exc_info()[2]
                # Get frame info 3 layers up
                frame_info = traceback.extract_tb(tb)[-4]
                # We should be in the "fx" function defined above
                assert os.path.split(frame_info[0])[1] == 'test_compute_test_value.py'
                assert frame_info[2] == 'fx'

        finally:
            theano.config.compute_test_value = orig_compute_test_value

    def test_scan_err2(self):
        # This test should not fail when building fx for the first time,
        # but when calling the scan's perform()
        orig_compute_test_value = theano.config.compute_test_value
        try:
            theano.config.compute_test_value = 'raise'

            k = T.iscalar("k")
            A = T.matrix("A")
            k.tag.test_value = 3
            A.tag.test_value = numpy.random.rand(5,3).astype(config.floatX)

            def fx(prior_result, A):
                return T.dot(prior_result, A)

            self.assertRaises(ValueError,
                    theano.scan,
                    fn=fx,
                    outputs_info=T.ones_like(A.T),
                    non_sequences=A,
                    n_steps=k)

            # Since we have to inspect the traceback,
            # we cannot simply use self.assertRaises()
            try:
                theano.scan(
                        fn=fx,
                        outputs_info=T.ones_like(A.T),
                        non_sequences=A,
                        n_steps=k)
                assert False
            except ValueError, e:
                # Get traceback
                tb = sys.exc_info()[2]
                # Get last frame info
                frame_info = traceback.extract_tb(tb)[-1]
                # We should be in scan_op.py, function 'perform'
                assert os.path.split(frame_info[0])[1] == 'scan_op.py'
                assert frame_info[2] == 'perform'

        finally:
            theano.config.compute_test_value = orig_compute_test_value

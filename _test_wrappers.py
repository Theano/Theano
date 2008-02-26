
import unittest
from wrappers import *


class _testCase_input(unittest.TestCase):
    def setUp(self):
        literal.hdb = {}
        literal.udb = {}
    def test_input_int(self):
        w = input(3)
        self.failUnless(isinstance(w, input.NN))
        self.failUnless(str(w.data.dtype) == input.int_dtype)
        self.failUnless(w.data == 3)
    def test_input_float(self):
        w = input(3.0)
        self.failUnless(isinstance(w, input.NN))
        self.failUnless(str(w.data.dtype) == input.float_dtype)
        self.failUnless(w.data == 3.0)


class _testCase_wrap(unittest.TestCase):
    def setUp(self):
        literal.hdb = {}
        literal.udb = {}
    def test_wrap_int(self):
        w = wrap(3)
        self.failUnless(isinstance(w, input.NN))
        self.failUnless(str(w.data.dtype) == input.int_dtype)
        self.failUnless(w.data == 3)
    def test_wrap_float(self):
        w = wrap(3.0)
        self.failUnless(isinstance(w, input.NN))
        self.failUnless(str(w.data.dtype) == input.float_dtype)
        self.failUnless(w.data == 3.0)


class _testCase_literal(unittest.TestCase):
    def setUp(self):
        literal.hdb = {}
        literal.udb = {}
    def test_int(self):
        w = literal(3)
        self.failUnless(isinstance(w, input.NN))
        self.failUnless(str(w.data.dtype) == input.int_dtype)
        self.failUnless(w.data == 3)

        u = literal(1+2)
        self.failUnless(u is w)

    def test_float(self):
        w = literal(3.0)
        self.failUnless(isinstance(w, input.NN))
        self.failUnless(str(w.data.dtype) == input.float_dtype)
        self.failUnless(w.data == 3.0)

        u = literal(1.0+2.0)
        self.failUnless(u is w)

    def test_mixed(self):
        f = literal(2.0)
        i = literal(2)
        self.failUnless(i is not f)


if __name__ == '__main__':
    unittest.main()


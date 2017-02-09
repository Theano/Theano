from nose.plugins.skip import SkipTest
import unittest

import theano.sandbox.mkl as mkl

if not mkl.mkl_available:
    raise SkipTest('Optional package MKL disabled')


class TestMKLStatus(unittest.TestCase):

    def test_mkl(self):
        print ('mkl_available: ' + str(mkl.mkl_available()))
        print ('mkl_version: ' + str(mkl.mkl_version()))

if __name__ == '__main__':
    unittest.main()

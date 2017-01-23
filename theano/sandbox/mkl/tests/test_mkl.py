import theano
import theano.sandbox.mkl as mkl
import unittest


class TestMKLStatus(unittest.TestCase):

    def test_mkl(self):
        dnn_stats = theano.config.dnn.enabled
        theano.config.dnn.enabled = "mkl"

        print ('mkl_available: ' + str(mkl.mkl_available()))
        print ('mkl_version: ' + str(mkl.mkl_version()))

        theano.config.dnn.enabled = dnn_stats


if __name__ == '__main__':
    unittest.main()

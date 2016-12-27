from __future__ import absolute_import, print_function, division

from nose.plugins.skip import SkipTest
import unittest
import numpy
import theano
import theano.tensor as tensor

import theano.sandbox.mkl as mkl
from theano.sandbox.mkl import mkl_elemwise, basic_ops

if not mkl.mkl_available:
    raise SkipTest('Optional package MKL disabled')

if theano.config.mode == 'FAST_COMPILE':
    mode_with_mkl = theano.compile.mode.get_mode('FAST_RUN').including('mkl')
    mode_without_mkl = theano.compile.mode.get_mode('FAST_RUN').excluding('mkl')
else:
    mode_with_mkl = theano.compile.mode.get_default_mode().including('mkl')
    mode_without_mkl = theano.compile.mode.get_default_mode().excluding('mkl')


class test_mkl_elemwise(unittest.TestCase):
    def test_elemwise_value(self):
        a = theano.tensor.ftensor4('a')
        b = theano.tensor.ftensor4('b')
        c = theano.tensor.ftensor4('c')

        a_internal = basic_ops.U2IElemwiseSum(inp_num=3, coeff=[1.0, 1.0, 1.0], uniq_id=1)(a)
        b_internal = basic_ops.U2IElemwiseSum(inp_num=3, coeff=[1.0, 1.0, 1.0], uniq_id=2)(b)
        c_internal = basic_ops.U2IElemwiseSum(inp_num=3, coeff=[1.0, 1.0, 1.0], uniq_id=3)(c)

        z_internal = mkl_elemwise.ElemwiseSum(inp_num=3, coeff=[1.0, 1.0, 1.0], uniq_id=4)(a_internal, b_internal, c_internal)
        z = basic_ops.I2U(uniq_id=5)(z_internal)
        f = theano.function([a, b, c], z)

        ival0 = numpy.random.rand(4, 4, 4, 4).astype(numpy.float32)
        ival1 = numpy.random.rand(4, 4, 4, 4).astype(numpy.float32)
        ival2 = numpy.random.rand(4, 4, 4, 4).astype(numpy.float32)
        assert numpy.allclose(f(ival0, ival1, ival2), ival0 + ival1 + ival2)
    
    def test_elemwise_U2I(self):
        a = theano.tensor.ftensor4('a')
        a_internal = basic_ops.U2IElemwiseSum(inp_num=1, coeff=[1.0, ], uniq_id=1)(a)
        a_out = basic_ops.I2U(uniq_id=2)(a_internal)
        f = theano.function([a], a_out)
        ival = numpy.random.rand(4, 4, 4, 4).astype(numpy.float32)
        assert numpy.allclose(f(ival), ival)

    def test_elemwise_wrong_dim(self):
        a = theano.tensor.fmatrix('a')
        try:
            basic_ops.U2IElemwiseSum(inp_num=1, coeff=[1.0, ], uniq_id=1)(a)
            raise Exception('No Exception when ndim is 2.')
        except TypeError:
            pass
        except Exception as e:
            raise Exception('test_elemwise_wrong_dim ' + str(e))

    def test_elemwise_user_layout(self):
        a = theano.tensor.ftensor4('a')
        b = theano.tensor.ftensor4('b')

        z = mkl_elemwise.ElemwiseSum(inp_num=2, coeff=[1.0, 1.0], uniq_id=1)(a, b)
        z_out = basic_ops.I2U(uniq_id=2)(z)
        f = theano.function([a, b], z_out, mode=mode_with_mkl)
        ival0 = numpy.random.rand(4, 4, 4, 4).astype(numpy.float32)
        ival1 = numpy.random.rand(4, 4, 4, 4).astype(numpy.float32)
        # assert numpy.allclose(f(ival0, ival1), ival0 + ival1)

    def test_elemwise_float64(self):
        old_floatX = theano.config.floatX
        theano.config.floatX = 'float64'
        
        a = theano.tensor.dtensor4('a')
        b = theano.tensor.dtensor4('b')
        c = theano.tensor.dtensor4('c')

        a_internal = basic_ops.U2IElemwiseSum(inp_num=3, coeff=[1.0, 1.0, 1.0], uniq_id=1)(a)
        b_internal = basic_ops.U2IElemwiseSum(inp_num=3, coeff=[1.0, 1.0, 1.0], uniq_id=2)(b)
        c_internal = basic_ops.U2IElemwiseSum(inp_num=3, coeff=[1.0, 1.0, 1.0], uniq_id=3)(c)

        z_internal = mkl_elemwise.ElemwiseSum(inp_num=3, coeff=[1.0, 1.0, 1.0], uniq_id=4)(a_internal, b_internal, c_internal)
        z = basic_ops.I2U(uniq_id=5)(z_internal)
        f = theano.function([a, b, c], z)

        ival0 = numpy.random.rand(4, 4, 4, 4).astype(theano.config.floatX)
        ival1 = numpy.random.rand(4, 4, 4, 4).astype(theano.config.floatX)
        ival2 = numpy.random.rand(4, 4, 4, 4).astype(theano.config.floatX)
        assert numpy.allclose(f(ival0, ival1, ival2), ival0 + ival1 + ival2)
        assert f(ival0, ival1, ival2).dtype == 'float64'
        theano.config.floatX = old_floatX

    def test_elemwise_float32(self):
        pass

    def test_elemwise_input_num(self):
        try:
            op1 = basic_ops.U2IElemwiseSum(inp_num=3, coeff=[1.0, 1.0], uniq_id=1)
            raise Exception('U2IElemwiseSUm No Exception when inp_num != len(coeff)')
        except ValueError:
            pass
        except Exception as e:
            raise Exception('test_elemwise_input_num ' + str(e))

        try:
            op2 = mkl_elemwise.ElemwiseSum(inp_num=3, coeff=[1.0, 1.0], uniq_id=2)
            raise Exception('ElemwiseSum No Exception when inp_num != len(coeff)')
        except ValueError:
            pass
        except Exception as e:
            raise Exception('test_elemwise_input_num ' + str(e))

    def test_elemwise_eq_hash(self):
        op1 = mkl_elemwise.ElemwiseSum(inp_num=3, coeff=[1.0, 1.0, 1.0], uniq_id=99)
        op2 = mkl_elemwise.ElemwiseSum(inp_num=3, coeff=[1.0, 1.0, 1.0], uniq_id=88)
        op3 = mkl_elemwise.ElemwiseSum(inp_num=3, coeff=[1.0, 1.0, 2.0], uniq_id=99)
        op4 = mkl_elemwise.ElemwiseSum(inp_num=2, coeff=[1.0, 1.0], uniq_id=99)

        assert op1 == op2
        assert op1 != op3
        assert op1 != op4

        assert hash(op1) == hash(op2)
        assert hash(op1) != hash(op3)
        # assert hash(op1) != hash(op4)



if __name__ == '__main__':
    unittest.main()

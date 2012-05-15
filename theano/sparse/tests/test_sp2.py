import time
import unittest

from nose.plugins.skip import SkipTest
import numpy
try:
    import scipy.sparse as sp
    import scipy.sparse
except ImportError:
    pass  # The variable enable_sparse will be used to disable the test file.

import theano

from theano import tensor as T
from theano import sparse as S
from theano.sparse.sandbox import sp2 as S2

from theano.tests import unittest_tools as utt

if S.enable_sparse == False:
    raise SkipTest('Optional package sparse disabled')

def as_sparse_format(data, format):
    if format == 'csc':
        return scipy.sparse.csc_matrix(data)
    elif format == 'csr':
        return scipy.sparse.csr_matrix(data)
    else:
        raise NotImplementedError()


def eval_outputs(outputs):
    return compile.function([], outputs)()[0]


def random_lil(shape, dtype, nnz):
    rval = sp.lil_matrix(shape, dtype=dtype)
    huge = 2 ** 30
    for k in range(nnz):
        # set non-zeros in random locations (row x, col y)
        idx = numpy.random.random_integers(huge, size=len(shape)) % shape
        value = numpy.random.rand()
        #if dtype *int*, value will always be zeros!
        if "int" in dtype:
            value = int(value * 100)
        rval.__setitem__(
                idx,
                value)
    return rval


class test_structured_add_s_v(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def test_structured_add_s_v_grad(self):
        sp_types = {'csc': sp.csc_matrix,
            'csr': sp.csr_matrix}
        
        for format in ['csr', 'csc']:
            for dtype in ['float32', 'float64']:
                spmat = sp_types[format](random_lil((4, 3), dtype, 3))
                mat = numpy.ones(3, dtype=dtype)
                
                S.verify_grad_sparse(S2.structured_add_s_v,
                    [spmat, mat], structured=True)
    
    def test_structured_add_s_v(self):
        sp_types = {'csc': sp.csc_matrix,
            'csr': sp.csr_matrix}
        
        for format in ['csr', 'csc']:
            for dtype in ['float32', 'float64']:
                x = S.SparseType(format, dtype=dtype)()
                y = T.vector(dtype=dtype)
                f = theano.function([x, y], S2.structured_add_s_v(x, y))
                
                spmat = sp_types[format](random_lil((4, 3), dtype, 3))
                spones = spmat.copy()
                spones.data = numpy.ones_like(spones.data)
                mat = numpy.ones(3, dtype=dtype)
                
                out = f(spmat, mat)
                
                assert numpy.all(out.toarray() == spones.multiply(spmat + mat))


class test_mul_s_v(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def test_structured_add_s_v_grad(self):
        sp_types = {'csc': sp.csc_matrix,
            'csr': sp.csr_matrix}
        
        for format in ['csr', 'csc']:
            for dtype in ['float32', 'float64']:
                spmat = sp_types[format](random_lil((4, 3), dtype, 3))
                mat = numpy.ones(3, dtype=dtype)
                
                S.verify_grad_sparse(S2.mul_s_v,
                    [spmat, mat], structured=True)
    
    def test_mul_s_v(self):
        sp_types = {'csc': sp.csc_matrix,
            'csr': sp.csr_matrix}
        
        for format in ['csr', 'csc']:
            for dtype in ['float32', 'float64']:
                x = S.SparseType(format, dtype=dtype)()
                y = T.vector(dtype=dtype)
                f = theano.function([x, y], S2.mul_s_v(x, y))
                
                spmat = sp_types[format](random_lil((4, 3), dtype, 3))
                spones = spmat.copy()
                spones.data = numpy.ones_like(spones.data)
                mat = numpy.ones(3, dtype=dtype)
                
                out = f(spmat, mat)
                
                assert numpy.all(out.toarray() == (spmat.toarray() * mat))

if __name__ == '__main__':
    unittest.main()

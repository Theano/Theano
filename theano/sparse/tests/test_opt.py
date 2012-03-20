import numpy
try:
    import scipy.sparse as sp
    import scipy.sparse
except ImportError:
    pass  # The variable enable_sparse will be used to disable the test file.

import theano
from theano import config, tensor
from theano.sparse import (enable_sparse, CSM, CSMProperties, csm_properties,
                           CSC, CSR)
from theano.sparse.tests.test_basic import random_lil
from theano.gof.python25 import any

if not enable_sparse:
    raise SkipTest('Optional package sparse disabled')


def test_local_csm_properties_csm():
    data = tensor.vector()
    indices, indptr, shape = (tensor.ivector(), tensor.ivector(),
                              tensor.ivector())
    mode = theano.compile.mode.get_default_mode()
    mode = mode.including("specialize", "local_csm_properties_csm")
    for CS, cast in [(CSC, sp.csc_matrix), (CSR, sp.csr_matrix)]:
        f = theano.function([data, indices, indptr, shape],
                            csm_properties(CS(data, indices, indptr, shape)),
                            mode=mode)
        #theano.printing.debugprint(f)
        assert not any(isinstance(node.op, (CSM, CSMProperties)) for node
                       in f.maker.env.toposort())
        v = cast(random_lil((10, 40),
                            config.floatX, 3))
        f(v.data, v.indices, v.indptr, v.shape)

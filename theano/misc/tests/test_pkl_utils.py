from numpy.testing import assert_allclose
from nose.plugins.skip import SkipTest

import theano.sandbox.cuda as cuda_ndarray

from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda.var import CudaNdarraySharedVariable
from theano.misc.pkl_utils import dump, load

if not cuda_ndarray.cuda_available:
    raise SkipTest('Optional package cuda disabled')


def test_dump_load():
    x = CudaNdarraySharedVariable('x', CudaNdarrayType((1, 1), name='x'),
                                  [[1]], False)

    with open('test', 'w') as f:
        dump(x, f)

    with open('test', 'r') as f:
        x = load(f)

    assert x.name == 'x'
    assert_allclose(x.get_value(), [[1]])

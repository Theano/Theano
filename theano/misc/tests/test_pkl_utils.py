from numpy.testing import assert_allclose
from nose.plugins.skip import SkipTest

import theano.sandbox.cuda as cuda_ndarray

from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda.var import CudaNdarraySharedVariable
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.misc.pkl_utils import dump, load

if not cuda_ndarray.cuda_enabled:
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


def test_dump_load_mrg():
    rng = MRG_RandomStreams(use_cuda=True)

    with open('test', 'w') as f:
        dump(rng, f)

    with open('test', 'r') as f:
        rng = load(f)

    assert type(rng) == MRG_RandomStreams


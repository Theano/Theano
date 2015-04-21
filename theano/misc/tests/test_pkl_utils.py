from numpy.testing import assert_allclose

from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda.var import CudaNdarraySharedVariable
from theano.misc.pkl_utils import dump, load


def test_dump_load():
    x = CudaNdarraySharedVariable('x', CudaNdarrayType((1, 1), name='x'),
                                  [[1]], False)

    with open('test', 'w') as f:
        dump(x, f)

    with open('test', 'r') as f:
        x = load(f)

    assert x.name == 'x'
    assert_allclose(x.get_value(), [[1]])
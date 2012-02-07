import numpy

from theano.tensor.utils import hash_from_ndarray


def test_hash_from_ndarray():
    hashs = []
    rng = numpy.random.rand(5, 5)

    for data in [-2, -1, 0, 1, 2, numpy.zeros((1, 5)), numpy.zeros((1, 6)),
                  # Data buffer empty but different shapes
                  numpy.zeros((1, 0)), numpy.zeros((2, 0)),
                  # Same data buffer and shapes but different strides
                  numpy.arange(25).reshape(5, 5),
                  numpy.arange(25).reshape(5, 5).T,
                  # Same data buffer, shapes and strides but different dtypes
                  numpy.zeros((5, 5), dtype="uint32"),
                  numpy.zeros((5, 5), dtype="int32"),

                  # Test slice
                  rng, rng[1:], rng[:4], rng[1:3], rng[::2], rng[::-1]
                  ]:
        data = numpy.asarray(data)
        hashs.append(hash_from_ndarray(data))

    assert len(set(hashs)) == len(hashs)

    # test that different type of views and their copy give the same hash
    assert hash_from_ndarray(rng[1:]) == hash_from_ndarray(rng[1:].copy())
    assert hash_from_ndarray(rng[1:3]) == hash_from_ndarray(rng[1:3].copy())
    assert hash_from_ndarray(rng[:4]) == hash_from_ndarray(rng[:4].copy())
    assert hash_from_ndarray(rng[::2]) == hash_from_ndarray(rng[::2].copy())
    assert hash_from_ndarray(rng[::-1]) == hash_from_ndarray(rng[::-1].copy())

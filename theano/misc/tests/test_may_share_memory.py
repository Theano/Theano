"""
test the tensor and sparse type. (gpuarray is tested in the gpuarray folder).
"""
from __future__ import absolute_import, print_function, division
import numpy as np
import theano

try:
    import scipy.sparse
    scipy_imported = True
except ImportError:
    scipy_imported = False

from theano.misc.may_share_memory import may_share_memory


def may_share_memory_core(a, b):
    va = a.view()
    vb = b.view()
    ra = a.reshape((4, 5))
    rb = b.reshape((4, 5))
    ta = a.T
    tb = b.T

    for a_, b_, rep in [(a, a, True), (b, b, True), (a, b, False),
                        (a, a[0], True), (a, a[:, 0], True), (a, a.T, True),
                        (a, (0,), False), (a, 1, False), (a, None, False),
                        (a, va, True), (b, vb, True), (va, b, False),
                        (a, vb, False), (a, ra, True), (b, rb, True),
                        (ra, b, False), (a, rb, False), (a, ta, True),
                        (b, tb, True), (ta, b, False), (a, tb, False)]:

        assert may_share_memory(a_, b_, False) == rep
        assert may_share_memory(b_, a_, False) == rep

    # test that it raise error when needed.
    for a_, b_, rep in [(a, (0,), False), (a, 1, False), (a, None, False), ]:
        assert may_share_memory(a_, b_, False) == rep
        assert may_share_memory(b_, a_, False) == rep
        try:
            may_share_memory(a_, b_)
            raise Exception("An error was expected")
        except TypeError:
            pass
        try:
            may_share_memory(b_, a_)
            raise Exception("An error was expected")
        except TypeError:
            pass


def test_may_share_memory():
    a = np.random.rand(5, 4)
    b = np.random.rand(5, 4)

    may_share_memory_core(a, b)

if scipy_imported:
    def test_may_share_memory_scipy():
        a = scipy.sparse.csc_matrix(scipy.sparse.eye(5, 3))
        b = scipy.sparse.csc_matrix(scipy.sparse.eye(4, 3))

        def as_ar(a):
            return theano._asarray(a, dtype='int32')
        for a_, b_, rep in [(a, a, True), (b, b, True), (a, b, False),
                            (a, a.data, True), (a, a.indptr, True),
                            (a, a.indices, True), (a, as_ar(a.shape), False),
                            (a.data, a, True), (a.indptr, a, True),
                            (a.indices, a, True), (as_ar(a.shape), a, False),
                            (b, b.data, True), (b, b.indptr, True),
                            (b, b.indices, True), (b, as_ar(b.shape), False),
                            (b.data, b, True), (b.indptr, b, True),
                            (b.indices, b, True), (as_ar(b.shape), b, False),
                            (b.data, a, False), (b.indptr, a, False),
                            (b.indices, a, False), (as_ar(b.shape), a, False)]:

            assert may_share_memory(a_, b_) == rep
            assert may_share_memory(b_, a_) == rep

        # test that it raise error when needed.
        for a_, b_, rep in [(a, (0,), False), (a, 1, False), (a, None, False)]:
            assert may_share_memory(a_, b_, False) == rep
            assert may_share_memory(b_, a_, False) == rep
            try:
                may_share_memory(a_, b_)
                raise Exception("An error was expected")
            except TypeError:
                pass
            try:
                may_share_memory(b_, a_)
                raise Exception("An error was expected")
            except TypeError:
                pass

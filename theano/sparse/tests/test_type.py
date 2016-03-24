from __future__ import absolute_import, print_function, division


def test_sparse_type():
    import theano.sparse
    # They need to be available even if scipy is not available.
    assert hasattr(theano.sparse, "SparseType")

from __future__ import absolute_import, print_function, division
from theano.gof.utils import hash_from_code


def hash_from_sparse(data):
    # We need to hash the shapes as hash_from_code only hashes
    # the data buffer. Otherwise, this will cause problem with shapes like:
    # (1, 0) and (2, 0)
    # We also need to add the dtype to make the distinction between
    # uint32 and int32 of zeros with the same shape.

    # Python hash is not strong, so use sha256 instead. To avoid having a too
    # long hash, I call it again on the contatenation of all parts.
    return hash_from_code(hash_from_code(data.data) +
                          hash_from_code(data.indices) +
                          hash_from_code(data.indptr) +
                          hash_from_code(str(data.shape)) +
                          hash_from_code(str(data.dtype)) +
                          hash_from_code(data.format))

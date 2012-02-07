import numpy

from theano.gof.cc import hash_from_code


def hash_from_ndarray(data):
    # We need to hash the shapes and strides as hash_from_code only hashes
    # the data buffer. Otherwise, this will cause problem with shapes like:
    # (1, 0) and (2, 0) and problem with inplace transpose.
    # We also need to add the dtype to make the distinction between
    # uint32 and int32 of zeros with the same shape and strides.

    # python hash are not strong, so I always use md5 in order not to have a
    # too long hash, I call it again on the concatenation of all parts.
    if not data.flags["C_CONTIGUOUS"] and not data.flags["F_CONTIGUOUS"]:
        data = numpy.ascontiguousarray(data)
    return hash_from_code(hash_from_code(data) +
                          hash_from_code(str(data.shape)) +
                          hash_from_code(str(data.strides)) +
                          hash_from_code(str(data.dtype)))

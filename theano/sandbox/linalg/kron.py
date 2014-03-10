from theano import tensor


def kron(a, b):
    """ Kronecker product

    Same as scipy.linalg.kron(a, b).

    :note: numpy.kron(a, b) != scipy.linalg.kron(a, b)!
        They don't have the same shape and order when
        a.ndim != b.ndim != 2.

    :param a: array_like
    :param b: array_like
    :return: array_like with a.ndim + b.ndim - 2 dimensions.

    """
    a = tensor.as_tensor_variable(a)
    b = tensor.as_tensor_variable(b)
    if (a.ndim + b.ndim <= 2):
        raise TypeError('kron: inputs dimensions must sum to 3 or more. '
                        'You passed %d and %d.' % (a.ndim, b.ndim))
    o = tensor.outer(a, b)
    o = o.reshape(tensor.concatenate((a.shape, b.shape)),
                  a.ndim + b.ndim)
    shf = o.dimshuffle(0, 2, 1, * range(3, o.ndim))
    if shf.ndim == 3:
        shf = o.dimshuffle(1, 0, 2)
        o = shf.flatten()
    else:
        o = shf.reshape((o.shape[0] * o.shape[2],
                         o.shape[1] * o.shape[3]) +
                        tuple([o.shape[i] for i in range(4, o.ndim)]))
    return o

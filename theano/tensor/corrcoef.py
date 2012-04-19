import theano


def corrcoef(X):
    """Return correlation coefficients."""

    Xm = (X.T - X.mean(1).T).T
    Xmn = (Xm.T / theano.tensor.sqrt((Xm ** 2.).sum(1)).T).T
    cc = theano.tensor.dot(Xmn, Xmn.T)
    return cc

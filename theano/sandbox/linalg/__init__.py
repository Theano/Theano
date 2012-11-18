from theano.tensor import as_tensor_variable as _as_tensor_variable
from ops import (cholesky, matrix_inverse, solve,
        diag, extract_diag, alloc_diag,
        det, psd, eig, eigh,
        trace, spectral_radius_bound)


def eigvals(a):
    "Compute the eigenvalues of a general matrix."
    a = _as_tensor_variable(a)
    assert a.ndim == 2
    return eig(a)[0]

def eigvalsh(a, UPLO='L'):
    "Compute the eigenvalues of a Hermitian or real symmetric matrix."
    a = _as_tensor_variable(a)
    assert a.ndim == 2    
    return eigh(a, UPLO)[0]

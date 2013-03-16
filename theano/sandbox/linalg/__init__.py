import theano.tensor as _T
from theano.tensor import as_tensor_variable as _as_tensor_variable
from ops import (cholesky, matrix_inverse, solve,
        diag, extract_diag, alloc_diag,
        det, psd, eig, eigh, pinv,
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

def svd(a, full_matrices=1, compute_uv=1):
    "Singular Value Decomposition."
    raise ValueError("Not implemented. Work in progress.")

def norm(x, ord=None):
    "Matrix or vector norm."
    x = _as_tensor_variable(x)
    if ord is None: # check the default case first and handle it immediately
        return _T.sqrt(_T.real((_T.conj(x) * x).sum()))
    raise ValueError("Not implemented. Work in progress.")

def lstsq(a, b, rcond=-1):
    "Return the least-squares solution to a linear matrix equation."
    raise ValueError("Not implemented. Work in progress.")

def tensorsolve(a, b, axes=None):
    "Solve the tensor equation ``a x = b`` for x."
    raise ValueError("Not implemented. Work in progress.")

def tensorinv(a, ind=2):
    "Compute the 'inverse' of an N-dimensional array."
    raise ValueError("Not implemented. Work in progress.")

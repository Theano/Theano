from __future__ import absolute_import, print_function, division
import logging
import warnings
from six.moves import xrange

import numpy as np

try:
    import scipy.linalg
    imported_scipy = True
except ImportError:
    # some ops (e.g. Cholesky, Solve, A_Xinv_b) won't work
    imported_scipy = False

from theano import tensor
import theano.tensor
from theano.tensor import as_tensor_variable
from theano.gof import Op, Apply

logger = logging.getLogger(__name__)

MATRIX_STRUCTURES = (
    'general',
    'symmetric',
    'lower_triangular',
    'upper_triangular',
    'hermitian',
    'banded',
    'diagonal',
    'toeplitz')


class Cholesky(Op):
    """
    Return a triangular matrix square root of positive semi-definite `x`.

    L = cholesky(X, lower=True) implies dot(L, L.T) == X.

    Parameters
    ----------
    lower : bool, default=True
        Whether to return the lower or upper cholesky factor
    on_error : ['raise', 'nan']
        If on_error is set to 'raise', this Op will raise a
        `scipy.linalg.LinAlgError` if the matrix is not positive definite.
        If on_error is set to 'nan', it will return a matrix containing
        nans instead.
    """
    # TODO: inplace
    # TODO: for specific dtypes
    # TODO: LAPACK wrapper with in-place behavior, for solve also

    __props__ = ('lower', 'destructive', 'on_error')

    def __init__(self, lower=True, on_error='raise'):
        self.lower = lower
        self.destructive = False
        if on_error not in ['raise', 'nan']:
            raise ValueError('on_error must be one of "raise" or ""nan"')
        self.on_error = on_error

    def infer_shape(self, node, shapes):
        return [shapes[0]]

    def make_node(self, x):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the Cholesky op")
        x = as_tensor_variable(x)
        assert x.ndim == 2
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        x = inputs[0]
        z = outputs[0]
        try:
            z[0] = scipy.linalg.cholesky(x, lower=self.lower).astype(x.dtype)
        except scipy.linalg.LinAlgError:
            if self.on_error == 'raise':
                raise
            else:
                z[0] = (np.zeros(x.shape) * np.nan).astype(x.dtype)

    def L_op(self, inputs, outputs, gradients):
        """
        Cholesky decomposition reverse-mode gradient update.

        Symbolic expression for reverse-mode Cholesky gradient taken from [#]_

        References
        ----------
        .. [#] I. Murray, "Differentiation of the Cholesky decomposition",
           http://arxiv.org/abs/1602.07527

        """

        dz = gradients[0]
        chol_x = outputs[0]

        # Replace the cholesky decomposition with 1 if there are nans
        # or solve_upper_triangular will throw a ValueError.
        if self.on_error == 'nan':
            ok = ~tensor.any(tensor.isnan(chol_x))
            chol_x = tensor.switch(ok, chol_x, 1)
            dz = tensor.switch(ok, dz, 1)

        # deal with upper triangular by converting to lower triangular
        if not self.lower:
            chol_x = chol_x.T
            dz = dz.T

        def tril_and_halve_diagonal(mtx):
            """Extracts lower triangle of square matrix and halves diagonal."""
            return tensor.tril(mtx) - tensor.diag(tensor.diagonal(mtx) / 2.)

        def conjugate_solve_triangular(outer, inner):
            """Computes L^{-T} P L^{-1} for lower-triangular L."""
            return solve_upper_triangular(
                outer.T, solve_upper_triangular(outer.T, inner.T).T)

        s = conjugate_solve_triangular(
            chol_x, tril_and_halve_diagonal(chol_x.T.dot(dz)))

        if self.lower:
            grad = tensor.tril(s + s.T) - tensor.diag(tensor.diagonal(s))
        else:
            grad = tensor.triu(s + s.T) - tensor.diag(tensor.diagonal(s))

        if self.on_error == 'nan':
            return [tensor.switch(ok, grad, np.nan)]
        else:
            return [grad]

cholesky = Cholesky()


class CholeskyGrad(Op):
    """
    """

    __props__ = ('lower', 'destructive')

    def __init__(self, lower=True):
        self.lower = lower
        self.destructive = False

    def make_node(self, x, l, dz):
        x = as_tensor_variable(x)
        l = as_tensor_variable(l)
        dz = as_tensor_variable(dz)
        assert x.ndim == 2
        assert l.ndim == 2
        assert dz.ndim == 2
        assert l.owner.op.lower == self.lower, (
            "lower/upper mismatch between Cholesky op and CholeskyGrad op"
        )
        return Apply(self, [x, l, dz], [x.type()])

    def perform(self, node, inputs, outputs):
        """
        Implements the "reverse-mode" gradient [#]_ for the
        Cholesky factorization of a positive-definite matrix.

        References
        ----------
        .. [#] S. P. Smith. "Differentiation of the Cholesky Algorithm".
           Journal of Computational and Graphical Statistics,
           Vol. 4, No. 2 (Jun.,1995), pp. 134-147
           http://www.jstor.org/stable/1390762

        """
        x = inputs[0]
        L = inputs[1]
        dz = inputs[2]
        dx = outputs[0]
        N = x.shape[0]
        if self.lower:
            F = np.tril(dz)
            for k in xrange(N - 1, -1, -1):
                for j in xrange(k + 1, N):
                    for i in xrange(j, N):
                        F[i, k] -= F[i, j] * L[j, k]
                        F[j, k] -= F[i, j] * L[i, k]
                for j in xrange(k + 1, N):
                    F[j, k] /= L[k, k]
                    F[k, k] -= L[j, k] * F[j, k]
                F[k, k] /= (2 * L[k, k])
        else:
            F = np.triu(dz)
            for k in xrange(N - 1, -1, -1):
                for j in xrange(k + 1, N):
                    for i in xrange(j, N):
                        F[k, i] -= F[j, i] * L[k, j]
                        F[k, j] -= F[j, i] * L[k, i]
                for j in xrange(k + 1, N):
                    F[k, j] /= L[k, k]
                    F[k, k] -= L[k, j] * F[k, j]
                F[k, k] /= (2 * L[k, k])
        dx[0] = F

    def infer_shape(self, node, shapes):
        return [shapes[0]]


class Solve(Op):
    """
    Solve a system of linear equations.

    For on CPU and GPU.
    """

    __props__ = ('A_structure', 'lower', 'overwrite_A', 'overwrite_b')

    def __init__(self,
                 A_structure='general',
                 lower=False,
                 overwrite_A=False,
                 overwrite_b=False):
        if A_structure not in MATRIX_STRUCTURES:
            raise ValueError('Invalid matrix structure argument', A_structure)
        self.A_structure = A_structure
        self.lower = lower
        self.overwrite_A = overwrite_A
        self.overwrite_b = overwrite_b

    def __repr__(self):
        return 'Solve{%s}' % str(self._props())

    def make_node(self, A, b):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the Solve op")
        A = as_tensor_variable(A)
        b = as_tensor_variable(b)
        assert A.ndim == 2
        assert b.ndim in [1, 2]

        # infer dtype by solving the most simple
        # case with (1, 1) matrices
        o_dtype = scipy.linalg.solve(
            np.eye(1).astype(A.dtype),
            np.eye(1).astype(b.dtype)).dtype
        x = tensor.tensor(
            broadcastable=b.broadcastable,
            dtype=o_dtype)
        return Apply(self, [A, b], [x])

    def perform(self, node, inputs, output_storage):
        A, b = inputs
        if self.A_structure == 'lower_triangular':
            rval = scipy.linalg.solve_triangular(
                A, b, lower=True)
        elif self.A_structure == 'upper_triangular':
            rval = scipy.linalg.solve_triangular(
                A, b, lower=False)
        else:
            rval = scipy.linalg.solve(A, b)
        output_storage[0][0] = rval

    # computes shape of x where x = inv(A) * b
    def infer_shape(self, node, shapes):
        Ashape, Bshape = shapes
        rows = Ashape[1]
        if len(Bshape) == 1:  # b is a Vector
            return [(rows,)]
        else:
            cols = Bshape[1]  # b is a Matrix
            return [(rows, cols)]

    def L_op(self, inputs, outputs, output_gradients):
        """
        Reverse-mode gradient updates for matrix solve operation c = A \\\ b.

        Symbolic expression for updates taken from [#]_.

        References
        ----------
        .. [#] M. B. Giles, "An extended collection of matrix derivative results
          for forward and reverse mode automatic differentiation",
          http://eprints.maths.ox.ac.uk/1079/

        """
        A, b = inputs
        c = outputs[0]
        c_bar = output_gradients[0]
        trans_map = {
            'lower_triangular': 'upper_triangular',
            'upper_triangular': 'lower_triangular'
        }
        trans_solve_op = Solve(
            # update A_structure and lower to account for a transpose operation
            A_structure=trans_map.get(self.A_structure, self.A_structure),
            lower=not self.lower
        )
        b_bar = trans_solve_op(A.T, c_bar)
        # force outer product if vector second input
        A_bar = -tensor.outer(b_bar, c) if c.ndim == 1 else -b_bar.dot(c.T)
        if self.A_structure == 'lower_triangular':
            A_bar = tensor.tril(A_bar)
        elif self.A_structure == 'upper_triangular':
            A_bar = tensor.triu(A_bar)
        return [A_bar, b_bar]

solve = Solve()
"""
Solves the equation ``a x = b`` for x, where ``a`` is a matrix and
``b`` can be either a vector or a matrix.

Note

Parameters
----------
a : `(M, M) symbolix matrix`
    A square matrix
b : `(M,) or (M, N) symbolic vector or matrix`
    Right hand side matrix in ``a x = b``


Returns
-------
x : `(M, ) or (M, N) symbolic vector or matrix`
    x will have the same shape as b
"""
# lower and upper triangular solves
solve_lower_triangular = Solve(A_structure='lower_triangular', lower=True)
"""Optimized implementation of :func:`theano.tensor.slinalg.solve` when A is lower triangular."""
solve_upper_triangular = Solve(A_structure='upper_triangular', lower=False)
"""Optimized implementation of :func:`theano.tensor.slinalg.solve` when A is upper triangular."""
# symmetric solves
solve_symmetric = Solve(A_structure='symmetric')
"""Optimized implementation of :func:`theano.tensor.slinalg.solve` when A is symmetric."""

# TODO: Optimizations to replace multiplication by matrix inverse
#      with solve() Op (still unwritten)


class Eigvalsh(Op):
    """
    Generalized eigenvalues of a Hermitian positive definite eigensystem.

    """

    __props__ = ('lower',)

    def __init__(self, lower=True):
        assert lower in [True, False]
        self.lower = lower

    def make_node(self, a, b):
        assert imported_scipy, (
            "Scipy not  available. Scipy is needed for the Eigvalsh op")

        if b == theano.tensor.NoneConst:
            a = as_tensor_variable(a)
            assert a.ndim == 2

            out_dtype = theano.scalar.upcast(a.dtype)
            w = theano.tensor.vector(dtype=out_dtype)
            return Apply(self, [a], [w])
        else:
            a = as_tensor_variable(a)
            b = as_tensor_variable(b)
            assert a.ndim == 2
            assert b.ndim == 2

            out_dtype = theano.scalar.upcast(a.dtype, b.dtype)
            w = theano.tensor.vector(dtype=out_dtype)
            return Apply(self, [a, b], [w])

    def perform(self, node, inputs, outputs):
        (w,) = outputs
        if len(inputs) == 2:
            w[0] = scipy.linalg.eigvalsh(a=inputs[0], b=inputs[1], lower=self.lower)
        else:
            w[0] = scipy.linalg.eigvalsh(a=inputs[0], b=None, lower=self.lower)

    def grad(self, inputs, g_outputs):
        a, b = inputs
        gw, = g_outputs
        return EigvalshGrad(self.lower)(a, b, gw)

    def infer_shape(self, node, shapes):
        n = shapes[0][0]
        return [(n,)]


class EigvalshGrad(Op):
    """
    Gradient of generalized eigenvalues of a Hermitian positive definite
    eigensystem.

    """

    # Note: This Op (EigvalshGrad), should be removed and replaced with a graph
    # of theano ops that is constructed directly in Eigvalsh.grad.
    # But this can only be done once scipy.linalg.eigh is available as an Op
    # (currently the Eigh uses numpy.linalg.eigh, which doesn't let you
    # pass the right-hand-side matrix for a generalized eigenproblem.) See the
    # discussion on github at
    # https://github.com/Theano/Theano/pull/1846#discussion-diff-12486764

    __props__ = ('lower',)

    def __init__(self, lower=True):
        assert lower in [True, False]
        self.lower = lower
        if lower:
            self.tri0 = np.tril
            self.tri1 = lambda a: np.triu(a, 1)
        else:
            self.tri0 = np.triu
            self.tri1 = lambda a: np.tril(a, -1)

    def make_node(self, a, b, gw):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the GEigvalsh op")
        a = as_tensor_variable(a)
        b = as_tensor_variable(b)
        gw = as_tensor_variable(gw)
        assert a.ndim == 2
        assert b.ndim == 2
        assert gw.ndim == 1

        out_dtype = theano.scalar.upcast(a.dtype, b.dtype, gw.dtype)
        out1 = theano.tensor.matrix(dtype=out_dtype)
        out2 = theano.tensor.matrix(dtype=out_dtype)
        return Apply(self, [a, b, gw], [out1, out2])

    def perform(self, node, inputs, outputs):
        (a, b, gw) = inputs
        w, v = scipy.linalg.eigh(a, b, lower=self.lower)
        gA = v.dot(np.diag(gw).dot(v.T))
        gB = - v.dot(np.diag(gw * w).dot(v.T))

        # See EighGrad comments for an explanation of these lines
        out1 = self.tri0(gA) + self.tri1(gA).T
        out2 = self.tri0(gB) + self.tri1(gB).T
        outputs[0][0] = np.asarray(out1, dtype=node.outputs[0].dtype)
        outputs[1][0] = np.asarray(out2, dtype=node.outputs[1].dtype)

    def infer_shape(self, node, shapes):
        return [shapes[0], shapes[1]]


def eigvalsh(a, b, lower=True):
    return Eigvalsh(lower)(a, b)


def kron(a, b):
    """ Kronecker product.

    Same as scipy.linalg.kron(a, b).

    Parameters
    ----------
    a: array_like
    b: array_like

    Returns
    -------
    array_like with a.ndim + b.ndim - 2 dimensions

    Notes
    -----
    numpy.kron(a, b) != scipy.linalg.kron(a, b)!
    They don't have the same shape and order when
    a.ndim != b.ndim != 2.

    """
    a = tensor.as_tensor_variable(a)
    b = tensor.as_tensor_variable(b)
    if (a.ndim + b.ndim <= 2):
        raise TypeError('kron: inputs dimensions must sum to 3 or more. '
                        'You passed %d and %d.' % (a.ndim, b.ndim))
    o = tensor.outer(a, b)
    o = o.reshape(tensor.concatenate((a.shape, b.shape)),
                  a.ndim + b.ndim)
    shf = o.dimshuffle(0, 2, 1, * list(range(3, o.ndim)))
    if shf.ndim == 3:
        shf = o.dimshuffle(1, 0, 2)
        o = shf.flatten()
    else:
        o = shf.reshape((o.shape[0] * o.shape[2],
                         o.shape[1] * o.shape[3]) +
                        tuple(o.shape[i] for i in xrange(4, o.ndim)))
    return o


class Expm(Op):
    """
    Compute the matrix exponential of a square array.

    """

    __props__ = ()

    def make_node(self, A):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the Expm op")

        A = as_tensor_variable(A)
        assert A.ndim == 2
        expm = theano.tensor.matrix(dtype=A.dtype)
        return Apply(self, [A, ], [expm, ])

    def perform(self, node, inputs, outputs):
        (A,) = inputs
        (expm,) = outputs
        expm[0] = scipy.linalg.expm(A)

    def grad(self, inputs, outputs):
        (A,) = inputs
        (g_out,) = outputs
        return [ExpmGrad()(A, g_out)]

    def infer_shape(self, node, shapes):
        return [shapes[0]]


class ExpmGrad(Op):
    """
    Gradient of the matrix exponential of a square array.

    """

    __props__ = ()

    def make_node(self, A, gw):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the Expm op")
        A = as_tensor_variable(A)
        assert A.ndim == 2
        out = theano.tensor.matrix(dtype=A.dtype)
        return Apply(self, [A, gw], [out, ])

    def infer_shape(self, node, shapes):
        return [shapes[0]]

    def perform(self, node, inputs, outputs):
        # Kalbfleisch and Lawless, J. Am. Stat. Assoc. 80 (1985) Equation 3.4
        # Kind of... You need to do some algebra from there to arrive at
        # this expression.
        (A, gA) = inputs
        (out,) = outputs
        w, V = scipy.linalg.eig(A, right=True)
        U = scipy.linalg.inv(V).T

        exp_w = np.exp(w)
        X = np.subtract.outer(exp_w, exp_w) / np.subtract.outer(w, w)
        np.fill_diagonal(X, exp_w)
        Y = U.dot(V.T.dot(gA).dot(U) * X).dot(V.T)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.ComplexWarning)
            out[0] = Y.astype(A.dtype)


expm = Expm()

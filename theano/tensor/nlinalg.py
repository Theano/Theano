import logging
import theano

logger = logging.getLogger(__name__)
import numpy

from theano.gof import Op, Apply

from theano.tensor import as_tensor_variable, dot, DimShuffle, Dot
from theano.tensor.blas import Dot22
from theano.tensor.opt import (register_stabilize,
        register_specialize, register_canonicalize)
from theano.gof import local_optimizer
from theano.gof.opt import Optimizer
from theano.gradient import DisconnectedType
from theano.tensor import basic as tensor


class MatrixPinv(Op):
    """Computes the pseudo-inverse of a matrix :math:`A`.

    The pseudo-inverse of a matrix A, denoted :math:`A^+`, is
    defined as: "the matrix that 'solves' [the least-squares problem]
    :math:`Ax = b`," i.e., if :math:`\\bar{x}` is said solution, then
    :math:`A^+` is that matrix such that :math:`\\bar{x} = A^+b`.

    Note that :math:`Ax=AA^+b`, so :math:`AA^+` is close to the identity matrix.
    This method is not faster then `matrix_inverse`. Its strength comes from
    that it works for non-square matrices.
    If you have a square matrix though, `matrix_inverse` can be both more
    exact and faster to compute. Also this op does not get optimized into a
    solve op.
    """

    __props__ = ()

    def __init__(self):
        pass

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        return Apply(self, [x], [x.type()])

    def perform(self, node, (x,), (z, )):
        z[0] = numpy.linalg.pinv(x).astype(x.dtype)

pinv = MatrixPinv()


class MatrixInverse(Op):
    """Computes the inverse of a matrix :math:`A`.

    Given a square matrix :math:`A`, ``matrix_inverse`` returns a square
    matrix :math:`A_{inv}` such that the dot product :math:`A \cdot A_{inv}`
    and :math:`A_{inv} \cdot A` equals the identity matrix :math:`I`.

    :note: When possible, the call to this op will be optimized to the call
           of ``solve``.
    """

    __props__ = ()

    def __init__(self):
        pass

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        return Apply(self, [x], [x.type()])

    def perform(self, node, (x,), (z, )):
        z[0] = numpy.linalg.inv(x).astype(x.dtype)

    def grad(self, inputs, g_outputs):
        r"""The gradient function should return

            .. math:: V\frac{\partial X^{-1}}{\partial X},

        where :math:`V` corresponds to ``g_outputs`` and :math:`X` to
        ``inputs``. Using the `matrix cookbook
        <http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274>`_,
        once can deduce that the relation corresponds to

            .. math:: (X^{-1} \cdot V^{T} \cdot X^{-1})^T.

        """
        x, = inputs
        xi = self(x)
        gz, = g_outputs
        #TT.dot(gz.T,xi)
        return [-matrix_dot(xi, gz.T, xi).T]

    def R_op(self, inputs, eval_points):
        r"""The gradient function should return

            .. math:: \frac{\partial X^{-1}}{\partial X}V,

        where :math:`V` corresponds to ``g_outputs`` and :math:`X` to
        ``inputs``.  Using the `matrix cookbook
        <http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274>`_,
        once can deduce that the relation corresponds to

            .. math:: X^{-1} \cdot V \cdot X^{-1}.

        """
        x, = inputs
        xi = self(x)
        ev, = eval_points
        if ev is None:
            return [None]
        return [-matrix_dot(xi, ev, xi)]

    def infer_shape(self, node, shapes):
        return shapes

matrix_inverse = MatrixInverse()


def matrix_dot(*args):
    """ Shorthand for product between several dots

    Given :math:`N` matrices :math:`A_0, A_1, .., A_N`, ``matrix_dot`` will
    generate the matrix product between all in the given order, namely
    :math:`A_0 \cdot A_1 \cdot A_2 \cdot .. \cdot A_N`.
    """
    rval = args[0]
    for a in args[1:]:
        rval = theano.tensor.dot(rval, a)
    return rval


class AllocDiag(Op):
    """
    Allocates a square matrix with the given vector as its diagonal.
    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, _x):
        x = as_tensor_variable(_x)
        if x.type.ndim != 1:
            raise TypeError('AllocDiag only works on vectors', _x)
        return Apply(self, [x], [theano.tensor.matrix(dtype=x.type.dtype)])

    def grad(self, inputs, g_outputs):
        return [extract_diag(g_outputs[0])]

    def perform(self, node, (x,), (z,)):
        if x.ndim != 1:
            raise TypeError(x)
        z[0] = numpy.diag(x)

    def infer_shape(self, node, shapes):
        x_s, = shapes
        return [(x_s[0], x_s[0])]

alloc_diag = AllocDiag()


class ExtractDiag(Op):
    """ Return the diagonal of a matrix.

    :note: work on the GPU.
    """
    def __init__(self, view=False):
        self.view = view
        if self.view:
            self.view_map = {0: [0]}

    def __eq__(self, other):
        return type(self) == type(other) and self.view == other.view

    def __hash__(self):
        return hash(type(self)) ^ hash(self.view)

    def make_node(self, _x):
        if not isinstance(_x, theano.Variable):
            x = as_tensor_variable(_x)
        else:
            x = _x

        if x.type.ndim != 2:
            raise TypeError('ExtractDiag only works on matrices', _x)
        return Apply(self, [x], [x.type.__class__(broadcastable=(False,),
                                                  dtype=x.type.dtype)()])

    def perform(self, node, ins, outs):
        """ For some reason numpy.diag(x) is really slow, so we
        implemented our own. """
        x, = ins
        z, = outs
        # zero-dimensional matrices ...
        if x.shape[0] == 0 or x.shape[1] == 0:
            z[0] = node.outputs[0].type.value_zeros((0,))
            return

        if x.shape[0] < x.shape[1]:
            rval = x[:, 0]
        else:
            rval = x[0]

        rval.strides = (x.strides[0] + x.strides[1],)
        if self.view:
            z[0] = rval
        else:
            z[0] = rval.copy()

    def __str__(self):
        return 'ExtractDiag{view=%s}' % self.view

    def grad(self, inputs, g_outputs):
        x = theano.tensor.zeros_like(inputs[0])
        xdiag = alloc_diag(g_outputs[0])
        return [theano.tensor.set_subtensor(
            x[:xdiag.shape[0], :xdiag.shape[1]],
            xdiag)]

    def infer_shape(self, node, shapes):
        x_s, = shapes
        shp = theano.tensor.min(node.inputs[0].shape)
        return [(shp,)]

extract_diag = ExtractDiag()
#TODO: optimization to insert ExtractDiag with view=True


def diag(x):
    """
    Numpy-compatibility method
    If `x` is a matrix, return its diagonal.
    If `x` is a vector return a matrix with it as its diagonal.

    * This method does not support the `k` argument that numpy supports.
    """
    xx = as_tensor_variable(x)
    if xx.type.ndim == 1:
        return alloc_diag(xx)
    elif xx.type.ndim == 2:
        return extract_diag(xx)
    else:
        raise TypeError('diag requires vector or matrix argument', x)


def trace(X):
    """
    Returns the sum of diagonal elements of matrix X.

    :note: work on GPU since 0.6rc4.
    """
    return extract_diag(X).sum()


class Det(Op):
    """Matrix determinant
    Input should be a square matrix
    """
    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        o = theano.tensor.scalar(dtype=x.dtype)
        return Apply(self, [x], [o])

    def perform(self, node, (x,), (z, )):
        try:
            z[0] = numpy.asarray(numpy.linalg.det(x), dtype=x.dtype)
        except Exception:
            print 'Failed to compute determinant', x
            raise

    def grad(self, inputs, g_outputs):
        gz, = g_outputs
        x, = inputs
        return [gz * self(x) * matrix_inverse(x).T]

    def infer_shape(self, node, shapes):
        return [()]

    def __str__(self):
        return "Det"
det = Det()


class Eig(Op):
    """Compute the eigenvalues and right eigenvectors of a square array.

    """
    _numop = staticmethod(numpy.linalg.eig)
    __props__ = ()

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        w = theano.tensor.vector(dtype=x.dtype)
        v = theano.tensor.matrix(dtype=x.dtype)
        return Apply(self, [x], [w, v])

    def perform(self, node, (x,), (w, v)):
        w[0], v[0] = [z.astype(x.dtype) for z in self._numop(x)]

    def infer_shape(self, node, shapes):
        n = shapes[0][0]
        return [(n,), (n, n)]

eig = Eig()


class Eigh(Eig):
    """
    Return the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.

    """
    _numop = staticmethod(numpy.linalg.eigh)
    __props__ = ('UPLO',)

    def __init__(self, UPLO='L'):
        assert UPLO in ['L', 'U']
        self.UPLO = UPLO

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        # Numpy's linalg.eigh may return either double or single
        # presision eigenvalues depending on installed version of
        # LAPACK.  Rather than trying to reproduce the (rather
        # involved) logic, we just probe linalg.eigh with a trivial
        # input.
        w_dtype = self._numop([[numpy.dtype(x.dtype).type()]])[0].dtype.name
        w = theano.tensor.vector(dtype=w_dtype)
        v = theano.tensor.matrix(dtype=x.dtype)
        return Apply(self, [x], [w, v])

    def perform(self, node, (x,), (w, v)):
        w[0], v[0] = self._numop(x, self.UPLO)

    def grad(self, inputs, g_outputs):
        r"""The gradient function should return

           .. math:: \sum_n\left(W_n\frac{\partial\,w_n}
                           {\partial a_{ij}} +
                     \sum_k V_{nk}\frac{\partial\,v_{nk}}
                           {\partial a_{ij}}\right),

        where [:math:`W`, :math:`V`] corresponds to ``g_outputs``,
        :math:`a` to ``inputs``, and  :math:`(w, v)=\mbox{eig}(a)`.

        Analytic formulae for eigensystem gradients are well-known in
        perturbation theory:

           .. math:: \frac{\partial\,w_n}
                          {\partial a_{ij}} = v_{in}\,v_{jn}


           .. math:: \frac{\partial\,v_{kn}}
                          {\partial a_{ij}} =
                \sum_{m\ne n}\frac{v_{km}v_{jn}}{w_n-w_m}
        """
        x, = inputs
        w, v = self(x)
        # Replace gradients wrt disconnected variables with
        # zeros. This is a work-around for issue #1063.
        gw, gv = _zero_disconnected([w, v], g_outputs)
        return [EighGrad(self.UPLO)(x, w, v, gw, gv)]


def _zero_disconnected(outputs, grads):
    l = []
    for o, g in zip(outputs, grads):
        if isinstance(g.type, DisconnectedType):
            l.append(o.zeros_like())
        else:
            l.append(g)
    return l


class EighGrad(Op):
    """Gradient of an eigensystem of a Hermitian matrix.

    """
    __props__ = ('UPLO',)

    def __init__(self, UPLO='L'):
        assert UPLO in ['L', 'U']
        self.UPLO = UPLO
        if UPLO == 'L':
            self.tri0 = numpy.tril
            self.tri1 = lambda a: numpy.triu(a, 1)
        else:
            self.tri0 = numpy.triu
            self.tri1 = lambda a: numpy.tril(a, -1)

    def make_node(self, x, w, v, gw, gv):
        x, w, v, gw, gv = map(as_tensor_variable, (x, w, v, gw, gv))
        assert x.ndim == 2
        assert w.ndim == 1
        assert v.ndim == 2
        assert gw.ndim == 1
        assert gv.ndim == 2
        out_dtype = theano.scalar.upcast(x.dtype, w.dtype, v.dtype,
                                         gw.dtype, gv.dtype)
        out = theano.tensor.matrix(dtype=out_dtype)
        return Apply(self, [x, w, v, gw, gv], [out])

    def perform(self, node, inputs, outputs):
        """
        Implements the "reverse-mode" gradient for the eigensystem of
        a square matrix.
        """
        x, w, v, W, V = inputs
        N = x.shape[0]
        outer = numpy.outer

        G = lambda n: sum(v[:, m] * V.T[n].dot(v[:, m]) / (w[n] - w[m])
                          for m in xrange(N) if m != n)
        g = sum(outer(v[:, n], v[:, n] * W[n] + G(n))
                for n in xrange(N))

        # Numpy's eigh(a, 'L') (eigh(a, 'U')) is a function of tril(a)
        # (triu(a)) only.  This means that partial derivative of
        # eigh(a, 'L') (eigh(a, 'U')) with respect to a[i,j] is zero
        # for i < j (i > j).  At the same time, non-zero components of
        # the gradient must account for the fact that variation of the
        # opposite triangle contributes to variation of two elements
        # of Hermitian (symmetric) matrix. The following line
        # implements the necessary logic.
        out = self.tri0(g) + self.tri1(g).T

        # The call to self.tri0 in perform upcast from float32 to
        # float64 or from int* to int64 in numpy 1.6.1 but not in
        # 1.6.2. We do not want version dependent dtype in Theano.
        # We think it should be the same as the output.
        outputs[0][0] = numpy.asarray(out, dtype=node.outputs[0].dtype)

    def infer_shape(self, node, shapes):
        return [shapes[0]]


def eigh(a, UPLO='L'):
    return Eigh(UPLO)(a)


class QRFull(Op):
    """
    Full QR Decomposition.
    Computes the QR decomposition of a matrix.
    Factor the matrix a as qr, where q is orthonormal
    and r is upper-triangular.
    """
    _numop = staticmethod(numpy.linalg.qr)
    __props__ = ('mode',)

    def __init__(self, mode):
        self.mode = mode

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2, "The input of qr function should be a matrix."
        q = theano.tensor.matrix(dtype=x.dtype)
        r = theano.tensor.matrix(dtype=x.dtype)
        return Apply(self, [x], [q, r])

    def perform(self, node, (x,), (q, r)):
        assert x.ndim == 2, "The input of qr function should be a matrix."

        q[0], r[0] = self._numop(x,
                                 self.mode)


class QRIncomplete(Op):
    """
    Incomplete QR Decomposition.
    Computes the QR decomposition of a matrix.
    Factor the matrix a as qr and return a single matrix.
    """
    _numop = staticmethod(numpy.linalg.qr)
    __props__ = ('mode',)

    def __init__(self, mode):
        self.mode = mode

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2, "The input of qr function should be a matrix."
        q = theano.tensor.matrix(dtype=x.dtype)
        return Apply(self, [x], [q])

    def perform(self, node, (x,), (q,)):
        assert x.ndim == 2, "The input of qr function should be a matrix."
        q[0] = self._numop(x,
                           self.mode)


def qr(a, mode="full"):
    """
    Computes the QR decomposition of a matrix.
    Factor the matrix a as qr, where q
    is orthonormal and r is upper-triangular.

    :type a:
        array_like, shape (M, N)
    :param a:
        Matrix to be factored.

    :type mode:
        one of 'reduced', 'complete', 'r', 'raw', 'full' and
        'economic', optional
    :keyword mode:
        If K = min(M, N), then

        'reduced'
          returns q, r with dimensions (M, K), (K, N)

        'complete'
           returns q, r with dimensions (M, M), (M, N)

        'r'
          returns r only with dimensions (K, N)

        'raw'
          returns h, tau with dimensions (N, M), (K,)

        'full'
          alias of 'reduced', deprecated (default)

        'economic'
          returns h from 'raw', deprecated.

        The options 'reduced', 'complete', and 'raw' are new in numpy
        1.8, see the notes for more information. The default is
        'reduced' and to maintain backward compatibility with earlier
        versions of numpy both it and the old default 'full' can be
        omitted. Note that array h returned in 'raw' mode is
        transposed for calling Fortran. The 'economic' mode is
        deprecated. The modes 'full' and 'economic' may be passed
        using only the first letter for backwards compatibility, but
        all others must be spelled out.

        Default mode is 'full' which is also default for numpy 1.6.1.

        :note: Default mode was left to full as full and reduced are
           both doing the same thing in the new numpy version but only
           full works on the old previous numpy version.

    :rtype q:
      matrix of float or complex, optional
    :return q:
      A matrix with orthonormal columns. When mode = 'complete' the
      result is an orthogonal/unitary matrix depending on whether or
      not a is real/complex. The determinant may be either +/- 1 in
      that case.

    :rtype r:
      matrix of float or complex, optional
    :return r:
      The upper-triangular matrix.
    """
    x = [[2, 1], [3, 4]]
    if isinstance(numpy.linalg.qr(x,mode), tuple):
        return QRFull(mode)(a)
    else:
        return QRIncomplete(mode)(a)


class SVD(Op):

    # See doc in the docstring of the function just after this class.
    _numop = staticmethod(numpy.linalg.svd)
    __props__ = ('full_matrices', 'compute_uv')

    def __init__(self, full_matrices=True, compute_uv=True):
        """
        full_matrices : bool, optional
            If True (default), u and v have the shapes (M, M) and (N, N),
            respectively.
            Otherwise, the shapes are (M, K) and (K, N), respectively,
            where K = min(M, N).
        compute_uv : bool, optional
            Whether or not to compute u and v in addition to s.
            True by default.
        """
        self.full_matrices = full_matrices
        self.compute_uv = compute_uv

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2, "The input of svd function should be a matrix."
        w = theano.tensor.matrix(dtype=x.dtype)
        u = theano.tensor.vector(dtype=x.dtype)
        v = theano.tensor.matrix(dtype=x.dtype)
        return Apply(self, [x], [w, u, v])

    def perform(self, node, (x,), (w, u, v)):
        assert x.ndim == 2, "The input of svd function should be a matrix."
        w[0], u[0], v[0] = self._numop(x,
                                       self.full_matrices,
                                       self.compute_uv)


def svd(a, full_matrices=1, compute_uv=1):
    """
    This function performs the SVD on CPU.

    :type full_matrices: bool, optional
    :param full_matrices:
        If True (default), u and v have the shapes (M, M) and (N, N),
        respectively.
        Otherwise, the shapes are (M, K) and (K, N), respectively,
        where K = min(M, N).
    :type compute_uv: bool, optional
    :param compute_uv:
        Whether or not to compute u and v in addition to s.
        True by default.

    :returns: U, V and D matrices.
    """
    return SVD(full_matrices, compute_uv)(a)


def test_matrix_inverse_solve():
    if not imported_scipy:
        raise SkipTest("Scipy needed for the Solve op.")
    A = theano.tensor.dmatrix('A')
    b = theano.tensor.dmatrix('b')
    node = matrix_inverse(A).dot(b).owner
    [out] = inv_as_solve.transform(node)
    assert isinstance(out.owner.op, Solve)


class lstsq(Op):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, x, y, rcond):
        x = theano.tensor.as_tensor_variable(x)
        y = theano.tensor.as_tensor_variable(y)
        rcond = theano.tensor.as_tensor_variable(rcond)
        return theano.Apply(self, [x, y, rcond],
                            [theano.tensor.matrix(), theano.tensor.dvector(),
                             theano.tensor.lscalar(), theano.tensor.dvector()])

    def perform(self, node, inputs, outputs):
        x = inputs[0]
        y = inputs[1]
        rcond = inputs[2]
        zz = numpy.linalg.lstsq(inputs[0], inputs[1], inputs[2])
        outputs[0][0] = zz[0]
        outputs[1][0] = zz[1]
        outputs[2][0] = numpy.array(zz[2])
        outputs[3][0] = zz[3]


def matrix_power(M, n):
    result = 1
    for i in xrange(n):
        result = theano.dot(result, M)
    return result


def norm(x,ord):
    x = as_tensor_variable(x)
    ndim = x.ndim
    if ndim == 0:
        raise ValueError("'axis' entry is out of bounds.")
    elif ndim == 1:
        if ord is None:
            return tensor.sum(x**2)**0.5
        elif ord == 'inf':
            return tensor.max(abs(x))
        elif ord == '-inf':
            return tensor.min(abs(x))
        elif ord == 0:
            return x[x.nonzero()].shape[0]
        else:
            try:
                z = tensor.sum(abs(x**ord))**(1./ord)
            except TypeError:
                raise ValueError("Invalid norm order for vectors.")
            return z
    elif ndim == 2:
        if ord is None or ord == 'fro':
            return tensor.sum(abs(x**2))**(0.5)
        elif ord == 'inf':
            return tensor.max(tensor.sum(abs(x), 1))
        elif ord == '-inf':
            return tensor.min(tensor.sum(abs(x), 1))
        elif ord == 1:
            return tensor.max(tensor.sum(abs(x), 0))
        elif ord == -1:
            return tensor.min(tensor.sum(abs(x),0))
        else:
            raise ValueError(0)
    elif ndim > 2:
        raise NotImplementedError("We don't support norm witn ndim > 2")

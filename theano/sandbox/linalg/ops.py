import logging

logger = logging.getLogger(__name__)
import numpy

from theano.gof import Op, Apply

from theano.tensor import as_tensor_variable, dot, DimShuffle, Dot
from theano.tensor.blas import Dot22
from theano import tensor
import theano.tensor
from theano.tensor.opt import (register_stabilize,
        register_specialize, register_canonicalize)
from theano.gof import local_optimizer
from theano.gof.opt import Optimizer
from theano.gradient import DisconnectedType
from theano.tensor.nlinalg import ( MatrixInverse,
                                    matrix_inverse,
                                    AllocDiag,
                                    alloc_diag,
                                    ExtractDiag,
                                    extract_diag,
                                    diag,
                                    trace,
                                    Det,
                                    det,
                                    Eig,
                                    eig,
                                    Eigh,
                                    EighGrad,
                                    eigh,
                                    matrix_dot,
                                    _zero_disconnected
                                    )

try:
    import scipy.linalg
    imported_scipy = True
except ImportError:
    # some ops (e.g. Cholesky, Solve, A_Xinv_b) won't work
    imported_scipy = False


class Hint(Op):
    """
    Provide arbitrary information to the optimizer

    These ops are removed from the graph during canonicalization
    in order to not interfere with other optimizations.
    The idea is that prior to canonicalization, one or more Features of the
    fgraph should register the information contained in any Hint node, and
    transfer that information out of the graph.

    """
    def __init__(self, **kwargs):
        self.hints = tuple(kwargs.items())
        self.view_map = {0: [0]}

    def __eq__(self, other):
        return type(self) == type(other) and self.hints == other.hints

    def __hash__(self):
        return hash((type(self), self.hints))

    def make_node(self, x):
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outstor):
        outstor[0][0] = inputs[0]

    def grad(self, inputs, g_out):
        return g_out


def is_hint_node(node):
    return isinstance(node.op, Hint)


def hints(variable):
    if hasattr(variable, 'fgraph'):
        try:
            return variable.fgraph.hints_feature.hints[variable]
        except AttributeError:
            return {}
    else:
        if is_hint_node(variable.owner):
            return dict(variable.owner.op.hints)
        else:
            return {}


@register_canonicalize
@local_optimizer([Hint])
def remove_hint_nodes(node):
    if is_hint_node(node):
        # transfer hints from graph to Feature
        try:
            for k, v in node.op.hints:
                node.fgraph.hints_feature.add_hint(node.inputs[0], k, v)
        except AttributeError:
            pass
        return node.inputs


class HintsFeature(object):
    """
    FunctionGraph Feature to track matrix properties

    This is a similar feature to variable 'tags'. In fact, tags are one way
    to provide hints.

    This class exists because tags were not documented well, and the
    semantics of how tag information should be moved around during
    optimizations was never clearly spelled out.

    Hints are assumptions about mathematical properties of variables.
    If one variable is substituted for another by an optimization,
    then it means that the assumptions should be transferred to the
    new variable.

    Hints are attached to 'positions in a graph' rather than to variables
    in particular, although Hints are originally attached to a particular
    positition in a graph *via* a variable in that original graph.

    Examples of hints are:
    - shape information
    - matrix properties (e.g. symmetry, psd, banded, diagonal)

    Hint information is propagated through the graph similarly to graph
    optimizations, except that adding a hint does not change the graph.
    Adding a hint is not something that debugmode will check.

    #TODO: should a Hint be an object that can actually evaluate its
    #      truthfulness?
    #      Should the PSD property be an object that can check the
    #      PSD-ness of a variable?

    """
    def add_hint(self, r, k, v):
        logger.debug('adding hint; %s, %s, %s' % (r, k, v))
        self.hints[r][k] = v

    def ensure_init_r(self, r):
        if r not in self.hints:
            self.hints[r] = {}

    #
    #
    # Feature inteface
    #
    #
    def on_attach(self, fgraph):
        assert not hasattr(fgraph, 'hints_feature')
        fgraph.hints_feature = self
        # Variable -> tuple(scalars) or None  (All tensor vars map to tuple)
        self.hints = {}
        for node in fgraph.toposort():
            self.on_import(fgraph, node, "on_attach")

    def on_import(self, fgraph, node, reason):
        if node.outputs[0] in self.hints:
            # this is a revert, not really an import
            for r in node.outputs + node.inputs:
                assert r in self.hints
            return

        for i, r in enumerate(node.inputs + node.outputs):
            # make sure we have shapes for the inputs
            self.ensure_init_r(r)

    def update_second_from_first(self, r0, r1):
        old_hints = self.hints[r0]
        new_hints = self.hints[r1]
        for k, v in old_hints.items():
            if k in new_hints and new_hints[k] is not v:
                raise NotImplementedError()
            if k not in new_hints:
                new_hints[k] = v

    def on_change_input(self, fgraph, node, i, r, new_r, reason):
        # TODO:
        # This tells us that r and new_r must have the same shape
        # if we didn't know that the shapes are related, now we do.
        self.ensure_init_r(new_r)
        self.update_second_from_first(r, new_r)
        self.update_second_from_first(new_r, r)

        # change_input happens in two cases:
        # 1) we are trying to get rid of r, or
        # 2) we are putting things back after a failed transaction.


class HintsOptimizer(Optimizer):
    """Optimizer that serves to add HintsFeature as an fgraph feature.
    """
    def __init__(self):
        Optimizer.__init__(self)

    def add_requirements(self, fgraph):
        fgraph.attach_feature(HintsFeature())

    def apply(self, fgraph):
        pass
# -1 should make it run right before the first merge
theano.compile.mode.optdb.register('HintsOpt',
                                   HintsOptimizer(),
                                   -1,
                                   'fast_run',
                                   'fast_compile')


def psd(v):
    """
    Apply a hint that the variable `v` is positive semi-definite, i.e.
    it is a symmetric matrix and :math:`x^T A x \ge 0` for any vector x.
    """
    return Hint(psd=True, symmetric=True)(v)


def is_psd(v):
    return hints(v).get('psd', False)


def is_symmetric(v):
    return hints(v).get('symmetric', False)


def is_positive(v):
    if hints(v).get('positive', False):
        return True
    #TODO: how to handle this - a registry?
    #      infer_hints on Ops?
    logger.debug('is_positive: %s' % str(v))
    if v.owner and v.owner.op == tensor.pow:
        try:
            exponent = tensor.get_scalar_constant_value(v.owner.inputs[1])
        except tensor.basic.NotScalarConstantError:
            return False
        if 0 == exponent % 2:
            return True
    return False


@register_stabilize
@local_optimizer([Dot, Dot22])
def inv_as_solve(node):
    if not imported_scipy:
        return False
    if isinstance(node.op, (Dot, Dot22)):
        l, r = node.inputs
        if l.owner and l.owner.op == matrix_inverse:
            return [solve(l.owner.inputs[0], r)]
        if r.owner and r.owner.op == matrix_inverse:
            if is_symmetric(r.owner.inputs[0]):
                return [solve(r.owner.inputs[0], l.T).T]
            else:
                return [solve(r.owner.inputs[0].T, l.T).T]


@register_canonicalize
@register_stabilize
@register_specialize
@local_optimizer([DimShuffle])
def no_transpose_symmetric(node):
    if isinstance(node.op, DimShuffle):
        x = node.inputs[0]
        if x.type.ndim == 2 and is_symmetric(x):
            #print 'UNDOING TRANSPOSE', is_symmetric(x), x.ndim
            if node.op.new_order == [1, 0]:
                return [x]


@register_stabilize
@local_optimizer(None) # XXX: solve is defined later and can't be used here
def psd_solve_with_chol(node):
    if node.op == solve:
        A, b = node.inputs  # result is solution Ax=b
        if is_psd(A):
            L = cholesky(A)
            #N.B. this can be further reduced to a yet-unwritten cho_solve Op
            #     __if__ no other Op makes use of the the L matrix during the
            #     stabilization
            Li_b = Solve('lower_triangular')(L, b)
            x = Solve('upper_triangular')(L.T, Li_b)
            return [x]


@register_stabilize
@register_specialize
@local_optimizer(None) # XXX: det is defined later and can't be used here
def local_det_chol(node):
    """
    If we have det(X) and there is already an L=cholesky(X)
    floating around, then we can use prod(diag(L)) to get the determinant.

    """
    if node.op == det:
        x, = node.inputs
        for (cl, xpos) in x.clients:
            if isinstance(cl.op, Cholesky):
                L = cl.outputs[0]
                return [tensor.prod(extract_diag(L) ** 2)]


@register_canonicalize
@register_stabilize
@register_specialize
@local_optimizer([tensor.log])
def local_log_prod_sqr(node):
    if node.op == tensor.log:
        x, = node.inputs
        if x.owner and isinstance(x.owner.op, tensor.elemwise.Prod):
            # we cannot always make this substitution because
            # the prod might include negative terms
            p = x.owner.inputs[0]

            # p is the matrix we're reducing with prod
            if is_positive(p):
                return [tensor.log(p).sum(axis=x.owner.op.axis)]

            #TODO: have a reduction like prod and sum that simply
            #      returns the sign of the prod multiplication.


@register_canonicalize
@register_stabilize
@register_specialize
@local_optimizer([tensor.log])
def local_log_pow(node):
    if node.op == tensor.log:
        x, = node.inputs
        if x.owner and x.owner.op == tensor.pow:
            base, exponent = x.owner.inputs
            #TODO: reason to be careful with dtypes?
            return [exponent * tensor.log(base)]


MATRIX_STRUCTURES = (
        'general',
        'symmetric',
        'lower_triangular',
        'upper_triangular',
        'hermitian',
        'banded',
        'diagonal',
        'toeplitz',
        )


class Cholesky(Op):
    """
    Return a triangular matrix square root of positive semi-definite `x`

    L = cholesky(X, lower=True) implies dot(L, L.T) == X
    """
    #TODO: inplace
    #TODO: for specific dtypes
    #TODO: LAPACK wrapper with in-place behavior, for solve also
    def __init__(self, lower=True):
        self.lower = lower
        self.destructive = False

    def props(self):
        return (self.lower,
                self.destructive)

    def __hash__(self):
        return hash((type(self), self.props()))

    def __eq__(self, other):
        return (type(self) == type(other) and self.props() == other.props())

    def infer_shape(self, node, shapes):
        return [shapes[0]]

    def __str__(self):
        if self.lower:
            lu = 'lower'
        else:
            lu = 'upper'
        if self.destructive:
            destr = 'destructive'
        else:
            destr = 'non-destructive'
        return 'Cholesky{%s,%s}' % (lu, destr)

    def make_node(self, x):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the Cholesky op")
        x = as_tensor_variable(x)
        assert x.ndim == 2
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        x = inputs[0]
        z = outputs[0]
        z[0] = scipy.linalg.cholesky(x, lower=self.lower).astype(x.dtype)

    def grad(self, inputs, gradients):
        return [CholeskyGrad(self.lower)(inputs[0], self(inputs[0]),
                                         gradients[0])]

cholesky = Cholesky()


class CholeskyGrad(Op):
    """
    """
    def __init__(self, lower=True):
        self.lower = lower
        self.destructive = False

    def props(self):
        return (self.lower,
                self.destructive)

    def __hash__(self):
        return hash((type(self), self.props()))

    def __eq__(self, other):
        return (type(self) == type(other) and self.props() == other.props())

    def __str__(self):
        if self.lower:
            lu = 'lower'
        else:
            lu = 'upper'
        if self.destructive:
            destr = 'destructive'
        else:
            destr = 'non-destructive'
        return 'CholeskyGrad{%s,%s}' % (lu, destr)

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
        """Implements the "reverse-mode" gradient [1]_ for the
        Cholesky factorization of a positive-definite matrix.

        .. [1] S. P. Smith. "Differentiation of the Cholesky Algorithm".
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
            F = numpy.tril(dz)
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
            F = numpy.triu(dz)
            M = N - 1
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


class MatrixPinv(Op):
    """Computes the pseudo-inverse of a matrix :math:`A`.

    The pseudo-inverse of a matrix A, denoted :math:`A^+`, is
    defined as: "the matrix that 'solves' [the least-squares problem]
    :math:`Ax = b`," i.e., if :math:`\\bar{x}` is said solution, then
    :math:`A^+` is that matrix such that :math:`\\bar{x} = A^+b`.

    Note that :math:`Ax=AA^+b`, so :math:`AA^+` is close to the identity
    matrix.
    This method is not faster then `matrix_inverse`. Its strength comes from
    that it works for non-square matrices.
    If you have a square matrix though, `matrix_inverse` can be both more
    exact and faster to compute. Also this op does not get optimized into a
    solve op.
    """
    def __init__(self):
        pass

    def props(self):
        """Function exposing different properties of each instance of the
        op.

        For the ``MatrixPinv`` op, there are no properties to be exposed.
        """
        return ()

    def __hash__(self):
        return hash((type(self), self.props()))

    def __eq__(self, other):
        return (type(self) == type(other) and self.props() == other.props())

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        return Apply(self, [x], [x.type()])

    def perform(self, node, (x,), (z, )):
        if imported_scipy:
            z[0] = scipy.linalg.pinv(x).astype(x.dtype)
        else:
            z[0] = numpy.linalg.pinv(x).astype(x.dtype)

    def __str__(self):
        return "MatrixPseudoInverse"

pinv = MatrixPinv()


class Solve(Op):
    """Solve a system of linear equations"""
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

    def props(self):
        return (self.A_structure,
                self.lower,
                self.overwrite_A,
                self.overwrite_b)

    def __hash__(self):
        return hash((type(self), self.props()))

    def __eq__(self, other):
        return type(self) == type(other) and self.props() == other.props()

    def __repr__(self):
        return 'Solve{%s}' % str(self.props())

    def make_node(self, A, b):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the Solve op")
        A = as_tensor_variable(A)
        b = as_tensor_variable(b)
        assert A.ndim == 2
        assert b.ndim in [1, 2]
        otype = tensor.tensor(
                broadcastable=b.broadcastable,
                dtype=(A * b).dtype)
        return Apply(self, [A, b], [otype])

    def perform(self, node, inputs, output_storage):
        A, b = inputs
        #TODO: use the A_structure to go faster
        output_storage[0][0] = scipy.linalg.solve(A, b)

    # computes shape of x where x = inv(A) * b
    def infer_shape(self, node, shapes):
        Ashape, Bshape = shapes
        rows = Ashape[1]
        if len(Bshape) == 1:  # b is a Vector
            return [(rows,)]
        else:
            cols = Bshape[1]  # b is a Matrix
            return [(rows, cols)]

solve = Solve()  # general solve

#TODO : SolveTriangular

#TODO: Optimizations to replace multiplication by matrix inverse
#      with solve() Op (still unwritten)


def spectral_radius_bound(X, log2_exponent):
    """
    Returns upper bound on the largest eigenvalue of square symmetrix matrix X.

    log2_exponent must be a positive-valued integer. The larger it is, the
    slower and tighter the bound.  Values up to 5 should usually suffice.  The
    algorithm works by multiplying X by itself this many times.

    From V.Pan, 1990. "Estimating the Extremal Eigenvalues of a Symmetric
    Matrix", Computers Math Applic. Vol 20 n. 2 pp 17-22.
    Rq: an efficient algorithm, not used here, is defined in this paper.
    """
    if X.type.ndim != 2:
        raise TypeError('spectral_radius_bound requires a matrix argument', X)
    if not isinstance(log2_exponent, int):
        raise TypeError('spectral_radius_bound requires an integer exponent',
                        log2_exponent)
    if log2_exponent <= 0:
        raise ValueError('spectral_radius_bound requires a strictly positive '
                         'exponent', log2_exponent)

    XX = X
    for i in xrange(log2_exponent):
        XX = tensor.dot(XX, XX)
    return tensor.pow(
            trace(XX),
            2 ** (-log2_exponent))


class Eig(Op):
    """Compute the eigenvalues and right eigenvectors of a square array.

    """
    _numop = staticmethod(numpy.linalg.eig)

    def props(self):
        """Function exposing different properties of each instance of the
        op.

        For the ``Eig`` op, there are no properties to be exposed.
        """
        return ()

    def __hash__(self):
        return hash((type(self), self.props()))

    def __eq__(self, other):
        return (type(self) == type(other) and self.props() == other.props())

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

    def __str__(self):
        return self._numop.__name__.capitalize()

eig = Eig()


class SVD(Op):

    # See doc in the docstring of the function just after this class.
    _numop = staticmethod(numpy.linalg.svd)

    def __init__(self, full_matrices=True, compute_uv=True):
        """
        inputs :
        --------
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

    def __hash__(self):
        return hash((type(self), self.props()))

    def __eq__(self, other):
        return (type(self) == type(other) and self.props() == other.props())

    def props(self):
        return self.full_matrices, self.compute_uv,

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2, "The input of svd function should be a matrix."
        w = theano.tensor.matrix(dtype=x.dtype)
        u = theano.tensor.matrix(dtype=x.dtype)
        v = theano.tensor.matrix(dtype=x.dtype)
        return Apply(self, [x], [w, u, v])

    def perform(self, node, (x,), (w, u, v)):
        assert x.ndim == 2, "The input of svd function should be a matrix."
        w[0], u[0], v[0] = self._numop(x,
                                       self.full_matrices,
                                       self.compute_uv)

    def __str__(self):
        return self._numop.__name__.capitalize()


def svd(a, full_matrices=1, compute_uv=1):
    """
    This function performs the SVD on CPU.

    Parameters :
    ------------

    full_matrices : bool, optional
        If True (default), u and v have the shapes (M, M) and (N, N),
        respectively.
        Otherwise, the shapes are (M, K) and (K, N), respectively,
        where K = min(M, N).
    compute_uv : bool, optional
        Whether or not to compute u and v in addition to s.
        True by default.

    Returns :
    -------
    U, V and D matrices.
    """
    return SVD(full_matrices, compute_uv)(a)


class QRFull(Op):
    """
    Full QR Decomposition.
    Computes the QR decomposition of a matrix.
    Factor the matrix a as qr, where q is orthonormal
    and r is upper-triangular.
    """
    _numop = staticmethod(numpy.linalg.qr)

    def __init__(self, mode):
        self.mode = mode

    def __hash__(self):
        return hash((type(self), self.props()))

    def __eq__(self, other):
        return (type(self) == type(other) and self.props() == other.props())

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2, "The input of qr function should be a matrix."
        q = theano.tensor.matrix(dtype=x.dtype)
        r = theano.tensor.matrix(dtype=x.dtype)
        return Apply(self, [x], [q, r])

    def props(self):
        return self.mode

    def perform(self, node, (x,), (q, r)):
        assert x.ndim == 2, "The input of qr function should be a matrix."

        q[0], r[0] = self._numop(x,
                                 self.mode)

    def __str__(self):
        return self._numop.__class__.__name__


class QRIncomplete(Op):
    """
    Incomplete QR Decomposition.
    Computes the QR decomposition of a matrix.
    Factor the matrix a as qr and return a single matrix.
    """
    _numop = staticmethod(numpy.linalg.qr)

    def __init__(self, mode):
        self.mode = mode

    def __hash__(self):
        return hash((type(self), self.props()))

    def __eq__(self, other):
        return (type(self) == type(other) and self.props() == other.props())

    def props(self):
        return self.mode

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2, "The input of qr function should be a matrix."
        q = theano.tensor.matrix(dtype=x.dtype)
        return Apply(self, [x], [q])

    def perform(self, node, (x,), (q,)):
        assert x.ndim == 2, "The input of qr function should be a matrix."
        q[0] = self._numop(x,
                           self.mode)

    def __str__(self):
        return self._numop.__class__.__name__


def qr(a, mode="full"):
    """
    Computes the QR decomposition of a matrix.
    Factor the matrix a as qr, where q
    is orthonormal and r is upper-triangular.

    Parameters :
    ------------

    a : array_like, shape (M, N)
        Matrix to be factored.

    mode : {'reduced', 'complete', 'r', 'raw', 'full', 'economic'}, optional
        If K = min(M, N), then
        'reduced' : returns q, r with dimensions (M, K), (K, N) (default)
        'complete' : returns q, r with dimensions (M, M), (M, N)
        'r' : returns r only with dimensions (K, N)
        'raw' : returns h, tau with dimensions (N, M), (K,)
        'full' : alias of 'reduced', deprecated
        'economic' : returns h from 'raw', deprecated. The options 'reduced',
        'complete', and 'raw' are new in numpy 1.8, see the notes for more
        information. The default is 'reduced' and to maintain backward
        compatibility with earlier versions of numpy both it and the old
        default 'full' can be omitted. Note that array h returned in 'raw'
        mode is transposed for calling Fortran. The 'economic' mode is
        deprecated. The modes 'full' and 'economic' may be passed using only
        the first letter for backwards compatibility, but all others
        must be spelled out.
        Default mode is 'full' which is also default for numpy 1.6.1.

        Note:   Default mode was left to full as full and reduced are both doing
                the same thing in the new numpy version but only full works on the old
                previous numpy version.
    Returns :
    ---------
    q : matrix of float or complex, optional
    A matrix with orthonormal columns. When mode = 'complete'
    the result is an orthogonal/unitary matrix depending on whether
    or not a is real/complex. The determinant may be either +/- 1 in that case.

    r : matrix of float or complex, optional
    The upper-triangular matrix.

    """
    x = [[2, 1], [3, 4]]
    if isinstance(numpy.linalg.qr(x,mode), tuple):
        return QRFull(mode)(a)
    else:
        return QRIncomplete(mode)(a)


def _zero_disconnected(outputs, grads):
    l = []
    for o, g in zip(outputs, grads):
        if isinstance(g.type, DisconnectedType):
            l.append(o.zeros_like())
        else:
            l.append(g)
    return l


class Eigh(Eig):
    """
    Return the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.

    """
    _numop = staticmethod(numpy.linalg.eigh)

    def __init__(self, UPLO='L'):
        assert UPLO in ['L', 'U']
        self.UPLO = UPLO

    def __str__(self):
        return 'Eigh{%s}' % self.UPLO

    def props(self):
        return self.UPLO,

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
        try:
            w[0], v[0] = self._numop(x, self.UPLO)
        except numpy.linalg.LinAlgError:
            logger.debug('Failed to find %s of %s' % (self._numop.__name__,
                                                      node.inputs[0]))
            raise

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


def eigh(a, UPLO='L'):
    return Eigh(UPLO)(a)


class EighGrad(Op):
    """Gradient of an eigensystem of a Hermitian matrix.

    """
    def __init__(self, UPLO='L'):
        assert UPLO in ['L', 'U']
        self.UPLO = UPLO
        if UPLO == 'L':
            self.tri0 = numpy.tril
            self.tri1 = lambda a: numpy.triu(a, 1)
        else:
            self.tri0 = numpy.triu
            self.tri1 = lambda a: numpy.tril(a, -1)

    def props(self):
        return (self.UPLO,)

    def __hash__(self):
        return hash((type(self), self.props()))

    def __eq__(self, other):
        return (type(self) == type(other) and self.props() == other.props())

    def __str__(self):
        return 'EighGrad{%s}' % self.UPLO

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
        r"""
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


class Eigvalsh(Op):
    """Generalized eigenvalues of a Hermetian positive definite eigensystem
    """

    def __init__(self, lower=True):
        assert lower in [True, False]
        self.lower = lower

    def props(self):
        return (self.lower,)

    def __hash__(self):
        return hash((type(self), self.props()))

    def __eq__(self, other):
        return (type(self) == type(other) and self.props() == other.props())

    def make_node(self, a, b):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the Eigvalsh op")
        a = as_tensor_variable(a)
        assert a.ndim == 2
        if not isinstance(b, (theano.Variable)):
            if b is None:
                b = theano.tensor.NoneConst
                out_dtype = a.dtype
            else:
                b = as_tensor_variable(b)
                out_dtype = theano.scalar.upcast(a.dtype, b.dtype)
        elif not isinstance(b.type, theano.tensor.NoneTypeT):
            b = as_tensor_variable(b)
            out_dtype = theano.scalar.upcast(a.dtype, b.dtype)
            assert b.ndim == 2
        else:
            out_dtype = a.dtype

        w = theano.tensor.vector(dtype=out_dtype)
        return Apply(self, [a, b], [w])

    def perform(self, node, (a, b), (w,)):
        w[0] = scipy.linalg.eigvalsh(a=a, b=b, lower=self.lower)

    def grad(self, inputs, g_outputs):
        a, b = inputs
        gw, = g_outputs
        return EigvalshGrad(self.lower)(a, b, gw)

    def infer_shape(self, node, shapes):
        n = shapes[0][0]
        return [(n,)]


class EigvalshGrad(Op):
    """Gradient of generalized eigenvalues of a Hermetian positive definite
    eigensystem
    """

    # Note: This Op (EigvalshGrad), should be removed and replaced with a graph
    # of theano ops that is constructed directly in Eigvalsh.grad.
    # But this can only be done once scipy.linalg.eigh is available as an Op
    # (currently the Eigh uses numpy.linalg.eigh, which doesn't let you
    # pass the right-hand-side matrix for a generalized eigenproblem.) See the
    # discussion on github at
    # https://github.com/Theano/Theano/pull/1846#discussion-diff-12486764

    def __init__(self, lower=True):
        assert lower in [True, False]
        self.lower = lower
        if lower:
            self.tri0 = numpy.tril
            self.tri1 = lambda a: numpy.triu(a, 1)
        else:
            self.tri0 = numpy.triu
            self.tri1 = lambda a: numpy.tril(a, -1)

    def props(self):
        return (self.lower,)

    def __hash__(self):
        return hash((type(self), self.props()))

    def __eq__(self, other):
        return (type(self) == type(other) and self.props() == other.props())

    def make_node(self, a, b, gw):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the GEigvalsh op")
        a, b, gw = map(as_tensor_variable, (a, b, gw))
        assert a.ndim == 2
        assert b.ndim == 2
        assert gw.ndim == 1

        out_dtype = theano.scalar.upcast(a.dtype, b.dtype, gw.dtype)
        out1 = theano.tensor.matrix(dtype=out_dtype)
        out2 = theano.tensor.matrix(dtype=out_dtype)
        return Apply(self, [a, b, gw], [out1, out2])

    def perform(self, node, (a, b, gw), outputs):
        w, v = scipy.linalg.eigh(a, b, lower=self.lower)
        gA = v.dot(numpy.diag(gw).dot(v.T))
        gB = - v.dot(numpy.diag(gw*w).dot(v.T))

        # See EighGrad comments for an explanation of these lines
        out1 = self.tri0(gA) + self.tri1(gA).T
        out2 = self.tri0(gB) + self.tri1(gB).T
        outputs[0][0] = numpy.asarray(out1, dtype=node.outputs[0].dtype)
        outputs[1][0] = numpy.asarray(out2, dtype=node.outputs[1].dtype)

    def infer_shape(self, node, shapes):
        return [shapes[0], shapes[1]]


def eigvalsh(a, b, lower=True):
    return Eigvalsh(lower)(a, b)


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
        if ord == None:
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
        if ord == None or ord == 'fro':
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
            raise ValueError()
    elif ndim > 2:
        raise NotImplementedError("We don't support norm witn ndim > 2")


class lstsq(theano.Op):
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
        return theano.Apply(self, [x, y, rcond], [y.type(), theano.tensor.dvector(), theano.tensor.lscalar(), theano.tensor.dvector()])

    def perform(self, node, inputs, outputs):
        x = inputs[0]
        y = inputs[1]
        rcond = inputs[2]
        zz = numpy.linalg.lstsq(inputs[0], inputs[1], inputs[2])            
        outputs[0][0] = zz[0]
        outputs[1][0] = zz[1]
        outputs[2][0] = zz[2]
        outputs[3][0] = zz[3]

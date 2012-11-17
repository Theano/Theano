import logging

logger = logging.getLogger(__name__)
import numpy

from theano.gof import Op, Apply

from theano.tensor import as_tensor_variable, dot, DimShuffle
from theano import tensor
import theano.tensor
from theano.tensor.opt import (register_stabilize,
        register_specialize, register_canonicalize)
from theano.gof import local_optimizer
from theano.gof.opt import Optimizer
from theano.gradient import grad_not_implemented, DisconnectedType

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
@local_optimizer([])
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
            self.on_import(fgraph, node)

    def on_import(self, fgraph, node):
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

    def on_change_input(self, fgraph, node, i, r, new_r):
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
@local_optimizer([])
def inv_as_solve(node):
    if not imported_scipy:
        return False
    if node.op == dot:
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
@local_optimizer([])
def no_transpose_symmetric(node):
    if isinstance(node.op, DimShuffle):
        x = node.inputs[0]
        if x.type.ndim == 2 and is_symmetric(x):
            #print 'UNDOING TRANSPOSE', is_symmetric(x), x.ndim
            if node.op.new_order == [1, 0]:
                return [x]


@register_stabilize
@local_optimizer([])
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
@local_optimizer([])
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
@local_optimizer([])
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
@local_optimizer([])
def local_log_pow(node):
    if node.op == tensor.log:
        x, = node.inputs
        if x.owner and x.owner.op == tensor.pow:
            base, exponent = x.owner.inputs
            #TODO: reason to be careful with dtypes?
            return [exponent * tensor.log(base)]


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
        assert l.owner.op.lower == self.lower, (
            "lower/upper mismatch between Cholesky op and CholeskyGrad op"
        )
        return Apply(self, [x, l, dz], [x.type()])

    def perform(self, node, inputs, outputs):
        """
        Implements the "reverse-mode" gradient for the Cholesky factorization
        of a positive-definite matrix.

        References
        ----------
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

    Note that :math:`Ax=AA^+b`, so :math:`AA^+` is close to the identity matrix.
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
        return Apply(self, [x], [x.type()])

    def perform(self, node, (x,), (z, )):
        try:
            if imported_scipy:
                z[0] = scipy.linalg.pinv(x).astype(x.dtype)
            else:
                z[0] = numpy.linalg.pinv(x).astype(x.dtype)
        except numpy.linalg.LinAlgError:
            logger.debug('Failed to invert %s' % str(node.inputs[0]))
            raise

    def __str__(self):
        return "MatrixPseudoInverse"

pinv = MatrixPinv()


class MatrixInverse(Op):
    """Computes the inverse of a matrix :math:`A`.

    Given a square matrix :math:`A`, ``matrix_inverse`` returns a square
    matrix :math:`A_{inv}` such that the dot product :math:`A \cdot A_{inv}`
    and :math:`A_{inv} \cdot A` equals the identity matrix :math:`I`.

    :note: When possible, the call to this op will be optimized to the call
           of ``solve``.
    """

    def __init__(self):
        pass

    def props(self):
        """Function exposing different properties of each instance of the
        op.

        For the ``MatrixInverse`` op, there are no properties to be exposed.
        """
        return ()

    def __hash__(self):
        return hash((type(self), self.props()))

    def __eq__(self, other):
        return (type(self) == type(other) and self.props() == other.props())

    def make_node(self, x):
        x = as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, (x,), (z, )):
        try:
            z[0] = numpy.linalg.inv(x).astype(x.dtype)
        except numpy.linalg.LinAlgError:
            logger.debug('Failed to invert %s' % str(node.inputs[0]))
            raise

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

    def __str__(self):
        return "MatrixInverse"

matrix_inverse = MatrixInverse()


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


class ExtractDiag(Op):
    """ Return the diagonal of a matrix. """
    def __init__(self, view=False):
        self.view = view
        if self.view:
            self.view_map = {0: [0]}

    def __eq__(self, other):
        return type(self) == type(other) and self.view == other.view

    def __hash__(self):
        return hash(type(self)) ^ hash(self.view)

    def make_node(self, _x):
        x = as_tensor_variable(_x)
        if x.type.ndim != 2:
            raise TypeError('ExtractDiag only works on matrices', _x)
        return Apply(self, [x], [tensor.vector(dtype=x.type.dtype)])

    def perform(self, node, ins, outs):
        """ For some reason numpy.diag(x) is really slow, so we
        implemented our own. """
        x, = ins
        z, = outs

        # zero-dimensional matrices ...
        if x.shape[0] == 0 or x.shape[1] == 0:
            z[0] = numpy.zeros(0, dtype=x.dtype)
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
        x = tensor.zeros_like(inputs[0])
        xdiag = alloc_diag(g_outputs[0])
        return [tensor.set_subtensor(
            x[:xdiag.shape[0], :xdiag.shape[1]],
            xdiag)]

    def infer_shape(self, node, shapes):
        x_s, = shapes
        shp = tensor.min(node.inputs[0].shape)
        return [(shp,)]

extract_diag = ExtractDiag()
#TODO: optimization to insert ExtractDiag with view=True


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
        return Apply(self, [x], [tensor.matrix(dtype=x.type.dtype)])

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


class Det(Op):
    """Matrix determinant
    Input should be a square matrix
    """
    def make_node(self, x):
        x = as_tensor_variable(x)
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


def trace(X):
    """
    Returns the sum of diagonal elements of matrix X.
    """
    return extract_diag(X).sum()


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


class A_Xinv_b(Op):
    """Product of form a inv(X) b"""
    def make_node(self, a, X, b):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the A_Xinv_b op")
        a = as_tensor_variable(a)
        b = as_tensor_variable(b)
        X = as_tensor_variable(X)
        o = theano.tensor.matrix(dtype=x.dtype)
        return Apply(self, [a, X, b], [o])

    def perform(self, ndoe, inputs, outstor):
        a, X, b = inputs
        if 1:
            L_factor = scipy.linalg.cho_factor(X)
            xb = scipy.linalg.cho_solve(L_factor, b)
            xa = scipy.linalg.cho_solve(L_factor, a.T)
            z = numpy.dot(xa.T, xb)
        else:
            raise NotImplementedError(self.X_structure)
        outstor[0][0] = z

    def grad(self, inputs, g_outputs):
        gz, = g_outputs
        a, X, b = inputs
        iX = matrix_inverse(X)
        ga = matrix_dot(gz, b.T, iX.T)
        gX = -matrix_dot(iX.T, a, gz, b.T, iX.T)
        gb = matrix_dot(ix.T, a.T, gz)
        return [ga, gX, gb]


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
        try:
            w[0], v[0] = [z.astype(x.dtype) for z in self._numop(x)]
        except numpy.linalg.LinAlgError:
            logger.debug('Failed to find %s of %s' % (self._numop.__name__,
                                                      node.inputs[0]))
            raise

    def infer_shape(self, node, shapes):
        n = shapes[0][0]
        return [(n,), (n, n)]

    def __str__(self):
        return self._numop.__name__.capitalize()

eig = Eig()


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

import numpy

from theano.gof import Op, Apply

from theano.tensor import as_tensor_variable, dot, DimShuffle
from theano import tensor
import theano.tensor
from theano.tensor.opt import (register_stabilize,
        register_specialize, register_canonicalize)
from theano.gof import local_optimizer
from theano.gof.opt import Optimizer

try:
    import scipy.linalg
except ImportError:
    pass # some ops (e.g. Cholesky) won't work

class Hint(Op):
    """
    Provide arbitrary information to the optimizer

    These ops are removed from the graph during canonicalization
    in order to not interfere with other optimizations.
    The idea is that prior to canonicalization, one or more Features of the env should
    register the information contained in any Hint node, and transfer that information out of
    the graph.

    """
    def __init__(self, **kwargs):
        self.hints = tuple(kwargs.items())
        self.view_map = {0:[0]}
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
    if hasattr(variable, 'env'):
        try:
            return variable.env.hints_feature.hints[variable]
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
        try:
            for k,v in node.op.hints:
                node.env.hints_feature.add_hint(node.inputs[0], k, v)
        except AttributeError:
            pass
        return node.inputs


class HintsFeature(object):
    """
    Env Feature to track matrix properties

    This is a similar feature to variable 'tags'. In fact, tags are one way to provide hints.

    This class exists because tags were not documented well, and the semantics of how tag
    information should be moved around during optimizations was never clearly spelled out.

    Hints are assumptions about mathematical properties of variables.
    If one variable is substituted for another by an optimization,
    then it means that the assumptions should be transferred to the new variable.

    Hints are attached to 'positions in a graph' rather than to variables in particular,
    although Hints are originally attached to a particular positition in a graph *via* a
    variable in that original graph.

    Examples of hints are:
    - shape information
    - matrix properties (e.g. symmetry, psd, banded, diagonal)

    Hint information is propagated through the graph similarly to graph optimizations,
    except that adding a hint does not change the graph.  Adding a hint is not something that
    debugmode will check.

    #TODO: should a Hint be an object that can actually evaluate its truthfulness?
    #      Should the PSD property be an object that can check the PSD-ness of a variable?

    """
    def add_hint(self, r, k, v):
        print 'adding hint', r, k, v
        self.hints[r][k] = v

    def ensure_init_r(self, r):
        if r not in self.hints:
            self.hints[r] = {}
    #
    #
    # Feature inteface
    #
    #
    def on_attach(self, env):
        assert not hasattr(env, 'hints_feature')
        env.hints_feature = self
        self.hints = {} # Variable -> tuple(scalars) or None  (All tensor vars map to tuple)
        for node in env.toposort():
            self.on_import(env, node)

    def on_import(self, env, node):
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
        for k,v in old_hints.items():
            if k in new_hints and new_hints[k] is not v:
                raise NotImplementedError()
            if k not in new_hints:
                new_hints[k] = v

    def on_change_input(self, env, node, i, r, new_r):
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
    """Optimizer that serves to add HintsFeature as an env feature.
    """
    def __init__(self):
        Optimizer.__init__(self)

    def add_requirements(self, env):
        env.extend(HintsFeature())

    def apply(self, env):
        pass
# -1 should make it run right before the first merge
theano.compile.mode.optdb.register('HintsOpt', HintsOptimizer(), -1, 'fast_run', 'fast_compile')


def PSD_hint(v):
    return Hint(psd=True,symmetric=True)(v)
def is_psd(v):
    return hints(v).get('psd', False)
def is_symmetric(v):
    return hints(v).get('symmetric', False)
def is_positive(v):
    if hints(v).get('positive', False):
        return True
    #TODO: how to handle this - a registry?
    #      infer_hints on Ops?
    print 'is_positive', v
    if v.owner and v.owner.op == tensor.pow:
        print 'try for pow', v, v.owner.inputs
        try:
            exponent = tensor.get_constant_value(v.owner.inputs[1])
        except TypeError:
            return False
        if 0 == exponent % 2:
            return True
    return False


@register_stabilize
@local_optimizer([])
def inv_as_solve(node):
    if node.op == dot:
        l,r = node.inputs
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
        if x.type.ndim==2 and is_symmetric(x):
            #print 'UNDOING TRANSPOSE', is_symmetric(x), x.ndim
            if node.op.new_order == [1,0]:
                return [x]

@register_stabilize
@local_optimizer([])
def psd_solve_with_chol(node):
    if node.op == solve:
        A, b = node.inputs #result is solution Ax=b
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
        for (cl,xpos) in x.clients:
            if isinstance(cl.op, Cholesky):
                L = cl.outputs[0]
                return [tensor.prod(extract_diag(L)**2)]


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
            print "AAA", p

            # p is the matrix we're reducing with prod
            if is_positive(p):
                return [tensor.log(p).sum(axis=x.owner.op.axis)]

            #TODO: have a reduction like prod and sum that simply returns the sign
            #      of the prod multiplication.

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

    L = cholesky(X, lower=True) implies dot(L.T,L)==X
    """
    #TODO: inplace
    #TODO: for specific dtypes
    def __init__(self, lower=True):
        self.lower = lower
        self.destructive = False
    def props(self):
        return (self.lower,
                self.destructive)
    def __hash__(self):
        return hash((type(self), self.props()))
    def __eq__(self, other):
        return (type(self)==type(other) and self.props() == other.props())
    def __repr__(self):
        if self.lower:
            lu = 'lower'
        else:
            lu = 'upper'
        if self.destructive:
            destr = 'destructive'
        else:
            destr = 'non-destructive'
        return 'Cholesky{%s,%s}'% (lu,destr)
    def make_node(self, x):
        x = as_tensor_variable(x)
        return Apply(self, [x], [x.type()])
    def perform(self, node, (x,), (z,)):
        z[0] = scipy.linalg.cholesky(x, lower=self.lower).astype(x.dtype)
    #def grad(self, (x, y), (gz,)):
        #return dot(gz, y), dot(x, gz) #no transposing necessary
cholesky = Cholesky()

class MatrixInverse(Op):
    """Compute a matrix inverse"""
    def __init__(self):
        pass
    def props(self):
        return ()
    def __hash__(self):
        return hash((type(self), self.props()))
    def __eq__(self, other):
        return (type(self)==type(other) and self.props() == other.props())
    def make_node(self, x):
        x = as_tensor_variable(x)
        return Apply(self, [x], [x.type()])
    def perform(self, node, (x,), (z, )):
        try:
            z[0] = numpy.linalg.inv(x).astype(x.dtype)
        except Exception:
            print 'Failed to invert', node.inputs[0]
            raise
    def grad(self, inputs, g_outputs):
        x, = inputs
        xi = self(x)
        gz, = g_outputs
        #TT.dot(gz.T,xi)
        return [-matrix_dot(xi,gz.T,xi).T]
    def __str__(self):
        return "MatrixInverse"
matrix_inverse = MatrixInverse()

class Solve(Op):
    """Solve a system of linear equations"""
    def __init__(self, A_structure='general', lower=False, overwrite_A=False, overwrite_b=False):
        if A_structure not in MATRIX_STRUCTURES:
            raise ValueError('Invalid matrix structure argument', A_structure)
        self.A_structure = A_structure
        self.lower=lower
        self.overwrite_A=overwrite_A
        self.overwrite_b=overwrite_b
    def props(self):
        return (self.A_structure,
                self.lower,
                self.overwrite_A,
                self.overwrite_b)
    def __hash__(self):
        return hash((type(self),self.props()))
    def __eq__(self, other):
        return type(self) == type(other) and self.props() == other.props()
    def __repr__(self):
        return 'Solve{%s}'%str(self.props())
    def make_node(self, A, b):
        A = as_tensor_variable(A)
        b = as_tensor_variable(b)
        return Apply(self, [A,b], [b.type()])
    def perform(self, node, inputs, output_storage):
        A, b = inputs
        #TODO: use the A_structure to go faster
        output_storage[0][0] = scipy.linalg.solve(A,b)
solve = Solve() # general solve

#TODO : SolveTriangular

#TODO: Optimizations to replace multiplication by matrix inverse with solve() Op (still unwritten)

class ExtractDiag(Op):
    def __init__(self, view=False):
        self.view = view
        if self.view:
            self.view_map = {0:[0]}
            self.perform = self.perform_view
        else:
            self.perform = self.perform_noview
    def __eq__(self, other):
        return type(self) == type(other) and self.view == other.view
    def __hash__(self):
        return hash(type(self))^hash(self.view)
    def make_node(self, _x):
        x = as_tensor_variable(_x)
        if x.type.ndim != 2:
            raise TypeError('ExtractDiag only works on matrices', _x)
        return Apply(self, [x], [tensor.vector(dtype=x.type.dtype)])
    def perform_noview(self, node, (x,), (z,)):
        #for some reason numpy.diag(x) is really slow
        N,M = x.shape
        assert N==M
        rval = x[0]
        rval.strides = (x.strides[0]+x.strides[1],)
        z[0] = rval.copy()
    def perform_view(self, node, (x,), (z,)):
        N,M = x.shape
        a,b = x.strides
        assert N==M
        rval = x[0]
        rval.strides = a+b,
        z[0] = rval
    def __str__(self):
        return 'ExtractDiag{view=%s}'%self.view
    def grad(self, inputs, g_outputs):
        return [alloc_diag(g_outputs[0])]
extract_diag = ExtractDiag()

#TODO: optimization to insert ExtractDiag with view=True

class AllocDiag(Op):
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
alloc_diag = AllocDiag()

def diag(x):
    """Numpy-compatibility method

    For vector `x`, return a zero matrix except for `x` as diagonal.
    """
    xx = as_tensor_variable(x)
    if xx.type.ndim == 1:
        return alloc_diag(xx)
    elif xx.type.ndim ==2:
        return extract_diag(xx)
    else:
        raise TypeError('diag requires vector or matrix argument', x)

class Det(Op):
    """matrix determinant
    TODO: move this op to another file that request scipy.
    """
    def make_node(self, x):
        x = as_tensor_variable(x)
        o = theano.tensor.scalar(dtype=x.dtype)
        return Apply(self, [x], [o])
    def perform(self, node, (x,), (z, )):
        try:
            z[0] = numpy.asarray(scipy.linalg.det(x), dtype=x.dtype)
        except Exception:
            print 'Failed to compute determinant', x
            raise
    def grad(self, inputs, g_outputs):
        gz, = g_outputs
        x, = inputs
        return [gz * self(x) * matrix_inverse(x).T]
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
    """
    XX = X
    for i in xrange(log2_exponent):
        XX = tensor.dot(XX, XX)
    return tensor.pow(
            trace(XX),
            2**(-log2_exponent))

class A_Xinv_b(Op):
    """Product of form a inv(X) b"""
    def make_node(self, a, X, b):
        a = as_tensor_variable(a)
        b = as_tensor_variable(b)
        X = as_tensor_variable(X)
        o = theano.tensor.matrix(dtype=x.dtype)
        return Apply(self, [a,X,b], [o])
    def perform(self, ndoe, inputs, outstor):
        a,X,b = inputs
        if 1:
            L_factor = scipy.linalg.cho_factor(X)
            xb = scipy.linalg.cho_solve(L_factor, b)
            xa = scipy.linalg.cho_solve(L_factor, a.T)
            z = numpy.dot(xa.T, xb)
        else:
            raise NotImplementedError(self.X_structure)
        outstor[0][0]=z
    def grad(self, inputs, g_outputs):
        gz, = g_outputs
        a,X,b = inputs
        iX = matrix_inverse(X)
        ga = matrix_dot(gz, b.T, iX.T)
        gX = -matrix_dot(iX.T, a, gz, b.T, iX.T)
        gb = matrix_dot(ix.T, a.T, gz)
        return [ga, gX, gb]

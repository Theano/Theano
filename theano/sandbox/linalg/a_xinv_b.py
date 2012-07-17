import time
import math
import numpy
from theano.gof import Op, Apply
from theano import tensor, function, printing
from theano.tests import unittest_tools as utt
from theano.sandbox.linalg import matrix_inverse, kron

try:
    import scipy.linalg
    imported_scipy = True
except ImportError:
    imported_scipy = False


class Solve(Op):
    """
    Solves the matrix equation a x = b for x.

    Parameters:

    a: array, shape (M, M)
    b: array, shape (M,) or (M, N)
    sym_pos: (boolean) Assume a is symmetric and positive definite.
    lower: (boolean) Use only data contained in the lower triangle of a,
        if sym_pos is true. Default is to use upper triangle.
    overwrite_a: (boolean) Allow overwriting data in a (may enhance
    performance).
    overwrite_b: (boolean) Allow overwriting data in b (may enhance
    performance).

    Returns :

    x: array, shape (M,) or (M, N) depending on b
    """

    def __init__(self, sym_pos=False, lower=False, overwrite_a=False,
                 overwrite_b=False):
        self.sym_pos = sym_pos
        self.lower = lower
        self.overwrite_a = overwrite_a
        self.overwrite_b = overwrite_b

    def __eq__(self, other):
        return (type(self) == type(other) and self.sym_pos == other.sym_pos and
                self.lower == other.lower and
                self.overwrite_a == other.overwrite_a and
                self.overwrite_b == other.overwite_b)

    def __hash__(self):
        return (hash(type(self)) ^ hash(self.sym_pos) ^ hash(self.lower) ^
                hash(self.overwrite_a) ^ hash(self.overwrite_b))

    def props(self):
        return (self.sym_pos, self.lower, self.overwrite_a, self.overwrite_b)

    def __str__(self):
        return "%s{%s, %s, %s, %s}" % (self.__class__.__name__,
                "sym_pos=".join(str(self.sym_pos)),
                "lower=".join(str(self.lower)),
                "overwrite_a".join(str(self.overwrite_a)),
                "overwrite_b=".join(str(self.overwrite_b)))

    def __repr__(self):
        return 'Solve{%s}' % str(self.props())

    def make_node(self, a, b):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the Solve op")
        a = tensor.as_tensor_variable(a)
        b = tensor.as_tensor_variable(b)
        if a.ndim != 2 or  b.ndim > 2 or b.ndim == 0:
            raise TypeError('%s: inputs have improper dimensions:\n'
                    '\'a\' must have two,'
                    ' \'b\' must have either one or two' %
                            self.__class__.__name__)

        out_type = tensor.TensorType(dtype=(a * b).dtype,
                                     broadcastable=b.type.broadcastable)()
        return Apply(self, [a, b], [out_type])

    def infer_shape(self, node, in_shapes):
        return [in_shapes[1]]

    def perform(self, node, inputs, output_storage):
        a, b = inputs

        if a.shape[0] != a.shape[1] or a.shape[1] != b.shape[0]:
            raise TypeError('%s: inputs have improper lengths' %
                            self.__class__.__name__)
        try:
            output_storage[0][0] = scipy.linalg.solve(a, b, self.sym_pos,
                        self.lower, self.overwrite_a, self.overwrite_b)
        except:
            pass
            #raise  Exception('%s: array \'a\' is singular'
            #              % self.__class__.__name__)

    def grad(self, inputs, cost_grad):
        """
        See The Matrix Reference Manual,
        Copyright 1998-2011 Mike Brookes, Imperial College, London, UK

        Note: In contrast with the usual mathematical presentation, in order
        to apply theano's 'reshape' function wich implements row-order
        (i.e. C order), the differential expressions below have been derived
        around the row-vectorizations of inputs 'a' and 'b'.
        """

        a, b = inputs
        ingrad = cost_grad
        ingrad = tensor.as_tensor_variable(ingrad)
        inv_a = matrix_inverse(a)

        if b.ndim == 1:
            prod_a_b = tensor.tensordot(-b.T, inv_a.T, axes=1)
            prod_a_b = tensor.shape_padleft(prod_a_b)
            jac_veca = kron(inv_a, prod_a_b)
            jac_b = inv_a
            outgrad_veca = tensor.tensordot(ingrad, jac_veca, axes=1)
            outgrad_a = tensor.reshape(outgrad_veca,
                        (inputs[0].shape[0], inputs[0].shape[0]))
            outgrad_b = tensor.tensordot(ingrad, jac_b, axes=1).flatten(ndim=1)

        else:
            ingrad_vec = ingrad.flatten(ndim=1)
            prod_a_b = tensor.tensordot(-b.T, inv_a.T, axes=1)
            jac_veca = kron(inv_a, prod_a_b)
            I_N = tensor.eye(tensor.shape(inputs[1])[1],
                             tensor.shape(inputs[1])[1])
            jac_vecb = kron(inv_a, I_N)
            outgrad_veca = tensor.tensordot(ingrad_vec, jac_veca, axes=1)
            outgrad_a = tensor.reshape(outgrad_veca,
                        (inputs[0].shape[0], inputs[0].shape[0]))
            outgrad_vecb = tensor.tensordot(ingrad_vec, jac_vecb, axes=1)
            outgrad_b = tensor.reshape(outgrad_vecb,
                        (inputs[1].shape[0], inputs[1].shape[1]))

        return [outgrad_a, outgrad_b]


def solve(a, b, sym_pos=False, lower=False, overwrite_a=False,
                 overwrite_b=False):
    localop = Solve(sym_pos, lower, overwrite_a, overwrite_b)
    return localop(a, b)


#TODO: Optimizations to replace multiplication by matrix inverse
#      with Ops solve() or solve_triangular()


class TestSolve(utt.InferShapeTester):

    rng = numpy.random.RandomState(43)

    def setUp(self):

        super(TestSolve, self).setUp()
        self.op_class = Solve
        self.op = solve

    def test_perform(self):

        x = tensor.dmatrix()
        y = tensor.dmatrix()
        f = function([x, y], self.op(x, y))
        a = numpy.random.rand(4, 4)
        for shp1 in [(4, 5), (4, 1)]:
            b = numpy.random.rand(*shp1)
            out = f(a, b)
            assert numpy.allclose(out, scipy.linalg.solve(a, b))

        a = numpy.random.rand(1, 1)
        for shp1 in [(1, 5), (1, 1)]:
            b = numpy.random.rand(*shp1)
            out = f(a, b)
            assert numpy.allclose(out, scipy.linalg.solve(a, b))

        y = tensor.dvector()
        f = function([x, y], self.op(x, y))
        a = numpy.random.rand(4, 4)
        b = numpy.random.rand(4)
        out = f(a, b)
        assert numpy.allclose(out, scipy.linalg.solve(a, b))

        x = tensor.dmatrix()
        y = tensor.dmatrix()
        f = function([x, y], self.op(x, y, True, True))
        a = numpy.random.rand(4, 4)
        a = numpy.dot(a, numpy.transpose(a))
        for shp1 in [(4, 5), (4, 1)]:
            b = numpy.random.rand(*shp1)
            out = f(a, b)
            assert numpy.allclose(out, scipy.linalg.solve(a, b, True, True))

        a = numpy.random.rand(1, 1)
        a = numpy.dot(a, numpy.transpose(a))
        for shp1 in [(1, 5), (1, 1)]:
            b = numpy.random.rand(*shp1)
            out = f(a, b)
            assert numpy.allclose(out, scipy.linalg.solve(a, b, True, True))

        y = tensor.dvector()
        f = function([x, y], self.op(x, y))
        a = numpy.random.rand(4, 4)
        a = numpy.dot(a, numpy.transpose(a))
        b = numpy.random.rand(4)
        out = f(a, b)
        assert numpy.allclose(out, scipy.linalg.solve(a, b, True, True))

    def test_gradient(self):

        utt.verify_grad(self.op, [numpy.random.rand(5, 5),
                                numpy.random.rand(5, 1)],
                        n_tests=1, rng=TestSolve.rng)

        utt.verify_grad(self.op, [numpy.random.rand(4, 4),
                                       numpy.random.rand(4, 3)],
                      n_tests=1, rng=TestSolve.rng)

        utt.verify_grad(self.op, [numpy.random.rand(4, 4),
                                         numpy.random.rand(4)],
                      n_tests=1, rng=TestSolve.rng)

    def test_infer_shape(self):

        x = tensor.dmatrix()
        y = tensor.dmatrix()

        self._compile_and_check([x, y], [self.op(x, y)],
                                [numpy.random.rand(5, 5),
                                 numpy.random.rand(5, 2)],
                                self.op_class)

        self._compile_and_check([x, y], [self.op(x, y)],
                                [numpy.random.rand(4, 4),
                                 numpy.random.rand(4, 1)],
                                self.op_class)
        y = tensor.dvector()
        self._compile_and_check([x, y], [self.op(x, y)],
                                [numpy.random.rand(4, 4),
                                 numpy.random.rand(4)],
                                self.op_class)


# function using solve
def a_xinv_b_solve(a, x, b, x_sym_pos=False, x_lower=False, overwrite_x=False,
                 overwrite_b=False):
    xinv_b = solve(x, b, sym_pos=x_sym_pos, lower=x_lower,
                   overwrite_a=overwrite_x, overwrite_b=overwrite_b)
    a_xinv_b = tensor.dot(a, xinv_b)
    return a_xinv_b


# function using 'matrix_inverse' as a benchmark
def a_xinv_b_inv(a, x, b):
    xinv = matrix_inverse(x)
    xinv_b = tensor.dot(xinv, b)
    a_xinv_b = tensor.dot(a, xinv_b)
    return a_xinv_b


# Existing Op in sandbox/linalg using Cholesky factors (valid only for x
# symmetric positive definite)
class A_Xinv_b(Op):
    """Product of form a inv(X) b"""
    def make_node(self, a, X, b):
        assert imported_scipy, (
            "Scipy not available. Scipy is needed for the A_Xinv_b op")
        a = tensor.as_tensor_variable(a)
        b = tensor.as_tensor_variable(b)
        X = tensor.as_tensor_variable(X)
        o = tensor.matrix(dtype=x.dtype)
        return Apply(self, [a, X, b], [o])

    def perform(self, ndoe, inputs, outstor):
        a, X, b = inputs

        def recur_solve(left, right):
            sol = numpy.zeros_like(right)
            for i in xrange(left.shape[1]):
                sol[i, ::] = ((right[i, ::] - numpy.dot(left[i, ::], sol))
                              / left[i, i])
            return sol

        if 1:
            # (16 jul 2012) 'scipy.linalg.cho_solve' appears to be defective
            # L_factor = scipy.linalg.cho_factor(X, lower=True)
            # xb = scipy.linalg.cho_solve(L_factor, b)
            # xa = scipy.linalg.cho_solve(L_factor, a.T)

            # we replace it with generic 'solve' which should make it
            # suboptimal in comparison with a single application of solve as
            # in 'a_xinv_b_solve' above
            L_factor = scipy.linalg.cholesky(x_val, lower=True,
                                            overwrite_a=False)
            xb = scipy.linalg.solve(L_factor, b)
            xa = scipy.linalg.solve(L_factor, a.T)

            # we can also attempt to replace it with 'recur_solve' above
            # (this performs worse in trials)
            #L_factor = scipy.linalg.cholesky(x_val, lower=True,
            #                                 overwrite_a=False)
            #xb = recur_solve(L_factor, b)
            #xa = recur_solve(L_factor, a.T)

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


if __name__ == "__main__":

    """
    t = TestSolve('setUp')
    t.setUp()
    t.test_perform()
    t.test_gradient()
    t.test_infer_shape()
    """

    # Speed and accuracy trials
    trials = 10000
    order = 500
    x = tensor.dmatrix()
    a = tensor.dmatrix()
    b = tensor.dmatrix()
    rnd_state = 43
    rnd_seed = 2479
    print '**********************', 'trials:', trials, \
           ' order:', order, ' rnd_state:', rnd_state, ' rnd_seed:', rnd_seed

    # 1.1 with solve for generic x matrix

    f_solve = function([a, x, b], a_xinv_b_solve(a, x, b))
    # printing.debugprint(f_solve)
    rng = numpy.random.RandomState(rnd_state)
    numpy.random.seed(rnd_seed)
    diag = numpy.zeros(shape=(order, order))
    for i in xrange(order):
        diag[i, i] = 10 ** ((float(i) / order) * (-10))
    diag = numpy.matrix(diag, copy=False)
    a_val = numpy.identity(order)
    sum_relerror = 0.
    sum_relresid = 0.
    sum_cond = 0.
    start = time.clock()
    for i in xrange(trials):
        #a_val = numpy.random.rand(order, order)
        z_val = numpy.matrix(numpy.random.rand(order, 1), copy=False)
        temp = 10 * numpy.random.rand(order, order)
        range_temp = numpy.matrix(numpy.linalg.svd(temp)[0], copy=False)
        x_val = range_temp * diag * range_temp.T
        b_val = x_val * z_val
        sol = f_solve(a_val, x_val, b_val)
        #sol = z_val
        sum_relerror += (numpy.linalg.norm(sol - z_val, 'fro')
                         / numpy.linalg.norm(z_val, 'fro'))
        sum_relresid += (numpy.linalg.norm(x_val * sol - b_val, 'fro')
                         / numpy.linalg.norm(b_val, 'fro'))
        sum_cond += numpy.linalg.cond(x_val)
    avg_time = (time.clock() - start) / trials
    avg_relerror = sum_relerror / trials
    avg_relresid = sum_relresid / trials
    avg_cond = sum_cond / trials

    print 'solve generic:'
    print 'avg_time:', avg_time, 'avg_relerror:', avg_relerror, \
        'avg_relresid:', avg_relresid, 'avg_cond:', avg_cond

    # 1.2 with solve for symmetric positive definite x matrix

    f_solve = function([a, x, b], a_xinv_b_solve(a, x, b, True, True))
    # printing.debugprint(f_solve)
    rng = numpy.random.RandomState(rnd_state)
    numpy.random.seed(rnd_seed)
    diag = numpy.zeros(shape=(order, order))
    for i in xrange(order):
        diag[i, i] = 10 ** ((float(i) / order) * (-10))
    diag = numpy.matrix(diag, copy=False)
    a_val = numpy.identity(order)
    sum_relerror = 0.
    sum_relresid = 0.
    sum_cond = 0.
    start = time.clock()
    for i in xrange(trials):
        #a_val = numpy.random.rand(order, order)
        z_val = numpy.matrix(numpy.random.rand(order, 1), copy=False)
        temp = 10 * numpy.random.rand(order, order)
        range_temp = numpy.matrix(numpy.linalg.svd(temp)[0], copy=False)
        x_val = range_temp * diag * range_temp.T
        b_val = x_val * z_val
        sol = f_solve(a_val, x_val, b_val)
        sum_relerror += (numpy.linalg.norm(sol - z_val, 'fro')
                         / numpy.linalg.norm(z_val, 'fro'))
        sum_relresid += (numpy.linalg.norm(x_val * sol - b_val, 'fro')
                         / numpy.linalg.norm(b_val, 'fro'))
        sum_cond += numpy.linalg.cond(x_val)
    avg_time = (time.clock() - start) / trials
    avg_relerror = sum_relerror / trials
    avg_relresid = sum_relresid / trials
    avg_cond = sum_cond / trials

    print 'solve symmetric pd:'
    print 'avg_time:', avg_time, 'avg_relerror:', avg_relerror, \
        'avg_relresid:', avg_relresid, 'avg_cond:', avg_cond

    # 1.3 with matrix_inverse for generic x matrix
    # (will not differ from 1.1 since optimization will substitute solve
    # for inversion)

    f_inv = function([a, x, b], a_xinv_b_inv(a, x, b))
    # printing.debugprint(f_inv)
    rng = numpy.random.RandomState(rnd_state)
    numpy.random.seed(rnd_seed)
    diag = numpy.zeros(shape=(order, order))
    for i in xrange(order):
        diag[i, i] = 10 ** ((float(i) / order) * (-10))
    diag = numpy.matrix(diag, copy=False)
    a_val = numpy.identity(order)
    sum_relerror = 0.
    sum_relresid = 0.
    sum_cond = 0.
    start = time.clock()
    for i in xrange(trials):
        #a_val = numpy.random.rand(order, order)
        z_val = numpy.matrix(numpy.random.rand(order, 1), copy=False)
        temp = 10 * numpy.random.rand(order, order)
        range_temp = numpy.matrix(numpy.linalg.svd(temp)[0], copy=False)
        x_val = range_temp * diag * range_temp.T
        b_val = x_val * z_val
        sol = f_inv(a_val, x_val, b_val)
        sum_relerror += (numpy.linalg.norm(sol - z_val, 'fro')
                         / numpy.linalg.norm(z_val, 'fro'))
        sum_relresid += (numpy.linalg.norm(x_val * sol - b_val, 'fro')
                         / numpy.linalg.norm(b_val, 'fro'))
        sum_cond += numpy.linalg.cond(x_val)
    avg_time = (time.clock() - start) / trials
    avg_relerror = sum_relerror / trials
    avg_relresid = sum_relresid / trials
    avg_cond = sum_cond / trials

    print 'inverse generic:'
    print 'avg_time:', avg_time, 'avg_relerror:', avg_relerror, \
        'avg_relresid:', avg_relresid, 'avg_cond:', avg_cond

    # 1.4 with Cholesky factorization for symmetric positive definite x matrix

    f_chol = function([a, x, b], A_Xinv_b()(a, x, b))
    # printing.debugprint(f_chol)
    rng = numpy.random.RandomState(rnd_state)
    numpy.random.seed(rnd_seed)
    diag = numpy.zeros(shape=(order, order))
    for i in xrange(order):
        diag[i, i] = 10 ** ((float(i) / order) * (-10))
    diag = numpy.matrix(diag, copy=False)
    a_val = numpy.identity(order)
    sum_relerror = 0.
    sum_relresid = 0.
    sum_cond = 0.
    start = time.clock()
    for i in xrange(trials):
        #a_val = numpy.random.rand(order, order)
        z_val = numpy.matrix(numpy.random.rand(order, 1), copy=False)
        temp = 10 * numpy.random.rand(order, order)
        range_temp = numpy.matrix(numpy.linalg.svd(temp)[0], copy=False)
        x_val = range_temp * diag * range_temp.T
        b_val = x_val * z_val
        sol = f_chol(a_val, x_val, b_val)
        sum_relerror += (numpy.linalg.norm(sol - z_val, 'fro')
                         / numpy.linalg.norm(z_val, 'fro'))
        sum_relresid += (numpy.linalg.norm(x_val * sol - b_val, 'fro')
                         / numpy.linalg.norm(b_val, 'fro'))
        sum_cond += numpy.linalg.cond(x_val)
    avg_time = (time.clock() - start) / trials
    avg_relerror = sum_relerror / trials
    avg_relresid = sum_relresid / trials
    avg_cond = sum_cond / trials

    print 'Cholesky symmetric pd:'
    print 'avg_time:', avg_time, 'avg_relerror:', avg_relerror, \
        'avg_relresid:', avg_relresid, 'avg_cond:', avg_cond


"""
SIMULATION TRIALS (on BART3, July 2012):
(cpu time in seconds)

********************** trials: 1000  order: 500  rnd_state: 43  rnd_seed: 6318
avg overhead time: 0.34572 
solve generic:
avg_time: 0.33519 avg_relerror: 4.63702286138e-07 avg_relresid: 5.24910351103e-16 avg_cond: 9549925881.16
solve symmetric pd:
avg_time: 0.35186 avg_relerror: 4.67273006345e-07 avg_relresid: 3.79045700032e-16 avg_cond: 9549925881.16
inverse generic:
avg_time: 0.34467 avg_relerror: 4.63702286138e-07 avg_relresid: 5.24910351103e-16 avg_cond: 9549925881.16
Cholesky symmetric pd: (with solve in perform)
avg_time: 0.37647 avg_relerror: 4.6826195279e-07 avg_relresid: 1.22016314633e-15 avg_cond: 9549925881.16
Cholesky symmetric pd: (with recur_solve in perform)
avg_time: 0.431 avg_relerror: 5.25561664987e-07 avg_relresid: 1.22966872309e-15 avg_cond: 9549925881.16


********************** trials: 10000  order: 500  rnd_state: 43  rnd_seed: 2479
avg overhead time: 0.327635 
solve generic:
avg_time: 0.34137 avg_relerror: 4.62322158822e-07 avg_relresid: 5.27845933921e-16 avg_cond: 9549925875.41
solve symmetric pd:
avg_time: 0.344791 avg_relerror: 4.63647228054e-07 avg_relresid: 3.79386681208e-16 avg_cond: 9549925875.41
inverse generic:
avg_time: 0.357983 avg_relerror: 4.62322158822e-07 avg_relresid: 5.27845933921e-16 avg_cond: 9549925875.41
Cholesky symmetric pd:
avg_time: 0.406816 avg_relerror: 4.64379928085e-07 avg_relresid: 1.2315400664e-15 avg_cond: 9549925875.41


"""
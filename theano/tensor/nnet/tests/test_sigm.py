import unittest
from itertools import imap

import numpy

import theano.tensor.inplace
from theano import tensor as T
from theano import config
from theano.tests import unittest_tools as utt
from theano.tensor.nnet import sigmoid, sigmoid_inplace, softplus, tensor
from theano.tensor.nnet.sigm import (
        compute_mul, is_1pexp, parse_mul_tree, perform_sigm_times_exp,
        register_local_1msigmoid, simplify_mul)


class T_sigmoid(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def test_elemwise(self):
        utt.verify_grad(sigmoid, [numpy.random.rand(3, 4)])


class T_softplus(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def test_elemwise(self):
        utt.verify_grad(softplus, [numpy.random.rand(3, 4)])


class T_sigmoid_opts(unittest.TestCase):

    def get_mode(self, excluding=[]):
        """
        Return appropriate mode for the tests.

        :param excluding: List of optimizations to exclude.

        :return: The current default mode unless the `config.mode` option is
        set to 'FAST_COMPILE' (in which case it is replaced by the 'FAST_RUN'
        mode), without the optimizations specified in `excluding`.
        """
        m = theano.config.mode
        if m == 'FAST_COMPILE':
            mode = theano.compile.mode.get_mode('FAST_RUN')
        else:
            mode = theano.compile.mode.get_default_mode()
        if excluding:
            return mode.excluding(*excluding)
        else:
            return mode

    def test_exp_over_1_plus_exp(self):
        m = self.get_mode(excluding=['local_elemwise_fusion'])

        x = T.vector()
        data = numpy.random.rand(54).astype(config.floatX)

        backup = config.warn.identify_1pexp_bug
        config.warn.identify_1pexp_bug = False
        try:
            # tests exp_over_1_plus_exp
            f = theano.function([x], T.exp(x) / (1 + T.exp(x)), mode=m)
            assert [node.op for node in f.maker.env.toposort()] == [sigmoid]
            f(data)
            f = theano.function([x], T.exp(x) / (2 + T.exp(x)), mode=m)
            assert [node.op for node in f.maker.env.toposort()] != [sigmoid]
            f(data)
            f = theano.function([x], T.exp(x) / (1 - T.exp(x)), mode=m)
            assert [node.op for node in f.maker.env.toposort()] != [sigmoid]
            f(data)
            f = theano.function([x], T.exp(x + 1) / (1 + T.exp(x)), mode=m)
            assert [node.op for node in f.maker.env.toposort()] != [sigmoid]
            f(data)

            # tests inv_1_plus_exp
            f = theano.function([x], T.fill(x, 1.0) / (1 + T.exp(-x)), mode=m)
            assert [node.op for node in f.maker.env.toposort()] == [sigmoid]
            f(data)
            f = theano.function([x], T.fill(x, 1.0) / (2 + T.exp(-x)), mode=m)
            assert [node.op for node in f.maker.env.toposort()] != [sigmoid]
            f(data)
            f = theano.function([x], T.fill(x, 1.0) / (1 - T.exp(-x)), mode=m)
            assert [node.op for node in f.maker.env.toposort()] != [sigmoid]
            f(data)
            f = theano.function([x], T.fill(x, 1.1) / (1 + T.exp(-x)), mode=m)
            assert [node.op for node in f.maker.env.toposort()] != [sigmoid]
            f(data)

            # tests inv_1_plus_exp with neg
            f = theano.function([x], T.fill(x, -1.0) / (1 + T.exp(-x)), mode=m)
            assert [node.op for node in f.maker.env.toposort()] == [sigmoid,
                    theano.tensor.inplace.neg_inplace]
            f(data)
            f = theano.function([x], T.fill(x, -1.0) / (1 - T.exp(-x)), mode=m)
            assert [node.op for node in f.maker.env.toposort()] != [sigmoid,
                    theano.tensor.inplace.neg_inplace]
            f(data)
            f = theano.function([x], T.fill(x, -1.0) / (2 + T.exp(-x)), mode=m)
            assert [node.op for node in f.maker.env.toposort()] != [sigmoid,
                    theano.tensor.inplace.neg_inplace]
            f(data)
            f = theano.function([x], T.fill(x, -1.1) / (1 + T.exp(-x)), mode=m)
            assert [node.op for node in f.maker.env.toposort()] != [sigmoid,
                    theano.tensor.inplace.neg_inplace]
            f(data)

            # tests double inv_1_plus_exp with neg
            # (-1)(exp(x)) / (1+exp(x))(1+exp(-x))
            # = (-1)/(1+exp(-x)) * exp(x)/(1+exp(x))
            # = - (sigm(x) * sigm(x))
            f = theano.function([x], (T.fill(x, -1.0) * T.exp(x)) /
                                ((1 + T.exp(x)) * (1 + T.exp(-x))), mode=m)
            assert [node.op for node in f.maker.env.toposort()] == [sigmoid,
                    T.mul, theano.tensor.inplace.neg_inplace]
            f(data)
            f = theano.function([x], (T.fill(x, -1.1) * T.exp(x)) /
                                ((1 + T.exp(x)) * (1 + T.exp(-x))), mode=m)
            assert [node.op for node in f.maker.env.toposort()] != [sigmoid,
                    T.mul, theano.tensor.inplace.neg_inplace]
            f(data)
            f = theano.function([x], (T.fill(x, -1.0) * T.exp(x)) /
                                ((2 + T.exp(x)) * (1 + T.exp(-x))), mode=m)
            assert [node.op for node in f.maker.env.toposort()] != [sigmoid,
                    T.mul, theano.tensor.inplace.neg_inplace]
            f(data)
            f = theano.function([x], (T.fill(x, -1.0) * T.exp(x)) /
                                ((1 + T.exp(x)) * (2 + T.exp(-x))), mode=m)
            assert [node.op for node in f.maker.env.toposort()] != [sigmoid,
                    T.mul, theano.tensor.inplace.neg_inplace]
            f(data)
            f = theano.function([x], (T.fill(x, -1.0) * T.exp(x)) /
                                ((1 + T.exp(x)) * (1 + T.exp(x))), mode=m)
            assert [node.op for node in f.maker.env.toposort()] != [sigmoid,
                    T.mul, theano.tensor.inplace.neg_inplace]
            f(data)
            f = theano.function([x], (T.fill(x, -1.0) * T.exp(x)) /
                                ((1 + T.exp(x)) * (2 + T.exp(-x))), mode=m)
            assert [node.op for node in f.maker.env.toposort()] != [sigmoid,
                    T.mul, theano.tensor.inplace.neg_inplace]
            f(data)

        finally:
            # Restore config option.
            config.warn.identify_1pexp_bug = backup

    def test_1msigmoid(self):
        if not register_local_1msigmoid:
            return

        m = self.get_mode()
        x = T.fmatrix()

        # tests exp_over_1_plus_exp
        f = theano.function([x], 1 - T.exp(x) / (1 + T.exp(x)), mode=m)
        assert [node.op for node in f.maker.env.toposort()] == [
            tensor.neg, sigmoid_inplace]

        # tests inv_1_plus_exp
        f = theano.function([x], 1 - T.fill(x, 1.0) / (1 + T.exp(-x)), mode=m)
        assert [node.op for node in f.maker.env.toposort()] == [tensor.neg,
                sigmoid_inplace]

    def test_local_sigm_times_exp(self):
        """
        Test the `local_sigm_times_exp` optimization.
        exp(x) * sigm(-x) -> sigm(x)
        exp(-x) * sigm(x) -> sigm(-x)
        """
        def match(func, ops):
            #print [node.op.scalar_op for node in func.maker.env.toposort()]
            assert [node.op for node in func.maker.env.toposort()] == ops
        m = self.get_mode(excluding=['local_elemwise_fusion', 'inplace'])
        x, y = tensor.vectors('x', 'y')

        f = theano.function([x], sigmoid(-x) * tensor.exp(x), mode=m)
        match(f, [sigmoid])

        f = theano.function([x], sigmoid(x) * tensor.exp(-x), mode=m)
        match(f, [tensor.neg, sigmoid])

        f = theano.function([x], -(-(-(sigmoid(x)))) * tensor.exp(-x), mode=m)
        match(f, [tensor.neg, sigmoid, tensor.neg])

        f = theano.function(
                [x, y],
                (sigmoid(x) * sigmoid(-y) * -tensor.exp(-x) *
                 tensor.exp(x * y) * tensor.exp(y)),
                mode=m)
        match(f, [sigmoid, tensor.mul, tensor.neg, tensor.exp, sigmoid,
                  tensor.mul, tensor.neg])

    def test_perform_sigm_times_exp(self):
        """
        Test the core function doing the `sigm_times_exp` optimization.

        It is easier to test different graph scenarios this way than by
        compiling a theano function.
        """
        x, y, z, t = tensor.vectors('x', 'y', 'z', 't')
        exp = tensor.exp

        def ok(expr1, expr2):
            trees = [parse_mul_tree(e) for e in (expr1, expr2)]
            perform_sigm_times_exp(trees[0])
            trees[0] = simplify_mul(trees[0])
            good = theano.gof.graph.is_same_graph(
                    compute_mul(trees[0]),
                    compute_mul(trees[1]))
            if not good:
                print trees[0]
                print trees[1]
                print '***'
                theano.printing.debugprint(compute_mul(trees[0]))
                print '***'
                theano.printing.debugprint(compute_mul(trees[1]))
            assert good
        ok(sigmoid(x) * exp(-x), sigmoid(-x))
        ok(-x * sigmoid(x) * (y * (-1 * z) * exp(-x)),
           -x * sigmoid(-x) * (y * (-1 * z)))
        ok(-sigmoid(-x) *
           (exp(y) * (-exp(-z) * 3 * -exp(x)) *
            (y * 2 * (-sigmoid(-y) * (z + t) * exp(z)) * sigmoid(z))) *
           -sigmoid(x),
           sigmoid(x) *
           (-sigmoid(y) * (-sigmoid(-z) * 3) * (y * 2 * ((z + t) * exp(z)))) *
           -sigmoid(x))
        ok(exp(-x) * -exp(-x) * (-sigmoid(x) * -sigmoid(x)),
           -sigmoid(-x) * sigmoid(-x))
        ok(-exp(x) * -sigmoid(-x) * -exp(-x),
           -sigmoid(-x))


class T_softplus_opts(unittest.TestCase):
    def setUp(self):
        if theano.config.mode == 'FAST_COMPILE':
            m = theano.compile.mode.get_mode('FAST_RUN').excluding(
                'local_elemwise_fusion')
        else:
            m = theano.compile.mode.get_default_mode().excluding(
                'local_elemwise_fusion')
        self.m = m
        utt.seed_rng()

    def test_logsigm_to_softplus(self):
        x = T.vector()

        out = T.log(sigmoid(x))
        f = theano.function([x], out, mode=self.m)
        topo = f.maker.env.toposort()
        assert len(topo) == 3
        assert isinstance(topo[0].op.scalar_op, theano.scalar.Neg)
        assert isinstance(topo[1].op.scalar_op,
                          theano.tensor.nnet.sigm.ScalarSoftplus)
        assert isinstance(topo[2].op.scalar_op, theano.scalar.Neg)
        f(numpy.random.rand(54).astype(config.floatX))

    def test_log1msigm_to_softplus(self):
        x = T.vector()

        out = T.log(1 - sigmoid(x))
        f = theano.function([x], out, mode=self.m)
        topo = f.maker.env.toposort()
        assert len(topo) == 2
        assert isinstance(topo[0].op.scalar_op,
                          theano.tensor.nnet.sigm.ScalarSoftplus)
        assert isinstance(topo[1].op.scalar_op, theano.scalar.Neg)
        f(numpy.random.rand(54).astype(config.floatX))

    def test_log1pexp_to_softplus(self):
        m = theano.config.mode
        if m == 'FAST_COMPILE':
            m = 'FAST_RUN'

        x = T.vector()

        out = T.log(1 + T.exp(x))
        f = theano.function([x], out, mode=self.m)
        topo = f.maker.env.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op.scalar_op,
                          theano.tensor.nnet.sigm.ScalarSoftplus)
        f(numpy.random.rand(54).astype(config.floatX))


class T_sigmoid_utils(unittest.TestCase):

    """
    Test utility functions found in 'sigm.py'.
    """

    def test_compute_mul(self):
        x, y, z = tensor.vectors('x', 'y', 'z')
        tree = (x * y) * -z
        mul_tree = parse_mul_tree(tree)
        assert parse_mul_tree(compute_mul(mul_tree)) == mul_tree
        assert theano.gof.graph.is_same_graph(
                                    compute_mul(parse_mul_tree(tree)), tree)

    def test_parse_mul_tree(self):
        x, y, z = tensor.vectors('x', 'y', 'z')
        assert parse_mul_tree(x * y) == [False, [[False, x], [False, y]]]
        assert parse_mul_tree(-(x * y)) == [True, [[False, x], [False, y]]]
        assert parse_mul_tree(-x * y) == [False, [[True, x], [False, y]]]
        assert parse_mul_tree(-x) == [True, x]
        assert parse_mul_tree((x * y) * -z) == [
                        False, [[False, [[False, x], [False, y]]], [True, z]]]

    def test_is_1pexp(self):
        backup = config.warn.identify_1pexp_bug
        config.warn.identify_1pexp_bug = False
        try:
            x = tensor.vector('x')
            exp = tensor.exp
            assert is_1pexp(1 + exp(x)) == (False, x)
            assert is_1pexp(exp(x) + 1) == (False, x)
            for neg, exp_arg in imap(is_1pexp, [(1 + exp(-x)), (exp(-x) + 1)]):
                assert not neg and theano.gof.graph.is_same_graph(exp_arg, -x)
            assert is_1pexp(1 - exp(x)) is None
            assert is_1pexp(2 + exp(x)) is None
            assert is_1pexp(exp(x) + 2) is None
            assert is_1pexp(exp(x) - 1) is None
            assert is_1pexp(-1 + exp(x)) is None
            assert is_1pexp(1 + 2 * exp(x)) is None
        finally:
            config.warn.identify_1pexp_bug = backup

from __future__ import absolute_import, print_function, division
import unittest

import numpy as np

from theano.compat import imap
import theano.tensor.inplace
from theano.tensor import basic as tensor
from theano import tensor as T
from theano import config
from theano.gof.opt import check_stack_trace
from theano.tests import unittest_tools as utt
from theano.tensor.nnet import (sigmoid, sigmoid_inplace,
                                softplus, ultra_fast_sigmoid, hard_sigmoid)
from theano.tensor.nnet.sigm import (
    compute_mul, is_1pexp, parse_mul_tree, perform_sigm_times_exp,
    register_local_1msigmoid, simplify_mul,
)
from theano.tensor.tests.test_basic import (makeBroadcastTester, copymod,
                                            check_floatX, upcast_int8_nfunc,
                                            _good_broadcast_unary_normal_no_complex)


class T_sigmoid(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def test_elemwise(self):
        utt.verify_grad(sigmoid, [np.random.rand(3, 4)])

SigmoidTester = makeBroadcastTester(
    op=sigmoid,
    expected=upcast_int8_nfunc(lambda inputs: check_floatX(
        inputs, 1 / (1 + np.exp(-inputs)))),
    good=copymod(_good_broadcast_unary_normal_no_complex,
                 without=['uint16']),  # The reason that 'uint16' is excluted is that
                                       # theano works well but numpy overflows resulting
                                       # in an assertion error.
    # grad=_grad_broadcast_unary_normal,
    name='SigmoidTester',
)

UltraFastSigmoidTester = makeBroadcastTester(
    op=ultra_fast_sigmoid,
    expected=upcast_int8_nfunc(lambda inputs: check_floatX(
        inputs, 1 / (1 + np.exp(-inputs)))),
    good=copymod(_good_broadcast_unary_normal_no_complex,
                 without=['uint16']),  # numpy fucnting overflows with uint16.
    # grad=_grad_broadcast_unary_normal,
    name='UltraFastSigmoidTester',
    # This is an approx of the sigmoid. That is why we raise eps
    eps=5e-2)

HardSigmoidTester = makeBroadcastTester(
    op=hard_sigmoid,
    expected=upcast_int8_nfunc(lambda inputs: check_floatX(
        inputs, 1 / (1 + np.exp(-inputs)))),
    good=copymod(_good_broadcast_unary_normal_no_complex,
                 without=['uint16']),  # numpy fucnting overflows with uint16.
    # grad=_grad_broadcast_unary_normal,
    name='HardSigmoidTester',
    # This is an approx of the sigmoid. That is why we raise eps
    eps=1e-1)


SoftplusTester = makeBroadcastTester(
    op=softplus,
    expected=upcast_int8_nfunc(lambda inputs: check_floatX(
        inputs, np.log1p(np.exp(inputs)))),
    good=dict(copymod(_good_broadcast_unary_normal_no_complex,
                      without=['uint8', 'uint16']),  # numpy fucnting overflows with uint16.
              uint8=[np.arange(0, 89, dtype='uint8')],  # the range is different in new added uint8.
              int8=[np.arange(-127, 89, dtype='int8')]),
    # grad=_grad_broadcast_unary_normal,
    name='SoftplusTester',
)


class T_softplus(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def test_elemwise(self):
        utt.verify_grad(softplus, [np.random.rand(3, 4)])


class T_sigmoid_opts(unittest.TestCase):

    def get_mode(self, excluding=None):
        """
        Return appropriate mode for the tests.

        :param excluding: List of optimizations to exclude.

        :return: The current default mode unless the `config.mode` option is
        set to 'FAST_COMPILE' (in which case it is replaced by the 'FAST_RUN'
        mode), without the optimizations specified in `excluding`.
        """
        if excluding is None:
            excluding = []
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
        data = np.random.rand(54).astype(config.floatX)

        backup = config.warn.identify_1pexp_bug
        config.warn.identify_1pexp_bug = False
        try:
            # tests exp_over_1_plus_exp
            f = theano.function([x], T.exp(x) / (1 + T.exp(x)), mode=m)
            assert [node.op for node in f.maker.fgraph.toposort()] == [sigmoid]
            f(data)
            f = theano.function([x], T.exp(x) / (2 + T.exp(x)), mode=m)
            assert [node.op for node in f.maker.fgraph.toposort()] != [sigmoid]
            f(data)
            f = theano.function([x], T.exp(x) / (1 - T.exp(x)), mode=m)
            assert [node.op for node in f.maker.fgraph.toposort()] != [sigmoid]
            f(data)
            f = theano.function([x], T.exp(x + 1) / (1 + T.exp(x)), mode=m)
            assert [node.op for node in f.maker.fgraph.toposort()] != [sigmoid]
            f(data)

            # tests inv_1_plus_exp
            f = theano.function([x], T.fill(x, 1.0) / (1 + T.exp(-x)), mode=m)
            # todo: solve issue #4589 first
            # assert check_stack_trace(f, ops_to_check=sigmoid)
            assert [node.op for node in f.maker.fgraph.toposort()] == [sigmoid]
            f(data)
            f = theano.function([x], T.fill(x, 1.0) / (2 + T.exp(-x)), mode=m)
            assert [node.op for node in f.maker.fgraph.toposort()] != [sigmoid]
            f(data)
            f = theano.function([x], T.fill(x, 1.0) / (1 - T.exp(-x)), mode=m)
            assert [node.op for node in f.maker.fgraph.toposort()] != [sigmoid]
            f(data)
            f = theano.function([x], T.fill(x, 1.1) / (1 + T.exp(-x)), mode=m)
            assert [node.op for node in f.maker.fgraph.toposort()] != [sigmoid]
            f(data)

            # tests inv_1_plus_exp with neg
            f = theano.function([x], T.fill(x, -1.0) / (1 + T.exp(-x)), mode=m)
            # todo: solve issue #4589 first
            # assert check_stack_trace(
            #     f, ops_to_check=[sigmoid, theano.tensor.inplace.neg_inplace])
            assert ([node.op for node in f.maker.fgraph.toposort()] ==
                    [sigmoid, theano.tensor.inplace.neg_inplace])
            f(data)
            f = theano.function([x], T.fill(x, -1.0) / (1 - T.exp(-x)), mode=m)
            assert ([node.op for node in f.maker.fgraph.toposort()] != [sigmoid,
                    theano.tensor.inplace.neg_inplace])
            f(data)
            f = theano.function([x], T.fill(x, -1.0) / (2 + T.exp(-x)), mode=m)
            assert ([node.op for node in f.maker.fgraph.toposort()] != [sigmoid,
                    theano.tensor.inplace.neg_inplace])
            f(data)
            f = theano.function([x], T.fill(x, -1.1) / (1 + T.exp(-x)), mode=m)
            assert ([node.op for node in f.maker.fgraph.toposort()] != [sigmoid,
                    theano.tensor.inplace.neg_inplace])
            f(data)

            # tests double inv_1_plus_exp with neg
            # (-1)(exp(x)) / (1+exp(x))(1+exp(-x))
            # = (-1)/(1+exp(-x)) * exp(x)/(1+exp(x))
            # = - (sigm(x) * sigm(x))
            f = theano.function([x], (T.fill(x, -1.0) * T.exp(x)) /
                                ((1 + T.exp(x)) * (1 + T.exp(-x))), mode=m)
            # todo: solve issue #4589 first
            # assert check_stack_trace(f, ops_to_check=[sigmoid, T.mul])
            assert ([node.op for node in f.maker.fgraph.toposort()] == [sigmoid,
                    T.mul])
            f(data)
            f = theano.function([x], (T.fill(x, -1.1) * T.exp(x)) /
                                ((1 + T.exp(x)) * (1 + T.exp(-x))), mode=m)
            assert ([node.op for node in f.maker.fgraph.toposort()] != [sigmoid,
                    T.mul, theano.tensor.inplace.neg_inplace])
            f(data)
            f = theano.function([x], (T.fill(x, -1.0) * T.exp(x)) /
                                ((2 + T.exp(x)) * (1 + T.exp(-x))), mode=m)
            assert ([node.op for node in f.maker.fgraph.toposort()] != [sigmoid,
                    T.mul, theano.tensor.inplace.neg_inplace])
            f(data)
            f = theano.function([x], (T.fill(x, -1.0) * T.exp(x)) /
                                ((1 + T.exp(x)) * (2 + T.exp(-x))), mode=m)
            assert ([node.op for node in f.maker.fgraph.toposort()] != [sigmoid,
                    T.mul, theano.tensor.inplace.neg_inplace])
            f(data)
            f = theano.function([x], (T.fill(x, -1.0) * T.exp(x)) /
                                ((1 + T.exp(x)) * (1 + T.exp(x))), mode=m)
            assert ([node.op for node in f.maker.fgraph.toposort()] != [sigmoid,
                    T.mul, theano.tensor.inplace.neg_inplace])
            f(data)
            f = theano.function([x], (T.fill(x, -1.0) * T.exp(x)) /
                                ((1 + T.exp(x)) * (2 + T.exp(-x))), mode=m)
            assert ([node.op for node in f.maker.fgraph.toposort()] != [sigmoid,
                    T.mul, theano.tensor.inplace.neg_inplace])
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
        assert check_stack_trace(f, ops_to_check=[tensor.neg, sigmoid_inplace])
        assert [node.op for node in f.maker.fgraph.toposort()] == [
            tensor.neg, sigmoid_inplace]

        # tests inv_1_plus_exp
        f = theano.function([x], 1 - T.fill(x, 1.0) / (1 + T.exp(-x)), mode=m)
        assert check_stack_trace(f, ops_to_check=[tensor.neg, sigmoid_inplace])
        assert ([node.op for node in f.maker.fgraph.toposort()] == [tensor.neg,
                sigmoid_inplace])

    def test_local_sigm_times_exp(self):
        # Test the `local_sigm_times_exp` optimization.
        # exp(x) * sigm(-x) -> sigm(x)
        # exp(-x) * sigm(x) -> sigm(-x)

        def match(func, ops):
            # print [node.op.scalar_op for node in func.maker.fgraph.toposort()]
            assert [node.op for node in func.maker.fgraph.toposort()] == ops
        m = self.get_mode(excluding=['local_elemwise_fusion', 'inplace'])
        x, y = tensor.vectors('x', 'y')

        f = theano.function([x], sigmoid(-x) * tensor.exp(x), mode=m)
        match(f, [sigmoid])
        assert check_stack_trace(f, ops_to_check=sigmoid)

        f = theano.function([x], sigmoid(x) * tensor.exp(-x), mode=m)
        match(f, [tensor.neg, sigmoid])
        assert check_stack_trace(f, ops_to_check=sigmoid)

        f = theano.function([x], -(-(-(sigmoid(x)))) * tensor.exp(-x), mode=m)
        match(f, [tensor.neg, sigmoid, tensor.neg])
        # assert check_stack_trace(f, ops_to_check=sigmoid)

        f = theano.function(
            [x, y],
            (sigmoid(x) * sigmoid(-y) * -tensor.exp(-x) *
                tensor.exp(x * y) * tensor.exp(y)), mode=m)
        topo = f.maker.fgraph.toposort()
        for op, nb in [(sigmoid, 2), (tensor.mul, 2),
                       (tensor.neg, 1), (tensor.exp, 1)]:
            assert sum([n.op == op for n in topo]) == nb
        # assert check_stack_trace(f, ops_to_check=[sigmoid, tensor.mul,
        #                                           tensor.exp])

    def test_perform_sigm_times_exp(self):
        # Test the core function doing the `sigm_times_exp` optimization.
        #
        # It is easier to test different graph scenarios this way than by
        # compiling a theano function.

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
                print(trees[0])
                print(trees[1])
                print('***')
                theano.printing.debugprint(compute_mul(trees[0]))
                print('***')
                theano.printing.debugprint(compute_mul(trees[1]))
            assert good
        ok(sigmoid(x) * exp(-x), sigmoid(-x))
        ok(-x * sigmoid(x) * (y * (-1 * z) * exp(-x)),
           -x * sigmoid(-x) * (y * (-1 * z)))
        ok(-sigmoid(-x) *
           (exp(y) * (-exp(-z) * 3 * -exp(x)) *
            (y * 2 * (-sigmoid(-y) * (z + t) * exp(z)) * sigmoid(z))) * -
           sigmoid(x),
           sigmoid(x) *
           (-sigmoid(y) * (-sigmoid(-z) * 3) * (y * 2 * ((z + t) * exp(z)))) *
           (-sigmoid(x)))
        ok(exp(-x) * -exp(-x) * (-sigmoid(x) * -sigmoid(x)),
           -sigmoid(-x) * sigmoid(-x))
        ok(-exp(x) * -sigmoid(-x) * -exp(-x),
           -sigmoid(-x))

    def test_grad_log1msigm(self):
        # At some point, this returned nan, because (1 - sigm(x)) was
        # on both the numerator and the denominator of a fraction,
        # but the two nodes in question had not been merged.
        x = tensor.matrix('x')
        lr = tensor.scalar('lr')

        s = sigmoid(x)
        l = T.log(1 - s)
        c = l.mean()
        ux = x - lr * theano.grad(c, x)

        # Before the optimization, inf and NaN will be produced in the graph,
        # and DebugMode will complain. Everything is fine afterwards.
        mode = self.get_mode()
        if not isinstance(mode, theano.compile.DebugMode):
            f = theano.function([x, lr], ux, mode=mode)
            ux_v = f([[50]], 0.1)
            assert not np.isnan(ux_v)

    def test_local_ultra_fast_sigmoid(self):
        x = tensor.matrix('x')
        s = sigmoid(x)

        mode = self.get_mode('local_ultra_fast_sigmoid')
        f = theano.function([x], s, mode=mode)
        assert check_stack_trace(f, ops_to_check=sigmoid)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == sigmoid

        mode = self.get_mode().including('local_ultra_fast_sigmoid')
        f = theano.function([x], s, mode=mode)
        assert check_stack_trace(f, ops_to_check=ultra_fast_sigmoid)
        topo = f.maker.fgraph.toposort()
        assert topo[0].op == ultra_fast_sigmoid
        assert len(topo) == 1
        f([[-50, -10, -4, -1, 0, 1, 4, 10, 50]])

    def test_local_hard_sigmoid(self):
        x = tensor.matrix('x')
        s = sigmoid(x)

        mode = self.get_mode('local_hard_sigmoid')
        f = theano.function([x], s, mode=mode)
        assert check_stack_trace(f, ops_to_check=sigmoid)
        topo = f.maker.fgraph.toposort()
        assert topo[0].op == sigmoid
        assert len(topo) == 1

        mode = self.get_mode().including('local_hard_sigmoid')
        f = theano.function([x], s, mode=mode)
        topo = f.maker.fgraph.toposort()
        assert not any([n.op == sigmoid for n in topo])
        f([[-50, -10, -4, -1, 0, 1, 4, 10, 50]])

        mode2 = mode.excluding('fusion').excluding('inplace')
        f2 = theano.function([x], s, mode=mode2)
        self.assertTrue(check_stack_trace(f2, ops_to_check=theano.tensor.clip))


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

        # Fix ticket #4581 first
        # assert check_stack_trace(
        #     f, ops_to_check=(theano.scalar.Neg,
        #                      theano.tensor.nnet.sigm.ScalarSoftplus))
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 3
        assert isinstance(topo[0].op.scalar_op, theano.scalar.Neg)
        assert isinstance(topo[1].op.scalar_op,
                          theano.tensor.nnet.sigm.ScalarSoftplus)
        assert isinstance(topo[2].op.scalar_op, theano.scalar.Neg)
        f(np.random.rand(54).astype(config.floatX))

    def test_log1msigm_to_softplus(self):
        x = T.matrix()

        out = T.log(1 - sigmoid(x))
        f = theano.function([x], out, mode=self.m)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 2
        assert isinstance(topo[0].op.scalar_op,
                          theano.tensor.nnet.sigm.ScalarSoftplus)
        assert isinstance(topo[1].op.scalar_op, theano.scalar.Neg)
        # assert check_stack_trace(f, ops_to_check='all')
        f(np.random.rand(54, 11).astype(config.floatX))

        # Same test with a flatten
        out = T.log(1 - T.flatten(sigmoid(x)))
        f = theano.function([x], out, mode=self.m)

        # assert check_stack_trace(f, ops_to_check='all')
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 3
        assert tensor.is_flat(topo[0].outputs[0])
        assert isinstance(topo[1].op.scalar_op,
                          theano.tensor.nnet.sigm.ScalarSoftplus)
        assert isinstance(topo[2].op.scalar_op, theano.scalar.Neg)
        f(np.random.rand(54, 11).astype(config.floatX))

        # Same test with a reshape
        out = T.log(1 - sigmoid(x).reshape([x.size]))
        f = theano.function([x], out, mode=self.m)
        topo = f.maker.fgraph.toposort()
        # assert len(topo) == 3
        assert any(isinstance(node.op, T.Reshape) for node in topo)
        assert any(isinstance(getattr(node.op, 'scalar_op', None),
                              theano.tensor.nnet.sigm.ScalarSoftplus)
                   for node in topo)
        f(np.random.rand(54, 11).astype(config.floatX))

    def test_log1pexp_to_softplus(self):
        m = theano.config.mode
        if m == 'FAST_COMPILE':
            m = 'FAST_RUN'

        x = T.vector()

        out = T.log(1 + T.exp(x))
        f = theano.function([x], out, mode=self.m)

        # Fix ticket #4581 first
        # assert check_stack_trace(f, ops_to_check='all')
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op.scalar_op,
                          theano.tensor.nnet.sigm.ScalarSoftplus)
        f(np.random.rand(54).astype(config.floatX))


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
            assert is_1pexp(1 + exp(x), False) == (False, x)
            assert is_1pexp(exp(x) + 1, False) == (False, x)
            for neg, exp_arg in imap(lambda x:
                                     is_1pexp(x, only_process_constants=False),
                                     [(1 + exp(-x)), (exp(-x) + 1)]):
                assert not neg and theano.gof.graph.is_same_graph(exp_arg, -x)
            assert is_1pexp(1 - exp(x), False) is None
            assert is_1pexp(2 + exp(x), False) is None
            assert is_1pexp(exp(x) + 2, False) is None
            assert is_1pexp(exp(x) - 1, False) is None
            assert is_1pexp(-1 + exp(x), False) is None
            assert is_1pexp(1 + 2 * exp(x), False) is None
        finally:
            config.warn.identify_1pexp_bug = backup

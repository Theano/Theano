from __future__ import absolute_import, print_function, division

from theano.gof.type import Type
from theano.gof.graph import Variable, Apply, Constant
from theano.gof.op import Op
from theano.gof.opt import *  # noqa
from theano.gof.fg import FunctionGraph
from theano.gof.toolbox import *  # noqa

from theano import tensor as T


def as_variable(x):
    if not isinstance(x, Variable):
        raise TypeError("not a Variable", x)
    return x


class MyType(Type):

    def filter(self, data):
        return data

    def __eq__(self, other):
        return isinstance(other, MyType)

    def __hash__(self):
        return hash(MyType)


def MyVariable(name):
    return Variable(MyType(), None, None, name=name)


class MyOp(Op):

    def __init__(self, name, dmap=None, x=None):
        self.name = name
        if dmap is None:
            dmap = {}
        self.destroy_map = dmap
        self.x = x

    def make_node(self, *inputs):
        inputs = list(map(as_variable, inputs))
        for input in inputs:
            if not isinstance(input.type, MyType):
                raise Exception("Error 1")
        outputs = [MyType()()]
        return Apply(self, inputs, outputs)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        # rval = (self is other) or (isinstance(other, MyOp) and self.x is not None and self.x == other.x and self.name == other.name)
        rval = (self is other) or (isinstance(other, MyOp) and self.x is not None and self.x == other.x)
        return rval

    def __hash__(self):
        # return hash(self.x if self.x is not None else id(self)) ^ hash(self.name)
        if self.x is not None:
            return hash(self.x)
        else:
            return id(self)


op1 = MyOp('Op1')
op2 = MyOp('Op2')
op3 = MyOp('Op3')
op4 = MyOp('Op4')
op5 = MyOp('Op5')
op6 = MyOp('Op6')
op_d = MyOp('OpD', {0: [0]})

op_y = MyOp('OpY', x=1)
op_z = MyOp('OpZ', x=1)


def inputs():
    x = MyVariable('x')
    y = MyVariable('y')
    z = MyVariable('z')
    return x, y, z


def PatternOptimizer(p1, p2, ign=False):
    return OpKeyOptimizer(PatternSub(p1, p2), ignore_newtrees=ign)


def TopoPatternOptimizer(p1, p2, ign=True):
    return TopoOptimizer(PatternSub(p1, p2), ignore_newtrees=ign)


class TestPatternOptimizer:

    def test_replace_output(self):
        # replacing the whole graph
        x, y, z = inputs()
        e = op1(op2(x, y), z)
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer((op1, (op2, '1', '2'), '3'),
                         (op4, '3', '2')).optimize(g)
        assert str(g) == "[Op4(z, y)]"

    def test_nested_out_pattern(self):
        x, y, z = inputs()
        e = op1(x, y)
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer((op1, '1', '2'),
                         (op4, (op1, '1'), (op2, '2'), (op3, '1', '2'))).optimize(g)
        assert str(g) == "[Op4(Op1(x), Op2(y), Op3(x, y))]"

    def test_unification_1(self):
        x, y, z = inputs()
        e = op1(op2(x, x), z)  # the arguments to op2 are the same
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer((op1, (op2, '1', '1'), '2'),  # they are the same in the pattern
                         (op4, '2', '1')).optimize(g)
        # So the replacement should occur
        assert str(g) == "[Op4(z, x)]"

    def test_unification_2(self):
        x, y, z = inputs()
        e = op1(op2(x, y), z)  # the arguments to op2 are different
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer((op1, (op2, '1', '1'), '2'),  # they are the same in the pattern
                         (op4, '2', '1')).optimize(g)
        # The replacement should NOT occur
        assert str(g) == "[Op1(Op2(x, y), z)]"

    def test_replace_subgraph(self):
        # replacing inside the graph
        x, y, z = inputs()
        e = op1(op2(x, y), z)
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer((op2, '1', '2'),
                         (op1, '2', '1')).optimize(g)
        assert str(g) == "[Op1(Op1(y, x), z)]"

    def test_no_recurse(self):
        # if the out pattern is an acceptable in pattern
        # and that the ignore_newtrees flag is True,
        # it should do the replacement and stop
        x, y, z = inputs()
        e = op1(op2(x, y), z)
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer((op2, '1', '2'),
                         (op2, '2', '1'), ign=True).optimize(g)
        assert str(g) == "[Op1(Op2(y, x), z)]"

    def test_multiple(self):
        # it should replace all occurrences of the pattern
        x, y, z = inputs()
        e = op1(op2(x, y), op2(x, y), op2(y, z))
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer((op2, '1', '2'),
                         (op4, '1')).optimize(g)
        assert str(g) == "[Op1(Op4(x), Op4(x), Op4(y))]"

    def test_nested_even(self):
        # regardless of the order in which we optimize, this
        # should work
        x, y, z = inputs()
        e = op1(op1(op1(op1(x))))
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer((op1, (op1, '1')),
                         '1').optimize(g)
        assert str(g) == "[x]"

    def test_nested_odd(self):
        x, y, z = inputs()
        e = op1(op1(op1(op1(op1(x)))))
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer((op1, (op1, '1')),
                         '1').optimize(g)
        assert str(g) == "[Op1(x)]"

    def test_expand(self):
        x, y, z = inputs()
        e = op1(op1(op1(x)))
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer((op1, '1'),
                         (op2, (op1, '1')), ign=True).optimize(g)
        assert str(g) == "[Op2(Op1(Op2(Op1(Op2(Op1(x))))))]"

    def test_ambiguous(self):
        # this test should always work with TopoOptimizer and the
        # ignore_newtrees flag set to False. Behavior with ignore_newtrees
        # = True or with other NavigatorOptimizers may differ.
        x, y, z = inputs()
        e = op1(op1(op1(op1(op1(x)))))
        g = FunctionGraph([x, y, z], [e])
        TopoPatternOptimizer((op1, (op1, '1')),
                             (op1, '1'), ign=False).optimize(g)
        assert str(g) == "[Op1(x)]"

    def test_constant_unification(self):
        x = Constant(MyType(), 2, name='x')
        y = MyVariable('y')
        z = Constant(MyType(), 2, name='z')
        e = op1(op1(x, y), y)
        g = FunctionGraph([y], [e])
        PatternOptimizer((op1, z, '1'),
                         (op2, '1', z)).optimize(g)
        assert str(g) == "[Op1(Op2(y, z), y)]"

    def test_constraints(self):
        x, y, z = inputs()
        e = op4(op1(op2(x, y)), op1(op1(x, y)))
        g = FunctionGraph([x, y, z], [e])

        def constraint(r):
            # Only replacing if the input is an instance of Op2
            return r.owner.op == op2
        PatternOptimizer((op1, {'pattern': '1',
                                'constraint': constraint}),
                         (op3, '1')).optimize(g)
        assert str(g) == "[Op4(Op3(Op2(x, y)), Op1(Op1(x, y)))]"

    def test_match_same(self):
        x, y, z = inputs()
        e = op1(x, x)
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer((op1, 'x', 'y'),
                         (op3, 'x', 'y')).optimize(g)
        assert str(g) == "[Op3(x, x)]"

    def test_match_same_illegal(self):
        x, y, z = inputs()
        e = op2(op1(x, x), op1(x, y))
        g = FunctionGraph([x, y, z], [e])

        def constraint(r):
            # Only replacing if the input is an instance of Op2
            return r.owner.inputs[0] is not r.owner.inputs[1]
        PatternOptimizer({'pattern': (op1, 'x', 'y'),
                          'constraint': constraint},
                         (op3, 'x', 'y')).optimize(g)
        assert str(g) == "[Op2(Op1(x, x), Op3(x, y))]"

    def test_multi(self):
        x, y, z = inputs()
        e0 = op1(x, y)
        e = op3(op4(e0), e0)
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer((op4, (op1, 'x', 'y')),
                         (op3, 'x', 'y')).optimize(g)
        assert str(g) == "[Op3(Op4(*1 -> Op1(x, y)), *1)]"

    def test_eq(self):
        # replacing the whole graph
        x, y, z = inputs()
        e = op1(op_y(x, y), z)
        g = FunctionGraph([x, y, z], [e])
        PatternOptimizer((op1, (op_z, '1', '2'), '3'),
                         (op4, '3', '2')).optimize(g)
        str_g = str(g)
        assert str_g == "[Op4(z, y)]"

#     def test_multi_ingraph(self):
#         # known to fail
#         x, y, z = inputs()
#         e0 = op1(x, y)
#         e = op4(e0, e0)
#         g = FunctionGraph([x, y, z], [e])
#         PatternOptimizer((op4, (op1, 'x', 'y'), (op1, 'x', 'y')),
#                          (op3, 'x', 'y')).optimize(g)
#         assert str(g) == "[Op3(x, y)]"


def OpSubOptimizer(op1, op2):
    return OpKeyOptimizer(OpSub(op1, op2))


class TestOpSubOptimizer:

    def test_straightforward(self):
        x, y, z = inputs()
        e = op1(op1(op1(op1(op1(x)))))
        g = FunctionGraph([x, y, z], [e])
        OpSubOptimizer(op1, op2).optimize(g)
        assert str(g) == "[Op2(Op2(Op2(Op2(Op2(x)))))]"

    def test_straightforward_2(self):
        x, y, z = inputs()
        e = op1(op2(x), op3(y), op4(z))
        g = FunctionGraph([x, y, z], [e])
        OpSubOptimizer(op3, op4).optimize(g)
        assert str(g) == "[Op1(Op2(x), Op4(y), Op4(z))]"


class NoInputOp(Op):
    __props__ = ('param',)

    def __init__(self, param):
        self.param = param

    def make_node(self):
        return Apply(self, [], [MyType()()])

    def perform(self, node, inputs, output_storage):
        output_storage[0][0] = self.param


class TestMergeOptimizer:

    def test_straightforward(self):
        x, y, z = inputs()
        e = op1(op2(x, y), op2(x, y), op2(x, z))
        g = FunctionGraph([x, y, z], [e])
        MergeOptimizer().optimize(g)
        assert str(g) == "[Op1(*1 -> Op2(x, y), *1, Op2(x, z))]"

    def test_constant_merging(self):
        x = MyVariable('x')
        y = Constant(MyType(), 2, name='y')
        z = Constant(MyType(), 2, name='z')
        e = op1(op2(x, y), op2(x, y), op2(x, z))
        g = FunctionGraph([x, y, z], [e])
        MergeOptimizer().optimize(g)
        strg = str(g)
        assert strg == "[Op1(*1 -> Op2(x, y), *1, *1)]" \
            or strg == "[Op1(*1 -> Op2(x, z), *1, *1)]"

    def test_deep_merge(self):
        x, y, z = inputs()
        e = op1(op3(op2(x, y), z), op4(op3(op2(x, y), z)))
        g = FunctionGraph([x, y, z], [e])
        MergeOptimizer().optimize(g)
        assert str(g) == "[Op1(*1 -> Op3(Op2(x, y), z), Op4(*1))]"

    def test_no_merge(self):
        x, y, z = inputs()
        e = op1(op3(op2(x, y)), op3(op2(y, x)))
        g = FunctionGraph([x, y, z], [e])
        MergeOptimizer().optimize(g)
        assert str(g) == "[Op1(Op3(Op2(x, y)), Op3(Op2(y, x)))]"

    def test_merge_outputs(self):
        x, y, z = inputs()
        e1 = op3(op2(x, y))
        e2 = op3(op2(x, y))
        g = FunctionGraph([x, y, z], [e1, e2])
        MergeOptimizer().optimize(g)
        assert str(g) == "[*1 -> Op3(Op2(x, y)), *1]"

    def test_multiple_merges(self):
        x, y, z = inputs()
        e1 = op1(x, y)
        e2 = op2(op3(x), y, z)
        e = op1(e1, op4(e2, e1), op1(e2))
        g = FunctionGraph([x, y, z], [e])
        MergeOptimizer().optimize(g)
        strg = str(g)
        # note: graph.as_string can only produce the following two possibilities, but if
        # the implementation was to change there are 6 other acceptable answers.
        assert strg == "[Op1(*1 -> Op1(x, y), Op4(*2 -> Op2(Op3(x), y, z), *1), Op1(*2))]" \
            or strg == "[Op1(*2 -> Op1(x, y), Op4(*1 -> Op2(Op3(x), y, z), *2), Op1(*1))]"

    def test_identical_constant_args(self):
        x = MyVariable('x')
        y = Constant(MyType(), 2, name='y')
        z = Constant(MyType(), 2, name='z')
        ctv_backup = config.compute_test_value
        config.compute_test_value = 'off'
        try:
            e1 = op1(y, z)
        finally:
            config.compute_test_value = ctv_backup
        g = FunctionGraph([x, y, z], [e1])
        MergeOptimizer().optimize(g)
        strg = str(g)
        assert strg == '[Op1(y, y)]' or strg == '[Op1(z, z)]'

    def est_one_assert_merge(self):
        # Merge two nodes, one has assert, the other not.
        x1 = T.matrix('x1')
        x2 = T.matrix('x2')
        e = T.dot(x1, x2) + T.dot(T.opt.assert_op(x1, (x1 > x2).all()), x2)
        g = FunctionGraph([x1, x2], [e])
        MergeOptimizer().optimize(g)
        strg = theano.printing.debugprint(g, file='str')
        strref = '''Elemwise{add,no_inplace} [id A] ''   4
 |dot [id B] ''   3
 | |Assert{msg='Theano Assert failed!'} [id C] ''   2
 | | |x1 [id D]
 | | |All [id E] ''   1
 | |   |Elemwise{gt,no_inplace} [id F] ''   0
 | |     |x1 [id D]
 | |     |x2 [id G]
 | |x2 [id G]
 |dot [id B] ''   3
'''
        assert strg == strref, (strg, strref)

    def est_both_assert_merge_1(self):
        # Merge two nodes, both have assert on the same node
        # with different conditions.
        x1 = T.matrix('x1')
        x2 = T.matrix('x2')
        x3 = T.matrix('x3')
        e = T.dot(T.opt.assert_op(x1, (x1 > x3).all()), x2) +\
            T.dot(T.opt.assert_op(x1, (x1 > x2).all()), x2)
        g = FunctionGraph([x1, x2, x3], [e])
        MergeOptimizer().optimize(g)
        strg = theano.printing.debugprint(g, file='str')
        strref1 = '''Elemwise{add,no_inplace} [id A] ''   6
 |dot [id B] ''   5
 | |Assert{msg='Theano Assert failed!'} [id C] ''   4
 | | |x1 [id D]
 | | |All [id E] ''   3
 | | | |Elemwise{gt,no_inplace} [id F] ''   1
 | | |   |x1 [id D]
 | | |   |x3 [id G]
 | | |All [id H] ''   2
 | |   |Elemwise{gt,no_inplace} [id I] ''   0
 | |     |x1 [id D]
 | |     |x2 [id J]
 | |x2 [id J]
 |dot [id B] ''   5
'''
        strref2 = '''Elemwise{add,no_inplace} [id A] ''   6
 |dot [id B] ''   5
 | |Assert{msg='Theano Assert failed!'} [id C] ''   4
 | | |x1 [id D]
 | | |All [id E] ''   3
 | | | |Elemwise{gt,no_inplace} [id F] ''   1
 | | |   |x1 [id D]
 | | |   |x2 [id G]
 | | |All [id H] ''   2
 | |   |Elemwise{gt,no_inplace} [id I] ''   0
 | |     |x1 [id D]
 | |     |x3 [id J]
 | |x2 [id G]
 |dot [id B] ''   5
'''
        # print(strg)
        assert strg == strref1 or strg == strref2, (strg, strref1, strref2)

    def est_both_assert_merge_2(self):
        # Merge two nodes, both have assert on different node
        x1 = T.matrix('x1')
        x2 = T.matrix('x2')
        x3 = T.matrix('x3')
        e = T.dot(T.opt.assert_op(x1, (x1 > x3).all()), x2) +\
            T.dot(x1, T.opt.assert_op(x2, (x2 > x3).all()))
        g = FunctionGraph([x1, x2, x3], [e])
        MergeOptimizer().optimize(g)
        strg = theano.printing.debugprint(g, file='str')
        strref = '''Elemwise{add,no_inplace} [id A] ''   7
 |dot [id B] ''   6
 | |Assert{msg='Theano Assert failed!'} [id C] ''   5
 | | |x1 [id D]
 | | |All [id E] ''   3
 | |   |Elemwise{gt,no_inplace} [id F] ''   1
 | |     |x1 [id D]
 | |     |x3 [id G]
 | |Assert{msg='Theano Assert failed!'} [id H] ''   4
 |   |x2 [id I]
 |   |All [id J] ''   2
 |     |Elemwise{gt,no_inplace} [id K] ''   0
 |       |x2 [id I]
 |       |x3 [id G]
 |dot [id B] ''   6
'''
        # print(strg)
        assert strg == strref, (strg, strref)

    def est_both_assert_merge_2_reverse(self):
        # Test case "test_both_assert_merge_2" but in reverse order
        x1 = T.matrix('x1')
        x2 = T.matrix('x2')
        x3 = T.matrix('x3')
        e = T.dot(x1, T.opt.assert_op(x2, (x2 > x3).all())) +\
            T.dot(T.opt.assert_op(x1, (x1 > x3).all()), x2)
        g = FunctionGraph([x1, x2, x3], [e])
        MergeOptimizer().optimize(g)
        strg = theano.printing.debugprint(g, file='str')
        strref = '''Elemwise{add,no_inplace} [id A] ''   7
 |dot [id B] ''   6
 | |Assert{msg='Theano Assert failed!'} [id C] ''   5
 | | |x1 [id D]
 | | |All [id E] ''   3
 | |   |Elemwise{gt,no_inplace} [id F] ''   1
 | |     |x1 [id D]
 | |     |x3 [id G]
 | |Assert{msg='Theano Assert failed!'} [id H] ''   4
 |   |x2 [id I]
 |   |All [id J] ''   2
 |     |Elemwise{gt,no_inplace} [id K] ''   0
 |       |x2 [id I]
 |       |x3 [id G]
 |dot [id B] ''   6
'''
        print(strg)
        assert strg == strref, (strg, strref)

    def test_merge_noinput(self):
        # Check that identical Apply nodes without inputs will be merged
        x = NoInputOp(param=0)()
        y = NoInputOp(param=0)()
        z = NoInputOp(param=1)()

        fg = FunctionGraph([], [x, y, z])
        MergeOptimizer().optimize(fg)
        no_input_ops = [n for n in fg.apply_nodes
                        if isinstance(n.op, NoInputOp)]
        assert len(no_input_ops) == 2, fg.apply_nodes


class TestEquilibrium(object):

    def test_1(self):
        x, y, z = map(MyVariable, 'xyz')
        e = op3(op4(x, y))
        g = FunctionGraph([x, y, z], [e])
        # print g
        opt = EquilibriumOptimizer(
            [PatternSub((op1, 'x', 'y'), (op2, 'x', 'y')),
             PatternSub((op4, 'x', 'y'), (op1, 'x', 'y')),
             PatternSub((op3, (op2, 'x', 'y')), (op4, 'x', 'y'))
             ],
            max_use_ratio=10)
        opt.optimize(g)
        # print g
        assert str(g) == '[Op2(x, y)]'

    def test_2(self):
        x, y, z = map(MyVariable, 'xyz')
        e = op1(op1(op3(x, y)))
        g = FunctionGraph([x, y, z], [e])
        # print g
        opt = EquilibriumOptimizer(
            [PatternSub((op1, (op2, 'x', 'y')), (op4, 'x', 'y')),
             PatternSub((op3, 'x', 'y'), (op4, 'x', 'y')),
             PatternSub((op4, 'x', 'y'), (op5, 'x', 'y')),
             PatternSub((op5, 'x', 'y'), (op6, 'x', 'y')),
             PatternSub((op6, 'x', 'y'), (op2, 'x', 'y'))
             ],
            max_use_ratio=10)
        opt.optimize(g)
        assert str(g) == '[Op2(x, y)]'

    def test_low_use_ratio(self):
        x, y, z = map(MyVariable, 'xyz')
        e = op3(op4(x, y))
        g = FunctionGraph([x, y, z], [e])
        # print 'before', g
        # display pesky warnings along with stdout
        # also silence logger for 'theano.gof.opt'
        _logger = logging.getLogger('theano.gof.opt')
        oldlevel = _logger.level
        _logger.setLevel(logging.CRITICAL)
        try:
            opt = EquilibriumOptimizer(
                [PatternSub((op1, 'x', 'y'), (op2, 'x', 'y')),
                 PatternSub((op4, 'x', 'y'), (op1, 'x', 'y')),
                 PatternSub((op3, (op2, 'x', 'y')), (op4, 'x', 'y'))
                 ],
                max_use_ratio=1. / len(g.apply_nodes))  # each opt can only be applied once
            opt.optimize(g)
        finally:
            _logger.setLevel(oldlevel)
        # print 'after', g
        assert str(g) == '[Op1(x, y)]'


def test_pre_constant_merge_slice():
    ms = theano.tensor.type_other.MakeSlice()(1)
    pre_constant_merge([ms])
    const_slice = theano.tensor.type_other.SliceConstant(
        type=theano.tensor.type_other.slicetype,
        data=slice(1, None, 2))
    adv = theano.tensor.subtensor.AdvancedSubtensor()(theano.tensor.matrix(),
                                                      [2, 3], const_slice)
    pre_constant_merge(adv)

    cst = pre_greedy_local_optimizer([theano.tensor.opt.constant_folding], ms)
    assert isinstance(cst, theano.tensor.type_other.SliceConstant)

    # Make sure constant of slice signature is hashable.
    hash(cst.signature())

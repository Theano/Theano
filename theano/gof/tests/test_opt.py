
from theano.gof.type import Type
from theano.gof.graph import Variable, Apply, Constant
from theano.gof.op import Op
from theano.gof.opt import *
from theano.gof.env import Env
from theano.gof.toolbox import *


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
    return Variable(MyType(), None, None, name = name)


class MyOp(Op):

    def __init__(self, name, dmap = {}, x = None):
        self.name = name
        self.destroy_map = dmap
        self.x = x

    def make_node(self, *inputs):
        inputs = map(as_variable, inputs)
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
        #rval = (self is other) or (isinstance(other, MyOp) and self.x is not None and self.x == other.x and self.name == other.name)
        rval = (self is other) or (isinstance(other, MyOp) and self.x is not None and self.x == other.x)
        return rval

    def __hash__(self):
        #return hash(self.x if self.x is not None else id(self)) ^ hash(self.name)
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

op_y = MyOp('OpY', x = 1)
op_z = MyOp('OpZ', x = 1)



def inputs():
    x = MyVariable('x')
    y = MyVariable('y')
    z = MyVariable('z')
    return x, y, z


PatternOptimizer = lambda p1, p2, ign=False: OpKeyOptimizer(PatternSub(p1, p2), ignore_newtrees=ign)
TopoPatternOptimizer = lambda p1, p2, ign=True: TopoOptimizer(PatternSub(p1, p2), ignore_newtrees=ign)

class TestPatternOptimizer:

    def test_replace_output(self):
        # replacing the whole graph
        x, y, z = inputs()
        e = op1(op2(x, y), z)
        g = Env([x, y, z], [e])
        PatternOptimizer((op1, (op2, '1', '2'), '3'),
                         (op4, '3', '2')).optimize(g)
        assert str(g) == "[Op4(z, y)]"

    def test_nested_out_pattern(self):
        x, y, z = inputs()
        e = op1(x, y)
        g = Env([x, y, z], [e])
        PatternOptimizer((op1, '1', '2'),
                         (op4, (op1, '1'), (op2, '2'), (op3, '1', '2'))).optimize(g)
        assert str(g) == "[Op4(Op1(x), Op2(y), Op3(x, y))]"

    def test_unification_1(self):
        x, y, z = inputs()
        e = op1(op2(x, x), z) # the arguments to op2 are the same
        g = Env([x, y, z], [e])
        PatternOptimizer((op1, (op2, '1', '1'), '2'), # they are the same in the pattern
                         (op4, '2', '1')).optimize(g)
        # So the replacement should occur
        assert str(g) == "[Op4(z, x)]"

    def test_unification_2(self):
        x, y, z = inputs()
        e = op1(op2(x, y), z) # the arguments to op2 are different
        g = Env([x, y, z], [e])
        PatternOptimizer((op1, (op2, '1', '1'), '2'), # they are the same in the pattern
                         (op4, '2', '1')).optimize(g)
        # The replacement should NOT occur
        assert str(g) == "[Op1(Op2(x, y), z)]"

    def test_replace_subgraph(self):
        # replacing inside the graph
        x, y, z = inputs()
        e = op1(op2(x, y), z)
        g = Env([x, y, z], [e])
        PatternOptimizer((op2, '1', '2'),
                         (op1, '2', '1')).optimize(g)
        assert str(g) == "[Op1(Op1(y, x), z)]"

    def test_no_recurse(self):
        # if the out pattern is an acceptable in pattern
        # and that the ignore_newtrees flag is True,
        # it should do the replacement and stop
        x, y, z = inputs()
        e = op1(op2(x, y), z)
        g = Env([x, y, z], [e])
        PatternOptimizer((op2, '1', '2'),
                         (op2, '2', '1'), ign=True).optimize(g)
        assert str(g) == "[Op1(Op2(y, x), z)]"

    def test_multiple(self):
        # it should replace all occurrences of the pattern
        x, y, z = inputs()
        e = op1(op2(x, y), op2(x, y), op2(y, z))
        g = Env([x, y, z], [e])
        PatternOptimizer((op2, '1', '2'),
                         (op4, '1')).optimize(g)
        assert str(g) == "[Op1(Op4(x), Op4(x), Op4(y))]"

    def test_nested_even(self):
        # regardless of the order in which we optimize, this
        # should work
        x, y, z = inputs()
        e = op1(op1(op1(op1(x))))
        g = Env([x, y, z], [e])
        PatternOptimizer((op1, (op1, '1')),
                         '1').optimize(g)
        assert str(g) == "[x]"

    def test_nested_odd(self):
        x, y, z = inputs()
        e = op1(op1(op1(op1(op1(x)))))
        g = Env([x, y, z], [e])
        PatternOptimizer((op1, (op1, '1')),
                         '1').optimize(g)
        assert str(g) == "[Op1(x)]"

    def test_expand(self):
        x, y, z = inputs()
        e = op1(op1(op1(x)))
        g = Env([x, y, z], [e])
        PatternOptimizer((op1, '1'),
                         (op2, (op1, '1')), ign=True).optimize(g)
        assert str(g) == "[Op2(Op1(Op2(Op1(Op2(Op1(x))))))]"

    def test_ambiguous(self):
        # this test should always work with TopoOptimizer and the
        # ignore_newtrees flag set to False. Behavior with ignore_newtrees
        # = True or with other NavigatorOptimizers may differ.
        x, y, z = inputs()
        e = op1(op1(op1(op1(op1(x)))))
        g = Env([x, y, z], [e])
        TopoPatternOptimizer((op1, (op1, '1')),
                             (op1, '1'), ign=False).optimize(g)
        assert str(g) == "[Op1(x)]"

    def test_constant_unification(self):
        x = Constant(MyType(), 2, name = 'x')
        y = MyVariable('y')
        z = Constant(MyType(), 2, name = 'z')
        e = op1(op1(x, y), y)
        g = Env([y], [e])
        PatternOptimizer((op1, z, '1'),
                         (op2, '1', z)).optimize(g)
        assert str(g) == "[Op1(Op2(y, z), y)]"

    def test_constraints(self):
        x, y, z = inputs()
        e = op4(op1(op2(x, y)), op1(op1(x, y)))
        g = Env([x, y, z], [e])
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
        g = Env([x, y, z], [e])
        PatternOptimizer((op1, 'x', 'y'),
                         (op3, 'x', 'y')).optimize(g)
        assert str(g) == "[Op3(x, x)]"

    def test_match_same_illegal(self):
        x, y, z = inputs()
        e = op2(op1(x, x), op1(x, y))
        g = Env([x, y, z], [e])
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
        g = Env([x, y, z], [e])
        PatternOptimizer((op4, (op1, 'x', 'y')),
                         (op3, 'x', 'y')).optimize(g)
        assert str(g) == "[Op3(Op4(*1 -> Op1(x, y)), *1)]"

    def test_eq(self):
        # replacing the whole graph
        x, y, z = inputs()
        e = op1(op_y(x, y), z)
        g = Env([x, y, z], [e])
        PatternOptimizer((op1, (op_z, '1', '2'), '3'),
                         (op4, '3', '2')).optimize(g)
        str_g = str(g)
        assert str_g == "[Op4(z, y)]"

#     def test_multi_ingraph(self):
#         # known to fail
#         x, y, z = inputs()
#         e0 = op1(x, y)
#         e = op4(e0, e0)
#         g = Env([x, y, z], [e])
#         PatternOptimizer((op4, (op1, 'x', 'y'), (op1, 'x', 'y')),
#                          (op3, 'x', 'y')).optimize(g)
#         assert str(g) == "[Op3(x, y)]"


OpSubOptimizer = lambda op1, op2: TopoOptimizer(OpSub(op1, op2))
OpSubOptimizer = lambda op1, op2: OpKeyOptimizer(OpSub(op1, op2))

class TestOpSubOptimizer:

    def test_straightforward(self):
        x, y, z = inputs()
        e = op1(op1(op1(op1(op1(x)))))
        g = Env([x, y, z], [e])
        OpSubOptimizer(op1, op2).optimize(g)
        assert str(g) == "[Op2(Op2(Op2(Op2(Op2(x)))))]"

    def test_straightforward_2(self):
        x, y, z = inputs()
        e = op1(op2(x), op3(y), op4(z))
        g = Env([x, y, z], [e])
        OpSubOptimizer(op3, op4).optimize(g)
        assert str(g) == "[Op1(Op2(x), Op4(y), Op4(z))]"


class TestMergeOptimizer:

    def test_straightforward(self):
        x, y, z = inputs()
        e = op1(op2(x, y), op2(x, y), op2(x, z))
        g = Env([x, y, z], [e])
        MergeOptimizer().optimize(g)
        assert str(g) == "[Op1(*1 -> Op2(x, y), *1, Op2(x, z))]"

    def test_constant_merging(self):
        x = MyVariable('x')
        y = Constant(MyType(), 2, name = 'y')
        z = Constant(MyType(), 2, name = 'z')
        e = op1(op2(x, y), op2(x, y), op2(x, z))
        g = Env([x, y, z], [e])
        MergeOptimizer().optimize(g)
        strg = str(g)
        assert strg == "[Op1(*1 -> Op2(x, y), *1, *1)]" \
            or strg == "[Op1(*1 -> Op2(x, z), *1, *1)]"

    def test_deep_merge(self):
        x, y, z = inputs()
        e = op1(op3(op2(x, y), z), op4(op3(op2(x, y), z)))
        g = Env([x, y, z], [e])
        MergeOptimizer().optimize(g)
        assert str(g) == "[Op1(*1 -> Op3(Op2(x, y), z), Op4(*1))]"

    def test_no_merge(self):
        x, y, z = inputs()
        e = op1(op3(op2(x, y)), op3(op2(y, x)))
        g = Env([x, y, z], [e])
        MergeOptimizer().optimize(g)
        assert str(g) == "[Op1(Op3(Op2(x, y)), Op3(Op2(y, x)))]"

    def test_merge_outputs(self):
        x, y, z = inputs()
        e1 = op3(op2(x, y))
        e2 = op3(op2(x, y))
        g = Env([x, y, z], [e1, e2])
        MergeOptimizer().optimize(g)
        assert str(g) == "[*1 -> Op3(Op2(x, y)), *1]"

    def test_multiple_merges(self):
        x, y, z = inputs()
        e1 = op1(x, y)
        e2 = op2(op3(x), y, z)
        e = op1(e1, op4(e2, e1), op1(e2))
        g = Env([x, y, z], [e])
        MergeOptimizer().optimize(g)
        strg = str(g)
        # note: graph.as_string can only produce the following two possibilities, but if
        # the implementation was to change there are 6 other acceptable answers.
        assert strg == "[Op1(*1 -> Op1(x, y), Op4(*2 -> Op2(Op3(x), y, z), *1), Op1(*2))]" \
            or strg == "[Op1(*2 -> Op1(x, y), Op4(*1 -> Op2(Op3(x), y, z), *2), Op1(*1))]"

    def test_identical_constant_args(self):
        x = MyVariable('x')
        y = Constant(MyType(), 2, name = 'y')
        z = Constant(MyType(), 2, name = 'z')
        ctv_backup = config.compute_test_value
        config.compute_test_value = 'off'
        try:
            e1 = op1(y, z)
        finally:
            config.compute_test_value = ctv_backup
        g = Env([x, y, z], [e1])
        MergeOptimizer().optimize(g)
        strg = str(g)
        assert strg == '[Op1(y, y)]' or strg == '[Op1(z, z)]'


class TestEquilibrium(object):

    def test_1(self):
        x, y, z = map(MyVariable, 'xyz')
        e = op3(op4(x, y))
        g = Env([x, y, z], [e])
        print g
        opt = EquilibriumOptimizer(
            [PatternSub((op1, 'x', 'y'), (op2, 'x', 'y')),
             PatternSub((op4, 'x', 'y'), (op1, 'x', 'y')),
             PatternSub((op3, (op2, 'x', 'y')), (op4, 'x', 'y'))
             ],
            max_use_ratio = 10)
        opt.optimize(g)
        print g
        assert str(g) == '[Op2(x, y)]'

    def test_2(self):
        x, y, z = map(MyVariable, 'xyz')
        e = op1(op1(op3(x, y)))
        g = Env([x, y, z], [e])
        print g
        opt = EquilibriumOptimizer(
            [PatternSub((op1, (op2, 'x', 'y')), (op4, 'x', 'y')),
             PatternSub((op3, 'x', 'y'), (op4, 'x', 'y')),
             PatternSub((op4, 'x', 'y'), (op5, 'x', 'y')),
             PatternSub((op5, 'x', 'y'), (op6, 'x', 'y')),
             PatternSub((op6, 'x', 'y'), (op2, 'x', 'y'))
             ],
            max_use_ratio = 10)
        opt.optimize(g)
        assert str(g) == '[Op2(x, y)]'

    def test_low_use_ratio(self):
        x, y, z = map(MyVariable, 'xyz')
        e = op3(op4(x, y))
        g = Env([x, y, z], [e])
        print 'before', g
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
                max_use_ratio = 1. / len(g.nodes)) # each opt can only be applied once
            opt.optimize(g)
        finally:
            _logger.setLevel(oldlevel)
        print 'after', g
        assert str(g) == '[Op1(x, y)]'

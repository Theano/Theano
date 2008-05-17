
import unittest

from type import Type
from graph import Result, Apply, Constant
from op import Op, Macro
from opt import *
from env import Env
from toolbox import *


def as_result(x):
    assert isinstance(x, Result)
    return x


class MyType(Type):

    def filter(self, data):
        return data

    def __eq__(self, other):
        return isinstance(other, MyType)


def MyResult(name):
    return Result(MyType(), None, None, name = name)


class MyOp(Op):

    def __init__(self, name, dmap = {}, x = None):
        self.name = name
        self.destroy_map = dmap
        self.x = x
    
    def make_node(self, *inputs):
        inputs = map(as_result, inputs)
        for input in inputs:
            if not isinstance(input.type, MyType):
                raise Exception("Error 1")
        outputs = [MyType()()]
        return Apply(self, inputs, outputs)

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self is other or isinstance(other, MyOp) and self.x is not None and self.x == other.x

    def __hash__(self):
        return self.x if self.x is not None else id(self)

op1 = MyOp('Op1')
op2 = MyOp('Op2')
op3 = MyOp('Op3')
op4 = MyOp('Op4')
op_d = MyOp('OpD', {0: [0]})

op_y = MyOp('OpY', x = 1)
op_z = MyOp('OpZ', x = 1)



def inputs():
    x = MyResult('x')
    y = MyResult('y')
    z = MyResult('z')
    return x, y, z


class _test_PatternOptimizer(unittest.TestCase):
    
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
        # if the out pattern is an acceptable in pattern,
        # it should do the replacement and stop
        x, y, z = inputs()
        e = op1(op2(x, y), z)
        g = Env([x, y, z], [e])
        PatternOptimizer((op2, '1', '2'),
                         (op2, '2', '1')).optimize(g)
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
                         (op2, (op1, '1'))).optimize(g)
        assert str(g) == "[Op2(Op1(Op2(Op1(Op2(Op1(x))))))]"

#     def test_ambiguous(self):
#         # this test is known to fail most of the time
#         # the reason is that PatternOptimizer doesn't go through
#         # the ops in topological order. The order is random and
#         # it does not visit ops that it creates.
#         x, y, z = inputs()
#         e = op1(op1(op1(op1(op1(x)))))
#         g = Env([x, y, z], [e])
#         PatternOptimizer((op1, (op1, '1')),
#                          (op1, '1')).optimize(g)
#         assert str(g) == "[Op1(x)]"

    def test_constant_unification(self):
        x = Constant(MyType(), 2, name = 'x')
        y = MyResult('y')
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
        def constraint(env, r):
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
        def constraint(env, r):
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
        assert str(g) == "[Op4(z, y)]"

#     def test_multi_ingraph(self):
#         # known to fail
#         x, y, z = inputs()
#         e0 = op1(x, y)
#         e = op4(e0, e0)
#         g = Env([x, y, z], [e])
#         PatternOptimizer((op4, (op1, 'x', 'y'), (op1, 'x', 'y')),
#                          (op3, 'x', 'y')).optimize(g)
#         assert str(g) == "[Op3(x, y)]"
        

# class _test_PatternDescOptimizer(unittest.TestCase):
    
#     def test_replace_output(self):
#         # replacing the whole graph
#         x, y, z = inputs()
#         e = op1(op2(x, y), z)
#         g = env([x, y, z], [e])
#         PatternDescOptimizer((Op1, (Op2, '1', '2'), '3'),
#                              (Op4, '3', '2')).optimize(g)
#         assert str(g) == "[Op4(z, y)]"

#     def test_eq(self):
#         x, y, z = inputs()
#         e = op1(op_y(x, y, 37, 88), op2(op_y(y, z, 1, 7)))
#         g = env([x, y, z], [e])
#         PatternDescOptimizer((op_z, '1', '2'),
#                              (op3, '2', '1')).optimize(g)
#         assert str(g) == "[Op1(Op3(y, x), Op2(OpZ(y, z)))]"


class _test_OpSubOptimizer(unittest.TestCase):
    
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


class _test_MergeOptimizer(unittest.TestCase):

    def test_straightforward(self):
        x, y, z = inputs()
        e = op1(op2(x, y), op2(x, y), op2(x, z))
        g = Env([x, y, z], [e])
        MergeOptimizer().optimize(g)
        assert str(g) == "[Op1(*1 -> Op2(x, y), *1, Op2(x, z))]"

    def test_constant_merging(self):
        x = MyResult('x')
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
        x = MyResult('x')
        y = Constant(MyType(), 2, name = 'y')
        z = Constant(MyType(), 2, name = 'z')
        e1 = op1(y, z)
        g = Env([x, y, z], [e1])
        MergeOptimizer().optimize(g)
        strg = str(g)
        self.failUnless(strg == '[Op1(y, y)]' or strg == '[Op1(z, z)]', strg)

#     def test_identical_constant_args_with_destroymap(self):
#         x, y, z = inputs()
#         y.data = 2.0
#         y.constant = False
#         z.data = 2.0
#         z.constant = True
#         e1 = op_d(y, z)
#         g = env([x, y, z], [e1])
#         MergeOptimizer().optimize(g)
#         strg = str(g)
#         self.failUnless(strg == '[OpD(y, z)]', strg)

#     def test_merge_with_destroyer_1(self):
#         x, y, z = inputs()
#         e1 = op_d(op1(x,y), y)
#         e2 = op_d(op1(x,y), z)
#         g = env([x, y, z], [e1,e2])
#         MergeOptimizer().optimize(g)
#         strg = str(g)
#         self.failUnless(strg == '[OpD(Op1(x, y), y), OpD(Op1(x, y), z)]', strg)

#     def test_merge_with_destroyer_2(self):
#         x, y, z = inputs()
#         e1 = op_d(op1(x,y), z)
#         e2 = op_d(op1(x,y), z)
#         g = env([x, y, z], [e1,e2])
#         MergeOptimizer().optimize(g)
#         strg = str(g)
#         self.failUnless(strg == '[*1 -> OpD(Op1(x, y), z), *1]', strg)



# class _test_ConstantFinder(unittest.TestCase):

#     def test_straightforward(self):
#         x, y, z = inputs()
#         y.data = 2
#         z.data = 2
#         e = op1(x, y, z)
#         g = env([x], [e])
#         ConstantFinder().optimize(g)
#         assert y.constant and z.constant
#         MergeOptimizer().optimize(g)
#         assert str(g) == "[Op1(x, y, y)]" \
#             or str(g) == "[Op1(x, z, z)]"

#     def test_deep(self):
#         x, y, z = inputs()
#         y.data = 2
#         z.data = 2
#         e = op1(op2(x, y), op2(x, y), op2(x, z))
#         g = env([x], [e])
#         ConstantFinder().optimize(g)
#         assert y.constant and z.constant
#         MergeOptimizer().optimize(g)
#         assert str(g) == "[Op1(*1 -> Op2(x, y), *1, *1)]" \
#             or str(g) == "[Op1(*1 -> Op2(x, z), *1, *1)]"

#     def test_destroyed_orphan_not_constant(self):
#         x, y, z = inputs()
#         y.data = 2
#         z.data = 2
#         e = op_d(x, op2(y, z)) # here x is destroyed by op_d
#         g = env([y], [e])
#         ConstantFinder().optimize(g)
#         assert not getattr(x, 'constant', False) and z.constant
#         MergeOptimizer().optimize(g)



class _test_ExpandMacro(unittest.TestCase):

    def test_straightforward(self):
        class Macro1(Macro):
            def make_node(self, x, y):
                return Apply(self, [x, y], [MyType()()])
            def expand(self, node):
                return [op1(y, x)]
        x, y, z = inputs()
        e = Macro1()(x, y)
        g = Env([x, y], [e])
        print g
        expand_macros.optimize(g)
        print g
        
    def test_loopy(self):
        class Macro1(Macro):
            def make_node(self, x, y):
                return Apply(self, [x, y], [MyType()()])
            def expand(self, node):
                return [Macro1()(y, x)]
        x, y, z = inputs()
        e = Macro1()(x, y)
        g = Env([x, y], [e])
        print g
        #expand_macros.optimize(g)
        TopDownOptimizer(ExpandMacro(), ignore_newtrees = True).optimize(g)
        print g
        



if __name__ == '__main__':
    unittest.main()



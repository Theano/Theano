
import unittest

from result import ResultBase
from op import Op
from ext import Destroyer
from opt import *
from env import Env
from toolbox import *


class MyResult(ResultBase):

    def __init__(self, name):
        ResultBase.__init__(self, role = None, name = name)
        self.data = [1000]

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def hash(self):
        return self.data



class MyOp(Op):
        
    def __init__(self, *inputs):
        for input in inputs:
            if not isinstance(input, MyResult):
                raise Exception("Error 1")
        self.inputs = inputs
        self.outputs = [MyResult(self.__class__.__name__ + "_R")]


class Op1(MyOp):
    pass

class Op2(MyOp):
    pass

class Op3(MyOp):
    pass

class Op4(MyOp):
    pass

class OpD(MyOp, Destroyer):
    def destroyed_inputs(self):
        return [self.inputs[0]]


import modes
modes.make_constructors(globals())


def inputs():
    x = modes.build(MyResult('x'))
    y = modes.build(MyResult('y'))
    z = modes.build(MyResult('z'))
    return x, y, z

def env(inputs, outputs, validate = True):
    return Env(inputs, outputs, features = [EquivTool], consistency_check = validate)


class _test_PatternOptimizer(unittest.TestCase):
    
    def test_replace_output(self):
        # replacing the whole graph
        x, y, z = inputs()
        e = op1(op2(x, y), z)
        g = env([x, y, z], [e])
        PatternOptimizer((Op1, (Op2, '1', '2'), '3'),
                         (Op4, '3', '2')).optimize(g)
        assert str(g) == "[Op4(z, y)]"
    
    def test_nested_out_pattern(self):
        x, y, z = inputs()
        e = op1(x, y)
        g = env([x, y, z], [e])
        PatternOptimizer((Op1, '1', '2'),
                         (Op4, (Op1, '1'), (Op2, '2'), (Op3, '1', '2'))).optimize(g)
        assert str(g) == "[Op4(Op1(x), Op2(y), Op3(x, y))]"

    def test_unification_1(self):
        x, y, z = inputs()
        e = op1(op2(x, x), z) # the arguments to op2 are the same
        g = env([x, y, z], [e])
        PatternOptimizer((Op1, (Op2, '1', '1'), '2'), # they are the same in the pattern
                         (Op4, '2', '1')).optimize(g)
        # So the replacement should occur
        assert str(g) == "[Op4(z, x)]"

    def test_unification_2(self):
        x, y, z = inputs()
        e = op1(op2(x, y), z) # the arguments to op2 are different
        g = env([x, y, z], [e])
        PatternOptimizer((Op1, (Op2, '1', '1'), '2'), # they are the same in the pattern
                         (Op4, '2', '1')).optimize(g)
        # The replacement should NOT occur
        assert str(g) == "[Op1(Op2(x, y), z)]"

    def test_replace_subgraph(self):
        # replacing inside the graph
        x, y, z = inputs()
        e = op1(op2(x, y), z)
        g = env([x, y, z], [e])
        PatternOptimizer((Op2, '1', '2'),
                         (Op1, '2', '1')).optimize(g)
        assert str(g) == "[Op1(Op1(y, x), z)]"

    def test_no_recurse(self):
        # if the out pattern is an acceptable in pattern,
        # it should do the replacement and stop
        x, y, z = inputs()
        e = op1(op2(x, y), z)
        g = env([x, y, z], [e])
        PatternOptimizer((Op2, '1', '2'),
                         (Op2, '2', '1')).optimize(g)
        assert str(g) == "[Op1(Op2(y, x), z)]"

    def test_multiple(self):
        # it should replace all occurrences of the pattern
        x, y, z = inputs()
        e = op1(op2(x, y), op2(x, y), op2(y, z))
        g = env([x, y, z], [e])
        PatternOptimizer((Op2, '1', '2'),
                         (Op4, '1')).optimize(g)
        assert str(g) == "[Op1(Op4(x), Op4(x), Op4(y))]"

    def test_nested_even(self):
        # regardless of the order in which we optimize, this
        # should work
        x, y, z = inputs()
        e = op1(op1(op1(op1(x))))
        g = env([x, y, z], [e])
        PatternOptimizer((Op1, (Op1, '1')),
                         '1').optimize(g)
        assert str(g) == "[x]"

    def test_nested_odd(self):
        x, y, z = inputs()
        e = op1(op1(op1(op1(op1(x)))))
        g = env([x, y, z], [e])
        PatternOptimizer((Op1, (Op1, '1')),
                         '1').optimize(g)
        assert str(g) == "[Op1(x)]"

    def test_expand(self):
        x, y, z = inputs()
        e = op1(op1(op1(x)))
        g = env([x, y, z], [e])
        PatternOptimizer((Op1, '1'),
                         (Op2, (Op1, '1'))).optimize(g)
        assert str(g) == "[Op2(Op1(Op2(Op1(Op2(Op1(x))))))]"

    def test_ambiguous(self):
        # this test is known to fail most of the time
        # the reason is that PatternOptimizer doesn't go through
        # the ops in topological order. The order is random and
        # it does not visit ops that it creates.
        x, y, z = inputs()
        e = op1(op1(op1(op1(op1(x)))))
        g = env([x, y, z], [e])
        PatternOptimizer((Op1, (Op1, '1')),
                         (Op1, '1')).optimize(g)
        assert str(g) == "[Op1(x)]"

    def test_constant_unification(self):
        x, y, z = inputs()
        x.constant = True
        x.value = 2
        z.constant = True
        z.value = 2
        e = op1(op1(x, y), y)
        g = env([y], [e])
        PatternOptimizer((Op1, z, '1'),
                         (Op2, '1', z)).optimize(g)
        assert str(g) == "[Op1(Op2(y, z), y)]"

    def test_constraints(self):
        x, y, z = inputs()
        e = op4(op1(op2(x, y)), op1(op1(x, y)))
        g = env([x, y, z], [e])
        def constraint(env, r):
            # Only replacing if the input is an instance of Op2
            return isinstance(r.owner, Op2)
        PatternOptimizer((Op1, {'pattern': '1',
                                'constraint': constraint}),
                         (Op3, '1')).optimize(g)
        assert str(g) == "[Op4(Op3(Op2(x, y)), Op1(Op1(x, y)))]"
        


class _test_OpSubOptimizer(unittest.TestCase):
    
    def test_straightforward(self):
        x, y, z = inputs()
        e = op1(op1(op1(op1(op1(x)))))
        g = env([x, y, z], [e])
        OpSubOptimizer(Op1, Op2).optimize(g)
        assert str(g) == "[Op2(Op2(Op2(Op2(Op2(x)))))]"
    
    def test_straightforward_2(self):
        x, y, z = inputs()
        e = op1(op2(x), op3(y), op4(z))
        g = env([x, y, z], [e])
        OpSubOptimizer(Op3, Op4).optimize(g)
        assert str(g) == "[Op1(Op2(x), Op4(y), Op4(z))]"


class _test_MergeOptimizer(unittest.TestCase):

    def test_straightforward(self):
        x, y, z = inputs()
        e = op1(op2(x, y), op2(x, y), op2(x, z))
        g = env([x, y, z], [e])
        MergeOptimizer().optimize(g)
        assert str(g) == "[Op1(*1 -> Op2(x, y), *1, Op2(x, z))]"

    def test_constant_merging(self):
        x, y, z = inputs()
        y.data = 2
        y.constant = True
        z.data = 2.0
        z.constant = True
        e = op1(op2(x, y), op2(x, y), op2(x, z))
        g = env([x, y, z], [e])
        MergeOptimizer().optimize(g)
        strg = str(g)
        assert strg == "[Op1(*1 -> Op2(x, y), *1, *1)]" \
            or strg == "[Op1(*1 -> Op2(x, z), *1, *1)]"

    def test_deep_merge(self):
        x, y, z = inputs()
        e = op1(op3(op2(x, y), z), op4(op3(op2(x, y), z)))
        g = env([x, y, z], [e])
        MergeOptimizer().optimize(g)
        assert str(g) == "[Op1(*1 -> Op3(Op2(x, y), z), Op4(*1))]"

    def test_no_merge(self):
        x, y, z = inputs()
        e = op1(op3(op2(x, y)), op3(op2(y, x)))
        g = env([x, y, z], [e])
        MergeOptimizer().optimize(g)
        assert str(g) == "[Op1(Op3(Op2(x, y)), Op3(Op2(y, x)))]"

    def test_merge_outputs(self):
        x, y, z = inputs()
        e1 = op3(op2(x, y))
        e2 = op3(op2(x, y))
        g = env([x, y, z], [e1, e2])
        MergeOptimizer().optimize(g)
        assert str(g) == "[*1 -> Op3(Op2(x, y)), *1]"

    def test_multiple_merges(self):
        x, y, z = inputs()
        e1 = op1(x, y)
        e2 = op2(op3(x), y, z)
        e = op1(e1, op4(e2, e1), op1(e2))
        g = env([x, y, z], [e])
        MergeOptimizer().optimize(g)
        strg = str(g)
        # note: graph.as_string can only produce the following two possibilities, but if
        # the implementation was to change there are 6 other acceptable answers.
        assert strg == "[Op1(*1 -> Op1(x, y), Op4(*2 -> Op2(Op3(x), y, z), *1), Op1(*2))]" \
            or strg == "[Op1(*2 -> Op1(x, y), Op4(*1 -> Op2(Op3(x), y, z), *2), Op1(*1))]"


class _test_ConstantFinder(unittest.TestCase):

    def test_straightforward(self):
        x, y, z = inputs()
        y.data = 2
        z.data = 2
        e = op1(x, y, z)
        g = env([x], [e])
        ConstantFinder().optimize(g)
        assert y.constant and z.constant
        MergeOptimizer().optimize(g)
        assert str(g) == "[Op1(x, y, y)]" \
            or str(g) == "[Op1(x, z, z)]"

    def test_deep(self):
        x, y, z = inputs()
        y.data = 2
        z.data = 2
        e = op1(op2(x, y), op2(x, y), op2(x, z))
        g = env([x], [e])
        ConstantFinder().optimize(g)
        assert y.constant and z.constant
        MergeOptimizer().optimize(g)
        assert str(g) == "[Op1(*1 -> Op2(x, y), *1, *1)]" \
            or str(g) == "[Op1(*1 -> Op2(x, z), *1, *1)]"

    def test_destroyed_orphan_not_constant(self):
        x, y, z = inputs()
        y.data = 2
        z.data = 2
        e = op_d(x, op2(y, z)) # here x is destroyed by op_d
        g = env([y], [e])
        ConstantFinder().optimize(g)
        assert not getattr(x, 'constant', False) and z.constant
        MergeOptimizer().optimize(g)



if __name__ == '__main__':
    unittest.main()





from core import *
import gof


# def pattern_opt(in_pattern, out_pattern):
#     def parse(x):
#         if isinstance(x, (list, tuple)):
#             return [parse(y) for y in x]
#         elif isinstance(x, wrapper):
#             return x.opclass
#         elif isinstance(x, str) or (hasattr(x, '__bases__') and issubclass(x, gof.op.Op)):
#             return x
#         else:
#             raise TypeError("Bad input type for pattern_opt.")
#     return gof.opt.PatternOptimizer(parse(in_pattern), parse(out_pattern))

# def op_sub(op1, op2):
#     if isinstance(op1, wrapper):
#         op1 = op1.opclass
#     if isinstance(op2, wrapper):
#         op2 = op2.opclass
#     return gof.opt.OpSubOptimizer(op1, op2)


pattern_opt = gof.PatternOptimizer
op_sub = gof.OpSubOptimizer


def export_opts(opts):
    for name, opt in opts:
        if name:
            globals()[name] = opt


# List of optimizations to perform. They are listed in the order they are applied.
opts = [

    ['double_transpose_eliminator', pattern_opt((transpose, (transpose, 'x')),
                                                'x')],

    ['addxx_to_twice',              pattern_opt((add, 'x', 'x'),
                                                (twice, 'x'))],

    ['twice_to_itwice',             op_sub(twice, itwice)],

    ['mulxx_to_sqr',                pattern_opt((mul, 'x', 'x'),
                                                (sqr, 'x'))],

    ['sqr_to_isqr',                 op_sub(sqr, isqr)],
    
    ['add_to_iadd',                 op_sub(add, iadd)],

    ['add_to_iadd_reverse',         pattern_opt((add, 'x', 'y'),
                                                (iadd, 'y', 'x'))],

    ['remove_copies',               gof.OpRemover(array_copy)],
    
    [None,                          gof.DummyRemover] # has to be at the end
    
    ]


export_opts(opts) # publish the optimizations performed under individual names


optimizer = gof.PythonOpt(gof.MergeOptMerge(gof.SeqOptimizer([opt for name, opt in opts])))
#optimizer = gof.PythonOpt(gof.SeqOptimizer([opt for name, opt in opts]))



#[isub(1.0, mul(0.1, iadd(transpose(dot(transpose(*2 -> sigmoid(dot(0.0, 1.0))), *4 -> mul(mul(neg(scal(mul(*3 -> sub(0.0, *1 -> sigmoid(dot(*2, transpose(1.0)))), fill(isqr(*3), 1.0)), 2.0)), *1), sub(1, *1)))), dot(transpose(0.0), mul(mul(dot(*4, 1.0), *2), sub(1, *2))))))]

#[isub(1.0, mul(0.1, iadd(dot(transpose(0.0), mul(mul(dot(*4 -> mul(mul(neg(scal(mul(*1 -> sub(0.0, *2 -> sigmoid(dot(*3 -> sigmoid(dot(0.0, 1.0)), transpose(1.0)))), fill(sqr(*1), 1.0)), 2.0)), *2), sub(1, *2)), 1.0), *3), sub(1, *3))), transpose(dot(transpose(*3), *4)))))]

#[isub(1.0, mul(0.1, iadd(dot(transpose(0.0), mul(mul(dot(*2 -> mul(mul(neg(scal(mul(*4 -> sub(0.0, *1 -> sigmoid(dot(*3 -> sigmoid(dot(0.0, 1.0)), transpose(1.0)))), fill(sqr(*4), 1.0)), 2.0)), *1), sub(1, *1)), 1.0), *3), sub(1, *3))), transpose(dot(transpose(*3), *2)))))]


# [ sqr(sub(0.0, sigmoid(dot(sigmoid(dot(0.0, 1.0)), transpose(1.0)))))]
# [isqr(sub(0.0, sigmoid(dot(sigmoid(dot(0.0, 1.0)), transpose(1.0)))))]

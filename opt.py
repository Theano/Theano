
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


pattern_opt = gof.opt.PatternOptimizer
op_sub = gof.opt.OpSubOptimizer


#def make_patterns(patterns):
#    return [name, pattern_opt(inp, outp) for name, inp, outp in patterns]

def export_opts(opts):
    for name, opt in opts:
        if name:
            globals()[name] = opt



# double_transpose_eliminator = pattern_opt((transpose, (transpose, 'x')), 'x')
# patterns = make_patterns(patterns)
# export_patterns(patterns)

# List of optimizations to perform. They are listed in the order they are applied.
opts = [

    ['double_transpose_eliminator', pattern_opt((transpose, (transpose, 'x')),
                                                'x')],

    ['addxx_to_twice',              pattern_opt((add, 'x', 'x'),
                                                (twice, 'x'))],

    ['twice_to_itwice',             op_sub(twice, itwice)],

    ['mulxx_to_twice',              pattern_opt((mul, 'x', 'x'),
                                                (sqr, 'x'))],

    ['sqr_to_isqr',                 op_sub(sqr, isqr)],
    
    ['add_to_iadd',                 op_sub(add, iadd)],

    ['add_to_iadd_reverse',         pattern_opt((add, 'x', 'y'),
                                                (iadd, 'y', 'x'))],

    ['remove_copies',               gof.opt.OpRemover(array_copy)],
    
    [None,                          gof.lib.DummyRemover] # has to be at the end
    
    ]

export_opts(opts) # publish the optimizations performed under individual names


# class AAA(gof.opt.Optimizer):

#     def __init__(self, opt):
#         self.opt = opt
    
#     def optimize(self, env):
#         build_mode()
#         self.opt.optimize(env)
#         pop_mode()


optimizer = gof.lib.PythonOpt(gof.opt.MergeOptMerge(gof.opt.SeqOptimizer([opt for name, opt in opts])))

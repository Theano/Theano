
from core import *
import gof

from gof import PatternOptimizer as pattern_opt, OpSubOptimizer as op_sub

"""
This variable is used in compile.prog as the optimizer for all programs built
using either compile.single, compile.to_func, and compile.prog.

"""

optimizer_begin = gof.SeqOptimizer([])
         #gof.SeqOptimizer([ opt for name, opt in [ ]])))
         #['double_transpose_eliminator', pattern_opt((transpose, (transpose, 'x')), 'x')],

         #['addxx_to_twice',              pattern_opt((add_elemwise, 'x', 'x'), (twice, 'x'))],

         #['twice_to_itwice',             op_sub(twice, itwice)],

         #['mulxx_to_sqr',                pattern_opt((mul_elemwise, 'x', 'x'), (sqr, 'x'))],

         #['sqr_to_isqr',                 op_sub(sqr, isqr)],
        
         #['add_to_iadd',                 op_sub(add_elemwise, iadd_elemwise)],

         #['add_to_iadd_reverse',         pattern_opt((add_elemwise, 'x', 'y'), (iadd_elemwise, 'y', 'x'))],

optimizer_end = gof.SeqOptimizer([gof.DummyRemover])
#         ['remove_copies',               gof.OpRemover(array_copy)],
#         [None,                          gof.DummyRemover] # has to be at the end

        
def optimizer(lst):
    seq_opt = gof.SeqOptimizer(optimizer_begin + lst + optimizer_end)
    rval = gof.PythonOpt(gof.MergeOptMerge(seq_opt))
    return rval
    

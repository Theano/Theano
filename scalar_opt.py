
from scalar import *
from gof import PatternOptimizer

c2 = constant(2.0)

opt1 = PatternOptimizer((Mul, 'x', 'x'), (Sqr, 'x'))
opt2 = PatternOptimizer((Pow, 'x', c2), (Sqr, 'x'))







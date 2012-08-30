import theano
import theano.tensor as T

M = theano.Module()
M.a, M.b, M.c = [T.dvector() for i in 1,2,3]

P = theano.Module()
P.m = M   # include a module by nesting
x = T.dvector()
P.f = theano.Method([x], None, {M.b: M.b + x})

p = P.make()  # this converts both M and P because M was nested within P
p.m.b = [4, 5, 6]
p.f(3)
print p.m.b
#  prints  array([7.,8.,9.])

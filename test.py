

# import gof


# gof.stealth.method_wrap(int, '__add__', [2, 1], )

# x = gof.stealth.wrap(3)
# y = gof.stealth.wrap(4)

# print x + y

import gof
import core
import numpy
import compile
import grad

# a = core.NumpyR(numpy.ones((3, 3)))
# b = core.NumpyR(numpy.ones((3, 3)))

# w = core.dot #core.wrapper(numpy.dot)

# core.start_build()
# r = a * (b * b)
# core.end_build()

# #r = w(a, w(b, b))

# print r
# print r.owner

# env = gof.Env([a, b], [r._obj])
# print env

# print r
# gof.ThunkLinker()(env)()
# print r


# core.start_build()
# a += b + c
# a = a + b
# a += a + core.transpose(b)
# core.end_build()


# # env = gof.Env(gof.graph.inputs([a]), [a])
# # print env

# # gof.ThunkLinker()(env)()

# # print a

# print gof.Env(gof.graph.inputs([a]), [a])
# prog = compile.single(a)
# print prog.env

# prog()

# print a


############################

# core.build_mode()
# dim = core.wrap(())
# dim2 = core.wrap((2, 2))
# a = core.zeros(dim, dtype='int32') #(core.NumpyR(numpy.ones((3, 3))))
# b = core.ones(dim2, 'int32') #(core.NumpyR(numpy.ones((3, 3))))
# c = core.zeros(dim, dtype='int32')
# d = a + (b + b) + c + numpy.ones(())
# e = d + (b * c)
# core.pop_mode()


# #print e
# #print gof.graph.ops([dim], [e])
# #1/0

# #print gof.Env([dim], [e])

# #f = compile.to_func([dim], [e])
# f = compile.to_func([a, b, c], [e])

# print f(1, 2, 3)
# #print f((2,2))


############################

# a = core.ones((2, 2))
# b = core.ones((2, 2))

# def f():
#     return (a + b) + (a + b)

# r = core.build(f)

# env = gof.Env([a, b], [r])
# print env
# gof.opt.MergeOptimizer().optimize(env)
# print env


# print compile.to_func([a, b], [r])(1, 2)


############################

a = core.ones((2, 2))
b = core.ones((2, 2))

def f():
    return (a + b) + (a + b)

r = core.build(f)

g = grad.grad(r, a)

core.print_graph(g)
core.print_graph(r)






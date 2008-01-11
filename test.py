

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

# #core.build_mode()
# dim = core.wrap(())
# dim2 = core.wrap((2, 2))
# a = core.zeros(dim, dtype='int32') #(core.NumpyR(numpy.ones((3, 3))))
# b = core.ones(dim2, 'int32') #(core.NumpyR(numpy.ones((3, 3))))
# c = core.zeros(dim, dtype='int32')

# d = a + (b + b) + c + numpy.ones(())
# e = d + (b * c)

# #core.pop_mode()

# print e

# #print e
# #print gof.graph.ops([dim], [e])
# #1/0

# #print gof.Env([dim], [e])

# #f = compile.to_func([dim], [e])

# # f = compile.to_func([a, b, c], [e])

# # print f(1, 2, 3)
# # #print f((2,2))


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

# a = core.ones((2, 2))
# b = core.ones((2, 2))

# def f():
#     return (a + b) + (a + b)

# r = core.build(f)

# g = grad.grad(r, a)

# core.print_graph(g)
# print [id(input) for input in g.owner.inputs]
# print gof.literals_db

# core.print_graph(r)


############################

# def dataset_1hot(x, targ, n):
#     """Return an looping iterator over 1-hot vectors

#     This function is a generator for the integers range(n) that works by
#     side-effect on the numpy ndarray mat.
#     On each iteration, mat is set (in-place) to the next element of an infinite
#     sequence of 1-hot vectors.

#     """
#     assert targ.size == 1

#     for i in xrange(n):
#         idx = i % x.shape[1]
#         x[:] = 0
#         x[0,idx] = 1
#         targ[0] = idx
#         yield i


# class sigmoid(core.omega_op):
#     def impl(x):
#         return 1.0 / (1.0 + numpy.exp(-x))
#     def grad(x, gz):
#         return gz * sigmoid(x) * (1 - sigmoid(x))


# x = core.zeros((1, 10))
# w = core.input(numpy.random.rand(10, 15))

# #print x.data, w.data

# def autoassociator(w, x):
#     forward = sigmoid(core.dot(sigmoid(core.dot(x, w)), w.T))
#     rec_error = core.sum(core.sqr(x - forward))
#     w -= 0.1 * grad.grad(rec_error, w)
#     return w, rec_error

# w2, rec_error = core.build(autoassociator, w, x)
# f = compile.to_func([w, x], [w2, rec_error])

# for i in dataset_1hot(x.data, numpy.ndarray((1, )), 10000):
#     w2, rec_error = f(w.data, x.data)
#     if not(i % 1000):
#         print rec_error

# print "done!"
# print w.data



# # 1 = mul(mul(neg(scal(mul(sub(0.736213102665, sigmoid(*3)), 1.0), 2.0)), sigmoid(*3)), sub(1, sigmoid(*3)))
# # 2 = transpose(0.11474051836)
# # 3 = dot(*2, *5)
# # 4 = dot(0.11474051836, 0.736213102665)
# # 5 = sigmoid(*4)
# # add(transpose(dot(*1, transpose(*5))), dot(mul(mul(dot(transpose(*2), *1), sigmoid(*4)), sub(1, sigmoid(*4))), transpose(0.736213102665)))





############################

# def fun():
#     a = core.NumpyR(numpy.zeros(()) + 200)
# #    b = numpy.ones(())
# #    a = a * core.sqrt(core.isqr(a))
#     a = a * core.isqr(a)
#     return a

# f = core.build(fun)

# g = compile.to_func(gof.graph.inputs([f]), [f])



############################

print core.ones((2, 2)) + 1

print numpy.ones((2, 2)) ** numpy.ones((2, 2))



import unittest
import gof, gof.opt

from theano import compile
from theano.compile.function_module import *
from theano.scalar import *

from theano import tensor
from theano import tensor as T
import random
import numpy as N


PatternOptimizer = lambda p1, p2, ign=True: gof.OpKeyOptimizer(gof.PatternSub(p1, p2), ignore_newtrees=ign)

def checkfor(testcase, fn, E):
    try:
        fn()
    except Exception, e:
        if isinstance(e, E):
            # we got the exception we wanted
            return
        else:
            # we did not get the exception we wanted
            raise
    # fn worked, but it shouldn't have
    testcase.fail()


# def graph1(): # (x+y) * (x/z)
#     x, y, z = floats('xyz')
#     o = mul(add(x, y), div(x, z))
#     return [x,y,z], [o]


# class T_Function(unittest.TestCase):
    
#     def test_noopt(self):
#         gi, go = graph1()
#         p = function(gi, go, optimizer = None, linker = 'py')
#         self.failUnless(p(1.0,3.0,4.0) == 1.0)

#     def test_opt(self):
#         opt = PatternOptimizer((div, '1', '2'), (div, '2', '1'))
#         gi, go = graph1()
#         p = function(gi,go, optimizer=opt.optimize, linker = 'py')
#         self.failUnless(p(1.,3.,4.) == 16.0)

#     def test_multiout(self):
#         def graph2():
#             x, y, z = floats('xyz')
#             o = mul(add(x, y), div(x, z))
#             return [x,y,z], [o, o.owner.inputs[1]]
#         opt = PatternOptimizer((div, '1', '2'), (div, '2', '1'))
#         gi, go = graph2()
#         p = function(gi,go, optimizer=opt.optimize)
#         a,b = p(1.,3.,4.)
#         self.failUnless(a == 16.0)
#         self.failUnless(b == 4.0)

#     def test_make_many_functions(self):
#         x, y, z = tensor.scalars('xyz')
#         e0, e1, e2 = x+y+z, x*y-z, z*z+x*x+y*y
#         f1 = function([x, y, z], [e0])
#         f2 = function([x, y, z], [e0])
#         f3 = function([x, y, z], [e1])
#         f4 = function([x, y, z], [e2])
#         f5 = function([e0], [e0 * e0])
#         ff = FunctionFactory([x, y, z], [e0])
#         f6 = ff.create()
#         f7 = ff.create()
#         f8 = ff.create()
#         f9 = ff.partial(1.0, 2.0)
#         assert f1(1.0, 2.0, 3.0) == 6.0
#         assert f2(1.0, 2.0, 3.0) == 6.0
#         assert f3(1.0, 2.0, 3.0) == -1.0
#         assert f4(1.0, 2.0, 3.0) == 14.0
#         assert f5(7.0) == 49.0
#         assert f6(1.0, 2.0, 3.0) == 6.0
#         assert f7(1.0, 2.0, 3.0) == 6.0
#         assert f8(1.0, 2.0, 3.0) == 6.0
#         assert f9(3.0) == 6.0

#     def test_no_inputs(self):
#         x, y, z = tensor.value(1.0), tensor.value(2.0), tensor.value(3.0)
#         e = x*x + y*y + z*z
#         assert function([], [e], linker = 'py')() == 14.0
#         assert function([], [e], linker = 'c')() == 14.0
#         assert function([], [e], linker = 'c|py')() == 14.0
#         assert function([], [e], linker = 'c&py')() == 14.0
#         assert eval_outputs([e]) == 14.0
#         assert fast_compute(e) == 14.0

#     def test_closure(self):
#         x, y, z = tensor.scalars('xyz')
#         v = tensor.value(numpy.zeros(()))
#         e = x + tensor.add_inplace(v, 1)
#         f = function([x], [e])
#         assert f(1.) == 2.
#         assert f(1.) == 3.
#         assert f(1.) == 4.

#     def test_borrow_true(self):
#         x, y, z = tensor.scalars('xyz')
#         e = x + y + z
#         f = function([x, y, z], [e], borrow_outputs = True)
#         res1 = f(1.0, 2.0, 3.0)
#         assert res1 == 6.0
#         res2 = f(1.0, 3.0, 5.0)
#         assert res1 is res2
#         assert res1 == 9.0
#         assert res2 == 9.0

#     def test_borrow_false(self):
#         x, y, z = tensor.scalars('xyz')
#         e = x + y + z
#         for linker in 'py c c|py c&py'.split():
#             f = function([x, y, z], [e], borrow_outputs = False, linker = linker)
#             res1 = f(1.0, 2.0, 3.0)
#             self.failUnless(res1 == 6.0, (res1, linker))
#             res2 = f(1.0, 3.0, 5.0)
#             self.failUnless(res1 is not res2, (res1, res2, linker))
#             self.failUnless(res1 == 6.0, (res1, linker))
#             self.failUnless(res2 == 9.0, (res2, linker))

#     def test_borrow_false_through_inplace(self):
#         x, y, z = tensor.scalars('xyz')
#         # if borrow_outputs is False, we must not reuse the temporary created for x+y
#         e = tensor.add_inplace(x + y, z)
#         for linker in 'py c c|py c&py'.split():
#             f = function([x, y, z], [e], borrow_outputs = False, linker = linker)
#             res1 = f(1.0, 2.0, 3.0)
#             self.failUnless(res1 == 6.0, (res1, linker))
#             res2 = f(1.0, 3.0, 5.0)
#             self.failUnless(res1 is not res2, (res1, res2, linker))
#             self.failUnless(res1 == 6.0, (res1, linker))
#             self.failUnless(res2 == 9.0, (res2, linker))


# class T_fast_compute(unittest.TestCase):

#     def test_straightforward(self):
#         x, y, z = tensor.value(1.0), tensor.value(2.0), tensor.value(3.0)
#         e = x*x + y*y + z*z
#         assert fast_compute(e) == 14.0
#         assert compile._fcache[(e, )]() == 14.0



class T_OpFromGraph(unittest.TestCase):

    def test_straightforward(self):
        x, y, z = T.matrices('xyz')
        e = x + y * z
        op = OpFromGraph([x, y, z], [e], mode='FAST_RUN')
        f = op(x, y, z) - op(y, z, x)
        fn = function([x, y, z], f)
        xv, yv, zv = N.ones((2, 2)), N.ones((2, 2))*3, N.ones((2, 2))*5
        assert numpy.all(8.0 == fn(xv, yv, zv))
        assert numpy.all(8.0 == fn(xv, yv, zv))
    
    def test_size_changes(self):
        x, y, z = T.matrices('xyz')
        e = T.dot(x, y)
        op = OpFromGraph([x, y], [e], mode='FAST_RUN')
        f = op(x, op(y, z))
        fn = function([x, y, z], f)
        xv, yv, zv = N.ones((2, 3)), N.ones((3, 4))*3, N.ones((4, 5))*5
        res = fn(xv, yv, zv)
        assert res.shape == (2, 5)
        assert numpy.all(180.0 == res)
        res = fn(xv, yv, zv)
        assert res.shape == (2, 5)
        assert numpy.all(180.0 == res)
    
    def test_grad(self):
        x, y, z = T.matrices('xyz')
        e = x + y * z
        op = OpFromGraph([x, y, z], [e], mode='FAST_RUN', grad_depth = 2)
        f = op(x, y, z)
        f = f - T.grad(f, y)
        fn = function([x, y, z], f)
        xv, yv, zv = N.ones((2, 2)), N.ones((2, 2))*3, N.ones((2, 2))*5
        assert numpy.all(11.0 == fn(xv, yv, zv))


class T_function(unittest.TestCase):
    def test_empty(self):
        fn = function([], []) #ok
        self.failUnless(fn() == [])

    def test_missing_inputs(self):

        MissingInputException = TypeError

        def fn():
            x,s = T.scalars('xs')
            fn = function([], [x])
        checkfor(self, fn, MissingInputException)

        def fn():
            x,s = T.scalars('xs')
            fn = function([s], [x])
        checkfor(self, fn, MissingInputException)

        def fn():
            x,s = T.scalars('xs')
            fn = function([s], x)
        checkfor(self, fn, MissingInputException)

        def fn():
            x,s = T.scalars('xs')
            fn = function([s], Out(x))
        checkfor(self, fn, MissingInputException)

        def fn():
            x,s = T.scalars('xs')
            fn = function([In(x, update=s+x)], x)
        checkfor(self, fn, MissingInputException)

        def fn():
            x,s = T.scalars('xs')
            fn = function([In(x, update=mul(s,s)+x)], x)
        checkfor(self, fn, MissingInputException)

    def test_input_anon_singleton(self):
        x,s = T.scalars('xs')
        fn = function([s,x], [x+s])
        self.failUnless(fn(2,3) == [5])
        # no state
        self.failUnless(fn(2,3) == [5])

    def test_input_anon_unpack(self):
        x,s = T.scalars('xs')
        fn = function([s,x], x+s)
        self.failUnless(fn(2,3) == 5)

    def test_naming_rule0(self):
        x,s = T.scalars('xs')
        f = function([x,s], x/s)
        self.failUnless(f(1,2) == 0.5)
        self.failUnless(f(2,1) == 2.0)
        self.failUnless(f(s=2,x=1) == 0.5)
        self.failUnless(f(x=2,s=1) == 2.0)
        self.failUnless(f(2, s=1) == 2.0)
        checkfor(self, lambda :f(2, x=2.0), TypeError) #got multiple values for keyword argument 'x'
        checkfor(self, lambda :f(x=1), TypeError) #takes exactly 2 non-keyword arguments (1 given)
        checkfor(self, lambda :f(s=1), TypeError) #takes exactly 2 non-keyword arguments (0 given)

    def test_naming_rule1(self):
        a = T.scalar() # the a is for 'anonymous' (un-named).
        x,s = T.scalars('xs')
        f = function([a, s], a/s)
        self.failUnless(f(1,2) == 0.5)
        self.failUnless(f(2,1) == 2.0)
        self.failUnless(f(2, s=1) == 2.0)
        checkfor(self, lambda:f(q=2,s=1), TypeError) #got unexpected keyword argument 'q'
        checkfor(self, lambda:f(a=2,s=1), TypeError) #got unexpected keyword argument 'a'

    def test_naming_rule2(self):
        a = T.scalar() # the a is for 'anonymous' (un-named).
        x,s = T.scalars('xs')

        #x's name is ignored because it is followed by anonymous parameter a.
        f = function([x, a, s], a/s)
        self.failUnless(f(9,1,2) == 0.5)
        self.failUnless(f(9,2,1) == 2.0)
        self.failUnless(f(9,2, s=1) == 2.0)
        checkfor(self, lambda:f(x=9,a=2,s=1), TypeError) #got unexpected keyword argument 'x'
        checkfor(self, lambda:f(5.0,x=9), TypeError) #got unexpected keyword argument 'x'

    def test_naming_rule3(self):
        a = T.scalar() # the a is for 'anonymous' (un-named).
        x,s = T.scalars('xs')

        #x's name is not ignored (as in test_naming_rule2) because a has a default value.
        f = function([x, In(a, value=1.0), s], a/s+x)
        self.failUnless(f(9,2,4) == 9.5) #can specify all args in order
        self.failUnless(f(9,2,s=4) == 9.5) # can give s as kwarg
        self.failUnless(f(9,s=4) == 9.25) # can give s as kwarg, get default a
        self.failUnless(f(x=9,s=4) == 9.25) # can give s as kwarg, omit a, x as kw
        checkfor(self, lambda:f(x=9,a=2,s=4), TypeError) #got unexpected keyword argument 'a'
        checkfor(self, lambda:f(), TypeError) #takes exactly 3 non-keyword arguments (0 given)
        checkfor(self, lambda:f(x=9), TypeError) #takes exactly 3 non-keyword arguments (1 given)

    def test_naming_rule4(self):
        a = T.scalar() # the a is for 'anonymous' (un-named).
        x,s = T.scalars('xs')

        f = function([x, In(a, value=1.0,name='a'), s], a/s+x)

        self.failUnless(f(9,2,4) == 9.5) #can specify all args in order
        self.failUnless(f(9,2,s=4) == 9.5) # can give s as kwarg
        self.failUnless(f(9,s=4) == 9.25) # can give s as kwarg, get default a
        self.failUnless(f(9,a=2,s=4) == 9.5) # can give s as kwarg, a as kwarg
        self.failUnless(f(x=9,a=2, s=4) == 9.5) # can give all kwargs
        self.failUnless(f(x=9,s=4) == 9.25) # can give all kwargs
        checkfor(self, lambda:f(), TypeError) #takes exactly 3 non-keyword arguments (0 given)
        checkfor(self, lambda:f(5.0,x=9), TypeError) #got multiple values for keyword argument 'x'

    def test_state_access(self):
        a = T.scalar() # the a is for 'anonymous' (un-named).
        x,s = T.scalars('xs')

        f = function([x, In(a, value=1.0,name='a'), In(s, value=0.0, update=s+a*x)], s+a*x)

        self.failUnless(f[a] == 1.0)
        self.failUnless(f[s] == 0.0)

        self.failUnless(f(3.0) == 3.0)
        self.failUnless(f(3.0,a=2.0) == 9.0) #3.0 + 2*3.0

        self.failUnless(f[a] == 1.0) #state hasn't changed permanently, we just overrode it last line
        self.failUnless(f[s] == 9.0)

        f[a] = 5.0
        self.failUnless(f[a] == 5.0)
        self.failUnless(f(3.0) == 24.0) #9 + 3*5
        self.failUnless(f[s] == 24.0)

    def test_same_names(self):
        a,x,s = T.scalars('xxx')
        #implicit names would cause error.  What do we do?
        f = function([a, x, s], a+x+s)
        self.failUnless(f(1,2,3) == 6)
        checkfor(self, lambda:f(1,2,x=3), TypeError)

    def test_weird_names(self):
        a,x,s = T.scalars('xxx')
        
        checkfor(self, lambda:function([In(a,name=[])],[]), TypeError)

        def t():
            f = function([In(a,name=set(['adsf',()]), value=1.0),
                          In(x,name=(), value=2.0),
                          In(s,name=T.scalar(), value=3.0)], a+x+s)
        checkfor(self, t, TypeError)

    def test_copy(self):
        a = T.scalar() # the a is for 'anonymous' (un-named).
        x,s = T.scalars('xs')

        f = function([x, In(a, value=1.0,name='a'), In(s, value=0.0, update=s+a*x, mutable=True)], s+a*x)

        g = copy(f)
        #if they both return, assume  that they return equivalent things.

        self.failIf(g.container[x].storage is f.container[x].storage)
        self.failIf(g.container[a].storage is f.container[a].storage)
        self.failIf(g.container[s].storage is f.container[s].storage)

        self.failIf(g.value[a] is not f.value[a]) # should not have been copied
        self.failIf(g.value[s] is f.value[s]) # should have been copied because it is mutable.
        self.failIf((g.value[s] != f.value[s]).any()) # its contents should be identical

        self.failUnless(f(2, 1) == g(2)) #they should be in sync, default value should be copied.
        self.failUnless(f(2, 1) == g(2)) #they should be in sync, default value should be copied.
        f(1,2) # put them out of sync
        self.failIf(f(1, 2) == g(1, 2)) #they should not be equal anymore.

    def test_shared_state0(self):
        a = T.scalar() # the a is for 'anonymous' (un-named).
        x,s = T.scalars('xs')

        f = function([x, In(a, value=1.0,name='a'), In(s, value=0.0, update=s+a*x, mutable=True)], s+a*x)
        g = function([x, In(a, value=1.0,name='a'), In(s, value=f.container[s], update=s-a*x, mutable=True)], s+a*x)

        f(1, 2)
        self.failUnless(f[s] == 2)
        self.failUnless(g[s] == 2)
        g(1, 2)
        self.failUnless(f[s] == 0)
        self.failUnless(g[s] == 0)


# class T_function_examples(unittest.TestCase):
#     def test_accumulator(self):
#         """Test low-level interface with state."""
#         x = T.scalar('x')
#         s = T.scalar('s')

#         fn, states = program_states(inputs = [x], outputs = [], states = [(s, 0, s+x)])

#         sum = 0
#         for inc in [1, 4, 5,23, -324]:
#             sum += inc
#             fn.run([inc], states)
#             assert sum == states[0].value


#     def test_misc0(self):

#         fn_inc, states_inc = function_states(\
#                 inputs = [x], outputs = [], states = [(s, 0, s+x)])

#         fn_inc2, states_inc2 = function_states(\
#                 inputs = [x], outputs = [], states = [(s, 0, s+x)])

#         fn_inc_copy = copy.copy(fn_inc) #USE fn copy

#         # run() is like __call__, but requires an explicit state argument

#         fn_inc.run([5], states_inc) #run on own state object
#         fn_inc2.run([3], states_inc) #run on compatible state object
#         assert states_inc[0].value == 8

#         states_inc_copy = copy.copy(states_inc) #USE state copy
#         fn_inc_copy.run([2], states_inc_copy)
#         assert states_inc[0].value == 10   #compatible

#         fn_dec, states_dec = function_states(\
#                 inputs = [x], outputs = [], states = [((s, s-x), states_inc[0])])

#         try:
#             fn_inc.run([5], states_dec) # wrong kind of state for given program
#             self.fail("fn accepted an invalid state argument")
#         except SpecificException:
#             raise NotImplementedError() #TODO
#         except Exception:
#             self.fail("fn accepted an invalid state argument")

#     def test_perceptron(self):
#         """Test high-level state interface."""

#         mu0 = numpy.array([1.0,0.0])
#         mu1 = numpy.array([0.0,0.1])
#         si0 = numpy.ones_like(mu0) #unit variance
#         si1 = numpy.ones_like(mu1) #unit variance

#         #implicit internal state
#         r_state = random.random_state()
#         label = r_state.bernoulli(0.5) 

#         #implicit internal state for each DiagGaussian
#         x = label * DiagGaussian(mu0, si0, state=r_state) \
#                 + (1 - label) * random.DiagGaussian(mu1, si1, state=r_state)

#         w = T.tensor.dvector()
#         b = T.tensor.dscalar()
#         lr = 0.01

#         decision = dot(x,w) + b > 0
#         new_w = w + neq(label, decision) * lr * x
#         new_b = b + neq(label, decision) * (label * (-lr) + (1-label)*lr)

#         init_w = numpy.array([0.0, 0.0])
#         init_b = 0.0

#         io_stream = T.function([], [label, x], state={'seed':(r_state, 42)})

#         perceptron_learn = T.function([x, label], [decision], 
#                 state={
#                     'w':((w, update_w), init_w),
#                     'b':((b, update_b), init_b),
#                     'lr':(lr, 0.01)})

#         perceptron_use = T.function([x], [decision],
#                 state={
#                     'w':(w, perceptron_learn.shared['w']),
#                     'b':(b, perceptron_learn.shared['b'])})

#         errs = 0
#         for i in xrange(100):
#             il, ix = io_stream()

#             d0 = perceptron_use(ix)
#             d1 = perceptron_learn(ix, il)

#             assert d0 == d1

#             errs += (d0 != d1)

#             print d0
#         print 'errs =', errs 


# class T_dict_interface(unittest.TestCase):

#     def test_keyword(self):
#         x = T.scalar('x')
#         y = T.scalar('y')
#         s = T.scalar('s')

#         fn = function(input_kw = {'a':x, 'b':y}, outputs = [], state = {'s':(s, 0, s+x/y)})

#         try:
#             fn(1, 1)
#             self.fail("non-keyword call accepted!")
#         except SpecificException:
#             raise NotImplementedError()
#         except Exception:
#             self.fail("non-keyword call accepted!")

#         try:
#             fn(a=1)
#             self.fail("incomplete call accepted!")
#         except SpecificException:
#             raise NotImplementedError()
#         except Exception:
#             self.fail("incomplete call accepted!")

#         try:
#             fn(a=1, b=1, c=1)
#             self.fail("overcomplete call accepted!")
#         except SpecificException:
#             raise NotImplementedError()
#         except Exception:
#             self.fail("overcomplete call accepted!")

#     def test_aliased_state(self):
#         """Test keyword input and copy."""
#         x = T.scalar('x')
#         y = T.scalar('y')
#         s = T.scalar('s')

#         fn = function(input_kw = {'a':x, 'b':y}, outputs = [], state = {'s':(s, 0, s+x/y)})
#         fn2 = fn.copy()
#         fn3 = fn.copy()

#         fn(a=2, b=5)
#         fn2(a=5, b=2)
#         fn3(b=2, a=5)
#         assert fn.state['s'] == 2.0/5
#         assert fn2.state['s'] == 5.0/2 
#         assert fn3.state['s'] == 5.0/2

#         #fn and fn3 use the same sort of state, so this is OK.
#         fn3.state = fn.state 

#         fn.state['s'] = 0
#         fn(a=1, b=1)   #increment the shared state
#         assert fn3.state['s'] == 1
#         fn3(a=-1, b=1) #decrement the shared state
#         assert fn.state['s'] == 0


if __name__ == '__main__':

    if 1:
        unittest.main()
    else:
        testcases = []
        testcases.append(T_function)

        #<testsuite boilerplate>
        testloader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for testcase in testcases:
            suite.addTest(testloader.loadTestsFromTestCase(testcase))
        unittest.TextTestRunner(verbosity=2).run(suite)
        #</boilerplate>

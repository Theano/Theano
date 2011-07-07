import copy
import cPickle
import numpy
import unittest


from theano import gof,config
from theano.scalar import mul
from theano.compile.io import In, Out
from theano.compile import function
from theano.gof.python25 import all, any

from theano import tensor
from theano import tensor as T
import theano

import numpy as N
from numpy.testing.noseclasses import KnownFailureTest

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


class T_function(unittest.TestCase):
    def test_none(self):
        fn = function([], None) #ok
        rval = fn()
        if rval == []:
            raise KnownFailureTest('See #254: Using None as function output leads to [] return value')
        else:
            assert rval is None

    def test_empty(self):
        fn = function([], []) #ok
        self.assertTrue(fn() == [])

    def test_extra_inputs(self):
        x,s = T.scalars('xs')
        fn = function([x], [x])
        self.assertRaises(TypeError,fn,1,2)

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
        self.assertTrue(fn(2,3) == [5])
        # no state
        self.assertTrue(fn(2,3) == [5])

    def test_input_anon_unpack(self):
        x,s = T.scalars('xs')
        fn = function([s,x], x+s)
        self.assertTrue(fn(2,3) == 5)

    def test_naming_rule0(self):
        x,s = T.scalars('xs')
        f = function([x,s], x/s)
        self.assertTrue(f(1,2) == 0.5)
        self.assertTrue(f(2,1) == 2.0)
        self.assertTrue(f(s=2,x=1) == 0.5)
        self.assertTrue(f(x=2,s=1) == 2.0)
        self.assertTrue(f(2, s=1) == 2.0)
        checkfor(self, lambda :f(2, x=2.0), TypeError) #got multiple values for keyword argument 'x'
        checkfor(self, lambda :f(x=1), TypeError) #takes exactly 2 non-keyword arguments (1 given)
        checkfor(self, lambda :f(s=1), TypeError) #takes exactly 2 non-keyword arguments (0 given)

    def test_naming_rule1(self):
        a = T.scalar() # the a is for 'anonymous' (un-named).
        x,s = T.scalars('xs')
        f = function([a, s], a/s)
        self.assertTrue(f(1,2) == 0.5)
        self.assertTrue(f(2,1) == 2.0)
        self.assertTrue(f(2, s=1) == 2.0)
        checkfor(self, lambda:f(q=2,s=1), TypeError) #got unexpected keyword argument 'q'
        checkfor(self, lambda:f(a=2,s=1), TypeError) #got unexpected keyword argument 'a'

    def test_naming_rule2(self):
        a = T.scalar() # the a is for 'anonymous' (un-named).
        x,s = T.scalars('xs')

        #x's name is ignored because it is followed by anonymous parameter a.
        f = function([x, a, s], a/s)
        self.assertTrue(f(9,1,2) == 0.5)
        self.assertTrue(f(9,2,1) == 2.0)
        self.assertTrue(f(9,2, s=1) == 2.0)
        checkfor(self, lambda:f(x=9,a=2,s=1), TypeError) #got unexpected keyword argument 'x'
        checkfor(self, lambda:f(5.0,x=9), TypeError) #got unexpected keyword argument 'x'

    def test_naming_rule3(self):
        a = T.scalar() # the a is for 'anonymous' (un-named).
        x,s = T.scalars('xs')

        #x's name is not ignored (as in test_naming_rule2) because a has a default value.
        f = function([x, In(a, value=1.0), s], a/s+x)
        self.assertTrue(f(9,2,4) == 9.5) #can specify all args in order
        self.assertTrue(f(9,2,s=4) == 9.5) # can give s as kwarg
        self.assertTrue(f(9,s=4) == 9.25) # can give s as kwarg, get default a
        self.assertTrue(f(x=9,s=4) == 9.25) # can give s as kwarg, omit a, x as kw
        checkfor(self, lambda:f(x=9,a=2,s=4), TypeError) #got unexpected keyword argument 'a'
        checkfor(self, lambda:f(), TypeError) #takes exactly 3 non-keyword arguments (0 given)
        checkfor(self, lambda:f(x=9), TypeError) #takes exactly 3 non-keyword arguments (1 given)

    def test_naming_rule4(self):
        a = T.scalar() # the a is for 'anonymous' (un-named).
        x,s = T.scalars('xs')

        f = function([x, In(a, value=1.0,name='a'), s], a/s+x)

        self.assertTrue(f(9,2,4) == 9.5) #can specify all args in order
        self.assertTrue(f(9,2,s=4) == 9.5) # can give s as kwarg
        self.assertTrue(f(9,s=4) == 9.25) # can give s as kwarg, get default a
        self.assertTrue(f(9,a=2,s=4) == 9.5) # can give s as kwarg, a as kwarg
        self.assertTrue(f(x=9,a=2, s=4) == 9.5) # can give all kwargs
        self.assertTrue(f(x=9,s=4) == 9.25) # can give all kwargs
        checkfor(self, lambda:f(), TypeError) #takes exactly 3 non-keyword arguments (0 given)
        checkfor(self, lambda:f(5.0,x=9), TypeError) #got multiple values for keyword argument 'x'

    def test_state_access(self):
        a = T.scalar() # the a is for 'anonymous' (un-named).
        x,s = T.scalars('xs')

        f = function([x, In(a, value=1.0,name='a'), In(s, value=0.0, update=s+a*x)], s+a*x)

        self.assertTrue(f[a] == 1.0)
        self.assertTrue(f[s] == 0.0)

        self.assertTrue(f(3.0) == 3.0)
        self.assertTrue(f(3.0,a=2.0) == 9.0) #3.0 + 2*3.0

        self.assertTrue(f[a] == 1.0) #state hasn't changed permanently, we just overrode it last line
        self.assertTrue(f[s] == 9.0)

        f[a] = 5.0
        self.assertTrue(f[a] == 5.0)
        self.assertTrue(f(3.0) == 24.0) #9 + 3*5
        self.assertTrue(f[s] == 24.0)

    def test_same_names(self):
        a,x,s = T.scalars('xxx')
        #implicit names would cause error.  What do we do?
        f = function([a, x, s], a+x+s)
        self.assertTrue(f(1,2,3) == 6)
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

        g = copy.copy(f)
        #if they both return, assume  that they return equivalent things.

        self.assertFalse(g.container[x].storage is f.container[x].storage)
        self.assertFalse(g.container[a].storage is f.container[a].storage)
        self.assertFalse(g.container[s].storage is f.container[s].storage)

        self.assertFalse(g.value[a] is not f.value[a]) # should not have been copied
        self.assertFalse(g.value[s] is f.value[s]) # should have been copied because it is mutable.
        self.assertFalse((g.value[s] != f.value[s]).any()) # its contents should be identical

        self.assertTrue(f(2, 1) == g(2)) #they should be in sync, default value should be copied.
        self.assertTrue(f(2, 1) == g(2)) #they should be in sync, default value should be copied.
        f(1,2) # put them out of sync
        self.assertFalse(f(1, 2) == g(1, 2)) #they should not be equal anymore.

    def test_shared_state0(self):
        a = T.scalar() # the a is for 'anonymous' (un-named).
        x,s = T.scalars('xs')

        f = function([x, In(a, value=1.0,name='a'), In(s, value=0.0, update=s+a*x, mutable=True)], s+a*x)
        g = function([x, In(a, value=1.0,name='a'), In(s, value=f.container[s], update=s-a*x, mutable=True)], s+a*x)

        f(1, 2)
        self.assertTrue(f[s] == 2)
        self.assertTrue(g[s] == 2)
        g(1, 2)
        self.assertTrue(f[s] == 0)
        self.assertTrue(g[s] == 0)

    def test_shared_state1(self):
        a = T.scalar() # the a is for 'anonymous' (un-named).
        x,s = T.scalars('xs')

        f = function([x, In(a, value=1.0,name='a'), In(s, value=0.0, update=s+a*x, mutable=True)], s+a*x)
        g = function([x, In(a, value=1.0,name='a'), In(s, value=f.container[s])], s+a*x)

        f(1, 2)
        self.assertTrue(f[s] == 2)
        self.assertTrue(g[s] == 2)
        f(1, 2)
        g(1, 2)
        self.assertTrue(f[s] == 4)
        self.assertTrue(g[s] == 4)

    def test_shared_state2(self):
        a = T.scalar() # the a is for 'anonymous' (un-named).
        x,s = T.scalars('xs')

        f = function([x, In(a, value=1.0,name='a'), In(s, value=0.0, update=s+a*x,
            mutable=False)], s+a*x)
        g = function([x, In(a, value=1.0,name='a'), In(s, value=f.container[s])], s+a*x)

        f(1, 2)
        self.assertTrue(f[s] == 2)
        self.assertTrue(g[s] == 2)
        f(1, 2)
        self.assertTrue(f[s] == 4)
        self.assertTrue(g[s] == 4)
        g(1, 2) # has no effect on state
        self.assertTrue(f[s] == 4)
        self.assertTrue(g[s] == 4)

    def test_shared_state_not_implicit(self):
        # This test is taken from the documentation in
        # doc/topics/function.txt. If it does not pass anymore and yet the
        # behavior is still intended the doc and the test should both be
        # updated accordingly.
        x, s = T.scalars('xs')
        inc = function([x, In(s, update=(s+x), value=10.0)], [])
        dec = function([x, In(s, update=(s-x), value=inc.container[s],
            implicit = False)], [])
        self.assertTrue(dec[s] is inc[s])
        inc[s] = 2
        self.assertTrue(dec[s] == 2)
        dec(1)
        self.assertTrue(inc[s] == 1)
        dec(1, 0)
        self.assertTrue(inc[s] == -1)
        self.assertTrue(dec[s] == -1)


    def test_constant_output(self):
        # Test that if the output is a constant, we respect the theano memory interface
        f = theano.function([],theano.tensor.constant([4]))
        print f.maker.env.toposort()
        out = f()
        assert (out==4).all()
        out[0]=3
        out2 = f()
        # If the following 2 asserts fail it mean Theano broke it's memory contract.
        assert out2 is not out
        assert (out2==4).all()

        # Test that if the output is a constant and borrow, we respect the theano memory interface
        f = theano.function([],Out(theano.tensor.constant([4]), borrow=True))
        print f.maker.env.toposort()
        out = f()
        assert (out==4).all()
        out[0]=3
        out2 = f()

        if isinstance(theano.compile.mode.get_default_mode(),
                      theano.compile.DebugMode):
            # In DebugMode, we don't implement optimization based on borrow on the output.
            assert (out2==4).all()
        else:
            assert out2 is out
            assert (out2==3).all()


    def test_borrow_input(self):
        """
        Tests that the contract for io.In is respected. When borrow=False, it should be
        impossible for outputs to be aliased to the input variables provided by the user,
        either through a view-map or a destroy map. New tests should be added in the future
        when borrow=True is implemented.
        """
        a = T.dmatrix()
        aval = numpy.random.rand(3,3)

        # when borrow=False, test that a destroy map cannot alias output to input
        f = theano.function([In(a, borrow=False)], Out(a+1, borrow=True))
        assert numpy.all(f(aval) == aval+1)
        assert not numpy.may_share_memory(aval, f(aval))

        # when borrow=False, test that a viewmap cannot alias output to input
        f = theano.function([In(a, borrow=False)], Out(a[0,:], borrow=True))
        assert numpy.all(f(aval) == aval[0,:])
        assert not numpy.may_share_memory(aval, f(aval))

    def test_borrow_output(self):
        a = T.dmatrix()
        f = function([a], Out(a, borrow=False))
        o = N.ones((3,3))
        assert o is not f(o) #function no longer permits aliasing outputs to inputs

        f = function([a], Out(a*4, borrow=False))
        o = N.ones((3,3))
        four = f(o)
        assert numpy.all(four==4)
        f(o+.1) #should not clobber the memory used to store four
        assert numpy.all(four==4)

        f = function([a], Out(a*4, borrow=True), mode=theano.Mode('c|py_nogc', 'fast_run'))
        o = N.ones((3,3))
        four = f(o)
        assert numpy.all(four==4)
        f(o+.1) #should clobber the memory used to store four
        assert not numpy.all(four==4)

class T_picklefunction(unittest.TestCase):

    def test_deepcopy(self):
        a = T.scalar() # the a is for 'anonymous' (un-named).
        x,s = T.scalars('xs')

        f = function([x, In(a, value=1.0,name='a'), In(s, value=0.0, update=s+a*x, mutable=True)], s+a*x)

        try:
            g = copy.deepcopy(f)
        except NotImplementedError, e:
            if e[0].startswith('DebugMode is not picklable'):
                return
            else:
                raise
        #if they both return, assume  that they return equivalent things.
        #print [(k,id(k)) for k in f.finder.keys()]
        #print [(k,id(k)) for k in g.finder.keys()]

        self.assertFalse(g.container[0].storage is f.container[0].storage)
        self.assertFalse(g.container[1].storage is f.container[1].storage)
        self.assertFalse(g.container[2].storage is f.container[2].storage)
        self.assertFalse(x in g.container)
        self.assertFalse(x in g.value)
        self.assertTrue(len(f.defaults) == len(g.defaults))
        print 'f.defaults = %s' % (f.defaults, )
        print 'g.defaults = %s' % (g.defaults, )
        self.assertTrue(all([f_req == g_req and f_feed == g_feed and
            f_val == g_val
            for ((f_req, f_feed, f_val), (g_req, g_feed, g_val)) in zip(
                f.defaults, g.defaults)]))

        self.assertFalse(g.value[1] is f.value[1]) # should not have been copied
        self.assertFalse(g.value[2] is f.value[2]) # should have been copied because it is mutable.
        self.assertFalse((g.value[2] != f.value[2]).any()) # its contents should be identical

        self.assertTrue(f(2, 1) == g(2)) #they should be in sync, default value should be copied.
        self.assertTrue(f(2, 1) == g(2)) #they should be in sync, default value should be copied.
        f(1,2) # put them out of sync
        self.assertFalse(f(1, 2) == g(1, 2)) #they should not be equal anymore.
        g(1, 2) # put them back in sync
        self.assertTrue(f(3) == g(3)) # They should be in sync again.

    def test_deepcopy_shared_container(self):
        # Ensure that shared containers remain shared after a deep copy.
        a, x = T.scalars('ax')

        h = function([In(a, value = 0.0)], a)
        f = function([x, In(a, value=h.container[a], implicit = True)], x + a)

        try:
            memo = {}
            ac = copy.deepcopy(a)
            memo.update({id(a): ac})
            hc = copy.deepcopy(h, memo = memo)
            memo.update({id(h): hc})
            fc = copy.deepcopy(f, memo = memo)
        except NotImplementedError, e:
            if e[0].startswith('DebugMode is not picklable'):
                return
            else:
                raise
        h[a] = 1
        hc[ac] = 2
        self.assertTrue(f[a] == 1)
        self.assertTrue(fc[ac] == 2)

    def test_pickle(self):
        a = T.scalar() # the a is for 'anonymous' (un-named).
        x,s = T.scalars('xs')

        f = function([x, In(a, value=1.0,name='a'), In(s, value=0.0, update=s+a*x, mutable=True)], s+a*x)

        try:
            # Note that here we also test protocol 0 on purpose, since it
            # should work (even though one should not use it).
            g = cPickle.loads(cPickle.dumps(f, protocol=0))
            g = cPickle.loads(cPickle.dumps(f, protocol=-1))
        except NotImplementedError, e:
            if e[0].startswith('DebugMode is not picklable'):
                return
            else:
                raise
        #if they both return, assume  that they return equivalent things.
        #print [(k,id(k)) for k in f.finder.keys()]
        #print [(k,id(k)) for k in g.finder.keys()]

        self.assertFalse(g.container[0].storage is f.container[0].storage)
        self.assertFalse(g.container[1].storage is f.container[1].storage)
        self.assertFalse(g.container[2].storage is f.container[2].storage)
        self.assertFalse(x in g.container)
        self.assertFalse(x in g.value)

        self.assertFalse(g.value[1] is f.value[1]) # should not have been copied
        self.assertFalse(g.value[2] is f.value[2]) # should have been copied because it is mutable.
        self.assertFalse((g.value[2] != f.value[2]).any()) # its contents should be identical

        self.assertTrue(f(2, 1) == g(2)) #they should be in sync, default value should be copied.
        self.assertTrue(f(2, 1) == g(2)) #they should be in sync, default value should be copied.
        f(1,2) # put them out of sync
        self.assertFalse(f(1, 2) == g(1, 2)) #they should not be equal anymore.

    def test_optimizations_preserved(self):
        a = T.dvector() # the a is for 'anonymous' (un-named).
        x = T.dvector('x')
        s = T.dvector('s')
        xm = T.dmatrix('x')
        sm = T.dmatrix('s')

        f = function([a, x, s, xm, sm], ((a.T.T)*(tensor.dot(xm, (sm.T.T.T)) + x).T * (x/x) + s))
        old_default_mode = config.mode
        old_default_opt  = config.optimizer
        old_default_link = config.linker
        try:
            try:
                str_f = cPickle.dumps(f, protocol=-1)
                config.mode = 'Mode'
                config.linker = 'py'
                config.optimizer = 'None'
                g = cPickle.loads(str_f)
                #print g.maker.mode
                #print compile.mode.default_mode
            except NotImplementedError, e:
                if e[0].startswith('DebugMode is not pickl'):
                    g = 'ok'
        finally:
            config.mode = old_default_mode
            config.optimizer = old_default_opt
            config.linker = old_default_link

        if g == 'ok':
            return

        assert f.maker is not g.maker
        assert f.maker.env is not g.maker.env
        tf = f.maker.env.toposort()
        tg = f.maker.env.toposort()
        assert len(tf) == len(tg)
        for nf, ng in zip(tf, tg):
            assert nf.op == ng.op
            assert len(nf.inputs) == len(ng.inputs)
            assert len(nf.outputs) == len(ng.outputs)
            assert [i.type for i in nf.inputs] == [i.type for i in ng.inputs]
            assert [i.type for i in nf.outputs] == [i.type for i in ng.outputs]


    def test_multiple_functions(self):
        a = T.scalar() # the a is for 'anonymous' (un-named).
        x,s = T.scalars('xs')
        v = T.vector('v')


        # put in some inputs
        list_of_things = [s, x, v]

        # some derived thing, whose inputs aren't all in the list
        list_of_things.append(a * x + s )

        f1 = function([x, In(a, value=1.0,name='a'), In(s, value=0.0, update=s+a*x, mutable=True)], s+a*x)
        list_of_things.append(f1)

        # now put in a function sharing container with the previous one
        f2 = function([x, In(a, value=1.0,name='a'), In(s, value=f1.container[s], update=s+a*x, mutable=True)], s+a*x)
        list_of_things.append(f2)

        assert isinstance(f2.container[s].storage, list)
        assert f2.container[s].storage is f1.container[s].storage

        # now put in a function with non-scalar
        v_value = numpy.asarray([2,3,4.], dtype=config.floatX)
        f3 = function([x, In(v, value=v_value)], x+v)
        list_of_things.append(f3)

        # try to pickle the entire things
        try:
            saved_format = cPickle.dumps(list_of_things, protocol=-1)
            new_list_of_things = cPickle.loads(saved_format)
        except NotImplementedError, e:
            if e[0].startswith('DebugMode is not picklable'):
                return
            else:
                raise

        # now test our recovered new_list_of_things
        # it should be totally unrelated to the original
        # it should be interdependent in the same way as the original

        ol = list_of_things
        nl = new_list_of_things

        for i in range(4):
            assert nl[i] != ol[i]
            assert nl[i].type == ol[i].type
            assert nl[i].type is not ol[i].type

        # see if the implicit input got stored
        assert ol[3].owner.inputs[1] is s
        assert nl[3].owner.inputs[1] is not s
        assert nl[3].owner.inputs[1].type == s.type

        # moving on to the functions...
        for i in range(4,7):
            assert nl[i] != ol[i]

        # looking at function number 1, input 's'
        assert nl[4][nl[0]] is not ol[4][ol[0]]
        assert nl[4][nl[0]] == ol[4][ol[0]]
        assert nl[4](3) == ol[4](3)

        # looking at function number 2, input 's'
        # make sure it's shared with the first function
        assert ol[4].container[ol[0]].storage is ol[5].container[ol[0]].storage
        assert nl[4].container[nl[0]].storage is nl[5].container[nl[0]].storage
        assert nl[5](3) == ol[5](3)
        assert nl[4].value[nl[0]] == 6

        assert numpy.all(nl[6][nl[2]] == numpy.asarray([2, 3., 4]))

    def test_broken_pickle_with_shared(self):
        saves = []
        def pers_save(obj):
            if isinstance(obj, numpy.ndarray):
                saves.append(obj)
                return len(saves)-1
            else:
                return None
        def pers_load(id):
            return saves[id]

        a = numpy.random.rand(4, 5)
        b = numpy.random.rand(5, 4)

        x = theano.tensor.matrix()
        y = theano.shared(b)

        f = theano.function([x], theano.tensor.dot(x, y))

        import StringIO
        fp = StringIO.StringIO()
        p = cPickle.Pickler(fp, 2)
        p.persistent_id = pers_save
        try:
            p.dump(f)
        except NotImplementedError, e:
            if e[0].startswith('DebugMode is not picklable'):
                return
            else:
                raise
        fp2 = StringIO.StringIO(fp.getvalue())
        fp.close()
        p = cPickle.Unpickler(fp2)
        p.persistent_load = pers_load
        f2 = p.load()
        fp2.close()

    def test_pickle_class_with_functions(self):

        blah = SomethingToPickle()
        assert blah.f2.container[blah.s].storage is blah.f1.container[blah.s].storage

        try:
            blah2 = copy.deepcopy(blah)
        except NotImplementedError, e:
            if e[0].startswith('DebugMode is not picklable'):
                return
            else:
                raise

        assert blah2.f2.container[blah2.s].storage is blah2.f1.container[blah2.s].storage

        assert blah.f1[blah.s] == blah2.f1[blah2.s]

        blah.f2(5)
        assert blah.f1[blah.s] != blah2.f1[blah2.s]



class SomethingToPickle(object):
    def __init__(self):
        a = T.scalar() # the a is for 'anonymous' (un-named).
        x,s = T.scalars('xs')
        v = T.vector('v')

        self.s = s
        self.x = x
        self.v = v

        self.e = a * x + s

        self.f1 = function([x, In(a, value=1.0,name='a'), In(s, value=0.0, update=s+a*x, mutable=True)], s+a*x)

        self.f2 = function([x, In(a, value=1.0,name='a'), In(s, value=self.f1.container[s], update=s+a*x, mutable=True)], s+a*x)


if __name__ == '__main__':

    if 1:
        unittest.main()
    elif 0:
        testcases = []
        testcases.append(T_function)

        #<testsuite boilerplate>
        testloader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for testcase in testcases:
            suite.addTest(testloader.loadTestsFromTestCase(testcase))
        unittest.TextTestRunner(verbosity=2).run(suite)
        #</boilerplate>
    elif 0:
        theano.config.mode = 'FAST_COMPILE'
        t = T_picklefunction()
        def fu(b):
            assert b
        t.failUnless = fu
        t.test_deepcopy_shared_container()

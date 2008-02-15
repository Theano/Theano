import gof
import core

class Grad(object):
    """A dictionary-like class, into which derivative expressions may be added.

    This class maps keys to their ids to deal with the ndarray, which is not
    hashable.
    
    Attributes: None

    Methods:

    add()
    bprop()
    __call__()
    __getitem__()
    """
    def __init__(self, dct={}):
        self.map = {}
        self.outputs = []
        for key,val in dct.items():
            self.add_output(key,val)
        self.did_bprop = False

    def __contains__(self, item):
        return item in self.map

    def __getitem__(self, item):
        """Map item to its id and retrieve it."""
        key = core.wrap(item)
        try:
            return self.map[key]
        except KeyError:
            return core.UNDEFINED

    def __setitem__(self, item, val):
        """Map item to its id and store internally."""
        self.map[item] = val

    def add_output(self, r, dr):
        self.add(r, dr)
        self.outputs.append(r)
        
    def add(self, r, dr):
        """Add dr to the sum of gradients associated with r.
        
        This function should be fed as follows:

        if dr is UNDEFINED:
            r could be anything
        else dr might be core.UNCOMPUTED:
            r may be uncomputed or NumpyR
        else dr will be isinstance(NumpyR):
            r may be uncomputed or NumpyR

        """
        if dr is core.UNDEFINED:
            # nothing to do
            pass
        else:
            if r.data is core.UNCOMPUTED or dr.data is core.UNCOMPUTED:
                pass
            else: # try some hacky checks to catch obvious mistakes
                if not hasattr(r.data, 'shape'):
                    raise ValueError(('Grad::add r lacks shape: type=',
                        type(r.data)))
                if not hasattr(dr.data, 'shape'):
                    raise ValueError(('Grad::add dr lacks shape: type=',
                        type(dr.data)))
                if r.data.shape != dr.data.shape:
                    raise ValueError(('Grad::add r, dr shape mismatch',
                        r.data.shape, dr.data.shape))

            # add dr to self[r]
            if r in self:
                self[r] = self[r] + dr
            else:
                self[r] = dr

    def bprop(self, maybe_redo=False):
        """Build a backpropagation graph.

        The gradient associated with each value is stored in <self> which
        inherits from dictionary.  The idea is that when we call
        op.update_gradient(self), that the op's update_gradient function calls
        back into <self>.add(), and says what gradient term goes with each of
        its inputs.  Most of the time, the gradients of the op's outputs are
        necessary for the op to compute the gradient wrt its inputs, so
        op.update_gradient will usually call <self>.__getitem__, (via the
        [] notation). 
        
        It is essential that the gradient of an op's outputs be fully computed
        before op.update_gradient is called, or else key errors may be raised
        and incorrect gradients will be computed.

        bprop sets the omega evaluation mode to be 'build', so no computations
        or allocations are done by bprop.
        """
        if not maybe_redo and self.did_bprop:
            raise Exception('bprop has already been done. Consider calling with maybe_redo=True.')
        core.build_mode()
        try:
            outputs = self.outputs
            inputs = gof.graph.inputs(outputs)
            for op in gof.graph.io_toposort(inputs, outputs).__reversed__():
                op.update_gradient(self)
        finally:
            core.pop_mode()
            self.did_bprop = True

    def __call__(self, item):
        """Return a derivative term.

        If the current omega evaluation mode is 'build_eval' then the node is
        computed if necessary.
        """
        if not self.did_bprop:
            raise Exception('Grad.__call__ only makes sense after a bprop')
        rval = self[item]
        if rval is not core.UNDEFINED \
                and core.current_mode() == 'build_eval':
            rval.compute()
        return rval

def grad(cost, param=None, cost_grad = 1.0):
    """Return symbolic expression of gradient of <cost> wrt <param>.

    If <param> is None, then return a Grad instance, from which the gradients of
    multiple objects can be retrieved using the __getitem__ or __call__ methods
    (as in function currying in languages such as scheme and OCaML).

    If <param> is not None, then return the gradient expression for 
    d cost / d param.

    """
    if core.current_mode() == 'eval':
        raise NotImplementedError('Gradient-related functions are not available in eval mode')

    rval = Grad({cost:core.wrap(cost_grad)})
    rval.bprop()
    if param is None:
        return rval
    else:
        return rval(param)

class update_gradient_via_grad:
    """Inherit from this class to add a convenient self.update_gradient function"""

    def update_gradient(self, grad_d):
        """Call self.grad() and add the result to grad_d

        This function is called by grad.Grad.bprop() to construct a symbolic gradient graph.

        self.grad is called like this:

            self.grad(*(self.inputs + [grad_d[output] for output in self.outputs]))

        In general, grad() should return a list of PythonR instances whose
        length matches that of self.inputs, and whose elements are the
        gradients of self.inputs.

        There is a (but often used) special feature in place to automatically
        wrap the return value of grad() in a list if it is a PythonR instance
        and the op is unary.  This makes many grad implementations a little
        cuter.

        """
        inputgs = self.grad(*(self.inputs + [grad_d[output] for output in self.outputs]))
        if len(self.inputs) == 1 and isinstance(inputgs, gof.PythonR):
            inputgs = [inputgs]
        else:
            assert len(inputgs) == len(self.inputs)
        for input, inputg in zip(self.inputs, inputgs):
            grad_d.add(input, inputg)

#
# UNIT TEST
#
import unittest
import numpy
import compile

class _testCase (unittest.TestCase):

    class posneg(core.omega_op):
        nout=2
        def impl(x): return x, -x
        def grad(x, gpos, gneg): return gpos - gneg

    class posnegzero(core.omega_op):
        nout=3
        def impl(x): return x, -x, 0.0
        def grad(x, gpos, gneg, gzero): return gpos - gneg

    def setUp(self):
        numpy.random.seed(1)
        core.build_eval_mode()

    def matinv(self,dim):
        w = core.wrap(numpy.random.rand(dim,dim))
        wi = core.wrap(numpy.random.rand(dim,dim))
        ident = core.wrap(numpy.identity(dim))

        for i in xrange(300):
            wwi = core.dot(w, wi)
            diff = wwi - ident
            ssdiff = core.sum((diff**2))
            if i == 0:
                str0 = str_ssdiff = str(ssdiff)

            #print ssdiff
            g = grad(ssdiff)
            gw = g(w)
            w.data += -0.4 * gw.data

        return str0, str(ssdiff)

    def matinv_compiled(self, dim):
        w = core.wrap(numpy.random.rand(dim,dim))
        wi = core.wrap(numpy.random.rand(dim,dim))
        ident = core.wrap(numpy.identity(dim))

        wwi = core.dot(w, wi)
        diff = wwi - ident
        ssdiff = core.sum((diff**2))
        str0 = str_ssdiff = str(ssdiff)

        #print ssdiff
        g = grad(ssdiff)
        gw = g(w)

        prog = compile.single(g(w),ssdiff)

        for i in xrange(300):
            prog()
            w.data += -0.4 * gw.data

        return str0, str(ssdiff)

    def test0(self):
        """Matrix inversion by gradient descent (eval mode)"""
        self.assertEqual(('2.67327580893', '0.000438649434819'), self.matinv(3))

    def test1(self):
        """Matrix inversion by gradient descent (compiled mode)"""
        self.assertEqual(('2.67327580893', '0.000438649434819'),
                self.matinv_compiled(3))

    def test_grad_wrt_ndarray_pointer(self):
        """Grad indexing by un-wrapped ndarray"""
        a = numpy.ones((4, 4))
        b = numpy.ones((4, 4))
        c = numpy.ones((4, 4))
        expr = core.sum(core.dot(core.add(a, b), c))
        g = grad(expr)
        g[a]

    def test_bprop_call_order(self):
        """Ensure call before bprop is illegal"""
        a = numpy.ones((3,3,3))
        b = core.exp(a)
        gb = Grad({b:core.wrap(a)})
        try:
            gb(a)
            self.assertEqual('should have raised',0)
        except Exception, e:
            self.assertEqual(str(e), 'Grad.__call__ only makes sense after a bprop')
            return
        self.assertEqual('should have caught, returned',0)

    def test_undefined_grad0(self):
        """Make sure posneg works with fully specified gradients"""

        a = numpy.ones((3,3,3))
        b,c = _testCase.posneg(a)

        g = Grad({b:core.wrap(a),c:core.wrap(a)})
        g.bprop()
        max = numpy.max(g(a))
        min = numpy.min(g(a))
        self.assertEqual(max, min)
        self.assertEqual(max, 0.0)

    def test_undefined_grad1(self):
        """Propagate undefined values through posneg's first gradient"""

        a = numpy.ones((3,3,3))
        b,c = _testCase.posneg(a)

        gb = Grad({b:core.wrap(a)})
        try:
            gb.bprop()
            self.assertEqual('should have raised',0)
        except AttributeError, e:
            self.assertEqual(str(e), "Keyword instance has no attribute 'shape'")
            return
        self.assertEqual("Should have been error", 0)
    
    def test_undefined_grad2(self):
        """Propagate undefined values through posneg's second gradient"""

        a = numpy.ones((3,3,3))
        b,c = _testCase.posneg(a)
        gc = Grad({c:core.wrap(a)})
        try:
            gc.bprop()
            self.assertEqual('should have raised',0)
        except AttributeError, e:
            self.assertEqual(str(e), "Keyword instance has no attribute 'shape'")
            return
        self.assertEqual("Should have been error", 0)

    def test_undefined_grad3(self):
        """Ignore undefined values properly"""

        a = numpy.ones((3,3,3))
        b,c,d = _testCase.posnegzero(a)
        #print b, c, d
        g = Grad({b:core.wrap(a), c:core.wrap(a)})
        g.bprop()
        max = numpy.max(g(a))
        min = numpy.min(g(a))
        self.assertEqual(max, min)
        self.assertEqual(max, 0.0)

    def test_repeat_bprop(self):
        """Refuse to repeat bprop"""

        a = numpy.ones((3,3,3))
        b,c,d = _testCase.posnegzero(a)
        #print b, c, d
        g = Grad({b:core.wrap(a), c:core.wrap(a)})
        g.bprop()
        try:
            g.bprop()
            self.assertEqual('should have raised')
        except Exception, e:
            self.assertEqual(str(e), 'bprop has already been done. Consider calling with maybe_redo=True.')
            return
        self.assertEqual('should have caught')

    def test_repeat_bprop1(self):
        """Force repeat bprop"""

        a = numpy.ones((3,3,3))
        z = numpy.zeros((3,3,3))
        b,c,d = _testCase.posnegzero(a)
        #print b, c, d
        g = Grad({b:core.wrap(a), c:core.wrap(z)})
        g.bprop()
        g.bprop(maybe_redo=True)
        max = numpy.max(g(a))
        min = numpy.min(g(a))
        self.assertEqual(max, min)
        self.assertEqual(max, 2.0)

    def tearDown(self):
        core.pop_mode()

if __name__ == '__main__':
    unittest.main()


import unittest
import theano
import numpy

from theano import config
from theano.tests  import unittest_tools as utt

'''
  Questions and notes about scan that should be answered :

   * Even though it does not make it publically known in
   the documentation, scan allows you to set both a return_steps
   flag and a store_steps flag ( the first one is a soft condition telling
   you how many steps to return, the second one determines how much memory to
   allocate). There is an optimization as well, that transforms return_steps to
   store_steps. Questions :
      - what happens if both flags are set ?
       answer: whatever return_steps says is ignored, and store_steps is used
      - the optimization works only with return_steps = -1; can it be made to work
        with other values ?
       answer: 6 Jul 2010 RP :it is a bit harry to figure out from the subtensors what
       exactly you need

   * Scan seems to do copies of every input variable. Is that needed?
   answer : probably not, but it doesn't hurt also ( what we copy is theano variables,
     which just cary information about the type / dimension of the data)


   * There is some of scan functionality that is not well documented
'''


class multiple_outputs_numeric_grad:
    """WRITEME"""
    type_eps = {'float64': 1e-7,
            'float32': 3e-3}

    def __init__(self, f, pt, ndarray_mask = None, eps=None):
        """Return the gradient of f at pt.

        This function computes the gradient by a one-sided finite differences of a
        fixed step size (eps).

        It is assumed that f(...) will return a scalar.
        :param eps: the stepsize for the finite differencing.  None means input
        dtype-dependent. See `type_eps`.
        """

        def prod(inputs):
            rval = 1
            for i in inputs:
                rval *= i
            return rval
        packed_pt = False
        if not isinstance(pt, (list, tuple)):
            pt = [pt]
            packed_pt = True

        # This mask tells us if we are dealing with an ndarray input or
        # something else ( a random state ? ) with which we shouldn't really
        # mess up
        if not ndarray_mask:
                ndarray_mask = [True for x in pt ]

        dtype_eps = multiple_outputs_numeric_grad.type_eps['float64']

        for i,p in enumerate(pt):
            if ndarray_mask[i]:
                pt[i] = numpy.array(p)
                _eps = multiple_outputs_numeric_grad.type_eps[str(pt[i].dtype)]
                if _eps > dtype_eps:
                        dtype_eps = _eps


        # Compute clean output:
        f_x = f(*pt)
        gx = []
        # now iterate over the elements of x and call f on those + delta x
        for i in xrange(len(pt)):
            if ndarray_mask[i]:
                # It is a ndarray that we can tweak
                if eps:
                    _eps = eps
                else:
                    _eps = dtype_eps
                if pt[i].ndim :
                    _g = []
                    # it has several dimensions:
                    for pos in xrange(prod(pt[i].shape)):
                        t = pt[i].copy()
                        t = t.flatten()
                        t[pos] += _eps
                        t = t.reshape(pt[i].shape)
                        f_eps = f(*(pt[:i]+[t]+pt[i+1:]))
                        _g.append(numpy.asarray((f_eps - f_x)/_eps))
                    gx.append(numpy.asarray(_g).reshape(pt[i].shape))
                else:
                    t= numpy.array(pt[i] + _eps)
                    f_eps = f(*(pt[:i]+[t]+pt[i+1:]))
                    gx.append(numpy.asarray((f_eps-f_x)/_eps))
        self.gx = gx

    @staticmethod
    def abs_rel_err(a,b,eps=1.0e-10):
        """Return a small number when a and b are close, relative to how big they are"""
        return abs(a-b) / (abs(a)+abs(b)+eps)

    def max_err(self, g_pt):
        """Return the biggest relative error between g_pt and self.gx"""
        if len(g_pt) != len(self.gx):
            raise ValueError('argument has wrong number of elements', len(g_pt))
        errs = []
        for i, (a, b) in enumerate(zip(g_pt, self.gx)):
            if a.shape != b.shape:
                raise ValueError('argument element %i has wrong shape %s' %(i,str((a.shape,
                    b.shape))))
            vv = multiple_outputs_numeric_grad.abs_rel_err(a,b)
            errs.append(numpy.max(multiple_outputs_numeric_grad.abs_rel_err(a,b)))
        if numpy.all(numpy.isfinite(errs)):
            return numpy.max(errs), numpy.argmax(errs)
        else:
            return float('inf'), 0



#TODO: Test this function, and if it works,
# use it with the normal verify_grad rather than the
# copy-and-pasted one above.
# Also - add a reference to this technique in the
# verify_grad method so that other ops with multiple outputs can be tested. DONE - rp
def scan_project_sum(*args, **kwargs):
    rng = theano.tensor.shared_randomstreams.RandomStreams(123)
    scan_outputs, updates = theano.scan(*args, **kwargs)
    if type(scan_outputs) not in [list,tuple]:
        scan_outputs = [scan_outputs]
    # we should ignore the random-state updates so that
    # the uniform numbers are the same every evaluation and on every call
    rng.add_default_updates = False
    factors = [ rng.uniform(size=s.shape, low = 0.1, high = 0.9) for s in scan_outputs ]
    # Random values (?)
    return (sum([(s*f).sum() for s,f in zip(scan_outputs,factors)]),updates)

def asarrayX(value):
    return theano._asarray(value, dtype=theano.config.floatX)



class T_Scan(unittest.TestCase):

    def setUp(self):
        utt.seed_rng()

    # generator network, only one output , type scalar ; no sequence or
    # non sequence arguments
    def test_generator_one_output_scalar(self):
        def f_pow2(x_tm1):
            return 2*x_tm1

        state = theano.tensor.scalar()
        n_steps = theano.tensor.scalar()
        output, updates = theano.scan(f_pow2, [],state, [],n_steps = n_steps, truncate_gradient
                = -1, go_backwards = False)
        my_f = theano.function([state,n_steps], output, updates = updates)

        rng = numpy.random.RandomState(utt.fetch_seed())
        state = asarrayX(rng.uniform())
        steps = 5

        numpy_values = numpy.array([ state*(2**(k+1)) for k in xrange(steps) ])
        theano_values = my_f(state,steps)
        assert numpy.allclose(numpy_values,theano_values)


    # simple rnn, one input, one state, weights for each; input/state
    # are vectors, weights are scalars
    def test_one_sequence_one_output_weights(self):
        def f_rnn(u_t,x_tm1,W_in, W):
            return u_t*W_in+x_tm1*W

        u    = theano.tensor.vector()
        x0   = theano.tensor.scalar()
        W_in = theano.tensor.scalar()
        W    = theano.tensor.scalar()

        output, updates = theano.scan(f_rnn, u,x0,[W_in,W], n_steps = None, truncate_gradient =
                -1, go_backwards = False)

        f2   = theano.function([u,x0,W_in,W], output, updates = updates)
        # get random initial values
        rng  = numpy.random.RandomState(utt.fetch_seed())
        v_u  = asarrayX(rng.uniform(size = (4,), low = -5., high = 5.))
        v_x0 = asarrayX(rng.uniform())
        W    = asarrayX(rng.uniform())
        W_in = asarrayX(rng.uniform())

        # compute the output in numpy
        v_out = numpy.zeros((4,))
        v_out[0] = v_u[0]*W_in + v_x0 * W
        for step in xrange(1,4):
            v_out[step] = v_u[step]*W_in + v_out[step-1] * W

        theano_values = f2(v_u,v_x0, W_in, W)
        assert numpy.allclose(theano_values, v_out)


    # simple rnn, one input, one state, weights for each; input/state
    # are vectors, weights are scalars; using shared variables
    def test_one_sequence_one_output_weights_shared(self):
        rng   = numpy.random.RandomState(utt.fetch_seed())
        u    = theano.tensor.vector()
        x0   = theano.tensor.scalar()
        W_in = theano.shared(asarrayX(rng.uniform()), name = 'w_in')
        W    = theano.shared(asarrayX(rng.uniform()), name ='w')

        def f_rnn_shared(u_t,x_tm1, tmp_W_in, tmp_W):
            return u_t*tmp_W_in+x_tm1*tmp_W

        output, updates = theano.scan(f_rnn_shared, u,x0,[W_in, W], n_steps =None,
                truncate_gradient= -1, go_backwards = False)
        f3    = theano.function([u,x0], output, updates = updates)
        # get random initial values

        v_u   = asarrayX(rng.uniform(size = (4,), low = -5., high = 5.))
        v_x0  = asarrayX(rng.uniform())
        # compute the output i numpy
        v_out = numpy.zeros((4,))
        v_out[0] = v_u[0]*W_in.value + v_x0*W.value
        for step in xrange(1,4):
            v_out[step] = v_u[step]*W_in.value + v_out[step-1]*W.value

        theano_values = f3(v_u, v_x0)
        assert  numpy.allclose(theano_values, v_out)


    # some rnn with multiple outputs and multiple inputs; other
    # dimension instead of scalars/vectors
    def test_multiple_inputs_multiple_outputs(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        vW_in2 = asarrayX(rng.uniform(size = (2,), low = -5.,high = 5.))
        vW     = asarrayX(rng.uniform(size = (2,2), low = -5.,high = 5.))
        vWout  = asarrayX(rng.uniform(size = (2,), low = -5.,high = 5.))
        vW_in1 = asarrayX(rng.uniform(size = (2,2), low = -5.,high = 5.))
        v_u1   = asarrayX(rng.uniform(size = (3,2), low = -5., high = 5.))
        v_u2   = asarrayX(rng.uniform(size = (3,), low = -5.,high = 5.))
        v_x0   = asarrayX(rng.uniform(size = (2,), low = -5.,high = 5.))
        v_y0   = asarrayX(rng.uniform())

        W_in2 = theano.shared(vW_in2, name='win2')
        W     = theano.shared(vW, name='w')
        W_out = theano.shared(vWout, name = 'wout')
        W_in1 = theano.tensor.matrix('win')
        u1    = theano.tensor.matrix('u1')
        u2    = theano.tensor.vector('u2')
        x0    = theano.tensor.vector('x0')
        y0    = theano.tensor.scalar('y0')

        def f_rnn_cmpl(u1_t, u2_t, x_tm1, y_tm1, W_in1):
            return [theano.dot(u1_t,W_in1) + u2_t* W_in2 + \
                    theano.dot(x_tm1, W), theano.dot(x_tm1, W_out)]

        outputs, updates = theano.scan(f_rnn_cmpl,[u1,u2],[x0,y0],W_in1, n_steps = None,
                truncate_gradient = -1, go_backwards = False)

        f4     = theano.function([u1,u2,x0,y0,W_in1], outputs, updates = updates)
        # compute the values in numpy
        v_x = numpy.zeros((3,2),dtype=theano.config.floatX)
        v_y = numpy.zeros((3,),dtype=theano.config.floatX)
        v_x[0] = numpy.dot(v_u1[0],vW_in1) + v_u2[0]*vW_in2 + numpy.dot(v_x0,vW)
        v_y[0] = numpy.dot(v_x0,vWout)
        for i in xrange(1,3):
            v_x[i] = numpy.dot(v_u1[i],vW_in1) + v_u2[i]*vW_in2 + numpy.dot(v_x[i-1],vW)
            v_y[i] = numpy.dot(v_x[i-1], vWout)

        (theano_x,theano_y) =  f4( v_u1, v_u2, v_x0, v_y0, vW_in1)

        assert numpy.allclose(theano_x , v_x)
        assert numpy.allclose(theano_y , v_y)


    # simple rnn, one input, one state, weights for each; input/state are
    # vectors, weights are scalars; using shared variables and past
    # taps (sequences and outputs)
    def test_using_taps_input_output(self):
        rng   = numpy.random.RandomState(utt.fetch_seed())
        vW    = asarrayX(rng.uniform())
        vW_in = asarrayX(rng.uniform())
        vu    = asarrayX(rng.uniform(size=(4,), low = -5., high = 5.))
        vx0   = asarrayX(rng.uniform(size=(2,), low = -5., high = 5.))

        u    = theano.tensor.vector()
        x0   = theano.tensor.vector()
        W_in = theano.shared(vW_in, name = 'w_in')
        W    = theano.shared(vW, name ='w')

        def f_rnn_shared(u_tm2, x_tm1, x_tm2):
            return u_tm2*W_in+x_tm1*W+x_tm2

        outputs, updates = theano.scan(f_rnn_shared, dict(input=u, taps=-2),
                dict(initial = x0, taps = [-1,-2]), [], n_steps = None, truncate_gradient = -1,
                go_backwards = False)

        f7   = theano.function([u,x0], outputs, updates = updates)
        theano_out = f7(vu,vx0)

        # compute output in numpy
        # a bit of explaining:
        # due to the definition of sequences taps in scan, v_0[0] is actually v_0[-2],
        # and v_0[1] is v_0[-1]. The values v_0[2] and v_0[3] do not get uesd ( because you
        # do not use v_0[t] in scan) which might seem strange, but then again why not use
        # v_0[t] instead of v_0[t-2] in a real application ??
        # also vx0[0] corresponds to vx0[-2], vx0[1] to vx0[-1]
        numpy_out = numpy.zeros((2,))
        numpy_out[0] = vu[0]*vW_in + vx0[1]*vW + vx0[0]
        numpy_out[1] = vu[1]*vW_in + numpy_out[0]*vW + vx0[1]

        assert numpy.allclose(numpy_out , theano_out)



    # simple rnn, one input, one state, weights for each; input/state are
    # vectors, weights are scalars; using shared variables and past
    # taps (sequences and outputs) and future taps for sequences
    def test_past_future_taps_shared(self):
        rng   = numpy.random.RandomState(utt.fetch_seed())
        vW    = asarrayX(rng.uniform())
        vW_in = asarrayX(rng.uniform())
        vu    = asarrayX(rng.uniform(size=(6,), low = -5., high = 5.))
        vx0   = asarrayX(rng.uniform(size=(2,), low = -5., high = 5.))

        u    = theano.tensor.vector()
        x0   = theano.tensor.vector()
        W_in = theano.shared(vW_in, name = 'w_in')
        W    = theano.shared(vW, name ='w')

        def f_rnn_shared(u_tm2,u_tp2, x_tm1, x_tm2):
            return (u_tm2+u_tp2)*W_in+x_tm1*W+x_tm2

        output,updates = theano.scan(f_rnn_shared, dict( input = u, taps=[-2,2]),\
                dict(initial = x0, taps = [-1,-2]), [], n_steps = None, truncate_gradient =-1,
                go_backwards = False)

        f8   = theano.function([u,x0], output, updates = updates)
        theano_out = f8(vu,vx0)
        # compute output in numpy
        numpy_out = numpy.zeros(2)
        # think of vu[0] as vu[-2], vu[4] as vu[2]
        # and vx0[0] as vx0[-2], vx0[1] as vx0[-1]
        numpy_out[0] = (vu[0]+vu[4])*vW_in + vx0[1]*vW + vx0[0]
        numpy_out[1] = (vu[1]+vu[5])*vW_in + numpy_out[0]*vW + vx0[1]

        assert numpy.allclose(numpy_out , theano_out)


    # simple rnn ; compute inplace version 1
    def test_inplace1(self):
        rng   = numpy.random.RandomState(utt.fetch_seed())
        vW    = asarrayX(numpy.random.uniform())
        vW_in = asarrayX(numpy.random.uniform())
        vu0   = asarrayX(rng.uniform(size=(3,), low = -5., high = 5.))
        vu1   = asarrayX(rng.uniform(size=(3,), low = -5., high = 5.))
        vu2   = asarrayX(rng.uniform(size=(3,), low = -5., high = 5.))
        vx0   = asarrayX(rng.uniform())
        vx1   = asarrayX(rng.uniform())

        u0   = theano.tensor.vector('u0')
        u1   = theano.tensor.vector('u1')
        u2   = theano.tensor.vector('u2')
        mu0  = theano.Param( u0, mutable = False)
        mu1  = theano.Param( u1, mutable = True)
        mu2  = theano.Param( u2, mutable = True)
        x0   = theano.tensor.scalar('x0')
        x1   = theano.tensor.scalar('y0')
        W_in = theano.shared(vW_in,'Win')
        W    = theano.shared(vW,'W')
        mode = theano.compile.mode.get_mode(None).including('inplace')
        def f_rnn_shared(u0_t,u1_t, u2_t, x0_tm1,x1_tm1):
            return [u0_t*W_in + x0_tm1*W + u1_t*u2_t, u0_t*W_in + x1_tm1*W+ u1_t+u2_t ]

        outputs, updates = theano.scan(f_rnn_shared, [u0,u1,u2],
                [dict( initial = x0, inplace =u2), dict(initial = x1, inplace = u1)],
                [], n_steps = None, truncate_gradient = -1, go_backwards = False, mode=mode )
        f9   = theano.function([mu0,mu1,mu2,x0,x1], outputs , updates = updates, mode = mode)

       # compute output in numpy
        numpy_x0 = numpy.zeros((3,))
        numpy_x1 = numpy.zeros((3,))
        numpy_x0[0] = vu0[0] * vW_in + vx0 * vW + vu1[0]*vu2[0]
        numpy_x1[0] = vu0[0] * vW_in + vx1 * vW + vu1[0]+vu2[0]
        for i in xrange(1,3):
            numpy_x0[i] = vu0[i]* vW_in + numpy_x0[i-1]*vW + vu1[i]*vu2[i]
            numpy_x1[i] = vu0[i]* vW_in + numpy_x1[i-1]*vW + vu1[i]+vu2[i]

        # note theano computes inplace, so call function after numpy equivalent is done
        (theano_x0, theano_x1) = f9(vu0,vu1,vu2,vx0,vx1)
        # assert that theano does what it should
        assert numpy.allclose( theano_x0 , numpy_x0)
        assert numpy.allclose( theano_x1 , numpy_x1)
        # assert that it was done in place
        assert numpy.allclose( theano_x0 , vu2)
        assert numpy.allclose( theano_x1 , vu1)

    # simple rnn ; compute inplace version 2
    def test_inplace2(self):
        rng   = numpy.random.RandomState(utt.fetch_seed())
        vW    = asarrayX(numpy.random.uniform())
        vW_in = asarrayX(numpy.random.uniform())
        vu0   = asarrayX(rng.uniform(size=(3,), low = -5., high = 5.))
        vu1   = asarrayX(rng.uniform(size=(4,), low = -5., high = 5.))
        vu2   = asarrayX(rng.uniform(size=(5,), low = -5., high = 5.))
        vx0   = asarrayX(rng.uniform())
        vx1   = asarrayX(rng.uniform())

        u0   = theano.tensor.vector('u0')
        u1   = theano.tensor.vector('u1')
        u2   = theano.tensor.vector('u2')
        mu0  = theano.Param( u0, mutable = True)
        mu1  = theano.Param( u1, mutable = True)
        mu2  = theano.Param( u2, mutable = True)
        x0   = theano.tensor.scalar('x0')
        x1   = theano.tensor.scalar('y0')
        W_in = theano.shared(vW_in,'Win')
        W    = theano.shared(vW,'W')
        mode = theano.compile.mode.get_mode(None).including('inplace')
        def f_rnn_shared(u0_t,u1_t,u1_tp1, u2_tm1,u2_t,u2_tp1, x0_tm1,x1_tm1):
            return [u0_t*W_in + x0_tm1*W + u1_t*u1_tp1, \
                    u0_t*W_in + x1_tm1*W+ u2_tm1+u2_t+u2_tp1 ]

        outputs, updates = theano.scan(f_rnn_shared,
                [u0,dict(input = u1, taps = [0,1]),dict( input = u2, taps= [-1,0,+1])],
                [dict( initial = x0, inplace =u2), dict(initial = x1, inplace = u1)],
                [], n_steps = None, truncate_gradient = -1, go_backwards = False, mode=mode )
        f9   = theano.function([mu0,mu1,mu2,x0,x1], outputs , updates = updates, mode = mode)

       # compute output in numpy
        numpy_x0 = numpy.zeros((3,))
        numpy_x1 = numpy.zeros((3,))
        numpy_x0[0] = vu0[0] * vW_in + vx0 * vW + vu1[0]*vu1[1]
        numpy_x1[0] = vu0[0] * vW_in + vx1 * vW + vu2[0]+vu2[1]+vu2[2]
        for i in xrange(1,3):
            numpy_x0[i] = vu0[i]* vW_in + numpy_x0[i-1]*vW + vu1[i]*vu1[i+1]
            numpy_x1[i] = vu0[i]* vW_in + numpy_x1[i-1]*vW + vu2[i]+vu2[i+1]+vu2[i+2]

        # note theano computes inplace, so call function after numpy equivalent is done
        (theano_x0, theano_x1) = f9(vu0,vu1,vu2,vx0,vx1)
        # assert that theano does what it should
        assert numpy.allclose( theano_x0 , numpy_x0)
        assert numpy.allclose( theano_x1 , numpy_x1)
        # assert that it was done in place
        # not that x0 should not be inplace of vu2 because you are using past values of u2,
        # and therefore you are not allowed to work inplace !!
        assert not numpy.allclose( theano_x0 , vu2[1:4])
        assert numpy.allclose( theano_x1 , vu1[0:3])



    # Shared variable with updates
    def test_shared_arguments_with_updates(self):
        rng = numpy.random.RandomState(utt.fetch_seed())

        vW1 = asarrayX(rng.rand(20,30))
        vW2 = asarrayX(rng.rand(30,20))
        vu1 = asarrayX(rng.rand(3,20))
        vu2 = asarrayX(rng.rand(3,30))
        vy0 = asarrayX(rng.rand(3,20))
        vy1 = asarrayX(rng.rand(20))
        vy2 = asarrayX(rng.rand(30))

        # Their is a bug when floatX=float32 when we remove this line.
        # The trace back is:
#Traceback (most recent call last):
#  File "/u/bastienf/repos/Theano/theano/tests/test_scan.py", line 434, in test_shared_arguments_with_updates
#    theano_y0,theano_y1,theano_y2 = f10(vu2, vy0)
#  File "/u/bastienf/repos/theano/compile/function_module.py", line 480, in __call__
#    self.fn()
#  File "/u/bastienf/repos/theano/compile/profilemode.py", line 59, in profile_f
#    raise_with_op(node)
#  File "/u/bastienf/repos/theano/compile/profilemode.py", line 52, in profile_f
#    th()
#  File "/u/bastienf/repos/theano/gof/cc.py", line 1141, in <lambda>
#    thunk = lambda p = p, i = node_input_storage, o = node_output_storage, n = node: p(n, [x[0] for x in i], o)
#  File "/u/bastienf/repos/theano/scan.py", line 922, in perform
#    inplace_map)
#  File "/u/bastienf/repos/theano/scan.py", line 1054, in scan
#    something = fn(*fn_args)
#  File "/u/bastienf/repos/theano/compile/function_module.py", line 458, in __call__
#    s.storage[0] = s.type.filter(arg, strict=s.strict)
#  File "/u/bastienf/repos/theano/tensor/basic.py", line 415, in filter
#    data = theano._asarray(data, dtype = self.dtype) #TODO - consider to pad shape with ones
#  File "/u/bastienf/repos/theano/misc/safe_asarray.py", line 30, in _asarray
#    rval = numpy.asarray(a, dtype=dtype, order=order)
#  File "/u/lisa/local/byhost/ceylon.iro.umontreal.ca//lib64/python2.5/site-packages/numpy/core/numeric.py", line 230, in asarray
#    return array(a, dtype, copy=False, order=order)
#TypeError: ('__array__() takes no arguments (1 given)', <theano.scan.Scan object at 0x3dbbf90>(?_steps, u1, u2, y0, y1, 0.0, W1, W2), 'Sequence id of Apply node=0')
#
#  This don't seam to be a theano related bug...
        #vu1 = rng.rand(3,20)
        vu1 = asarrayX(rng.rand(3,20))

        W1 = theano.shared(vW1,'W1')
        W2 = theano.shared(vW2,'W2')
        u1 = theano.shared(vu1,'u1')
        y1 = theano.shared(vy1,'y1')

        def f(u1_t, u2_t, y0_tm3, y0_tm2, y0_tm1, y1_tm1):
            y0_t = theano.dot(theano.dot(u1_t,W1),W2) + 0.1*y0_tm1 + \
                    0.33*y0_tm2 + 0.17*y0_tm3
            y1_t = theano.dot(u2_t, W2) + y1_tm1
            y2_t = theano.dot(u1_t, W1)
            nwW1 = W1 + .1
            nwW2 = W2 + .05
            # return outputs followed by a list of updates
            return ([y0_t, y1_t, y2_t], [( W1,nwW1), (W2, nwW2)])

        u2 = theano.tensor.matrix('u2')
        y0 = theano.tensor.matrix('y0')

        outputs,updates = theano.scan(f, [u1,u2], [ dict(initial = y0, taps = [-3,-2,-1]),y1,
            None], [], n_steps = None, go_backwards = False, truncate_gradient = -1)
        f10 = theano.function([u2,y0], outputs, updates = updates)
        theano_y0,theano_y1,theano_y2 = f10(vu2, vy0)

        # do things in numpy
        numpy_y0 = numpy.zeros((6,20))
        numpy_y1 = numpy.zeros((4,20))
        numpy_y2 = numpy.zeros((3,30))
        numpy_y0[:3] = vy0
        numpy_y1[0]  = vy1
        numpy_W1     = vW1.copy()
        numpy_W2    = vW2.copy()
        for idx in xrange(3):
            numpy_y0[idx+3] = numpy.dot( numpy.dot(vu1[idx,:], numpy_W1), numpy_W2) + \
                    0.1*numpy_y0[idx+2] + 0.33*numpy_y0[idx+1] + 0.17*numpy_y0[idx]
            numpy_y1[idx+1] = numpy.dot( vu2[idx,:], numpy_W2) + numpy_y1[idx]
            numpy_y2[idx]   = numpy.dot( vu1[idx,:], numpy_W1)
            numpy_W1 = numpy_W1 + .1
            numpy_W2 = numpy_W2 + .05

        assert numpy.allclose( theano_y0 , numpy_y0[3:])
        assert numpy.allclose( theano_y1 , numpy_y1[1:])
        assert numpy.allclose( theano_y2 , numpy_y2    )
        assert numpy.allclose( W1.value  , numpy_W1    )
        assert numpy.allclose( W2.value  , numpy_W2    )



    def test_simple_shared_random(self):

        theano_rng = theano.tensor.shared_randomstreams.RandomStreams(utt.fetch_seed())

        values, updates = theano.scan(lambda : theano_rng.uniform((2,),-1,1), [],[],[],n_steps
                = 5, truncate_gradient = -1, go_backwards = False)
        my_f = theano.function([], values, updates = updates )

        rng_seed = numpy.random.RandomState(utt.fetch_seed()).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed)) #int() is for 32bit

        numpy_v = numpy.zeros((10,2))
        for i in xrange(10):
            numpy_v[i] = rng.uniform(-1,1,size = (2,))

        theano_v = my_f()
        assert numpy.allclose( theano_v , numpy_v [:5,:])
        theano_v = my_f()
        assert numpy.allclose( theano_v , numpy_v[5:,:])



    def test_gibbs_chain(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        v_W       = numpy.array(rng.rand(20,30) -.5, dtype = 'float32')
        v_vsample = numpy.array(rng.binomial(1,0.5, size=(3,20), ), dtype = 'float32')
        v_bvis    = numpy.array(rng.rand(20) -.5, dtype='float32')
        v_bhid    = numpy.array(rng.rand(30) -.5, dtype='float32')
        W       = theano.shared(v_W)
        bhid    = theano.shared(v_bhid)
        bvis    = theano.shared(v_bvis)
        vsample = theano.tensor.matrix(dtype='float32')
        trng = theano.tensor.shared_randomstreams.RandomStreams(utt.fetch_seed())

        def f(vsample_tm1):
            hmean_t   = theano.tensor.nnet.sigmoid(theano.dot(vsample_tm1,W)+ bhid)
            hsample_t = theano.tensor.cast(trng.binomial(hmean_t.shape,1,hmean_t),dtype='float32')
            vmean_t   = theano.tensor.nnet.sigmoid(theano.dot(hsample_t,W.T)+ bvis)
            return theano.tensor.cast(trng.binomial(vmean_t.shape,1,vmean_t), dtype='float32')

        theano_vsamples, updates = theano.scan(f, [], vsample,[], n_steps = 10,
                truncate_gradient=-1, go_backwards = False)
        my_f = theano.function([vsample], theano_vsamples[-1], updates = updates)

        _rng = numpy.random.RandomState(utt.fetch_seed())
        rng_seed = _rng.randint(2**30)
        nrng1 = numpy.random.RandomState(int(rng_seed)) # int() is for 32bit

        rng_seed = _rng.randint(2**30)
        nrng2 = numpy.random.RandomState(int(rng_seed)) # int() is for 32bit
        def numpy_implementation(vsample):
            for idx in range(10):
                hmean = 1./(1. + numpy.exp(-(numpy.dot(vsample,v_W) + v_bhid)))
                hsample = numpy.array(nrng1.binomial(1,hmean, size = hmean.shape), dtype='float32')
                vmean  = 1./(1. + numpy.exp(-(numpy.dot(hsample,v_W.T) + v_bvis)))
                vsample = numpy.array(nrng2.binomial(1,vmean, size = vmean.shape),dtype='float32')

            return vsample

        t_result = my_f(v_vsample)
        n_result = numpy_implementation(v_vsample)

        assert numpy.allclose( t_result , n_result)


    def test_only_shared_no_input_no_output(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        v_state = asarrayX(rng.uniform())
        state = theano.shared(v_state)
        def f_2():
            return {state: 2*state}
        n_steps = theano.tensor.scalar()
        output, updates = theano.scan(f_2,[],[],[],n_steps = n_steps, truncate_gradient = -1,
                go_backwards = False)
        this_f = theano.function([n_steps], output, updates = updates)
        n_steps = 3
        this_f(n_steps)
        numpy_state = v_state* (2**(n_steps))
        assert numpy.allclose(state.value, numpy_state)

    def test_map_functionality(self):
        def f_rnn(u_t):
            return u_t + 3

        u    = theano.tensor.vector()

        outputs, updates = theano.scan(f_rnn, u,[],[], n_steps =None , truncate_gradient = -1,
                go_backwards = False)

        f2    = theano.function([u], outputs, updates = updates)
        rng = numpy.random.RandomState(utt.fetch_seed())

        v_u   = rng.uniform(size=(5,), low = -5., high = 5.).astype(config.floatX)
        numpy_result = v_u + 3
        theano_result = f2(v_u)
        assert numpy.allclose(theano_result , numpy_result)


    def test_map(self):
        v = theano.tensor.vector()
        abs_expr,abs_updates = theano.map(lambda x: abs(x), v,[],
                truncate_gradient = -1, go_backwards = False)
        f = theano.function([v],abs_expr,updates = abs_updates)

        rng = numpy.random.RandomState(utt.fetch_seed())
        vals = rng.uniform(size=(10,), low = -5., high = 5.).astype(config.floatX)
        abs_vals = abs(vals)
        theano_vals = f(vals)
        assert numpy.allclose(abs_vals , theano_vals)

    def test_backwards(self):
        def f_rnn(u_t,x_tm1,W_in, W):
            return u_t*W_in+x_tm1*W

        u    = theano.tensor.vector()
        x0   = theano.tensor.scalar()
        W_in = theano.tensor.scalar()
        W    = theano.tensor.scalar()

        output, updates = theano.scan(f_rnn, u,x0,[W_in,W], n_steps = None, truncate_gradient =
                -1, go_backwards = True)

        f2   = theano.function([u,x0,W_in,W], output, updates = updates)
        # get random initial values
        rng  = numpy.random.RandomState(utt.fetch_seed())
        v_u  = asarrayX(rng.uniform(size=(4,), low=-5., high=5.))
        v_x0 = asarrayX(rng.uniform())
        W    = asarrayX(rng.uniform())
        W_in = asarrayX(rng.uniform())

        # compute the output in numpy
        v_out = numpy.zeros((4,))
        v_out[0] = v_u[3]*W_in + v_x0 * W
        for step in xrange(1,4):
            v_out[step] = v_u[3-step]*W_in + v_out[step-1] * W

        theano_values = f2(v_u,v_x0, W_in, W)
        assert numpy.allclose( theano_values , v_out)

    def test_reduce(self):
        v = theano.tensor.vector()
        s = theano.tensor.scalar()
        result, updates = theano.reduce(lambda x,y: x+y, v,s)

        f = theano.function([v,s], result, updates = updates)
        rng = numpy.random.RandomState(utt.fetch_seed())
        v_v = rng.uniform(size = (5,), low = -5., high = 5.).astype(config.floatX)
        assert abs(numpy.sum(v_v) - f(v_v, 0.)) < 1e-3


    def test_grad_one_output(self):
        def f_rnn(u_t,x_tm1,W_in, W):
            return u_t*W_in+x_tm1*W

        u    = theano.tensor.vector('u')
        x0   = theano.tensor.scalar('x0')
        W_in = theano.tensor.scalar('W_in')
        W    = theano.tensor.scalar('W')

        cost, updates = scan_project_sum(f_rnn, u, x0, [W_in,W], n_steps = None,
                truncate_gradient = -1, go_backwards = False)
        gu,gx0,gW_in,gW = theano.tensor.grad(cost, [u,x0,W_in, W])
        grad_fn = theano.function([u,x0,W_in, W], [gu,gx0,gW_in, gW],
                updates = updates, no_default_updates = True)
        cost_fn = theano.function([u,x0,W_in, W], cost, updates = updates,
                no_default_updates = True, allow_input_downcast=True)

        # get random initial values
        rng  = numpy.random.RandomState(utt.fetch_seed())
        v_u  = numpy.array(rng.uniform( size = (10,), low = -.5, high = .5),dtype=theano.config.floatX)
        v_x0 = numpy.array(rng.uniform(), dtype= theano.config.floatX)
        W    = numpy.array(rng.uniform(), dtype= theano.config.floatX)
        W_in = numpy.array(rng.uniform(), dtype= theano.config.floatX)
        num_grad = multiple_outputs_numeric_grad(cost_fn, [v_u, v_x0, W_in, W])
        analytic_grad = grad_fn(v_u, v_x0, W_in, W)
        max_err, max_err_pos = num_grad.max_err(analytic_grad)

        if max_err > 1e-2:
            raise Exception(theano.tensor.verify_grad.E_grad,
                    (max_err, 1e-2, max_err_pos))


    def test_grad_multiple_outs(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        vW_in2 = asarrayX(rng.uniform(size = (2,), low = -.1,high = .1))
        vW     = asarrayX(rng.uniform(size = (2,2), low = -.1,high = .1))
        vWout  = asarrayX(rng.uniform(size = (2,), low = -.1,high = .1))
        vW_in1 = asarrayX(rng.uniform(size = (2,2), low = -.1,high = .1))
        v_u1   = asarrayX(rng.uniform(size = (7,2), low = -.1, high = .1))
        v_u2   = asarrayX(rng.uniform(size = (7,), low = -.1,high = .1))
        v_x0   = asarrayX(rng.uniform(size = (2,), low = -.1,high = .1))
        v_y0   = asarrayX(rng.uniform())

        W_in2 = theano.shared(vW_in2, name='win2')
        W     = theano.shared(vW, name='w')
        W_out = theano.shared(vWout, name = 'wout')
        W_in1 = theano.tensor.matrix('win')
        u1    = theano.tensor.matrix('u1')
        u2    = theano.tensor.vector('u2')
        x0    = theano.tensor.vector('x0')
        y0    = theano.tensor.scalar('y0')

        def f_rnn_cmpl(u1_t, u2_t, x_tm1, y_tm1, W_in1):
            return [theano.dot(u1_t,W_in1) + u2_t* W_in2 + \
                    theano.dot(x_tm1, W), theano.dot(x_tm1, W_out)]

        cost, updates = scan_project_sum(f_rnn_cmpl,[u1,u2],[x0,y0],W_in1, n_steps = None,
                truncate_gradient = -1, go_backwards = False)
        vparams = [v_u1, v_u2, v_x0, v_y0,vW_in1]
        params = [u1,u2,x0,y0,W_in1 ]
        gparams = theano.tensor.grad(cost, params)
        grad_fn = theano.function([u1,u2,x0,y0,W_in1], gparams,
                updates = updates, no_default_updates = True)
        cost_fn = theano.function([u1,u2,x0,y0,W_in1], cost,
                updates = updates, no_default_updates = True,
                allow_input_downcast=True)

        num_grad = multiple_outputs_numeric_grad(cost_fn,[v_u1,v_u2,v_x0,v_y0,vW_in1])
        analytic_grad = grad_fn(v_u1,v_u2, v_x0,v_y0, vW_in1)
        max_err, max_err_pos = num_grad.max_err(analytic_grad)

        if max_err > 1e-2:
            raise Exception(theano.tensor.verify_grad.E_grad,
                    (max_err, 1e-2, max_err_pos))


    def test_grad_multiple_outs_taps(self):
        l = 5
        rng = numpy.random.RandomState(utt.fetch_seed())
        vW_in2 = asarrayX(rng.uniform(size = (2,), low = -.2,high = .2))
        vW     = asarrayX(rng.uniform(size = (2,2), low = -.2,high = .2))
        vWout  = asarrayX(rng.uniform(size = (2,), low = -.2,high = .2))
        vW_in1 = asarrayX(rng.uniform(size = (2,2), low = -.2,high = .2))
        v_u1   = asarrayX(rng.uniform(size = (l,2), low = -.2, high = .2))
        v_u2   = asarrayX(rng.uniform(size = (l+2,2), low = -.2,high = .2))
        v_x0   = asarrayX(rng.uniform(size = (2,), low = -.2,high = .2))
        v_y0   = asarrayX(rng.uniform(size = (4,)))

        W_in2 = theano.shared(vW_in2, name='win2')
        W     = theano.shared(vW, name='w')
        W_out = theano.shared(vWout, name = 'wout')
        W_in1 = theano.tensor.matrix('win')
        u1    = theano.tensor.matrix('u1')
        u2    = theano.tensor.matrix('u2')
        x0    = theano.tensor.vector('x0')
        y0    = theano.tensor.vector('y0')

        def f_rnn_cmpl(u1_t, u2_tm1, u2_t, u2_tp1, x_tm1, y_tm1, y_tm3, W_in1):
            return [theano.dot(u1_t,W_in1) + (u2_t+u2_tm1*u2_tp1)* W_in2 + \
                    theano.dot(x_tm1, W), (y_tm1+y_tm3)*theano.dot(x_tm1, W_out)]
        cost, updates = scan_project_sum(f_rnn_cmpl,[u1,
            dict(input=u2,taps=[-1,0,1])],[x0,dict(initial=y0,
                taps=[-1,-3])],W_in1, n_steps = None,
                truncate_gradient = -1, go_backwards = False)
        vparams = [v_u1, v_u2, v_x0, v_y0,vW_in1]
        params = [u1,u2,x0,y0,W_in1 ]
        gparams = theano.tensor.grad(cost, params)
        grad_fn = theano.function([u1,u2,x0,y0,W_in1], gparams,
                updates = updates, no_default_updates = True)
        cost_fn = theano.function([u1,u2,x0,y0,W_in1], cost,
                updates = updates, no_default_updates = True)

        num_grad = multiple_outputs_numeric_grad(cost_fn,[v_u1,v_u2,v_x0,v_y0,vW_in1])
        analytic_grad = grad_fn(v_u1,v_u2, v_x0,v_y0, vW_in1)
        max_err, max_err_pos = num_grad.max_err(analytic_grad)
        if max_err > 1e-2:
            raise Exception(theano.tensor.verify_grad.E_grad,
                    (max_err, 1e-2, max_err_pos))

    def test_grad_multiple_outs_taps_backwards(self):
        l = 5
        rng = numpy.random.RandomState(utt.fetch_seed())
        vW_in2 = asarrayX(rng.uniform(size = (2,), low = -.2,high = .2))
        vW     = asarrayX(rng.uniform(size = (2,2), low = -.2,high = .2))
        vWout  = asarrayX(rng.uniform(size = (2,), low = -.2,high = .2))
        vW_in1 = asarrayX(rng.uniform(size = (2,2), low = -.2,high = .2))
        v_u1   = asarrayX(rng.uniform(size = (l,2), low = -.2, high = .2))
        v_u2   = asarrayX(rng.uniform(size = (l+2,2), low = -.2,high = .2))
        v_x0   = asarrayX(rng.uniform(size = (2,), low = -.2,high = .2))
        v_y0   = asarrayX(rng.uniform(size = (4,)))

        W_in2 = theano.shared(vW_in2, name='win2')
        W     = theano.shared(vW, name='w')
        W_out = theano.shared(vWout, name = 'wout')
        W_in1 = theano.tensor.matrix('win')
        u1    = theano.tensor.matrix('u1')
        u2    = theano.tensor.matrix('u2')
        x0    = theano.tensor.vector('x0')
        y0    = theano.tensor.vector('y0')

        def f_rnn_cmpl(u1_t, u2_tm1, u2_t, u2_tp1, x_tm1, y_tm1, y_tm3, W_in1):
            return [theano.dot(u1_t,W_in1) + (u2_t+u2_tm1*u2_tp1)* W_in2 + \
                    theano.dot(x_tm1, W), (y_tm1+y_tm3)*theano.dot(x_tm1, W_out)]
        cost, updates = scan_project_sum(f_rnn_cmpl,[u1,
            dict(input=u2,taps=[-1,0,1])],[x0,dict(initial=y0,
                taps=[-1,-3])],W_in1, n_steps = None,
                truncate_gradient = -1, go_backwards = True)
        vparams = [v_u1, v_u2, v_x0, v_y0,vW_in1]
        params = [u1,u2,x0,y0,W_in1 ]
        gparams = theano.tensor.grad(cost, params)
        grad_fn = theano.function([u1,u2,x0,y0,W_in1], gparams,
                updates = updates, no_default_updates = True)
        cost_fn = theano.function([u1,u2,x0,y0,W_in1], cost,
                updates = updates, no_default_updates = True)

        num_grad = multiple_outputs_numeric_grad(cost_fn,[v_u1,v_u2,v_x0,v_y0,vW_in1])
        analytic_grad = grad_fn(v_u1,v_u2, v_x0,v_y0, vW_in1)
        max_err, max_err_pos = num_grad.max_err(analytic_grad)
        if max_err > 1e-2:
            raise Exception(theano.tensor.verify_grad.E_grad,
                    (max_err, 1e-2, max_err_pos))



    def test_grad_multiple_outs_some_uncomputable(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        vW_in = asarrayX(rng.uniform(size = (2,2), low = -.1,high = .1))
        v_u   = asarrayX(rng.uniform(size = (5,2), low = -.1, high = .1))
        v_x0  = asarrayX(rng.uniform(size = (2,), low = -.1,high = .1))

        W_in = theano.tensor.matrix('win')
        u    = theano.tensor.matrix('u1')
        x0    = theano.tensor.vector('x0')
        # trng  = theano.tensor.shared_randomstreams.RandomStreams(utt.fetch_seed())

        def f_rnn_cmpl(u_t, x_tm1,  W_in):
            trng1 = theano.tensor.shared_randomstreams.RandomStreams(123)
            x_t = theano.dot(u_t, W_in) + x_tm1 + trng1.uniform(low=-.1, high=.1)
            return x_t

        cost, updates = scan_project_sum(f_rnn_cmpl,u,x0,W_in, n_steps = None,
                truncate_gradient = -1, go_backwards = False)
        vparams = [v_u, v_x0,vW_in]
        params = [u,x0,W_in ]
        gparams = theano.tensor.grad(cost, params)
        grad_fn = theano.function([u,x0,W_in], gparams,
                updates = updates, no_default_updates = True)
        cost_fn = theano.function([u,x0,W_in], cost,
                updates = updates, no_default_updates = True)
        def reset_rng_cost_fn(*args):
            for idx,arg in enumerate(cost_fn.maker.expanded_inputs):
                if arg.value and type(arg.value.data) == type(numpy.random.RandomState(123)):
                    cost_fn.maker.expanded_inputs[idx].value.data = numpy.random.RandomState(123)
            return cost_fn(*args)

        def reset_rng_grad_fn(*args):
            for idx,arg in enumerate(grad_fn.maker.expanded_inputs):
                if arg.value and type(arg.value.data)==type(numpy.random.RandomState(123)):
                    grad_fn.maker.expanded_inputs[idx].value.data = numpy.random.RandomState(123)
            return grad_fn(*args)

        num_grad = multiple_outputs_numeric_grad(reset_rng_cost_fn,\
                [v_u,v_x0,vW_in] )
        analytic_grad = reset_rng_grad_fn(v_u, v_x0, vW_in)
        max_err, max_err_pos = num_grad.max_err(analytic_grad)

        if max_err > 1e-2:
            raise Exception(theano.tensor.verify_grad.E_grad,
                    (max_err, 1e-2, max_err_pos))

    def test_grad_multiple_outs_some_truncate(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        vW_in = asarrayX(rng.uniform(size = (2,2), low = -.1,high = .1))
        v_u   = asarrayX(rng.uniform(size = (5,2), low = -.1, high = .1))
        v_x0  = asarrayX(rng.uniform(size = (2,), low = -.1,high = .1))

        W_in = theano.tensor.matrix('win')
        u    = theano.tensor.matrix('u1')
        x0    = theano.tensor.vector('x0')
        # trng  = theano.tensor.shared_randomstreams.RandomStreams(utt.fetch_seed())

        def f_rnn_cmpl(u_t, x_tm1,  W_in):
            trng1 = theano.tensor.shared_randomstreams.RandomStreams(123)
            x_t = theano.dot(u_t, W_in) + x_tm1 + trng1.uniform(low=-.1, high=.1)
            return x_t

        cost, updates = scan_project_sum(f_rnn_cmpl,u,x0,W_in, n_steps = None,
                truncate_gradient = 3, go_backwards = False)
        vparams = [v_u, v_x0,vW_in]
        params = [u,x0,W_in ]
        gparams = theano.tensor.grad(cost, params)
        grad_fn = theano.function([u,x0,W_in], gparams,
                updates = updates, no_default_updates = True)
        cost_fn = theano.function([u,x0,W_in], cost,
                updates = updates, no_default_updates = True)
        def reset_rng_cost_fn(*args):
            for idx,arg in enumerate(cost_fn.maker.expanded_inputs):
                if arg.value and type(arg.value.data) == type(numpy.random.RandomState(123)):
                    cost_fn.maker.expanded_inputs[idx].value.data = numpy.random.RandomState(123)
            return cost_fn(*args)

        def reset_rng_grad_fn(*args):
            for idx,arg in enumerate(grad_fn.maker.expanded_inputs):
                if arg.value and type(arg.value.data)==type(numpy.random.RandomState(123)):
                    grad_fn.maker.expanded_inputs[idx].value.data = numpy.random.RandomState(123)
            return grad_fn(*args)

        num_grad = multiple_outputs_numeric_grad(reset_rng_cost_fn,\
                [v_u,v_x0,vW_in] )
        analytic_grad = reset_rng_grad_fn(v_u, v_x0, vW_in)


        assert len(analytic_grad[0]) == 3


    def test_draw_as_input_to_scan(self):
        trng = theano.tensor.shared_randomstreams.RandomStreams(123)

        x = theano.tensor.matrix('x')
        y = trng.binomial(size = x.shape, p = x)
        z,updates = theano.scan(lambda a:a, non_sequences=y, n_steps=2)

        f = theano.function([x],[y,z], updates = updates)

        rng = numpy.random.RandomState(utt.fetch_seed())
        nx = rng.uniform( size = (10,10) ).astype(config.floatX)
        ny1,nz1 = f(nx)
        ny2,nz2 = f(nx)


        assert numpy.allclose([ny1,ny1], nz1)
        assert numpy.allclose([ny2,ny2], nz2)
        assert not numpy.allclose(ny1,ny2)


    def test_grad_of_shared(self):
        x1 = theano.shared(3.)
        x1.name = 'x1'
        x2 = theano.tensor.vector('x2')
        y, updates = theano.scan(lambda v: v*x1, sequences = x2)
        m = theano.tensor.grad(y.sum(), x1)

        f = theano.function([x2], m)
        print f([2,3])
        assert numpy.allclose(f([2,3]) , 5)

    def test_computing_gradient(self):
        x1 = theano.tensor.scalar()
        x2 = theano.shared(numpy.array([1,2,3,4,5]))
        K = x2*x1

        out,updates = theano.scan(lambda i,v: theano.tensor.grad(K[i], v),
                sequences = theano.tensor.arange(K.shape[0]), non_sequences=x1)
        f = theano.function([x1], out)

        print f(3.)
        assert numpy.all( f(3.) != 0. )



    '''
    def test_shared_updates(self):
        X = theano.shared( numpy.array( [[1,2,3],[4,5,6]]))

        out,updates = theano.scan( lambda :{X: X+1}, outputs_info = [], non_sequences= [],
                sequences = [], n_steps = 10)

        f = theano.function([],[], updates = updates)

        f()

        print X.value
    '''

    def test_scan_output_padding(self):
        """
        Scan outputs are usually lists, whose entries correspond to the intermediate result.
        When n_steps=1, some extra machinery is required in order to mimic this interface. Scan
        thus calls tensor.shape_padleft on the inner function outputs.

        However, this is not the proper behavior for:
        * shared variables : these should not be padded in any way
        * when return_steps is explicitely set to 1. Output should NOT be a list, but a tensor
          corresponding to the result of the last iteration.

        This unit test addresses the bug fix of changeset ba7157e95cb1.
        """
        a = theano.tensor.vector()
        init_a = theano.tensor.vector()
        b = theano.shared(numpy.random.rand(5,4))

        def inner_func(a):
            return a+1, {b:2*b}

        out, updates = theano.scan(inner_func,
                outputs_info = [{'initial': init_a, 'return_steps': 1}],
                n_steps=1)
        assert out.type.ndim == a.type.ndim
        assert updates[b].type.ndim == b.type.ndim

        out, updates = theano.scan(inner_func, outputs_info=[init_a], n_steps=1)
        assert out.type.ndim == a.type.ndim+1
        assert updates[b].type.ndim == b.type.ndim


    def test_scan_as_tensor_on_gradients(self):
        """
        Bug reported by cityhall on scan when computing the gradients
        """
        to_scan = theano.tensor.dvector('to_scan')
        seq     = theano.tensor.dmatrix('seq')
        f1      = theano.tensor.dscalar('f1')

        def scanStep(prev, seq, f1):
           return prev + f1 * seq

        scanned, _ = theano.scan(fn = scanStep, \
                                sequences    = [seq], \
                                outputs_info = [to_scan], \
                                non_sequences  = [f1])

        f_scan = theano.function(inputs=[to_scan, seq, f1], outputs=scanned)
        f_scan([1,2,3], numpy.arange(12).reshape([4,3]), 1.)

        t_grad = theano.tensor.grad(scanned.sum(), wrt=[to_scan, f1],
        consider_constant=[seq])
        f_grad = theano.function(inputs=[to_scan, seq, f1], outputs=t_grad)

        f_scan([1,2,3], numpy.arange(12).reshape([4,3]), 1.)
        f_grad([1,2,3], numpy.arange(12).reshape([4,3]), 1.)

    def caching_nsteps_by_scan_op(self):

        W       = theano.tensor.matrix('weights')
        initial = theano.tensor.vector('initial')
        inpt    = theano.tensor.matrix('inpt')

        def one_step(x_t, h_tm1, W):
          expr = T.dot(h_tm1, W) + x_t
          return expr

        expr, _ = theano.scan(
          fn=one_step,
          sequences=[inpt],
          outputs_info=[initial],
          non_sequences=[W])

        floatX = theano.config.floatX
        sh = expr.shape[0]
        init_val = theano.shared( numpy.ones(5, dtype=floatX))
        inpt_val = theano.shared( numpy.ones((5,5), dtype=floatX))
        shapef = theano.function([W], expr,
                                 givens={initial: init_val,
                                         inpt: inpt_val })
        # First execution to cache n_steps
        val0 = numpy.ones((5,5), dtype = floatX)
        shapef(val0)


        cost = expr.sum()
        d_cost_wrt_W = T.grad(cost, [W])
        init_val = theano.shared( numpy.zeros(5, dtype =floatX))
        f = theano.function([W, inpt], d_cost_wrt_W,
                             givens={initial: init_val})

        rval = numpy.asarray([[5187989]*5]*5, dtype = floatX)
        x = numpy.ones((5,5), dtype = floatX)
        y = numpy.ones((10,5), dtype = floatX)
        t_rval = f( x,y)
        assert numpy.allclose( t_rval, rval)


    def only_one_output_of_grad_of_scan(self):

        initial = theano.tensor.scalar('initial')

        floatX = theano.config.floatX
        def one_step( h_tm1):
          return h_tm1 + numpy.asarray(1., dtype=floatX)

        h, _ = theano.scan(
          fn=one_step,
          outputs_info=[initial],
          n_steps = 3
        )

        gh = TT.grad(h[-1], initial)

        f = theano.function([initial], gh)
        assert numpy.allclose( f(1.), 1.)



if __name__ == '__main__':
    unittest.main()


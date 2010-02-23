from nose.plugins.skip import SkipTest

import unittest
import theano
import numpy

import random
import numpy.random
from theano.tests  import unittest_tools as utt

def verify_grad(op, pt, n_tests=2, rng=None, eps = None, tol = None, 
                mode = None, cast_to_output_type = False):
    pt = [numpy.array(p) for p in pt]

    _type_tol = dict( float32=1e-2, float64=1e-4)

    if tol is None:
        tol = max(_type_tol[str(p.dtype)] for p in pt)

    if rng is None:
        rng = numpy.random
        utt.seed_rng()
    
    def function(inputs, outputs):
        if mode is None:
            f = theano.function(inputs, outputs, accept_inplace=True)
        else:
            f = theano.function(inputs,outputs,accept_inplace=True, mode=mode)
        return f

    for test_num in xrange(n_tests):
        tensor_pt=[theano.tensor.value(p.copy(),name='input %i'%i) 
                                       for i,p in enumerate(pt)]
    # op outputs
    o_outputs = op(*tensor_pt)
    if not (type(o_outputs) in (list,tuple)):
        o_outputs = [ o_outputs ]
    o_fn = function(tensor_pt, o_outputs)
    o_fn_outs = o_fn(*[p.copy() for p in pt])

    if not type(o_fn_outs) in (list,tuple):
        o_fn_outs = [o_fn_outs]

    random_projection = rng.rand(*o_fn_outs[0].shape)
    if cast_to_output_type:
        random_projection = numpy.array(random_projection, 
                             dtype = o_fn_outs[0].dtype)
    t_r = theano.tensor.as_tensor_variable(random_projection)
    cost = theano.tensor.sum( t_r * o_outputs[0])
    for i, o in enumerate(o_fn_outs[1:] ):
        random_projection = rng.rand(*o.shape)
        if cast_to_output_type:
            random_projection = numpy.array(random_projection,
                                            dtype=o_outputs[i].dtype)
        t_r  = theano.tensor.as_tensor_variable(random_projection)
        cost += theano.tensor.sum( t_r * o_outputs[i])
    cost_fn = function(tensor_pt, cost)
    num_grad = theano.tensor.numeric_grad(cost_fn,[p.copy() for p in pt],eps)
    g_cost = theano.tensor.as_tensor_variable(1.0,name='g_cost')
    if cast_to_output_type:
        g_cost = cast(g_cost, o_output.dtype)
    symbolic_grad = theano.tensor.grad(cost, tensor_pt, g_cost)
    

    grad_fn = function(tensor_pt,symbolic_grad)
    analytic_grad = grad_fn(*[p.copy() for p in pt])
    if not isinstance(analytic_grad, (list,tuple)):
        analytic_grad = [analytic_grad]

    max_err, max_err_pos = num_grad.max_err(analytic_grad)
    if max_err > tol:
        raise Exception(theano.tensor.verify_grad.E_grad, 
                                    (max_err, tol, max_err_pos))


#TODO: Test this function, and if it works,
# use it with the normal verify_grad rather than the 
# copy-and-pasted one above.
# Also - add a reference to this technique in the 
# verify_grad method so that other ops with multiple outputs can be tested.
def scan_project_sum(*args, **kwargs):
    rng = shared_randomstreams.RandomStreams()
    scan_outputs = scan(*args, **kwargs)
    # we should ignore the random-state updates so that
    # the uniform numbers are the same every evaluation and on every call
    rng.add_default_updates = False
    return  sum([(s * rng.uniform(size=s.shape)).sum() for s in scan_outputs])




def compareArrays(a,b):
    if type(a) in (list,tuple):
        a = numpy.array(a)
    if type(b) in (list, tuple):
        b = numpy.array(b)

    return numpy.all( abs(a-b) < 1e-5)



 
class T_Scan(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    # generator network, only one output , type scalar ; no sequence or 
    # non sequence arguments
    def test_1(self):


      def f_pow2(x_tm1):
        return 2*x_tm1
    
      s = theano.tensor.dscalar()
      n_steps = theano.tensor.dscalar()
      Y, updts = theano.scan(f_pow2, [],s, [],n_steps = n_steps)
      f1 = theano.function([s,n_steps], Y, updates = updts)
     
      assert(compareArrays(f1(1,3), [2,4,8]))
    # simple rnn, one input, one state, weights for each; input/state are 
    # vectors, weights are scalars
    def test_2(self):


        def f_rnn(u_t,x_tm1,W_in, W):
            return u_t*W_in+x_tm1*W
    
        u    = theano.tensor.dvector()
        x0   = theano.tensor.dscalar()
        W_in = theano.tensor.dscalar()
        W    = theano.tensor.dscalar()

        Y, updts = theano.scan(f_rnn, u,x0,[W_in,W])
    
        f2    = theano.function([u,x0,W_in,W], Y, updates = updts)
        v_u   = numpy.array([1.,2.,3.,4.])
        v_x0  = numpy.array(1)
        v_out = numpy.array([1.1,1.3,1.6,2.])
        assert(compareArrays( f2(v_u,v_x0,.1,1), v_out   ) )

    # simple rnn, one input, one state, weights for each; input/state are 
    # vectors, weights are scalars; using shared variables
    def test_3(self):
    
        u    = theano.tensor.dvector()
        x0   = theano.tensor.dscalar()
        W_in = theano.shared(.1, name = 'w_in')
        W    = theano.shared(1., name ='w')
    
        def f_rnn_shared(u_t,x_tm1, l_W_in, l_W):
            return u_t*l_W_in+x_tm1*l_W
    
        Y, updts = theano.scan(f_rnn_shared, u,x0,[W_in, W] )

        f3    = theano.function([u,x0], Y, updates = updts)
        v_u   = numpy.array([1.,2.,3.,4.])
        v_x0  = numpy.array(1.)
        v_out = numpy.array([1.1,1.3,1.6,2.])
        assert(compareArrays(f3(v_u,v_x0),v_out))

    # some rnn with multiple outputs and multiple inputs; other dimension 
    # instead of scalars/vectors
    def test_4(self):
    
        W_in2 = theano.shared(numpy.array([1.,2.]), name='win2')
        W     = theano.shared(numpy.array([[2.,1.],[1.,1.]]), name='w')
        W_out = theano.shared(numpy.array([.5,1.]), name = 'wout')
        W_in1 = theano.tensor.dmatrix('win')
        u1    = theano.tensor.dmatrix('u1')
        u2    = theano.tensor.dvector('u2')
        x0    = theano.tensor.dvector('x0')
        y0    = theano.tensor.dscalar('y0')
    
        def f_rnn_cmpl(u1_t, u2_t, x_tm1, y_tm1, W_in1):
            return [theano.dot(u1_t,W_in1) + u2_t* W_in2 + \
                    theano.dot(x_tm1, W), theano.dot(x_tm1, W_out)]

        Y, updts = theano.scan(f_rnn_cmpl,[u1,u2],[x0,y0],W_in1)

        f4     = theano.function([u1,u2,x0,y0,W_in1], Y, updates = updts)
        v_u1   = numpy.array([[1.,2.],[1.,2.],[1.,2.]])
        v_u2   = numpy.array([1.,2.,3.])
        v_x0   = numpy.array([0.,0.])
        v_y0   = numpy.array(1)
        v_Win1 = numpy.array([[1.,1.],[1.,1.]])
        v_x    = numpy.array([[4.,5.],[18.,16.],[58.,43.]])
        v_y    = numpy.array([0.,7.,25.])
        (x,y) =  f4( v_u1, v_u2, v_x0, v_y0, v_Win1)
         
        assert( compareArrays(x,v_x)) 
        assert( compareArrays(y,v_y))

    # simple rnn, one input, one state, weights for each; input/state are 
    # vectors, weights are scalars; using shared variables and past 
    # taps (sequences and outputs)
    def test_5(self):
    
        u    = theano.tensor.dvector()
        x0   = theano.tensor.dvector()
        W_in = theano.shared(.1, name = 'w_in')
        W    = theano.shared(1., name ='w')
    
        def f_rnn_shared(u_tm2, x_tm1, x_tm2):
            return u_tm2*W_in+x_tm1*W+x_tm2
    
        Y, updates = theano.scan(f_rnn_shared, u,x0, [], \
                 sequences_taps = {0:[-2]}, outputs_taps = {0:[-1,-2]})

        f7   = theano.function([u,x0], Y, updates = updates)
        v_u  = numpy.asarray([1.,2.,3.,4.])
        v_x0 = numpy.asarray([1.,2.])
        out  = numpy.asarray([3.1,5.3])
        assert (compareArrays( out, f7(v_u, v_x0)))
      
    # simple rnn, one input, one state, weights for each; input/state are 
    # vectors, weights are scalars; using shared variables and past 
    # taps (sequences and outputs) and future taps for sequences
    def test_6(self):
    
        u    = theano.tensor.dvector()
        x0   = theano.tensor.dvector()
        W_in = theano.shared(.1, name = 'w_in')
        W    = theano.shared(1., name ='w')
    
        def f_rnn_shared(u_tm2,u_tp2, x_tm1, x_tm2):
            return (u_tm2+u_tp2)*W_in+x_tm1*W+x_tm2
    
        Y,updts = theano.scan(f_rnn_shared, u,x0, [], \
                 sequences_taps = {0:[-2,2]}, outputs_taps = {0:[-1,-2]})

        f8   = theano.function([u,x0], Y, updates = updts)
        v_u  = numpy.array([1.,2.,3.,4.,5.,6.])
        v_x0 = numpy.array([1.,2.])
        out  = numpy.array([3.6, 6.4])

        assert (compareArrays( out, f8(v_u, v_x0) ) )
    
    # simple rnn ; compute inplace
    def test_7(self):
        
        u    = theano.tensor.dvector()
        mu   = theano.Param( u, mutable = True)
        x0   = theano.tensor.dscalar()
        W_in = theano.shared(.1)
        W    = theano.shared(1.)

        def f_rnn_shared(u_t, x_tm1):
            return u_t*W_in + x_tm1*W
        Y, updts = theano.scan(f_rnn_shared, u, x0,[], \
                    inplace_map={0:0} )
        f9   = theano.function([mu,x0], Y , updates = updts)
        v_u  = numpy.array([1.,2.,3.])
        v_x0 = numpy.array(1.)

        out = f9(v_u, v_x0)
        v_out = numpy.array([1.1,1.3,1.6])

        assert (compareArrays(out, v_out))
        assert (compareArrays(v_u, out))
    
    # Shared variable with updates
    def test_8(self):
       W1_vals = numpy.random.rand(20,30)
       W2_vals = numpy.random.rand(30,20)
       u1_vals = numpy.random.rand(3,20)
       u2_vals = numpy.random.rand(3,30)
       y0_vals = numpy.random.rand(3,20)
       y1_vals = numpy.random.rand(20)
       y2_vals = numpy.random.rand(30)
    
       W1 = theano.shared(W1_vals)
       W2 = theano.shared(W2_vals)

       u1 = theano.shared(u1_vals)
       y1 = theano.shared(y1_vals)

       def f(u1_t, u2_t, y0_tm3, y0_tm2, y0_tm1, y1_tm1):
            y0_t = theano.dot(theano.dot(u1_t,W1),W2) + 0.1*y0_tm1 + \
                                             0.33*y0_tm2 + 0.17*y0_tm3
            y1_t = theano.dot(u2_t, W2) + y1_tm1
            y2_t = theano.dot(u1_t, W1)
            nwW1 = W1 + .1
            nwW2 = W2 + .05
            return ([y0_t, y1_t, y2_t], [(W1,nwW1), (W2, nwW2)])

       u2 = theano.tensor.matrix()
       y0 = theano.tensor.matrix()
       y2 = theano.tensor.vector()

       Y,upds = theano.scan(f, [u1,u2], [y0,y1,y2],[], outputs_taps = {0:[-3,-2,-1], 2:[]})

       f = theano.function([u2,y0,y2], Y, updates = upds)


       vls = f(u2_vals, y0_vals, y2_vals)

       # do things in numpy
       v_y0 = numpy.zeros((6,20))
       v_y1 = numpy.zeros((4,20))
       v_y2 = numpy.zeros((3,30))
       v_y0[:3] = y0_vals
       v_y1[0]  = y1_vals
       vW1      = W1_vals.copy()
       vW2      = W2_vals.copy()
       for idx in xrange(3):
          v_y0[idx+3] = numpy.dot( numpy.dot(u1_vals[idx,:], vW1), vW2) + \
                        0.1*v_y0[idx+2] + 0.33*v_y0[idx+1] + 0.17*v_y0[idx]
          v_y1[idx+1] = numpy.dot( u2_vals[idx,:], vW2) + v_y1[idx]
          v_y2[idx]   = numpy.dot( u1_vals[idx,:], vW1)
          vW1 = vW1 + .1
          vW2 = vW2 + .05

    def test_9(self):

        W_vals  = numpy.random.rand(20,30) -.5
        vis_val = numpy.random.binomial(1,0.5, size=(3,20))
        bvis = numpy.random.rand(20) -.5
        bhid = numpy.random.rand(30) -.5

        tW  = theano.shared(W_vals)
        tbh = theano.shared(bhid)
        tbv = theano.shared(bvis)

        vis = theano.tensor.matrix()

        trng = theano.tensor.shared_randomstreams.RandomStreams(123)


        def f(vsample):
            hmean = theano.tensor.nnet.sigmoid(theano.dot(vsample,tW)+ tbh)
            hsample = trng.binomial(hmean.shape,1,hmean)
            vmean = theano.tensor.nnet.sigmoid(theano.dot(hsample,tW.T)+ tbv)
            return trng.binomial(vsample.shape,1,vsample)

        
        v_vals, updts = theano.scan(f, [], [vis],[], n_steps = 10,
                     sequences_taps = {}, outputs_taps = {})

        my_f = theano.function([vis], v_vals[-1], updates = updts)


        def numpy_implementation(vsample):

            rng = numpy.random.RandomState(123)
            b1  = numpy.random.RandomState(rng.randint(2**30))
            b2  = numpy.random.RandomState(rng.randint(2**30))

            for idx in range(10):
                hmean = 1./(1. + numpy.exp(-(numpy.dot(vsample,W_vals) + bhid)))
                hsample = b1.binomial(1,hmean, size = hmean.shape)
                vmean  = 1./(1. + numpy.exp(-(numpy.dot(hsample,W_vals.T) + bvis)))
                vsample = b2.binomial(1,vsample, size = vsample.shape)

            return vsample

        t_res = my_f(vis_val)
        n_res = numpy_implementation(vis_val)

        assert (compareArrays(t_res, n_res))

    def test_10(self):

      s = theano.shared(1)


      def f_pow2():
        return {s: 2*s}
    
      n_steps = theano.tensor.dscalar()
      Y, updts = theano.scan(f_pow2, [],[], [],n_steps = n_steps)
      f1 = theano.function([n_steps], Y, updates = updts)
      f1(3)
      assert(compareArrays(s.value, 8))
 
    '''
    # test gradient simple network 
    def test_10(self):
        pass

     TO TEST: 
        - test gradient (one output)
        - test gradient (multiple outputs)
        - test gradient (go_bacwards) 
        - test gradient (multiple outputs / some uncomputable )
        - test gradient (truncate_gradient)
        - test_gradient (taps past/future)
        - optimization !? 
    '''

    def test_map_functionality(self):
        raise SkipTest('Map functionality not implemented yet')

        def f_rnn(u_t):
            return u_t + 3
    
        u    = theano.tensor.dvector()

        Y, updts = theano.scan(f_rnn, sequences=u, outputs_taps={0:[]})
    
        f2    = theano.function([u], Y, updates = updts)
        v_u   = numpy.array([1.,2.,3.,4.])
        assert compareArrays(f2(v_u), v_u+3)


if __name__ == '__main__':
    unittest.main()

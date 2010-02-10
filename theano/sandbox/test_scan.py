
import unittest
import theano
import theano.sandbox.scan


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
        return (2*x_tm1, {})
    
      s = theano.tensor.dscalar()
      n_steps = theano.tensor.dscalar()
      Y = theano.sandbox.scan.scan(f_pow2, [],s, [],n_steps = n_steps)
    
      f1 = theano.function([s,n_steps], Y)
      
      assert(compareArrays(f1(1,3), [2,4,8]))

    # simple rnn, one input, one state, weights for each; input/state are 
    # vectors, weights are scalars
    def test_2(self):
        def f_rnn(u_t,x_tm1,W_in, W):
            return (u_t*W_in+x_tm1*W, {})
    
        u    = theano.tensor.dvector()
        x0   = theano.tensor.dscalar()
        W_in = theano.tensor.dscalar()
        W    = theano.tensor.dscalar()

        Y = theano.sandbox.scan.scan(f_rnn, u,x0,[W_in,W])
    
        f2    = theano.function([u,x0,W_in,W], Y)
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
    
        def f_rnn_shared(u_t,x_tm1):
            return (u_t*W_in+x_tm1*W, {})
    
        Y = theano.sandbox.scan.scan(f_rnn_shared, u,x0,[])

        f3    = theano.function([u,x0], Y)
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

        Y = theano.sandbox.scan.scan(f_rnn_cmpl,[u1,u2],[x0,y0],W_in1)
    
        f4     = theano.function([u1,u2,x0,y0,W_in1], Y)
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


    # basic ESN using updates 
    def test_5(self): 
        W_in = theano.shared(numpy.array([1.,1.]), name='win')
        W    = theano.shared(numpy.array([[.1,0.],[.0,.1]]),name='w')
        W_out= theano.shared(numpy.array([.5,1.]), name='wout')
    
        u  = theano.tensor.dvector('u')
        x  = theano.shared(numpy.array([0.,0.]),'x')
        y0 = theano.tensor.dscalar('y0')
    
        def f_ESN(u_t):
            return ( theano.dot(x,W_out), \
             { x: W_in*u_t + theano.dot(x,W) } )
    
        Y = theano.sandbox.scan.scan(f_ESN,u,y0,[],outputs_taps={0:[]})
    
        f5    = theano.function([u,y0],Y)
        v_u   = numpy.array([1.,2.,3.])
        v_y0  = numpy.array(0.)
        v_out  = numpy.array([0.,1.5,3.15])
        out = f5( v_u, v_y0 )
        assert( compareArrays(v_out, out))

    # basic ESN using updates ; moving backwards
    def test_6(self): 
        W_in = theano.shared(numpy.array([1.,1.]), name='win')
        W    = theano.shared(numpy.array([[.1,0.],[.0,.1]]),name='w')
        W_out= theano.shared(numpy.array([.5,1.]), name='wout')
    
        u  = theano.tensor.dvector('u')
        x  = theano.shared(numpy.array([0.,0.]),'x')
        y0 = theano.tensor.dvector('y0')
    
        def f_ESN(u_t):
            return ( theano.dot(x,W_out), \
             { x: W_in*u_t + theano.dot(x,W) } )
    
        Y = theano.sandbox.scan.scan(f_ESN,u,y0,[],outputs_taps={0:[]}, \
                                     go_backwards = True)
    
        f6    = theano.function([u,y0],Y)
        v_u   = numpy.array([1.,2.,3.])
        v_y0  = numpy.array([0])
        v_out = numpy.array([0.,4.5,3.45])
        out   = f6(v_u, v_y0)
        
        assert( compareArrays(out, v_out))

    # simple rnn, one input, one state, weights for each; input/state are 
    # vectors, weights are scalars; using shared variables and past 
    # taps (sequences and outputs)
    def test_7(self):
    
        u    = theano.tensor.dvector()
        x0   = theano.tensor.dvector()
        W_in = theano.shared(.1, name = 'w_in')
        W    = theano.shared(1., name ='w')
    
        def f_rnn_shared(u_tm2, x_tm1, x_tm2):
            return (u_tm2*W_in+x_tm1*W+x_tm2, {})
    
        Y = theano.sandbox.scan.scan(f_rnn_shared, u,x0, [], \
                 sequences_taps = {0:[-2]}, outputs_taps = {0:[-1,-2]})

        f7   = theano.function([u,x0], Y)
        v_u  = numpy.asarray([1.,2.,3.,4.])
        v_x0 = numpy.asarray([1.,2.])
        out  = numpy.asarray([3.1,5.3])
        assert (compareArrays( out, f7(v_u, v_x0)))
        
    # simple rnn, one input, one state, weights for each; input/state are 
    # vectors, weights are scalars; using shared variables and past 
    # taps (sequences and outputs) and future taps for sequences
    def test_8(self):
    
        u    = theano.tensor.dvector()
        x0   = theano.tensor.dvector()
        W_in = theano.shared(.1, name = 'w_in')
        W    = theano.shared(1., name ='w')
    
        def f_rnn_shared(u_tm2,u_tp2, x_tm1, x_tm2):
            return ((u_tm2+u_tp2)*W_in+x_tm1*W+x_tm2, {})
    
        Y = theano.sandbox.scan.scan(f_rnn_shared, u,x0, [], \
                 sequences_taps = {0:[-2,2]}, outputs_taps = {0:[-1,-2]})

        f8   = theano.function([u,x0], Y)
        v_u  = numpy.array([1.,2.,3.,4.,5.,6.])
        v_x0 = numpy.array([1.,2.])
        out  = numpy.array([3.6, 6.4])

        assert (compareArrays( out, f8(v_u, v_x0) ) )
        
    '''
    # simple rnn ; compute inplace
    def test_9(self):
        
        u    = theano.tensor.dvector()
        mu   = theano.Param( u, mutable = True)
        x0   = theano.tensor.dvector()
        W_in = theano.shared(.1)
        W    = theano.shared(1.)

        def f_rnn_shared(u_t, x_tm1):
            return (u_t*W_in + x_tm1*W, {})
        Y = theano.sandbox.scan.scan(f_rnn_shared, u, x0,[], \
                    inplace_map={0:0} )
        f9   = theano.function([mu,x0], Y , #mode = 'FAST_RUN')
                                mode = 'DEBUG_MODE')
        v_u  = numpy.array([1.,2.,3.])
        v_x0 = numpy.array([1.])

        out = f9(v_u, v_x0)
        v_out = numpy.array([1.1,1.3,1.6])

        assert (compareArrays(out, v_out))
        print v_u
        assert (compareArrays(v_u, out))

    '''
    # test gradient simple network 
    def test_10(self):
        pass

    '''
     TO TEST: 
        - test gradient (one output)
        - test gradient (multiple outputs)
        - test gradient (go_bacwards) 
        - test gradient (multiple outputs / some uncomputable )
        - test gradient (truncate_gradient)
        - test gradient (force_gradient)
        - test_gradient (taps past/future)
    '''


if __name__ == '__main__':
    unittest.main()

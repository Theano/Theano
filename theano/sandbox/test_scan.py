from scan import Scan

import unittest
import theano

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






 
class T_Scan(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()



    # generator network, only one output , type scalar ; no sequence or 
    # non sequence arguments
    def test_1():
      def f_pow2(x_tm1):
        return (2*x_tm1, {})
    
      s = theano.tensor.dvector()
      n_steps = theano.tensor.dscalar()
      Y = theano.sandbox.scan.scan(f_pow2, [],s, [],n_steps = n_steps)
    
      f1 = theano.function([s,n_steps], Y)
      assert( numpy.any(f1([1],3)== [2,4,8])  )

    # simple rnn, one input, one state, weights for each; input/state are 
    # vectors, weights are scalars
    def test_2():
        def f_rnn(u_t,x_tm1,W_in, W):
            return (u_t*W_in+x_tm1*W, {})
    
        u    = theano.tensor.dvector()
        x0   = theano.tensor.dvector()
        W_in = theano.tensor.dscalar()
        W    = theano.tensor.dscalar()

        Y = theano.sandbox.scan.scan(f_rnn, u,x0,[W_in,W])
    
        f2 = theano.function([u,x0,W_in,W], Y)
        
        assert(numpy.any(f2([1,2,3,4],[1],.1,1)== \
                numpy.array([1.1,1.3,1.6,2.])))

    # simple rnn, one input, one state, weights for each; input/state are 
    # vectors, weights are scalars; using shared variables
    def test_3():
    
        u    = theano.tensor.dvector()
        x0   = theano.tensor.dvector()
        W_in = theano.shared(.1, name = 'w_in')
        W    = theano.shared(1., name ='w')
    
        def f_rnn_shared(u_t,x_tm1):
            return (u_t*W_in+x_tm1*W, {})
    
        Y = theano.sandbox.scan.scan(f_rnn_shared, u,x0,[])

        f3 = theano.function([u,x0], Y)
        
        assert(numpy.any(f3([1,2,3,4],[1])== numpy.array([1.1,1.3,1.6,2.])))


    # some rnn with multiple outputs and multiple inputs; other dimension 
    # instead of scalars/vectors
    def test_4():
    
        W_in2 = theano.shared(numpy.array([1.,2.]), name='win2')
        W     = theano.shared(numpy.array([[2.,1.],[1.,1.]]), name='w')
        W_out = theano.shared(numpy.array([.5,1.]), name = 'wout')
        W_in1 = theano.tensor.dmatrix('win')
        u1    = theano.tensor.dmatrix('u1')
        u2    = theano.tensor.dvector('u2')
        x0    = theano.tensor.dmatrix('x0')
        y0    = theano.tensor.dvector('y0')
    
        def f_rnn_cmpl(u1_t, u2_t, x_tm1, y_tm1, W_in1):
            return ({}, [theano.dot(u1_t,W_in1) + u2_t* W_in2 + \
                    theano.dot(x_tm1, W), theano.dot(x_tm1, W_out)])

        Y = theano.sandbox.scan.scan(f_rnn_cmpl,[u1,u2],[x0,y0],W_in1)
    
        f4 = theano.function([u1,u2,x0,y0,W_in1], Y)
    
        (x,y) =  f4( numpy.array([[1,2],[1,2],[1,2]]), \
                  numpy.array([1,2,3]),             \
                  numpy.array([[0,0]]),             \
                  numpy.array([1]),                 \
                  numpy.array([[1,1],[1,1]]))
    
        assert( numpy.all(x == numpy.array([[4.,5.],[18.,16.],[58.,43.]])))
        assert( numpy.all(y == numpy.array([0.,7.,25.])))


    # basic ESN using updates 
    def test_5(): 
        W_in = theano.shared(numpy.array([1.,1.]), name='win')
        W    = theano.shared(numpy.array([[.1,0.],[.0,.1]]),name='w')
        W_out= theano.shared(numpy.array([.5,1.]), name='wout')
    
        u  = theano.tensor.dvector('u')
        x  = theano.shared(numpy.array([0.,0.]),'x')
        y0 = theano.tensor.dvector('y0')
    
        def f_ESN(u_t):
            return ( theano.dot(x,W_out), \
             { x: W_in*u_t + theano.dot(x,W) } )
    
        Y = theano.sandbox.scan.scan(f_ESN,u,y0,[],outputs_taps={0:[]})
    
        f5 = theano.function([u,y0],Y)
        assert( f5( numpy.array([1,2,3]), numpy.array([0])) == \
                 numpy.array([0.,1.4,3.15]))

    # basic ESN using updates ; moving backwards
    def test_6(): 
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
    
        f6 = theano.function([u,y0],Y)
        assert( f6( numpy.array([1,2,3]), numpy.array([0])) == \
                 numpy.array([0., 4.5, 3.45]))


    '''
     TO TEST: 
        - test taps (for sequences and outputs )
        - test gradient (one output)
        - test gradient (multiple outputs)
        - test gradient (go_bacwards) 
        - test gradient (multiple outputs / some uncomputable )
        - test gradient (truncate_gradient)
        - test gradient (force_gradient) 
        - test inplace map
    '''


if __name__ == '__main__':
    unittest.main()

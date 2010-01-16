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






# Naming convention : 
#  u_1,u_2,..   -> sequences
#  s_1,s_2,..   -> initial states
#  w_1,w_2,..   -> non-sequences
###################################
 
class T_Scan(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

   def test_one(self):
      pass

if __name__ == '__main__':
    unittest.main()

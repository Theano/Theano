import numpy
import theano
import theano.sandbox.scan



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
    
    assert(numpy.any(f2([1,2,3,4],[1],.1,1)== numpy.array([1.1,1.3,1.6,2.])))

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

## Why dot doesn;t work with scalars !??
## Why  *  doesn't support SharedVariable and TensorVariable

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


if __name__=='__main__':

    test_1()
    test_2()
    test_3()
    test_4()
    test_5()
    test_6()





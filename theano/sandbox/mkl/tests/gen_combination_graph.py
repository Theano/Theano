import theano
import numpy as np
from theano import tensor as T
from theano.tensor.signal import pool

def run_test(direction = 'forward'):
    print ('=' * 60)
    print ('generate relu_pool graph before and after opt for %s pass' % direction)

    x = T.ftensor4('x')
    maxpoolshp = (2, 2)
    ignore_border = True
    mode = 'max'

    imval = np.random.rand(4, 2, 16, 16).astype(np.float32)

    reluOut = T.nnet.relu(x)
    poolOut = pool.pool_2d(reluOut, maxpoolshp, ignore_border, mode=mode)
    if direction == 'forward':
        theano.printing.pydotprint(poolOut, outfile="relu_pool_before_opt.png", var_with_name_simple=True)
        f = theano.function(inputs=[x], outputs=[poolOut])
        theano.printing.pydotprint(f, outfile="relu_pool_after_opt.png", var_with_name_simple=True)
        f(imval)
    elif direction == 'backward':
        poolSum = T.sum(poolOut)
        poolBackward = T.grad(poolSum, [x])
        theano.printing.pydotprint(poolBackward, outfile="relu_poolBackward_before_opt.png", var_with_name_simple=True)
        f = theano.function(inputs=[x], outputs=poolBackward)
        theano.printing.pydotprint(f, outfile="relu_poolBackward_after_opt.png", var_with_name_simple=True)
        f(imval)
    else:
        print ("Invalid direction, only forward or backward allowed!")
    

if __name__ == '__main__':
    run_test('forward')
    run_test('backward')

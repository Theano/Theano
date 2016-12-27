import theano
from theano import tensor as T
import numpy as np


def run_test(direction='forward', x=T.ftensor4('x')):
    print ('=' * 60)
    print ('generate relu graph before and after opt for %s pass' % direction)
    y = T.nnet.relu(x)

    input_ndim = x.type.ndim

    if input_ndim == 4:
        imval = np.random.rand(4, 2, 4, 4).astype(np.float32)
    elif input_ndim == 2:
        imval = np.random.rand(4, 2).astype(np.float32)
    else:
        raise NotImplementedError("Not implemented for the %d ndims" % input_ndim)

    if direction == 'forward':
        theano.printing.pydotprint(y, outfile="relu_%dD_before_opt.png" % input_ndim, var_with_name_simple=True)
        f = theano.function(inputs=[x], outputs=y)
        theano.printing.pydotprint(f, outfile="relu_%dD_after_opt.png" % input_ndim, var_with_name_simple=True)
        f(imval)
    elif direction == 'backward':
        reluSum = T.sum(y)
        reluBackward = T.grad(reluSum, [x])
        theano.printing.pydotprint(reluBackward, outfile="reluGrad_%dD_before_opt.png" % input_ndim, var_with_name_simple=True)
        f = theano.function(inputs=[x], outputs=reluBackward)
        theano.printing.pydotprint(f, outfile="reluGrad_%dD_after_opt.png" % input_ndim, var_with_name_simple=True)
        f(imval)
    else:
        print ("Invalid direction, only forward or backward allowed!")


if __name__ == '__main__':
    x = T.ftensor4('x_4D')
    run_test('forward', x)
    run_test('backward', x)

    x = T.fmatrix('x_2D')
    run_test('forward', x)
    run_test('backward', x)

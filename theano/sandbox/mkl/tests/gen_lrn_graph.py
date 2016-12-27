
import theano
from theano import tensor as T

import theano.sandbox.mkl
from theano.tensor.nnet.lrn import lrn

x = T.ftensor4()

y = lrn(x)


theano.printing.pydotprint(y, outfile='lrn_fwd_before.png', var_with_name_simple=True)

f = theano.function([x], y)

theano.printing.pydotprint(f, outfile='lrn_fwd_after.png', var_with_name_simple=True)



z = T.grad(T.sum(y), [x])

theano.printing.pydotprint(z, outfile='lrn_bwd_before.png', var_with_name_simple=True)

f1 = theano.function([x], z)

theano.printing.pydotprint(f1, outfile='lrn_bwd_after.png', var_with_name_simple=True)

import theano
import theano.tensor as T
from theano import printing


X = T.matrix('X')
results, updates = theano.scan(
    fn=lambda x: 2*x.sum() + 3,
    outputs_info=None,
    sequences=[X],
    non_sequences=None
)
printing.debugprint(results)

from __future__ import absolute_import, print_function, division
import theano
import theano.tensor as tt
from six.moves import xrange

k = tt.iscalar("k")
A = tt.vector("A")


def inner_fct(prior_result, A):
    return prior_result * A
# Symbolic description of the result
result, updates = theano.scan(fn=inner_fct,
                              outputs_info=tt.ones_like(A),
                              non_sequences=A, n_steps=k)

# Scan has provided us with A**1 through A**k.  Keep only the last
# value. Scan notices this and does not waste memory saving them.
final_result = result[-1]

power = theano.function(inputs=[A, k],
                        outputs=final_result,
                        updates=updates)

print(power(list(range(10)), 2))
#[  0.   1.   4.   9.  16.  25.  36.  49.  64.  81.]

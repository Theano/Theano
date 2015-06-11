from __future__ import print_function
import theano
import theano.tensor as T

k = T.iscalar("k"); A = T.vector("A")

def inner_fct(prior_result, A): return prior_result * A
# Symbolic description of the result
result, updates = theano.scan(fn=inner_fct,
                              outputs_info=T.ones_like(A),
                              non_sequences=A, n_steps=k)

# Scan has provided us with A**1 through A**k.  Keep only the last
# value. Scan notices this and does not waste memory saving them.
final_result = result[-1]

power = theano.function(inputs=[A,k], outputs=final_result,
                        updates=updates)

print(power(range(10),2))

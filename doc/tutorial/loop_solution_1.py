
# Theano tutorial
# Solution to Exercise in section 'Loop'

"""

# 1. First example (runs satisfactorily)

import theano
import theano.tensor as T

theano.config.warn.subtensor_merge_bug = False

k = T.iscalar("k")
A = T.vector("A")


def inner_fct(prior_result, A):
    return prior_result * A

# Symbolic description of the result
result, updates = theano.scan(fn=inner_fct,
                            outputs_info=T.ones_like(A),
                            non_sequences=A, n_steps=k)

# Scan has provided us with A ** 1 through A ** k.  Keep only the last
# value. Scan notices this and does not waste memory saving them.
final_result = result[-1]

power = theano.function(inputs=[A, k], outputs=final_result,
                      updates=updates)

print power(range(10), 2)
# [  0.   1.   4.   9.  16.  25.  36.  49.  64.  81.]


# 2. Second example (runs satisfactorily)

import numpy
import theano
import theano.tensor as T

coefficients = theano.tensor.vector("coefficients")
x = T.scalar("x")
max_coefficients_supported = 10000

# Generate the components of the polynomial
full_range = theano.tensor.arange(max_coefficients_supported)
components, updates = theano.scan(fn=lambda coeff, power, free_var:
                                   coeff * (free_var ** power),
                                outputs_info=None,
                                sequences=[coefficients, full_range],
                                non_sequences=x)
polynomial = components.sum()
calculate_polynomial1 = theano.function(inputs=[coefficients, x],
                                     outputs=polynomial)

test_coeff = numpy.asarray([1, 0, 2], dtype=numpy.float32)
print calculate_polynomial1(test_coeff, 3)
# 19.0
"""


# 3. Reduction performed inside scan

# TODO: repair this code: yields 56.0 instead of 19.0

import numpy
import theano
import theano.tensor as T

theano.config.warn.subtensor_merge_bug = False

coefficients = theano.tensor.vector("coefficients")
x = T.scalar("x")
max_coefficients_supported = 10000

# Generate the components of the polynomial
full_range = theano.tensor.arange(max_coefficients_supported)


outputs_info = T.as_tensor_variable(numpy.asarray(0, 'float64'))

components, updates = theano.scan(fn=lambda prior_value, coeff, power, free_var:
                                 prior_value + (coeff * (free_var ** power)),
                                 outputs_info=outputs_info,
                                 sequences=[coefficients, full_range],
                                 non_sequences=x)

polynomial = components[-1]
calculate_polynomial = theano.function(inputs=[coefficients, x],
                                     outputs=polynomial, updates=updates)

test_coeff = numpy.asarray([1, 0, 2], dtype=numpy.float32)
print calculate_polynomial(test_coeff, 3)
# 19.0

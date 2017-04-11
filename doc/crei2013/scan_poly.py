from __future__ import absolute_import, print_function, division
import numpy as np

import theano
import theano.tensor as tt

coefficients = theano.tensor.vector("coefficients")
x = tt.scalar("x")
max_coefficients_supported = 10000

# Generate the components of the polynomial
full_range = theano.tensor.arange(max_coefficients_supported)
components, updates = theano.scan(fn=lambda coeff, power, free_var:
                                  coeff * (free_var ** power),
                                  outputs_info=None,
                                  sequences=[coefficients, full_range],
                                  non_sequences=x)
polynomial = components.sum()
calculate_polynomial = theano.function(inputs=[coefficients, x],
                                       outputs=polynomial)

test_coeff = np.asarray([1, 0, 2], dtype=np.float32)
print(calculate_polynomial(test_coeff, 3))
# 19.0

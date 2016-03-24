from __future__ import absolute_import, print_function, division
import theano
a = theano.tensor.vector("a") # declare variable
b = a + a**10                 # build symbolic expression
f = theano.function([a], b)   # compile function
print(f([0,1,2]))
# prints `array([0,2,1026])`

theano.printing.pydotprint(b, outfile="pics/f_unoptimized.png", var_with_name_simple=True)
theano.printing.pydotprint(f, outfile="pics/f_optimized.png", var_with_name_simple=True)

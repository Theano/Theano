import os

import numpy as np

import theano

a = theano.tensor.vector()
b = theano.tensor.vector()
for inps, out in [([a], theano.tensor.exp(a)),  # 1 input/1 outputs
                  ([a, b], a + b),  # 2 inputs
                  ((a, b), [a, b]),  # 2 outputs, 2 deepcopy ops
                  ((a, b), [a + b, a - b]),
              ]:
    f = theano.function(inps, out, theano.Mode(linker='c'),
                        on_unused_input='ignore')
    theano.printing.debugprint(f)
    #filename = f.fn.thunks[0].filename  # with linker=vm
    filename = f.fn.filename  # with linker=c
    print filename

#theano.shared_lib(f, name='libtheano_exp')
#f(np.arange(10))
    x = os.system(filename[:-3])
    if x != 0:
        exit(x)


# Test raise error if no c code
try:
    theano.function((a,), [theano.tensor.argmax(a)],
                    theano.Mode(linker='c'),
                    on_unused_input='ignore')
    assert False, "Expected an error"
except NotImplementedError:
    pass
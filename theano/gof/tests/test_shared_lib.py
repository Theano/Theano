from __future__ import absolute_import, print_function, division
import os
import theano
import subprocess


def test_basic():
    a = theano.tensor.vector()
    b = theano.tensor.vector()
    for inps, out in [([a], theano.tensor.exp(a)),  # 1 input/1 outputs
                      ([a, b], a + b),  # 2 inputs
                      ((a, b), [a, b]),  # 2 outputs, 2 deepcopy ops
                      ((a, b), [a + b, a - b])]:
        linker = theano.gof.CLinker(c_callable=True)
        mode = theano.Mode(linker=linker)
        f = theano.function(inps, out, mode=mode,
                            on_unused_input='ignore')
        theano.printing.debugprint(f, print_type=True)
        filename = f.fn.filename  # with linker=c

        dir = os.path.split(filename)[0]
        p = subprocess.Popen(["make", "exec"], cwd=dir)
        p.wait()
        x = os.system(os.path.join(dir, 'exec'))
        assert x == 0, "The executable crashed!"

        # # Test raise error if no c code
        # try:
        #     theano.function((a,), [theano.tensor.argmax(a)],
        #                     theano.Mode(linker='c'),
        #                     on_unused_input='ignore')
        #     assert False, "Expected an error"
        # except NotImplementedError:
        #     pass

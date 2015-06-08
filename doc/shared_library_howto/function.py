import theano
import theano.tensor as T
import numpy as np

if __name__ == '__main__':
    x = T.matrix('x')
    y = T.vector('y')
    linker = theano.gof.CLinker(c_callable=True)
    mode = theano.Mode(linker=linker)
    f = theano.function([x, y], [2 * x, T.dot(y, x)], mode=mode)
    print f.fn.filename

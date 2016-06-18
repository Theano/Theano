import theano
import theano.tensor as T
import numpy as np

if __name__ == '__main__':
    x = T.matrix('x')
    W_values = np.asarray([1, 2, 3], dtype=theano.config.floatX)
    W = theano.shared(value=W_values, name='W', borrow=True)
    linker = theano.gof.CLinker(c_callable=True)
    mode = theano.Mode(linker=linker)
    f = theano.function([x], [2 * x, T.dot(W, x)], mode=mode)
    print(f.fn.filename)

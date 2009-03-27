import numpy
from theano.compile import Module, Method
import theano.tensor as T

class MatrixAccumulator(Module):

    def __init__(self):
        super(MatrixAccumulator, self).__init__() # don't forget this
        self.inc = T.dscalar()
        self.state = T.dmatrix()
        self.new_state = self.inc + self.state
        self.add = Method(inputs = self.inc,
                          outputs = self.new_state,
                          updates = {self.state: self.new_state})
        self.sub = Method(inputs = self.inc,
                          outputs = None,
                          updates = {self.state: self.state - self.inc})

    def _instance_print_state(self, acc):
        print '%s is: %s' % (self.state, acc.state)

    def _instance_initialize(self, acc, nrows, ncols):
        acc.state = numpy.zeros((nrows, ncols))

if __name__ == '__main__':
    m = Accumulator()
    acc = m.make(2, 5) # this calls m._instance_initialize(acc, 2, 5)

    acc.print_state()

    # OUTPUT:
    # state is: [[ 0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.]]

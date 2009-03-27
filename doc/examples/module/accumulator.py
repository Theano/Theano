from theano.compile import Module, Method
import theano.tensor as T

class Accumulator(Module):

    def __init__(self):
        super(Accumulator, self).__init__() # don't forget this
        self.inc = T.dscalar()
        self.state = T.dscalar()
        self.new_state = self.inc + self.state
        self.add = Method(inputs = self.inc,
                          outputs = self.new_state,
                          updates = {self.state: self.new_state})
        self.sub = Method(inputs = self.inc,
                          outputs = None,
                          updates = {self.state: self.state - self.inc})

if __name__ == '__main__':
    m = Accumulator()
    acc = m.make(state = 0)

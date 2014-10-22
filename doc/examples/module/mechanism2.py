from theano.compile import Module, ModuleInstance, Method
import theano.tensor as T

class AccumulatorInstance(ModuleInstance):

    def print_state(self):
        # self.component points to the Module from which this was compiled.
        print '%s is: %s' % (self.component.state, self.state)

class Accumulator(Module):

    # This line tells Theano to instantiate an AccumulatorInstance
    # when make() is called.
    InstanceType = AccumulatorInstance

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

if __name__=='__main__':
    m = Accumulator()
    acc = m.make(state = 0)
    acc.print_state() # --> prints "state is: 0.0"


import core
import gof
from numpy import random as r


# def rwrap(f):
#     wrapped = 
#     def ret(self, *args):
        
    

class RandomState(gof.Op, gof.ext.IONames):

    input_names = ['seed']

    def __init__(self, seed):
        inputs = [wrap(seed)]
        outputs = [ResultValue()]
        gof.Op.__init__(self, inputs, outputs)

    def thunk(self):
        def f():
            self.out.storage = r.RandomState(self.seed.storage)
        return f

    




class Random(object):

    def __init__(seed):
        self.state = core.wrap(seed)

    


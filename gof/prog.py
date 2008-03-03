

from env import Env
from utils import AbstractFunctionError


class Prog:

    def __init__(self, inputs, outputs, optimizer, linker_class, features = []):
        self.inputs = inputs
        if isinstance(outputs, dict):
            for name, output in outputs.items():
                setattr(self, name, output)
            self.outputs = outputs.values()
        else:
            self.outputs = outputs
        self.optimizer = optimizer
        self.env = Env(self.inputs, self.outputs, features, False)
        self.env.add_feature(EquivTool)
        self.linker = linker_class(self.env)

    def build(self):
        self.optimizer.optimize(self.env)
        

    def equiv(self, r):
        return self.env.equiv(r)

    def __getitem__(self, r):
        if isinstance(r, str):
            return getattr(self, r)
        else:
            return self.equiv(r)

    def __setitem__(self, r, value):
        if isinstance(r, tuple):
            for a, b in zip(r, value):
                self.__setitem__(a, b)
        else:
            self[r].data = value













# import compile

import env
import link
from features import EquivTool

class Prog:

    def __init__(self, inputs, outputs, optimizer, linker, features = []):
        self.optimizer = optimizer
        self.linker = linker

        features = set(features)
        features.add(EquivTool)
        self.env = env.Env(inputs, outputs, features) #, False)
        self.optimizer.optimize(self.env)
        self.perform = self.linker(self.env)
        self.outputs = outputs

#     def __optimize__(self):
#         self.optimizer.apply(self.env)
#         self.order = self.env.toposort()

    def equiv(self, r):
        return self.env.equiv(r)

    def __getitem__(self, r):
        return self.equiv(r)

    def __setitem__(self, r, value):
        if isinstance(r, tuple):
            for a, b in zip(r, value):
                self.__setitem__(a, b)
        else:
            self.equiv(r).set_value(value)

    def __call__(self, *args):
        self.perform()
        for output in self.outputs:
            output.set_value(self[output])
        return self.outputs
#        return [output for output in self.env.outputs]



#         if args:
#             for input, arg in zip(self.inputs, args):
#                 if arg is not None:
#                     input.value = arg
#         for thunk, op in zip(self.thunks, self.order):
#             try:
#                 thunk()
#             except Exception, e:
#                 raise e.__class__("Error in " + str(op) + ": " + str(e))
                
#         return [output.value for output in self.outputs]


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

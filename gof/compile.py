import env
import tools
import utils

class Compiler:
    """ What is this?  Please document.

    """

    def __init__(self, optimizer, features):
        self.features = set(features)
        self.features.update(optimizer.require())
        self.optimizer = optimizer

    def compile(self, inputs, outputs, features):
        features = self.features.union(features)
        e = env.Env(inputs, outputs, features, False)
        self.optimizer.apply(e)
        if not e.consistent():
            raise env.InconsistencyError("The graph is inconsistent.")
        return e

    def __call__(self, inputs, outputs, features):
        return self.compile(inputs, outputs, features)






#     def __init__(self, inputs, outputs, preprocessors, features, optimizer):
#         self.inputs = inputs
#         self.outputs = outputs
#         self.features = features
#         self.optimizer = optimizer

#         features = features + [tools.EquivTool] + optimizer.require()
#         features = utils.uniq_features(features)
        
#         self.env = env.Env(inputs,
#                            outputs,
#                            features,
#                            False)

#         if not self.env.consistent():
#             raise env.InconsistencyError("The graph is inconsistent.")

#         self.__optimize__()
#         self.thunks = [op.thunk() for op in self.order]

#     def __optimize__(self):
#         self.optimizer.apply(self.env)
#         self.order = self.env.toposort()

#     def equiv(self, r):
#         return self.env.equiv(r)

#     def __getitem__(self, r):
#         return self.equiv(r)

#     def __setitem__(self, r, value):
#         if isinstance(r, tuple):
#             for a, b in zip(r, value):
#                 self.__setitem__(a, b)
#         else:
#             self.equiv(r).set_value(value)

#     def __call__(self, *args):
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

















# import env
# import opt
# from value import AsValue


# class Prog:

#     def __init__(self, inputs, outputs, optimizer):
#         self.inputs = inputs
#         self.outputs = outputs
#         self.env = env.Env(inputs,
#                            outputs,
#                            False,
#                            op_db = env.OpDb,
#                            changed = env.ChangeListener,
# #                           pr = env.PrintListener,
#                            scope = env.ScopeListener)
# ##        self.adjustments = adjustments
#         self.optimizer = optimizer
# ##        if self.adjustments:
# ##            self.adjustments.apply(self.env)
#         if not self.env.consistent():
#             raise env.InconsistencyError("The graph is inconsistent.")
#         self.optimizer.apply(self.env)
#         self.order = self.env.toposort()
#         print "==================="
#         for op in self.order:
#             print op
#         print "==================="
#         self.thunks = [op.thunk() for op in self.order]

#     def equiv(self, v):
#         v = AsValue(v)
#         return self.env.equiv(v)

#     def __getitem__(self, v):
#         return self.equiv(v).storage

#     def __setitem__(self, v, value):
#         if isinstance(v, tuple):
#             for a, b in zip(v, value):
#                 self.__setitem__(a, b)
#         else:
#             self.equiv(v).value = value

#     def __call__(self, *args):
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


# def prog(i, o):
#     if not isinstance(i, (list, tuple)):
#         i = [i]
#     if not isinstance(o, (list, tuple)):
#         o = [o]

#     i = [AsValue(input) for input in i]
#     o = [AsValue(output) for output in o]

#     return Prog(i,
#                 o,
#                 opt.TagFilterMultiOptimizer(opt.opt_registry, None, None))



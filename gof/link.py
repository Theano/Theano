
# from features import Tool

from utils import AbstractFunctionError
import utils


class Linker:

    def __init__(self, env):
        self.env = env

    def make_thunk(self, inplace = False):
        """
        This function must return a triplet (function, input_results, output_results)
        where function is a thunk that operates on the returned results. If inplace
        is True, the input_results and output_results lists will be the same as the
        inputs and outputs of the graph provided to the Linker. Else, independent
        results will be returned.

        Example:
         e = x + y
         env = Env([x, y], [e])
         fn, (new_x, new_y), (new_e, ) = MyLinker(env).make_thunk(inplace)
         new_x.data = 1.0
         new_y.data = 2.0
         fn()
         print new_e.data # 3.0
         print e.data # 3.0 iff inplace == True (else unknown)
        """
        raise AbstractFunctionError()

    def make_function(self, inplace = False):
        """
        Returns a function that takes values corresponding to the inputs of the
        env used by this Linker and returns values corresponding the the outputs
        of that env. If inplace is True, the calculations will operate in the
        same storage the env uses, else independent storage will be allocated
        for the function.
        
        Example:
         e = x + y
         env = Env([x, y], [e])
         fn = MyLinker(env).make_function(inplace)
         print fn(1.0, 2.0) # 3.0
         print e.data # 3.0 iff inplace == True (else unknown)
        """
        thunk, inputs, outputs = self.make_thunk(inplace)

        def execute(*args):
            for arg, result in zip(args, inputs):
                result.data = arg
            thunk()
            return utils.to_return_values([result.data for result in outputs])

        return execute





class PerformLinker(Linker):

    def make_thunk(self, inplace = False):
        if inplace:
            env = self.env
        else:
            env = self.env.clone(True)
        order = env.toposort()
        thunks = [op.perform for op in order]
        def f():
            for thunk in thunks:
                thunk()
        return f, env.inputs, env.outputs

#         self.thunk = f
#         self.order = order
#         self.thunks = thunks


class ProfilePerformLinker(Linker):

    def compile(self):
        order = self.env.toposort()
        thunks = [op.perform for op in order]
        self.n_calls = 0
        self.n_thunks = 0
        self.times = [0.0 for op in self.order]
        def f():
            for thunk in thunks:
                thunk()
        self.thunk = f
        self.order = order
        self.thunks = thunks
    
    def slow_call(self):
        """Run the program, timing each thunk."""
        for i, thunk in enumerate(self.thunks):
            start_time = time.time()
            thunk()
            self.times[i] += time.time() - start_time
            self.n_thunks += 1
        self.n_calls += 1

    def fast_call(self):
        """Run the program, but only time the entire loop."""
        start_time = time.time()
        for thunk in self.thunks:
            thunk()
        self.n_thunks += len(self.thunks)
        self.n_calls += 1
        self.times[0] += time.time() - start_time

    __call__ = slow_call

    def dump(self, proportion=True):
        """Print statistics accumulated so far."""
        total_time = sum(self.times)
        print self.n_calls, 'calls took', total_time, 'seconds to evaluate',
        print self.n_thunks, 'thunks'

        if 0:
            print 'Proportion of CPU per op'
            for op, t in zip(self.order, self.times):
                s_op = str(op).split()[0][1:]
                print "  %-35s %4.5f"% (s_op, t/total_time)

        print 'Proportion of CPU per op class'
        dct = {}
        for op, t in zip(self.order, self.times):
            s_op = str(op).split()[0][1:]
            dct[s_op] = dct.get(s_op, 0.0) + t
        for t, s_op in reversed(sorted([(t,op) for op, t in dct.items()])):
            if proportion:
                print "  %-35s %4.5f"% (s_op, t/total_time)
            else:
                print "  %-35s %4.5f"% (s_op, t)

    



# class Linker(Tool):

#     def compile(self):
#         raise AbstractFunctionError()

#     def run(self):
#         raise AbstractFunctionError()

    




# def perform_linker(env, target = None):
#     order = env.toposort()
#     thunks = [op.perform for op in order]
#     def ret():
#         for thunk in thunks:
#             thunk()
#     if not target:
#         return ret
#     else:
#         raise NotImplementedError("Cannot write thunk representation to a file.")


# def perform_linker_nochecks(env, target = None):
#     order = env.toposort()
#     thunks = [op._perform for op in order]
#     def ret():
#         for thunk in thunks:
#             thunk()
#     if not target:
#         return ret
#     else:
#         raise NotImplementedError("Cannot write thunk representation to a file.")


# def cthunk_linker(env):
#     order = env.toposort()
#     thunks = []
#     cstreak = []

#     def append_cstreak():
#         if cstreak:
#             thunks.append(cutils.create_cthunk_loop(*cstreak))
#             cstreak = []
#     def ret():
#         for thunk in thunks:
#             thunk()

#     for op in order:
#         if hasattr(op, 'cthunk'):
#             cstreak.append(op.cthunk())
#         else:
#             append_cstreak()
#             thunks.append(op.perform)

#     if len(thunks) == 1:
#         return thunks[0]
#     else:
#         return ret


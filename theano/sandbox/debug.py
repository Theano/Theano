
from theano import gof
import sys


class DebugException(Exception):
    pass

class DebugLinker(gof.WrapLinker):

    def __init__(self,
                 linkers,
                 debug_pre = [],
                 debug_post = [],
                 copy_originals = False,
                 check_types = True,
                 compare_variables = True,
                 compare_fn = (lambda x, y: x == y)):
        gof.WrapLinker.__init__(self,
                                linkers = linkers,
                                wrapper = self.wrapper)

        self.env = None

        self.compare_fn = compare_fn
        
        self.copy_originals = copy_originals
        if check_types not in [None, True]:
            self.check_types = check_types
        if compare_variables not in [None, True]:
            self.compare_variables = compare_variables

        if not isinstance(debug_pre, (list, tuple)):
            debug_pre = [debug_pre]
        self.debug_pre = debug_pre

        if not isinstance(debug_post, (list, tuple)):
            debug_post = [debug_post]
        self.debug_post = debug_post
        if check_types is not None:
            self.debug_post.append(self.check_types)
        if compare_variables is not None:
            self.debug_post.append(self.compare_variables)

    def accept(self, env, no_recycling = []):
        return gof.WrapLinker.accept(self,
                                     env = env,
                                     no_recycling = no_recycling)

    def store_value(self, i, node, *thunks):
        th1 = thunks[0]
        for r, oval in zip(node.outputs, th1.outputs):
            r.step = i
            r.value = oval[0]
            if self.copy_originals:
                r.original_value = copy(oval[0])

    def check_types(self, i, node, *thunks):
        for thunk, linker in zip(thunks, self.linkers):
            for r in node.outputs:
                try:
                    r.type.filter(r.value, strict = True)
                except TypeError, e:
                    exc_type, exc_value, exc_trace = sys.exc_info()
                    exc = DebugException(e, "The output %s was filled with data with the wrong type using linker " \
                                         ("%s. This happened at step %i of the program." % (r, linker, i)) + \
                                         "For more info, inspect this exception's 'original_exception', 'debugger', " \
                                         "'output_at_fault', 'step', 'node', 'thunk' and 'linker' fields.")
                    exc.debugger = self
                    exc.original_exception = e
                    exc.output_at_fault = r
                    exc.step = i
                    exc.node = node
                    exc.thunk = thunk
                    exc.linker = linker
                    raise DebugException, exc, exc_trace

    def compare_variables(self, i, node, *thunks):
        thunk0 = thunks[0]
        linker0 = self.linkers[0]
        for thunk, linker in zip(thunks[1:], self.linkers[1:]):
            for o, output0, output in zip(node.outputs, thunk0.outputs, thunk.outputs):
                if not self.compare_fn(output0[0], output[0]):
                    exc = DebugException(("The variables from %s and %s for output %s are not the same. This happened at step %i." % (linker0, linker, o, step)) + \
                                         "For more info, inspect this exception's 'debugger', 'output', 'output_value1', 'output_value2', " \
                                         "'step', 'node', 'thunk1', 'thunk2', 'linker1' and 'linker2' fields.")
                    exc.debugger = self
                    exc.output = o
                    exc.output_value1 = output0
                    exc.output_value2 = output
                    exc.step = i
                    exc.node = node
                    exc.thunk1 = thunk0
                    exc.thunk2 = thunk
                    exc.linker1 = linker0
                    exc.linker2 = linker
                    raise exc

    def pre(self, f, inputs, order, thunk_groups):
        env = f.env
        for r in env.variables:
            if r.owner is None:
                r.step = "value" # this will be overwritten if r is an input
            else:
                r.step = None
            r.value = None
            r.original_value = None
            if r.owner is None and r not in env.inputs:
                r.value = r.data
                if self.copy_originals:
                    r.original_value = copy(r.data)
        for idx, (i, r) in enumerate(zip(inputs, env.inputs)):
            r.step = "input %i" % idx
            r.value = i
            if self.copy_originals:
                r.original_value = copy(i)
        for node, thunk_group in zip(order, thunk_groups):
            node.step = None

    def wrapper(self, i, node, *thunks):
        try:
            node.step = i
            for f in self.debug_pre:
                f(i, node, *thunks)
            for thunk in thunks:
                thunk()
            self.store_value(i, node, *thunks)
            for f in self.debug_post:
                f(i, node, *thunks)
        except Exception, e:
            exc_type, exc_value, exc_trace = sys.exc_info()
            if isinstance(e, DebugException):
                raise
            exc = DebugException(e, ("An exception occurred while processing node %s at step %i of the program." % (node, i)) + \
                                 "For more info, inspect this exception's 'original_exception', 'debugger', 'step', 'node' and 'thunks' fields.")
            exc.debugger = self
            exc.original_exception = e
            exc.step = i
            exc.node = node
            exc.thunks = thunks
            raise DebugException, exc, exc_trace



def print_info(i, node, *thunks):
    print "step %i, node %s" % (i, node)

def print_from(i, node, *thunks):
    print "parents:", ", ".join(str(input.step) for input in node.inputs)

def print_input_shapes(i, node, *thunks):
    print "input shapes:", ", ".join(str(input.value.shape) if hasattr(input.value, 'shape') else 'N/A' for input in node.inputs)

def print_input_types(i, node, *thunks):
    print "input types:", ", ".join(str(type(input.value)) for input in node.inputs)

def print_sep(i, node, *thunks):
    print "==================================="

import numpy
def numpy_compare(a, b, tolerance = 1e-6):
    if isinstance(a, numpy.ndarray):
        return (abs(a - b) <= tolerance).all()
    else:
        return a == b


def numpy_debug_linker(pre, post = []):
    return DebugLinker([gof.OpWiseCLinker],
                       pre,
                       post,
                       compare_fn = numpy_compare)



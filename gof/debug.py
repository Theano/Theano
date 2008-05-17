
import link


from functools import partial

class DebugException(Exception):
    pass

class DebugLinker(link.MetaLinker):

    def __init__(self,
                 env,
                 linkers,
                 debug_pre = [],
                 debug_post = [],
                 copy_originals = False,
                 check_types = True,
                 compare_results = True,
                 no_recycling = [],
                 compare_fn = lambda x, y: x == y):
        link.MetaLinker.__init__(self, env = env,
                                 linkers = linkers,
                                 wrapper = self.wrapper,
                                 no_recycling = no_recycling)

        self.compare_fn = compare_fn
        
        self.copy_originals = copy_originals
        if check_types not in [None, True]:
            self.check_types = check_types
        if compare_results not in [None, True]:
            self.compare_results = compare_results

        if not isinstance(debug_pre, (list, tuple)):
            debug_pre = [debug_pre]
        self.debug_pre = debug_pre

        if not isinstance(debug_post, (list, tuple)):
            debug_post = [debug_post]
        self.debug_post = debug_post
        if check_types is not None:
            self.debug_post.append(self.check_types)
        if compare_results is not None:
            self.debug_post.append(self.compare_results)

    def store_value(self, i, node, *thunks):
        th1 = thunks[0]
        for r, oval in zip(node.outputs, th1.outputs):
            r.step = i
            r.value = oval[0]
            if self.copy_originals:
                r.original_value = copy(oval[0])

    def check_types(self, debug, i, node, *thunks):
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

    def compare_results(self, debug, i, node, *thunks):
        thunk0 = thunks[0]
        linker0 = self.linkers[0]
        for thunk, linker in zip(thunks[1:], self.linkers[1:]):
            for o, output0, output in zip(node.outputs, thunk0.outputs, thunk.outputs):
                if not self.compare_fn(output0[0], output[0]):
                    exc = DebugException(("The results from %s and %s for output %s are not the same. This happened at step %i." % (linker0, linker, o, step)) + \
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
        for r in env.results:
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

    def wrapper(self, th, i, node, *thunks):
        try:
            node.step = i
            for f in self.debug_pre:
                f(th, i, node, *thunks)
            for thunk in thunks:
                thunk()
            self.store_value(i, node, *thunks)
            for f in self.debug_post:
                f(th, i, node, *thunks)
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

    def make_thunk(self, **kwargs):
        inplace = kwargs.pop("inplace", False)
        
        if inplace:
            e, equiv = self.env, None
        else:
            e, equiv = self.env.clone_get_equiv()

        class Debug:
            def __init__(self, thunk, env, equiv):
                self.thunk = thunk
                self.env = env
                self.equiv = equiv
            def __call__(self):
                self.thunk()
            def __getitem__(self, item):
                equiv = self.equiv
                if not isinstance(item, Apply) and not isinstance(item, Result):
                    raise TypeError("__getitem__ expects an Apply or Result instance.")
                if not hasattr(item, 'env') or item.env is not e:
                    if equiv is None:
                        raise Exception("item does not belong to this graph and has no equivalent")
                    else:
                        return equiv[item]
                else:
                    return item

        bk = self.no_recycling
        self.no_recycling = map(equiv.__getitem__, self.no_recycling)
        th, inputs, outputs = link.MetaLinker.make_thunk(self, alt_env = e, wrapf = lambda f: Debug(f, e, equiv), **kwargs)
        self.no_recycling = bk
        
        return th, inputs, outputs




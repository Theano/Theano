""" Provides `DebugMode`


"""
import time, copy, sys
from StringIO import StringIO

from .. import gof

from ..gof import Env, graph, utils, link
from ..gof.link import WrapLinkerMany, raise_with_op
from ..gof.cutils import run_cthunk
from ..gof.cc import OpWiseCLinker, CLinker
import numpy

from ..compile.function_module import (FunctionMaker,
        Function, 
        infer_reuse_pattern,
        SymbolicInput,
        SymbolicInputKit,
        SymbolicOutput,
        Supervisor)

class DebugModeError(Exception):
    pass

class BadClinkerOutput(DebugModeError):
    """Exception: a c implementation and python implementation don't agree"""

    r = None
    """TODO"""

    a = None
    """TODO"""

    b = None
    """TODO"""

    def __init__(self, r, a, b):
        """Initialize members"""
        super(BadClinkerOutput, self).__init__()
        self.r = r
        self.a = a
        self.b = b

class BadOptimization(DebugModeError):
    """Exception: some result and its substitute take different runtime values."""

    new_r = None
    """TODO"""

    r_val = None
    """TODO"""

    new_r_val = None
    """TODO"""

    reasons = []
    """TODO"""

    snapshots = []
    """TODO"""

    def __init__(self, new_r, r_val, new_r_val, reasons, snapshots):
        """Initialize members"""
        super(BadOptimization, self).__init__()
        self.new_r = new_r
        self.r_val = r_val
        self.new_r_val = new_r_val
        self.reasons = reasons
        self.snapshots = snapshots

    #def __str__(self):
        #return self.str_diagnostic() #debatable...

    def str_diagnostic(self):
        """TODO: what does this mean?  How to interpret?  """
        sio = StringIO()
        print >> sio, "  Result:", id(self.new_r), self.new_r 
        print >> sio, "  Op", self.new_r.owner
        print >> sio, "  Value Type:", type(self.new_r_val)
        print >> sio, "  Old Value: ", self.r_val
        print >> sio, "  Value: ", self.new_r_val
        print >> sio, "  Reason: ", [(str(reason), id(old_r)) for reason, old_r in self.reasons[self.new_r]]
        print >> sio, "  Snapshots:"
        for s in self.snapshots[self.new_r]:
            print >> sio, "  BEFORE"
            print >> sio, s[1]
            print >> sio, "  AFTER"
            print >> sio, s[2]
        return sio.getvalue()


def debugprint(a, prefix='', depth=-1, done=None, file=sys.stdout):
    if depth==0:
        return
    done = set() if done is None else done
    if hasattr(a, 'op'):
        print >> file, prefix, a.op, id(a)
        if id(a) not in done:
            done.add(id(a))
            for i in a.inputs:
                if i.owner:
                    debugprint(i.owner, prefix+'  ', depth=depth-1, done=done, file=file)
                else:
                    print >> file, prefix+'  ', i, id(i)
    else:
        print >> file, prefix+'  ', a, id(a)

    return file

class Event(object):
    def __init__(self, kind, node, idx=None, reason=None):
        self.kind = kind
        if node == 'output':
            self.node = 'output'
            self.op = 'output'
        else:
            self.node = node
            self.op = node.op
        self.idx = idx
        self.reason = reason

    def __str__(self):
        if self.kind == 'change':
            return ' '.join(['change', 
                self.reason, 
                str(self.op), 
                str(self.idx),
                str(len(self.node.inputs))])
        else:
            return str(self.__dict__)

    def __eq__(self, other):
        rval = type(self) == type(other) 
        if rval:
            for attr in ['kind', 'op', 'idx', 'reason']:
                rval = rval and getattr(self, attr) == getattr(other, attr)
        return rval

    def __ne__(self, other):
        return not (self == other)

class ResultEquivalenceTracker(object):
    def __init__(self):
        self.env = None

    def on_attach(self, env):
        assert self.env is None
        self.equiv = {}
        self.active_nodes = set()
        self.inactive_nodes = set()
        self.env = env
        self.all_results_ever = []
        self.reasons = {}
        self.replaced_by = {}
        self.snapshots = {}
        self.event_list = []

    def on_detach(self, env):
        assert env is self.env
        self.env = None

    def on_prune(self, env, node):
        self.event_list.append(Event('prune', node))
        #print 'PRUNING NODE', node, id(node)
        assert node in self.active_nodes
        assert node not in self.inactive_nodes
        self.active_nodes.remove(node)
        self.inactive_nodes.add(node)

    def on_import(self, env, node):
        self.event_list.append(Event('import', node))

        #print 'NEW NODE', node, id(node)
        assert node not in self.active_nodes
        self.active_nodes.add(node)

        if node in self.inactive_nodes:
            self.inactive_nodes.remove(node)
            for r in node.outputs:
                assert r in self.equiv
        else:
            for r in node.outputs:
                assert r not in self.equiv
                self.equiv[r] = set([r])
                self.all_results_ever.append(r)
                self.reasons.setdefault(r, [])
                self.replaced_by.setdefault(r, [])
                self.snapshots.setdefault(r, [])
            for r in node.inputs:
                self.reasons.setdefault(r, [])
                self.replaced_by.setdefault(r, [])
                self.snapshots.setdefault(r, [])

    def on_change_input(self, env, node, i, r, new_r, reason=None):
        #print 'CHANGE by', reason, 'to use', new_r, type(new_r)
        self.event_list.append(Event('change', node, reason=str(reason), idx=i))

        self.reasons.setdefault(new_r, [])
        self.replaced_by.setdefault(new_r, [])
        self.snapshots.setdefault(new_r, [])
        if (reason, r) not in self.reasons[new_r]:
            self.reasons[new_r].append((reason, r))
            self.replaced_by[r].append((reason, new_r))
            self.snapshots[new_r].append((
                reason, 
                debugprint(r.owner, prefix='  ', depth=6, file=StringIO()).getvalue(),
                debugprint(new_r.owner,prefix='  ',  depth=6, file=StringIO()).getvalue()))

        if r in self.equiv:
            r_set = self.equiv[r]
        else:
            r_set = self.equiv.setdefault(r, set([r]))
            self.all_results_ever.append(r)

        if new_r in self.equiv:
            new_r_set = self.equiv[new_r]
        else:
            new_r_set = self.equiv.setdefault(new_r, set([new_r]))
            self.all_results_ever.append(new_r)

        assert new_r in new_r_set
        assert r in r_set


        # update one equivalence set to contain the other
        # transfer all the elements of the old one to the new one
        r_set.update(new_r_set)
        for like_new_r in new_r_set:
            self.equiv[like_new_r] = r_set
            assert like_new_r in r_set

        assert self.equiv[r] is r_set
        assert self.equiv[new_r] is r_set

    def printstuff(self):
        for key in self.equiv:
            print key
            for e in self.equiv[key]:
                print '  ', e

def optcheck_env(input_specs, output_specs, accept_inplace = False):
    orig_inputs = [spec.result for spec in input_specs]
    updates = [spec.update for spec in input_specs if spec.update]
    orig_outputs = [spec.result for spec in output_specs] + updates

    inputs, outputs = gof.graph.clone(orig_inputs, orig_outputs)
    equivalence_tracker = ResultEquivalenceTracker()
    env = gof.env.Env(inputs, outputs,
            features=[equivalence_tracker,
                gof.DestroyHandler(do_imports_on_attach=False)])

    if not accept_inplace:
        for node in env.nodes:
            if getattr(node.op, 'destroy_map', None):
                raise TypeError("Graph must not contain inplace operations", node)

    # We need to protect all immutable inputs from inplace operations.
    env.extend(Supervisor(input for spec, input in zip(input_specs, inputs) if not (spec.mutable or (hasattr(env, 'destroyers') and env.destroyers(input)))))
    return env, map(SymbolicOutput, updates), equivalence_tracker


class DebugModeLinker(gof.link.LocalLinker):
    def __init__(self, maker):
        super(gof.LocalLinker, self).__init__()
        self.env = None
        self.maker = maker

    def accept(self, env, no_recycling = []):
        if self.env is not None and self.env is not env:
            assert type(self) is DebugModeLinker
            return type(self)(self.env, self.maker).accept(env, no_recycling)
        self.env = env
        self.no_recycling = no_recycling
        return self

    def make_all(self, profiler = None, input_storage = None, output_storage = None):
        env = self.env
        #order = env.toposort()

        #Compute a topological ordering that IGNORES the destroy_map of destructive Ops.
        #This will be OK, because every thunk is evaluated on a copy of its input.
        # If the copy.copy function produces an object that is aliased to the original one,
        # then this evaluation mode will not work.  It works for ndarrays.
        order_outputs = copy.copy(env.equivalence_tracker.all_results_ever)
        order_outputs.reverse()
        order = graph.io_toposort(env.inputs, order_outputs)

        no_recycling = self.no_recycling

        input_storage, output_storage, storage_map = link.map_storage(env, order, input_storage, output_storage)

        thunks_py = [] #python thunks
        thunks_c = [] #c thunks
        for node in order:
            node_input_storage = [storage_map[r] for r in node.inputs]
            node_output_storage = [storage_map[r] for r in node.outputs]
            try:
                if not self.maker.mode.check_c_code:
                    raise utils.AbstractFunctionError()
                e = Env(*graph.clone(node.inputs, node.outputs))
                e.toposort = lambda: e.nodes #WARNING: STOCHASTIC ORDER

                if any(isinstance(input, graph.Value) for input in node.inputs):
                    desc = None
                else:
                    desc = (node.op,
                            tuple(input.type for input in node.inputs),
                            tuple(input.type for input in node.inputs),
                            tuple(output in no_recycling for output in node.outputs),
                            tuple(node.inputs.count(input) for input in node.inputs))

                try:
                    cl = self.__cache__.get(desc)
                except Exception, exc:
                    #print >> sys.stderr, "INFO: failed to hash %s: %s. Node will not be cached." % (node, exc)
                    cl = None
                if cl is None:
                    cl = CLinker().accept(e, [r for r, r2 in zip(e.outputs, node.outputs) if r2 in no_recycling])
                    if desc is not None:
                        try:
                            self.__cache__[desc] = cl
                        except:
                            pass

                thunk, node_input_filters, node_output_filters = cl.make_thunk(
                    input_storage = node_input_storage,
                    output_storage = node_output_storage)
                thunk.inputs = node_input_storage
                thunk.outputs = node_output_storage
                thunks_c.append(thunk)
            except utils.AbstractFunctionError:
                thunks_c.append(None)


            p = node.op.perform
            thunk = (lambda p = p, i = node_input_storage, o = node_output_storage, n =
                    node: p(n, [x[0] for x in i], o))
            thunk.inputs = node_input_storage
            thunk.outputs = node_output_storage
            thunk.perform = p
            thunks_py.append(thunk)

        if no_recycling is True:
            no_recycling = storage_map.values()
            no_recycling = utils.difference(no_recycling, input_storage)
        else:
            no_recycling = [storage_map[r] for r in no_recycling if r not in env.inputs]

        #####
        # This is the function that runs when you evaluate the graph
        #####
        def f():
            for x in no_recycling:
                x[0] = None

            equiv_vals = {}
            problematic = set()
            # r_vals are the true values associated with each result in the graph
            # they should not change during the evaluation of this function, even when the
            # graph has destructive ops in it
            #
            # This dictionary is used to populate the storage_map as necessary
            r_vals = {} 
            assert len(thunks_py) == len(order)

            #put the initial values into the r_vals
            for r in storage_map:
                if storage_map[r][0] is not None:
                    r_vals[r] = copy.copy(storage_map[r][0])
                    storage_map[r][0] = None

            try:
                # compute the value of all results
                for i, (thunk_py, thunk_c, node) in enumerate(zip(thunks_py, thunks_c, order)):

                    #put a copy of each input into the storage_map
                    for r in node.inputs:
                        storage_map[r][0] = copy.copy(r_vals[r])

                    thunk_py()

                    #retrieve a copy of each output from the storage_map
                    for r in node.outputs:
                        if r in r_vals:
                            # r has been constant-folded
                            if not r.type.values_eq_enough(r_vals[r], storage_map[r][0]):
                                raise Exception('BadConstantFold', (r,  r_vals[r],
                                    storage_map[r][0])) #TODO: make a proper exception class for this

                        else:
                            r_vals[r] = copy.copy(storage_map[r][0])

                    if thunk_c:
                        for r in node.outputs:
                            storage_map[r][0] = None #clear the storage_map for the thunk_c

                        if 0:
                            # TODO: check that Op didn't change any inputs that it wasn't allowed to
                            # (Hint: use the destroy_map attribute)
                            raise NotImplementedError()
                        else:
                            for r in node.inputs:
                                storage_map[r][0] = copy.copy(r_vals[r])

                        thunk_c()

                        for r in node.outputs:
                            # compares the version from thunk_py (in r_vals)
                            # to the version produced by thunk_c (in storage_map)
                            if not r.type.values_eq_enough(r_vals[r], storage_map[r][0]):
                                raise BadClinkerOutput(r, r_vals[r], storage_map[r][0])

            except:
                raise_with_op(node)

            # iterate over results looking for values that don't match the values of the
            # results they replaced.  This is the sign of a broken optimization.
            for i, node in enumerate(order):
                for new_r in node.outputs:
                    for reason, r in env.equivalence_tracker.reasons[new_r]:
                        problem = False

                        #check if the value for new_r doesn't match the value for r
                        new_r_val = r_vals[new_r]
                        r_val = r_vals[r]
                        assert r.type == new_r.type

                        if not r.type.values_eq_enough(r_val, new_r_val):
                            raise BadOptimization(new_r, r_val, new_r_val,
                                    env.equivalence_tracker.reasons,
                                    env.equivalence_tracker.snapshots)

        f.allow_gc = True
        return f, [link.Container(input, storage) for input, storage in zip(env.inputs, input_storage)], \
            [link.Container(output, storage, True) for output, storage in zip(env.outputs, output_storage)], \
            thunks_py, order

NODEFAULT = ['NODEFAULT']
class DebugModeFunctionMaker(FunctionMaker): #inheritance buys a few helper functions
    verbose = 0
    """Verbosity level of compile-time and run-time checks. (Default 0: silent)"""


    def __init__(self, inputs, outputs, optimizer, mode,
            accept_inplace = False, 
            function_builder = Function):
        """
        :type inputs: a list of SymbolicInput instances

        :type outputs: a list of SymbolicOutput instances
                    outputs may also be a single Result (not a list), in which
                    case the functions produced by FunctionMaker will return
                    their output value directly

        :param accept_inplace: True iff it is acceptable to have inplace operations
                    in the graph from the inputs to the outputs
        """

        # Handle the case where inputs and/or outputs is a single Result (not in a list)
        unpack_single = False
        if not isinstance(outputs, (list, tuple)):
            unpack_single = True
            outputs = [outputs]
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        # Wrap them in In or Out instances if needed.
        inputs, outputs =  map(self.wrap_in, inputs), map(self.wrap_out, outputs)
        _inputs = gof.graph.inputs([o.result for o in outputs] + [i.update for i in inputs if getattr(i, 'update', False)])
        indices = [[input] + self.expand_in(input, _inputs) for input in inputs]
        expanded_inputs = reduce(list.__add__, [list(z) for x, y, z in indices], [])

        # make the env
        for i in xrange(mode.stability_patience):
            env, additional_outputs, equivalence_tracker = optcheck_env(expanded_inputs, outputs, accept_inplace)
            env.equivalence_tracker = equivalence_tracker
            # optimize the env
            optimizer(env)
            if i:
                li = env.equivalence_tracker.event_list
                l0 = env0.equivalence_tracker.event_list
                if li != l0 :
                    print >> sys.stderr, "WARNING: Optimization process is unstable"
                    for j in xrange(max(len(li), len(l0))):
                        if li[j] != l0[j]:
                            print >> sys.stderr, "* ", j
                            print >> sys.stderr, "  ", str(li[j]) if j < len(li) else '-'
                            print >> sys.stderr, "  ", str(l0[j]) if j < len(l0) else '-'
                        else:
                            pass

                    print >> sys.stderr, "EXITING"
                    sys.exit(1)
                    break
                else:
                    if self.verbose:
                        print >> sys.stdout, "OPTCHECK: optimization", i, "of", len(li), "events was stable."
            else:
                env0 = env


        del env0
        self.env = env
        #equivalence_tracker.printstuff()

        linker = DebugModeLinker(self)


        #the 'no_borrow' outputs are the ones for which that we can't return the internal storage pointer.
        no_borrow = [output for output, spec in zip(env.outputs, outputs+additional_outputs) if not spec.borrow]
        if no_borrow:
            self.linker = linker.accept(env, no_recycling = infer_reuse_pattern(env, no_borrow))
        else:
            self.linker = linker.accept(env)
        
        self.indices = indices
        self.inputs = inputs
        self.expanded_inputs = expanded_inputs
        self.outputs = outputs
        self.unpack_single = unpack_single
        self.accept_inplace = accept_inplace
        self.function_builder = function_builder
        self.mode = mode

    def create(self, defaults = None, trustme = False):
        """
        Create a function.

        defaults -> a list matching the inputs list and providing default values
                    if the default for an input is None, then that input is a
                    required input. For an input with an update, the default
                    acts as initialization.
        trustme -> disables some exceptions, used internally
        """
        if defaults is None:
            defaults = [None]*len(self.inputs)
        input_storage = [] # list of independent one-element lists, will be passed to the linker
        _defaults = []

        # The following loop is to fill in the input_storage and _defaults lists.
        for (input, indices, subinputs), default in zip(self.indices, defaults):
            __default = default

            if isinstance(default, gof.Container):
                # If the default is a gof.Container, this means we want to share
                # the same storage. This is done by appending default.storage
                # to input_storage
                if indices is not None:
                    raise TypeError("Cannot take a Container instance as default for a SymbolicInputKit.")
                input_storage.append(default.storage)
                default = None
                required = False
            elif isinstance(input, SymbolicInputKit):
                # If the input is a SymbolicInputKit, it represents more than
                # one storage unit. The indices and subinputs lists represent which
                # of the kit's inputs are active in this graph, so we make as many
                # storage units as needed
                if isinstance(default, (list, tuple)) \
                        and all(isinstance(x, gof.Container) for x in default):
                    if len(default) == len(indices):
                        input_storage += [x.storage for x in default]
                    elif len(default) > len(indices):
                        input_storage += [default[i].storage for i in indices]
                    else:
                        raise ValueError('Not enough storage for SymbolicInputKit', input, indices, default)
                    default = NODEFAULT
                else:
                    input_storage += [[None] for i in indices]
            else:
                # Normal case: one new, independent storage unit
                input_storage.append([None])

            # Filling _defaults. Each entry is a tuple of three elements:
            # (required, refeed, value)
            # - required means that the user must provide a value when calling the function
            # - refeed means that we want to put the default back in the storage after each function call
            # - value is the value that will be put in the storage initially

            # Even though a SymbolicInputKit represents more than one input,
            # we still only have one entry for the defaults list.
            if isinstance(input, SymbolicInputKit):
                if default is NODEFAULT:
                    _defaults.append((False, False, None))
                elif default is None:
                    _defaults.append((True, True, None))
                else:
                    _defaults.append((False, False, default))
            elif input.update is not None:
                # If the input has an update, then (logically) it is not required since
                # it is just a parameter and of course we don't want to refeed the default
                # back into the storage as it would defeat the point of updating it. We
                # always do this policy.
                if default is None:
                    if trustme or isinstance(__default, gof.Container):
                        _defaults.append((False, False, None))
                    else:
                        # This might catch some bugs early
                        raise ValueError("A default (initial) value is required for an input which can update itself.", input)
                else:
                    _defaults.append((False, False, default))
            else:
                if default is None:
                    if trustme or isinstance(__default, gof.Container):
                        _defaults.append((False, False, None))
                    else:
                        # No default, so this is a required input. Nothing to feed back, initial value is None.
                        _defaults.append((True, False, None))
                else:
                    # Default value. It is not required, but we want to put it back into the storage
                    # everytime so it behaves like most programming languages' default values
                    _defaults.append((False, True, default))
        defaults = _defaults

        # Get a function instance
        _fn, _i, _o = self.linker.make_thunk(input_storage = input_storage)
        fn = self.function_builder(_fn, _i, _o, self.indices, self.outputs, defaults, self.unpack_single, self)
        return fn



from ..compile.mode import Mode, register_mode
class DebugMode(Mode):
    """Evaluation Mode that detects optimization errors.

    A basic premise of how theano works is that every node that is replaced during optimization should compute the same thing as its replacement.

    Normally such replacements run instead of the originals.
    This Mode runs the original and the replacement, and then checks that they both compute the
    same thing.
    If their values are different, the optimization that created the replacement is probably
    broken.

    """
    # This function will be used to create a FunctionMaker in 
    # function_module.function
    def function_maker(self, i,o,m, *args, **kwargs):
        assert m is self
        return DebugModeFunctionMaker(i, o, self.optimizer, self, *args, **kwargs)
    def __init__(self, 
            optimizer='fast_run', 
            stability_patience=10,
            check_c_code=True):
        super(DebugMode, self).__init__(
                optimizer=optimizer,
                linker=DebugModeLinker)
        self.stability_patience = stability_patience
        self.check_c_code = check_c_code
register_mode('DEBUG_MODE',DebugMode(optimizer='fast_run'))

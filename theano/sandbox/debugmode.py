""" Mode that runs all nodes, even those which have been optimized out. 

A basic premise of how theano works is that every node that is replaced during optimization should compute the same thing as its replacement.

Normally theano's optimizations work by running such replacements instead of the originals.
This debugging tool does a different thing.  It runs the original and the replacement, and then
checks that they both compute the same thing.
If their values are different, the optimization that created the replacement is probably
broken.


"""
import time, copy, sys
from StringIO import StringIO

from .. import gof

from ..gof import Env, graph, utils, link
from ..gof.link import WrapLinkerMany, raise_with_op
from ..gof.cutils import run_cthunk
from ..gof.cc import OpWiseCLinker, CLinker
from ..compile.mode import Mode
import numpy


from ..compile.function_module import (convert_function_input, 
        FunctionMaker,
        predefined_modes,
        Function, 
        infer_reuse_pattern,
        SymbolicInput,
        SymbolicInputKit,
        SymbolicOutput,
        Supervisor)

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
        self.snapshots = {}

    def on_detach(self, env):
        assert env is self.env
        self.env = None

    def on_prune(self, env, node):
        #print 'PRUNING NODE', node, id(node)
        assert node in self.active_nodes
        assert node not in self.inactive_nodes
        self.active_nodes.remove(node)
        self.inactive_nodes.add(node)

    def on_import(self, env, node):
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
                self.snapshots.setdefault(r, [])
            for r in node.inputs:
                self.reasons.setdefault(r, [])
                self.snapshots.setdefault(r, [])

    def on_change_input(self, env, node, i, r, new_r, reason=None):
        #print 'CHANGE by', reason, 'to use', new_r, type(new_r)

        self.reasons.setdefault(new_r, [])
        self.snapshots.setdefault(new_r, [])
        if (reason, r) not in self.reasons[new_r]:
            self.reasons[new_r].append((reason, r))
            self.snapshots[new_r].append((
                reason, 
                debugprint(r.owner, prefix='  ', depth=6, file=StringIO()).getvalue(),
                debugprint(new_r.owner,prefix='  ',  depth=6, file=StringIO()).getvalue()))
            self.reasons[r].append(('replaced by', new_r))

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

class OptCheckLinker(OpWiseCLinker):
    def make_all(self, profiler = None, input_storage = None, output_storage = None):
        env = self.env
        #order = env.toposort()
        order_outputs = copy.copy(env.equivalence_tracker.all_results_ever)
        order_outputs.reverse()
        order = graph.io_toposort(env.inputs, order_outputs)
        no_recycling = self.no_recycling

        input_storage, output_storage, storage_map = link.map_storage(env, order, input_storage, output_storage)

        thunks = []
        for node in order:
            node_input_storage = [storage_map[r] for r in node.inputs]
            node_output_storage = [storage_map[r] for r in node.outputs]
            try:
                raise NotImplementedError('need to copy destroyed inputs')
                e = Env(*graph.clone(node.inputs, node.outputs))
                e.toposort = lambda: e.nodes

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
                thunks.append(thunk)
            except (NotImplementedError, utils.AbstractFunctionError):
                if self.fallback_on_perform:
                    p = node.op.perform
                    thunk = (lambda p = p, i = node_input_storage, o = node_output_storage, n =
                            node: p(n, [copy.copy(x[0]) for x in i], o))
                    thunk.inputs = node_input_storage
                    thunk.outputs = node_output_storage
                    thunk.perform = p
                    thunks.append(thunk)
                else:
                    raise

        if no_recycling is True:
            no_recycling = storage_map.values()
            no_recycling = utils.difference(no_recycling, input_storage)
        else:
            no_recycling = [storage_map[r] for r in no_recycling if r not in env.inputs]

        def f():
            for x in no_recycling:
                x[0] = None
            try:
                equiv_vals = {}
                problematic = set()
                r_vals = {}
                assert len(thunks) == len(order)
                for i, (thunk, node) in enumerate(zip(thunks, order)):
                    thunk()

                    for r in node.outputs:
                        r_set = env.equivalence_tracker.equiv[r]
                        this_r_val = copy.copy(storage_map[r][0])
                        r_vals[r] = this_r_val
                        assert this_r_val is not None

                        if id(r_set) not in equiv_vals:
                            #print 'get correct', r_set
                            equiv_vals[id(r_set)] = this_r_val
                        else:
                            correct_r_val = equiv_vals[id(r_set)]

                            # TODO:  use r.type.val_cmp(correct_r_val, this_r_val)
                            # That function doesn't exist yet though..
                            if type(correct_r_val) != type(this_r_val):
                                problematic.add(r)
                            elif type(correct_r_val) is numpy.ndarray:
                                if not numpy.allclose(correct_r_val, this_r_val):
                                    problematic.add(r)
                            else:
                                print 'Ignoring comparison of instances of', type(correct_r_val)

                if problematic:
                    #print out the summary of the first problematic equivalence group
                    min_member = []
                    for problem_r in problematic:
                        problem_r_set = env.equivalence_tracker.equiv[problem_r]
                        for i, n in enumerate(order):
                            if problem_r_set.intersection(n.outputs):
                                break
                        min_member.append((i, problem_r_set))
                    min_member.sort()

                    problematic_set = min_member[0][1]

                    print "OPTCHECK FAILURE"
                    for r in problematic_set:
                        print "  Result:", id(r), r 
                        print "  Op", r.owner
                        print "  Value Type:", type(r_vals[r])
                        print "  Value: ", r_vals[r]
                        print "  Reason: ", [(str(reason), id(old_r)) for reason, old_r in env.equivalence_tracker.reasons[r]]
                        print "  Snapshots:"
                        for s in env.equivalence_tracker.snapshots[r]:
                            print "  BEFORE"
                            print s[1]
                            print "  AFTER"
                            print s[2]
                        print ""

                    raise Exception("OptCheckFailure")

            except:
                raise_with_op(node)

        f.allow_gc = self.allow_gc

        return f, [link.Container(input, storage) for input, storage in zip(env.inputs, input_storage)], \
            [link.Container(output, storage, True) for output, storage in zip(env.outputs, output_storage)], \
            thunks, order

NODEFAULT = ['NODEFAULT']
class OptCheckFunctionMaker(FunctionMaker):

    def __init__(self, inputs, outputs, optimizer,
            accept_inplace = False, function_builder = Function):
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
        env, additional_outputs, equivalence_tracker = optcheck_env(expanded_inputs, outputs, accept_inplace)
        self.env = env

        linker = OptCheckLinker()

        # optimize the env
        optimizer(env)

        env.equivalence_tracker = equivalence_tracker
        #equivalence_tracker.printstuff()


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

class OptCheck(Mode):
    # This function will be used to create a FunctionMaker in 
    # function_module.function
    def function_maker(self, i,o,m, *args, **kwargs):
        assert m is self
        return OptCheckFunctionMaker(i, o, self.optimizer, *args, **kwargs)
    def __init__(self, optimizer='fast_run'):
        super(OptCheck, self).__init__(
                optimizer=optimizer,
                linker=OptCheckLinker)




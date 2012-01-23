"""
Defines the base class for optimizations as well as a certain
amount of useful generic optimization tools.
"""


import copy, logging, sys, time

import numpy

import graph
from env import InconsistencyError
import utils
import unify
import toolbox
import op
import theano
from theano import config
from theano.gof.python25 import any, all, deque
from theano.configparser import AddConfigVar, BoolParam, config

#if sys.version_info[:2] >= (2,5):
#  from collections import defaultdict

_logger = logging.getLogger('theano.gof.opt')

AddConfigVar('time_seq_optimizer',
        "Should SeqOptimizer print the time taked by each of its optimizer",
        BoolParam(False),
        in_c_key=False)

import destroyhandler as dh
import traceback

_optimizer_idx = [0]

def _list_of_nodes(env):
    return list(graph.io_toposort(env.inputs, env.outputs))

class Optimizer(object):
    """WRITEME
    An L{Optimizer} can be applied to an L{Env} to transform it.
    It can represent an optimization or in general any kind
    of transformation you could apply to an L{Env}.
    """

    def __hash__(self):
        if not hasattr(self, '_optimizer_idx'):
            self._optimizer_idx = _optimizer_idx[0]
            _optimizer_idx[0] += 1
        return self._optimizer_idx

    def apply(self, env):
        """WRITEME
        Applies the optimization to the provided L{Env}. It may use all
        the methods defined by the L{Env}. If the L{Optimizer} needs
        to use a certain tool, such as an L{InstanceFinder}, it can do
        so in its L{add_requirements} method.
        """
        pass

    def optimize(self, env, *args, **kwargs):
        """WRITEME
        This is meant as a shortcut to::
          opt.add_requirements(env)
          opt.apply(env)
        """
        self.add_requirements(env)
        self.apply(env, *args, **kwargs)

    def __call__(self, env):
        """WRITEME
        Same as self.optimize(env)
        """
        return self.optimize(env)

    def add_requirements(self, env):
        """WRITEME
        Add features to the env that are required to apply the optimization.
        For example:
          env.extend(History())
          env.extend(MyFeature())
          etc.
        """
        pass

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        name = getattr(self, 'name', None)
        print >> stream, "%s%s %s id=%i" %(' '*level, self.__class__.__name__,
                                           name, id(self))

class FromFunctionOptimizer(Optimizer):
    """WRITEME"""
    def __init__(self, fn):
        self.apply = fn
    def add_requirements(self, env):
        # Added by default
        #env.extend(toolbox.ReplaceValidate())
        pass

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print >> stream, "%s%s id=%i" %(' '*level,
                str(self.apply),
                id(self))

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

def optimizer(f):
    """decorator for FromFunctionOptimizer"""
    rval = FromFunctionOptimizer(f)
    rval.__name__ = f.__name__
    return rval



class SeqOptimizer(Optimizer, list):
    #inherit from Optimizer first to get Optimizer.__hash__
    """WRITEME
    Takes a list of L{Optimizer} instances and applies them
    sequentially.
    """
    @staticmethod
    def warn(exc, self, optimizer):
        """Default failure_callback for SeqOptimizer
        """
        _logger.error("SeqOptimizer apply %s"% str(optimizer))
        _logger.error("Traceback:")
        _logger.error(traceback.format_exc())
        if config.on_opt_error == 'raise':
            raise exc

    def __init__(self, *opts, **kw):
        """WRITEME"""
        if len(opts) == 1 and isinstance(opts[0], (list, tuple)):
            opts = opts[0]
        self[:] = opts
        self.failure_callback = kw.pop('failure_callback', None)

    def apply(self, env):
        """WRITEME
        Applies each L{Optimizer} in self in turn.
        """
        l=[]
        nb_node_before = len(env.nodes)
        for optimizer in self:
            try:
                t0=time.time()
                optimizer.optimize(env)
                l.append(float(time.time()-t0))
            except AssertionError: # do not catch Assertion failures
                raise
            except Exception, e:
                if self.failure_callback:
                    self.failure_callback(e, self, optimizer)
                    continue
                else:
                    raise
        if config.time_seq_optimizer:
            print "SeqOptimizer",
            if hasattr(self,"name"): print self.name,
            elif hasattr(self,"__name__"): print self.__name__,
            print " time %.3fs for %d/%d nodes before/after optimization"%(sum(l),nb_node_before,len(env.nodes))

            ll=[]
            for opt in self:
                if hasattr(opt,"__name__"):
                    ll.append((opt.__name__,opt.__class__.__name__))
                else:
                    ll.append((opt.name,opt.__class__.__name__))
            lll=zip(l,ll)
            def cmp(a,b):
                if a[0]==b[0]: return 0
                if a[0]<b[0]: return -1
                return 1
            lll.sort(cmp)
            print lll

    def __eq__(self, other):
        #added to override the list's __eq__ implementation
        return id(self) == id(other)

    def __neq__(self, other):
        #added to override the list's __neq__ implementation
        return id(self) != id(other)


    def __str__(self):
        return "SeqOpt(%s)" % list.__str__(self)

    def __repr__(self):
        return list.__repr__(self)

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        name = getattr(self, 'name', None)
        print >> stream, "%s%s %s id=%i" %(' '*level, self.__class__.__name__, name, id(self))
        # This way, -1 will do all depth
        if depth != 0:
            depth -= 1
            for opt in self:
                opt.print_summary(stream, level=level+2, depth=depth)




class _metadict:
    """WRITEME"""
    # dict that accepts unhashable keys
    # uses an associative list
    # for internal use only
    def __init__(self):
        self.d = {}
        self.l = []
    def __getitem__(self, item):
        return self.get(item, None)
    def __setitem__(self, item, value):
        try:
            self.d[item] = value
        except Exception:
            for i, (key,val) in enumerate(self.l):
                if key == item:
                    self.l[i] = (item, value)
                    return
            self.l.append((item, value))
    def get(self, item, default):
        try:
            return self.d[item]
        except Exception:
            for item2, value in self.l:
                try:
                    if item == item2:
                        return value
                    if item.equals(item2):
                        return value
                except Exception:
                    if item is item2:
                        return value
            else:
                return default
    def clear(self):
        self.d = {}
        self.l = []
    def __str__(self):
        return "(%s, %s)" % (self.d, self.l)


class MergeOptimizer(Optimizer):
    """
    Merges parts of the graph that are identical and redundant.

    The basic principle is that if two Applies have ops that compare equal, and
    identical inputs, then they do not both need to be computed. The clients of
    one are transferred to the other and one of them is removed from the graph.
    This procedure is carried out in input->output order through the graph.

    The first step of merging is constant-merging, so that all clients of an
    int(1) for example, are transferred to a particular instance of int(1).
    """
    def __init__(self, skip_const_merge=False):
        self.skip_const_merge = skip_const_merge

    def add_requirements(self, env):
        # Added by default
        #env.extend(toolbox.ReplaceValidate())
        pass

    def apply_constant_merge(self, env):
        seen_constants = set()
        const_sig = _metadict()     # variable -> variable.signature()  (for constants)
        const_sig_inv = _metadict() # signature -> variable (for constants)
        for node in _list_of_nodes(env):
            for i, c in enumerate([r for r in node.inputs if isinstance(r, graph.Constant)]):
                if id(c) in seen_constants:
                    continue
                else:
                    seen_constants.add(id(c))
                sig = c.signature()
                other_c = const_sig_inv.get(sig, None)
                if other_c is not None:
                    # multiple names will clobber each other..
                    # we adopt convention to keep the last name
                    if c.name:
                        other_c.name = c.name
                    env.replace_validate(c, other_c, reason='Constant Merge')
                else:
                    #this is a new constant
                    const_sig[c] = sig
                    const_sig_inv[sig] = c

    def apply_node_merge(self, env):
        # we clear the dicts because the Constants signatures are not necessarily hashable
        # and it's more efficient to give them an integer like the other Variables

        nodes_seen = {}

        for node_idx, node in enumerate(_list_of_nodes(env)):
            #
            # these asserts ensure that the env has set the clients field properly the clients
            # should at least contain `node` itself!
            #
            if node.inputs:
                assert len(node.inputs[0].clients) > 0
                assert (node,0) in node.inputs[0].clients
                merge_candidates = [(nodes_seen[c],c) for (c,i) in node.inputs[0].clients if c in nodes_seen]
            else:
                merge_candidates = []
            merge_candidates.sort()
            nodes_seen[node] = node_idx
            #print 'NODE', node, merge_candidates, node.inputs[0].clients
            for candidate_idx, candidate in merge_candidates:
                if len(node.inputs) != len(candidate.inputs):
                    continue
                inputs_match = all(node_in is cand_in for node_in, cand_in in zip(node.inputs, candidate.inputs))
                if inputs_match and node.op == candidate.op:
                    assert node is not candidate
                    #
                    #transfer clients from node to candidate
                    #
                    success = True
                    assert len(node.outputs) == len(candidate.outputs)
                    pairs = zip(node.outputs, candidate.outputs)

                    #transfer names
                    for node_output, cand_output in pairs:
                        #clobber old name with new one
                        #it's arbitrary... one of the names has to go
                        if node_output.name:
                            cand_output.name = node_output.name
                    try:
                        env.replace_all_validate(pairs, reason="Merge")
                    except InconsistencyError, e:
                        success = False

                    if success:
                        #break out of the candidate loop
                        break
                    else:
                        #try the next candidate
                        pass


    #TODO: Consider splitting this into a separate optimizer (SeqOptimizer)
    def apply(self, env):
        if not self.skip_const_merge:
            self.apply_constant_merge(env)
        self.apply_node_merge(env)

merge_optimizer = MergeOptimizer()


def is_same_graph_with_merge(var1, var2, givens={}):
    """
    Merge-based implementation of `theano.gof.graph.is_same_graph`.

    See help on `theano.gof.graph.is_same_graph` for additional documentation.
    """
    # Copy variables since the MergeOptimizer will modify them.
    copied = copy.deepcopy([var1, var2, givens])
    vars = copied[0:2]
    givens = copied[2]
    # Create Env.
    inputs = theano.gof.graph.inputs(vars)
    env = theano.gof.env.Env(inputs, vars)
    # Perform Variable substitution.
    for to_replace, replace_by in givens.iteritems():
        env.replace(to_replace, replace_by)
    # Perform merge optimization.
    merge_optimizer.optimize(env)
    # When two variables perform the same computations, they will have the same
    # owner in the optimized graph.
    # We need to be careful with the special case where the owner is None,
    # which happens when the graph is made of a single Variable.
    # We also need to make sure we replace a Variable if it is present in
    # `givens`.
    vars_replaced = [givens.get(v, v) for v in vars]
    o1, o2 = [v.owner for v in vars_replaced]
    if o1 is None and o2 is None:
        # Comparing two single-Variable graphs: they are equal if they are
        # the same Variable.
        return vars_replaced[0] == vars_replaced[1]
    else:
        return o1 is o2


def MergeOptMerge(opt):
    """WRITEME
    Returns an Optimizer that merges the graph then applies the
    optimizer in opt and then merges the graph again in case the
    opt introduced additional similarities.
    """
    merger = merge_optimizer
    return SeqOptimizer([merger, opt, merger])


def pre_constant_merge(vars):
    """
    Merge constants in the subgraph used to compute nodes in `vars`.

    `vars` is a list of nodes, and we want to merge together nodes
    that are constant inputs used to compute nodes in that list.

    :note: This function will ignore nodes that are in an env.
           It is used to pre-merge nodes generated inside an optimization,
           before it is inserted in the env.
           It is useful if there are many such replacements to make,
           so that DebugMode will not check each of them.
    """

    seen_var = set()
    const_sig = {}     # variable -> variable.signature()  (for constants)
    const_sig_inv = {} # signature -> variable (for constants)
    def recursive_merge(var):
        if var in seen_var:
            return var
        if not hasattr(var, 'owner'):
            return var
        if var.owner and hasattr(var.owner, "env"):
            return var
        seen_var.add(var)
        if isinstance(var, graph.Constant):
            sig = var.signature()
            if sig in const_sig_inv:
                return const_sig_inv[sig]
            const_sig_inv[sig] = var
            return var
        if var.owner:
            for idx,inp in enumerate(var.owner.inputs):
                var.owner.inputs[idx] = recursive_merge(inp)
        return var

    return map(recursive_merge, vars)

########################
### Local Optimizers ###
########################

class LocalOptimizer(object):
    """A class for node-based optimizations.

    Instances should implement the transform function,
    and be passed to configure a env-based Optimizer instance.
    """

    def __hash__(self):
        if not hasattr(self, '_optimizer_idx'):
            self._optimizer_idx = _optimizer_idx[0]
            _optimizer_idx[0] += 1
        return self._optimizer_idx

    def transform(self, node):
        """Transform a subgraph whose output is `node`.

        Subclasses should implement this function so that it returns one of two
        kinds of things:

        - False to indicate that no optimization can be applied to this `node`; or

        - <list of variables> to use in place of `node`'s outputs in the greater graph.

        :type node: an Apply instance

        """

        raise utils.MethodNotDefined("transform", type(self), self.__class__.__name__)

    def add_requirements(self, env):
        """If this local optimization wants to add some requirements to the env,
        This is the place to do it."""
        # Added by default
        #env.extend(toolbox.ReplaceValidate())
        pass

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print >> stream, "%s%s id=%i" %(' '*level, self.__class__.__name__, id(self))

class FromFunctionLocalOptimizer(LocalOptimizer):
    """WRITEME"""
    def __init__(self, fn, tracks = []):
        self.transform = fn
        self._tracks = tracks
    def tracks(self):
        return self._tracks
    def __str__(self):
        return getattr(self, '__name__', '<FromFunctionLocalOptimizer instance>')
    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print >> stream, "%s%s id=%i" %(' '*level,
                str(self.transform),
                id(self))

def local_optimizer(*tracks):
    def decorator(f):
        """WRITEME"""
        rval = FromFunctionLocalOptimizer(f, tracks)
        rval.__name__ = f.__name__
        return rval
    return decorator


class LocalOptGroup(LocalOptimizer):
    """WRITEME"""

    def __init__(self, *optimizers):
        self.opts = optimizers
        self.reentrant = any(getattr(opt, 'reentrant', True) for opt in optimizers)
        self.retains_inputs = all(getattr(opt, 'retains_inputs', False) for opt in optimizers)

    def __str__(self):
        return getattr(self, '__name__', '<theano.gof.opt.LocalOptGroup instance>'+str([str(o) for o in self.opts]))

    def transform(self, node):
        for opt in self.opts:
            repl = opt.transform(node)
            if repl:
                return repl

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print >> stream, "%s%s id=%i" %(' '*level, self.__class__.__name__, id(self))
        if depth != 0:
            depth -= 1
            for lopt in self.opts:
                lopt.print_summary(stream, level=level+2, depth=depth)


class _LocalOpKeyOptGroup(LocalOptGroup):
    """WRITEME"""

    def __init__(self, optimizers):
        if any(not hasattr(opt, 'op_key'), optimizers):
            raise TypeError("All LocalOptimizers passed here must have an op_key method.")
        CompositeLocalOptimizer.__init__(self, optimizers)

    def op_key(self):
        return [opt.op_key() for opt in self.opts]


class OpSub(LocalOptimizer):
    """WRITEME
    Replaces the application of a certain op by the application of
    another op that take the same inputs as what they are replacing.

    e.g. OpSub(add, sub) ==> add(div(x, y), add(y, x)) -> sub(div(x, y), sub(y, x))
    """

    reentrant = False      # an OpSub does not apply to the nodes it produces
    retains_inputs = True  # all the inputs of the original node are transferred to the outputs

    def __init__(self, op1, op2, transfer_tags = True):
        """
        op1.make_node and op2.make_node must take the same number of
        inputs and have the same number of outputs.
        """
        self.op1 = op1
        self.op2 = op2
        self.transfer_tags = transfer_tags

    def op_key(self):
        return self.op1

    def tracks(self):
        return [[self.op1]]

    def transform(self, node):
        if node.op != self.op1:
            return False
        repl = self.op2.make_node(*node.inputs)
        if self.transfer_tags:
            repl.tag = copy.copy(node.tag)
            for output, new_output in zip(node.outputs, repl.outputs):
                new_output.tag = copy.copy(output.tag)
        return repl.outputs

    def __str__(self):
        return "%s -> %s" % (self.op1, self.op2)


class OpRemove(LocalOptimizer):
    """WRITEME
    Removes all applications of an op by transferring each of its
    outputs to the corresponding input.
    """

    reentrant = False      # no nodes are added at all

    def __init__(self, op):
        self.op = op

    def op_key(self):
        return self.op

    def tracks(self):
        return [[self.op]]

    def transform(self, node):
        if node.op != self.op:
            return False
        return node.inputs

    def __str__(self):
        return "%s(x) -> x" % (self.op)

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print >> stream, "%s%s(%s) id=%i" %(' '*level,
                self.__class__.__name__,
                str(self.op),
                id(self))


class PatternSub(LocalOptimizer):
    """WRITEME
    @todo update

    Replaces all occurrences of the input pattern by the output pattern:

     input_pattern ::= (op, <sub_pattern1>, <sub_pattern2>, ...)
     input_pattern ::= dict(pattern = <input_pattern>,
                            constraint = <constraint>)
     sub_pattern ::= input_pattern
     sub_pattern ::= string
     sub_pattern ::= a Constant instance
     sub_pattern ::= int
     sub_pattern ::= float
     constraint ::= lambda env, expr: additional matching condition

     output_pattern ::= (op, <output_pattern1>, <output_pattern2>, ...)
     output_pattern ::= string
     output_pattern ::= int
     output_pattern ::= float

    Each string in the input pattern is a variable that will be set to
    whatever expression is found in its place. If the same string is
    used more than once, the same expression must be found in those
    places. If a string used in the input pattern is used in the
    output pattern, the matching expression will be inserted in its
    place. The input pattern cannot just be a string but the output
    pattern can.

    If you put a constant variable in the input pattern, there will be a
    match iff a constant variable with the same value and the same type
    is found in its place.

    You can add a constraint to the match by using the dict(...)  form
    described above with a 'constraint' key. The constraint must be a
    function that takes the env and the current Variable that we are
    trying to match and returns True or False according to an
    arbitrary criterion.

    Examples:
     PatternSub((add, 'x', 'y'), (add, 'y', 'x'))
     PatternSub((multiply, 'x', 'x'), (square, 'x'))
     PatternSub((subtract, (add, 'x', 'y'), 'y'), 'x')
     PatternSub((power, 'x', Constant(double, 2.0)), (square, 'x'))
     PatternSub((boggle, {'pattern': 'x',
                                'constraint': lambda expr: expr.type == scrabble}),
                      (scrabble, 'x'))
    """

    def __init__(self, in_pattern, out_pattern, allow_multiple_clients = False,
            skip_identities_fn = None, name = None, pdb = False):
        """
        Creates a PatternSub that replaces occurrences of
        in_pattern by occurrences of out_pattern.

        :param in_pattern: the input pattern that we want to replace
        :param out_pattern: the replacement pattern
        :param allow_multiple_clients: if False, the pattern matching will fail
                                       if one of the subpatterns has more than
                                       one client.
        :param pdb: if True, we invoke pdb when the first node in the pattern match.
        """
        self.in_pattern = in_pattern
        self.out_pattern = out_pattern
        if isinstance(in_pattern, (list, tuple)):
            self.op = self.in_pattern[0]
        elif isinstance(in_pattern, dict):
            self.op = self.in_pattern['pattern'][0]
        else:
            raise TypeError("The pattern to search for must start with a specific Op instance.")
        self.__doc__ = self.__class__.__doc__ + "\n\nThis instance does: " + str(self) + "\n"
        self.allow_multiple_clients = allow_multiple_clients
        self.skip_identities_fn = skip_identities_fn
        if name:
            self.__name__ = name
        self.pdb = pdb

    def skip_identities(self, expr):
        if self.skip_identities_fn:
            return self.skip_identities_fn(expr)

    def op_key(self):
        return self.op

    def tracks(self):
        def helper(pattern, sofar):
            if isinstance(pattern, (list, tuple)):
                sofar = sofar + (pattern[0],)
                return reduce(tuple.__add__,
                              tuple(helper(p, sofar) for p in pattern[1:]),
                              ())
            elif isinstance(pattern, dict):
                return helper(pattern['pattern'], sofar)
            else:
                return (sofar,)
        return set(helper(self.in_pattern, ()))

    def transform(self, node):
        """
        Checks if the graph from node corresponds to in_pattern. If it does,
        constructs out_pattern and performs the replacement.
        """
        if node.op != self.op:
            return False

        def match(pattern, expr, u, allow_multiple_clients = False, pdb = False):
            def retry_with_equiv():
                expr_equiv = self.skip_identities(expr)
                if expr_equiv is None:
                    return False
                #TODO: Not sure how to handle multiple_clients flag
                ###print 'retrying match', pattern, expr_equiv
                return match(pattern, expr_equiv, u,
                        allow_multiple_clients=allow_multiple_clients)

            if isinstance(pattern, (list, tuple)):
                if expr.owner is None:
                    return False
                if not (expr.owner.op == pattern[0]) or (not allow_multiple_clients and len(expr.clients) > 1):
                    return retry_with_equiv()
                if len(pattern) - 1 != len(expr.owner.inputs):
                    return retry_with_equiv()
                for p, v in zip(pattern[1:], expr.owner.inputs):
                    u = match(p, v, u, self.allow_multiple_clients)
                    if not u:
                        return False
            elif isinstance(pattern, dict):
                try:
                    real_pattern = pattern['pattern']
                except KeyError:
                    raise KeyError("Malformed pattern: %s (expected key 'pattern')" % pattern)
                constraint = pattern.get('constraint', lambda expr: True)
                if constraint(expr):
                    return match(real_pattern, expr, u, pattern.get('allow_multiple_clients', allow_multiple_clients))
                else:
                    return retry_with_equiv()
            elif isinstance(pattern, basestring):
                v = unify.Var(pattern)
                if u[v] is not v and u[v] is not expr:
                    return retry_with_equiv()
                else:
                    u = u.merge(expr, v)
            elif isinstance(pattern, (int, float)) and isinstance(expr, graph.Constant):
                if numpy.all(theano.tensor.constant(pattern).value==expr.value):
                    return u
                else:
                    return retry_with_equiv()
            elif isinstance(pattern, graph.Constant) and isinstance(expr, graph.Constant) and pattern.equals(expr):
                return u
            else:
                return retry_with_equiv()
            if pdb:
                import pdb;pdb.set_trace()
            return u

        def build(pattern, u):
            if isinstance(pattern, (list, tuple)):
                args = [build(p, u) for p in pattern[1:]]
                return pattern[0](*args)
            elif isinstance(pattern, basestring):
                return u[unify.Var(pattern)]
            elif isinstance(pattern, (int,float)):
                return pattern
            else:
                return pattern.clone()
        u = match(self.in_pattern, node.out, unify.Unification(), True, self.pdb)
        if u:
            p = self.out_pattern
            new = build(p, u)
            ####print "PatternSub matched:", new
            return [new]
        else:
            return False

    def __str__(self):
        if getattr(self,'__name__',None):
            return self.__name__
        def pattern_to_str(pattern):
            if isinstance(pattern, (list, tuple)):
                return "%s(%s)" % (str(pattern[0]), ", ".join([pattern_to_str(p) for p in pattern[1:]]))
            elif isinstance(pattern, dict):
                return "%s subject to %s" % (pattern_to_str(pattern['pattern']), str(pattern.get('constraint', 'no conditions')))
            else:
                return str(pattern)
        return "%s -> %s" % (pattern_to_str(self.in_pattern), pattern_to_str(self.out_pattern))

    def __repr__(self):
        return str(self)

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        name = getattr(self, '__name__', getattr(self, 'name', None))
        print >> stream, "%s%s %s(%s, %s) id=%i" %(' '*level,
                self.__class__.__name__,
                name,
                str(self.in_pattern),
                str(self.out_pattern),
                id(self))


##################
### Navigators ###
##################

# Use the following classes to apply LocalOptimizers


class NavigatorOptimizer(Optimizer):
    """Abstract class

    """
    @staticmethod
    def warn(exc, nav, repl_pairs, local_opt):
        """failure_callback for NavigatorOptimizer: print traceback
        """
        _logger.error("Optimization failure due to: %s" % str(local_opt))
        _logger.error("TRACEBACK:")
        _logger.error(traceback.format_exc())
        if isinstance(exc, AssertionError) or config.on_opt_error == 'raise':
            raise exc
    @staticmethod
    def warn_inplace(exc, nav, repl_pairs, local_opt):
        """failure_callback for NavigatorOptimizer: ignore InconsistencyErrors, print traceback
        """
        if isinstance(exc, InconsistencyError):
            return
        return NavigatorOptimizer.warn(exc, nav, repl_pairs, local_opt)
    @staticmethod
    def warn_ignore(exc, nav, repl_pairs, local_opt):
        """failure_callback for NavigatorOptimizer: ignore all errors
        """
        pass

    def __init__(self, local_opt, ignore_newtrees = 'auto', failure_callback = None):
        """
        :param local_opt:  a LocalOptimizer to apply over a Env (or None is Ok too).
        :param ignore_newtrees:
            - True: new subgraphs returned by an optimization is not a candidate for optimization
            - False: new subgraphs returned by an optimization is a candidate for optimization
            - 'auto': let the local_opt set this parameter via its 'reentrant' attribute.
        :param failure_callback:
            a function that takes (exception, navigator, [(old, new),
            (old,new),...]) and we call it if there's an exception.

            If the trouble is from local_opt.transform(), the new variables will be 'None'.

            If the trouble is from validation (the new types don't match for
            example) then the new variables will be the ones created by
            transform().

            If this parameter is None, then exceptions are not caught here (raised normally).
        """
        self.local_opt = local_opt
        if ignore_newtrees == 'auto':
            self.ignore_newtrees = not getattr(local_opt, 'reentrant', True)
        else:
            self.ignore_newtrees = ignore_newtrees
        self.failure_callback = failure_callback

    def attach_updater(self, env, importer, pruner, chin = None):
        """Install some Env listeners to help the navigator deal with the ignore_trees-related functionality.

        :param importer: function that will be called whenever when optimizations add stuff to the graph.
        :param pruner: function to be called when optimizations remove stuff from graph.
        :param chin: "on change input" called whenever an node's inputs change.

        :returns: The Env plugin that handles the three tasks.  Keep this around so that you can detach later!

        """
        if self.ignore_newtrees:
            importer = None

        if importer is None and pruner is None:
            return None

        class Updater:
            if importer is not None:
                def on_import(self, env, node):
                    importer(node)
            if pruner is not None:
                def on_prune(self, env, node):
                    pruner(node)
            if chin is not None:
                def on_change_input(self, env, node, i, r, new_r):
                    chin(node, i, r, new_r)

        u = Updater()
        env.extend(u)
        return u

    def detach_updater(self, env, u):
        """Undo the work of attach_updater.

        :param u: a return-value of attach_updater

        :returns: None.
        """
        if u is not None:
            env.remove_feature(u)

    def process_node(self, env, node, lopt = None):
        """
        This function will use `lopt` to `transform` the `node`.  The `transform` method will
        return either False or a list of Variables that are intended to replace `node.outputs`.

        If the env accepts the replacement, then the optimization is successful, and this
        function returns True.

        If there are no replacement candidates or the env rejects the replacements, this
        function returns False.

        :param env:  an Env
        :param node: an Apply instance in `env`
        :param lopt: a LocalOptimizer instance that may have a better idea for how to compute
        node's outputs.
        :rtype: Bool
        :returns: True iff the `node`'s outputs were replaced in the `env`.

        """
        lopt = lopt or self.local_opt
        try:
            replacements = lopt.transform(node)
        except Exception, e:
            if self.failure_callback is not None:
                self.failure_callback(e, self, [(x, None) for x in node.outputs], lopt)
                return False
            else:
                raise
        if replacements is False or replacements is None:
            return False
        if not isinstance(replacements, (tuple, list)):
            raise TypeError('Optimizer %s gave wrong type of replacement. Expected list or tuple.' % lopt)
        if len(node.outputs) != len(replacements):
            raise ValueError('Optimizer %s gave wrong number of replacements' % lopt)
        # If an output would be replaced by itself, no need to perform
        # the replacement
        repl_pairs = [(r, rnew) for r, rnew in zip(node.outputs, replacements)
                if rnew is not r]
        if len(repl_pairs) == 0:
            return False
        try:
            env.replace_all_validate(repl_pairs, reason=lopt)
            return True
        except Exception, e:
            # This means the replacements were rejected by the env.
            #
            # This is not supposed to happen.  The default failure_callback will print a
            # traceback as a warning.
            if self.failure_callback is not None:
                self.failure_callback(e, self, repl_pairs, lopt)
                return False
            else:
                raise

    def add_requirements(self, env):
        super(NavigatorOptimizer, self).add_requirements(env)
        # Added by default
        #env.extend(toolbox.ReplaceValidate())
        if self.local_opt:
            self.local_opt.add_requirements(env)

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print >> stream, "%s%s (%i)" %(' '*level, self.__class__.__name__, id(self))
        if depth != 0:
            self.local_opt.print_summary(stream, level=level+2, depth=depth-1)


class TopoOptimizer(NavigatorOptimizer):
    """WRITEME"""

    def __init__(self, local_opt, order = 'in_to_out', ignore_newtrees = False, failure_callback = None):
        if order not in ['out_to_in', 'in_to_out']:
            raise ValueError("order must be 'out_to_in' or 'in_to_out'")
        self.order = order
        NavigatorOptimizer.__init__(self, local_opt, ignore_newtrees, failure_callback)

    def apply(self, env, start_from = None):
        if start_from is None: start_from = env.outputs
        q = deque(graph.io_toposort(env.inputs, start_from))
        def importer(node):
            if node is not current_node:
                q.append(node)
        def pruner(node):
            if node is not current_node:
                try:
                    q.remove(node)
                except ValueError:
                    pass

        u = self.attach_updater(env, importer, pruner)
        try:
            while q:
                if self.order == 'out_to_in':
                    node = q.pop()
                else:
                    node = q.popleft()
                current_node = node
                self.process_node(env, node)
        except Exception:
            self.detach_updater(env, u)
            raise
        self.detach_updater(env, u)



class OpKeyOptimizer(NavigatorOptimizer):
    """WRITEME"""

    def __init__(self, local_opt, ignore_newtrees = False, failure_callback = None):
        if not hasattr(local_opt, 'op_key'):
            raise TypeError("LocalOptimizer for OpKeyOptimizer must have an 'op_key' method.")
        NavigatorOptimizer.__init__(self, local_opt, ignore_newtrees, failure_callback)

    def apply(self, env):
        op = self.local_opt.op_key()
        if isinstance(op, (list, tuple)):
            q = reduce(list.__iadd__, map(env.get_nodes, op))
        else:
            q = list(env.get_nodes(op))
        def importer(node):
            if node is not current_node:
                if node.op == op: q.append(node)
        def pruner(node):
            if node is not current_node and node.op == op:
                try:
                    q.remove(node)
                except ValueError:
                    pass
        u = self.attach_updater(env, importer, pruner)
        try:
            while q:
                node = q.pop()
                current_node = node
                self.process_node(env, node)
        except Exception:
            self.detach_updater(env, u)
            raise
        self.detach_updater(env, u)

    def add_requirements(self, env):
        """
        Requires the following features:
          - NodeFinder
          - ReplaceValidate(Added by default)
        """
        super(OpKeyOptimizer, self).add_requirements(env)
        env.extend(toolbox.NodeFinder())



class ChangeTracker:
    def __init__(self):
        self.changed = False

    def on_import(self, env, node):
        self.changed = True

    def on_change_input(self, env, node, i, r, new_r):
        self.changed = True

    def reset(self):
        self.changed = False

    def on_attach(self, env):
        env.change_tracker = self

class EquilibriumOptimizer(NavigatorOptimizer):
    def __init__(self,
                 optimizers,
                 failure_callback = None,
                 max_depth = None,
                 max_use_ratio = None):
        """
        :param optimizers:  list or set of local or global optimizations to apply until
            equilibrium.

        :param max_use_ratio: each optimizer can be applied at most (size of graph * this number)

        :param max_depth: TODO what does this do? (EquilibriumDB sets it to 5)

        """

        super(EquilibriumOptimizer, self).__init__(
            None,
            ignore_newtrees = True,
            failure_callback = failure_callback)
        self.local_optimizers = []
        self.global_optimizers = []

        for opt in optimizers:
            if isinstance(opt, LocalOptimizer):
                self.local_optimizers.append(opt)
            else:
                self.global_optimizers.append(opt)
        self.max_depth = max_depth
        self.max_use_ratio = max_use_ratio
        assert self.max_use_ratio is not None, 'max_use_ratio has to be a number'

    def add_requirements(self, env):
        super(EquilibriumOptimizer, self).add_requirements(env)
        env.extend(ChangeTracker())

    def apply(self, env, start_from = None):
        if start_from is None:
            start_from = env.outputs
        changed = True
        max_use_abort = False
        opt_name = None
        process_count = {}

        while changed and not max_use_abort:
            changed = False

            #apply global optimizer
            env.change_tracker.reset()
            for gopt in self.global_optimizers:
                gopt.apply(env)
            if env.change_tracker.changed:
                changed = True

            #apply local optimizer
            for node in start_from:
                assert node in env.outputs

            q = deque(graph.io_toposort(env.inputs, start_from))

            max_use = len(q) * self.max_use_ratio
            def importer(node):
                if node is not current_node:
                    q.append(node)
            def pruner(node):
                if node is not current_node:
                    try:
                        q.remove(node)
                    except ValueError:
                        pass

            u = self.attach_updater(env, importer, pruner)
            try:
                while q:
                    node = q.pop()
                    current_node = node
                    for lopt in self.local_optimizers:
                        process_count.setdefault(lopt, 0)
                        if process_count[lopt] > max_use:
                            max_use_abort = True
                            opt_name = (getattr(lopt, "name", None)
                                        or getattr(lopt, "__name__", None) or "")
                        else:
                            lopt_change = self.process_node(env, node, lopt)
                            if lopt_change:
                                process_count[lopt] += 1
                                changed = True
                                if node not in env.nodes:
                                    break # go to next node
            finally:
                self.detach_updater(env, u)
            self.detach_updater(env, u) #TODO: erase this line, it's redundant at best
        if max_use_abort:
            _logger.error("EquilibriumOptimizer max'ed out by '%s'" % opt_name
                          + ". You can safely raise the current threshold of "
                          + "%f with the theano flag 'optdb.max_use_ratio'." %
                          config.optdb.max_use_ratio)

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        name = getattr(self, 'name', None)
        print >> stream, "%s%s %s id=%i" %(' '*level, self.__class__.__name__, name, id(self))
        if depth != 0:
            for lopt in self.local_optimizers:
                lopt.print_summary(stream, level=level+2, depth=depth-1)


#################
### Utilities ###
#################

def _check_chain(r, chain):
    """WRITEME"""

    chain = list(reversed(chain))
    while chain:
        elem = chain.pop()
        if elem is None:
            if not r.owner is None:
                return False
        elif r.owner is None:
            return False
        elif isinstance(elem, op.Op):
            if not r.owner.op == elem:
                return False
        else:
            try:
                if issubclass(elem, op.Op) and not isinstance(r.owner.op, elem):
                    return False
            except TypeError:
                return False
        if chain:
            r = r.owner.inputs[chain.pop()]
    #print 'check_chain', _check_chain.n_calls
    #_check_chain.n_calls += 1

    # The return value will be used as a Boolean, but some Variables cannot
    # be used as Booleans (the results of comparisons, for instance)
    return (r is not None)
#_check_chain.n_calls = 0

def check_chain(r, *chain):
    """WRITEME"""
    if isinstance(r, graph.Apply):
        r = r.outputs[0]
    return _check_chain(r, reduce(list.__iadd__, ([x, 0] for x in chain)))


def pre_greedy_local_optimizer(list_optimizations, out):
    '''
    This function traverses the computation graph described by all
    ``node`` in the graph before the variable out but that are not in the env.
    it applies each of the local_optimizations on the traversed graph.

    Its main use is to apply locally constant folding when generating
    the graph of the indices of a subtensor.

    We should not apply optimizations on node that are in env.
    So we don't optimize node that have an attribute env.

    :note: This don't do an equilibrium... So if there is optimization
           like local_upcast_elemwise_constant_inputs in the list, that
           add additional node to the inputs of the node, it can
           be needed to call this function multiple time.
    '''
    def local_recursive_function( list_opt, out, optimized_vars, depth):
        if not getattr(out, 'owner', None):
            return [out], optimized_vars
        node = out.owner
        if hasattr(node, 'env'):
            return node.outputs, optimized_vars
        for idx, inp in enumerate(node.inputs):
            if inp in optimized_vars:
                nw_in = optimized_vars[inp]
            else:
                if inp.owner:
                    outs, optimized_vars = local_recursive_function(
                        list_opt
                        , inp
                        , optimized_vars
                        , depth+1)
                    for k,v in zip(inp.owner.outputs, outs):
                        optimized_vars[k] = v
                    nw_in = outs[inp.owner.outputs.index(inp)]

                else:
                    nw_in = inp
                    optimized_vars[inp] = inp
            node.inputs[idx] = nw_in

        results = node.outputs
        for opt in list_opt:
            ret = opt.transform(node)
            if ret is not False and ret is not None:
                assert len(ret) == len(node.outputs)
                for k,v in zip(node.outputs, ret):
                    optimized_vars[k] = v
                results = ret
                if ret[0].owner :
                    node = out.owner
                else:
                    break
        return results, optimized_vars

    final_outs, optimized_nodes = local_recursive_function(
        list_optimizations, out, {}, 0)
    return final_outs[0]




############
### Misc ###
############

class InplaceOptimizer(Optimizer):

    def __init__(self, inplace):
        self.inplace = inplace

    def apply(self, env):
        self.inplace(env)

    def add_requirements(self, env):
        env.extend(dh.DestroyHandler())


class PureThenInplaceOptimizer(Optimizer):

    def __init__(self, pure, inplace):
        self.pure = pure
        self.inplace = inplace

    def apply(self, env):
        self.pure(env)
        env.extend(dh.DestroyHandler())
        self.inplace(env)

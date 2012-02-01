"""Provides `DebugMode`, an evaluation mode for debugging theano internals.

:TODO: add support for IfElse Op, LazyLinker, PureOp, etc.

"""
__docformat__ = "restructuredtext en"

import time, copy, sys, copy_reg, gc, os
from StringIO import StringIO

import numpy

import theano
from theano import gof
from theano.gof import Env, graph, utils, link
from theano.gof.link import raise_with_op
from theano.gof.cc import CLinker
from theano.configparser import (config, AddConfigVar, BoolParam, IntParam,
        StrParam)
from theano.compile.function_module import (FunctionMaker,
        Function,
        infer_reuse_pattern,
        SymbolicInputKit,
        SymbolicOutput,
        Supervisor)
from theano.compile.mode import Mode, register_mode

AddConfigVar('DebugMode.patience',
        "Optimize graph this many times to detect inconsistency",
        IntParam(10, lambda i: i > 0),
        in_c_key=False)

AddConfigVar('DebugMode.check_c',
        "Run C implementations where possible",
        BoolParam(True),
        in_c_key=False)

AddConfigVar('DebugMode.check_py',
        "Run Python implementations where possible",
        BoolParam(True),
        in_c_key=False)

AddConfigVar('DebugMode.check_finite',
        "True -> complain about NaN/Inf results",
        BoolParam(True),
        in_c_key=False)

AddConfigVar('DebugMode.check_strides',
        ("Check that Python- and C-produced ndarrays have same strides.  "
            "On difference: (0) - ignore, (1) warn, or (2) raise error"),
        IntParam(1, lambda i: i in (0,1,2)),
        in_c_key=False)

AddConfigVar('DebugMode.warn_input_not_reused',
        ("Generate a warning when the destroy_map or view_map tell that an op work inplace, but the op did not reuse the input for its output."
         ),
        BoolParam(True),
        in_c_key=False)

def is_valid_check_preallocated_output_param(param):
    if not isinstance(param, basestring):
        return False
    valid = ["previous", "c_contiguous", "f_contiguous", "neg_strides", "ALL", ""]
    for p in param.split(":"):
        if p not in valid:
            return False
    return True

AddConfigVar('DebugMode.check_preallocated_output',
        ('Test thunks with pre-allocated memory as output storage. '
         'This is a list of strings separated by ":". Valid values are: '
         '"previous" (previously-returned memory), '
         '"c_contiguous", "f_contiguous", '
         '"neg_strides" (negative strides), and '
         '"ALL" (all of the above).'),
        StrParam('ALL', is_valid=is_valid_check_preallocated_output_param),
        in_c_key=False)

import logging
_logger=logging.getLogger("theano.compile.debugmode")
_logger.setLevel(logging.WARNING)

# Filter to avoid duplicating optimization warnings
class NoDuplicateOptWarningFilter(logging.Filter):
    prev_msgs = set([])
    def filter(self, record):
        msg = record.getMessage()
        if msg.startswith('Optimization Warning: '):
            if msg in self.prev_msgs:
                return False
            else:
                self.prev_msgs.add(msg)
                return True
        return True

_logger.addFilter(NoDuplicateOptWarningFilter())

########################
#
# Exceptions
#
########################

class DebugModeError(Exception):
    """Generic Exception raised to indicate an internal theano problem"""
    pass

class BadCLinkerOutput(DebugModeError):
    """Exception: an Op's c_code and perform implementations don't agree."""

    r = None
    """The `Variable` instance for which conflicting values were computed"""

    val_py = None
    """The value computed by `r.owner.op.perform`"""

    val_c = None
    """The value computed by `r.owner.op.c_code`"""

    def __init__(self, r, val_py, val_c):
        """Initialize members"""
        DebugModeError.__init__(self)#to be compatible with python2.4
        self.r = r
        self.val_py = val_py
        self.val_c = val_c

    def offending_op(self):
        """Return the Op class whose c_code and perform implementations didn't match"""
        return type(self.r.owner.op)

    def __str__(self):
        return self.str_diagnostic()

    def str_diagnostic(self):
        """Return a pretty multiline string representating the cause of the exception"""
        sio = StringIO()
        print >> sio, "BadCLinkerOutput"
        print >> sio, "  variable:", self.r
        print >> sio, "  Outputs Type    :", self.r.type
        print >> sio, "  Inputs Type:", [i.type for i in self.r.owner.inputs]
        print >> sio, "  Apply   :", self.r.owner
        print >> sio, "  val_py  :", self.val_py
        print >> sio, "  val_c   :", self.val_c
        print >> sio, "  op      :", self.offending_op()
        try:
            ssio = StringIO()
            print >> ssio, "  PyValue shape, dtype, strides, min, max, n_inf, n_nan:",
            print >> ssio, self.val_py.shape,
            print >> ssio, self.val_py.dtype,
            print >> ssio, self.val_py.strides,
            print >> ssio, self.val_py.min(),
            print >> ssio, self.val_py.max(),
            print >> ssio, numpy.isinf(self.val_py).sum(),
            print >> ssio, numpy.isnan(self.val_py).sum(),
            # only if all succeeds to we add anything to sio
            print >> sio, ssio.getvalue()
        except Exception:
            pass
        try:
            ssio = StringIO()
            print >> ssio, "  CValue shape, dtype, strides, min, max, n_inf, n_nan:",
            print >> ssio, self.val_c.shape,
            print >> ssio, self.val_c.dtype,
            print >> ssio, self.val_c.strides,
            print >> ssio, self.val_c.min(),
            print >> ssio, self.val_c.max(),
            print >> ssio, numpy.isinf(self.val_c).sum(),
            print >> ssio, numpy.isnan(self.val_c).sum(),
            # only if all succeeds to we add anything to sio
            print >> sio, ssio.getvalue()
        except Exception:
            pass
        try:
            ov=numpy.asarray(self.val_c)
            nv=numpy.asarray(self.val_py)
            ssio = StringIO()
            absdiff = numpy.absolute(nv-ov)
            print >> ssio, "  Max Abs Diff: ", numpy.max(absdiff)
            print >> ssio, "  Mean Abs Diff: ", numpy.mean(absdiff)
            print >> ssio, "  Median Abs Diff: ", numpy.median(absdiff)
            print >> ssio, "  Std Abs Diff: ", numpy.std(absdiff)
            reldiff = numpy.absolute(nv-ov) / (numpy.absolute(nv)+numpy.absolute(ov))
            print >> ssio, "  Max Rel Diff: ", numpy.max(reldiff)
            print >> ssio, "  Mean Rel Diff: ", numpy.mean(reldiff)
            print >> ssio, "  Median Rel Diff: ", numpy.median(reldiff)
            print >> ssio, "  Std Rel Diff: ", numpy.std(reldiff)
            # only if all succeeds to we add anything to sio
            print >> sio, ssio.getvalue()
        except Exception:
            pass
        return sio.getvalue()

class BadOptimization(DebugModeError):
    """Exception: some variable and its substitute take different runtime values.
    """

    new_r = None
    """A `Variable` instance that took a different value from `old_r`, but which replaced `old_r`."""

    old_r = None
    """A `Variable` instance that was replaced by `new_r`."""

    old_r_val = None
    """The value computed for `old_r`."""

    new_r_val = None
    """The value computed for `new_r`."""

    reason = None
    """An object that indicates why old_r was turned into new_r.

    Convention is that this is the name of the optimization that requested the replacement.
    """

    old_graph = ""
    """A multiline string representation of the graph leading to old_r, at the time of the replacement."""

    new_graph = ""
    """A multiline string representation of the graph leading to new_r, at the time of the replacement."""

    def __init__(self, old_r, new_r, old_r_val, new_r_val, reason, old_graph, new_graph):
        """Initialize members"""
        DebugModeError.__init__(self)#to be compatible with python2.4
        self.old_r = old_r
        self.new_r = new_r
        self.old_r_val = old_r_val
        self.new_r_val = new_r_val
        self.reason = reason
        self.old_graph = old_graph
        self.new_graph = new_graph

    def __str__(self):
        return self.str_diagnostic()

    def str_diagnostic(self):
        """Return a pretty multiline string representating the cause of the exception"""
        sio = StringIO()
        val_str_len_limit = 800
        print >> sio, "BadOptimization Error", super(BadOptimization, self).__str__()
        print >> sio, "  Variable: id", id(self.new_r), self.new_r
        print >> sio, "  Op", self.new_r.owner
        print >> sio, "  Value Type:", type(self.new_r_val)
        try:
            ssio = StringIO()
            print >> ssio, "  Old Value shape, dtype, strides:",
            print >> ssio, self.old_r_val.shape,
            print >> ssio, self.old_r_val.dtype,
            print >> ssio, self.old_r_val.strides
            # only if all succeeds to we add anything to sio
            print >> sio, ssio.getvalue()
        except Exception:
            pass

        str_old_r_val = str(self.old_r_val)
        if len(str_old_r_val) > val_str_len_limit:
            print >> sio, "  Old Value: ", str(self.old_r_val)[:val_str_len_limit], '...'
        else:
            print >> sio, "  Old Value: ", str(self.old_r_val)

        try:
            ssio = StringIO()
            print >> ssio, "  New Value shape, dtype, strides:",
            print >> ssio, self.new_r_val.shape,
            print >> ssio, self.new_r_val.dtype,
            print >> ssio, self.new_r_val.strides
            # only if all succeeds to we add anything to sio
            print >> sio, ssio.getvalue()
        except Exception:
            pass
        str_new_r_val = str(self.new_r_val)
        if len(str_new_r_val) > val_str_len_limit:
            print >> sio, "  New Value: ", str(self.new_r_val)[:val_str_len_limit], '...'
        else:
            print >> sio, "  New Value: ", str(self.new_r_val)

        try:
            ov=numpy.asarray(self.old_r_val)
            nv=numpy.asarray(self.new_r_val)
            ssio = StringIO()
            print >> ssio, "  Max Abs Diff: ", numpy.max(numpy.absolute(nv-ov))
            print >> ssio, "  Mean Abs Diff: ", numpy.mean(numpy.absolute(nv-ov))
            print >> ssio, "  Median Abs Diff: ", numpy.median(numpy.absolute(nv-ov))
            print >> ssio, "  Std Abs Diff: ", numpy.std(numpy.absolute(nv-ov))

            # N.B. the maximum(..., 1e-8) protects against div by 0 when
            #      nv == ov == 0
            reldiff = (numpy.absolute(nv - ov)
                    / numpy.maximum(
                        numpy.absolute(nv) + numpy.absolute(ov),
                        1e-8))
            print >> ssio, "  Max Rel Diff: ", numpy.max(reldiff)
            print >> ssio, "  Mean Rel Diff: ", numpy.mean(reldiff)
            print >> ssio, "  Median Rel Diff: ", numpy.median(reldiff)
            print >> ssio, "  Std Rel Diff: ", numpy.std(reldiff)
            # only if all succeeds to we add anything to sio
            print >> sio, ssio.getvalue()
        except Exception:
            pass

        print >> sio, "  Reason: ", str(self.reason)
        print >> sio, "  Old Graph:"
        print >> sio, self.old_graph
        print >> sio, "  New Graph:"
        print >> sio, self.new_graph
        print >> sio, ""
        print >> sio, "Hint: relax the tolerance by setting tensor.cmp_sloppy=1"
        print >> sio, "  or even tensor.cmp_sloppy=2 for less-strict comparison"
        return sio.getvalue()

class BadDestroyMap(DebugModeError):
    """Exception: Some perform() or c_code() modified an input that wasn't in the destroy_map"""
    def __init__(self, node, idx, old_val, new_val, perform):
        #super(BadDestroyMap, self).__init__()
        DebugModeError.__init__(self)#to be compatible with python2.4
        self.node = node
        self.idx = idx
        self.old_val = old_val
        self.new_val = new_val
        self.perform = perform

    def __str__(self):
        sio = StringIO()
        print >> sio, "  node:", self.node
        print >> sio, "  perform:", self.perform
        print >> sio, "  node.inputs:", [(str(i), id(i)) for i in self.node.inputs]
        print >> sio, "  destroy_map:", getattr(self.node.op, 'destroy_map', {})
        print >> sio, "  changed input idx:", self.idx
        print >> sio, "  changed input type:", self.node.inputs[self.idx].type
        print >> sio, "  repr (old val):", repr(self.old_val)
        print >> sio, "  repr (new val):", repr(self.new_val)
        try:
            npy_old_val = numpy.asarray(self.old_val)
            npy_new_val = numpy.asarray(self.new_val)
            print >> sio, "  value dtype (new <space> old):", npy_new_val.dtype, npy_old_val.dtype
            print >> sio, "  value shape (new <space> old):", npy_new_val.shape, npy_old_val.shape
            print >> sio, "  value min (new <space> old):", npy_new_val.min(), npy_old_val.min()
            print >> sio, "  value max (new <space> old):", npy_new_val.max(), npy_old_val.max()
            delta = npy_new_val - npy_old_val
            print >> sio, "  value min (new-old):", delta.min()
            print >> sio, "  value max (new-old):", delta.max()
            print >> sio, "  value argmin (new-old):", numpy.unravel_index(delta.argmin(), npy_new_val.shape)
            print >> sio, "  value argmax (new-old):", numpy.unravel_index(delta.argmax(), npy_new_val.shape)
            print >> sio, "  location of first 10 mismatches:", numpy.transpose(numpy.nonzero(delta))[:10]
            print >> sio, ""
        except Exception, e:
            print >> sio, "(Numpy-hints failed with: %s)" %str(e)
        print >> sio, "  Hint: this can also be caused by a deficient values_eq_approx() or __eq__() implementation [which compared input values]"
        return sio.getvalue()

class BadViewMap(DebugModeError):
    """Exception: Some perform() or c_code() created a memory alias that wasn't in the view_map"""
    def __init__(self, node, output_idx, out_storage, in_alias_idx=None, out_alias_idx=None):
        #super(BadViewMap, self).__init__()
        DebugModeError.__init__(self)#to be compatible with python2.4
        self.node = node
        self.output_idx = output_idx
        self.out_storage = out_storage
        self.in_alias_idx = in_alias_idx
        self.out_alias_idx = out_alias_idx

    def __str__(self):
        sio = StringIO()
        print >> sio, "  node:", self.node
        print >> sio, "  node.inputs:", [(str(i), id(i)) for i in self.node.inputs]
        print >> sio, "  node.outputs:", [(str(i), id(i)) for i in self.node.outputs]
        print >> sio, "  view_map:", getattr(self.node.op, 'view_map', {})
        print >> sio, "  destroy_map:", getattr(self.node.op, 'destroy_map', {})
        print >> sio, "  aliased output:", self.output_idx
        print >> sio, "  aliased output storage:", self.out_storage
        if self.in_alias_idx:
            print >> sio, "  aliased to inputs:", self.in_alias_idx
        if self.out_alias_idx:
            print >> sio, "  aliased to outputs:", self.out_alias_idx
        return sio.getvalue()

class StochasticOrder(DebugModeError):
    """Exception: Repeated Optimizations of the same graph do not give identical results.

    The most common cause is that an Optimization iterates over some objects in a
    memory-address-dependent order (such as id() or object.hash()).  If you see this error and
    you think it is related to optimizations within Theano, email theano-dev with the message
    attached to this exception.

    """
    pass

class InvalidValueError(DebugModeError):
    """Exception: some Op an output value that is inconsistent with the Type of that output"""
    def __init__(self, r, v, client_node=None, hint='none', specific_hint='none'):
        #super(InvalidValueError, self).__init__()
        DebugModeError.__init__(self)#to be compatible with python2.4
        self.r = r
        self.v = v
        self.client_node = client_node
        self.hint=hint
        self.specific_hint=specific_hint

    def __str__(self):
        r, v = self.r, self.v
        type_r = r.type
        type_v = type(v)
        v_val = str(v)[0:100]
        v_dtype = 'N/A'
        v_shape = 'N/A'
        v_min = 'N/A'
        v_max = 'N/A'
        v_isfinite = 'N/A'
        try:
            v_shape = v.shape
            v_dtype = v.dtype
            v_min = v.min()
            v_max = v.max()
            v_isfinite = numpy.all(numpy.isfinite(v))
        except Exception:
            pass
        client_node = self.client_node
        hint = self.hint
        specific_hint = self.specific_hint
        context = debugprint(r, prefix='  ', depth=12, file=StringIO()).getvalue()
        return """InvalidValueError
        type(variable) = %(type_r)s
        variable       = %(r)s
        type(value)    = %(type_v)s
        dtype(value)   = %(v_dtype)s
        shape(value)   = %(v_shape)s
        value          = %(v_val)s
        min(value)     = %(v_min)s
        max(value)     = %(v_max)s
        isfinite       = %(v_isfinite)s
        client_node    = %(client_node)s
        hint           = %(hint)s
        specific_hint  = %(specific_hint)s
        context        = ...\n%(context)s
        """ % locals()

########################
#
# Private Functions
#
########################


def debugprint(r, prefix='', depth=-1, done=None, print_type=False,
               file=sys.stdout, print_destroy_map=False, print_view_map=False,
               order=[]):
    """Print the graph leading to `r` to given depth.

    :param r: Variable instance
    :param prefix: prefix to each line (typically some number of spaces)
    :param depth: maximum recursion depth (Default -1 for unlimited).
    :param done: set of Apply instances that have already been printed
    :param print_type: wether to print the Variable type after the other infos
    :param file: file-like object to which to print
    :param print_destroy_map: wether to print the op destroy_map after ofther info
    :param print_view_map: wether to print the op view_map after ofther info
    :param order: If not empty will print the index in the toposort.
    """
    if depth==0:
        return

    if done is None:
        done = set()

    if print_type:
        type_str = ' <%s>' % r.type
    else:
        type_str = ''

    if hasattr(r.owner, 'op'):
        # this variable is the output of computation,
        # so just print out the apply
        a = r.owner

        r_name = getattr(r, 'name', '')
        # normally if the name isn't set, it'll be None, so
        # r_name == None here
        if r_name is None:
            r_name = ''

        if print_destroy_map:
            destroy_map_str=str(getattr(r.owner.op,'destroy_map',''))
        else:
            destroy_map_str=''

        if print_view_map:
            view_map_str=str(getattr(r.owner.op,'view_map',''))
        else:
            view_map_str=''
        if destroy_map_str and destroy_map_str!='{}':
            destroy_map_str='d='+destroy_map_str
        if view_map_str and view_map_str!='{}':
            view_map_str='v='+view_map_str

        o=''
        if order:
            o = str(order.index(r.owner))
        if len(a.outputs) == 1:
            print >> file, '%s%s [@%i]%s \'%s\' %s %s %s' % (prefix, a.op, id(r),
                                                          type_str, r_name,
                                                          destroy_map_str,
                                                          view_map_str,
                                                          o)
        else:
            print >> file, '%s%s.%i [@%i]%s \'%s\' %s %s %s' % (prefix, a.op,
                                                             a.outputs.index(r),
                                                             id(r), type_str,
                                                             r_name,
                                                             destroy_map_str,
                                                             view_map_str,
                                                             o)
        if id(a) not in done:
            done.add(id(a))
            for i in a.inputs:
                debugprint(i, prefix+' |', depth=depth-1, done=done,
                           print_type=print_type, file=file, order=order)
    else:
        #this is a variable
        print >> file, '%s%s [@%i]%s' % (prefix, r, id(r), type_str)

    return file

def _optcheck_env(input_specs, output_specs, accept_inplace = False):
    """Create an Env for debugging.

    :param input_specs: env inputs
    :type input_specs: WRITEME
    :param output_specs: env outputs
    :type output_specs: WRITEME
    :param accept_inplace: are inplace ops permitted in the original graph?
    :type accept_inplace: Bool
    :rtype: `Env`
    :returns: a new Env with a cloned graph, with debugging `Feature` instances already installed.

    """
    orig_inputs = [spec.variable for spec in input_specs]
    updates = [spec.update for spec in input_specs if spec.update]
    orig_outputs = [spec.variable for spec in output_specs] + updates

    inputs, outputs = gof.graph.clone(orig_inputs, orig_outputs)
    equivalence_tracker = _VariableEquivalenceTracker()
    env = gof.env.Env(inputs, outputs,
            #DestroyHandler is not needed because it is actually installed by an optimization
            # after canonicalization.  This variables in a big speed gain.
            #features=[equivalence_tracker, gof.DestroyHandler(do_imports_on_attach=False)])
            features=[equivalence_tracker])

    if not accept_inplace:
        for node in env.nodes:
            if getattr(node.op, 'destroy_map', None):
                raise TypeError("Graph must not contain inplace operations", node)

    # We need to protect all immutable inputs from inplace operations.
    env.extend(Supervisor(input for spec, input in zip(input_specs, inputs) if not (spec.mutable or (hasattr(env, 'destroyers') and env.destroyers(input)))))

    # If named nodes are replaced, keep the name
    env.extend(gof.toolbox.PreserveNames())

    return env, map(SymbolicOutput, updates), equivalence_tracker

def _check_inputs(node, storage_map, r_vals, dr_vals, active_nodes, clobber_dr_vals=True,
        perform=None, warn_input_not_reused=True):
    """Raise BadDestroyMap if necessary, update dr_vals"""
    destroyed_idx_list = []
    destroy_map = getattr(node.op, 'destroy_map', {})
    for o_pos, i_pos_list in destroy_map.iteritems():
        destroyed_idx_list.extend(i_pos_list)
    destroyed_res_list = [node.inputs[i] for i in destroyed_idx_list]

    if warn_input_not_reused and destroyed_res_list:
        dmap=getattr(node.op,'destroy_map',{})
        for oo,ii in dmap.iteritems():
            out_var = storage_map[node.outputs[oo]][0]
            in_var = storage_map[node.inputs[ii[0]]][0]
            if isinstance (node.op, theano.compile.mode.OutputGuard):
                # The point of OutputGuard is to be declared as destructive
                # while not destroying anything
                continue
            if out_var is not in_var:
                _logger.warning("Optimization Warning: input idx %d marked "
                        "as destroyed was not changed for node '%s'",
                        ii[0], str(node))

    if warn_input_not_reused:
        vmap=getattr(node.op,'view_map',{})
        for oo,ii in vmap.iteritems():
            out_var = storage_map[node.outputs[oo]][0]
            in_var = storage_map[node.inputs[ii[0]]][0]
            # We don't try to optimize simple scalar and empty ndarray,
            # as this is not worth our time. This happen at least in
            # Subtensor when the output is a scalar But this depend on
            # the version of numpy!
            if getattr(out_var,'size',2)<=1:
                continue
            if isinstance(node.op, theano.compile.mode.OutputGuard):
                # This class is not in the final graph.
                continue
            if not _may_share_memory(out_var, in_var):
                _logger.warning("Optimization Warning: input idx %d marked "
                        "as viewed but new memory allocated by node '%s'",
                        ii[0], str(node))

    for r_idx, r in enumerate(node.inputs):
        if not r.type.values_eq(r_vals[r], storage_map[r][0]):
            # some input node 'r' got changed by running the node
            # this may or may not be ok...
            if r in destroyed_res_list:
                # ok, we expected r to be destroyed
                if node in active_nodes:
                    if dr_vals.get(r, (0, node))[1] is not node:
                        # bad: there should only be one active node that destroys any variable
                        raise Exception('failure in topological ordering')
                    if clobber_dr_vals:
                        dr_vals[r] = (storage_map[r][0], node) #no copy, this is the last use of this variable
                    storage_map[r][0] = None #make sure that dr_vals[r] doens't get used again
            else:
                raise BadDestroyMap(node, r_idx, r_vals[r], storage_map[r][0], perform)


def _check_viewmap(node, storage_map):
    """
    This functions raises a BadViewMap exception when it detects the following:
    - output node storages aliased to input storage, with no declaration in view_map
    - if not aliased to an input, check if two outputs are aliased together
      and used subsequently in the graph
    """

    for oi, onode in enumerate(node.outputs):

        good_alias, bad_alias = {}, {}
        outstorage = storage_map[onode][0]
        instorage_id = [id(storage_map[i][0]) for i in node.inputs]

        # TODO: investigate ways in which other Types may be aliased
        # TODO: consider adding a function to Type to detect aliasing
        danger_flag = id(outstorage) in instorage_id or\
                      (type(outstorage)==numpy.ndarray and
                       outstorage.flags['OWNDATA']==False)

        if danger_flag:
            # first find out which input it aliases
            view_map = getattr(node.op, 'view_map', {})
            destroy_map = getattr(node.op, 'destroy_map', {})

            # In theory, theano's view_map only allows for 1 output to alias 1 input
            # Checking for multiple aliases just in case...

            for ii, inode in enumerate(node.inputs):

                if _may_share_memory(outstorage, storage_map[inode][0]):

                    nodeid = id(inode)
                    bad_alias[nodeid] = ii

                    # check that the aliasing was declared in [view|destroy]_map
                    if ([ii]==view_map.get(oi,None) or\
                        [ii]==destroy_map.get(oi,None)):

                        good_alias[nodeid] = bad_alias.pop(nodeid)

            #TODO: make sure this is correct
            # According to OB, duplicate inputs are rejected on build graph time
            # if they cause problems. So if they are here it should be ok.
            for key,val in good_alias.iteritems():
                bad_alias.pop(key, None)
            if bad_alias:
                raise BadViewMap(node, oi, outstorage, bad_alias.values())

        #if its not aliased to input, check output->output aliasing
        if not good_alias and _is_used_in_graph(onode):
            for other_oi, other_onode in enumerate(node.outputs):
                if other_oi==oi: continue

                other_storage = storage_map[other_onode][0]
                # check to see if we share memory with this other output
                # this is not a problem if the node is not actually used
                if _is_used_in_graph(other_onode) and \
                        _may_share_memory(outstorage, other_storage):
                    raise BadViewMap(node, oi, outstorage, out_alias_idx=other_oi)

def _may_share_memory(a, b):
    from theano.misc.may_share_memory import may_share_memory
    return may_share_memory(a,b,False)

def _is_function_output(node):
    """
    Returns True if the node in question is the a final output of the graph
    """
    return node.clients==[('output', 1)]

def _is_used_in_graph(node):
    return not(_is_function_output(node) or node.clients==[])

def _check_strides_match(a, b, warn_err, op):
    """
    param: warn_err: if 0, no warning, if 1 warning, if 2 error
    """
    if warn_err==0: return

    try:
        strides_eq = a.strides == b.strides
    except Exception:
        return # no strides

    if not strides_eq:
        e = TypeError('Stride mismatch', (a.shape, b.shape, a.strides, b.strides, str(op)))
        if warn_err==2:
            raise e
        else:
            print >> sys.stderr, 'WARNING:', e

def _lessbroken_deepcopy(a):
    """
    :param a: any object

    Returns a copy of `a` that shares no internal storage with the original.  A deep copy.
    This function handles numpy arrays specially to avoid some bug I had one time... (possibly
    about copying 0-d arrays?)
    """
    # this exists because numpy copies are broken
    if type(a) is numpy.ndarray:
        rval = numpy.array(a, copy=True, dtype=a.dtype)
    else:
        rval = copy.deepcopy(a)

    assert type(rval) == type(a)
    if isinstance(rval, numpy.ndarray):
        assert rval.dtype == a.dtype
    return rval

def _find_bad_optimizations0(order, reasons, r_vals):
    """Use a simple algorithm to find broken optimizations.

    This algorithm is simple to understand, but sometimes when there's a problem it identifies
    the wrong optimization as the culprit.  The problem stems from the fact that results are
    not evaluated in chronological order (looking at when they were introduced to the graph).
    """
    # iterate over variables looking for values that don't match the values of the
    # variables they replaced.  This is the sign of a broken optimization.
    for i, node in enumerate(order):
        for new_r in node.outputs:
            for reason, r, old_graph_str, new_graph_str in reasons[new_r]:
                problem = False

                #check if the value for new_r doesn't match the value for r
                new_r_val = r_vals[new_r]
                r_val = r_vals[r]
                assert r.type == new_r.type

                if hasattr(new_r,'values_eq_approx'):
                    check = new_r.values_eq_approx(r_val, new_r_val)
                else:
                    check = r.type.values_eq_approx(r_val, new_r_val)
                if not check:
                    raise BadOptimization(old_r=r,
                            new_r=new_r,
                            old_r_val=r_val,
                            new_r_val=new_r_val,
                            reason=reason,
                            old_graph=old_graph_str,
                            new_graph=new_graph_str)

def _find_bad_optimizations1(order, reasons, r_vals):
    # iterate over variables looking for values that don't match the values of the
    # variables they replaced.  This is the sign of a broken optimization.

    #identify sets of variables that are supposed to be equivalent
    equivalence_sets = {}
    program_position = {} #node -> order idx

    for i, node in enumerate(order):
        program_position[node] = i
        for new_r in node.outputs:
            equivalence_sets.setdefault(new_r, set([new_r]))
            for reason, r, old_graph_str, new_graph_str in reasons[new_r]:
                equivalence_sets[new_r].update(equivalence_sets.setdefault(r, set([r])))
                for er in equivalence_sets[r]:
                    equivalence_sets[er] = equivalence_sets[new_r]

    #identify equivalence sets that are broken
    equivalence_sets_broken = {} #id(set) -> Bool
    there_is_a_problem = False
    for r, r_equiv in equivalence_sets.iteritems():
        if id(r_equiv) not in equivalence_sets_broken:
            equivalence_sets_broken[id(r_equiv)] = False
            #loop over the variables in the set comparing them to be equal enough
            re0 = None
            for re in r_equiv:
                if re0:
                    new_r_val = r_vals[re]
                    r_val = r_vals[re0]
                    assert re.type == re0.type
                    if not re.type.values_eq_approx(r_val, new_r_val):
                        equivalence_sets_broken[id(r_equiv)] = True
                        there_is_a_problem = True
                re0 = re

    if there_is_a_problem:
        # which broken equivalence set has the earliest-occurring element?
        first_broken_set = None
        for i, node in enumerate(order):
            for r in node.outputs:
                r_equiv = equivalence_sets[r]
                if equivalence_sets_broken[id(r_equiv)]:
                    first_broken_set = r_equiv
        #TODO finish this to produce good diagnostic information
        print first_broken_set
        raise Exception('broken')

def _find_bad_optimizations2(order, reasons, r_vals):
    """Use a simple algorithm to find broken optimizations.

    This algorithm is simple to understand, but sometimes when there's a problem it identifies
    the wrong optimization as the culprit.  The problem stems from the fact that results are
    not evaluated in chronological order (looking at when they were introduced to the graph).
    """

    checked_variables = set()

    def check_variable_norec(new_r):
        """Verify that `r` has the same value as the results it replaces """
        for reason, r, old_graph_str, new_graph_str in reasons[new_r]:
            new_r_val = r_vals[new_r]
            r_val = r_vals[r]

            if (r.type != new_r.type) or (not r.type.values_eq_approx(r_val, new_r_val)):
                raise BadOptimization(old_r=r,
                        new_r=new_r,
                        old_r_val=r_val,
                        new_r_val=new_r_val,
                        reason=reason,
                        old_graph=old_graph_str,
                        new_graph=new_graph_str)

    def check_variable(r):
        if r in checked_variables:
            return
        checked_variables.add(r)

        # (recursively) first check all the variables that could make r look bad:
        list_of_vars = [old_r for (reason, old_r, olds, news) in reasons[r]]
        if (None is not r.owner):
            list_of_vars += r.owner.inputs

        for var_that_could_make_r_look_bad in \
              list_of_vars:
                #backport
                #[old_r for (reason, old_r, olds, news) in reasons[r]] \
                #+ ([] if (None is r.owner) else r.owner.inputs):
            check_variable(var_that_could_make_r_look_bad)

        check_variable_norec(r)


    # iterate over variables looking for values that don't match the values of the
    # variables they replaced.  This is the sign of a broken optimization.
    for i, node in enumerate(order):
        for new_r in node.outputs:
            check_variable(new_r)

_find_bad_optimizations = _find_bad_optimizations0

def _check_preallocated_output(node, thunk, prealloc_modes, def_val,
        storage_map, r_vals, dr_vals, perform, active_order_set):
    '''Try to apply thunk() on different output storages'''

    # To avoid circular imports
    from theano.tensor import TensorType
    from theano.sandbox.cuda import cuda_available, CudaNdarrayType
    if cuda_available:
        from theano.sandbox.cuda import CudaNdarray

    # List of (name, map) pairs of the settings to test
    prealloc_maps = []
    # TODO: Sparse, Scalar
    # TODO: wrong shape, more stride patterns

    # reuse_output: use a copy of the same storage returned the first time
    # TODO: optimization warning if the storage in reuse_outputs
    # is not reused
    if 'previous' in prealloc_modes or 'ALL' in prealloc_modes:
        reuse_outputs = {}
        for r in node.outputs:
            # We want to reuse the exact same memory buffer,
            # so we keep the copy in r_vals
            new_r = _lessbroken_deepcopy(r_vals[r])
            reuse_outputs[r] = r_vals[r]
            r_vals[r] = new_r

        prealloc_maps.append(('previous', reuse_outputs))

    # c_cont_output: use a c-continuous array
    # (for TensorType and CudaNdarray, else None)
    if 'c_contiguous' in prealloc_modes or 'ALL' in prealloc_modes:
        c_cont_outputs = {}
        for r in node.outputs:
            if isinstance(r.type, (TensorType, CudaNdarrayType)):
                # Build a C-contiguous buffer
                new_buf = numpy.zeros(
                        shape=r_vals[r].shape,
                        dtype=r_vals[r].dtype,
                        order='C')
                new_buf += def_val
                if isinstance(r.type, CudaNdarrayType):
                    new_buf = CudaNdarray(new_buf)
                c_cont_outputs[r] = new_buf

        if len(c_cont_outputs):
            prealloc_maps.append(('c_contiguous', c_cont_outputs))

    # f_cont_output: use a fortran-continuous ndarray
    # (for TensorType, only)
    if 'f_contiguous' in prealloc_modes or 'ALL' in prealloc_modes:
        f_cont_outputs = {}
        for r in node.outputs:
            if isinstance(r.type, TensorType):
                new_buf = numpy.zeros(
                        shape=r_vals[r].shape,
                        dtype=r_vals[r].dtype,
                        order='F')
                new_buf += def_val
                f_cont_outputs[r] = new_buf

        if len(f_cont_outputs):
            prealloc_maps.append(('f_contiguous', f_cont_outputs))

    if 'neg_strides' in prealloc_maps:
        raise NotImplementedError('Negative strides in check_preallocated_output')

    for (name, out_map) in prealloc_maps:
        # _logger.debug('name = %s, perform = %s', name, perform)
        # Copy the inputs over again
        for r in node.inputs:
            storage_map[r][0] = _lessbroken_deepcopy(r_vals[r])

        # Get the appropriate output storages
        # (no copy)
        for r in node.outputs:
            storage_map[r][0] = out_map.get(r, None)

        thunk()

        # Check outputs
        for r in node.outputs:
            if not r.type.is_valid_value(storage_map[r][0]):
                raise InvalidValueError(r, storage_map[r][0],
                        hint='%s with %s output' % (perform, name),
                        specific_hint=r.type.value_validity_msg(storage_map[r][0]))

        _check_inputs(node, storage_map, r_vals, dr_vals, active_order_set,
                      clobber_dr_vals=False,
                      perform='%s with output %s' % (perform, name),
                      warn_input_not_reused=False)

        _check_viewmap(node, storage_map)

        for r in node.outputs:
            if not r.type.values_eq_approx(r_vals[r], storage_map[r][0]):
                # TODO: indicate it is not a C/Py problem
                raise BadCLinkerOutput(r, val_py=r_vals[r], val_c=storage_map[r][0])

        # Clear storage_map
        for r in node.outputs:
            storage_map[r][0] = None

class _EnvEvent(object):
    """A record of an event in the life of an Env.

    The __eq__ function is important here, as it is the basis for comparing optimization runs.
    """

    kind = ""
    """One of 'import', 'change', 'prune'"""

    node = None
    """Either 'output' or an Apply instance"""

    op = None
    """Either 'output' or an Op instance"""

    idx = None
    """change events involve an position index of the input variable"""

    reason = None
    """change events sometimes have a reason"""

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
            if (self.op != 'output'):
                msg = str(len(self.node.inputs))
            else:
                msg = ''

            return ' '.join(['change',
                self.reason,
                str(self.op),
                str(self.idx),
                msg])
                #backport
                #str(len(self.node.inputs)) if (self.op != 'output') else ''])
        else:
            return str(self.__dict__)

    def __eq__(self, other):
        rval = type(self) == type(other)
        if rval:
            # nodes are not compared because this comparison is supposed to be true for
            # corresponding events that happen in different Env instances (different graphs)
            for attr in ['kind', 'op', 'idx', 'reason']:
                rval = rval and getattr(self, attr) == getattr(other, attr)
        return rval

    def __ne__(self, other):
        return not (self == other)

class _VariableEquivalenceTracker(object):
    """A Env Feature that keeps tabs on an Env and tries to detect problems."""

    env = None
    """WRITEME"""

    equiv = None
    """WRITEME"""

    active_nodes = None
    """WRITEME"""

    inactive_nodes = None
    """WRITEME"""

    all_variables_ever = None
    """WRITEME"""

    reasons = None
    """WRITEME"""

    replaced_by = None
    """WRITEME"""

    event_list = None
    """WRITEME"""

    def __init__(self):
        self.env = None

    def on_attach(self, env):
        assert self.env is None
        self.equiv = {}
        self.active_nodes = set()
        self.inactive_nodes = set()
        self.env = env
        self.all_variables_ever = []
        self.reasons = {}
        self.replaced_by = {}
        self.event_list = []

    def on_detach(self, env):
        assert env is self.env
        self.env = None

    def on_prune(self, env, node):
        self.event_list.append(_EnvEvent('prune', node))
        #print 'PRUNING NODE', node, id(node)
        assert node in self.active_nodes
        assert node not in self.inactive_nodes
        self.active_nodes.remove(node)
        self.inactive_nodes.add(node)

    def on_import(self, env, node):
        self.event_list.append(_EnvEvent('import', node))

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
                self.all_variables_ever.append(r)
                self.reasons.setdefault(r, [])
                self.replaced_by.setdefault(r, [])
            for r in node.inputs:
                self.reasons.setdefault(r, [])
                self.replaced_by.setdefault(r, [])

    def on_change_input(self, env, node, i, r, new_r, reason=None):
        #print 'CHANGE by', reason, 'to use', new_r, type(new_r)
        self.event_list.append(_EnvEvent('change', node, reason=str(reason), idx=i))

        self.reasons.setdefault(new_r, [])
        self.replaced_by.setdefault(new_r, [])

        append_reason = True
        for tup in self.reasons[new_r]:
            if tup[0] == reason and tup[1] is r:
                append_reason = False

        if append_reason:
            # N.B. compute the debugprint now, because future optimizations will change the
            # graph
            self.reasons[new_r].append((reason
                , r
                , debugprint(r, prefix='  ', depth=6, file=StringIO()).getvalue()
                , debugprint(new_r, prefix='  ',  depth=6, file=StringIO()).getvalue()))
            self.replaced_by[r].append((reason, new_r))

        if r in self.equiv:
            r_set = self.equiv[r]
        else:
            r_set = self.equiv.setdefault(r, set([r]))
            self.all_variables_ever.append(r)

        if new_r in self.equiv:
            new_r_set = self.equiv[new_r]
        else:
            new_r_set = self.equiv.setdefault(new_r, set([new_r]))
            self.all_variables_ever.append(new_r)

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

class _Linker(gof.link.LocalLinker):
    """Special debugging linker"""
    def __init__(self, maker):
        super(gof.LocalLinker, self).__init__()
        self.env = None
        self.maker = maker

    def accept(self, env, no_recycling = []):
        if self.env is not None and self.env is not env:
            assert type(self) is _Linker
            return type(self)(self.env, self.maker).accept(env, no_recycling)
        self.env = env
        self.no_recycling = no_recycling
        return self

    def make_all(self, profiler = None, input_storage = None, output_storage = None):

        if 1:
            #can't import at toplevel because of circular import
            # TODO: don't do this ugly hacky way of setting the filter_checks_isfinite
            from theano.tensor import TensorType #to set filter_check_isfinite
            from theano import tests # for config.unittests.rseed
        env = self.env
        input_storage_ = input_storage
        output_storage_ = output_storage
        #order = env.toposort()

        #Compute a topological ordering that IGNORES the destroy_map of destructive Ops.
        #This will be OK, because every thunk is evaluated on a copy of its input.
        order_outputs = copy.copy(env.equivalence_tracker.all_variables_ever)
        order_outputs.reverse()
        order = graph.io_toposort(env.inputs, order_outputs)

        active_order = env.toposort()  #an ordering of just the active nodes
        active_order_set = set(active_order)

        no_recycling = self.no_recycling

        input_storage, output_storage, storage_map = link.map_storage(env, order,
                input_storage_, output_storage_)

        thunks_py = [] #python thunks
        thunks_c = [] #c thunks

        for node in order:
            node_input_storage = [storage_map[r] for r in node.inputs]
            node_output_storage = [storage_map[r] for r in node.outputs]

            try:
                if not self.maker.mode.check_c_code:
                    raise utils.MethodNotDefined()
                # Ops that do not inherit from gof.op.Op don't have certain
                # methods defined that the CLinker expects (Scan is an
                # exmaple, ifelse is another of such classes that inherit
                # directly from PureOp)
                if not isinstance(node.op, gof.op.Op):
                    raise utils.MethodNotDefined()
                e = Env(*graph.clone(node.inputs, node.outputs))
                e.toposort = lambda: e.nodes #WARNING: STOCHASTIC ORDER
                #  Specifically... e.nodes is a set, but of only 1 element

                cl = CLinker().accept(e, [r for r, r2 in zip(e.outputs, node.outputs) if r2 in no_recycling])

                thunk, node_input_filters, node_output_filters = cl.make_thunk(
                    input_storage = node_input_storage,
                    output_storage = node_output_storage)
                thunk.inputs = node_input_storage
                thunk.outputs = node_output_storage
                thunks_c.append(thunk)
            except (NotImplementedError, utils.MethodNotDefined):
                thunks_c.append(None)

            # Pure ops don't really have a perform ( or their perform just
            # raises an not implemented exception), so in those cases we
            # consider that we don't have a python implementation
            if ((self.maker.mode.check_py_code or thunks_c[-1] is None) and
                node.op.perform.func_code != gof.op.PureOp.perform.func_code):
                p = node.op.perform
                thunk = (lambda p = p, i = node_input_storage, o = node_output_storage, n =
                         node: p(n, [x[0] for x in i], o))
                thunk.inputs = node_input_storage
                thunk.outputs = node_output_storage
                thunk.perform = p
                thunks_py.append(thunk)
            else:
                thunks_py.append(None)

            # If the op define its own make_thunk, check it
            if node.op.make_thunk.im_func != theano.gof.Op.make_thunk.im_func:
                compute_map = {}
                for k in node.inputs:
                    compute_map[k] = [True]
                for k in node.outputs:
                    compute_map[k] = [False]
                thunk = node.op.make_thunk(node,
                                           storage_map,
                                           compute_map,
                                           no_recycling)

                # Right now there is no op that when called check if
                # its ouputs are computed and don't recompute itself.
                # I think it is not a good idea to do so as we only
                # call thunk when we want them computed. So those
                # check would be useless. In case some ops do it at
                # some point, we reset the compute_map of outputs to
                # False.
                #
                # Note RP: this warp_thunk doesn't work. What happens is
                # that for all ops that have a make_thunk, the same instance
                # of `wrap_thunk` gets used ( that has the same `thunk`
                # function, probably the one of the first of all those ops (
                # or the last .. I'm not sure). I don't know suffcient about
                # how python works to understand why. A bunch of tests fail
                # because of this, one of them being
                # theano/scan_module/tests/scan_tests.py:T_Scan.test_backwards
                #def wrap_thunk():
                #    for k in node.outputs:
                #        compute_map[k] = [False]
                #    thunk()

                if thunks_py[-1] is None:
                    thunks_py[-1] = thunk
                elif thunks_c[-1] is None:
                    thunks_c[-1] = thunk
                else:
                    _logger.warn("We won't check the perform function of node '%s' but we will check its make_thunk function"%node)
                    thunks_py[-1] = thunk

        if no_recycling is True:
            no_recycling = storage_map.values()
            no_recycling = utils.difference(no_recycling, input_storage)
        else:
            no_recycling = [storage_map[r] for r in no_recycling if r not in env.inputs]

        # Precompute some things for storage pre-allocation
        prealloc_modes = config.DebugMode.check_preallocated_output.split(':')
        try:
            def_val = int(config.unittests.rseed)
        except ValueError:
            def_val = 666

        #####
        # This is the function that runs when you evaluate the graph
        #####
        def f():
            ####
            # Note: `f` ignores the compute_map and evaluates the nodes in
            # topological order. In some sense, this is ok, and can be used
            # for now.
            #####
            _logger.debug("starting a DebugMode call")
            for x in no_recycling:
                x[0] = None

            # nest all this in try-finally to put storage *back* into storage_map when an
            # exception is raised
            original_storage_map_keys = [r for r in storage_map if r.owner is None]

            try:
                equiv_vals = {}
                problematic = set()
                # r_vals are the true values associated with each variable in the graph
                # they should not change during the evaluation of this function, even when the
                # graph has destructive ops in it
                #
                # This dictionary is used to populate the storage_map as necessary
                r_vals = {}

                # dr_vals are the values taken by variables after being destroyed
                dr_vals = {}
                assert len(thunks_py) == len(order)

                # transfer the initial values from the storage_map to the r_vals
                _logger.debug("DEBUGMODE: transfer initial values")
                # r_vals_initialized keeps track of the values that have
                # actually been transferred from storage_map to r_vals
                r_vals_initialized = []
                for r in storage_map:
                    if (r.owner is None):
                        if not r.type.is_valid_value(storage_map[r][0]):
                            # None may be a valid input value (for instance,
                            # for a Generic object). We only want to raise
                            # an error if it is not valid.
                            if (storage_map[r][0] is None):
                                raise InvalidValueError(r, storage_map[r][0],
                                       hint="Graph Input '%s' is missing" % str(r))
                            raise InvalidValueError(r, storage_map[r][0],
                                   hint=("Graph Input '%s' has invalid value "
                                       "%s" % (r, storage_map[r][0])))
                        r_vals[r] = storage_map[r][0]
                        storage_map[r][0] = None
                        r_vals_initialized.append(r)

                # TODO: store them in another map, and test the thunks on
                # them as output storages.
                for r in storage_map:
                    if r in env.outputs:
                        storage_map[r][0] = None

                #####
                #  Precondition: the storage map is empty, transferred completely to r_vals
                #####
                for r, s in storage_map.iteritems():
                    if s[0] is not None:
                        print r, s
                    assert s[0] is None

                #try:
                # compute the value of all variables
                for i, (thunk_py, thunk_c, node) in enumerate(zip(thunks_py, thunks_c, order)):
                    this_node_destroyed_variables = set()

                    _logger.debug("%i - starting node %i %s", i, i, node)

                    # put a copy of each input into the storage_map
                    # also, check that inputs have valid values
                    for r in node.inputs:
                        assert isinstance(r, gof.Variable)
                        assert r in r_vals
                        # print >> sys.stderr,i,  "DEBUGMODE: deepcopy input ", r
                        storage_map[r][0] = _lessbroken_deepcopy(r_vals[r])
                        if not r.type.is_valid_value(storage_map[r][0]):
                            raise InvalidValueError(r, storage_map[r][0], client_node=node)

                    ## On the first call to thunk_py(), its output storage will be None
                    if thunk_py:
                        _logger.debug("%i - running thunk_py with None as "
                                "output storage", i)
                        try:
                            thunk_py()
                        except utils.MethodNotDefined:
                            thunk_py = None #shouldn't have put it into the list in the first place

                    if thunk_py:
                        # check output values for type-correctness
                        for r in node.outputs:
                            if not r.type.is_valid_value(storage_map[r][0]):
                                raise InvalidValueError(r, storage_map[r][0], hint='perform output', specific_hint = r.type.value_validity_msg(storage_map[r][0]))

                        _check_inputs(node, storage_map, r_vals, dr_vals, active_order_set,
                                      clobber_dr_vals=True, perform='py',
                                      warn_input_not_reused=config.DebugMode.warn_input_not_reused)

                        _check_viewmap(node, storage_map)

                        # Retrieve each output from the storage_map
                        # The return values of this first run will be the reference ones
                        for r in node.outputs:
                            assert r not in r_vals
                            # print >> sys.stderr, i, "DEBUGMODE storing reference output %x" % id(storage_map[r][0])
                            r_vals[r] = storage_map[r][0]
                            storage_map[r][0] = None #clear the storage_map of outputs for the thunk_c

                        if config.DebugMode.check_preallocated_output:
                            _logger.debug(
                                    '%i - calling _check_preallocated_output '
                                    'with thunk_py', i)
                            _check_preallocated_output(
                                    node=node,
                                    thunk=thunk_py,
                                    prealloc_modes=prealloc_modes,
                                    def_val=def_val,
                                    storage_map=storage_map,
                                    r_vals=r_vals,
                                    dr_vals=dr_vals,
                                    perform='py',
                                    active_order_set=active_order_set)

                        # print >> sys.stderr, i, "DEBUGMODE thunk_py %100s %50s %30s" % (node,
                            #[(id(o), numpy.asarray(storage_map[o][0])[0,0]) for o in node.inputs],
                            #[(id(o), numpy.asarray(storage_map[o][0])[0,0]) for o in node.outputs])
                        sys.stdout.flush()

                    if thunk_c:

                        clobber = True
                        if thunk_py:
                            for r in node.inputs:
                                # if thunk_py ran, and we still got this far,
                                # it means that the destroy_map of the Op (and view_map) are
                                # accurate
                                # so we can assume that inputs not marked as destroyed have in
                                # fact not been destroyed.
                                # Therefore... we only need to overwrite inputs that *have*
                                # been marked as destroyed.

                                #TODO: The following was tried on revision 6c613932a63c,
                                # and made lots of tests fail, some complaining about
                                # AttributeError: 'Env' object has no attribute 'destroyers'
                                # some giving plain wrong numerical results.
                                #if env.destroyers(r):
                                #    storage_map[r][0] = _lessbroken_deepcopy(r_vals[r])

                                storage_map[r][0] = _lessbroken_deepcopy(r_vals[r])

                            clobber = False

                        _logger.debug("%i - running thunk_c", i)
                        ## First time, with None in output_storage
                        try:
                            thunk_c()
                        except Exception:
                            raise_with_op(node)

                        for r in node.outputs:
                            # check output values for type-correctness
                            if not r.type.is_valid_value(storage_map[r][0]):
                                raise InvalidValueError(r, storage_map[r][0], hint='c output')

                            if thunk_py:
                                assert r in r_vals #because we put it in during the thunk_py branch
                                # check for stride correctness (may raise exception)
                                _check_strides_match(r_vals[r], storage_map[r][0],
                                    self.maker.mode.require_matching_strides, node.op)

                        _check_inputs(node, storage_map, r_vals, dr_vals, active_order_set,
                                      clobber_dr_vals=clobber, perform='c',
                                      warn_input_not_reused=config.DebugMode.warn_input_not_reused)

                        _check_viewmap(node, storage_map)

                        # Check with Python result
                        for r in node.outputs:
                            if r in r_vals:
                                #print >> sys.stderr, i, "DEBUGMODE clearing output", r
                                # compares the version from thunk_py (in r_vals)
                                # to the version produced by thunk_c (in storage_map)
                                if not r.type.values_eq_approx(r_vals[r], storage_map[r][0]):
                                    #import pdb; pdb.set_trace()
                                    #r.type.values_eq_approx(r_vals[r], storage_map[r][0])
                                    raise BadCLinkerOutput(r, val_py=r_vals[r], val_c=storage_map[r][0])
                            else:
                                #print >> sys.stderr, i, "DEBUGMODE storing reference output %x" % id(storage_map[r][0])
                                #retrieve each output from the storage_map
                                r_vals[r] = storage_map[r][0]
                            storage_map[r][0] = None #clear the storage_map for the thunk_c

                        if config.DebugMode.check_preallocated_output:
                            def thunk():
                                try:
                                    thunk_c()
                                except Exception:
                                    raise_with_op(node)
                            _logger.debug(
                                    '%i - calling _check_preallocated_output '
                                    'with thunk_c', i)
                            _check_preallocated_output(
                                    node=node,
                                    thunk=thunk,
                                    prealloc_modes=prealloc_modes,
                                    def_val=def_val,
                                    storage_map=storage_map,
                                    r_vals=r_vals,
                                    dr_vals=dr_vals,
                                    perform='c code',
                                    active_order_set=active_order_set)

                        # print >> sys.stderr, i, "DEBUGMODE thunk_c  %100s %50s %30s" % (node,
                            #[(id(o), numpy.asarray(storage_map[o][0])[0,0]) for o in node.inputs],
                            #[(id(o), numpy.asarray(storage_map[o][0])[0,0]) for o in node.outputs])
                        sys.stdout.flush()


                    # we're done with this thunk
                    # clear everything out of the storage_map
                    for r in node.inputs:
                        #print >> sys.stderr, i, "DEBUGMODE clearing input", r
                        storage_map[r][0] = None
                    _logger.debug("%i - done with node", i)

                if False:
                    #This could be useful to help finding refcount problem.
                    #But it is very slow and it is not sure it will help.
                    gc.collect()

                _find_bad_optimizations(order, env.equivalence_tracker.reasons, r_vals)

                #####
                #  Postcondition: the input and output variables are in the storage map, nothing more
                #####

                # Nothing should be in storage map after evaluating each the thunk (specifically the
                # last one)
                for r, s in storage_map.iteritems():
                    assert type(s) is list
                    assert s[0] is None

                # store our output variables to their respective storage lists
                for output, storage in zip(env.outputs, output_storage):
                    storage[0] = r_vals[output]

                # transfer all inputs back to their respective storage lists
                for r in r_vals:
                    if r.owner is None:
                        if r in env.inputs:
                            assert storage_map[r] is input_storage[env.inputs.index(r)]
                        storage_map[r][0] = r_vals[r]

                # if an input was destroyed, the destroyed value should be returned
                for r in dr_vals:
                    assert dr_vals[r][0] is not None
                    if r.owner is None:
                        assert r in env.inputs
                        #HACK TO LOOK LIKE A REAL DESTRUCTIVE ACTION TOOK PLACE
                        if type(dr_vals[r][0]) is numpy.ndarray \
                                and dr_vals[r][0].dtype == storage_map[r][0].dtype \
                                and dr_vals[r][0].shape == storage_map[r][0].shape:
                            if len(dr_vals[r][0].shape):
                                storage_map[r][0][:] = dr_vals[r][0]
                            else:
                                storage_map[r][0].itemset(dr_vals[r][0])
                        else:
                            storage_map[r][0] = dr_vals[r][0]
            except Exception:
                # Restore the initial state of storage_map
                for r in storage_map:
                    if r in original_storage_map_keys:
                        # If r was transferred to r_vals, put it back
                        if r in r_vals_initialized:
                            storage_map[r][0] = r_vals[r]
                    else:
                        # clear out any partially-computed stuff
                        storage_map[r][0] = None
                raise

            #print ""
            #print output_storage
            #print dr_vals
            #print storage_map
            for r in storage_map:
                if (r.owner is None):
                    if not r.type.is_valid_value(None):
                        assert storage_map[r][0] is not None


            ###############
            # Done debugmode function call 'f'
            ##############

        def run_with_tensortype_filter_check(f):
            def deco():
                # WARNING: this is a global mechanism...
                # so it will screw up if we are trying to use
                # multiple modes at once.
                old_filter_checks_isfinite = TensorType.filter_checks_isfinite
                TensorType.filter_checks_isfinite = self.maker.mode.check_isfinite
                try:
                    return f()
                finally:
                    # put back the filter_checks_isfinite
                    TensorType.filter_checks_isfinite = old_filter_checks_isfinite
            return deco

        f = run_with_tensortype_filter_check(f)

        f.allow_gc = True
        assert len(env.inputs) == len(input_storage)
        assert len(env.outputs) == len(output_storage)
        #print 'make_all returning output', [id(z) for z in output_storage]
        return f, [link.Container(input, storage, readonly=False) for input, storage in zip(env.inputs, input_storage)], \
            [link.Container(output, storage, readonly=True) for output, storage in zip(env.outputs, output_storage)], \
            thunks_py, order

_NODEFAULT = ['NODEFAULT']
class _Maker(FunctionMaker): #inheritance buys a few helper functions
    """Special debugging FunctionMaker
    """
    verbose = 0
    """Verbosity level of compile-time and run-time checks. (Default 0: silent)"""


    def __init__(self, inputs, outputs, optimizer, mode,
            accept_inplace = False,
            function_builder = Function,
            profile=None):
        """
        :type inputs: a list of SymbolicInput instances

        :type outputs: a list of SymbolicOutput instances
                    outputs may also be a single Variable (not a list), in which
                    case the functions produced by FunctionMaker will return
                    their output value directly

        :param accept_inplace: True iff it is acceptable to have inplace operations
                    in the graph from the inputs to the outputs

        :note: this function sets TensorType.filter_checks_isfinite when `mode.check_isfinite` is True

        """
        self.profile = profile
        # Handle the case where inputs and/or outputs is a single Variable (not in a list)
        unpack_single = False
        return_none = False
        if outputs is None:
            return_none = True
            outputs = []
        if not isinstance(outputs, (list, tuple)):
            unpack_single = True
            outputs = [outputs]
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        # Wrap them in In or Out instances if needed.
        inputs, outputs =  map(self.wrap_in, inputs), map(self.wrap_out, outputs)
        _inputs = gof.graph.inputs([o.variable for o in outputs] + [i.update for i in inputs if getattr(i, 'update', False)])

#TODO: REMOVE THIS CRUFT - it's complicated for SymbolicInputKits
        indices = [[input] + self.expand_in(input, _inputs) for input in inputs]
        expanded_inputs = reduce(list.__add__, [list(z) for x, y, z in indices], [])

        assert expanded_inputs == inputs  #JB - I added this to make sure we could delete above

        # make the env
        for i in xrange(mode.stability_patience):
            env, additional_outputs, equivalence_tracker = _optcheck_env(expanded_inputs, outputs, accept_inplace)
            env.equivalence_tracker = equivalence_tracker

            # optimize the env
            compute_test_value_orig = theano.config.compute_test_value
            try:
                theano.config.compute_test_value = "off"
                optimizer(env)

                theano.compile.function_module.insert_deepcopy(env, inputs, outputs+additional_outputs)
            finally:
                theano.config.compute_test_value = compute_test_value_orig

            if i:
                li = env.equivalence_tracker.event_list
                l0 = env0.equivalence_tracker.event_list
                if li != l0 :
                    infolog = StringIO()
                    print >> infolog, "WARNING: Optimization process is unstable..."
                    print >> infolog, "  (HINT: Ops that the nodes point to must compare equal)"
                    print >> infolog, "(event index)  (one event trace)  (other event trace)"
                    print >> infolog, "-----------------------------------------------------"
                    for j in xrange(max(len(li), len(l0))):
                        if j >= len(li):
                            print >> infolog, 'trailing event in optimization 0 :', j
                            print >> infolog, '   ', str(l0[j])
                        elif j >= len(l0):
                            print >> infolog, 'trailing event in optimization', i, ':', j
                            print >> infolog, '   ', str(li[j])
                        elif li[j] != l0[j]:
                            print >>infolog, 'non-equal optimization events', i, ':', j
                            print >>infolog, '   ', str(l0[j])
                            print >>infolog, '   ', str(li[j])
                            #print >> infolog, "* ", j,
                            #if j < len(li):
                            #  msg =  str(li[j])
                            #else:
                            #  msg = '-'
                            #print >> infolog, "  ", msg
                            #if j < len(l0):
                            #  msg = str(l0[j])
                            #else:
                            #  msg = '-'
                            #print >> infolog, "  ", msg
                        else:
                            pass
                    raise StochasticOrder(infolog.getvalue())
                else:
                    if self.verbose:
                        print >> sys.stderr, "OPTCHECK: optimization", i, "of", len(li), "events was stable."
            else:
                env0 = env


        del env0
        self.env = env
        #equivalence_tracker.printstuff()

        linker = _Linker(self)


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
        self.return_none = return_none
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
                    default = _NODEFAULT
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
                if default is _NODEFAULT:
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
        fn = self.function_builder(_fn, _i, _o, self.indices, self.outputs, defaults, self.unpack_single, self.return_none, self)
        return fn

def _pickle_DebugMode_Maker(maker):
    raise NotImplementedError('DebugMode is not picklable (yet)')
copy_reg.pickle(_Maker, _pickle_DebugMode_Maker)

########################
#
# API symbol: DebugMode
#
########################

class DebugMode(Mode):
    """Evaluation Mode that detects internal theano errors.

    This mode catches several kinds of internal error:

    - inconsistent c_code and perform implementations (see `BadCLinkerOutput`)

    - a variable replacing another when their runtime values don't match.  This is a symptom of
      an incorrect optimization step, or faulty Op implementation (raises `BadOptimization`)

    - stochastic optimization ordering (raises `StochasticOrder`)

    - incomplete `destroy_map` specification (raises `BadDestroyMap`)

    - an op that returns an illegal value not matching the output Variable Type (raises
      InvalidValueError)

    Each of these exceptions inherits from the more generic `DebugModeError`.

    If there are no internal errors, this mode behaves like FAST_RUN or FAST_COMPILE, but takes
    a little longer and uses more memory.

    If there are internal errors, this mode will raise an `DebugModeError` exception.

    :remark: The work of debugging is implemented by the `_Maker`, `_Linker`, and
    `_VariableEquivalenceTracker` classes.

    """

    stability_patience = config.DebugMode.patience
    """
    When checking for the stability of optimization, recompile the graph this many times.
    """

    check_c_code = config.DebugMode.check_c
    """
    Should we evaluate (and check) the `c_code` implementations?
    """

    check_py_code = config.DebugMode.check_py
    """
    Should we evaluate (and check) the `perform` implementations? Always checked if no `c_code`.
    """

    check_isfinite = config.DebugMode.check_finite
    """
    Should we check for (and complain about) NaN/Inf ndarray elements?
    """

    require_matching_strides = config.DebugMode.check_strides
    """
    Should we check for (and complain about) Ops whose python and C outputs are ndarrays with
    different strides? (This can catch bugs, but is generally overly strict.) 0 no check, 1 warn, 2 err.
    """

    # This function will be used to create a FunctionMaker in
    # function_module.function
    def function_maker(self, i,o,m, *args, **kwargs):
        """Return an instance of `_Maker` which handles much of the debugging work"""
        assert m is self
        return _Maker(i, o, self.optimizer, self, *args, **kwargs)

    def __init__(self,
            optimizer='fast_run',
            stability_patience=None,
            check_c_code=None,
            check_py_code=None,
            check_isfinite=None,
            require_matching_strides=None,
            linker=None):
        """Initialize member variables.

        If any of these arguments (except optimizer) is not None, it overrides the class default.
        The linker arguments is not used. It is set their to allow Mode.requiring() and some other fct to work with DebugMode too.
        """
        if linker is not None and not issubclass(linker, _Linker):
            raise Exception("DebugMode can use only its own linker! Don't give him one to use it.", linker)

        super(DebugMode, self).__init__(
                optimizer=optimizer,
                linker=_Linker)

        if stability_patience is not None:
            self.stability_patience = stability_patience

        if check_c_code is not None:
            self.check_c_code = check_c_code

        if check_py_code is not None:
            self.check_py_code = check_py_code

        if check_isfinite is not None:
            self.check_isfinite = check_isfinite

        if require_matching_strides is not None:
            self.require_matching_strides = require_matching_strides

        if not (self.check_c_code or self.check_py_code):
            raise ValueError('DebugMode has to check at least one of c and py code')

register_mode('DEBUG_MODE',DebugMode(optimizer='fast_run'))

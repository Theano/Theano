"""Tensor optimizations addressing the ops in basic.py
"""
# TODO: intelligent merge for mul/add
# TODO: 0*x -> 0

import logging
_logger = logging.getLogger('theano.tensor.opt')

from theano import gof
from theano.gof import opt, InconsistencyError, TopoOptimizer, graph
from theano.gof import Variable, Constant
from theano.gof.utils import MethodNotDefined
from theano.configparser import config
from elemwise import Elemwise, DimShuffle
from theano import scalar
import basic as T
import inplace as I
import numpy, theano
import numpy as N #guys... please don't do this in the library :(
import operator
import itertools
import sys, os
from theano import compile  #to register the optimizer built by this file

from theano.gof.python25 import any, all
from theano.gof.opt import Optimizer
from theano.gof import toolbox, DestroyHandler
from basic import get_constant_value


# Utilities

def out2in(*local_opts):
    """WRITEME """
    return opt.TopoOptimizer(opt.LocalOptGroup(*local_opts),
                             order = 'out_to_in',
                             failure_callback=TopoOptimizer.warn_inplace)

def in2out(*local_opts, **kwargs):
    """WRITEME """
    return opt.TopoOptimizer(opt.LocalOptGroup(*local_opts),
                             order = 'in_to_out',
                             failure_callback=TopoOptimizer.warn_inplace,
                             **kwargs)

def _fill_chain(new_out, orig_inputs):
    for i in orig_inputs:
        new_out = T.fill(i, new_out)
    return [new_out]

def encompasses_broadcastable(b1, b2):
    """
    Returns True if the broadcastable patterns b1 and b2 are such that b2 is
    broadcasted to b1's shape and not the opposite.

    :param b1: the broadcastable attribute of a tensor type
    :param b2: the broadcastable attribute of a tensor type
    """
    if len(b1) < len(b2):
        return False
    b1 = b1[-len(b2):]
    return not any(v1 and not v2 for v1, v2 in zip(b1, b2))

def merge_broadcastables(broadcastables):
    return [all(bcast) for bcast in zip(*broadcastables)]

def scalarconsts_rest(inputs):
    """Partition a list of variables into two kinds:
    scalar constants, and the rest."""
    consts = []
    origconsts = []
    nonconsts = []
    for i in inputs:
        try:
            v = get_constant_value(i)
            consts.append(v)
            origconsts.append(i)
        except:
            nonconsts.append(i)
    return consts, origconsts, nonconsts

def broadcast_like(value, template, env):
    """Return a Variable with the same shape and dtype as the template,
    filled by broadcasting value through it. `value` will be casted as necessary.

    """
    shape_of = env.shape_feature.shape_of
    if template not in shape_of:
        raise NotImplementedError('broadcast_like currently requires the template Variable to be in the env already')
    rval = T.alloc(T.cast(value, template.dtype), *shape_of[template])
    assert rval.type == template.type
    return rval


@gof.optimizer
def insert_inplace_optimizer(env):
    """
    Usage: inplace_optimizer.optimize(env)
    
    Attempts to replace all Broadcast ops by versions of them
    that operate inplace. It operates greedily: for each Broadcast
    Op that is encountered, for each output, tries each input to
    see if it can operate inplace on that input. If so, makes the
    change and go to the next output or Broadcast Op.

    Examples:
      x + y + z -> x += y += z
      (x + y) * (x * y) -> (x += y) *= (x * y) or (x + y) *= (x *= y)
    """
    for node in list(graph.io_toposort(env.inputs, env.outputs)):
        op = node.op
        if not isinstance(op, Elemwise):
            continue
        baseline = op.inplace_pattern
        candidate_outputs = [i for i in xrange(len(node.outputs)) if i not in baseline]
        candidate_inputs = [i for i in xrange(len(node.inputs)) if i not in baseline.values()]
        for candidate_output in candidate_outputs:
            for candidate_input in candidate_inputs:
                inplace_pattern = dict(baseline, **{candidate_output: candidate_input})
                try:
                    new = Elemwise(
                        op.scalar_op.__class__(
                            scalar.transfer_type(
                                *[inplace_pattern.get(i, None) \
                                        for i in xrange(len(node.outputs))])),
                        inplace_pattern).make_node(*node.inputs)
                    env.replace_all_validate(zip(node.outputs, new.outputs),
                            reason="insert_inplace_optimizer")
                except (ValueError, TypeError, InconsistencyError), e:
                    continue
                candidate_inputs.remove(candidate_input)
                node = new
                baseline = inplace_pattern
                break
compile.optdb.register('inplace_opt', insert_inplace_optimizer, 75, 'fast_run', 'inplace') 

def register_canonicalize(lopt, *tags, **kwargs):
    name = (kwargs and kwargs.pop('name')) or lopt.__name__
    compile.optdb['canonicalize'].register(name, lopt, 'fast_run', *tags)
    return lopt

def register_specialize(lopt, *tags, **kwargs):
    name = (kwargs and kwargs.pop('name')) or lopt.__name__
    compile.optdb['specialize'].register(name, lopt, 'fast_run', *tags)
    return lopt

def register_stabilize(lopt, *tags, **kwargs):
    name = (kwargs and kwargs.pop('name')) or lopt.__name__
    compile.optdb['stabilize'].register(name, lopt, 'fast_run', *tags)
    return lopt

######################
# DimShuffle lifters #
######################

@gof.local_optimizer([None, None])
def local_dimshuffle_lift(node):
    """
    "Lifts" DimShuffle through Elemwise operations and merges
    consecutive DimShuffles. Basically, applies the following
    transformations on the whole graph:

    DimShuffle(Elemwise(x, y)) => Elemwise(DimShuffle(x), DimShuffle(y))
    DimShuffle(DimShuffle(x)) => DimShuffle(x)

    After this transform, clusters of Elemwise operations are
    void of DimShuffle operations.
    """
    op = node.op
    if not isinstance(op, DimShuffle):
        return False

    input = node.inputs[0]
    inode = input.owner
    if inode and isinstance(inode.op, Elemwise) and (len(input.clients)==1):
        return inode.op.make_node(*[DimShuffle(input.type.broadcastable,
                                               op.new_order,
                                               op.inplace)(input) for input in inode.inputs]).outputs
    if inode and isinstance(inode.op, DimShuffle):
        new_order = [x == 'x' and 'x' or inode.op.new_order[x] for x in op.new_order]
        inplace = op.inplace and inode.op.inplace
        iinput = inode.inputs[0]
        if new_order == range(len(new_order)) and (len(new_order) == iinput.type.ndim):
            return [iinput]
        else:
            return DimShuffle(iinput.type.broadcastable, new_order, inplace).make_node(iinput).outputs


register_canonicalize(local_dimshuffle_lift)
register_specialize(local_dimshuffle_lift)



#####################################
# ShapeFeature, Shape optimizations
#####################################

class MakeVector(T.Op):
    """Concatenate a number of scalars together into a vector

    This is a simple version of stack() that introduces far less cruft into the graph.
    
    """
    def __init__(self, dtype='int64'):
        self.dtype = dtype
    def __eq__(self, other):
        return type(self) == type(other) and self.dtype == other.dtype
    def __hash__(self):
        return hash(type(self)) ^ hash(self.dtype)
    def make_node(self, *inputs):
        inputs = map(T.as_tensor_variable, inputs)
        if not all(a.type == inputs[0].type for a in inputs):
            raise TypeError('This MakeVector instance requires inputs of same type %s' %
                    inputs[0].type)
        if inputs:
            dtype = inputs[0].type.dtype
        else:
            dtype = self.dtype
        #bcastable = (len(inputs) == 1)
        bcastable = False
        otype = T.TensorType(
                broadcastable=(bcastable,),
                dtype=dtype)
        return T.Apply(self, inputs, [otype()])
    def __str__(self):
        return self.__class__.__name__
    def perform(self, node, inputs, (out,)):
        # not calling theano._asarray as optimization
        if out[0] is None:
            out[0] = theano._asarray(inputs, dtype=node.outputs[0].dtype)
        else:
            # assume that out has correct dtype.  there is no cheap way to check
            out[0][...] = inputs

make_vector = MakeVector()

class MakeVectorPrinter:
    def process(self, r, pstate):
        if r.owner is None:
            raise TypeError("Can only print make_vector.")
        elif isinstance(r.owner.op, MakeVector):
            return "[%s]" % ", ".join(pstate.pprinter.process(input, pstate.clone(precedence = 1000)) for input in r.owner.inputs)
        else:
            raise TypeError("Can only print make_vector.")
T.pprint.assign(lambda pstate, r: r.owner and isinstance(r.owner.op, MakeVector), MakeVectorPrinter())

class Shape_i(T.Op):
    """
    L{Op} to return the shape of a matrix.

    @note: Non-differentiable.
    """
    def __init__(self, i):
        self.i = i
    def __hash__(self):
        return hash(type(self)) ^ self.i
    def __eq__(self, other):
        return type(self) == type(other) and self.i == other.i
    def __str__(self):
        return '%s{%i}'%(self.__class__.__name__, self.i)
    def make_node(self, x):
        # x could be one of a number of types
        # the only thing we require is that the variable have a .ndim,
        # and that the value have a .shape
        if not isinstance(x, T.Variable):
            raise TypeError('x must be Variable with ndim attribute', x)
        if x.ndim <= self.i:
            raise TypeError('x has too few dimensions for Shape_i', (x, self.i))
        return T.Apply(self, [x], [T.lscalar()])
    def perform(self, node, (x, ), (out, )):
        if out[0] is None:
            out[0] = theano._asarray(x.shape[self.i], dtype='int64')
        else:
            out[0][...] = x.shape[self.i]
    def c_code_cache_version(self):
        return (0,1)
    def c_code(self, node, name, (x, ), (out, ), sub):
        i = self.i
        if isinstance(node.inputs[0].type,T.TensorType):
            return """
            if(!%(out)s)
            %(out)s=(PyArrayObject*)PyArray_ZEROS(0, NULL, PyArray_INT64, 0);
            ((npy_int64*)PyArray_DATA(%(out)s))[0]=%(x)s->dimensions[%(i)s];
            """%locals()
        
        elif node.inputs[0].type.__class__.__name__=="CudaNdarrayType":
            #Don't want to import cuda stuff here.
            return """
            if(!%(out)s)
            %(out)s=(PyArrayObject*)PyArray_ZEROS(0, NULL, PyArray_INT64, 0);
            ((npy_int64*)PyArray_DATA(%(out)s))[0]=CudaNdarray_HOST_DIMS(%(x)s)[%(i)s];
            """%locals()
        else:
            return super(Shape_i, self).c_code(node, name, (x,), (out,), sub)
    def grad(self, (x,), (gz,)):
        return [None]

class ShapeFeature(object):
    """Graph optimizer for removing all calls to shape()
    
    This optimizer replaces all Shapes and Subtensors of Shapes with Shape_i and MakeVector
    Ops.

    This optimizer has several goals:
    1. to 'lift' Shapes to as close to the inputs as possible.  
    2. to infer the shape of every node in the graph in terms of the input shapes.
    3. remove all fills (T.second, T.fill) from the graph

    Lifting shapes as close to the inputs as possible is important for canonicalization because
    it is very bad form to have to compute something just to know how big it will be.  Firstly,
    it is a waste of time to compute such outputs.  But it is important to get rid of these
    outputs as early as possible in the compilation process because the
    extra computations make it appear as if many internal graph nodes have multiple clients.
    Many optimizations refuse to work on nodes with multiple clients.

    Lifting is done by using an `<Op>.infer_shape` function if one is present, or else using a
    conservative default.  An Op that supports shape-lifting should define a 
    infer_shape(self, node, input_shapes) function.  The argument input_shapes is a tuple
    of tuples... there is an interior tuple for each input to the node.  The tuple has as many
    elements as dimensions.  The element in position i of tuple j represents the i'th shape
    component of the j'th input.  The function should return a tuple of tuples.  One output
    tuple for each node.output.  Again, the i'th element of the j'th output tuple represents
    the output[j].shape[i] of the function.  If an output is not a TensorType, then None should
    be returned instead of a tuple for that output.

    For example the infer_shape for a matrix-matrix product would accept 
    input_shapes=((x0,x1), (y0,y1)) and return ((x0, y1),).
    

    Inferring the shape of internal nodes in the graph is important for doing size-driven
    optimizations.  If we know how big various intermediate results will be, we can estimate
    the cost of many Ops accurately, and generate c-code that is specific [e.g. unrolled] to
    particular sizes.

    If you can determine the shape only in some case, return NotImplementedError when you can't

    .. note::

        Right now there is only the ConvOp that could really take advantage of this shape
        inference, but it is worth it even just for the ConvOp.  All that's necessary to do
        shape inference is 1) to mark shared inputs as having a particular shape,
        either via a .tag or some similar hacking; and 2) to add an optional Param() argument
        to promise that inputs will have a certain shape (or even to have certain shapes in
        certain dimensions). We can't automatically infer the shape of shared variable as
        they can change of shape during the execution by default.
        (NOT IMPLEMENTED YET)


    """
    def shape_i(self, i):
        def op_deco(r):
            if r.type.broadcastable[i]:
                return self.lscalar_one
            else:
                return Shape_i(i)(r)
        return op_deco

    def shape_tuple(self, r):
        return tuple([self.shape_i(i)(r) for i in xrange(r.ndim)])

    def default_infer_shape(self, node, i_shapes):
        rval = []
        for r in node.outputs:
            try:
                rval.append(self.shape_tuple(r))
            except AttributeError:
                rval.append(None)
        return rval

    def unpack(self, s_i):
        # unpack the s_i that the Op returned
        assert s_i is not None
        if s_i == 1:
            # don't make the optimizer merge a zillion ones together
            return self.lscalar_one
        if type(s_i) is int or isinstance(s_i, numpy.integer):
            # this shape is a constant
            assert s_i >= 0
            return T.constant(s_i, dtype='int64')
        if type(s_i) in (tuple,list):
            # this dimension is the same as many of the inputs
            # which tells us that if one of the inputs is known, 
            # the others all become known.
            # TODO: should be implemented in Elemwise, and Dot
            #
            # worst case, we loop over shape_of and replace things
            raise NotImplementedError(s_i)
        elif s_i.type.dtype[:3] in ('int', 'uint'):
            if getattr(s_i.type, 'ndim', 0):
                raise TypeError('Shape element must be scalar', s_i)
            return s_i
        else:
            raise TypeError('Unsupported shape element', 
                    s_i, type(s_i), getattr(s_i, 'type', None))

    def set_shape(self, r, s):
        assert r not in self.shape_of
        if s is None:
            self.shape_of[r] = s
        else:
            self.shape_of[r] = tuple([self.unpack(s_i) for s_i in s])

    def init_r(self,r):
        if r not in self.shape_of:
            try:
                self.set_shape(r, self.shape_tuple(r))
            except AttributeError:
                self.set_shape(r,None)

    def make_vector_shape(self, r):
        return make_vector(*self.shape_of[r])
    #
    #
    # Feature inteface
    #
    #
    def on_attach(self, env):
        assert not hasattr(env, 'shape_feature')
        env.shape_feature = self
        self.shape_of = {} # Variable -> tuple(scalars) or None  (All tensor vars map to tuple)
        self.scheduled = {} # Variable -> 
        self.lscalar_one = T.constant(1, dtype='int64')
        assert self.lscalar_one.type == T.lscalar
        for node in env.toposort():
            self.on_import(env, node)

    def on_import(self, env, node):
        if node.outputs[0] in self.shape_of:
            # this is a revert, not really an import
            for r in node.outputs + node.inputs:
                assert r in self.shape_of
            return

        for i, r in enumerate(node.inputs):
            # make sure we have shapes for the inputs
            self.init_r(r)

        try:
            shape_infer = node.op.infer_shape
        except AttributeError:
            shape_infer = self.default_infer_shape

        try:
            o_shapes = shape_infer(node, [self.shape_of[r] for r in node.inputs])
        except NotImplementedError:
            o_shapes = self.default_infer_shape(node, [self.shape_of[r] for r in node.inputs])
        except Exception, e:
            _logger.error('Failed to infer_shape from Op %s (i_shapes=%s): %s %s'% (node.op,
                [self.shape_of[r] for r in node.inputs],
                type(e), str(e)))
            o_shapes = self.default_infer_shape(node, [self.shape_of[r] for r in node.inputs])

        # this is packed information
        # an element of o_shapes is either None or a tuple
        #   elements of the tuple can be either strings, or ints

        assert len(o_shapes) == len(node.outputs)

        for r, s in zip(node.outputs, o_shapes):
            self.set_shape(r, s)

    def on_change_input(self, env, node, i, r, new_r):
        # TODO:
        # This tells us that r and new_r must have the same shape
        # if we didn't know that the shapes are related, now we do.
        self.init_r(new_r)
        # change_input happens in two cases:
        # 1) we are trying to get rid of r, or
        # 2) we are putting things back after a failed transaction.

        # In case 1, if r has a shape_i client, we will want to replace the shape_i of r with
        # the shape of new_r.  Say that r is *scheduled*.
        # At that point, node is no longer a client of r, but of new_r
        for (shpnode, idx) in (r.clients + [(node, i)]):
            if isinstance(getattr(shpnode,'op', None), Shape_i):
                self.scheduled[shpnode] = new_r
        # In case 2, if r is a variable that we've scheduled for shape update, then we
        # should cancel it.
        # TODO: store some kind of reverse index?
        for k,v in self.scheduled.items():
            if v == r:
                del self.scheduled[k]

class ShapeOptimizer(Optimizer):
    """Optimizer that serves to add ShapeFeature as an env feature.
    """
    def __init__(self):
        Optimizer.__init__(self)

    def add_requirements(self, env):
        env.extend(ShapeFeature())

    def apply(self, env):
        pass

# -1 should make it run right before the first merge
theano.compile.mode.optdb.register('ShapeOpt', ShapeOptimizer(), -1, 'fast_run', 'fast_compile')

@register_specialize
@register_canonicalize
@gof.local_optimizer([T.fill])
def local_fill_to_alloc(node):
    """fill(s,v) -> alloc(v, shape(s))

    This is an important optimization because with the shape_to_shape_i optimization, the
    dependency on 's' is often removed.
    
    """
    if node.op == T.fill:
        r, v = node.inputs
        if v.type == node.outputs[0].type:
            # this is a useless fill, erase it.
            rval = [v]
        elif v.type.broadcastable == node.outputs[0].type.broadcastable:
            # this is a cast
            rval = [T.cast(v, node.outputs[0].type.dtype)]
        else:
            # we are broadcasting v somehow
            shape_of = node.env.shape_feature.shape_of
            # TODO: cut out un-necessary dimshuffles of v
            rval = [T.alloc(T.cast(v, node.outputs[0].dtype), *shape_of[node.outputs[0]])]
        
        #if rval[0].type != node.outputs[0].type:
            #print >> sys.stderr, theano.printing.debugprint(node.outputs[0], file='str')

        assert rval[0].type == node.outputs[0].type, ('rval', rval[0].type,
                'orig', node.outputs[0].type,
                'node', node,
                )#theano.printing.debugprint(node.outputs[0], file='str'))
        return rval

@register_specialize
@register_canonicalize
@gof.local_optimizer([T.alloc])
def local_useless_alloc(node):
    """
    if the input type is the same as the output type(dtype and broadcast)
    their is no change in the shape of the input. So this is just a simple copy
    of the input. This is not needed.
    """
    if node.op == T.alloc:
        if node.inputs[0].type == node.outputs[0].type:
            return [node.inputs[0]]

@register_specialize
@register_canonicalize
@gof.local_optimizer([T._shape])
def local_shape_to_shape_i(node):
    if node.op == T._shape:
        shape_feature = node.env.shape_feature
        return [shape_feature.make_vector_shape(node.inputs[0])]

@register_specialize
@register_canonicalize
@gof.local_optimizer([T._shape])
def local_track_shape_i(node):
    try:
        shape_feature = node.env.shape_feature
    except:
        return
    if node in shape_feature.scheduled:
        assert isinstance(node.op, Shape_i)
        replacement = shape_feature.scheduled[node]
        return [shape_feature.shape_of[replacement][node.op.i]]

@register_specialize
@register_canonicalize
@gof.local_optimizer([T.Subtensor])
def local_subtensor_make_vector(node):
    # replace all subtensor(make_vector) like:
    # [a,b,c][0] -> a
    # [a,b,c][0:2] -> [a,b]
    # we can do this for constant indexes
    if isinstance(node.op, T.Subtensor):
        shape_feature = node.env.shape_feature
        x = node.inputs[0]
        if x.owner and x.owner.op == make_vector:
            try:
                idx, = node.op.idx_list
            except:
                #'how can you have multiple indexes into a shape?'
                raise

            if isinstance(idx, (scalar.Scalar, T.TensorType)):
                # The idx is a Scalar, ie a Type. This means the actual index 
                # is contained in node.inputs[1]
                old_idx, idx = idx, node.inputs[1]
                assert idx.type == old_idx

            if isinstance(idx, (int, numpy.integer)):
                return [x.owner.inputs[idx]]
            elif isinstance(idx, Variable):
                # if it is a constant we can do something with it
                try:
                    v = get_constant_value(idx)
                    return [x.owner.inputs[v]]
                except:
                    pass
            else:
                # it is a slice of ints and/or Variables
                #TODO: check subtensor to see if it can contain constant variables,
                #      and if it can, then try to unpack them.
                try:
                    return [make_vector(*x.owner.inputs.__getitem__(idx))]
                except TypeError:
                    pass
                except:
                    _logger.error('failed to index with "%s"' % str(idx))
                    raise

@register_canonicalize
@register_specialize
@gof.local_optimizer([T.Elemwise])
def local_useless_eq(node):
    """eq(x,x) -> 1
    """
    if isinstance(node.op, T.Elemwise) and node.op.scalar_op == theano.scalar.eq and len(node.inputs)==2:
        if node.inputs[0]==node.inputs[1]:
            #it is the same var in the graph. That will always be true
            return [T.fill(node.inputs[0], T.constant(1.0, dtype=node.outputs[0].type.dtype))]

@register_canonicalize
@register_specialize
@gof.local_optimizer([T.Elemwise])
def local_useless_neq(node):
    """neq(x,x) -> 0
    """
    if isinstance(node.op, T.Elemwise) and node.op.scalar_op == theano.scalar.neq and len(node.inputs)==2:
        if node.inputs[0]==node.inputs[1]:
            #it is the same var in the graph. That will always be true
            return [T.fill(node.inputs[0], T.constant(0.0, dtype=node.outputs[0].type.dtype))]

#TODO: the other optimization for and, or, xor, le and ge see ticket #496.

@register_specialize
@gof.local_optimizer([T.Elemwise])
def local_alloc_unary(node):
    """unary(alloc(x, shp)) -> alloc(unary(x), shp)
    """
    if isinstance(node.op, T.Elemwise) and len(node.inputs)==1:
        a = node.inputs[0]
        if a.owner and isinstance(a.owner.op, T.Alloc):
            x = a.owner.inputs[0]
            shp = a.owner.inputs[1:]
            v = node.op(x)
            return [T.alloc(T.cast(v, node.outputs[0].dtype), *shp)]

class Assert(T.Op):
    view_map={0:[0]}
    def make_node(self, value, *conds):
        cond = [T.as_tensor_variable(c) for c in conds]
        assert numpy.all([c.type.ndim == 0 for c in cond])
        return gof.Apply(self, [value]+cond, [value.type()])
    
    def __str__(self):
        return self.__class__.__name__
    def perform(self, node, inputs, (out,)):
        v = inputs[0]
        out[0]=v
        assert numpy.all(inputs[1:])

    def __eq__(self, other):
        return type(self)==type(other)
    def __hash__(self):        return hash(type(self))
    def grad(self,input,output_gradients):
        return output_gradients
    def c_code(self, node, name, inames, onames, sub):
        value = inames[0]
        out = onames[0]
        check = []
        fail = sub['fail']
        for idx in range(len(inames)-1):
            i=inames[idx+1]
            dtype=node.inputs[idx+1].dtype
            check.append('if(!((npy_%(dtype)s*)PyArray_DATA(%(i)s))[0]){PyErr_SetString(PyExc_AssertionError,"Theano Assert failed!");%(fail)s}'%locals())
        check = "\n".join(check)
        return """
        %(check)s
        %(out)s = %(value)s;
        Py_INCREF(%(value)s);
        """%locals()
        pass
    def c_code_cache_version(self):
        return (0,1)

    def infer_shape(self, node, input_shapes):
        return [input_shapes[0]]
    
assert_ = Assert()

@register_specialize
@gof.local_optimizer([Assert])
def local_remove_useless_assert(node):
    if isinstance(node.op, Assert):
        cond=[]
        for c in node.inputs[1:]:
            try:
                const = get_constant_value(c)
                
                if 0!=const.ndim or const==0:
                    #Should we raise an error here? How to be sure it is not catched?
                    cond.append(c)
            except TypeError:
                cond.append(c)
        
        if len(cond)==0:
            return [node.inputs[0]]
        if len(cond)!=len(node.inputs)-1:
            return [assert_(node.inputs[0],*cond)]

@gof.local_optimizer([T.Alloc])
def local_alloc_elemwise(node):
    """
    elemwise(alloc(x, shp), ..., y.TensorType(BROADCAST CONDITION))
      -> elemwise(x, y.TensorType(no broadcast flag))

    elemwise(dimshuffle(alloc(x, shp)),... ,y.TensorType(BROADCAST CONDITION))
      -> elemwise(x, y.TensorType(no broadcast flag))

    BROADCAST CONDITION: the condition is that the one input that are not to be optimized to have the same braodcast pattern as the output

         We can change the alloc by a dimshuffle as the elemwise already have the shape info.
         The dimshuffle will be faster to exec
    """
    if not isinstance(node.op, T.Elemwise):
        return False
    if len(node.outputs)>1:
        #This is a supposition this code make that I'm not sure is always true.
        assert all([list(o.type.broadcastable) == list(node.outputs[0].type.broadcastable) for o in node.outputs[1:]])

    if not any([list(i.type.broadcastable)==list(node.outputs[0].type.broadcastable) for i in node.inputs]):
        return False
    if not any([i.owner and (isinstance(i.owner.op,T.Alloc) or \
                             (isinstance(i.owner.op,T.DimShuffle) and i.owner.inputs[0].owner and \
                              isinstance(i.owner.inputs[0].owner.op,T.Alloc))) for i in node.inputs]):
        return False
    no_broad_idx = -1
    for idx,i in enumerate(node.inputs):
        if not i.owner:
            if list(i.type.broadcastable) == [False,]*i.type.ndim:
                no_broad_idx = idx
                break
            else:
                continue
        if not any(i.type.broadcastable) and not isinstance(i.owner.op, T.Alloc):
            no_broad_idx = idx
            break
        elif list(i.type.broadcastable)==list(node.outputs[0].type.broadcastable) \
             and not isinstance(i.owner.op, T.Alloc) \
             and not (isinstance(i.owner.op, T.DimShuffle) and i.owner.inputs[0].owner and \
                      isinstance(i.owner.inputs[0].owner.op,T.Alloc)):
            no_broad_idx = idx
            break
            
    assert no_broad_idx>=0
    assert_op = node.inputs[no_broad_idx]
    cmp_op = assert_op
    new = []
    
    for i in node.inputs:
        if i.owner and isinstance(i.owner.op,T.Alloc) and i.owner.inputs[0].type != i.owner.outputs[0].type:
            #when i.owner.inputs[0].type == i.owner.outputs[0].type we will remove that alloc later

            assert i.type.ndim == cmp_op.ndim
            if theano.config.experimental.local_alloc_elemwise_assert:
                assert_op = assert_(assert_op,*[T.eq(i.shape[idx],cmp_op.shape[idx])\
                                                    for idx in range(i.type.ndim) \
                                                    if not i.type.broadcastable[idx]])
            new.append(i.owner.inputs[0])
        elif i.owner and isinstance(i.owner.op, T.DimShuffle) and i.owner.inputs[0].owner \
             and isinstance(i.owner.inputs[0].owner.op,T.Alloc):
            assert i.type.ndim == cmp_op.type.ndim
            if theano.config.experimental.local_alloc_elemwise_assert:
                assert_op = assert_(assert_op,*[T.eq(i.shape[idx],cmp_op.shape[idx]) for idx \
                                                    in range(i.type.ndim) if not i.type.broadcastable[idx]])
            new.append(i.owner.inputs[0].owner.inputs[0])
        else: new.append(i)
    new[no_broad_idx]=assert_op
    if theano.config.experimental.local_alloc_elemwise_assert:
        assert assert_op.owner.op is assert_
    return [node.op(*new)]

#TODO, global optimizer that lift the assert to the beginning of the graph.
#TODO, var.tag.shape to propagate the shape and lower the overhead of this op
#TODO, when all inputs can be optimized do all except one

theano.configparser.AddConfigVar('experimental.local_alloc_elemwise',
        "If True enable the experimental optimization local_alloc_elemwise",
        theano.configparser.BoolParam(False),
        )
#This version if faster but not as save.
theano.configparser.AddConfigVar('experimental.local_alloc_elemwise_assert',
        "If False enable the experimental optimization local_alloc_elemwise but WITHOUT assert into the graph!",
        theano.configparser.BoolParam(True),
        )
if theano.config.experimental.local_alloc_elemwise:
    #enabled by default when the lifter of assert is done.
    register_specialize(local_alloc_elemwise)
else:
    #don't register them in fast_run by default to have them disabled by default
    #disable them by default as we are not sure it is always a good idea to replace an alloc with multiple op.
    compile.optdb['specialize'].register("local_alloc_elemwise", local_alloc_elemwise)

############################
# Constant Canonicalization
############################

@register_canonicalize
@gof.local_optimizer([])
def local_upcast_elemwise_constant_inputs(node):
    """This explicitly upcasts constant inputs to elemwise Ops, when those Ops do implicit upcasting anyway.

    Rationale: it helps merge things like (1-x) and (1.0 - x).
    """
    if len(node.outputs)>1:
        return
    try:
        shape_i = node.env.shape_feature.shape_i
    except AttributeError:
        shape_i = None
    if isinstance(node.op, T.Elemwise):
        scalar_op = node.op.scalar_op
        #print "aa", scalar_op.output_types_preference
        if getattr(scalar_op,'output_types_preference',None) in (T.scal.upgrade_to_float, T.scal.upcast_out):
            # this is the kind of op that we can screw with the input dtypes by upcasting
            # explicitly
            output_dtype = node.outputs[0].type.dtype
            new_inputs = []
            for i in node.inputs:
                if i.type.dtype == output_dtype:
                    new_inputs.append(i)
                else:
                    try:
                        cval_i = get_constant_value(i)    # works only for scalars
                        if all(i.broadcastable):
                            new_inputs.append(T.cast(cval_i, output_dtype))
                        else:
                            if shape_i is None:
                                return
                            new_inputs.append(T.alloc(T.cast(cval_i, output_dtype),
                                *[shape_i(d)(i) for d in xrange(i.ndim)]))
                            #print >> sys.stderr, "AAA", *[Shape_i(d)(i) for d in xrange(i.ndim)]
                    except TypeError:
                        if isinstance(i, T.TensorConstant): #for the case of a non-scalar
                            new_inputs.append(T.cast(i, output_dtype))
                        else:
                            new_inputs.append(i)

            if new_inputs != node.inputs:
                rval = [node.op(*new_inputs)]
                if rval[0].type != node.outputs[0].type:
                    print >> sys.stderr, "NODE:", node
                    print >> sys.stderr, "NODE INPUT TYPES:", [i.type for i in node.inputs]
                    print >> sys.stderr, "NODE OUTPUT TYPES:", [o.type for o in node.outputs]
                    print >> sys.stderr, "RVAL:", rval
                    print >> sys.stderr, "NEW INPUT TYPES:", [i.type for i in new_inputs]
                    print >> sys.stderr, "RVAL INPUT TYPES:", [i.type for i in rval[0].owner.inputs]
                    print >> sys.stderr, "RVAL TYPES:", [o.type for o in rval]
                assert rval[0].type == node.outputs[0].type, (node, rval[0])
                return rval

##################
# Subtensor opts #
##################

@register_canonicalize
@gof.local_optimizer([])
def local_subtensor_unary(node):
    """
    unary(x)[idx] -> unary(x[idx])
    """
    if isinstance(node.op, T.Subtensor):
        u = node.inputs[0]
        if u.owner and isinstance(u.owner.op, T.Elemwise) and len(u.owner.inputs)==1:
            idx = node.inputs[1:]
            x_idx = node.op(u.owner.inputs[0], *idx)
            return [u.owner.op(x_idx)]

@register_canonicalize
@gof.local_optimizer([None])
def local_IncSubtensor_serialize(node):
    """
    When using Subtensor, gradient graphs can be ugly.

    If we ask for grad(f(a[0]), a), we are going to get something like

        IncSubtensor(Elemwise{second}(a, 0), g(f(a[0])), [0])

    This might be ugly, but at least it's as fast as you could want.  If we ask for
    grad(f(a[0], a[1], a[2]), a), it's much worse...

        Elemwise{Add}
            IncSubtensor(Elemwise{second}(a, 0), g(f(a[0])), [0])
            IncSubtensor(Elemwise{second}(a, 0), g(f(a[1])), [1])
            IncSubtensor(Elemwise{second}(a, 0), g(f(a[2])), [2])

    This is much worse because this time we have to produce 3 matrices the size of 'a', just so
    we can add them together. 
    
    This Op rearranges IncSubtensor's that all work on the same initial argument (here,
    Elemwise{second}(a,0)) into a chain.  The advantage of the chain structure is that each one
    can be optimized later in the pipeline to operate inplace.

    Ideally, the op will do something like this:

    #
    #  add(x, incsubtensor(b, c), incsubtensor(b, d))
    #  -> incsubtensor(incsubtensor(add(x,b,b), c), d)
    
    """
    def movable(i):
        # Return True iff this is a incsubtensor that we can move
        return i.owner \
                and isinstance(i.owner.op, T.IncSubtensor) \
                and i.type == o_type \
                and len(i.clients) == 1 \
                and not i.owner.op.set_instead_of_inc

    if node.op == T.add:
        o_type = node.outputs[0].type

        movable_inputs = [i for i in node.inputs if movable(i)]

        if movable_inputs:
            new_inputs = [i for i in node.inputs if not movable(i)] \
                    + [mi.owner.inputs[0] for mi in movable_inputs]
            new_add = T.add(*new_inputs)

            # stack up the new incsubtensors
            tip = new_add
            for mi in movable_inputs:
                assert tip.type == o_type
                assert tip.type == mi.owner.inputs[0].type
                tip = mi.owner.op(tip, *mi.owner.inputs[1:])
            return [tip]

        #print incsub_inputs, [id(i.owner.inputs[0]) for i in incsub_inputs]


#after priority 50 Destructive inplace operations
#gemm is the first one now, at priority 70

@gof.local_optimizer([None])
def local_inplace_setsubtensor(node):
    """
    Also work for GpuIncSubtensor
    """
    if isinstance(node.op, T.IncSubtensor) and not node.op.inplace:
        new_op = node.op.__class__(node.op.idx_list, inplace=True, \
                        set_instead_of_inc=node.op.set_instead_of_inc)
        new_node = new_op(*node.inputs)
        return [new_node]
    return False
compile.optdb.register('inplace_setsubtensor', TopoOptimizer(local_inplace_setsubtensor,
    failure_callback=TopoOptimizer.warn_inplace), 60, 'fast_run', 'inplace') #DEBUG


####################
# Rebroadcast opts #
####################

@register_canonicalize
@register_specialize
@gof.local_optimizer([T.Rebroadcast])
def local_useless_rebroadcast(node):
    """
    Remove Rebroadcast if id does not actually change the broadcasting pattern
    """
    if isinstance(node.op, T.Rebroadcast):
        x = node.inputs[0]
        if numpy.all(x.broadcastable == node.outputs[0].broadcastable):
            # No broadcastable flag was modified
            return [x]
        else:
            # Keep the flags that modify something
            new_axis = {}
            for dim, bc in node.op.axis.items():
                if x.broadcastable[dim] != bc:
                    new_axis[dim] = bc
            if new_axis == node.op.axis:
                # All flags are useful
                return
            else:
                return [T.Rebroadcast(*new_axis.items())(x)]


@register_canonicalize
@register_specialize
@gof.local_optimizer([T.Rebroadcast])
def local_rebroadcast_lift(node):
    """
    Lifts Rebroadcast through unary Elemwise operations,
    and merges consecutive Rebroadcasts.

    Rebroadcast(Elemwise(x)) => Elemwise(Rebroadcast(x))
    Rebroadcast(Rebroadcast(x)) => Rebroadcast(x)
    """
    op = node.op
    if not isinstance(op, T.Rebroadcast):
        return False

    input = node.inputs[0]
    inode = input.owner
    if inode and isinstance(inode.op, Elemwise) and len(inode.inputs) == 1:
        if len(input.clients)==1:
            rval = inode.op.make_node(T.Rebroadcast(*op.axis.items())(inode.inputs[0])).outputs
            return rval
    if inode and isinstance(inode.op, T.Rebroadcast):
        # the "axis" specification in the outer Rebroadcast overrides
        # the axis of the inner one
        axis = inode.op.axis.copy()
        axis.update(op.axis)
        iinput = inode.inputs[0]
        rval = [T.Rebroadcast(*axis.items())(iinput)]
        return rval

def apply_rebroadcast_opt(rval):
    """
    Apply as many times as required the optimization local_useless_rebroadcast 
    and local_rebroadcast_lift.

    :param rval: a Variable
    :retrun: a Variable. The same if not optimisation can be applied.
    """

    changed = True
    while changed and rval.owner:
      changed = False
      rval2 = theano.tensor.opt.local_useless_rebroadcast.transform(rval.owner)
      if rval2: 
        assert len(rval2)==1
        rval = rval2[0]
        changed = True
      if rval.owner:
        rval2 = theano.tensor.opt.local_rebroadcast_lift.transform(rval.owner)
        if rval2:
          assert len(rval2)==1
          rval = rval2[0]
          changed = True
    return rval

##################
# Reshape opts   #
##################


@gof.local_optimizer([None, None])
def local_reshape_chain(node):
    """
    Reshape(Reshape(shape1),shape2) -> Reshape(shape2)
    """
    if not opt.check_chain(node, T.Reshape, T.Reshape):
        return False
    
    # TODO: this can permit a failing program to run by eliminating the the lower
    #       reshape
    return [node.op(node.inputs[0].owner.inputs[0], node.inputs[1])]
register_canonicalize(local_reshape_chain)

if 0:
    # TODO: Test that this optimziation works.
    @register_canonicalize
    @gof.local_optimizer([])
    def local_scalar_reshape(node):
        """Eliminate reshape Ops whose inputs and outputs are scalars """
        if isinstance(node.op, T.Reshape):
            x, shp = node.inputs
            if x.ndim == 0 and T.get_vector_length(shp)==0:
                return [x]

if 0:
    # TODO: Finish writing and testing this optimization.
    #       The idea is that if we can prove the output to this sum
    #       has a zero-size dimension, then it can be replaced by an appropriately typed and
    #       broadcasted zero.
    @register_canonicalize
    @gof.local_optimizer([])
    def local_sum_over_empty(node):
        if isinstance(node.op, T.Sum):
            y, = node.outputs
            y_shape = node.env.shape_feature.shape_of[y]

            def tmp(thing):
                try: 
                    return T.get_constant_value(thing)
                except (TypeError, ValueError), e:
                    print e, thing.owner.inputs[0]
                    return None
            print 'LOCAL SUM EMPTY', [tmp(s) for s in y_shape]

##################
# Middleman cuts #
##################

@gof.local_optimizer([None, T.fill])
def local_fill_cut(node):
    """
    f(fill(a,b), c) -> f(b, c)
    If c.type == a.type.
    """

    # this optimization is essentially for getting broadcasting to replace fill. 
    # This is always possible when using a Compound Elemwise operation, 
    # but it is not always possible without one (consider filling a large matrix with a scalar,
    # and then adding another scalar.  The only numbers that count are the two scalars, but we
    # can't ignore the large matrix because it gives the shape of the result.

    if not opt.check_chain(node, T.Elemwise):
        return False
    
    output = node.outputs[0]
    try:
        #reference is some input with the same type as the input but that is not produced by a fill
        reference = [input
                     for input in node.inputs
                     if input.type == output.type and (not input.owner or input.owner.op != T.fill)][0]
    except IndexError:
        return False

    new_inputs = []
    for input in node.inputs:
        if opt.check_chain(input, T.fill):
            model, filling = input.owner.inputs
            if encompasses_broadcastable(reference.type.broadcastable,
                                         filling.type.broadcastable):
                new_inputs.append(filling)
                continue
        new_inputs.append(input)

    if new_inputs == node.inputs:
        return False

    rval = node.op(*new_inputs)
    if isinstance(rval, gof.Variable):
        return rval.owner.outputs
    else:
        return rval[0].owner.outputs

register_canonicalize(local_fill_cut)

register_canonicalize(gof.OpRemove(T.tensor_copy), name='remove_tensor_copy' )

@gof.local_optimizer([None, T.fill])
def local_fill_sink(node):
    """
    f(fill(a, b), fill(c, d), e) -> fill(a, fill(c, f(b, d, e)))
    """
    if not (node.op and isinstance(node.op, T.Elemwise) and node.op != T.fill):
        return False
    models = []
    inputs = []
    for input in node.inputs:
        if input.owner and input.owner.op == T.fill:
            models.append(input.owner.inputs[0])
            inputs.append(input.owner.inputs[1])
        else:
            inputs.append(input)
    if inputs == node.inputs:
        return False
    c = node.op(*inputs)
    for model in models:
        c = T.fill(model, c)
    return [c]

register_canonicalize(local_fill_sink)


################
# Canonization #
################

class Canonizer(gof.LocalOptimizer):
    """
    Simplification tool.

    Usage: Canonizer(main, inverse, reciprocal, calculate)
    
    * main: a suitable Op class that is commutative, associative and
            takes one to an arbitrary number of inputs, e.g. add or
            mul
    * inverse: an Op class such that inverse(main(x, y), y) == x
               e.g. sub or true_div
    * reciprocal: a function such that main(x, reciprocal(y)) ==
                  inverse(x, y) e.g. neg or inv

    * calculate: function that takes a list of numpy.ndarray instances
                 for the numerator, another list for the denumerator,
                 and calculates inverse(main(*num), main(*denum)). It
                 takes a keyword argument, aslist. If True, the value
                 should be returned as a list of one element, unless
                 the value is such that value = main(). In that case,
                 the return value should be an empty list.

    The variable is a local_optimizer. It is best used with a TopoOptimizer in
    in_to_out order.

    Examples:
      T = theano.tensor
      add_canonizer = Canonizer(T.add, T.sub, T.neg, lambda n, d: sum(n) - sum(d))
      mul_canonizer = Canonizer(T.mul, T.true_div, T.inv, lambda n, d: prod(n) / prod(d))
    
    Examples of optimizations mul_canonizer can perform:
      x / x -> 1
      (x * y) / x -> y
      x / y / x -> 1 / y
      x / y / z -> x / (y * z)
      x / (y / z) -> (x * z) / y
      (a / b) * (b / c) * (c / d) -> a / d
      (2.0 * x) / (4.0 * y) -> (0.5 * x) / y
      2 * x / 2 -> x
      x * y * z -> Elemwise(T.mul){x,y,z} #only one pass over the memory.
                !-> Elemwise(T.mul){x,Elemwise(T.mul){y,z}}
    """

    def __init__(self, main, inverse, reciprocal, calculate, use_reciprocal = True):
        self.main = main
        self.inverse = inverse
        self.reciprocal = reciprocal
        self.calculate = calculate
        self.use_reciprocal = use_reciprocal

        self.external_simplifiers = []

    def add_simplifier(self, simplifier, reason):
        self.external_simplifiers.append((reason, simplifier))

    def tracks(self):
        return [[self.main, None], [self.inverse, None], [self.reciprocal, None]]

    def get_num_denum(self, input):
        """
        This extract two lists, num and denum, such that the input is:
        self.inverse(self.main(*num), self.main(*denum)). It returns
        the two lists in a (num, denum) pair.

        For example, for main, inverse and reciprocal = *, / and inv(),

        input -> returned value (num, denum)

        x*y -> ([x, y], [])
        inv(x) -> ([], [x])
        inv(x) * inv(y) -> ([], [x, y])
        x*y/z -> ([x, y], [z])
        log(x) / y * (z + x) / y -> ([log(x), z + x], [y, y])
        (((a / b) * c) / d) -> ([a, c], [b, d])
        a / (b / c) -> ([a, c], [b])
        log(x) -> ([log(x)], [])
        x**y -> ([x**y], [])
        x * y * z -> ([x, y, z], [])

        """

        # This function is recursive.
        # The idea is that there is a get_num_denum recursion in which the internal ops are all
        # one of (main, inverse, reciprocal, DimShuffle) and the internal data nodes all have
        # the dtype of the 'input' argument. The leaf-Variables of the graph covered by the
        # recursion may be of any Variable type.

        if 0:
            # UPDATE: This logic makes it impossible to recognize some important patterns
            # (e.g. variants on the x/x)
            # and it is screwing up the RBM free energy gradient.
            #TODO: review this
            if len(input.clients) > 1:
                # this logic is too conservative, but doing it is better than not doing it.
                #
                # we don't want to canonize a subgraph that we will need to compute anyway for the other clients.
                # This check is too conservative because if the other clients are also in the subgraph we are canonizing,
                # then we should [probably?] recurse anyway.
                return [input], []

        if input.owner is None or input.owner.op not in [self.main, self.inverse, self.reciprocal]:
            if input.owner and isinstance(input.owner.op, T.DimShuffle):
                # If input is a DimShuffle of some input which does something like this:
                # * change a vector of length N into a 1xN row matrix
                # * change a scalar into a 1x1x1 tensor
                # * in general, complete the shape of a tensor with broadcastable 1s to the *left*
                # Then we will simply discard the DimShuffle and return the num/denum of its input
                dsn = input.owner    # dimshuffle node
                dsop = dsn.op        # dimshuffle op
                dsi0 = dsn.inputs[0] # the first input of the dimshuffle i.e. the ndarray to redim

                # The compatible order is a DimShuffle "new_order" of the form:
                # ('x', ..., 'x', 0, 1, 2, ..., dimshuffle_input.type.ndim)

                # That kind of DimShuffle only adds broadcastable
                # dimensions on the left, without discarding any
                # existing broadcastable dimension and is inserted
                # automatically by Elemwise when the inputs have
                # different numbers of dimensions (hence why we can
                # discard its information - we know we can retrieve it
                # later on).
                compatible_order = ('x',) * (input.type.ndim - dsi0.type.ndim) + tuple(range(dsi0.type.ndim))
                if dsop.new_order == compatible_order:
                    # If the "new_order" is the one we recognize,
                    # we return the num_denum of the dimshuffled input.
                    return self.get_num_denum(input.owner.inputs[0])
                else:
                    # This is when the input isn't produced by main, inverse or reciprocal.
                    return [input], []
            else:
                return [input], []
        num = []
        denum = []
        parent = input.owner

        # We get the (num, denum) pairs for each input
        #pairs = [self.get_num_denum(input2) if input2.type.dtype == input.type.dtype else ([input2], []) for input2 in parent.inputs]
        pairs = [self.get_num_denum(input2) for input2 in parent.inputs]

        if parent.op == self.main:
            # If we have main(x, y), numx, denumx, numy and denumy
            # then num is concat(numx, numy) and denum is concat(denumx, denumy)
            # note that main() can have any number of arguments >= 0
            # concat is list concatenation
            num = reduce(list.__iadd__, map(operator.itemgetter(0), pairs))
            denum = reduce(list.__iadd__, map(operator.itemgetter(1), pairs))
        elif parent.op == self.inverse:
            # If we have inverse(x, y), numx, denumx, numy and denumy
            # then num is concat(numx, denumy) and denum is concat(denumx, numy)
            # note that inverse() is binary
            num = pairs[0][0] + pairs[1][1]
            denum = pairs[0][1] + pairs[1][0]
        elif parent.op == self.reciprocal:
            # If we have reciprocal(x), numx, denumx
            # then num is denumx and denum is numx
            # note that reciprocal() is unary
            num = pairs[0][1]
            denum = pairs[0][0]
        return num, denum

    def merge_num_denum(self, num, denum):
        """
        Utility function which takes two lists, num and denum, and
        returns something which is equivalent to inverse(main(*num),
        main(*denum)), but depends on the length of num and the length
        of denum (in order to minimize the number of operations).

        Let n = len(num) and d = len(denum):

        n=0, d=0: neutral element (given by self.calculate([], []))
                  (for example, this would be 0 if main is addition
                  and 1 if main is multiplication)
        n=1, d=0: num[0]
        n=0, d=1: reciprocal(denum[0])
        n=1, d=1: inverse(num[0], denum[0])
        n=0, d>1: reciprocal(main(*denum))
        n>1, d=0: main(*num)
        n=1, d>1: inverse(num[0], main(*denum))
        n>1, d=1: inverse(main(*num), denum[0])
        n>1, d>1: inverse(main(*num), main(*denum))

        Given the values of n and d to which they are associated, all
        of the above are equivalent to:
        inverse(main(*num), main(*denum))
        """

        ln, ld = len(num), len(denum)
        if not ln and not ld:
            return T.as_tensor_variable(self.calculate([], []))
        if not ln:
            if self.use_reciprocal:
                return self.reciprocal(self.merge_num_denum(denum, []))
            else:
                ln = [self.calculate([], [], aslist = False)]
        if not ld:
            if ln == 1:
                # num[0] should always be a variable
                assert isinstance(num[0], gof.Variable)
                return num[0]
            else:
                return self.main(*num)
        return self.inverse(self.merge_num_denum(num, []),
                            self.merge_num_denum(denum, []))

    @staticmethod
    def get_constant(v):
        """
        Returns a numeric constant if v is a Constant or, well, a
        numeric constant. If v is a plain Variable, returns None.
        """
        if isinstance(v, Variable):
            try:
                return get_constant_value(v)
            except TypeError:
                return None
        else:
            return v

    def simplify(self, num, denum):
        """
        Shorthand for: self.simplify_constants(*self.simplify_factors(num, denum))
        """
        rval = self.simplify_constants(*self.simplify_factors(num, denum))
        for reason, simplifier in self.external_simplifiers:
            # TODO: document that 'reason' is associated with this simplification
            #       to help auditing when things go wrong
            rval = simplifier(*rval)
        return rval

    def simplify_factors(self, num, denum):
        """
        For any Variable r which is both in num and denum, removes it
        from both lists. Modifies the lists inplace. Returns the
        modified lists. For example:

        [x], [x] -> [], []
        [x, y], [x] -> [y], []
        [a, b], [c, d] -> [a, b], [c, d]
        """
        for v in list(num):
            if v in denum:
                num.remove(v)
                denum.remove(v)
        return num, denum

    def simplify_constants(self, orig_num, orig_denum):
        """

        Finds all constants in orig_num and orig_denum (using
        get_constant) and puts them together into a single
        constant. The constant is inserted as the first element of the
        numerator. If the constant is the neutral element, it is
        removed from the numerator. Examples:

        Let main be multiplication:

        [2, 3, x], [] -> [6, x], []
        [x, y, 2], [4, z] -> [0.5, x, y], [z]
        [x, 2, y], [z, 2] -> [x, y], [z]
        """

        # Lists representing the numerator and denumerator
        num, denum = list(orig_num), list(orig_denum)
        out_type = self.merge_num_denum(orig_num, orig_denum).type

        # Lists representing the *constant* elements of num and denum
        numct, denumct = [], []
        
        for v in orig_num:
            ct = self.get_constant(v)
            if ct is not None:
                # We found a constant in the numerator!
                # We remove it from num
                num.remove(v)
                # We add it to numct
                numct.append(ct)
        for v in orig_denum:
            ct = self.get_constant(v)
            if ct is not None:
                denum.remove(v)
                denumct.append(ct)

        if self.use_reciprocal or num:
            # This will calculate either:
            # [inverse(main(*numct), main(*denumct))]
            # [] - if inverse(main(*numct), main(*denumct)) is the neutral element
            ct = self.calculate(numct, denumct, aslist = True, out_type=out_type)
        else:
            # This happens if we don't allow the reciprocal and the
            # numerator is empty. That means we will need to represent
            # reciprocal(x) like inverse(neutral_element, x) so
            # we can't allow ct == []
            # TODO: why is this branch needed when merge_num_denum does it for us?
            ct = [self.calculate(numct, denumct, aslist = False, out_type=out_type)]

        # Wrapping ct in a Constant with the right dtype
        ct = [T.constant(c, dtype=out_type.dtype) for c in ct]

        if orig_num and len(numct) == 1 and len(denumct) == 0 and ct and\
                N.all([c.data for c in ct] == self.get_constant(orig_num[0])):
            # this is an important trick :( if it so happens that:
            # * there's exactly one constant on the numerator and none on the denominator
            # * it's not the neutral element (ct is an empty list in that case)
            # * the constant is the same as the first argument in the numerator
            # Then we return very exactly the original num/denum
            # If we don't do that the optimizer will just loop infinitely because
            # it will not catch on that there are no changes to be made and everytime
            # it will want to replace something by the same thing...
            return orig_num, orig_denum
        return ct + num, denum

    def transform(self, node):
        op = node.op
        if op not in [self.main, self.inverse, self.reciprocal]:
            return False

        inputs = node.inputs
        out = node.outputs[0]
        assert len(node.outputs) == 1

        # check if any of the clients of this node would be part of this canonized graph...
        # if so, we do nothing and wait for them to be transformed.
        def _bypass_dimshuffle(n):
            if isinstance(n.op, DimShuffle) and len(n.outputs[0].clients) <= 1:
                return _bypass_dimshuffle(n.outputs[0].clients.__iter__().next()[0])
            else:
                return n
        for c,c_idx in out.clients:
            if c=='output': continue
            if _bypass_dimshuffle(c).op in [self.main, self.inverse, self.reciprocal]:
                return False

        # Here we make the canonical version of the graph around this node
        # See the documentation of get_num_denum and simplify
        orig_num, orig_denum = self.get_num_denum(node.outputs[0])
        num, denum = self.simplify(list(orig_num), list(orig_denum))


        def same(x, y):
            return len(x) == len(y) and all(N.all(xe == ye) for xe, ye in zip(x, y))

        if same(orig_num, num) and same(orig_denum, denum):
            # We return False if there are no changes
            return False

        new = self.merge_num_denum(num, denum)
        if new.type.dtype != out.type.dtype:
            #new = T.fill(out, new)
            elem_op = T.Elemwise(scalar.Identity(scalar.specific_out(getattr(scalar, out.type.dtype))))
            new = elem_op(new)

        assert (new.type == out.type) == (not (new.type != out.type))

        if not (new.type == out.type):
            new = _fill_chain(new, node.inputs)[0]

        if new.type == out.type:
            return [new]
        else:
            _logger.warning(' '.join(('CANONIZE FAILED: new, out = ', new, ',', out, 'types',
                new.type, ',', out.type)))
            return False

    def __str__(self):
        return getattr(self, 'name', 'Canonizer(%s, %s, %s)' % (self.main, self.inverse, self.reciprocal))


def mul_calculate(num, denum, aslist=False, out_type=None):
    if not num and not denum:
        # Smallest 1 possible.
        if aslist:
            return []
        else:
            return N.int8(1)

    # Make sure we do not accidently upcast data types.
    if out_type is None:
        out_dtype = scalar.upcast(*[v.dtype for v in (num+denum)])
    else:
        out_dtype = out_type.dtype
    one = theano._asarray(1, dtype=out_dtype)

    v = reduce(N.multiply, num, one) / reduce(N.multiply, denum, one)
    if aslist:
        if N.all(v == 1):
            return []
        else:
            return [v]
    return v

local_mul_canonizer = Canonizer(T.mul, T.true_div, T.inv, mul_calculate, False)
register_canonicalize(local_mul_canonizer, name = 'local_mul_canonizer')

@gof.local_optimizer([T.neg])
def local_neg_to_mul(node):
    if node.op == T.neg:
        return [T.mul(numpy.array(-1, dtype = node.inputs[0].dtype), 
            node.inputs[0])]
register_canonicalize(local_neg_to_mul)

@register_specialize
@gof.local_optimizer([])
def local_sum_mul_by_scalar(node):
    """sum(scalar * smth) -> scalar * sum(smth)
    """
    # TODO: if the the thing inside the Sum is a division, 
    # we should get at the numerator....
    if isinstance(node.op, T.Sum):
        thing_summed, = node.inputs
        if thing_summed.owner and thing_summed.owner.op == T.mul:
            terms = thing_summed.owner.inputs
            scalars = [t.dimshuffle() for t in terms if numpy.all(t.type.broadcastable)]
            non_scalars = [t for t in terms if not numpy.all(t.broadcastable)]
            if scalars:
                if len(scalars) > 1:
                    if len(non_scalars) > 1:
                        return [T.mul(T.mul(*scalars), node.op(T.mul(*non_scalars)))]
                    elif len(non_scalars) == 1:
                        return [T.mul(T.mul(*scalars), node.op(non_scalars[0]))]
                    else:
                        return [T.mul(*scalars)]
                else:
                    if len(non_scalars) > 1:
                        return [T.mul(scalars[0], node.op(T.mul(*non_scalars)))]
                    elif len(non_scalars) == 1:
                        return [T.mul(scalars[0], node.op(non_scalars[0]))]
                    else:
                        return [scalars[0]]
        if thing_summed.owner and thing_summed.owner.op == T.neg:
            return [T.neg(node.op(thing_summed.owner.inputs[0]))]

@register_canonicalize
@gof.local_optimizer([])
def local_sum_div_dimshuffle(node):
    '''sum(a / dimshuffle{...}(b), axis=l) -> sum(a, axis=l) / b,
    if dimension l of the DimShuffle is 'x'.'''
    # TODO: extend it to product, and quotient of products

    if isinstance(node.op, T.Sum):
        axis = node.op.axis
        #print 'axis =', axis
        thing_summed = node.inputs[0]
        dimshuffled = None
        if thing_summed.owner and thing_summed.owner.op == T.true_div:
            numerator, denominator = thing_summed.owner.inputs
            if isinstance(numerator.owner.op, T.DimShuffle):
                new_order = numerator.owner.op.new_order
                #print 'new_order =', new_order
                # check compatibility
                compatible_dims = True
                for ax in axis:
                    if len(new_order) <= ax or new_order[ax] != 'x':
                        compatible_dims = False
                        break
                if compatible_dims:
                    #print 'getting num out'
                    new_num = numerator.owner.inputs[0]
                    return [T.true_div(new_num, node.op(denominator))]
                #else:
                #    print 'incompatible dims:', axis, new_order

            if isinstance(denominator.owner.op, T.DimShuffle):
                new_order = denominator.owner.op.new_order
                #print 'new_order =', new_order
                # check compatibility
                compatible_dims = True
                for ax in axis:
                    #print 'ax =', ax
                    #print 'len(new_order) =', len(new_order)
                    #print 'new_order[ax] =', new_order[ax]
                    if len(new_order) <= ax or new_order[ax] != 'x':
                        compatible_dims = False
                        break
                if compatible_dims:
                    #print 'getting denom out'
                    new_denom = denominator.owner.inputs[0]
                    return [T.true_div(node.op(numerator), new_denom)]
                #else:
                #    print 'incompatible dims:', axis, new_order


@register_canonicalize
@gof.local_optimizer([])
def local_sum_all_to_none(node):
    """Sum{0,1,...N} -> Sum{}"""
    if isinstance(node.op, T.Sum):
        # if all the axes are named, then use None as a shorthand
        # this permits more merging
        if node.op.axis is None:
            return
        if set(node.op.axis) == set(range(node.inputs[0].type.ndim)):
            return [T.Sum(axis=None)(node.inputs[0])]

@register_canonicalize
@gof.local_optimizer([])
def local_sum_sum(node):
    """Sum(Sum()) -> Sum"""
    if isinstance(node.op, T.Sum):
        summed, = node.inputs
        if len(summed.clients) == 1:
            if summed.owner and isinstance(summed.owner.op, T.Sum):
                if summed.owner.op.axis is None:
                    # special case of local_cut_useless_reduce
                    return [T.Sum(None)(summed.owner.inputs[0])]
                if node.op.axis is None:
                    # we're summing up everything anyway so lets 
                    # do it all at once
                    return [T.Sum(None)(summed.owner.inputs[0])]

                # figure out which dimensions of the original input are preserved
                alldims = range(summed.owner.inputs[0].type.ndim)
                
                # trim out the dimensions that were removed by the first sum
                alldims = [d for i,d in enumerate(alldims) if i in summed.owner.op.axis]

                # trim out the dimensions removed by second sum
                alldims = [d for i,d in enumerate(alldims) if i in node.op.axis]

                # figure out an axis argument that combines the effect of both
                newaxis = [i for i in xrange(summed.owner.inputs[0].type.ndim)
                        if i not in alldims]

                combined_sum = T.Sum(newaxis)
                return [combined_sum(summed.owner.inputs[0])]

@register_canonicalize
@gof.local_optimizer([])
def local_cut_useless_reduce(node):
    """Sum(a, axis=[]) -> a  """
    if isinstance(node.op, T.CAReduce):
        summed, = node.inputs
        # if reduce were doing anything, the output ndim would be reduced
        if summed.type == node.outputs[0].type:
            return [summed]

@gof.local_optimizer([T.mul])
def local_mul_to_neg(node):
    if node.op == T.mul and N.all(local_mul_canonizer.get_constant(node.inputs[0]) == -1.0):
        return [-local_mul_canonizer.merge_num_denum(node.inputs[1:], [])]
    else:
        return False
register_specialize(local_mul_to_neg)

@register_specialize
@gof.local_optimizer([T.neg])
def local_neg_neg(node):
    # other specializations shouldn't put this in, 
    # but sometimes they do
    if node.op == T.neg:
        if node.inputs[0].owner and node.inputs[0].owner.op == T.neg:
            return [node.inputs[0].owner.inputs[0]]

@register_specialize
@gof.local_optimizer([T.neg])
def local_neg_div_neg(node):
    """- (-a / b) -> a / b

    Also performs - (c / b) -> ((-c) / b) when c is a scalar constant.
    """
    if node.op == T.neg:
        if node.inputs[0].owner and node.inputs[0].owner.op == T.true_div:
            frac = node.inputs[0]
            num, denom = frac.owner.inputs
            if num.owner and num.owner.op == T.neg:
                if len(frac.clients) == 1:
                    # No other clients of the original division
                    new_num = num.owner.inputs[0]
                    return [T.true_div(new_num, denom)]
            elif numpy.all(num.broadcastable) and isinstance(num, Constant):
                if len(frac.clients) == 1:
                    new_num = -num.data
                    return [T.true_div(new_num, denom)]


@gof.local_optimizer([T.mul])
def local_mul_zero(node):
    """As part of canonicalization, we replace multiplication by zero with zero.
    """
    if node.op == T.mul:
        otype = node.outputs[0].type

        for i in node.inputs:
            try:
                value = get_constant_value(i)
            except TypeError:
                continue
            #print 'MUL by value', value, node.inputs
            if N.all(value == 0):
                #print '... returning zeros'
                return _fill_chain(theano._asarray(0, dtype=otype.dtype), node.inputs)
register_canonicalize(local_mul_zero)

@gof.local_optimizer([T.true_div])
def local_div_to_inv(node):
    if node.op == T.true_div and N.all(local_mul_canonizer.get_constant(node.inputs[0]) == 1.0):
        return [T.inv(local_mul_canonizer.merge_num_denum(node.inputs[1:], []))]
    else:
        return False
register_specialize(local_div_to_inv)

@gof.local_optimizer([T.inv])
def local_inv_canon(node):
    if node.op == T.inv:
        return [T.pow(node.inputs[0], -1.0)]
    else:
        return False
register_canonicalize(local_inv_canon)

@gof.local_optimizer([T.pow])
def local_pow_canonicalize(node):
    if node.op == T.pow:
        if N.all(local_mul_canonizer.get_constant(node.inputs[1]) == 0):
            return [broadcast_like(1, node.outputs[0], node.env)]
        if N.all(local_mul_canonizer.get_constant(node.inputs[1]) == 1):
            return [broadcast_like(node.inputs[0], node.outputs[0], node.env)]
    else:
        return False
register_canonicalize(local_pow_canonicalize)

@register_specialize
@gof.local_optimizer([T.mul])
def local_mul_to_sqr(node):
    """x*x -> sqr(x)

    This is faster on the GPU when memory fetching is a big part of the computation time.
    """
    if node.op == T.mul:
        if len(node.inputs)==2:
            if node.inputs[0] is node.inputs[1]:
                return [T.sqr(node.inputs[0])]

@gof.local_optimizer([T.pow])
def local_pow_specialize(node):
    #here, we are past the point of canonicalization, so we don't want to put in un-necessary fills.
    if node.op == T.pow:
        #the idea here is that we have pow(x, y)
        odtype = node.outputs[0].dtype
        xsym = node.inputs[0]
        ysym = node.inputs[1]
        y = local_mul_canonizer.get_constant(ysym)
        if (y is not None) \
                and encompasses_broadcastable(xsym.type.broadcastable, ysym.type.broadcastable):
            rval = None

            if N.all(y == 2):
                rval = [T.sqr(xsym)]
            if N.all(y == 1):
                rval = [xsym]
            if N.all(y == 0):
                rval = [T.fill(xsym, numpy.asarray(1, dtype=odtype))]
            if N.all(y == 0.5):
                rval = [T.sqrt(xsym)]
            if N.all(y == -0.5):
                rval = [T.inv(T.sqrt(xsym))]
            if N.all(y == -1):
                rval = [T.inv(xsym)]
            if N.all(y == -2):
                rval = [T.inv(T.sqr(xsym))]

            # Optimize all integral powers in [-RANGE, RANGE]
            if config.experimental.pow and rval is None and abs(y)==int(abs(y)) and abs(y) <= 512:# 512 is too small for the cpu and too big for some gpu!
                pow2 = [xsym]
                pow2_scal = [theano.scalar.Scalar(xsym.dtype)()]
                y_to_do = abs(y)
                for i in range(int(numpy.log2(y_to_do))):
                    pow2.append(T.sqr(pow2[i]))
                    pow2_scal.append(theano.scalar.sqr(pow2_scal[i]))
                rval1 = None
                rval1_scal = None
                while y_to_do>0:
                    log_to_do = int(numpy.log2(y_to_do))                    
                    if rval1:
                        rval1 *= pow2[log_to_do]
                        rval1_scal *= pow2_scal[log_to_do]
                    else: 
                        rval1 = pow2[log_to_do]
                        rval1_scal = pow2_scal[log_to_do]
                    y_to_do -= 2**log_to_do

                if abs(y)>2:
                    #We fuse all the pow together here to make compilation faster
                    rval1 = Elemwise(theano.scalar.Composite([pow2_scal[0]],[rval1_scal])).make_node(xsym)
                if y<0:
                    rval = [T.inv(rval1)]
                else:
                    rval = [rval1]
            if rval:
                rval[0] = T.cast(rval[0], odtype)
                assert rval[0].type == node.outputs[0].type, (rval, node.outputs)
                return rval
    else:
        return False
register_specialize(local_pow_specialize)
theano.configparser.AddConfigVar('experimental.pow',
        "Transform a pow to a constant integer to a graph of mul. Fast on cpu, but more work needed for gpu.",
        theano.configparser.BoolParam(False),
        )

@gof.local_optimizer([T.mul])
def local_mul_specialize(node):
    """Remove special-case constants from mul arguments
    """
    # here, we are past the point of canonicalization, so we don't want to put in un-necessary fills.
    #
    # at this point [post canonicalize], mul() may have many inputs.
    if node.op == T.mul:
        #the idea here is that we have pow(x, y)
        neg = False
        new_inputs = []
        for input in node.inputs:
            # remove any neg arguments 
            while input.owner and input.owner.op == T.neg:
                neg ^= True
                input = input.owner.inputs[0]

            # remove special case arguments of 1, -1 or 0
            y = local_mul_canonizer.get_constant(input)
            if N.all(y == 1.0):
                continue
            elif N.all(y == -1.0):
                neg ^= True #toggles
            elif N.all(y == 0.0):
                # if we find any zero, we just return right away
                return [T.alloc(numpy.asarray(0, dtype=node.outputs[0].dtype),
                        *node.env.shape_feature.shape_of[node.outputs[0]])]
            else:
                new_inputs.append(input)

        if new_inputs != node.inputs:
            if new_inputs:
                if len(new_inputs) == 1:
                    if neg:
                        rval = -new_inputs[0]
                    else:
                        rval = new_inputs[0]
                else:
                    if neg:
                        rval = -T.mul(*new_inputs)
                    else:
                        rval = T.mul(*new_inputs)

                return [T.alloc(T.cast(rval, node.outputs[0].dtype),
                        *node.env.shape_feature.shape_of[node.outputs[0]])]
            else:
                # there are no variable inputs to mul
                # N.B. this could have been constant-folded...
                if neg:
                    # return output's worth of -1
                    return [T.alloc(
                        numpy.asarray(-1, dtype=node.outputs[0].dtype),
                            *node.env.shape_feature.shape_of[node.outputs[0]])]
                else:
                    # return output's worth of 1
                    return [T.alloc(
                        numpy.asarray(1, dtype=node.outputs[0].dtype),
                            *node.env.shape_feature.shape_of[node.outputs[0]])]

register_specialize(local_mul_specialize)

@gof.local_optimizer([T.add])
def local_add_specialize(node):
    def fill_chain(v):
        return _fill_chain(v, node.inputs)

    #here, we are past the point of canonicalization, so we don't want to put in un-necessary fills.
    if node.op == T.add:
        new_inputs = []
        for input in node.inputs:
            try:
                y = get_constant_value(input)
            except TypeError:
                y = input
            if N.all(y == 0.0):
                continue
            new_inputs.append(input)

        if len(new_inputs) < len(node.inputs):
            if len(new_inputs) == 0:
                #we got rid of the entire expression!
                return fill_chain(T.TensorConstant(T.TensorType(dtype=node.outputs[0].type.dtype,
                    broadcastable = [True] * node.outputs[0].ndim), N.asarray(0)))

            if len(new_inputs) == 1:
                return fill_chain(new_inputs[0])
            else:
                return fill_chain(T.add(*new_inputs))
    else:
        return False
register_specialize(local_add_specialize)

# neg_to_mul = out2in(gof.LocalOptGroup(local_neg_to_mul))
# mul_to_neg = out2in(gof.LocalOptGroup(local_mul_to_neg))

mul_canonizer = in2out(gof.LocalOptGroup(local_mul_canonizer, local_fill_cut, local_fill_sink))

@register_stabilize
@gof.local_optimizer([T.log])
def local_log1p(node):
    # log(1+exp(x)) -> log1p(x)
    if node.op == T.log:
        log_arg, = node.inputs
        if log_arg.owner and log_arg.owner.op == T.add:
            scalars, scalar_inputs, nonconsts = \
                    scalarconsts_rest(log_arg.owner.inputs)
            # scalar_inputs are potentially dimshuffled and fill'd scalars
            if scalars and numpy.allclose(numpy.sum(scalars), 1):
                if not nonconsts:
                    pass # leave for constant-merge
                if len(nonconsts)==1:
                    return _fill_chain(T.log1p(nonconsts[0]), scalar_inputs)
                else:
                    return _fill_chain(T.log1p(T.add(*nonconsts)), scalar_inputs)

#TODO: in canonicalize, change log10 and log2 -> log
@register_stabilize
@gof.local_optimizer([T.log])
def local_log_add(node):
    # log(exp(x)+exp(y))
    #
    # Suppose x >= y
    # log(exp(x) + exp(y))
    # log(exp(x) * (1 + exp(y)/exp(x)))
    # x + log(1 + exp(y)/exp(x))
    # x + log1p(exp(y)/exp(x))
    # x + log1p(exp(y-x))
    if node.op == T.log:
        z = node.inputs[0]
        if z.owner and z.owner.op == T.add:
            zi = z.owner.inputs
            pre_exp = [ x.owner.inputs[0] for x in zi if x.owner and x.owner.op == T.exp]
            if len(pre_exp) == len(zi):
                # all arguments to add are exp(<something>)
                max_pre = T.maximum(*pre_exp)
                return [max_pre + T.log1p(T.exp(T.add(*[p - max_pre for p in pre_exp])))]

def add_calculate(num, denum, aslist = False, out_type=None):
    #TODO: make sure that this function and mul_calculate are similar
    if out_type is None:
      zero = 0.0
    else:
      zero = theano._asarray(0, dtype=out_type.dtype)
    #zero = 0.0 if out_type is None else theano._asarray(0, dtype=out_type.dtype)
    v = reduce(N.add, num, zero) - reduce(N.add, denum, zero)
    if aslist:
        if N.all(v == 0):
            return []
        else:
            return [v]
    return v

local_add_canonizer = Canonizer(T.add, T.sub, T.neg, add_calculate)
add_canonizer = in2out(gof.LocalOptGroup(local_add_canonizer, local_fill_cut, local_fill_sink))

register_canonicalize(local_add_canonizer, name = 'local_add_canonizer')


##################
# Distributivity #
##################


def distribute_greedy(pos_pairs, neg_pairs, num, denum, minscore = 0):
    # each pair in pos_pairs and neg_pairs is a num/denum pair. this
    # function attempts to add num and denum to the corresponding parts
    # of each pair, and counts how many multiplications/divisions can
    # be saved in that way.

    # each division is counted like div_cost multiplications
    # (typically, division costs more so we are willing to multiply more
    # in order to divide less)
    # 1.5 was obtained through an informal test and may very well be
    # platform dependent
    div_cost = 1.5

    score = len(num) + div_cost * len(denum) # score is number of operations saved, higher is better
    new_pos_pairs = list(itertools.starmap(local_mul_canonizer.simplify,
                                           [(n+num, d+denum) for (n, d) in pos_pairs]))
    new_neg_pairs = list(itertools.starmap(local_mul_canonizer.simplify,
                                           [(n+num, d+denum) for (n, d) in neg_pairs]))
    for (n, d), (nn, dd) in zip(pos_pairs + neg_pairs, new_pos_pairs + new_neg_pairs):
        # We calculate how many operations we are saving with the new num and denum
        score += len(n) + div_cost * len(d) - len(nn) - div_cost * len(dd)
    if score <= minscore:
        # the change is not applied because it adds too many operations
        return False, pos_pairs, neg_pairs
    return True, new_pos_pairs, new_neg_pairs

def attempt_distribution(factor, num, denum):
    # we try to insert each num and each denum in the factor
    # returns: changes?, new_factor, new_num, new_denum
    # if there are changes, new_num and new_denum contain all the numerators
    # and denumerators that could not be distributed in the factor
    pos, neg = local_add_canonizer.get_num_denum(factor)
    if len(pos) == 1 and not neg:
        return False, factor, num, denum
    pos_pairs = map(local_mul_canonizer.get_num_denum, pos)
    neg_pairs = map(local_mul_canonizer.get_num_denum, neg)
    change = False
    for n in list(num):
        success, pos_pairs, neg_pairs = distribute_greedy(pos_pairs, neg_pairs, [n], [])
        if success:
            change = True
            num.remove(n)
    for d in list(denum):
        success, pos_pairs, neg_pairs = distribute_greedy(pos_pairs, neg_pairs, [], [d])
        if success:
            change = True
            denum.remove(d)
    if not change:
        return change, factor, num, denum
    else:
        return change, local_add_canonizer.merge_num_denum(
            list(itertools.starmap(local_mul_canonizer.merge_num_denum, pos_pairs)),
            list(itertools.starmap(local_mul_canonizer.merge_num_denum, neg_pairs))), num, denum

@gof.local_optimizer([T.mul, T.add, T.mul], [T.mul, T.sub, T.mul],
                     [T.mul, T.add, T.true_div], [T.mul, T.sub, T.true_div])
def local_greedy_distributor(node):
    """
    This optimization tries to apply distributivity of multiplication
    to addition in order to reduce the number of multiplications
    and/or divisions that must be done. The algorithm weighs division
    more than multiplication to account for the former's slightly
    greater computational cost.

    The following expressions are simplified:
    1. ((a/x + b/y) * x * y) --> a*y + b*x
    2. ((a/x + b) * x) --> a + b*x

    The following expressions are not simplified:
    3. ((a + b) * x) -/-> a*x + b*x

    This optimization aims to reduce computational cost. It may also
    increase numerical stability, e.g. when x and/or y tend to 0 in
    example 1.
    """

    out = node.outputs[0]
    num, denum = local_mul_canonizer.get_num_denum(out)
    if len(num) == 1 and not denum:
        return False

    new_num, new_denum = [], []

    change = False

    for candidate in list(num):
        if candidate not in num:
            continue
        num.remove(candidate)
        _change, candidate, num, denum = attempt_distribution(candidate, num, denum)
        change |= _change
        new_num.append(candidate)

    for candidate in list(denum):
        if candidate not in denum:
            continue
        denum.remove(candidate)
        _change, candidate, denum, num = attempt_distribution(candidate, denum, num)
        change |= _change
        new_denum.append(candidate)

    if not change:
        return False

    new_num += num
    new_denum += denum

    rval = local_mul_canonizer.merge_num_denum(new_num, new_denum)

    if not (rval.type == out.type):
        #WHY DOES THIS HAPPEN?
        return False

    return [rval]

register_canonicalize(local_greedy_distributor)
register_stabilize(local_greedy_distributor)



@gof.local_optimizer([None])
def constant_folding(node):
    for input in node.inputs:
        if not isinstance(input, Constant):
            return False
    try:
        storage = [[None] for output in node.outputs]
        node.op.perform(node, [x.data for x in node.inputs], storage)
    except MethodNotDefined:
        tmp_inputs = [x.type() for x in node.inputs]
        f = compile.function(
                inputs=tmp_inputs,
                outputs=node.op.make_node(*tmp_inputs).outputs,
                mode=compile.Mode(linker='c|py',optimizer=None))
        xvals = f(*[x.data for x in node.inputs])
        storage = [[xv] for xv in xvals]

    msg = []
    assert len(storage) == len(node.outputs)
    for s, output in zip(storage, node.outputs):
        try:
            constant = output.type.Constant
        except:
            constant = Constant
        msg += [constant(output.type, s[0])]
    return msg

register_canonicalize(constant_folding, 'fast_compile')
register_stabilize(constant_folding) # because 
register_specialize(constant_folding)


inplace_matrix_transpose = T.DimShuffle([False,False], [1,0], inplace=True)
local_transposed_dot = gof.PatternSub((inplace_matrix_transpose, (T.dot, 'x', 'y')),
        (T.dot, (inplace_matrix_transpose, 'y'), (inplace_matrix_transpose, 'x')))
register_canonicalize(local_transposed_dot, name='local_transposed_dot')

# ###############
# # Loop fusion #
# ###############
def local_elemwise_fusion_op(OP):
    """
    We parametrise it to make it work for Elemwise and GpuElemwise op.
    """
    def local_fuse(node):
        """
        As part of specialisation, we fusion two consecutif elemwise op of the same shape.

        For mixed dtype, we let the Compise op do the cast. It let the C compile do the cast.
        The number of dimension is validated at call time by theano itself.

        """
        # META TODO:  PUT THESE THINGS IN TRAC, NOT TODO NOTES!!
        # TODO: use broadcast flag?

        # TODO: don't do this optimization as a localOptimizer.  Analyze the graph in terms of
        # elemwise subgraphs, and then replace each subgraph with a Composite version.

        # TODO: use malloc and copy to transfer arguments that don't fit within the parameter space
        # of 256 bytes
        #
        # TODO: Merge with multiple output to merge when an inputs have multiple clients. This can't be done with a local optimiser.
        # TODO: Related: Support composites with multiple outputs

        # TODO: Use Composite to combine Elemwise and Reduce operations.  We have to loop over the
        # data anyway... might as well sum it up while we're at it (this can be trickier than i'm
        # making it seound here. The data-traversal should be done contiguously, and the summing-up
        # might not be easy or worthwhile if the summation axis doesn't line up with a contiguous
        # dimension)

        if not isinstance(node.op, OP):
            return False
        nb_elemwise=0
        inputs=[]#inputs of the new Elemwise op.
        s_inputs = []#inputs of the new scalar op.
        s_g=[]#graph of scalar, what will by done in the inner loop.
        for i in node.inputs:
            do_fusion = False
            catch = False
            if i.owner and isinstance(i.owner.op, OP) and len(i.clients)<=1:
                #if the scalar_op don't have a c implementation, we skip its fusion to allow the fusion of the other ops.
                do_fusion=True
                try:
                    s_input = [scalar.Scalar(x.dtype).make_variable() for x in i.owner.inputs]
                    s_op=i.owner.op.scalar_op(*s_input)
                    i.owner.op.scalar_op.c_code(s_op.owner,"test_presence_of_c_code",
                                                ["x" for x in i.owner.inputs],
                                                "z",{})
                except MethodNotDefined:
                    catch = True
                except NotImplementedError:
                    catch = True
                if catch:
                    _logger.info("%s does not implement the c_code function. As well as being potentially slow, this disables loop fusion of this op." % str(i.owner.op.scalar_op))
                    do_fusion=False

            if do_fusion:
                nb_elemwise+=1
                inputs.extend(i.owner.inputs)
                s_inputs.extend(s_input)
                s_g.append(s_op)
            else:
                inputs.append(i)
                s=scalar.Scalar(i.dtype).make_variable()
                s_inputs.append(s)
                s_g.append(s)

        #if no inputs have are an elemwise, there is nothing to fuse.
        if nb_elemwise==0:
    #        print "local_elemwise_fusion: no elemwise in inputs. Nothing to fuse."
            return False

        otype = node.outputs[0].type
        s_new_out=node.op.scalar_op(*s_g)
        try:
            s_new_out.owner.op.c_code(s_new_out.owner, "test_presence_of_c_code",
                             ["x" for x in s_g],
                             "z",{}) 
        except MethodNotDefined:
            _logger.info("%s does not implement the c_code function. As well as being potentially slow, this disables loop fusion of this op." % str(s_new_out.owner.op))
            return False
        except NotImplementedError:
            _logger.info("%s does not implement the c_code function. As well as being potentially slow, this disables loop fusion of this op." % str(s_new_out.owner.op))
            return False

        #create the composite op.
        C = scalar.Composite(s_inputs,[s_new_out])

        #create the new node.
        n=OP(C).make_node(*inputs)
        assert len(n.outputs)==1
        assert node.outputs[0].dtype==n.outputs[0].dtype

        # There is a hard limit of 256 bytes for the formal argument list to a GPU kernel function.
        # Here, we estimate how many bytes the new Op will need, and abort if it needs too much.
        if True:
            argument_limit = 240  # 16 bytes are used for block and thread coords etc.
            #TODO: read in from architecture to make this 4 or 8
            int_size = 8
            ptr_size = 8
            argument_size = int_size #for numels
            argument_size += int_size *  inputs[0].type.ndim # for the shape
            argument_size += sum((ptr_size + int_size * i.type.ndim) for i in n.inputs)
            argument_size += sum((ptr_size + int_size * i.type.ndim) for i in n.outputs)
            if argument_size >= argument_limit:
                _logger.info('loop fusion failed because Op would exceed kernel argument limit.')
                return False

    #    print "local_elemwise_fusion: FUSED",nb_elemwise+1,"elemwise!"
        return n.outputs
    return local_fuse

local_elemwise_fusion = local_elemwise_fusion_op(T.Elemwise)

class FusionOptimizer(Optimizer):
    """Graph optimizer for Fusion of elemwise operations"""
    def __init__(self, local_optimizer):
        Optimizer.__init__(self)
        self.optimizer = local_optimizer

    def add_requirements(self, env):
        env.extend(toolbox.ReplaceValidate())
        env.extend(DestroyHandler())

    def apply(self, env):
        did_something = True
        while did_something:
            nodelist = list(env.toposort())
            did_something = False
            for node in nodelist:
                new_outputs = self.optimizer(node)
                if new_outputs:
                    assert len(new_outputs) == len(node.outputs)
                    try:
                        env.replace_all_validate(
                                zip(node.outputs, new_outputs),
                                reason = self.__class__.__name__)
                        did_something = True
                        break
                    except InconsistencyError, e:
                        #TODO: retry other applications of gemm (see comment in _gemm_from_node
                        pass


if config.tensor.local_elemwise_fusion:
    _logger.debug("enabling optimization fusion elemwise in fast_run")
    compile.optdb.register('elemwise_fusion', FusionOptimizer(local_elemwise_fusion), 71.00, 'fast_run', 'fusion', 'local_elemwise_fusion')
else:
    _logger.debug("not enabling optimization fusion elemwise in fast_run")
    compile.optdb.register('elemwise_fusion', FusionOptimizer(local_elemwise_fusion), 71.00, 'fusion', 'local_elemwise_fusion')



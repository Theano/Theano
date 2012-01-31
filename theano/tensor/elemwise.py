from copy import copy

import numpy

import elemwise_cgen as cgen
import theano
from theano import gof
from theano.gof import Apply, Op
from theano import scalar
from theano.scalar import Scalar
from theano.printing import min_informative_str, pprint
from theano.gof.python25 import all, any
config = theano.config



# tensor depends on elemwise to provide definitions for several ops
# but elemwise needs to make TensorType instances, so we have these as
# placeholders and the tensor module fills them
def as_tensor_variable(data):
    raise Exception("Circular dependencies prevent using this here. import tensor before elemwise")

def TensorType(*inputs, **kwargs):
    raise Exception("Circular dependencies prevent using this here. import tensor before elemwise")

def TensorVariable(*inputs, **kwargs):
    raise Exception("Circular dependencies prevent using this here. import tensor before elemwise")

def TensorConstant(*inputs, **kwargs):
    raise Exception("Circular dependencies prevent using this here. import tensor before elemwise")

# Define common subsets of dtypes (as strings).
discrete_dtypes = map(str, scalar.discrete_types)
continuous_dtypes = map(str, scalar.continuous_types)


##################
### DimShuffle ###
##################

class DimShuffle(Op):
    """
    Allows to reorder the dimensions of a tensor or insert or remove
    broadcastable dimensions.

    In the following examples, 'x' means that we insert a broadcastable
    dimension and a numerical index represents the dimension of the same
    rank in the tensor passed to perform.

    Examples:
      DimShuffle((False, False, False), ['x', 2, 'x', 0, 1])

       This op will only work on 3d tensors with no broadcastable
       dimensions.  The first dimension will be broadcastable,
       then we will have the third dimension of the input tensor as
       the second of the resulting tensor, etc. If the tensor has
       shape (20, 30, 40), the resulting tensor will have dimensions
       (1, 40, 1, 20, 30). (AxBxC tensor is mapped to 1xCx1xAxB tensor)

      DimShuffle((True, False), [1])

       This op will only work on 2d tensors with the first dimension broadcastable.
       The second dimension of the input tensor will be the first dimension of
       the resulting tensor. If the tensor has shape (1, 20), the resulting tensor
       will have shape (20, ).

    More examples:
      DimShuffle((), ['x']) -> make a 0d (scalar) into a 1d vector
      DimShuffle((False, False), [0, 1]) -> identity
      DimShuffle((False, False), [1, 0]) -> inverts the first and second dimensions
      DimShuffle((False,), ['x', 0]) -> make a row out of a 1d vector (N to 1xN)
      DimShuffle((False,), [0, 'x']) -> make a column out of a 1d vector (N to Nx1)
      DimShuffle((False, False, False), [2, 0, 1]) -> AxBxC to CxAxB
      DimShuffle((False, False), [0, 'x', 1]) -> AxB to Ax1xB
      DimShuffle((False, False), [1, 'x', 0]) -> AxB to Bx1xA

    The reordering of the dimensions can be done in numpy with the transpose function.
    Adding, subtracting dimensions can be done with reshape.
    """

    def __init__(self, input_broadcastable, new_order, inplace = False):
        """
        Usage: DimShuffle(input_broadcastable, new_order, inplace = False)

        - input_broadcastable: the expected broadcastable pattern of the
                               input
        - new_order: a list representing the relationship between the
                     input's dimensions and the output's dimensions. Each
                     element of the list can either be an index or 'x'.
        - inplace: if True, the output will be a view of the input.
                   If False, the output will be a copy of the input.

        If j = new_order[i] is an index, the output's ith dimension
          will be the input's jth dimension.
        If new_order[i] is 'x', the output's ith dimension will
          be 1 and Broadcast operations will be allowed to do broadcasting
          over that dimension.

        If input.broadcastable[i] == False then i must be found in new_order.
        Broadcastable dimensions, on the other hand, can be discarded.
        """
        input_broadcastable = tuple(input_broadcastable)
        self.input_broadcastable = input_broadcastable
        new_order = tuple(new_order)
        self.new_order = new_order
        self.inplace = inplace

        for i in xrange(len(new_order)-1):
            j = new_order[i]
            if j != 'x' and j in new_order[i+1:]:
                raise ValueError("The same input dimension may not appear twice in the list of output dimensions", (new_order))

        # list of dimensions of the input to drop
        self.drop = []
        i2j = {} # this maps i before dropping dimensions to j after dropping dimensions so self.shuffle can be set properly later on
        j = 0
        for i, b in enumerate(input_broadcastable):
            if i not in new_order:
                # we want to drop this dimension because it's not a value in new_order
                if b == 1: # 1 aka True
                    self.drop.append(i)
                else:
                    # we cannot drop non-broadcastable dimensions
                    raise ValueError("You cannot drop a non-broadcastable dimension.", (input_broadcastable, new_order))
            else:
                i2j[i] = j
                j += 1

        # transposition of non-broadcastable dimensions
        # This is how the dimensions will be permuted, without accounting for the extra
        # 'x' broadcastable dimensions to insert.
        self.shuffle = [i2j[x] for x in new_order if x != 'x']

        # list of dimensions of the output that are broadcastable and were not in the original input
        self.augment = [i for i, x in enumerate(new_order) if x == 'x']

        if self.inplace:
            self.view_map = {0: [0]}

        self._rehash()

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['_hashval']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._rehash()

    def make_node(self, _input):
        input = as_tensor_variable(_input)
        ib = tuple(input.type.broadcastable)
        if not ib == self.input_broadcastable:
            raise TypeError("The number of dimensions and/or broadcastable pattern of the input is incorrect for this op. Expected %s, got %s." % (self.input_broadcastable, ib))
        ob = []
        for value in self.new_order:
            if value == 'x':
                ob.append(True)
            else:
                ob.append(ib[value])

        output = TensorType(dtype = input.type.dtype,
                        broadcastable = ob).make_variable()

        return Apply(self, [input], [output])

    def __eq__(self, other):
        # it's probably not necessary to compare input_broadcastable
        return type(self) == type(other) \
            and self.inplace == other.inplace \
            and self.new_order == other.new_order \
            and self.input_broadcastable == other.input_broadcastable

    def _rehash(self):
        self._hashval = hash(type(self).__name__) ^ hash(type(self).__module__) ^ hash(self.inplace) \
                ^ hash(self.new_order) ^ hash(self.input_broadcastable)

    def __hash__(self):
        return self._hashval

    def __str__(self):
        if self.inplace:
            return "InplaceDimShuffle{%s}" % ",".join(str(x) for x in self.new_order)
        else:
            return "DimShuffle{%s}" % ",".join(str(x) for x in self.new_order)

    def perform(self, node, inp, out):
        input, = inp
        storage, = out
        # drop
        res = input
        if type(res) != numpy.ndarray:
            raise TypeError(res)
        shape = list(res.shape)
        for drop in reversed(self.drop):
            shape.pop(drop)
        res = res.reshape(shape)

        # transpose
        res = res.transpose(self.shuffle)

        # augment
        shape = list(res.shape)
        for augm in self.augment:
            shape.insert(augm, 1)
        res = res.reshape(shape)

        # copy (if not inplace)
        if not self.inplace:
            res = numpy.copy(res)

        storage[0] = numpy.asarray(res) #asarray puts scalars back into array

    def infer_shape(self, node, shapes):
        ishp, = shapes
        ishp = list(ishp)
        for drop in reversed(self.drop):
            del ishp[drop]
        # transpose
        rval = [ishp[i] for i in self.shuffle]

        # augment
        for augm in self.augment:
            rval.insert(augm, 1)
        return [rval]

    def R_op(self, inputs, eval_points):
        if None in eval_points:
            return [None]
        return self.make_node(*eval_points).outputs

    def c_code(self, node, name, inp, out, sub):
        input, = inp
        res, = out
        basename = input + '__view_or_copy'

        def statements(lst):
            return ';\n'.join(lst) + ';'

        nd_in = len(self.input_broadcastable)
        nd_out = len(self.new_order)

        check_input_nd = [('if (%(input)s->nd != ' + str(nd_in) + ')'
                '{PyErr_SetString(PyExc_NotImplementedError, "input nd"); %(fail)s;}')]

        clear_output = ['if (%(res)s) {Py_XDECREF(%(res)s);}']

        #get the copy / view of the input depending on whether we're doing things inplace or not.
        if self.inplace:
            get_base = ['{ PyArrayObject * %(basename)s = %(input)s', 'Py_INCREF((PyObject*)%(basename)s)']
        else:
            get_base = [('{ PyArrayObject * %(basename)s = (PyArrayObject*)PyArray_FromAny((PyObject*)%(input)s, NULL,'
                    '0, 0, NPY_ALIGNED|NPY_ENSURECOPY, NULL)')]

        shape_statements = ['npy_intp dimensions[%i]'%nd_out]
        for i, o in enumerate(self.new_order):
            if o != 'x':
                shape_statements += [('dimensions['+str(i)+'] = %(basename)s->dimensions['+str(o)+']')]
            else:
                shape_statements += [('dimensions['+str(i)+'] = 1')]
        #backport
        #shape_statements += [('dimensions['+str(i)+'] = %(basename)s->dimensions['+str(o)+']')
        #    if o != 'x' else
        #    ('dimensions['+str(i)+'] = 1')
        #    for i, o in enumerate(self.new_order)]


        strides_statements = ['npy_intp strides[%i]'%nd_out]

        #set the strides of the non-broadcasted dimensions
        for i, o in enumerate(self.new_order):
            if o != 'x':
                strides_statements += [('strides['+str(i)+'] = %(basename)s->strides['+str(o)+']')]
            else:
                strides_statements += [('strides['+str(i)+'] = 0')]
        #backport
        #strides_statements += [('strides['+str(i)+'] = %(basename)s->strides['+str(o)+']')
        #    if o != 'x' else
        #    ('strides['+str(i)+'] = 0')
        #    for i, o in enumerate(self.new_order)]

        # set the strides of the broadcasted dimensions
        # this algorithm is from numpy: PyArray_Newshape() in cvs/numpy/numpy/core/src/multiarraymodule.c
        if nd_out > 0:
            strides_statements.append(
                'if (strides[' +
                str(nd_out) +
                '-1] == 0) strides[' +
                str(nd_out) +
                '-1] = %(basename)s->descr->elsize'
            )
        for i in xrange(nd_out-2,-1, -1):
            strides_statements.append("if (strides[%(i)s] == 0) strides[%(i)s] = strides[%(i)s+1] * dimensions[%(i)s+1]"%dict(i=str(i)))

        #
        # PyObject* PyArray_New(PyTypeObject* subtype, int nd, npy_intp* dims, int type_num,
        #                       npy_intp* strides, void* data, int itemsize, int flags, PyObject* obj)
        #
        close_bracket = [
                #create a new array,
                ('%(res)s = (PyArrayObject*)PyArray_New(&PyArray_Type, '
                            '' + str(nd_out) + ', dimensions, '
                            'PyArray_TYPE(%(basename)s), strides, '
                            '%(basename)s->data, PyArray_ITEMSIZE(%(basename)s), '
                            #borrow only the writable flag from the base
                            # the NPY_OWNDATA flag will default to 0.
                            '(NPY_WRITEABLE*PyArray_ISWRITEABLE(%(basename)s)), NULL)'),
                #recalculate flags: CONTIGUOUS, FORTRAN, ALIGNED
                'PyArray_UpdateFlags(%(res)s, NPY_UPDATE_ALL)',
                #we are making a view in both inplace and non-inplace cases
                '%(res)s->base = (PyObject*)%(basename)s',
                '}']

        full_code = statements(check_input_nd
                + clear_output
                + get_base
                + shape_statements
                + strides_statements
                + close_bracket)

        if 0:
            print 'C_CODE'
            print ''
            print self
            print "IN BROAD", self.input_broadcastable
            print "NEW ORDER", self.new_order
            print "SHUFFLE", self.shuffle
            print "AUGMENT", self.augment
            print '------------'
            print ''
            print full_code

            if 0:
                import sys
                sys.exit()

        return full_code % dict(locals(), **sub)

    def c_code_cache_version(self):
        return (2,)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        gz = as_tensor_variable(gz)
        grad_order = ['x'] * len(x.type.broadcastable)
        for i, v in enumerate(self.new_order):
            if v != 'x':
                grad_order[v] = i
        # Do not make the DimShuffle inplace as an optimization at the
        # canonicalization optimization phase will remove the implace.
        # The inplace will be reintroduced automatically later in the graph.
        return [DimShuffle(gz.type.broadcastable, grad_order)(Elemwise(scalar.identity)(gz))]



class DimShufflePrinter:

    def __p(self, new_order, pstate, r):
        if new_order != () and  new_order[0] == 'x':
            return "%s" % self.__p(new_order[1:], pstate, r)
#            return "[%s]" % self.__p(new_order[1:], pstate, r)
        if list(new_order) == range(r.type.ndim):
            return pstate.pprinter.process(r)
        if list(new_order) == list(reversed(range(r.type.ndim))):
            return "%s.T" % pstate.pprinter.process(r)
        return "DimShuffle{%s}(%s)" % (", ".join(map(str, new_order)), pstate.pprinter.process(r))

    def process(self, r, pstate):
        if r.owner is None:
            raise TypeError("Can only print DimShuffle.")
        elif isinstance(r.owner.op, DimShuffle):
            ord = r.owner.op.new_order
            return self.__p(ord, pstate, r.owner.inputs[0])
        else:
            raise TypeError("Can only print DimShuffle.")

pprint.assign(lambda pstate, r: r.owner and isinstance(r.owner.op, DimShuffle), DimShufflePrinter())



################
### Elemwise ###
################

class Elemwise(Op):
    """
    Generalizes a scalar op to tensors.

    All the inputs must have the same number of dimensions. When the
    Op is performed, for each dimension, each input's size for that
    dimension must be the same. As a special case, it can also be 1
    but only if the input's broadcastable flag is True for that
    dimension. In that case, the tensor is (virtually) replicated
    along that dimension to match the size of the others.

    The dtypes of the outputs mirror those of the scalar Op that is
    being generalized to tensors. In particular, if the calculations
    for an output are done inplace on an input, the output type must
    be the same as the corresponding input type (see the doc of
    scalar.ScalarOp to get help about controlling the output type)

    Examples:
      Elemwise(add) # represents + on tensors (x + y)
      Elemwise(add, {0 : 0}) # represents the += operation (x += y)
      Elemwise(add, {0 : 1}) # represents += on the second argument (y += x)
      Elemwise(mul)(rand(10, 5), rand(1, 5)) # the second input is completed along the first dimension to match the first input
      Elemwise(true_div)(rand(10, 5), rand(10, 1)) # same but along the second dimension
      Elemwise(int_div)(rand(1, 5), rand(10, 1)) # the output has size (10, 5)
      Elemwise(log)(rand(3, 4, 5))
    """

    def __init__(self, scalar_op, inplace_pattern = {}, name = None, nfunc_spec = None):
        """
        Usage: Elemwise(scalar_op, inplace_pattern = {})

        * scalar_op: an instance of a subclass of scalar.ScalarOp which works uniquely on
                     scalars
        * inplace_pattern: a dictionary that maps the index of an output to the
                           index of an input so the output is calculated inplace using
                           the input's storage. (Just like destroymap, but without the lists.)
        * nfunc_spec: either None or a tuple of three elements, (nfunc_name, nin, nout) such
                      that getattr(numpy, nfunc_name) implements this operation, takes nin
                      inputs and abs(nout) outputs (nout < 0 if the numpy function
                      does not provide the option of providing a numpy array to store the
                      results in). Note that nin cannot always be inferred from the scalar op's
                      own nin field because that value is sometimes 0 (meaning a variable number
                      of inputs), whereas the numpy function may not have varargs. NOTE: as of
                      now, the sign of the nout field is ignored (some work needs to be done
                      to resize the destinations when needed).
        """
        self.name = name
        self.scalar_op = scalar_op
        self.inplace_pattern = inplace_pattern
        self.destroy_map = dict((o, [i]) for o, i in inplace_pattern.items())

        self.ufunc = None
        self.nfunc = None
        self.nfunc_spec = nfunc_spec
        if nfunc_spec:
            self.nfunc = getattr(numpy, nfunc_spec[0])
        elif scalar_op.nin > 0:
            self.ufunc = numpy.frompyfunc(scalar_op.impl, scalar_op.nin, scalar_op.nout)

        #precompute the hash of this node
        self._rehash()

    def __getstate__(self):
        d = copy(self.__dict__)
        d.pop('ufunc')
        d.pop('nfunc')
        d.pop('__epydoc_asRoutine', None)
        d.pop('_hashval')
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.ufunc = None
        self.nfunc = None
        if getattr(self, 'nfunc_spec', None):
            self.nfunc = getattr(numpy, self.nfunc_spec[0])
        elif self.scalar_op.nin > 0:
            self.ufunc = numpy.frompyfunc(self.scalar_op.impl, self.scalar_op.nin, self.scalar_op.nout)
        self._rehash()

    def make_node(self, *inputs):
        """
        If the inputs have different number of dimensions, their shape
        is left-completed to the greatest number of dimensions with 1s
        using DimShuffle.
        """

        inputs = map(as_tensor_variable, inputs)
        shadow = self.scalar_op.make_node(*[Scalar(dtype=i.type.dtype)() for i in inputs])

        target_length = max([input.type.ndim for input in inputs])

        args = []
        for input in inputs:
            length = input.type.ndim
            difference = target_length - length
            if not difference:
                args.append(input)
            else:
                # TODO: use LComplete instead
                args.append(DimShuffle(
                    input.type.broadcastable,
                    ['x']*difference + range(length),
                    inplace = True)(input))
        inputs = args

        #HERE: all the broadcast dims have the same length now

        #cleverness: we iterate over the first, second, third broadcast flag of all inputs in
        #parallel... the all() gives us each output broadcastable bit in turn.

        #it is multiplied by nout because Elemwise supports multiple outputs (nout of them)
        out_broadcastables = [[all(bcast) for bcast in zip(*[input.type.broadcastable for input in inputs])]] * shadow.nout

        #inplace_pattern maps output idx -> input idx
        inplace_pattern = self.inplace_pattern
        if inplace_pattern:
            for overwriter, overwritten in inplace_pattern.items():
                for ob, ib in zip(out_broadcastables[overwriter], inputs[overwritten].type.broadcastable):
                    if ib and not ob:
                        raise ValueError("Operation cannot be done inplace on an input with broadcasted dimensions.")
        out_dtypes = [o.type.dtype for o in shadow.outputs]
        if any(inputs[i].type.dtype != out_dtypes[o] for o, i in inplace_pattern.items()):
            raise TypeError("Cannot do an inplace operation on incompatible data types.",
                    ([i.type.dtype for i in inputs], out_dtypes, inplace_pattern))
        outputs = [TensorType(dtype = dtype, broadcastable = broadcastable)() for dtype, broadcastable in zip(out_dtypes, out_broadcastables)]
        return Apply(self, inputs, outputs)

    def __eq__(self, other):
        if type(self) == type(other):
            items = self.inplace_pattern.items()
            other_items = other.inplace_pattern.items()
            items.sort()
            other_items.sort()
            rval = (self.scalar_op == other.scalar_op) and (items == other_items)
            return rval
        return False

    def _rehash(self):
        items = self.inplace_pattern.items()
        items.sort()
        first_part = [k for k,v in items]
        second_part = []
        for k,v in items:
            if isinstance(v, (tuple, list)):
                second_part += [tuple(v)]
            else:
                second_part += [v]
        tuple_items = tuple(first_part + second_part)
        #backport
        #tuple_items = tuple([k for k,v in items] + [(tuple(v) if isinstance(v, (tuple, list)) else v) for k,v in items])
        h = hash('Elemwise') ^ hash(self.scalar_op) ^ hash(tuple_items)
        assert h == getattr(self,'_hashval', h)
        self._hashval = h

    def __hash__(self):
        return self._hashval

    def __str__(self):
        if self.name is None:
            if self.inplace_pattern:
                items = self.inplace_pattern.items()
                items.sort()
                return "Elemwise{%s}%s" % (self.scalar_op, str(items))
            else:
                return "Elemwise{%s}" % (self.scalar_op)
        else:
            return self.name

    def R_op(self, inputs, eval_points):
        outs = self.make_node(*inputs).outputs
        rval = [None for x in outs]
        # For each output
        for idx, out in enumerate(outs):
            # make such that _bgrads computes only the gradients of the
            # current output on the inputs ( and not all outputs)
            ograds = [ theano.tensor.zeros_like(x) for x in outs]
            ograds[idx] = theano.tensor.ones_like(out)

            bgrads = self._bgrad(inputs, ograds)
            rop_out = None

            for jdx, (inp, eval_point) in enumerate(zip(inputs,
                                                        eval_points)):
                # if None, then we can just ignore this branch ..
                # what we do is to assume that for any non-differentiable
                # branch, the gradient is actually 0, which I think is not
                # the right thing to do .. have to talk to Ian and James
                # about it

                if bgrads[jdx] is None:
                    pass
                elif eval_point is not None:
                    if rop_out is None:
                        rop_out = bgrads[jdx]*eval_point
                    else:
                        rop_out = rop_out + bgrads[jdx]*eval_point

            rval[idx] = rop_out

        return rval

    def grad(self, inputs, ograds):

        #compute grad with respect to broadcasted input
        rval = self._bgrad(inputs,ograds)

        #sum out the broadcasted dimensions
        for i, ipt in enumerate(inputs):
            if rval[i] is None:
                continue

            # list of all the dimensions that are broadcastable for input[i] so we
            # can sum over them
            # todo: only count dimensions that were effectively broadcasted
            to_sum = [j for j, bcast in enumerate(ipt.type.broadcastable) if bcast]

            if to_sum:
                shuffle = []
                j = 0
                for bcast in ipt.type.broadcastable:
                    if bcast == 1:
                        shuffle.append('x')
                    else:
                        shuffle.append(j)
                        j += 1
                    #close if
                #close for
                sr = Sum(axis = to_sum)(rval[i])
                sr = sr.dimshuffle(shuffle)
                #sr = DimShuffle(sr.type.broadcastable, shuffle)(sr)
                rval[i] = sr
            #close if
        #close for

        return rval


    def _bgrad(self, inputs, ograds):
        # returns grad, with respect to broadcasted versions of inputs

        # Gradients (especially on the final costs) don't have to be symbolic
        # e.g., ograds will be [ 1. ] if your objective is c and the output
        # of the current apply node is c
        ograds = map(as_tensor_variable, ograds)

        prev_setting = theano.config.compute_test_value

        try:

            theano.config.compute_test_value = 'off'

            scalar_inputs = [Scalar(dtype = t.type.dtype)() for t in inputs]
            scalar_ograds = [Scalar(dtype = ograd.type.dtype)() for ograd in ograds]
            scalar_igrads = self.scalar_op.grad(scalar_inputs, scalar_ograds)

        finally:

            theano.config.compute_test_value = prev_setting

        nd = len(inputs[0].type.broadcastable) # this is the same for everyone
        def transform(r):
            # From a graph of ScalarOps, make a graph of Broadcast ops.
            if r in scalar_inputs:
                return inputs[scalar_inputs.index(r)]
            if r in scalar_ograds:
                return ograds[scalar_ograds.index(r)]
            node = r.owner
            if node is None:
                # the gradient contains a constant, translate it as
                # an equivalent TensorType of size 1 and proper number of dimensions
                res = TensorConstant(TensorType(dtype = r.type.dtype,
                                            broadcastable = ()),
                                     numpy.asarray(r.data)) # .reshape(b)
                return DimShuffle((), ['x']*nd, inplace = True)(res)
            new_r = Elemwise(node.op, {})(*[transform(ipt) for ipt in node.inputs])
            return new_r
        ret = []
        for scalar_igrad, ipt in zip(scalar_igrads, inputs):
            if scalar_igrad is None:
                # undefined gradient
                ret.append(None)
                continue
            ret.append( transform(scalar_igrad))


        return ret

    def perform(self, node, inputs, output_storage):
        maxsize = max(len(input.shape) for input in inputs)
        for dims in zip(*[[(1, True)]*(maxsize - len(input.shape)) + zip(input.shape, sinput.type.broadcastable)
                          for input, sinput in zip(inputs, node.inputs)]):
            if max(d for d,b in dims) != 1 and (1, False) in dims:
                # yes there may be more compact ways to write this code,
                # but please maintain python 2.4 compatibility (no "x if c else y")
                msg = []
                assert len(inputs) == len(node.inputs)
                for input, sinput in zip(inputs, node.inputs):
                    assert len(input.shape) == len(sinput.type.broadcastable)
                    msg2 = []
                    for d, b in zip(input.shape, sinput.type.broadcastable):
                        if b:
                            msg2 += ['*']
                        else:
                            msg2 += [str(d)]
                    msg.append('(%s)' % ", ".join(msg2))

                base_exc_str = 'Dimension mismatch; shapes are %s' % (
                               ', '.join(msg))
                if config.exception_verbosity == 'high':
                    msg_chunks = [base_exc_str]
                    for i, ipt in enumerate(node.inputs):
                        msg_chunks.append('input %d: %s' %
                                          (i, min_informative_str(ipt)))
                    raise ValueError('\n'.join(msg_chunks))
                else:
                    raise ValueError(base_exc_str)

                #backport
                #raise ValueError('Dimension mismatch; shapes are %s' %
                #                 ', '.join('(%s)' % ', '.join('*' if b else str(d)
                #                                              for d, b in zip(input.shape, sinput.type.broadcastable))
                #                           for input, sinput in zip(inputs, node.inputs)))
                # Other mismatches will be caught by the ufunc
        if not self.inplace_pattern:
            for output, storage in zip(node.outputs, output_storage):
                odat = storage[0]
                shape = [max(values) for values in zip(*[input.shape for input in inputs])]
                if odat is not None:
                    # reuse storage if we can
                    odat.resize(shape, refcheck = 0)
                else:
                    odat = numpy.ndarray(shape, dtype = output.type.dtype)
                storage[0] = odat
        else:
            for i, (output, storage) in enumerate(zip(node.outputs, output_storage)):
                #i is an output idx
                if i in self.inplace_pattern:
                    odat = inputs[self.inplace_pattern[i]]
                else:
                    odat = storage[0]
                    shape = [max(values) for values in zip(*[input.shape for input in inputs])]
                    if odat is not None:
                        odat.resize(shape, refcheck = 0)
                    else:
                        odat = numpy.ndarray(shape, dtype = output.type.dtype)
                storage[0] = odat

        ufunc_args = inputs # + output_storage
        if self.nfunc and len(inputs) == self.nfunc_spec[1]:
            ufunc = self.nfunc
            nout = self.nfunc_spec[2]
            if nout < 0:
                nout = -nout
            # Unfortunately, the else case does not allow us to
            # directly feed the destination arguments to the nfunc
            # since it sometimes requires resizing. Doing this
            # optimization is probably not worth the effort, since we
            # should normally run the C version of the Op.
        else:
            # the second calling form is used because in certain versions of numpy
            # the first (faster) version leads to segfaults
            ufunc = self.ufunc or numpy.frompyfunc(self.scalar_op.impl, len(inputs), self.scalar_op.nout)
            nout = ufunc.nout

        try:
            variables = ufunc(*ufunc_args)
        except Exception, e:
            errormsg = 'While computing '+str(node.outputs)+': Failed calling ufunc for op', self.scalar_op,\
                        'for params of shape', [arg.shape for arg in ufunc_args]
            e.args = e.args + errormsg
            raise
        if nout == 1:
            variables = [variables]
        for variable, storage, nout in zip(variables, output_storage, node.outputs):
            if str(getattr(variable, "dtype", "")) == 'object':
                # Since numpy 1.6, function created with numpy.frompyfunc
                # always return an ndarray with dtype object
                variable = numpy.asarray(variable, dtype=nout.dtype)
            if hasattr(variable, 'shape') and storage[0].shape != variable.shape:
                if numpy.prod(variable.shape) == 0:
                    # numpy don't resize from a shape (1,5) to (0,5)
                    # This bypass the inplace... But I it is important in this case.
                    storage[0] = variable
                    continue
                storage[0].resize(variable.shape)

            if storage[0].shape:
                storage[0][:] = variable
            else:
                storage[0].itemset(variable)
            assert str(storage[0].dtype) != 'object'
        # the following should be used instead of the previous loop, unfortunately it tends to segfault
        # self.ufunc(*(ufunc_args+[s[0] for s in output_storage]))

    def infer_shape(self, node, i_shapes):
        rval = []
        for o in node.outputs:
            oshp = []
            for dim, b in enumerate(o.type.broadcastable):
                b_dim = None
                if b: # this is broadcastable
                    b_dim = 1
                else: # there must be some input that is not broadcastable in dimension 'dim'
                    for ishp, i in zip(i_shapes,node.inputs):
                        if isinstance(i.type,theano.scalar.Scalar):
                            continue #we skip scalar
                        if not i.type.broadcastable[dim]:
                            # input i is not broadcastable in position dim
                            # therefore if its shape is known, we can use it
                            # as the output shape
                            if ishp[dim]:
                                b_dim = ishp[dim]
                                break
                # b_dim might still be None, if every input's shape was unknown in dimension 'dim'
                oshp.append(b_dim)
                # TODO: it would be interesting to return the constraining information that if
                # one of the inputs shape[dim] is known and another input's shape[dim] is not,
                # that we can now assume that the other input's shape[dim] is the same as the
                # first.
            rval.append(tuple(oshp))
        return rval

    def _c_all(self, node, nodename, inames, onames, sub):
        _inames = inames
        _onames = onames

        inames = gof.utils.uniq(inames)
        inputs = gof.utils.uniq(node.inputs)

        defines = ""
        undefs = ""

        # The destroy map is a map of output indices to input indices
        # that overwrite them.  We just convert them to the actual
        # Variables.
        dmap = dict([(node.outputs[o], [node.inputs[i]])
                     for o, i in self.inplace_pattern.iteritems()])

        # dtypes of the inputs
        idtypes = [input.type.dtype_specs()[1] for input in inputs]

        # These are the outputs that we will need to allocate
        # (output, name, name of the c type), transposed
        real = zip(*[(r, s, r.type.dtype_specs()[1])
                     for r, s in zip(node.outputs, onames) if r not in dmap])
        if real:
            real_outputs, real_onames, real_odtypes = real
        else:
            real_outputs, real_onames, real_odtypes = [], [], []

        # Outputs that are aliased with an input (inplace)
        # (output, name), transposed (c type name not needed since we don't
        # need to allocate.
        aliased = zip(*[(r, s)
                        for (r, s) in zip(node.outputs, onames) if r in dmap])
        if aliased:
            aliased_outputs, aliased_onames = aliased
        else:
            aliased_outputs, aliased_onames = [], []

        # for each input:
        # same as range(ndim), but with 'x' at all broadcastable positions
        orders = [[x and 'x' or i
                   for i, x in enumerate(input.type.broadcastable)]
                  for input in inputs]

        # number of nested loops we will need (all inputs have same
        # dimensionality)
        nnested = len(orders[0])
        sub = dict(sub)
        for i, (input, iname) in enumerate(zip(inputs, inames)):
            # the c generators will substitute the input names for
            # references to loop variables lv0, lv1, ...
            sub['lv%i' % i] = iname

        decl = cgen.make_declare(orders, idtypes, sub)
        checks = cgen.make_checks(orders, idtypes, sub)

        alloc = ""
        # We loop over the "real" outputs, i.e., those that are not
        # inplace (must be allocated) and we declare/allocate/check
        # them
        for output, oname, odtype in zip(real_outputs, real_onames, real_odtypes):
            i += 1 # before this loop, i = number of inputs
            sub['lv%i' % i] = oname
            sub['olv'] = oname
            alloc += cgen.make_declare([range(nnested)], [odtype],
                                       dict(sub, lv0 = oname))
            alloc += cgen.make_alloc(orders, odtype, sub)
            alloc += cgen.make_checks([range(nnested)], [odtype],
                                      dict(sub, lv0 = oname))
        olv_index = i # index of the last output

        # We loop over the "aliased" outputs, i.e., those that are
        # inplace (overwrite the contents of one of the inputs) and
        # make the output pointers point to theur corresponding input
        # pointers.
        for output, oname in zip(aliased_outputs, aliased_onames):
            olv_index = inputs.index(dmap[output][0])
            iname = inames[olv_index]
            # We make the output point to the corresponding input and
            # decrease the reference of whatever the output contained
            # prior to this
            alloc += """
            if (%(oname)s) {
                Py_XDECREF(%(oname)s);
            }
            %(oname)s = %(iname)s;
            Py_XINCREF(%(oname)s);
            """ % locals()
            # We alias the scalar variables
            defines += "#define %(oname)s_i %(iname)s_i" % locals()
            undefs += "#undef %(oname)s_i" % locals()

        # Note: here, olv_index is either the index of the last output
        # which is allocated, OR, if there are any aliased outputs,
        # the index of the last of these aliased outputs.

        # We declare the scalar variables used in the inner loop to do
        # the element-wise computation. Aliased scalar variables need
        # not be declared, as they are #defined in defines
        task_decl = "".join(["%(dtype)s& %(name)s_i = *%(name)s_iter;\n" % locals()
                             for name, dtype in zip(inames + list(real_onames),
                                                    idtypes + list(real_odtypes))])

        # We generate the C code of the inner loop using the scalar op
        task_code = self.scalar_op.c_code(
                Apply(self.scalar_op,
                      [Scalar(dtype = input.type.dtype)() for input in node.inputs],
                      [Scalar(dtype = output.type.dtype)() for output in node.outputs]),
                nodename + '_scalar_',
                ["%s_i" % s for s in _inames],
                ["%s_i" % s for s in onames],
                sub)
        code = """
        {
            %(defines)s
            %(task_decl)s
            %(task_code)s
            %(undefs)s
        }
        """ % locals()

        loop = cgen.make_reordered_loop(
                init_loop_orders = orders + [range(nnested)] * len(real_onames),
                olv_index = olv_index,
                dtypes = idtypes + list(real_odtypes),
                inner_task = code,
                sub = sub)
        return decl, checks, alloc, loop

    def c_code(self, node, nodename, inames, onames, sub):
        code = "\n".join(self._c_all(node, nodename, inames, onames, sub))
        return code

    def c_headers(self):
        return ['<vector>', '<algorithm>']

    def c_support_code(self):
        return self.scalar_op.c_support_code()

    def c_support_code_apply(self, node, nodename):
        support_code = self.scalar_op.c_support_code_apply(node,
                nodename + '_scalar_')
        return support_code

    def c_code_cache_version_apply(self, node):
        version = [6] # the version corresponding to the c code in this Op

        # now we insert versions for the ops on which we depend...
        scalar_node = Apply(self.scalar_op,
                [Scalar(dtype = input.type.dtype)() for input in node.inputs],
                [Scalar(dtype = output.type.dtype)() for output in node.outputs])
        version.extend(self.scalar_op.c_code_cache_version_apply(scalar_node))
        for i in node.inputs + node.outputs:
            version.extend(Scalar(dtype=i.type.dtype).c_code_cache_version())
        if all(version):
            return tuple(version)
        else:
            return ()

# def elemwise_to_scal(env):
#     mapping = {}
#     inputs = []
#     outputs = []
#     for node in env.io_toposort():
#         if not isinstance(node.op, Elemwise):
#             raise TypeError('All ops in the graph must be Elemwise.')



################
### CAReduce ###
################

class CAReduce(Op):
    """
    Reduces a scalar operation along the specified axis(es).

    The output will have the same shape as the input minus the reduced
    dimensions. It will contain the variable of accumulating all values
    over the reduced dimensions using the specified scalar op.

    Examples:
     CAReduce(add) -> sum
     CAReduce(mul) -> product
     CAReduce(maximum) -> max
     CAReduce(minimum) -> min
     CAReduce(or_) -> any # not lazy
     CAReduce(and_) -> all # not lazy
     CAReduce(xor) -> a bit at 1 tell that there was an odd number of bit at
                      that position that where 1.
                      0 it was an even number ...

    In order to (eventually) optimize memory usage patterns,
    L{CAReduce} makes zero guarantees on the order in which it
    iterates over the dimensions and the elements of the
    array(s). Therefore, to ensure consistent variables, the scalar
    operation represented by the reduction must be both commutative
    and associative (eg add, multiply, maximum, binary or/and/xor - but not
    subtract, divide or power).
    """

    def __init__(self, scalar_op, axis=None):
        """
        Usage: CAReduce(scalar_op, axis = None)

        * scalar_op: a binary scalar op with only one output.
                     It must be commutative and associative.
        * axis: - the dimension along which we want to reduce
                - list of dimensions that we want to reduce
                - if None, all dimensions are reduced
        """
        if scalar_op.nin not in [-1, 2] or scalar_op.nout != 1:
            raise NotImplementedError("CAReduce only supports binary functions with a single output.")
        self.scalar_op = scalar_op

        if axis is None:
            self.axis = axis
        elif isinstance(axis, int):
            self.axis = (axis,)
        else:
            self.axis = list(set(axis))
            self.axis.sort()
            self.axis = tuple(self.axis)

        # This is probably a speed up of the implementation
        if isinstance(scalar_op, theano.scalar.basic.Add):
            self.ufunc = numpy.add
        elif isinstance(scalar_op, theano.scalar.basic.Mul):
            self.ufunc = numpy.multiply
        elif isinstance(scalar_op, theano.scalar.basic.Maximum):
            self.ufunc = numpy.maximum
        elif isinstance(scalar_op, theano.scalar.basic.Minimum):
            self.ufunc = numpy.minimum
        elif isinstance(scalar_op, theano.scalar.basic.AND):
            self.ufunc = numpy.bitwise_and
        elif isinstance(scalar_op, theano.scalar.basic.OR):
            self.ufunc = numpy.bitwise_or
        elif isinstance(scalar_op, theano.scalar.basic.XOR):
            self.ufunc = numpy.bitwise_xor
        else:
            self.ufunc = numpy.frompyfunc(scalar_op.impl, 2, 1)

    def _output_dtype(self, input_dtype):
        return input_dtype

    def make_node(self, input):
        input = as_tensor_variable(input)

        if self.axis is not None:
            for axis in self.axis:
                if axis >= input.type.ndim or (axis<0 and abs(axis)>input.type.ndim):
                    raise ValueError('Not enough dimensions on %s to reduce on axis %s' % (input, axis))
        input = as_tensor_variable(input)
        axis = self.axis
        if axis is None:
            axis = range(len(input.type.broadcastable))
        if any([a<0 for a in axis]):
            axis2=[]
            for a in self.axis:
                if a<0:
                    axis2.append(a+input.type.ndim)
                else:
                    axis2.append(a)
            assert len(axis)==len(axis2)
            axis = tuple(axis2)
            op = self.__class__(self.scalar_op, axis)
        else:
            op = self
        broadcastable = [x for i, x in enumerate(input.type.broadcastable)
                         if i not in axis]
        output = TensorType(dtype=self._output_dtype(input.type.dtype),
                            broadcastable=broadcastable)()
        return Apply(op, [input], [output])

    def __getstate__(self):
        d = copy(self.__dict__)
        d.pop('ufunc')
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.ufunc = numpy.frompyfunc(self.scalar_op.impl, 2, 1)

    def __eq__(self, other):
        return type(self) == type(other) and self.scalar_op == other.scalar_op and self.axis == other.axis

    def __hash__(self):
        if self.axis is None:
            return hash(self.scalar_op)
        else:
            return hash(self.scalar_op) ^ hash(tuple(self.axis))

    def __str__(self):
        if self.axis is not None:
            return "Reduce{%s}{%s}" % (self.scalar_op, ", ".join(str(x) for x in self.axis))
        else:
            return "Reduce{%s}" % self.scalar_op

    def perform(self, node, inp, out):
        input, = inp
        output, = out
        axis = self.axis
        if axis is None:
            axis = range(input.ndim)
        variable = input
        to_reduce = reversed(sorted(axis))
        if to_reduce:
            for dimension in to_reduce:
                # If it's a zero-size array, use scalar_op.identity if available
                if variable.shape[dimension] == 0:
                    if hasattr(self.scalar_op, 'identity'):
                        variable = numpy.array(self.scalar_op.identity)
                        break
                    else:
                        raise ValueError("Input (%s) has zero-size on axis %s, but self.scalar_op (%s) has no attribute 'identity'" % (variable, dimension, self.scalar_op))
                else:
                    # Numpy 1.6 has a bug where you sometimes have to specify
                    # "dtype='object'" in reduce for it to work, if the ufunc
                    # was built with "frompyfunc". We need to find out if we
                    # are in one of these cases (only "object" is supported in
                    # the output).
                    if ((self.ufunc.ntypes == 1)
                            and (self.ufunc.types[0][-1] == 'O')):
                        variable = self.ufunc.reduce(variable, dimension,
                                dtype='object')
                    else:
                        variable = self.ufunc.reduce(variable, dimension)

            variable = numpy.asarray(variable)
            if numpy.may_share_memory(variable, input):
                # perhaps numpy is clever for reductions of size 1?  We don't want this.
                variable = variable.copy()
            output[0] = theano._asarray(variable, dtype = node.outputs[0].type.dtype)
        else:
            output[0] = numpy.copy(variable)

    def infer_shape(self, node, shapes):
        ishape, = shapes
        axis = self.axis
        if axis is None:
            return (),
        return [ishape[i] for (i,b) in enumerate(node.inputs[0].type.broadcastable) if i not in axis],


    def _c_all(self, node, name, inames, onames, sub):

        input = node.inputs[0]
        output = node.outputs[0]

        iname = inames[0]
        oname = onames[0]

        idtype = input.type.dtype_specs()[1]
        odtype = output.type.dtype_specs()[1]

        axis = self.axis
        if axis is None:
            axis = range(len(input.type.broadcastable))

        if len(axis) == 0:
            op = Elemwise(scalar.identity)
            return op._c_all(op.make_node(input), name, inames, onames, sub)

        order1 = [i for i in xrange(input.type.ndim) if i not in axis]
        order = order1 + list(axis)

        nnested = len(order1)

        sub = dict(sub)
        for i, (input, iname) in enumerate(zip(node.inputs, inames)):
            sub['lv%i' % i] = iname

        decl = cgen.make_declare([order], [idtype], sub)
        checks = cgen.make_checks([order], [idtype], sub)

        alloc = ""
        i += 1
        sub['lv%i' % i] = oname
        sub['olv'] = oname
        alloc += cgen.make_declare([range(nnested) + ['x'] * len(axis)], [odtype], dict(sub, lv0 = oname))
        alloc += cgen.make_alloc([order1], odtype, sub)
        alloc += cgen.make_checks([range(nnested) + ['x'] * len(axis)], [odtype], dict(sub, lv0 = oname))

        if hasattr(self.scalar_op,'identity'):
            identity = self.scalar_op.identity
        elif self.scalar_op in [scalar.maximum, scalar.minimum]:
            if self.scalar_op == scalar.maximum:
                scal_name = 'maximum'
                if input.type.dtype in ["float32","float64"]:
                    identity = "-__builtin_inf()"
                elif input.type.dtype.startswith("uint"):
                    # numpy1.5.1 don't define NPY_MIN_UINT*
                    identity = "0"
                else:
                    identity = "NPY_MIN_"+str(input.type.dtype).upper()
            if self.scalar_op == scalar.minimum:
                scal_name = 'minimum'
                if input.type.dtype in ["float32","float64"]:
                    identity = "__builtin_inf()"
                else:
                    identity = "NPY_MAX_"+str(input.type.dtype).upper()
            fail = sub["fail"]
            pattern=[0]*len(node.inputs[0].broadcastable)
            axis = self.axis
            if axis == None: axis = range(len(pattern))
            for i in axis:
                pattern[i]=1
            pattern_ = str(pattern)[1:-1]
            decl +="""int tosum[]={%(pattern_)s};"""%locals()
            alloc += """
for(int i=0;i<%(iname)s->nd;i++){
  if(PyArray_DIMS(%(iname)s)[i]==0 && tosum[i]){
    PyErr_Format(PyExc_ValueError, "Input of CAReduce{%(scal_name)s} has zero-size on axis %%d",i);
    %(fail)s;
  }
}
                   """%locals()
        else:
            raise TypeError("The CAReduce.scalar_op must have an identity field.")

        task0_decl = "%(dtype)s& %(name)s_i = *%(name)s_iter;\n%(name)s_i = %(identity)s;" % dict(dtype = odtype,
                                                                                                  name = onames[0],
                                                                                                  identity = identity)

        task1_decl = "%(dtype)s& %(name)s_i = *%(name)s_iter;\n" % dict(dtype = idtype, name = inames[0])

        task1_code = self.scalar_op.c_code(Apply(self.scalar_op,
                                                 [Scalar(dtype = input.type.dtype)() for input in node.inputs*2],
                                                 [Scalar(dtype = output.type.dtype)() for input in node.outputs]),
                                           None,
                                           ["%s_i" % onames[0], "%s_i" % inames[0]],
                                           ["%s_i" % onames[0]],
                                           sub)
        code1 = """
        {
            %(task1_decl)s
            %(task1_code)s
        }
        """ % locals()

        if node.inputs[0].type.ndim:
            if len(axis) == 1:
                all_code = [("", "")] * nnested + [(task0_decl, code1), ""]
            else:
                all_code = [("", "")] * nnested + [(task0_decl, "")] + [("", "")] * (len(axis) - 2) + [("", code1), ""]
        else:
            all_code = [task0_decl + code1]
        loop = cgen.make_loop([order, range(nnested) + ['x'] * len(axis)], [idtype, odtype], all_code, sub)
        return decl, checks, alloc, loop

    def c_code(self, node, name, inames, onames, sub):
        code = "\n".join(self._c_all(node, name, inames, onames, sub))
        return code

    def c_headers(self):
        # Sometimes, Elemwise's c_code is returned, so we need its headers
        return ['<vector>', '<algorithm>']

    def c_code_cache_version_apply(self, node):
        version = [4] # the version corresponding to the c code in this Op

        # now we insert versions for the ops on which we depend...
        scalar_node = Apply(self.scalar_op,
                [Scalar(dtype = input.type.dtype)() for input in node.inputs],
                [Scalar(dtype = output.type.dtype)() for output in node.outputs])
        version.extend(self.scalar_op.c_code_cache_version_apply(scalar_node))
        for i in node.inputs + node.outputs:
            version.extend(Scalar(dtype=i.type.dtype).c_code_cache_version())
        if all(version):
            return tuple(version)
        else:
            return ()


class All(CAReduce):
    """ Applies `bitwise and` to all the values of a tensor along the
    specified axis(es).

    Equivalent to CAReduce(scalar.and_, axis=axis)
    """
    def __init__(self, axis=None):
        CAReduce.__init__(self, scalar.and_, axis)

    def _output_dtype(self, idtype):
        return "int8"

    def __str__(self):
        if self.axis is None:
            return "All"
        else:
            return "All{%s}" % ", ".join(map(str, self.axis))


class Any(CAReduce):
    """ Applies `bitwise or` to all the values of a tensor along the
    specified axis(es).

    Equivalent to CAReduce(scalar.or_, axis=axis)
    """
    def __init__(self, axis=None):
        CAReduce.__init__(self, scalar.or_, axis)

    def _output_dtype(self, idtype):
        return "int8"

    def __str__(self):
        if self.axis is None:
            return "Any"
        else:
            return "Any{%s}" % ", ".join(map(str, self.axis))


class CAReduceDtype(CAReduce):
    """
    Reduces a scalar operation along the specified axis(es).

    This subclass of CAReduce accepts an additional "dtype" parameter,
    that specifies which dtype will be used for the accumulation.

    If no dtype is provided, one will be inferred so as not to lose
    too much precision.
    """

    def __init__(self, scalar_op, axis=None, dtype=None):
        """
        Usage: CAReduceDtype(scalar_op, axis=None, dtype=None)

        :param scalar_op: a binary scalar op with only one output.
                     It must be commutative and associative.

        :axis:  - the dimension along which we want to reduce
                - list of dimensions that we want to reduce
                - if None, all dimensions are reduced

        :param dtype: The dtype of the internal accumulator and returned
        tensor. If None, then we use the default dtype which is the same as the
        input tensor's dtype except when:
            - the input dtype is a signed integer of precision < 64 bit, in
              which case we use int64
            - the input dtype is an unsigned integer of precision < 64 bit, in
              which case we use uint64
        This behavior is similar in spirit to that of numpy (except numpy
        uses the default machine integer while we always use 64 bit integers to
        avoid platform-dependent behavior).

        """
        CAReduce.__init__(self, scalar_op, axis=axis)
        self.dtype = dtype

    def __eq__(self, other):
        return CAReduce.__eq__(self, other) and self.dtype == other.dtype

    def __hash__(self):
        return CAReduce.__hash__(self) ^ hash(self.dtype)

    def _output_dtype(self, idtype):
        dtype = self.dtype
        if dtype is None:
            # If input has an discrete dtype, upcast it to 64
            return dict(
                    int8='int64',
                    int16='int64',
                    int32='int64',
                    uint8='uint64',
                    uint16='uint64',
                    uint32='uint64',
                    ).get(idtype, idtype)
        elif dtype in continuous_dtypes and idtype in discrete_dtypes:
            # Specifying a continuous output for discrete input is OK
            return dtype
        else:
            # The conversion has to be considered an upcast.
            upcasted_dtype = scalar.upcast(idtype, dtype)
            if dtype != upcasted_dtype:
                raise TypeError(
                        'Cannot build %s node with input dtype %s '
                        'and output dtype %s, as precision would be lost. '
                        'To correct this error, you can either:\n'
                        '  - not specify a dtype, or\n'
                        '  - use a dtype at least as precise as %s.\n'
                        'If you are expecting the precision loss, you can '
                        'use tensor.cast(..., dtype="%s"), either on your '
                        'input, or on the output of the reduce operation.'
                        % (self, idtype, dtype, upcasted_dtype, dtype))
            return dtype

    def make_node(self, input):
        # We need to redefine make_node so that, if self.dtype is None,
        # we can infer what dtype should be, and create a node from an Op
        # of the appropriate dtype.
        dtype = self._output_dtype(input.dtype)
        assert dtype is not None
        if dtype == self.dtype:
            # Don't build another instance
            op = self
        else:
            op = self.__class__(axis=self.axis, dtype=dtype)
        return CAReduce.make_node(op, input)


class Sum(CAReduceDtype):
    """
    Sums all the values of a tensor along the specified axis(es).

    Equivalent to CAReduceDtype(scalar.add, axis=axis, dtype=dtype),
    with the difference that this defines the gradient of sum wrt its
    tensor input.
    """

    def __init__(self, axis=None, dtype=None):
        """
        Constructor.

        :param axis: Axis(es) along which the tensor should be summed
        (use None to sum over all axes, and a list or tuple to sum along more
        than one axis).

        :param dtype: The dtype of the internal accumulator and returned
        tensor. If None, then we use the default dtype which is the same as the
        input tensor's dtype except when:
            - the input dtype is a signed integer of precision < 64 bit, in
              which case we use int64
            - the input dtype is an unsigned integer of precision < 64 bit, in
              which case we use uint64
        """
        CAReduceDtype.__init__(self, scalar.add, axis=axis, dtype=dtype)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        gz = as_tensor_variable(gz)
        axis = self.axis
        if axis is None:
            axis = range(x.type.ndim)
        if axis == ():
            return gz,
        new_dims = []
        i = 0
        for j, _ in enumerate(x.type.broadcastable):
            if j in axis:
                new_dims.append('x')
            else:
                new_dims.append(i)
                i += 1
        return Elemwise(scalar.second)(
                        x, DimShuffle(gz.type.broadcastable, new_dims)(gz)),

    def R_op(self, inputs, eval_points):
        # There is just one element in inputs and eval_points, the axis are
        # part of self
        if None in eval_points:
            return [None]
        return self.make_node(*eval_points).outputs

    def __str__(self):
        if self.axis is None:
            return "Sum"
        else:
            return "Sum{%s}" % ", ".join(map(str, self.axis))


class Prod(CAReduceDtype):
    """
    Multiplies all the values of a tensor along the specified axis(es).

    Equivalent to CAReduce(scalar.prod, axis = axis), with the
    difference that this defines the gradient of prod wrt its tensor
    input.
    """
    def __init__(self, axis=None, dtype=None, no_zeros_in_input=False):
        CAReduceDtype.__init__(self, scalar.mul, axis=axis, dtype=dtype)
        self.no_zeros_in_input = no_zeros_in_input

    def __setstate__(self, dct):
        super(Prod, self).__setstate__(dct)
        # Add default value to be able to reload old pickled objects.
        if 'no_zeros_in_input' not in dct:
            self.no_zeros_in_input = False

    def __eq__(self, other):
        return (CAReduceDtype.__eq__(self, other)
                and self.no_zeros_in_input == other.no_zeros_in_input)

    def __hash__(self):
        return (CAReduceDtype.__hash__(self) ^
                hash(self.no_zeros_in_input))

    def grad(self, inp, grads):
        '''
        The grad of this Op could be very easy, it is was not for the case
        where zeros are present in a given "group" (ie. elements reduced
        together to form the product).

        If no zeros are found in the elements of the product, then the
        partial derivative of the product relative to one of the elements
        (one of the inputs) is simply the product of the other elements.
        That's easy to see from the chain rule.

        Now the trick (with no zeros) is to take the overall product, then
        for every original element, the partial derivative is given by
        this product divided by the element itself (which equals the product
        of the other terms). This is easy to do by broadcasting the original
        product.

        (Note that we also need to broadcast-multiply by the "incoming gradient",
        ie. the gradient of the cost relative to the output/product).

        -----

        With zeros, things get more complicated. For a given group, we have 3
        cases:
        * No zeros in the group. Use previous trick.
        * If only one zero is present, then the gradient for that element is
            non-zero, but is zero for all others.
        * If more than one zero is present, then all the derivatives are zero.

        For the last two cases (with 1 or more zeros), we can't use the division
        trick, as this gives divisions by 0.

        Implementing that case-by-case logic is not as trivial, so a bunch of
        hacks are piled down here to do it. Notably, for the "only one zero"
        case, there's a special Op that computes the product of the elements
        in the group, minus the zero (see ProdWithoutZero). The trick is then
        to use the division trick for groups with no zero, to use the
        ProdWithoutZeros op where there's only one zero, and to output a
        derivative of zero for any element part of a group with more than
        one zero.

        I do this by first counting the number of zeros in each group (see
        the "T.eq()" bits), then taking this or that behavior (see T.switch)
        based on the result of this count.
        '''
        prod_in, = inp
        gz, = grads
        if prod_in.dtype[0:3] in ('int','uin'):
            return [None]


        # Prepare the broadcasting that is used everywhere to broadcast
        # over the original groups (ie. broadcast over the elements of a given
        # product)
        gz = as_tensor_variable(gz)
        axis = self.axis
        if axis is None:
            axis = range(prod_in.type.ndim)
        if axis == ():
            return gz,
        new_dims = []
        i = 0
        for j, _ in enumerate(prod_in.type.broadcastable):
            if j in axis:
                new_dims.append('x')
            else:
                new_dims.append(i)
                i += 1

        # result of the product, broadcastable over groups
        prod_out = self(prod_in).dimshuffle(new_dims)
        # incoming gradient, broadcastable over groups
        gz = gz.dimshuffle(new_dims)

        # division trick if we don't have zeros. This will contain
        # NaNs to be eliminated in the T.switch if we do have zeros.
        grad_case_without_zeros = (gz * prod_out / prod_in)

        if self.no_zeros_in_input:
            # this handles inputs with zeros, but only certain input shapes
            return [grad_case_without_zeros]
        else:
            T = theano.tensor

            where_zeros = T.eq(prod_in, 0.0)
            sum_where_zeros = T.sum(where_zeros, axis=self.axis)
            groups_with_single_zero = T.eq(sum_where_zeros, 1).dimshuffle(new_dims)
            # tensor with 0 everywhere except for those places where
            # a 0 part of a group with a single zero was to be found
            where_single_zero = groups_with_single_zero * where_zeros
            # further optimization to avoid computing ProdWithoutZeros
            # if the incoming gradient is 0
            where_gz_not_zero = T.neq(gz, 0.0)
            # only take ProdWithoutZeros for the groups with single zeros
            # with non-null incoming gradient
            where_to_take_prod_without_zeros = \
                        groups_with_single_zero * where_gz_not_zero
            # preprocess the original input so that we set 0 everywhere
            # except for groups that contain a single zero, to avoid computing
            # multiplications on other groups
            prod_without_zeros_in = where_to_take_prod_without_zeros * prod_in
            # TODO: put lazy switch here, if it'd work
            # this is pretty efficient already (no multiplication if 0), but
            # it'd be even better if we had a lazy if per element
            prod_without_zeros = ProdWithoutZeros(axis=self.axis)(prod_without_zeros_in)
            prod_without_zeros = prod_without_zeros.dimshuffle(new_dims)

            groups_without_zeros = T.eq(sum_where_zeros, 0).dimshuffle(new_dims)

            final_grad = T.switch(groups_without_zeros, grad_case_without_zeros,
                            T.switch(where_single_zero, prod_without_zeros, 0.0) * gz)

            return [final_grad]

    def __str__(self):
        if self.axis is None:
            return "Prod"
        else:
            return "Prod{%s}" % ", ".join(map(str, self.axis))

    def c_code_cache_version(self):
        return ()

class MulWithoutZeros(scalar.BinaryScalarOp):
    # "identity" here is zero, as in Reduce we don't want to start
    # with reducing (1, something_else): this leads to the erronous
    # case where a vector of zeros is reduced by binary reductions
    # of (1, 0), which always ends up as 1 (ie. the result for
    # the c version, for the product of [0,0,0], is 1.0)

    identity = 0.
    commutative = True
    associative = True
    def impl(self, x, y):
        if x == 0:
            return y
        if y == 0:
            return x
        return x*y

    def c_code(self, node, name, inp, out, sub):
        x, y = inp
        z, = out
        return ("%(z)s = ((%(x)s == 0) ? (%(y)s) : " + \
                    "((%(y)s == 0) ? (%(x)s) : ((%(y)s)*(%(x)s))) );") % locals()

    def c_code_cache_version(self):
        return (1,)
mul_without_zeros = MulWithoutZeros(scalar.upcast_out, name = 'mul_without_zeros')

class ProdWithoutZeros(CAReduceDtype):
    def __init__(self, axis=None, dtype=None):
        CAReduceDtype.__init__(self, mul_without_zeros, axis=axis, dtype=dtype)

    def __str__(self):
        if self.axis is None:
            return "ProdWithoutZeros"
        else:
            return "ProdWithoutZeros{%s}" % ", ".join(map(str, self.axis))

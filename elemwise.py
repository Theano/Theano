
import elemwise_cgen as cgen

import numpy
from gof import Op, Apply
import scalar
from scalar import Scalar
import gof
from gof.python25 import all


# tensor depends on elemwise to provide definitions for several ops
# but elemwise needs to make Tensor instances, so we have these as
# placeholders and the tensor module fills them
def as_tensor(data):
    raise Exception("Circular dependencies prevent using this here. import tensor before elemwise")

def Tensor(*inputs, **kwargs):
    raise Exception("Circular dependencies prevent using this here. import tensor before elemwise")

def TensorResult(*inputs, **kwargs):
    raise Exception("Circular dependencies prevent using this here. import tensor before elemwise")

def TensorConstant(*inputs, **kwargs):
    raise Exception("Circular dependencies prevent using this here. import tensor before elemwise")


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

        # list of dimensions of the input to drop
        self.drop = []
        i2j = {} # this maps i before dropping dimensions to j after dropping dimensions so self.shuffle can be set properly later on
        j = 0
        for i, b in enumerate(input_broadcastable):
            if i not in new_order:
                # we want to drop this dimension because it's not a value in new_order
                if b == 1:
                    self.drop.append(i)
                else:
                    # we cannot drop non-broadcastable dimensions
                    raise NotImplementedError("You cannot drop a non-broadcastable dimension.")
            else:
                i2j[i] = j
                j += 1

        # transposition of non-broadcastable dimensions
        self.shuffle = [i2j[x] for x in new_order if x != 'x']

        # list of dimensions of the output that are broadcastable and were not in the original input
        self.augment = [i for i, x in enumerate(new_order) if x == 'x']

        if self.inplace:
            self.view_map = {0: [0]}

    def make_node(self, input):
        ib = tuple(input.type.broadcastable)
        if not ib == self.input_broadcastable:
            raise TypeError("The number of dimensions and/or broadcastable pattern of the input is incorrect for this op. Expected %s, got %s." % (self.input_broadcastable, ib))
        ob = []
        for value in self.new_order:
            if value == 'x':
                ob.append(True)
            else:
                ob.append(ib[value])

        output = Tensor(dtype = input.type.dtype,
                        broadcastable = ob).make_result()
        return Apply(self, [input], [output])

    def __eq__(self, other):
        # it's probably not necessary to compare input_broadcastable
        return type(self) == type(other) \
            and self.inplace == other.inplace \
            and self.new_order == other.new_order \
            and self.input_broadcastable == other.input_broadcastable

    def __hash__(self):
        return hash(self.inplace) ^ hash(self.new_order) ^ hash(self.input_broadcastable)

    def __str__(self):
        if self.inplace:
            return "InplaceDimShuffle{%s}" % ",".join(str(x) for x in self.new_order)
        else:
            return "DimShuffle{%s}" % ",".join(str(x) for x in self.new_order)

    def perform(self, node, (input, ), (storage, )):
        # drop
        res = input
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

        storage[0] = res

    def grad(self, (x, ), (gz, )):
        gz = as_tensor(gz)
        grad_order = ['x'] * len(x.type.broadcastable)
        for i, v in enumerate(self.new_order):
            if v != 'x':
                grad_order[v] = i
        return DimShuffle(gz.type.broadcastable, grad_order)(gz),


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
      Elemwise(div)(rand(10, 5), rand(10, 1)) # same but along the second dimension
      Elemwise(div)(rand(1, 5), rand(10, 1)) # the output has size (10, 5)
      Elemwise(log)(rand(3, 4, 5))
    """

    def __init__(self, scalar_op, inplace_pattern = {}, name = None):
        """
        Usage: Elemwise(scalar_op, inplace_pattern = {})

        * scalar_op: an instance of a subclass of scalar.ScalarOp which works uniquely on
                     scalars
        * inplace_pattern: a dictionary that maps the index of an output to the
                           index of an input so the output is calculated inplace using
                           the input's storage.
        """
        self.name = name
        self.scalar_op = scalar_op
        self.inplace_pattern = inplace_pattern
        self.destroy_map = dict((o, [i]) for o, i in inplace_pattern.items())
        if scalar_op.nin > 0:
            self.ufunc = numpy.frompyfunc(scalar_op.impl, scalar_op.nin, scalar_op.nout)
        else:
            self.ufunc = None

    def make_node(self, *inputs):
        """
        If the inputs have different number of dimensions, their shape
        is left-completed to the greatest number of dimensions with 1s
        using DimShuffle.
        """

        inputs = map(as_tensor, inputs)
        shadow = self.scalar_op.make_node(*[Scalar(dtype = t.type.dtype)() for t in inputs])

        target_length = max([input.type.ndim for input in inputs])

        args = []
        for input in inputs:
            length = input.type.ndim
            difference = target_length - length
            if not difference:
                args.append(input)
            else:
                # TODO: use LComplete instead
                args.append(DimShuffle(input.type.broadcastable, ['x']*difference + range(length), inplace = True)(input))
        inputs = args

#         # Following conditions should always be true?
#         try:
#             assert len(set([len(input.type.broadcastable) for input in inputs])) == 1
#         except (AssertionError, AttributeError):
#             raise TypeError("All inputs to a Broadcast subclass must be Tensor instances and their broadcastable fields must all have the same length.", inputs)

        out_broadcastables = [[all(bcast) for bcast in zip(*[input.type.broadcastable for input in inputs])]] * shadow.nout
        inplace_pattern = self.inplace_pattern
        if inplace_pattern:
            for overwriter, overwritten in inplace_pattern.items():
                for ob, ib in zip(out_broadcastables[overwriter], inputs[overwritten].type.broadcastable):
                    if ib and not ob:
                        raise ValueError("Operation cannot be done inplace on an input with broadcasted dimensions.")
        out_dtypes = [o.type.dtype for o in shadow.outputs]
        if any(inputs[i].type.dtype != out_dtypes[o] for i, o in inplace_pattern.items()):
            raise TypeError("Cannot do an inplace operation on incompatible data types.", [i.type.dtype for i in inputs], out_dtypes)
        outputs = [Tensor(dtype = dtype, broadcastable = broadcastable)() for dtype, broadcastable in zip(out_dtypes, out_broadcastables)]
        return Apply(self, inputs, outputs)

    def __eq__(self, other):
        return type(self) == type(other) and self.scalar_op == other.scalar_op and self.inplace_pattern == other.inplace_pattern

    def __hash__(self):
        return hash(self.scalar_op) ^ hash(tuple(self.inplace_pattern.items()))

    def __str__(self):
        if self.name is None:
            if self.inplace_pattern:
                return "Elemwise{%s}%s" % (self.scalar_op, str(self.inplace_pattern))
            else:
                return "Elemwise{%s}" % (self.scalar_op)
        else:
            return self.name

    def grad(self, inputs, ograds):
        ograds = map(as_tensor, ograds) # this shouldn't be necessary...
        scalar_inputs = [Scalar(dtype = t.type.dtype)() for t in inputs]
        scalar_ograds = [Scalar(dtype = ograd.type.dtype)() for ograd in ograds]
        scalar_igrads = self.scalar_op.grad(scalar_inputs, scalar_ograds)
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
                # an equivalent Tensor of size 1 and proper number of dimensions
                res = TensorConstant(Tensor(dtype = r.type.dtype,
                                            broadcastable = ()),
                                     numpy.asarray(r.data)) # .reshape(b)
                return DimShuffle((), ['x']*nd, inplace = True)(res)
            new_r = Elemwise(node.op, {})(*[transform(input) for input in node.inputs])
            return new_r
        ret = []
        for scalar_igrad, input in zip(scalar_igrads, inputs):
            if scalar_igrad is None:
                # undefined gradient
                ret.append(None)
                continue
            r = transform(scalar_igrad)

            # list of all the dimensions that are broadcastable for that input so we
            # can sum over them
            # todo: only count dimensions that were effectively broadcasted
            to_sum = [i for i, bcast in enumerate(input.type.broadcastable) if bcast]

            if to_sum:
                shuffle = []
                j = 0
                for bcast in input.type.broadcastable:
                    if bcast == 1:
                        shuffle.append('x')
                    else:
                        shuffle.append(j)
                        j += 1
                sr = Sum(axis = to_sum)(r)
                sr = DimShuffle(sr.type.broadcastable, shuffle)(sr)
                ret.append(sr)
            else:
                ret.append(r)
        return ret

    def perform(self, node, inputs, output_storage):
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
        # the second calling form is used because in certain versions of numpy
        # the first (faster) version leads to segfaults
        ufunc_args = inputs # + output_storage
        ufunc = self.ufunc or numpy.frompyfunc(self.scalar_op.impl, len(inputs), self.scalar_op.nout)
        results = ufunc(*ufunc_args)
        if ufunc.nout == 1: results = [results]
        for result, storage in zip(results, output_storage):
            if storage[0].shape:
                storage[0][:] = result
            else:
                storage[0].itemset(result)
        # the following should be used instead of the previous loop, unfortunately it tends to segfault
        # self.ufunc(*(ufunc_args+[s[0] for s in output_storage]))

    def _c_all(self, node, name, inames, onames, sub):
        _inames = inames
        _onames = onames

        inames = gof.utils.uniq(inames)
        inputs = gof.utils.uniq(node.inputs)

        defines = ""
        undefs = ""
        dmap = dict([(node.outputs[i], [node.inputs[o]]) for i, o in self.inplace_pattern.items()])

        idtypes = [input.type.dtype_specs()[1] for input in inputs]

        real = zip(*[(r, s, r.type.dtype_specs()[1])
                     for r, s in zip(node.outputs, onames) if r not in dmap])
        if real:
            real_outputs, real_onames, real_odtypes = real
        else:
            real_outputs, real_onames, real_odtypes = [], [], []

        aliased = zip(*[(r, s)
                        for (r, s) in zip(node.outputs, onames) if r in dmap])
        if aliased:
            aliased_outputs, aliased_onames = aliased
        else:
            aliased_outputs, aliased_onames = [], []

        orders = [[x and 'x' or i for i, x in enumerate(input.type.broadcastable)] for input in inputs]
        nnested = len(orders[0])
        sub = dict(sub)
        for i, (input, iname) in enumerate(zip(inputs, inames)):
            sub['lv%i' % i] = iname
        decl = cgen.make_declare(orders, idtypes, sub)
        checks = cgen.make_checks(orders, idtypes, sub)

        alloc = ""
        for output, oname, odtype in zip(real_outputs, real_onames, real_odtypes):
            i += 1
            sub['lv%i' % i] = oname
            sub['olv'] = oname
            alloc += cgen.make_declare([range(nnested)], [odtype], dict(sub, lv0 = oname))
            alloc += cgen.make_alloc(orders, odtype, sub)
            alloc += cgen.make_checks([range(nnested)], [odtype], dict(sub, lv0 = oname))

        for output, oname in zip(aliased_outputs, aliased_onames):
            iname = inames[inputs.index(dmap[output][0])]
            alloc += """
            if (%(oname)s) {
                Py_XDECREF(%(oname)s);
            }
            %(oname)s = %(iname)s;
            Py_XINCREF(%(oname)s);
            """ % locals()
            defines += "#define %(oname)s_i %(iname)s_i" % locals()
            undefs += "#undef %(oname)s_i" % locals()

        task_code = self.scalar_op.c_code(Apply(self.scalar_op,
                                                [Scalar(dtype = input.type.dtype)() for input in node.inputs],
                                                [Scalar(dtype = output.type.dtype)() for input in node.outputs]),
                                          None,
                                          ["%s_i" % s for s in _inames],
                                          ["%s_i" % s for s in onames],
                                          sub)
        task_decl = "".join(["%(dtype)s& %(name)s_i = *%(name)s_iter;\n" % locals() for name, dtype in zip(inames + list(real_onames), idtypes + list(real_odtypes))])
        code = """
        {
            %(defines)s
            %(task_decl)s
            %(task_code)s
            %(undefs)s
        }
        """ % locals()
        if nnested:
            all_code = [("", "")] * (nnested - 1) + [("", code)] + [""]
        else:
            all_code = [code]
        loop = cgen.make_loop(orders + [range(nnested)] * len(real_onames), idtypes + list(real_odtypes), all_code, sub)
        return decl, checks, alloc, loop

    def c_code(self, node, name, inames, onames, sub):
        code = "\n".join(self._c_all(node, name, inames, onames, sub))
        return code



################
### CAReduce ###
################

class CAReduce(Op):
    """
    Reduces a scalar operation along the specified axis(es).

    The output will have the same shape as the input minus the reduced
    dimensions. It will contain the result of accumulating all values
    over the reduced dimensions using the specified scalar op.

    Examples:
     CAReduce(add) -> sum
     CAReduce(mul) -> product
     CAReduce(_or) -> any # not lazy
     CAReduce(_and) -> all # not lazy

    In order to (eventually) optimize memory usage patterns,
    L{CAReduce} makes zero guarantees on the order in which it
    iterates over the dimensions and the elements of the
    array(s). Therefore, to ensure consistent results, the scalar
    operation represented by the reduction must be both commutative
    and associative (eg add, multiply, binary or/and/xor - but not
    subtract, divide or power).
    """

    def __init__(self, scalar_op, axis = None):
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
        if isinstance(axis, int):
            self.axis = [axis]
        else:
            self.axis = axis
        self.ufunc = numpy.frompyfunc(scalar_op.impl, 2, 1)

    def make_node(self, input):
        input = as_tensor(input)
        axis = self.axis
        if axis is None:
            axis = range(len(input.type.broadcastable))
        output = Tensor(dtype = input.type.dtype,
                        broadcastable = [x for i, x in enumerate(input.type.broadcastable) if i not in axis])()
        return Apply(self, [input], [output])

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

    def perform(self, node, (input, ), (output, )):
        axis = self.axis
        if axis is None:
            axis = range(input.ndim)
        result = input
        to_reduce = reversed(sorted(axis))
        if to_reduce:
            for dimension in to_reduce:
                result = self.ufunc.reduce(result, dimension)
            output[0] = numpy.asarray(result, dtype = node.outputs[0].type.dtype)
        else:
            output[0] = numpy.copy(result)

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

        if axis == ():
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

        task0_decl = "%(dtype)s& %(name)s_i = *%(name)s_iter;\n%(name)s_i = %(identity)s;" % dict(dtype = odtype,
                                                                                                  name = onames[0],
                                                                                                  identity = self.scalar_op.identity)

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


class Sum(CAReduce):
    """
    Sums all the values of a tensor along the specified axis(es).

    Equivalent to CAReduce(scalar.add, axis = axis), with the
    difference that this defines the gradient of sum wrt its tensor
    input.
    """
    def __init__(self, axis = None):
        CAReduce.__init__(self, scalar.add, axis)

    def grad(self, (x, ), (gz, )):
        gz = as_tensor(gz)
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
        return Elemwise(scalar.second)(x, DimShuffle(gz.type.broadcastable, new_dims)(gz)),

    def __str__(self):
        if self.axis is None:
            return "Sum"
        else:
            return "Sum{%s}" % ", ".join(map(str, self.axis))



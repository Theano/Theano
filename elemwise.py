
import elemwise_cgen as cgen

import numpy
from gof import Op, Viewer, Destroyer
#from base_tensor import BaseTensor as Tensor
import scalar
from scalar import upcast, Scalar
import gof
from gof.python25 import all


def astensor(data):
    raise Exception("Circular dependencies prevent using this here. import tensor before elemwise")

def Tensor(*inputs, **kwargs):
    raise Exception("Circular dependencies prevent using this here. import tensor before elemwise")


##################
### DimShuffle ###
##################

class DimShuffle(Op, Viewer):
    """
    Usage: DimShuffle(input, new_order, inplace = True)

    * input: a Tensor instance
    * new_order: a list representing the relationship between the
                 input's dimensions and the output's dimensions. Each
                 element of the list can either be an index or 'x'.
    * inplace: if True, the output will be a view of the input.
               If False, the output will be a copy of the input.

    If j = new_order[i] is an index, the output's ith dimension
      will be the input's jth dimension.
    If new_order[i] is 'x', the output's ith dimension will
      be 1 and Broadcast operations will be allowed to do broadcasting
      over that dimension.

    If input.broadcastable[i] == False then i must be found in new_order.
    Broadcastable dimensions, on the other hand, can be discarded.

    Examples:
      # t<n> represents a n-d tensor
      DimShuffle(t2, [0, 1]) -> identity
      DimShuffle(t2, [1, 0]) -> inverts the first and second dimensions
      DimShuffle(t1, ['x', 0]) -> make a row out of a 1d vector
      DimShuffle(t1, [0, 'x']) -> make a column out of a 1d vector
      DimShuffle(t3, [2, 0, 1]) -> like doing t3.transpose((2, 0, 1)) in numpy
      DimShuffle(t2, [0, 'x', 1]) -> like doing t3.reshape((t3.shape[0], 1, t3.shape[1])) in numpy
      DimShuffle(t2, [1, 'x', 0]) -> like doing t3.T.reshape((t3.shape[0], 1, t3.shape[1])) in numpy
    """
    
    def __init__(self, input, new_order, inplace = True):

        input = astensor(input)

        ib = input.broadcastable
        ob = []
        for value in new_order:
            if value == 'x':
                self.has_x = True
                ob.append(1)
            else:
                ob.append(ib[value])
        
        output = Tensor(dtype = input.dtype,
                        broadcastable = ob)

        self.new_order = new_order
        self.inputs = input,
        self.outputs = output,

        self.inplace = inplace

        # list of dimensions of the input to drop
        self.drop = []
        i2j = {} # this maps i before dropping dimensions to j after dropping dimensions so self.shuffle can be set properly later on
        j = 0
        for i, b in enumerate(ib):
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

    def clone_with_new_inputs(self, *new_inputs):
        return DimShuffle(new_inputs[0], self.new_order, self.inplace)
    
    def view_map(self):
        if self.inplace:
            return {self.outputs[0]: [self.inputs[0]]}
        else:
            return {}

    def desc(self):
        return (self.__class__, tuple(self.new_order))

    def strdesc(self):
        return "DimShuffle{%s}" % "".join(str(x) for x in self.new_order)

    def perform(self):
        # drop
        res = self.inputs[0].data
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

        self.outputs[0].data = res

    def grad(self, (x, ), (gz, )):
        grad_order = ['x'] * len(self.inputs[0].broadcastable)
        for i, x in enumerate(self.new_order):
            if x != 'x':
                grad_order[x] = i
        return DimShuffle(gz, grad_order).out,
        
    def __str__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, str(self.inputs[0]), self.new_order)



#################
### Broadcast ###
#################

class Broadcast(Op, Destroyer):
    """
    Generalizes a scalar op to tensors.
    
    Usage: Broadcast(scalar_opclass, inputs, inplace_pattern = {})

    * scalar_opclass: a class that extends scalar.ScalarOp, works uniquely on
                      scalars and can be instantiated from the list of its inputs
    * inputs: a list of Tensor instances
    * inplace_pattern: a dictionary that maps the index of an output to the
                       index of an input so the output is calculated inplace using
                       the input's storage.

    All the inputs must have the same number of dimensions. When the
    Op is performed, for each dimension, each input's size for that
    dimension must be the same. As a special case, it can also be 1
    but only if the input's broadcastable flag is True for that
    dimension. In that case, the tensor is (virtually) replicated
    along that dimension to match the size of the others.

    The dtypes of the outputs mirror those of the scalar Op that is
    being generalized to tensors. However, if the calculations for an
    output are done inplace on an input, it will keep the same dtype
    as the input (in a nutshell, int + float -> float but int += float -> int)

    Examples:
      Broadcast(Add, rand(10, 5), rand(10, 5), {0 : 0}) # this does input0 += input1
      Broadcast(Add, rand(10, 5), rand(10, 5), {0 : 1}) # this does input1 += input0
      Broadcast(Mul, rand(10, 5), rand(1, 5)) # the second input is completed along the first dimension to match the first input
      Broadcast(Div, rand(10, 5), rand(10, 1)) # same but along the second dimension
      Broadcast(Div, rand(1, 5), rand(10, 1)) # the output has size (10, 5)
      Broadcast(Log, rand(3, 4, 5))
    """

    def __init__(self, scalar_opclass, inputs, inplace_pattern = {}):

        inputs = map(astensor, inputs)
        
        try:
            assert len(set([len(input.broadcastable) for input in inputs])) == 1
        except (AssertionError, AttributeError):
            raise TypeError("All inputs to a Broadcast subclass must be Tensor instances and their broadcastable fields must all have the same length.", self.__class__)

        # self.shadow is an instance of scalar_opclass used to get values for all the properties we need (dtypes, gradient, etc.)
        self.shadow = scalar_opclass(*[Scalar(dtype = t.dtype) for t in inputs])
        
        self.nin = self.shadow.nin
        self.nout = self.shadow.nout
        out_broadcastables = [[1*all(bcast) for bcast in zip(*[input.broadcastable for input in inputs])]] * self.nout

        if inplace_pattern:
            for overwriter, overwritten in inplace_pattern.items():
                for ob, ib in zip(out_broadcastables[overwriter], inputs[overwritten].broadcastable):
                    if ib and not ob:
                        raise ValueError("Operation cannot be done inplace on an input with broadcasted dimensions.")

        out_dtypes = [t.dtype for t in self.shadow.outputs]
        def get_dtype(i):
            # If an operation is done inplace, the dtype of the output
            # will be the same as the dtype of the input it overwrites
            # eg int + float -> float, but int += float -> int
            input_idx = inplace_pattern.get(i, None)
            if input_idx is not None:
                return inputs[input_idx].dtype
            else:
                return out_dtypes[i]
        out_dtypes = map(get_dtype, xrange(self.nout))
        self.inputs = inputs
        self.outputs = [Tensor(dtype = dtype, broadcastable = broadcastable) for dtype, broadcastable in zip(out_dtypes, out_broadcastables)]
        self.inplace_pattern = inplace_pattern
        self.scalar_opclass = scalar_opclass
        self.ufunc = numpy.frompyfunc(self.shadow.impl, self.shadow.nin, self.shadow.nout)

    def clone_with_new_inputs(self, *new_inputs):
        return Broadcast(self.scalar_opclass, new_inputs, self.inplace_pattern)

    def desc(self):
        return (Broadcast, self.scalar_opclass, tuple(self.inplace_pattern.items()))

    def strdesc(self):
        if self.inplace_pattern:
            return "Broadcast{%s}%s" % (self.shadow.strdesc(), str(self.inplace_pattern))
        else:
            return "Broadcast{%s}" % (self.shadow.strdesc())

    def destroy_map(self):
        ret = {}
        for key, value in self.inplace_pattern.items():
            ret[self.outputs[key]] = [self.inputs[value]]
        return ret

    def grad(self, inputs, ograds):
        shadow = self.shadow
        scalar_ograds = [Scalar(dtype = ograd.dtype) for ograd in ograds]
        scalar_igrads = shadow.grad(shadow.inputs, scalar_ograds)
        nd = len(inputs[0].broadcastable) # this is the same for everyone
        def transform(r):
            # From a graph of ScalarOps, make a graph of Broadcast ops.
            if r in shadow.inputs:
                return inputs[shadow.inputs.index(r)]
            if r in scalar_ograds:
                return ograds[scalar_ograds.index(r)]
            op = r.owner
            if op is None:
                # the gradient contains a constant, translate it as
                # an equivalent Tensor of size 1 and proper number of dimensions
                b = [1] * nd
                res = astensor(numpy.asarray(r.data).reshape(b),
                               broadcastable = b)
                return res
            op_class = op.__class__
            bcasted = Broadcast(op_class, [transform(input) for input in op.inputs], {}).out
            return bcasted
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
            to_sum = [i for i, bcast in enumerate(input.broadcastable) if bcast]

            if to_sum:
                shuffle = []
                j = 0
                for bcast in input.broadcastable:
                    if bcast == 1:
                        shuffle.append('x')
                    else:
                        shuffle.append(j)
                        j += 1
                sr = Sum(r, axis = to_sum).out
                sr = DimShuffle(sr, shuffle).out
                ret.append(sr)
            else:
                ret.append(r)
        return ret

    def perform(self):
        output_storage = []
        if not self.inplace_pattern:
            for output in self.outputs:
                odat = output.data
                shape = [max(values) for values in zip(*[input.data.shape for input in self.inputs])]
                if odat is not None:
                    # reuse storage if we can
                    odat.resize(shape, refcheck = 0)
                else:
                    odat = numpy.ndarray(shape, dtype = output.dtype)
                output_storage.append(odat)
                output.data = odat
        else:
            for i, output in enumerate(self.outputs):
                if i in self.inplace_pattern:
                    odat = self.inputs[self.inplace_pattern[i]].data
                else:
                    odat = output.data
                    shape = [max(values) for values in zip(*[input.data.shape for input in self.inputs])]
                    if odat is not None:
                        odat.resize(shape)
                    else:
                        odat = numpy.ndarray(shape, dtype = output.dtype)
                output_storage.append(odat)
                output.data = odat
        self.ufunc(*([input.data for input in self.inputs] + output_storage))

    def _c_all(self, inames, onames, sub):
        defines = ""
        undefs = ""
        dmap = self.destroy_map()

        idtypes = [input.dtype_specs()[1] for input in self.inputs]

        real = zip(*[(r, s, r.dtype_specs()[1])
                     for r, s in zip(self.outputs, onames) if r not in dmap])
        if real:
            real_outputs, real_onames, real_odtypes = real
        else:
            real_outputs, real_onames, real_odtypes = [], [], []

        aliased = zip(*[(r, s)
                        for (r, s) in zip(self.outputs, onames) if r in dmap])
        if aliased:
            aliased_outputs, aliased_onames = aliased
        else:
            aliased_outputs, aliased_onames = [], []
        
        orders = [[x and 'x' or i for i, x in enumerate(input.broadcastable)] for input in self.inputs]
        nnested = len(orders[0])
        sub = dict(sub)
        for i, (input, iname) in enumerate(zip(self.inputs, inames)):
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
            iname = inames[self.inputs.index(dmap[output][0])]
            alloc += """
            if (%(oname)s) {
                Py_XDECREF(%(oname)s);
            }
            %(oname)s = %(iname)s;
            Py_XINCREF(%(oname)s);
            """ % locals()
            defines += "#define %(oname)s_i %(iname)s_i" % locals()
            undefs += "#undef %(oname)s_i" % locals()
        
        task_code = self.shadow.c_code(["%s_i" % s for s in inames],
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
        
    def c_code(self, inames, onames, sub):
        code = "\n".join(self._c_all(inames, onames, sub))
        return code
         


def make_broadcast(scalar_opclass, inplace_pattern = {}, name = None):
    if name is None:
        name = "Tensor" + scalar_opclass.__name__
        
    scalar_name = scalar_opclass.__name__
    previous_doc = Broadcast.__doc__

    scalar_doc = scalar_opclass.__doc__ or ""
    if scalar_doc:
        scalar_doc = """
    %(scalar_name)s documentation:
        %(scalar_doc)s
        """ % locals()

    doc = """
    Usage: %(name)s(*inputs)
    Equivalent to: Broadcast(scalar.%(scalar_name)s, inputs, %(inplace_pattern)s)

    Performs Scalar %(scalar_name)s on each element of the
    input tensors.
    %(scalar_doc)s
    Documention for Broadcast:
    ==================================================
    %(previous_doc)s
    ==================================================
    """ % locals()

    class New(Broadcast):
        __doc__ = doc
        def __init__(self, *inputs):
            Broadcast.__init__(self, scalar_opclass, inputs, inplace_pattern)
        def clone_with_new_inputs(self, *new_inputs):
            return New(*new_inputs)
        @classmethod
        def desc(cls):
            return (Broadcast, scalar_opclass, tuple(inplace_pattern.items()))
    New.__name__ = name
    return New

def wrap_broadcast(op):
    def instantiate(*inputs):
        inputs = map(astensor, inputs)
        
        target_length = max([len(input.broadcastable) for input in inputs])
        args = []
        for input in inputs:
            length = len(input.broadcastable)
            difference = target_length - length
            if not difference:
                args.append(input)
            else:
                args.append(DimShuffle(input, ['x']*difference + range(length)).out)
        return op(*args)
    instantiate.__name__ = "instantiate{%s}" % op.__name__
    instantiate.__doc__ = op.__doc__
    return instantiate



################
### CAReduce ###
################

class CAReduce(Op):
    """
    Usage: CAReduce(scalar_opclass, inputs, dimensions_to_reduce = None)

    * scalar_opclass: a binary scalar op with only one output.
                      It will be instantiated as such:
                      scalar_opclass.__init__([Scalar(t.dtype) for t in inputs])
                      It must be commutative and associative.
    * inputs: list of Tensor instances
    * dimensions_to_reduce: list of dimensions that we want to reduce
                            if None, all dimensions are reduced

    The output will have the same shape as the input minus the reduced
    dimensions. It will contain the result of accumulating all values
    over the reduced dimensions using the specified scalar op.

    Examples:
     CAReduce(Add, inputs) -> sum(inputs)
     CAReduce(Mul, inputs) -> product(inputs)
     CAReduce(Or, inputs) -> any(inputs) # not lazy
     CAReduce(And, inputs) -> all(inputs) # not lazy
     CAReduce(Xor, inputs) -> sum(inputs != 0) % 2

    In order to optimize memory usage patterns, L{CAReduce} makes zero
    guarantees on the order in which it iterates over the dimensions
    and the elements of the array(s). Therefore, to ensure consistent
    results, the scalar operation represented by the reduction must be
    both commutative and associative (eg add, multiply, binary
    or/and/xor - but not subtract, divide or power).
    """
    
    def __init__(self, scalar_opclass, inputs, dimensions_to_reduce = None):
        inputs = map(astensor, inputs)

        self.shadow = scalar_opclass(*[Scalar(dtype = inputs[0].dtype) for i in xrange(len(inputs) + 1)])
        
        if self.shadow.nin != 2 or self.shadow.nout != 1:
            raise NotImplementedError("CAReduce only supports binary functions with a single output.")
        if len(inputs) != 1:
            raise TypeError("Only one argument expected.")
        if dimensions_to_reduce is None:
            dimensions_to_reduce = range(len(inputs[0].broadcastable))

        self.inputs = inputs
        self.outputs = [Tensor(dtype = inputs[0].dtype,
                               broadcastable = [x for i, x in enumerate(inputs[0].broadcastable) if i not in dimensions_to_reduce])]

        self.dimensions_to_reduce = dimensions_to_reduce
        self.scalar_opclass = scalar_opclass
        self.ufunc = numpy.frompyfunc(self.shadow.impl, self.shadow.nin, self.shadow.nout)

    def desc(self):
        return (self.__class__, self.scalar_opclass, tuple(self.dimensions_to_reduce))
        
    def strdesc(self):
        if set(self.dimensions_to_reduce) != set(xrange(len(self.inputs[0].broadcastable))):
            return "Reduce{%s}{%s}" % (self.scalar_opclass.__name__, "".join(str(x) for x in self.dimensions_to_reduce))
        else:
            return "Reduce{%s}" % self.scalar_opclass.__name__
        
    def clone_with_new_inputs(self, *new_inputs):
        return CAReduce(self.scalar_opclass, new_inputs, self.dimensions_to_reduce)
        
    def perform(self):
        result = self.inputs[0].data
        to_reduce = reversed(sorted(self.dimensions_to_reduce))
        if to_reduce:
            for dimension in to_reduce:
                result = self.ufunc.reduce(result, dimension)
            self.outputs[0].data = result
        else:
            self.outputs[0].data = numpy.copy(result)

    def _c_all(self, inames, onames, sub):

        input = self.inputs[0]
        output = self.outputs[0]

        iname = inames[0]
        oname = onames[0]
        
        idtype = input.dtype_specs()[1]
        odtype = output.dtype_specs()[1]

        tosum = self.dimensions_to_reduce

        if tosum == ():
            return Broadcast(scalar.Identity, (input, ))._c_all(inames, onames, sub)

        order1 = [i for i in xrange(len(input.broadcastable)) if i not in tosum]
        order = order1 + list(tosum)
        
        nnested = len(order1)

        sub = dict(sub)
        for i, (input, iname) in enumerate(zip(self.inputs, inames)):
            sub['lv%i' % i] = iname

        decl = cgen.make_declare([order], [idtype], sub)
        checks = cgen.make_checks([order], [idtype], sub)

        alloc = ""
        i += 1
        sub['lv%i' % i] = oname
        sub['olv'] = oname
        alloc += cgen.make_declare([range(nnested) + ['x'] * len(tosum)], [odtype], dict(sub, lv0 = oname))
        alloc += cgen.make_alloc([order1], odtype, sub)
        alloc += cgen.make_checks([range(nnested) + ['x'] * len(tosum)], [odtype], dict(sub, lv0 = oname))

        task0_decl = "%(dtype)s& %(name)s_i = *%(name)s_iter;\n%(name)s_i = %(identity)s;" % dict(dtype = odtype,
                                                                                                  name = onames[0],
                                                                                                  identity = self.shadow.identity)

        task1_decl = "%(dtype)s& %(name)s_i = *%(name)s_iter;\n" % dict(dtype = idtype, name = inames[0])
        task1_code = self.shadow.c_code(["%s_i" % onames[0], "%s_i" % inames[0]],
                                        ["%s_i" % onames[0]],
                                        sub)
        code1 = """
        {
            %(task1_decl)s
            %(task1_code)s
        }
        """ % locals()

        if len(tosum) == 1:
            all_code = [("", "")] * nnested + [(task0_decl, code1), ""]
        else:
            all_code = [("", "")] * nnested + [(task0_decl, "")] + [("", "")] * (len(tosum) - 2) + [("", code1), ""]
        
#         if nnested:
#             all_code = [("", "")] * (nnested - 1) + [("", code)] + [""]
#         else:
#             all_code = [code]

#        print [order, range(nnested) + ['x'] * len(tosum)]
        
        loop = cgen.make_loop([order, range(nnested) + ['x'] * len(tosum)], [idtype, odtype], all_code, sub)
        return decl, checks, alloc, loop
        
    def c_code(self, inames, onames, sub):
        code = "\n".join(self._c_all(inames, onames, sub))
#        print code
        return code

    def __str__(self):
        input = self.inputs[0]
        if len(input.broadcastable) == len(self.dimensions_to_reduce):
            return "%s:%s(%s)" % (self.__class__.__name__,
                                  self.scalar_opclass.__name__,
                                  str(input))
        else:
            return "%s:%s(%s, axis = %s)" % (self.__class__.__name__,
                                             self.scalar_opclass.__name__,
                                             str(input),
                                             self.dimensions_to_reduce)
        
        

def make_reduce(scalar_opclass, name = None):
    if getattr(scalar_opclass, 'commutative', False) \
            and getattr(scalar_opclass, 'associative', False):
        reducer = CAReduce
    else:
        raise NotImplementedError("The scalar op class to reduce must be commutative and associative.")

    scalar_name = scalar_opclass.__name__
    if name is None:
        name = "Reduce" + scalar_name
    previous_doc = reducer.__doc__

    doc = """
    Usage: %(name)s(input, axis)
    Equivalent to: CAReduce(%(scalar_name)s, input, axis)

    Reduces the input over the specified axis.

    Documention for CAReduce:
    ==================================================
    %(previous_doc)s
    ==================================================
    """ % locals()

    class New(reducer):
        __doc__ = doc
        def __init__(self, *inputs, **kwargs):
            reducer.__init__(self, scalar_opclass, inputs, kwargs.get('axis', None))
        def clone_with_new_inputs(self, *new_inputs):
            return New(*new_inputs, **dict(axis = self.dimensions_to_reduce))
        def __str__(self):
            input = self.inputs[0]
            if len(input.broadcastable) == len(self.dimensions_to_reduce):
                return "%s(%s)" % (self.__class__.__name__,
                                   str(input))
            else:
                return "%s(%s, axis = %s)" % (self.__class__.__name__,
                                              str(input),
                                              self.dimensions_to_reduce)
    New.__name__ = name
    return New

_Sum = make_reduce(scalar.Add, '_Sum')
class Sum(_Sum):
    __doc__ = _Sum.__doc__
    def grad(self, (x, ), (gz, )):
        if self.dimensions_to_reduce == ():
            return gz,
        new_dims = []
        i = 0
        for j, _ in enumerate(x.broadcastable):
            if j in self.dimensions_to_reduce:
                new_dims.append('x')
            else:
                new_dims.append(i)
                i += 1
        return Broadcast(scalar.Second, (x, DimShuffle(gz, new_dims).out)).out, 


def reduce(op):
    if getattr(op, 'commutative', True) and getattr(op, 'associative', True):
        reducer = CAReduce
    else:
        raise NotImplementedError("The scalar op class to reduce must be commutative and associative.")
    def instantiate(*inputs):
        return reducer(op, inputs, dimensions_to_reduce)
    return instantiate




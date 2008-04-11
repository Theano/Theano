
import elemwise_cgen as cgen

import numpy
from gof import Op, Viewer, Destroyer
from base_tensor import BaseTensor as Tensor
import scalar
from scalar import upcast, Scalar
import gof
from gof.python25 import all

def astensor(data):
    assert isinstance(data, Tensor)
    return data


##################
### DimShuffle ###
##################

class DimShuffle(Op, Viewer):
    """
    @todo: DOCUMENTATION? --jpt
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

        self.drop = []
        self.augment = []
        i2j = {}
        j = 0
        for i, b in enumerate(ib):
            if i not in new_order:
                if b == 1:
                    self.drop.append(i)
                else:
                    raise NotImplementedError("You cannot drop a non-broadcastable dimension.")
            else:
                i2j[i] = j
                j += 1

        self.shuffle = [i2j[x] for x in new_order if x != 'x']
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
        res = self.inputs[0].data
        shape = list(res.shape)
        for drop in reversed(self.drop):
            shape.pop(drop)
        res = res.reshape(shape)
        
        res = res.transpose(self.shuffle)

        shape = list(res.shape)
        for augm in self.augment:
            shape.insert(augm, 1)
        res = res.reshape(shape)

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


class Transpose(DimShuffle):

    def __init__(self, input):
        DimShuffle.__init__(self, input, range(len(input.broadcastable)-1, -1, -1), False)

    def clone_with_new_inputs(self, *new_inputs):
        return Transpose(new_inputs[0])

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, str(self.inputs[0]))



#################
### Broadcast ###
#################

class Broadcast(Op, Destroyer):
    """
    @todo: DOCUMENTATION? --jpt
    """

    def __init__(self, scalar_opclass, inputs, inplace_pattern = {}):

        inputs = map(astensor, inputs)
        
        try:
            assert len(set([len(input.broadcastable) for input in inputs])) == 1
        except (AssertionError, AttributeError):
            raise TypeError("All inputs to a Broadcast subclass must be Tensor instances and their broadcastable fields must all have the same length.", self.__class__)
        self.nin = scalar_opclass.nin
        self.nout = scalar_opclass.nout
        out_broadcastables = [[1*all(bcast) for bcast in zip(*[input.broadcastable for input in inputs])]] * self.nout

        if inplace_pattern:
            for overwriter, overwritten in inplace_pattern.items():
                for ob, ib in zip(out_broadcastables[overwriter], inputs[overwritten].broadcastable):
                    if ib and not ob:
                        raise ValueError("Operation cannot be done inplace on an input with broadcasted dimensions.")

        upcasted = upcast(*[input.dtype for input in inputs])
        def get_dtype(i):
            input_idx = inplace_pattern.get(i, None)
            if input_idx is not None:
                return inputs[input_idx].dtype
            else:
                return upcasted
        out_dtypes = map(get_dtype, xrange(self.nout))
        self.inputs = inputs
        self.outputs = [Tensor(dtype = dtype, broadcastable = broadcastable) for dtype, broadcastable in zip(out_dtypes, out_broadcastables)]
        self.inplace_pattern = inplace_pattern
        self.scalar_opclass = scalar_opclass
        self.shadow = scalar_opclass(*[Scalar(dtype = t.dtype) for t in self.inputs])
        self.ufunc = numpy.frompyfunc(self.shadow.impl, scalar_opclass.nin, scalar_opclass.nout)

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
        def transform(r):
            if r in shadow.inputs:
                return inputs[shadow.inputs.index(r)]
            if r in scalar_ograds:
                return ograds[scalar_ograds.index(r)]
            op = r.owner
            if op is None:
                b = [1] * len(inputs[0].broadcastable)
                res = astensor(numpy.asarray(r.data).reshape(b),
                               broadcastable = b)
                return res
            op_class = op.__class__
            bcasted = Broadcast(op_class, [transform(input) for input in op.inputs], {}).out
            return bcasted
        ret = []
        for scalar_igrad, input in zip(scalar_igrads, inputs):
            r = transform(scalar_igrad)
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
    class New(Broadcast):
        def __init__(self, *inputs):
            Broadcast.__init__(self, scalar_opclass, inputs, inplace_pattern)
        def clone_with_new_inputs(self, *new_inputs):
            return New(*new_inputs)
        @classmethod
        def desc(cls):
            return (Broadcast, scalar_opclass, tuple(inplace_pattern.items()))
    if name is not None:
        New.__name__ = name
    else:
        New.__name__ = "Tensor" + scalar_opclass.__name__
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
    return instantiate



################
### CAReduce ###
################

class CAReduce(Op):
    """
    CAReduce(scalar_op, inputs, dimensions_to_reduce = None, init = None, shortcut = False)
    
    The number of inputs must be the difference between the number of
    outputs of scalar_op and its number of inputs. L{CAReduce} holds
    scalar states, the accumulators, in proportion to the number of
    outputs of scalar_op and it updates them iteratively::

      for x, y, ... in input0, input1, ...
        scalar_state <- scalar_op(scalar_state, x, y, ...)}

    The initial states are init if provided (they must be scalars),
    else if there are as many states as inputs, a sample from each
    input will be taken as initialization, else an error will be
    raised.

    If shortcut is True and the scalar op has a 'tbd' field, the
    iteration will try to stop as soon as it encounters the value
    specified for that field and will return it immediately, eg
    multiply/and will return 0 at first sight of 0 and 'or' will
    return 1 at first sight of 1.

    In order to optimize memory usage patterns, L{CAReduce} makes zero
    guarantees on the order in which it iterates over the dimensions
    and the elements of the array(s). Therefore, to ensure consistent
    results, the scalar operation represented by the reduction must be
    both commutative and associative (eg add, multiply, binary
    or/and/xor - but not subtract, divide or power).
    """
    
    def __init__(self, scalar_opclass, inputs, dimensions_to_reduce = None):
        inputs = map(astensor, inputs)
        
        if scalar_opclass.nin != 2 or scalar_opclass.nout != 1:
            raise NotImplementedError("CAReduce only supports binary functions with a single output.")
        if len(inputs) != 1:
            raise TypeError("Only one argument expected.")
        if dimensions_to_reduce is None:
            dimensions_to_reduce = range(len(inputs[0].broadcastable))

        self.nin = 1
        self.nout = 1

        self.inputs = inputs
        self.outputs = [Tensor(dtype = inputs[0].dtype,
                               broadcastable = [x for i, x in enumerate(inputs[0].broadcastable) if i not in dimensions_to_reduce])]

        self.dimensions_to_reduce = dimensions_to_reduce
        self.scalar_opclass = scalar_opclass
        self.shadow = scalar_opclass(*[Scalar(dtype = inputs[0].dtype) for i in xrange(scalar_opclass.nin)])
        self.ufunc = numpy.frompyfunc(self.shadow.impl, scalar_opclass.nin, scalar_opclass.nout)

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
    if getattr(scalar_opclass, 'commutative', True) \
            and getattr(scalar_opclass, 'associative', True):
        reducer = CAReduce
    else:
        raise NotImplementedError("The scalar op class to reduce must be commutative and associative.")
    
    class New(reducer):
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
    if name is not None:
        New.__name__ = name
    else:
        New.__name__ = "Reduce" + scalar_opclass.__name__
    return New

class Sum(make_reduce(scalar.Add)):
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




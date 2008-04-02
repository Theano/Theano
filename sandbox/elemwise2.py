

# foldl(f, fold_inputs, init) =>
#     fold_inputs = init;
#     for loop_inputs in c_order(difference(inputs, fold_inputs)):
#         fold_inputs = f(fold_inputs, loop_inputs)
# a+b+c+d => ((a+b)+c)+d

# foldr(f, fold_inputs, init) =>
#     fold_inputs = init;
#     for loop_inputs in reversed_c_order(difference(inputs, fold_inputs)):
#         fold_inputs = f(fold_inputs, loop_inputs)
# a**b**c**d => a**(b**(c**d))

# foldx(f, fold_inputs, init) =>
#     fold_inputs = init;
#     for loop_inputs in any_order(difference(inputs, fold_inputs)):
#         fold_inputs = f(fold_inputs, loop_inputs)
# a+b+c+d => ((a+b)+c)+d
# a+b+c+d => a+(b+(c+d))
# a+b+c+d => (a+b)+(c+d)

# foldx <=> f.associative
# f.associative => (foldl => foldx) and (foldr => foldx)


# z = a*b + b*c + c*d + d*e
# z: (0, 0, 0, 0)
# a: (0, 0, 0, 0)       => (0, 0, 1, 0, 0, 1) => loop order: 1, 2, 3, 4, x, x
# b: (0, 0)             => (1, 1, 1, 0, 0, 1) => loop order: x, x, x, 1, 2, x
# c: (0, 0, S, 0, 0, S) => (0, 0, S, 0, 0, S) => loop order: 1, 2, 4, 5, 3, 6
# d: (1, 0, 1)          => (1, 1, 1, 0, 1, 1) => loop order: x, x, x, 2, x, x
# e: (S, 0, 0, S, 0, 0) =>                    => loop order: 2, 3, 5, 6, 1, 4



# strategy: (broadcasted, folded, fold_method)
# (2, 1, 1, 3, 1), (1, 7, 1, 1, 4)
#          (2, 7, 1, 3, 4), (1, 1, 8, 1, 4)
#                  (2, 7, 8, 3, 4)

# (2, 3, 4, 5), (7, 3, 4, 8)
#    (2, 3, 4), (3, 4)
#        (2, 3, 4)


class ElemwiseGroup:

    def __init__(self):
        self.





def compile_env(env):

    mappings = {}
    

    order = env.io_toposort()

    for op in reversed(order):
        if not isinstance(op, Elemwise):
            raise TypeError("Unsupported op type for the Elemwise compiler.", op)
        for input in op.input_policy:
            strategies.setdefault()




def elemwise_op_gen(op, modalities):
    """
    * op: z = x + y
      modalities: {z: foldx(0, x, y)}
      result: Z = sum(Y)
      
    * op: z = x + y
    """

def broadcasting_cgen(op):
    template = op.c_foreach()
    




class DimShuffle(Op, Viewer):

    def __init__(self, input, new_order, inplace = True):

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
        
        self.numorder = [x for x in new_order if type(x) == int]
        self.is_transposition = sorted(new_order) == range(length(ib))
        self.dup_dims = len(set(self.numorder)) != len(self.numorder)
        self.all_dims = len(set(self.numorder)) == len(ib)
        if self.dup_dims or not self.all_dims:
            raise NotImplementedError("You must provide a permutation of *all* the input dimensions with *no duplicates*.")
    
    def view_map(self):
        if self.inplace:
            return {self.outputs[0]: [self.inputs[0]]}
        else:
            return {}

    def perform(self):
        res = self.inputs[0].data.transpose(self.numorder)
        shape = list(res.shape)
        new_shape = []
        for entry in new_order:
            if entry == 'x':
                new_shape.append(1)
            else:
                new_shape.append(shape.pop())
        res = res.reshape(new_shape)
        if not inplace:
            res = numpy.copy(res)
        self.outputs[0].data = res

    def __str__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, str(self.inputs[0]), self.new_order)


class Transpose(DimShuffle):

    def __init__(self, input):
        DimShuffle.__init__(self, input, range(len(input.broadcastable)-1, -1, -1))


class Broadcast(Op, Destroyer):

    def __init__(self, scalar_opclass, inputs, inplace_pattern):
        try:
            assert len(set([len(input.broadcastable) for input in inputs]) == 1)
        except (AssertionError, AttributeError):
            raise TypeError("All inputs to a Broadcast subclass must be Tensor instances and their broadcastable fields must all have the same length.", self.__class__)
        out_broadcastables = [[1*all(bcast) for bcast in zip(*[input.broadcastable for input in inputs])]] * self.nout
        upcasted = upcast(*[input.dtype for input in inputs])
        def get_dtype(i):
            input_idx = inplace_pattern.get(i, [None])
            if input_idx is not None:
                return inputs[input_idx].dtype
            else:
                return upcasted
        out_dtypes = map(get_dtype, xrange(self.nout))
        self.inputs = inputs
        self.outputs = [Tensor(dtype = dtype, broadcastable = broadcastable) for dtype, broadcastable in zip(out_dtypes, out_broadcastables)]
        self.inplace_pattern = inplace_pattern
        self.scalar_opclass = scalar_opclass
        self.shadow = scalar_opclass([Scalar(dtype = t.dtype) for t in self.inputs])
        self.ufunc = numpy.frompyfunc(scalar_opclass.impl, scalar_opclass.nin, scalar_opclass.nout)

    def id(self):
        return (self.__class__, self.scalar_opclass, self.inplace_pattern)

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
            op_class = op.__class__
            bcasted = Broadcast(op_class, [transform(input) for input in op.inputs], {})
            return bcasted
        ret = []
        for scalar_igrad, input in zip(scalar_igrads, inputs):
            r = transform(scalar_igrad)
            to_sum = [i for i, bcast in enumerate(input.broadcastable) if bcast]
            if to_sum:
                ret.append(Sum(r, to_sum))
            else:
                ret.append(r)
        return ret

    def perform(self):
        output_storage = []
        if not self.inplace_pattern:
            for output in self.outputs:
                odat = output.data
                if odat is not None:
                    odat.resize(self.inputs[0].data.shape)
                else:
                    odat = numpy.ndarray(self.inputs[0].data.shape, dtype = output.dtype)
                output_storage.append(odat)
        else:
            for i, output in enumerate(self.outputs):
                if i in self.inplace_pattern:
                    odat = self.inputs[self.inplace_pattern[i]].data
                else:
                    odat = output.data
                    if odat is not None:
                        odat.resize(self.inputs[0].data.shape)
                    else:
                        odat = numpy.ndarray(self.inputs[0].data.shape, dtype = output.dtype)
                output_storage.append(odat)
        self.ufunc(*([input.data for input in self.inputs] + output_storage))


def broadcast(op):
    def instantiate(*inputs):
        target_length = max([len(input.broadcastable) for input in inputs])
        args = []
        for input in inputs:
            difference = target_length - len(input.broadcastable)
            if not difference:
                args.append(input)
            else:
                args.append(DimShuffle(input, ['x']*difference + range(length)))
        return op(*args)


class CAReduce(Op):
    """
    CAReduce(scalar_op, inputs, dimensions_to_reduce = None, init = None, shortcut = False)
    
    The number of inputs must be the difference between the number of
    outputs of scalar_op and its number of inputs. CAReduce holds
    scalar states, the accumulators, in proportion to the number of
    outputs of scalar_op and it updates them iteratively:
    for x, y, ... in input0, input1, ...
      scalar_state <- scalar_op(scalar_state, x, y, ...)

    The initial states are init if provided (they must be scalars),
    else if there are as many states as inputs, a sample from each
    input will be taken as initialization, else an error will be
    raised.

    If shortcut is True and the scalar op has a 'tbd' field, the
    iteration will try to stop as soon as it encounters the value
    specified for that field and will return it immediately, eg
    multiply/and will return 0 at first sight of 0 and 'or' will
    return 1 at first sight of 1.

    In order to optimize memory usage patterns, CAReduce makes zero
    guarantees on the order in which it iterates over the dimensions
    and the elements of the array(s). Therefore, to ensure consistent
    results, the scalar operation represented by the reduction must be
    both commutative and associative (eg add, multiply, binary
    or/and/xor - but not subtract, divide or power).
    """
    
    def __init__(self, scalar_opclass, inputs, dimensions_to_reduce = None):
        if scalar_opclass.nin != 2 or scalar_opclass.nout != 1:
            raise NotImplementedError("CAReduce only supports binary functions with a single output.")

def reduce(op, dimensions_to_reduce):
    if getattr(op, 'commutative', True) and getattr(op, 'associative', True):
        reducer = CAReduce
    else:
        raise NotImplementedError("The scalar op class to reduce must be commutative and associative.")
    def instantiate(*inputs):
        return reducer(op, inputs, dimensions_to_reduce)
    return instantiate



# class Elemwise(TensorOp):

#     def propagate_dtype(self, idtypes):
#         raise AbstractFunctionError

#     def propagate_broadcastable(self, ibroadcastables):
#         raise AbstractFunctionError

#     def _calculate_elemwise_strategy(self, input_strategies):
#         raise AbstractFunctionError




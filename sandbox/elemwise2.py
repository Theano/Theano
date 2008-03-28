

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

    def __init__(self, input, new_order):

        ib = input.broadcastable
        ob = []
        for value in new_order:
            if value == 'x':
                ob.append(1)
            else:
                ob.append(ib[value])
        
        output = Tensor(dtype = input.dtype,
                        broadcastable = ob)

        self.new_order = new_order
        self.inputs = input,
        self.outputs = output,

    def view_map(self):
        return {self.outputs[0]: [self.inputs[0]]}

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
        


def broadcast2(op):
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


class FoldX(Op):

    def __init__(self, scalar_opclass, inputs, to_fold):
        pass




# class Elemwise(TensorOp):

#     def propagate_dtype(self, idtypes):
#         raise AbstractFunctionError

#     def propagate_broadcastable(self, ibroadcastables):
#         raise AbstractFunctionError

#     def _calculate_elemwise_strategy(self, input_strategies):
#         raise AbstractFunctionError




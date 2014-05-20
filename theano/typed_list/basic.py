from type import TypedListType

from theano.gof import Apply, Constant, Op, Variable
from theano.tensor.type_other import SliceType
from theano import tensor as T


class _typed_list_py_operators:

    def __getitem__(self, index):
        return GetItem()(self, index)

    def append(self, toAppend):
        return Append()(self, toAppend)

    def extend(self, toAppend):
        return Extend()(self, toAppend)

    ttype = property(lambda self: self.type.ttype)


class TypedListVariable(_typed_list_py_operators, Variable):
    """
    Subclass to add the typed list operators to the basic `Variable` class.
    """

TypedListType.Variable = TypedListVariable


class GetItem(Op):
    """
    get specified slice of a typed list
    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x, index):
        assert isinstance(x.type, TypedListType)
        assert isinstance(index, Variable)
        if isinstance(index.type, SliceType):
            return Apply(self, [x, index], [x.type()])
        elif isinstance(index, T.TensorVariable) and index.ndim == 0:
            return Apply(self, [x, index], [x.ttype()])
        else:
            raise TypeError('Expected scalar or slice as index.')

    def perform(self, node, (x, index), (out, )):
        if not isinstance(index, slice):
            index = int(index)
        out[0] = x[index]

    def __str__(self):
        return self.__class__.__name__


class Append(Op):
    """
    #append an element at the end of another list
    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x, toAppend):
        assert isinstance(x.type, TypedListType)
        assert x.ttype == toAppend.type
        return Apply(self, [x, toAppend], [x.type()])

    def perform(self, node, (x, toAppend), (out, )):
        out[0] = x
        out[0].append(toAppend)

    def __str__(self):
        return self.__class__.__name__

    destroy_map = {0: [0]}


class Extend(Op):
    """
    append all element of a list at the end of another list
    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x, toAppend):
        assert isinstance(x.type, TypedListType)
        assert x.type == toAppend.type
        return Apply(self, [x, toAppend], [x.type()])

    def perform(self, node, (x, toAppend), (out, )):
        out[0] = x
        out[0].extend(toAppend)

    def __str__(self):
        return self.__class__.__name__

    destroy_map = {0: [0]}

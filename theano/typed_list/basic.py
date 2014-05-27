from type import TypedListType

from theano.gof import Apply, Constant, Op, Variable
from theano.tensor.type_other import SliceType
from theano import tensor as T

import numpy


class _typed_list_py_operators:

    def __getitem__(self, index):
        return GetItem()(self, index)

    def append(self, toAppend):
        return Append()(self, toAppend)

    def extend(self, toAppend):
        return Extend()(self, toAppend)

    def insert(self, index, toInsert):
        return Insert()(self, index, toInsert)

    def remove(self, toRemove):
        return Remove()(self, toRemove)

    def reverse(self):
        return Reverse()(self)

    def count(self, elem):
        return Count()(self, elem)

    #name "index" is already used by an attribute
    def ind(self, elem):
        return Index()(self, elem)

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
        if not isinstance(index, Variable):
            if isinstance(index, slice):
                index = Constant(SliceType(), index)
                return Apply(self, [x, index], [x.type()])
            else:
                index = T.constant(index, ndim=0)
                return Apply(self, [x, index], [x.ttype()])
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

    def __init__(self, inplace=False):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x, toAppend):
        assert isinstance(x.type, TypedListType)
        assert x.ttype == toAppend.type
        return Apply(self, [x, toAppend], [x.type()])

    def perform(self, node, (x, toAppend), (out, )):
        if not self.inplace:
            out[0] = list(x)
        else:
            out[0] = x
        out[0].append(toAppend)

    def __str__(self):
        return self.__class__.__name__


class Extend(Op):
    """
    append all element of a list at the end of another list
    """

    def __init__(self, inplace=False):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x, toAppend):
        assert isinstance(x.type, TypedListType)
        assert x.type == toAppend.type
        return Apply(self, [x, toAppend], [x.type()])

    def perform(self, node, (x, toAppend), (out, )):
        if not self.inplace:
            out[0] = list(x)
        else:
            out[0] = x
        out[0].extend(toAppend)

    def __str__(self):
        return self.__class__.__name__


class Insert(Op):

    def __init__(self, inplace=False):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x, index, toInsert):
        assert isinstance(x.type, TypedListType)
        assert x.ttype == toInsert.type
        if not isinstance(index, Variable):
            index = T.constant(index, ndim=0)
        else:
            assert isinstance(index, T.TensorVariable) and index.ndim == 0
        return Apply(self, [x, index, toInsert], [x.type()])

    def perform(self, node, (x, index, toInsert), (out, )):
        if not self.inplace:
            out[0] = list(x)
        else:
            out[0] = x
        out[0].insert(index, toInsert)

    def __str__(self):
        return self.__class__.__name__


class Remove(Op):

    def __init__(self, inplace=False):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x, toRemove):
        assert isinstance(x.type, TypedListType)
        assert x.ttype == toRemove.type
        return Apply(self, [x, toRemove], [x.type()])

    def perform(self, node, (x, toRemove), (out, )):

        if not self.inplace:
            out[0] = list(x)
        else:
            out[0] = x

        """
        inelegant workaround for ValueError: The truth value of an
        array with more than one element is ambiguous. Use a.any() or a.all()
        being thrown when trying to remove a matrix from a matrices list
        """
        if isinstance(toRemove, numpy.ndarray):
            for y in range(out[0].__len__()):
                if numpy.array_equal(out[0][y], toRemove):
                    del out[0][y]
                    break
        else:
            out[0].remove(toRemove)

    def __str__(self):
        return self.__class__.__name__


class Reverse(Op):

    def __init__(self, inplace=False):
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x):
        assert isinstance(x.type, TypedListType)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inp, (out, )):

        if not self.inplace:
            out[0] = list(inp[0])
        else:
            out[0] = inp[0]
        out[0].reverse()

    def __str__(self):
        return self.__class__.__name__


class Index(Op):

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x, elem):
        assert isinstance(x.type, TypedListType)
        assert x.ttype == elem.type
        return Apply(self, [x, elem], [T.scalar()])

    def perform(self, node, (x, elem), (out, )):
        """
        inelegant workaround for ValueError: The truth value of an
        array with more than one element is ambiguous. Use a.any() or a.all()
        being thrown when trying to remove a matrix from a matrices list
        """
        if isinstance(elem, numpy.ndarray):
            for y in range(x.__len__()):
                if numpy.array_equal(x[y], elem):
                    out[0] = numpy.asarray([y])
                    break
        else:
            out[0] = numpy.asarray([x.index(elem)])

    def __str__(self):
        return self.__class__.__name__


class Count(Op):

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x, elem):
        assert isinstance(x.type, TypedListType)
        assert x.ttype == elem.type
        return Apply(self, [x, elem], [T.scalar()])

    def perform(self, node, (x, elem), (out, )):
        """
        inelegant workaround for ValueError: The truth value of an
        array with more than one element is ambiguous. Use a.any() or a.all()
        being thrown when trying to remove a matrix from a matrices list
        """
        if isinstance(elem, numpy.ndarray):
            out[0] = 0
            for y in range(x.__len__()):
                if numpy.array_equal(x[y], elem):
                    out[0] += 1
            out[0] = numpy.asarray([out[0]])
        else:
            out[0] = numpy.asarray([x.count(elem)])

    def __str__(self):
        return self.__class__.__name__

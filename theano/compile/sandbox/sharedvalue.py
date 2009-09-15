"""Provide a simple user friendly API """
__docformat__ = 'restructuredtext en'

import copy
import numpy

import theano.tensor.basic
from theano.gof import Container, Variable, generic
from theano.tensor import TensorType
from theano.scalar import Scalar
from theano.compile import function


class SharedVariable(Variable):
    """
    Variable that is (defaults to being) shared between functions that it appears in.

    """

    #Container object
    container = None
    """
    A container to use for this SharedVariable when it is an implicit function parameter.

    :type: `Container`
    """

    def __init__(self, name, type, value, strict, container=None):
        """
        :param name: The name for this variable (see `Variable`).

        :param type: The type for this variable (see `Variable`).

        :param value: A value to associate with this variable (a new container will be created).

        :param strict: True -> assignments to .value will not be casted or copied, so they must
        have the correct type.

        :param container: The container to use for this variable. Illegal to pass this as well
        as a value.

        For more user-friendly constructor, see `shared`

        """
        super(SharedVariable, self).__init__(type=type, name=name, owner=None, index=None)

        if container is not None:
            self.container = container
            if (value is not None) or (strict is not None):
                raise TypeError('value and strict are ignored if you pass a container here')
        else:
            if container is not None:
                raise TypeError('Error to specify both value and container')
            self.container = Container(self,
                    storage=[type.filter(value, strict=strict)],
                    readonly=False,
                    strict=strict)

    def __set(self,new_value):
        self.container.value = new_value

    def __get(self):
        return self.container.value

    def clone(self):
        cp = self.__class__(
                name=self.name,
                type=self.type, 
                value=None,
                strict=None,
                container=self.container)
        cp.tag = copy.copy(self.tag)
        return cp

    value = property(__get, __set)
    #value = self.container.value #GD- would've thought mapping one property to another would work

    """Read/write the non-symbolic value associated with this SharedVariable.
    
    If the SharedVariable is shared, changes to this value will be visible to all functions using
    this SharedVariable.  If this SharedVariable is not shared, a change will not be visible to
    functions that were created before the change.

    """

    def filter_update(self, update):
        """When this shared variable is updated by a pfunc, the update value will be run through this function.
        This is a good spot to cast or convert the update expression as necessary.

        Default behaviour is to return `update` unmodified if it is a Variable, otherwise to create a SharedVariable for it by calling ``shared(update)``.

        :param update: the new value for this shared variable when updated by a pfunc.

        :returns: a Variable whose value will be assigned to this SharedVariable by a pfunc.
        """
        if not isinstance(update, Variable):
            # The value for the update is not a Variable: we cast it into
            # a shared Variable so that it can be used by 'function'. Note that
            # it means the update value may change if it is mutable and its
            # value is modified after the function is created.
            update = shared(update)
        return update

def shared_constructor(ctor):
    shared.constructors.append(ctor)
    return ctor

def shared(value, name=None, strict=False, **kwargs):
    """Return a SharedVariable Variable, initialized with a copy or reference of `value`.

    This function iterates over constructor functions (see `shared_constructor`) to find a
    suitable SharedVariable subclass.

    :note: 
    By passing kwargs, you effectively limit the set of potential constructors to those that
    can accept those kwargs.
    
    """
    for ctor in reversed(shared.constructors):
        try:
            return ctor(value, name=name, strict=strict, **kwargs)
        except TypeError:
            continue
    # This may happen when kwargs were supplied
    # if kwargs were given, the generic_constructor won't be callable.
    #
    # This was done on purpose, the rationale being that if kwargs were supplied,
    # the user didn't want them to be ignored.
    raise TypeError('No suitable SharedVariable constructor could be found', (value, kwargs))
shared.constructors = []

@shared_constructor
def generic_constructor(value, name=None, strict=False):
    """SharedVariable Constructor"""
    return SharedVariable(type=generic, value=value, name=name, strict=strict)


class TensorSharedVariable(SharedVariable, theano.tensor.basic._tensor_py_operators):
    pass
@shared_constructor
def tensor_constructor(value, name=None, strict=False):
    """SharedVariable Constructor for TensorType"""
    if not isinstance(value, numpy.ndarray):
        raise TypeError()

    bcast = [b==1 for b in value.shape]
    type = TensorType(value.dtype, broadcastable=bcast)
    return TensorSharedVariable(type=type, value=value, name=name, strict=strict)

@shared_constructor
def scalar_constructor(value, name=None, dtype=None, strict=False):
    """SharedVariable constructor for scalar values. Defaults to int64 or float64"""  
    if not isinstance(value, (float,int)):
        raise TypeError()
    # use float64 and int64 by default, user can override
    if not dtype:
        dtype = 'int64' if isinstance(value,int) else 'float64'
    type = Scalar(dtype)
    return TensorSharedVariable(type=type, value=numpy.asarray(value), name=name, strict=strict)


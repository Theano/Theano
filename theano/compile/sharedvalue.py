"""Provide a simple user friendly API to Theano-managed memory"""
__docformat__ = 'restructuredtext en'

# Standard imports
import copy
import logging
import traceback
import warnings

# Python 2.4 doesn't contain this, but it's just a convenience, so replace
# it with a no-op decorator if it doesn't exist.
try:
    from functools import wraps
except ImportError:
    def wraps(*arg):
        def _wrap(f):
            return f
        return _wrap

# Theano imports
from theano import config
from theano.configparser import (TheanoConfigParser, AddConfigVar, EnumStr,
        StrParam, IntParam, FloatParam, BoolParam)
from theano.gof import Container, Variable, generic

_logger = logging.getLogger('theano.compile.sharedvalue')

AddConfigVar('shared.value_borrows',
        ("DEPRECATED. You should not use the 'value' property of shared"
            " variables, but use the .get_value() and .set_value() methods."
            " False: shared variables 'value' property is guaranteed to not"
            " alias theano-managed memory. True: no guarantee, but faster."),
        BoolParam(True),
        in_c_key=False)

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

    # default_update
    # If this member is present, its value will be used as the "update" for
    # this Variable, unless another update value has been passed to "function",
    # or the "no_default_updates" list passed to "function" contains it.

    def __init__(self, name, type, value, strict, allow_downcast=None, container=None):
        """
        :param name: The name for this variable (see `Variable`).

        :param type: The type for this variable (see `Variable`).

        :param value: A value to associate with this variable (a new container will be created).

        :param strict: True -> assignments to .value will not be casted or copied, so they must
        have the correct type.

        :param allow_downcast: Only applies if `strict` is False.
        True -> allow assigned value to lose precision when casted during assignment.
        False -> never allow precision loss.
        None -> only allow downcasting of a Python float to a scalar floatX.

        :param container: The container to use for this variable. Illegal to pass this as well
        as a value.

        :note: For more user-friendly constructor, see `shared`

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
                    storage=[type.filter(value, strict=strict, allow_downcast=allow_downcast)],
                    readonly=False,
                    strict=strict,
                    allow_downcast=allow_downcast)

    def get_value(self, borrow=False, return_internal_type=False):
        """Get the non-symbolic value associated with this SharedVariable.

        :param borrow:
            True to permit returning of an object aliased to internal memory.
        :param return_internal_type:
            True to permit the returning of an arbitrary type object used internally to store
            the shared variable.

        Only with borrow=False and return_internal_type=True does this function guarantee that
        you actually get the internal object.  But in that case, you may get different return
        types when using different compute devices.

        """
        if borrow:
            return self.container.value
        else:
            return copy.deepcopy(self.container.value)

    def set_value(self,new_value, borrow=False):
        """Set the non-symbolic value associated with this SharedVariable.

        :param borrow:
            True to use the new_value directly, potentially creating problems
            related to aliased memory.

        Changes to this value will be visible to all functions using this SharedVariable.
        """
        if borrow:
            self.container.value = new_value
        else:
            self.container.value = copy.deepcopy(new_value)

    def clone(self):
        cp = self.__class__(
                name=self.name,
                type=self.type,
                value=None,
                strict=None,
                container=self.container)
        cp.tag = copy.copy(self.tag)
        return cp

    @wraps(get_value)
    def value(self, *args, **kwargs):
        """Alias for get_value."""
        return self.get_value(*args, **kwargs)

    def filter_update(self, update):
        """
        When this shared variable is updated by a pfunc, the update value will be run through this function.

        This is a good spot to cast or convert the update expression as necessary.

        Default behaviour is to return `update` unmodified if it is a Variable, otherwise to create a SharedVariable for it by calling ``shared(update)``.

        :param update: the new value for this shared variable when updated by a pfunc.

        :returns: a Variable whose value will be assigned to this SharedVariable by a pfunc.

        :note: The return value of this function must match the self.type, or else pfunc()
        will raise a TypeError.
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

def shared(value, name=None, strict=False, allow_downcast=None, **kwargs):
    """Return a SharedVariable Variable, initialized with a copy or reference of `value`.

    This function iterates over constructor functions (see `shared_constructor`) to find a
    suitable SharedVariable subclass.

    :note:
    By passing kwargs, you effectively limit the set of potential constructors to those that
    can accept those kwargs.
    :note: Some shared variable have 'borrow' as extra kwargs.
           `See <http://deeplearning.net/software/theano/tutorial/aliasing.html#borrowing-when-creating-shared-variables>`_ for detail.

    """
    for ctor in reversed(shared.constructors):
        try:
            return ctor(value, name=name, strict=strict,
                    allow_downcast=allow_downcast, **kwargs)
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
def generic_constructor(value, name=None, strict=False, allow_downcast=None):
    """SharedVariable Constructor"""
    return SharedVariable(type=generic, value=value, name=name, strict=strict,
            allow_downcast=allow_downcast)

"""Provide a simple user friendly API """
__docformat__ = 'restructuredtext en'

import traceback
import copy
from theano.gof import Container, Variable, generic

import logging
_logger = logging.getLogger('theano.compile.sharedvalue')
_logger.setLevel(logging.DEBUG)
def debug(*msg): _logger.debug(' '.join(str(m) for m in msg))
def info(*msg): _logger.info(' '.join(str(m) for m in msg))
def warn(*msg): _logger.warn(' '.join(str(m) for m in msg))
def warning(*msg): _logger.warning(' '.join(str(m) for m in msg))
def error(*msg): _logger.error(' '.join(str(m) for m in msg))

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

    def get_value(self, borrow=False):
        """Get the non-symbolic value associated with this SharedVariable.

        :param borrow: 
            True to return the internal value directly, potentially creating problems related
            to aliased memory.

        If the return value is mutable, and you have used borrow=True to get at the internal
        value, then you should be careful about changing it.  If you modify it, call
        set_value(rval, borrow=True) to tell Theano that you modified it.  (Theano may have
        cached computations based on the old value.)
        
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

    value = property(get_value, set_value, 
            doc="shortcut for self.get_value() and self.set_value() which COPIES data")


    def filter_update(self, update):
        """When this shared variable is updated by a pfunc, the update value will be run through this function.
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


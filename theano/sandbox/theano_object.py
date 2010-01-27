"""DRAFT: TheanoObject

N.B. the gotcha with this design is listed in the documentation of `TheanoObject`

"""
import theano
from theano import tensor
import numpy

def theano_type(x):
    """Return a theano Type instance suitable for containing value `x`."""
    if type(x) is int:
        return tensor.lscalar
    else:
        raise NotImplementedError()

class symbolic_fn_callable(object):
    """This is the class whose instance you get when you access a symbolic function in a
    `TheanoObject`.

    When you call a symbolic function (`symbolic_fn`) of a TheanoObject the `__call__` of this
    class handles your request.

    You can also access the symbolic outputs and updates of a symbolic function though this
    class.

    .. code-block:: python
       
       class T(TheanoObject):
          @symbolic_fn
          def add(self, x):
             ...
             add_outputs = ...
             add_updates = ...
             return RVal(add_outputs, add_updates)
       t = T() 
       t.add.outputs(5)         # returns `add_outputs` from when `x=theano_type(5)`
       t.add.updates(5)         # returns `add_updates` from when `x=theano_type(5)`
       t.add.theano_function(5) # returns the `Function` compiled when `x=theano_type(5)`
       t.add(5)                 # runs the `Function` compiled when `x=theano_type(5)`
                                #     with arguments `(5,)`
    """
    def __init__(self, fn, mode):
        self.fn = fn
        self.mode = mode

    def on(self, o_self):
        """Silly method to work with symbolic_fn.__get__"""
        self.o_self = o_self
        return self
    
    def run_symbolic(self, *args, **kwargs):
        return self.o_self._get_method_impl(self.fn, self.o_self, args, kwargs, mode=self.mode)

    def __call__(self, *args, **kwargs):
        return self.run_symbolic(*args, **kwargs)['theano_function'](*args, **kwargs)

    def theano_function(self, *args, **kwargs):
        return self.run_symbolic(*args, **kwargs)['theano_function']

    def outputs(self, *args, **kwargs):
        return self.run_symbolic(*args, **kwargs)['outputs']

    def updates(self, *args, **kwargs):
        return self.run_symbolic(*args, **kwargs)['updates']

class symbolic_fn(object):
    """A property-like class for decorating symbolic functions in `TheanoObject`
    """
    def __init__(self, fn, mode=None):
        self.fn = fn
        self.callable = symbolic_fn_callable(fn, mode)
    
    def __get__(self, o_self, o_cls):
        return self.callable.on(o_self)

    def __set__(self, o_self, new_val):
        pass
        #return NotImplemented

def symbolic_fn_opts(**kwargs):
    """Return a decorator for symbolic_functions in a `TheanoObject`

    `kwargs` passed here are passed to `theano.function` via `symbolic_fn`
    """
    def deco(f):
        return symbolic_fn(f, **kwargs)
    return deco

class RVal(object):
    """A Return-Value object for a `symbolic_fn` """
    outputs = []
    """The method will compute values for the variables in this list"""
    
    updates = {}
    """The method will update module variables in this dictionary

    For items ``(k,v)`` in this dictionary, ``k`` must be a `symbolic_member` of some module.
    On each call to this compiled function, the value of ``k`` will be replaced with the
    computed value of the Variable ``v``.
    
    """
    def __init__(self, outputs, updates={}):
        self.outputs = outputs
        assert type(updates) is dict
        self.updates = updates

class TheanoObject(object):
    """Base for Theano-supported classes

    This class provides support for symbolic_fn class attributes.
    These will be compiled on demand so that they can be used just like normal (non-symbolic)
    methods.
    
    The symbolic functions in a TheanoObject can share member variables that have been created
    using the `symbolic_member` method.

    :note: Other variables (ones not created using ``self.symbolic_member``) referred to in the
    body of a symbolic function will *not* be shared between symbolic functions, or between
    symbolic functions and this class.  These other variables will be locked away in the
    closure of a symbolic function when that function is compiled.  
    

    :warning: It is not recommended for code to interleave
    (a) changes to non-symbolic instance variables with
    (b) calls to symbolic functions that use those instance variables. 
    A symbolic function may be
    compiled multiple times because it must be compiled for each set of argument types.
    Each time the function is compiled, the values of non-symbolic variables will be locked
    into the compiled function.  Subsequent changes to those non-symbolic instance variables
    will not have any effect on the behaviour of the already-compiled symbolic function.

    :todo: Is there an efficient way of recognizing when a compiled symbolic function is stale,
    wrt the current values of the class's instance variables?

    - One option is to re-evaluate symbolic functions symbolically and see if the graph can be
      completely merged with the original graph.  This is not fast enough to do all the time by
      default though.

    """
    def __init__(self):
        self.module_method_cache = {}

    def _get_method_impl(self, fn, o_self, args, kwargs, mode):
        """Retrieve information about the symbolic function (`fn`) in TheanoObject instance
        `o_self`, being evaluated on arguments `args` and `kwargs`.

        :rtype: dict with entries 'theano_function', 'outputs', 'updates'

        :return: the theano function compiled for these arguments, the symbolic outputs of that
        function, and the symbolic updates performed by that function.

        :note: This function caches return values in self.`module_method_cache`.

        :todo: This may at some point become a class-level cache rather than an instance-level
        cache.

        """
        if kwargs:
            raise NotImplementedError()

        cache = self.module_method_cache

        args_types = tuple(theano_type(arg) for arg in args)
        key = (fn, args_types)

        if key not in cache:
            inputs = [a() for a in args_types]
            print 'compiling', fn, 'for inputs', inputs
            rval = fn(o_self, *inputs)

            print 'compiling to compute outputs', rval.outputs

            if isinstance(rval.outputs, (tuple, list)):
                all_required_inputs = theano.gof.graph.inputs(rval.outputs)
            else:
                all_required_inputs = theano.gof.graph.inputs([rval.outputs])

            # construct In instances for the symbolic_member instances that can automatically be
            # included here.
            module_inputs = [theano.compile.io.In(
                    variable=v, 
                    value=v._theanoclass_container,
                    mutable=(v in rval.updates),
                    update=rval.updates.get(v, None))
                for v in all_required_inputs \
                        if hasattr(v, '_theanoclass_container') and not (v in inputs)]

            cache[key] = dict(theano_function=theano.function(inputs+module_inputs, rval.outputs),
                    updates=rval.updates,
                    outputs=rval.outputs,
                    mode=mode)

        return cache[key]

    def symbolic_member(self, ival, name=None):
        """Create a Variable instance to hold value `ival`.

        This function also immediately creates a Container object for ival.
        When the returned Variable is used as input to a `TheanoObject` `symbolic_fn`, (but
        does not appear as an argument to that symbolic_fn), then this Container will be used to
        retrieve (and store) values for the Variable.

        This Variable's Container's contents can be retrieved by its `get()` method.
        This Variable's Container's contents can be written using its `set(newval)` method.

        """
        if type(ival) is not int:
            raise NotImplementedError()

        v = tensor.lscalar(name)
        v._theanoclass_container = \
                theano.gof.Container(v, 
                        storage = [theano._asarray(ival, dtype='int64')],
                        readonly=False)
        assert not hasattr(v, 'set')
        assert not hasattr(v, 'get')
        v.get = lambda : v._theanoclass_container.data
        def setval_in_v(newval):
            v._theanoclass_container.data = newval
        v.set = setval_in_v
        return v


        


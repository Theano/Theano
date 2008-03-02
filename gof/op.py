
"""
Contains the Op class, which is the base interface for all operations
compatible with gof's graph manipulation routines.
"""

# from result import BrokenLinkError
from utils import ClsInit, all_bases, all_bases_collect, AbstractFunctionError
import graph

from copy import copy


__all__ = ['Op']


class Op(object):
    """
    Op represents a computation on the storage in its 'inputs' slot,
    the results of which are stored in the Result instances in the
    'outputs' slot. The owner of each Result in the outputs list must
    be set to this Op and thus any Result instance is in the outputs
    list of at most one Op, its owner. It is the responsibility of the
    Op to ensure that it owns its outputs and it is encouraged (though
    not required) that it creates them.

    After construction, self.inputs and self.outputs should only be
    modified through the set_input and set_output methods.
    """

    __slots__ = ['_inputs', '_outputs']

    _default_output_idx = 0

    def default_output(self):
        """Returns the default output of this Op instance, typically self.outputs[0]."""
        try:
            return self.outputs[self._default_output_idx]
        except (IndexError, TypeError):
            raise AttributeError("Op does not have a default output.")

    out = property(default_output, 
            doc = "Same as self.outputs[0] if this Op's has_default_output field is True.")


    def __init__(self, *inputs):
        raise AbstractFunctionError("Op is an abstract class. Its constructor does nothing, you must override it.")

    
#     def __init__(self, *inputs):
#         self._inputs = None
#         self._outputs = None
#         self.set_inputs(inputs)
#         self.validate_update()


#     def __get_input(self, i):
#         input = self._inputs[i]
# #         if input.replaced:
# #             raise BrokenLinkError()
#         return input
#     def __set_input(self, i, new_input):
#         self._inputs[i] = new_input


#     def __get_inputs(self):
# #         for input in self._inputs:
# #             if input.replaced:
# #                 raise BrokenLinkError()
#         return self._inputs
#     def __set_inputs(self, new_inputs):
#         self._inputs = list(new_inputs)


#     def __get_output(self, i):
#         return self._outputs[i]
#     def __set_output(self, i, new_output):
#         raise Exception("Cannot change outputs.")
# #         old_output = self._outputs[i]
# #         if old_output != new_output:
# #             old_output.replaced = True
# #             try:
# #                 # We try to reuse the old storage, if there is one
# #                 new_output.data = old_output.data
# #             except:
# #                 pass
# #             new_output.role = (self, i)
# #             self._outputs[i] = new_output


#     def __get_outputs(self):
#         return self._outputs
#     def __set_outputs(self, new_outputs):
#         if self._outputs is None:
#             for i, output in enumerate(new_outputs):
#                 output.role = (self, i)
#             self._outputs = new_outputs
#             return True
#         raise Exception("Cannot change outputs.")
# #         if len(self._outputs) != len(new_outputs):
# #             raise TypeError("The new outputs must be exactly as many as the previous outputs.")

# #         for i, new_output in enumerate(new_outputs):
# #             self.__set_output(i, new_output)


#     def get_input(self, i):
#         return self.__get_input(i)
#     def set_input(self, i, new_input):
#         old_input = self.__get_input(i)
#         try:
#             self.__set_input(i, new_input)
#             return self.validate_update()
#         except:
#             self.__set_input(i, old_input)
#             self.validate_update()
#             raise


#     def get_inputs(self):
#         return self.__get_inputs()
#     def set_inputs(self, new_inputs):
#         old_inputs = self.__get_inputs()
#         try:
#             self.__set_inputs(new_inputs)
#             return self.validate_update()
#         except:
#             self._inputs = old_inputs
#             raise


#     def get_output(self, i):
#         return self.__get_output(i)

#     def get_outputs(self):
#         return self.__get_outputs()


    def get_input(self, i):
        return self._inputs[i]        
    def set_input(self, i, new):
        self._inputs[i] = new

    def get_inputs(self):
        return self._inputs
    def set_inputs(self, new):
        self._inputs = list(new)

    def get_output(self, i):
        return self._outputs[i]

    def get_outputs(self):
        return self._outputs
    def set_outputs(self, new):
        if not hasattr(self, '_outputs') or self._outputs is None:
            for i, output in enumerate(new):
                output.role = (self, i)
            self._outputs = list(new)
        else:
            raise Exception("Can only set outputs once, to initialize them.")

    #create inputs and outputs as read-only attributes
    inputs = property(get_inputs, set_inputs, doc = "The list of this Op's input Results.")
    outputs = property(get_outputs, set_outputs, doc = "The list of this Op's output Results.")


#     def validate_update(self):
#         """
#         (Abstract) This function must do two things:
#         * validate: check all the inputs in self.inputs to ensure
#                     that they have the right type for this Op, etc.
#                     If the validation fails, raise an exception.
#         * update: if self.outputs is None, create output Results
#                   and set the Op's outputs. Else, fail or update
#                   the outputs in place.
#         If any changes were made to the outputs, return True. Else,
#         return False.
#         """
#         raise AbstractFunctionError()


#     def repair(self):
#         """
#         Repairs all the inputs that are broken links to use what
#         they were replaced with. Then, calls self.validate_update()
#         to validate the new inputs and make new outputs.
#         """
#         changed = False
#         repaired_inputs = []
#         old_inputs = self._inputs
#         for input in self._inputs:
#             if input.replaced:
#                 changed = True
#                 role = input.role.old_role
#                 input = role[0].outputs[role[1]]
#             repaired_inputs.append(input)
#         if changed:
#             try:
#                 self.__set_inputs(repaired_inputs)
#                 self.validate_update()
#             except:
#                 self._inputs = old_inputs
#                 raise
#         return changed


    #
    # copy
    #

    def __copy__(self):        
        """
        Shallow copy of this Op. The inputs are the exact same, but
        the outputs are recreated because of the one-owner-per-result
        policy.
        """
        return self.__class__(*self.inputs)


    #
    # String representation
    #

    def __str__(self):
        return graph.op_as_string(self.inputs, self)

    def __repr__(self):
        return str(self)


    #
    # perform
    #
    
    def perform(self):
        """
        (abstract) Performs the computation associated to this Op,
        places the result(s) in the output Results and gives them
        the Computed status.
        """
        raise AbstractFunctionError()


    #
    # C code generators
    #

    def c_var_names(self):
        """
        Returns ([list of input names], [list of output names]) for
        use as C variables.
        """
        return [["i%i" % i for i in xrange(len(self.inputs))],
                ["o%i" % i for i in xrange(len(self.outputs))]]

    def c_validate(self):
        """
        Returns C code that checks that the inputs to this function
        can be worked on. If a failure occurs, set an Exception
        and insert "%(fail)s".
        
        You may use the variable names defined by c_var_names()
        """
        raise AbstractFunctionError()

    def c_validate_cleanup(self):
        """
        Clean up things allocated by c_validate().
        """
        raise AbstractFunctionError()

    def c_update(self):
        """
        Returns C code that allocates and/or updates the outputs
        (eg resizing, etc.) so they can be manipulated safely
        by c_code.
        
        You may use the variable names defined by c_var_names()
        """
        raise AbstractFunctionError()

    def c_update_cleanup(self):
        """
        Clean up things allocated by c_update().
        """
        raise AbstractFunctionError()

    def c_code(self):
        """
        Returns C code that does the computation associated to this
        Op. You may assume that input validation and output allocation
        have already been done.
        
        You may use the variable names defined by c_var_names()
        """
        raise AbstractFunctionError()

    def c_code_cleanup(self):
        """
        Clean up things allocated by c_code().
        """
        raise AbstractFunctionError()



class GuardedOp(Op):

    def set_input(self, i, new):
        old = self._inputs[i]
        if old is new:
            return
        try:
            if not old.same_properties(new):
                raise TypeError("The new input must have the same properties as the previous one.")
        except AbstractFunction:
            pass
        Op.set_input(self, i, new)

    def set_inputs(self, new):
        if not hasattr(self, '_inputs') or self_inputs is None:
            Op.set_inputs(self, new)
        else:
            if not len(new) == len(self._inputs):
                raise TypeError("The new inputs are not as many as the previous ones.")
            for i, new in enumerate(new):
                self.set_input(i, new)
        

#     def __init__(self, inputs, outputs, use_self_setters = False):
#         """
#         Initializes the '_inputs' and '_outputs' slots and sets the
#         owner of all outputs to self.

#         If use_self_setters is False, Op::set_input and Op::set_output
#         are used, which do the minimum checks and manipulations. Else,
#         the user defined set_input and set_output functions are
#         called (in any case, all inputs and outputs are initialized
#         to None).
#         """
#         self._inputs = [None] * len(inputs)
#         self._outputs = [None] * len(outputs)

#         if use_self_setters:
#             for i, input in enumerate(inputs):
#                 self.set_input(i, input, validate = False)
#             for i, output in enumerate(outputs):
#                 self.set_output(i, output, validate = False)
#             self.validate()
#         else:
#             for i, input in enumerate(inputs):
#                 Op.set_input(self, i, input, validate = False)
#             for i, output in enumerate(outputs):
#                 Op.set_output(self, i, output, validate = False)
#             self.validate()
        
#         self.validate()

        


#     def set_input(self, i, input, allow_changes = False, validate = True):
#         """
#         Sets the ith input of self.inputs to input. i must be an
#         integer in the range from 0 to len(self.inputs) - 1 and input
#         must be a Result instance. The method may raise a GofTypeError
#         or a GofValueError accordingly to the semantics of the Op, if
#         the new input is of the wrong type or has the wrong
#         properties.

#         If i > len(self.inputs), an IndexError must be raised. If i ==
#         len(self.inputs), it is allowed for the Op to extend the list
#         of inputs if it is a vararg Op, else an IndexError should be
#         raised.

#         For a vararg Op, it is also allowed to have the input
#         parameter set to None for 0 <= i < len(self.inputs), in which
#         case the rest of the inputs will be shifted left. In any other
#         situation, a ValueError should be raised.

#         In some cases, set_input may change some outputs: for example,
#         a change of an input from float to double might require the
#         output's type to also change from float to double. If
#         allow_changes is True, set_input is allowed to perform those
#         changes and must return a list of pairs, each pair containing
#         the old output and the output it was replaced with (they
#         _must_ be different Result instances). See Op::set_output for
#         important information about replacing outputs. If
#         allow_changes is False and some change in the outputs is
#         required for the change in input to be correct, a
#         PropagationError must be raised.

#         This default implementation sets the ith input to input and
#         changes no outputs. It returns None.
#         """
#         previous = self.inputs[i]
#         self.inputs[i] = input
#         if validate:
#             try:
#                 self.validate()
#             except:
#                 # this call gives a subclass the chance to undo the set_outputs 
#                 # that it may have triggered...
#                 # TODO: test this functionality!
#                 self.set_input(i, previous, True, False)


#     def set_output(self, i, output, validate = True):
#         """
#         Sets the ith output to output. The previous output, which is
#         being replaced, must be invalidated using Result::invalidate.
#         The new output must not already have an owner, or its owner must
#         be self. It cannot be a broken link, unless it used to be at this
#         spot, in which case it can be reinstated.

#         For Ops that have vararg output lists, see the regulations in
#         Op::set_input.
#         """
#         if isinstance(output.owner, BrokenLink) \
#            and output.owner.owner is self \
#            and output.owner.index == i:
#             output.revalidate()
#         else:
#             output.set_owner(self, i) # this checks for an already existing owner
#         previous = self.outputs[i]
#         if previous:
#             previous.invalidate()
#         self.outputs[i] = output
#         if validate:
#             try:
#                 self.validate()
#             except:
#                 self.set_output(i, previous, False)


#     def _dontuse_repair(self, allow_changes = False):
#         """
#         This function attempts to repair all inputs that are broken
#         links by calling set_input on the new Result that replaced
#         them. Note that if a set_input operation invalidates one or
#         more outputs, new broken links might appear in the other ops
#         that use this op's outputs.

#         It is possible that the new inputs are inconsistent with this
#         op, in which case an exception will be raised and the previous
#         inputs (and outputs) will be restored.

#         refresh returns a list of (old_output, new_output) pairs
#         detailing the changes, if any.
#         """
#         backtrack = []
#         try:
#             for i, input in enumerate(self.inputs):
#                 link = input.owner
#                 if isinstance(link, BrokenLink):
#                     current = link.owner.outputs[link.index]
#                     dirt = self.set_input(i, current, allow_changes)
#                     backtrack.append((i, input, dirt))
#         except:
#             # Restore the inputs and outputs that were successfully changed.
#             for i, input, dirt in backtrack:
#                 self.inputs[i] = input
#                 if dirt:
#                     for old, new in dirt:
#                         new.invalidate()
#                         old.revalidate()
#                         self.outputs[self.outputs.index(new)] = old
#             raise
#         all_dirt = []
#         for i, input, dirt in backtrack:
#             if dirt:
#                 all_dirt += dirt
#         return all_dirt

        
#     def perform(self):
#         """
#         Performs the computation on the inputs and stores the results
#         in the outputs. This function should check for the validity of
#         the inputs and raise appropriate errors for debugging (for
#         executing without checks, override _perform).

#         An Op may define additional ways to perform the computation
#         that are more efficient (e.g. a piece of C code or a C struct
#         with direct references to the inputs and outputs), but
#         perform() should always be available in order to have a
#         consistent interface to execute graphs.
#         """
#         raise NotImplementedError

#     def _perform(self):
#         """
#         Performs the computation on the inputs and stores the results
#         in the outputs, like perform(), but is not required to check
#         the existence or the validity of the inputs.
#         """
#         return self.perform()

#     @classmethod
#     def require(cls):
#         """
#         Returns a set of Feature subclasses that must be used by any
#         Env manipulating this kind of op. For instance, a Destroyer
#         requires ext.DestroyHandler to guarantee that various
#         destructive operations don't interfere.

#         By default, this collates the __require__ field of this class
#         and the __require__ fields of all classes that are directly or
#         indirectly superclasses to this class into a set.
#         """
#         r = set()

#         bases = all_bases(cls, lambda cls: hasattr(cls, '__env_require__'))

#         for base in bases:
#             req = base.__env_require__
#             if isinstance(req, (list, tuple)):
#                 r.update(req)
#             else:
#                 r.add(req)
#         return r


#     def validate(self):
#         """
#         This class's __validate__ function will be called, as well as
#         the __validate__ functions of all base classes down the class
#         tree. If you do not want to execute __validate__ from the base
#         classes, set the class variable __validate_override__ to True.
#         """
#         vfns = all_bases_collect(self.__class__, 'validate')
#         for vfn in vfns:
#             vfn(self)


#     def __copy__(self):
#         """
#         Copies the inputs list shallowly and copies all the outputs
#         because of the one owner per output restriction.
#         """
#         new_inputs = copy(self.inputs)
#         # We copy the outputs because they are tied to a single Op.
#         new_outputs = [copy(output) for output in self.outputs]
#         op = self.__class__(new_inputs, new_outputs)
#         op._inputs = new_inputs
#         op._outputs = new_outputs
#         for i, output in enumerate(op.outputs):
#             # We adjust _owner and _index manually since the copies
#             # point to the previous op (self).
#             output._owner = op
#             output._index = i
#         return op


#     def __deepcopy__(self, memo):
#         """
#         Not implemented. Use gof.graph.clone(inputs, outputs) to copy
#         a subgraph.
#         """
#         raise NotImplementedError("Use gof.graph.clone(inputs, outputs) to copy a subgraph.")



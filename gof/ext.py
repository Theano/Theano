
from copy import copy
from op import Op
from lib import DummyOp
from features import Listener, Constraint, Orderings
from env import InconsistencyError
from utils import ClsInit
import graph


#TODO: move mark_outputs_as_destroyed to the place that uses this function
#TODO: move Return to where it is used.
__all__ = ['IONames', 'mark_outputs_as_destroyed']


class IONames:
    """
    Requires assigning a name to each of this Op's inputs and outputs.
    """

    __metaclass__ = ClsInit

    input_names = ()
    output_names = ()
    
    @staticmethod
    def __clsinit__(cls, name, bases, dct):
        for names in ['input_names', 'output_names']:
            if names in dct:
                x = getattr(cls, names)
                if isinstance(x, str):
                    x = [x,]
                    setattr(cls, names, x)
                if isinstance(x, (list, tuple)):
                    x = [a for a in x]
                    setattr(cls, names, x)
                    for i, varname in enumerate(x):
                        if not isinstance(varname, str) or hasattr(cls, varname) or varname in ['inputs', 'outputs']:
                            raise TypeError("In %s: '%s' is not a valid input or output name" % (cls.__name__, varname))
                        # Set an attribute for the variable so we can do op.x to return the input or output named "x".
                        setattr(cls, varname,
                                property(lambda op, type=names.replace('_name', ''), index=i:
                                         getattr(op, type)[index]))
                else:
                    print 'ERROR: Class variable %s::%s is neither list, tuple, or string' % (name, names)
                    raise TypeError, str(names)
            else:
                setattr(cls, names, ())

#     def __init__(self, inputs, outputs, use_self_setters = False):
#         assert len(inputs) == len(self.input_names)
#         assert len(outputs) == len(self.output_names)
#         Op.__init__(self, inputs, outputs, use_self_setters)

    def __validate__(self):
        assert len(self.inputs) == len(self.input_names)
        assert len(self.outputs) == len(self.output_names)
                
    @classmethod
    def n_inputs(cls):
        return len(cls.input_names)
        
    @classmethod
    def n_outputs(cls):
        return len(cls.output_names)
        
    def get_by_name(self, name):
        """
        Returns the input or output which corresponds to the given name.
        """
        if name in self.input_names:
            return self.input_names[self.input_names.index(name)]
        elif name in self.output_names:
            return self.output_names[self.output_names.index(name)]
        else:
            raise AttributeError("No such input or output name for %s: %s" % (self.__class__.__name__, name))




class Return(DummyOp):
    """
    Dummy op which represents the action of returning its input
    value to an end user. It "destroys" its input to prevent any
    other Op to overwrite it.
    """
    def destroy_map(self): return {self.out:[self.inputs[0]]}


def mark_outputs_as_destroyed(outputs):
    return [Return(output).out for output in outputs]


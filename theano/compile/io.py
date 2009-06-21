"""Define `SymbolicInput`, `SymbolicOutput`, `In`, `Out` """
__docformat__ = 'restructuredtext en'

from theano import gof

class SymbolicInput(object):
    """
    Represents a symbolic input for use with function or FunctionMaker.

    variable: a Variable instance. 
        This will be assigned a value before running the function,
        not computed from its owner.

    name: Any type. (If autoname=True, defaults to variable.name). 
        If name is a valid Python identifier, this input can be set by kwarg, and its value
        can be accessed by self.<name>.

    update: Variable instance (default: None)
        value (see previous) will be replaced with this expression variable after each function call.
        If update is None, the update will be the default value of the input.

    mutable: Bool (default: False if update is None, True if update is not None)
        True: permit the compiled function to modify the python object being passed as the input
        False: do not permit the compiled function to modify the python object being passed as the input.

    strict: Bool (default: False)
        True: means that the value you pass for this input must have exactly the right type
        False: the value you pass for this input may be casted automatically to the proper type

    autoname: Bool (default: True)
        See the name option.

    implicit: Bool (default: False)
        See help(In). Note that 'None' is not allowed here, since we are in the
        symbolic case.
    """

    def __init__(self, variable, name=None, update=None, mutable=None, strict=False, autoname=True,
            implicit=False):
        assert implicit is not None # Safety check.
        self.variable = variable
        self.name = variable.name if (autoname and name is None) else name
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("name must be a string! (got: %s)" % self.name)
        self.update = update
        self.mutable = mutable if (mutable is not None) else (update is not None)
        self.strict = strict
        self.implicit = implicit

    def __str__(self):
        if self.update:
            return "In(%s -> %s)" % (self.variable, self.update)
        else:
            return "In(%s)" % self.variable

    def __repr__(self):
        return str(self)


class SymbolicInputKit(object):
    """
    Represents a group ("kit") of SymbolicInputs. If fed into function or
    FunctionMaker, only the inputs which are needed to compile the function
    properly will be taken.

    A SymbolicInputKit provides the distribute function in order to set or
    initialize several inputs from a single value. Specialized Kits should
    override it.
    """

    def __init__(self, name):
        if not isinstance(name, str):
            raise TypeError('naem must be a string (got: %s)' % name)
        self.name = name
        self.sinputs = []
        self.variables = []

    def add_input(self, sinput):
        """
        Add a SymbolicInput to this SymbolicInputKit. It will be given the
        next available index.
        """
        self.sinputs.append(sinput)
        self.variables.append(sinput.variable)

    def distribute(self, value, indices, containers):
        """
        Given a list of indices corresponding to SymbolicInputs in this kit
        as well as a corresponding list of containers, initialize all the
        containers using the provided value.
        """
        raise NotImplementedError

    def complete(self, inputs):
        """
        Given inputs (a list of Variable instances), checks through all
        the SymbolicInputs in the kit and return a sorted list of
        indices and a list of their corresponding SymbolicInputs such
        that each of them represents some variable in the inputs list.

        Not all the provided inputs will have a corresponding
        SymbolicInput in the kit.
        """
        ret = []
        for input in inputs:
            try:
                i = self.variables.index(input)
                ret.append((i, self.sinputs[i]))
            except ValueError:
                pass
        ret.sort()
        if not ret:
            return [[], []]
        return zip(*ret)


class In(SymbolicInput):
    """
    Represents a symbolic input for use with function or FunctionMaker.

    variable: a Variable instance. 
        This will be assigned a value before running the function,
        not computed from its owner.

    name: Any type. (If autoname=True, defaults to variable.name). 
        If name is a valid Python identifier, this input can be set by kwarg, and its value
        can be accessed by self.<name>.

    value: Any type.
        The initial/default value for this input. If update is None, this input acts just like
        an argument with a default value in Python. If update is not None, changes to this
        value will "stick around", whether due to an update or a user's explicit action.

    update: Variable instance (default: None)
        value (see previous) will be replaced with this expression variable after each function call.
        If update is None, the update will be the default value of the input.

    mutable: Bool (default: False if update is None, True if update is not None)
        True: permit the compiled function to modify the python object being passed as the input
        False: do not permit the compiled function to modify the python object being passed as the input.

    strict: Bool (default: False)
        True: means that the value you pass for this input must have exactly the right type
        False: the value you pass for this input may be casted automatically to the proper type

    autoname: Bool (default: True)
        See the name option.

    implicit: Bool or None (default: None)
        True: This input is implicit in the sense that the user is not allowed
            to provide a value for it. Requires 'value' to be set. Setting an
            input as implicit allows Theano to directly share containers when
            'value' is an existing container.
        False: The user can provide a value for this input. In this case,
            containers will not be shared (to avoid accidentally overwriting a
            container's content with an input value provided by the user).
        None: Automatically choose between True or False depending on the
            situation. It will be set to False in all cases except if 'value'
            is a container (so that it can be shared by default).
    """
    def __init__(self, variable, name=None, value=None, update=None,
            mutable=None, strict=False, autoname=True,
            implicit=None):
        if implicit is None:
            implicit = isinstance(value, gof.Container)
        super(In, self).__init__(variable, name, update, mutable, strict,
                autoname, implicit = implicit)
        self.value = value
        if self.implicit and value is None:
            raise TypeError('An implicit input must be given a default value')


class SymbolicOutput(object):
    """
    Represents a symbolic output for use with function or FunctionMaker.

    borrow: set this to True to indicate that a reference to
            function's internal storage may be returned. A value
            returned for this output might be clobbered by running
            the function again, but the function might be faster.
    """
    
    def __init__(self, variable, borrow=False):
        self.variable = variable
        self.borrow = borrow

    def __str__(self):
        return "Out(%s)" % self.variable

Out = SymbolicOutput



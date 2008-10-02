

class SymbolicInput(object):
    """
    Represents a symbolic input for use with function or FunctionMaker.

    result: a Result instance. 
        This will be assigned a value before running the function,
        not computed from its owner.

    name: Any type. (If autoname=True, defaults to result.name). 
        If name is a valid Python identifier, this input can be set by kwarg, and its value
        can be accessed by self.<name>.

    update: Result instance (default: None)
        value (see previous) will be replaced with this expression result after each function call.
        If update is None, the update will be the default value of the input.

    mutable: Bool (default: False if update is None, True if update is not None)
        True: permit the compiled function to modify the python object being passed as the input
        False: do not permit the compiled function to modify the python object being passed as the input.

    strict: Bool (default: False)
        True: means that the value you pass for this input must have exactly the right type
        False: the value you pass for this input may be casted automatically to the proper type

    autoname: Bool (default: True)
        See the name option.
    """

    def __init__(self, result, name=None, update=None, mutable=None, strict=False, autoname=True):
        self.result = result
        self.name = result.name if (autoname and name is None) else name
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("name must be a string! (got: %s)" % self.name)
        self.update = update
        self.mutable = mutable if (mutable is not None) else (update is not None)
        self.strict = strict

    def __str__(self):
        if self.update:
            return "In(%s -> %s)" % (self.result, self.update)
        else:
            return "In(%s)" % self.result

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
        self.results = []

    def add_input(self, sinput):
        """
        Add a SymbolicInput to this SymbolicInputKit. It will be given the
        next available index.
        """
        self.sinputs.append(sinput)
        self.results.append(sinput.result)

    def distribute(self, value, indices, containers):
        """
        Given a list of indices corresponding to SymbolicInputs in this kit
        as well as a corresponding list of containers, initialize all the
        containers using the provided value.
        """
        raise NotImplementedError

    def complete(self, inputs):
        """
        Given inputs (a list of Result instances), checks through all
        the SymbolicInputs in the kit and return a sorted list of
        indices and a list of their corresponding SymbolicInputs such
        that each of them represents some result in the inputs list.

        Not all the provided inputs will have a corresponding
        SymbolicInput in the kit.
        """
        ret = []
        for input in inputs:
            try:
                i = self.results.index(input)
                ret.append((i, self.sinputs[i]))
            except ValueError:
                pass
        ret.sort()
        return zip(*ret)


class In(SymbolicInput):
    """
    Represents a symbolic input for use with function or FunctionMaker.

    result: a Result instance. 
        This will be assigned a value before running the function,
        not computed from its owner.

    name: Any type. (If autoname=True, defaults to result.name). 
        If name is a valid Python identifier, this input can be set by kwarg, and its value
        can be accessed by self.<name>.

    value: Any type.
        The initial/default value for this input. If update is None, this input acts just like
        an argument with a default value in Python. If update is not None, changes to this
        value will "stick around", whether due to an update or a user's explicit action.

    update: Result instance (default: None)
        value (see previous) will be replaced with this expression result after each function call.
        If update is None, the update will be the default value of the input.

    mutable: Bool (default: False if update is None, True if update is not None)
        True: permit the compiled function to modify the python object being passed as the input
        False: do not permit the compiled function to modify the python object being passed as the input.

    strict: Bool (default: False)
        True: means that the value you pass for this input must have exactly the right type
        False: the value you pass for this input may be casted automatically to the proper type

    autoname: Bool (default: True)
        See the name option.
    """
    def __init__(self, result, name=None, value=None, update=None, mutable=None, strict=False, autoname=True):
        super(In, self).__init__(result, name, update, mutable, strict, autoname)
        self.value = value


class SymbolicOutput(object):
    """
    Represents a symbolic output for use with function or FunctionMaker.

    borrow: set this to True to indicate that a reference to
            function's internal storage may be returned. A value
            returned for this output might be clobbered by running
            the function again, but the function might be faster.
    """
    
    def __init__(self, result, borrow=False):
        self.result = result
        self.borrow = borrow

Out = SymbolicOutput



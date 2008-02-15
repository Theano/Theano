"""
This file defines the Exceptions that may be raised by graph manipulations.
"""

class GofError(Exception):
    pass


class GofTypeError(GofError):
    pass

class GofValueError(GofError):
    pass


class PropagationError(GofError):
    pass


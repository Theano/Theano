
#Functions related to processing the names of objects

class _TagGenerator:
    """ Class for giving abbreviated tags like to objects.
        Only really intended for internal use """
    def __init__(self):
        self.cur_tag_number = 0

    def get_tag(self):
        rval = self.from_number(self.cur_tag_number)

        self.cur_tag_number += 1

        return rval

    def from_number(self, number):
        """ Converts number to string by rendering it in base 26 using
            capital letters as digits """

        base = 26

        rval = ""

        if number == 0:
            rval = 'A'

        while number != 0:
            remainder = number % base
            new_char = chr(ord('A')+remainder)
            rval = new_char + rval
            number /= base

        return rval



def min_informative_str(obj, indent_level = 0, _prev_obs = None, _tag_generator = None):
    """
    Returns a string specifying to the user what obj is
    The string will print out as much of the graph as is needed
    for the whole thing to be specified in terms only of constants
    or named variables


    Parameters
    ----------
    obj: the name to convert to a string
    indent_level: the number of tabs the tree should start printing at
                  (nested levels of the tree will get more tabs)
    _prev_obs: should only be used to by min_informative_str
                    a dictionary mapping previously converted
                    objects to short tags
    """

    if _prev_obs is None:
        _prev_obs = {}

    indent = '\t' * indent_level


    if obj in _prev_obs:
        tag = _prev_obs[obj]

        return indent + '<' + tag + '>'

    if _tag_generator is None:
        _tag_generator = _TagGenerator()

    cur_tag = _tag_generator.get_tag()

    _prev_obs[obj] = cur_tag


    if hasattr(obj, '__array__'):
        name = '<ndarray>'
    elif hasattr(obj, 'name') and obj.name is not None:
        name = obj.name
    elif hasattr(obj, 'owner') and obj.owner is not None:
        name = str(obj.owner.op)
        for ipt in obj.owner.inputs:
            name += '\n' + min_informative_str(ipt,
                    indent_level = indent_level + 1,
                    _prev_obs = _prev_obs, _tag_generator = _tag_generator)
    else:
        name = str(obj)


    prefix = cur_tag + '. '

    rval = indent + prefix + name

    return rval

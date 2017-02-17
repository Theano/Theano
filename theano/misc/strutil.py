from __future__ import absolute_import, print_function, division
from six.moves import xrange


def render_string(string, sub):
    """
    string: a string, containing formatting instructions
    sub: a dictionary containing keys and values to substitute for
        them.

    returns: string % sub

    The only difference between this function and the % operator
    is that it raises an exception with a more informative error
    message than the % operator does.
    """
    try:
        finalCode = string % sub
    except Exception as E:
        # If unable to render the string, render longer and longer
        # initial substrings until we find the minimal initial substring
        # that causes an error
        i = 0
        while i <= len(string):
            try:
                finalCode = string[0:i] % sub
            except Exception as F:
                if str(F) == str(E):
                    raise Exception(
                        string[0:i] + "<<<< caused exception " + str(F))
            i += 1
        assert False
    return finalCode


def pretty_format(string):
    lines = string.split('\n')

    lines = [strip_leading_white_space(line) for line in lines]

    indent = 0
    for i in xrange(len(lines)):
        indent -= lines[i].count('}')
        if indent < 0:
            indent = 0
        #
        lines[i] = ('    ' * indent) + lines[i]
        indent += lines[i].count('{')
    #

    rval = '\n'.join(lines)

    return rval


def strip_leading_white_space(line):
    while len(line) > 0 and (line[0] == ' ' or line[0] == '\t'):
        line = line[1:]
    return line

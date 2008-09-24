"""script to generate doc/oplist.txt """
__docformat__ = "restructuredtext en"
import sys
import gof

def isOpClass(thing):
    return hasattr(thing, 'perform') and not isinstance(thing, gof.Op)

def isOpConstructor(thing, module):
    return hasattr(thing, 'perform') and isinstance(thing, gof.Op)\
            or thing in getattr(module, '_constructor_list', [])

def print_title(title_string, under_char):
    print title_string
    print under_char * len(title_string)
    print ""

def chomp(s):
    """interpret and left-align a docstring"""

    if 'subtensor' in s: 
        debug = 0
    else:
        debug = 0

    r = []
    leadspace = True
    for c in s:
        if leadspace and c in ' \n\t':
            continue
        else:
            leadspace = False

        if c == '\n':
            if debug:
                print >> sys.stderr, 'breaking'
            break
        if c == '\t': 
            c = ' ';
        r.append(c)

    if debug: 
        print >> sys.stderr, r

    return "".join(r)


import scalar, sparse, tensor

print_title("Theano Op List", "~")
print ""
print ".. contents:: "
print ""

for module in [scalar, sparse, tensor]:
    print_title('module: `%s`' % module.__name__, '=')

    print_title('Op Classes', '-')

    for symbol_name in dir(module):

        symbol = getattr(module, symbol_name)

        if isOpClass(symbol):
            print ""
            print "- :api:`%s.%s`" % (symbol.__module__, symbol_name)
            docstring = getattr(symbol, '__doc__', "")

            if not docstring: 
                print " ", '(no doc)'
            elif len(docstring) < 50:
                print " ", chomp(docstring)
            else:
                print " ", chomp(docstring[:40]), "..."
    # a little trailing whitespace
    print ""

    print_title('Op Constructors', '-')
    for symbol_name in dir(module):

        symbol = getattr(module, symbol_name)

        if isOpConstructor(symbol, module):
            print ""
            print "- :api:`%s.%s`" % (symbol.__module__, symbol_name)
            docstring = getattr(symbol, '__doc__', "")

            if not docstring: 
                print " ", 'No documentation'
            elif len(docstring) < 50:
                print " ", chomp(docstring)
            else:
                print " ", chomp(docstring[:40]), "..."
    # a little trailing whitespace
    print ""


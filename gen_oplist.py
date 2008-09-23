import sys
import gof
import tensor

def isOp(thing):
    return hasattr(thing, 'perform')

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


for module in [tensor]:
    title = 'Ops in module: `%s`' % module.__name__
    print title
    print '-' * len(title)

    for symbol_name in dir(module):

        symbol = getattr(module, symbol_name)

        if isOp(symbol):
            print ""
            print "- :api:`%s.%s`" % (module.__name__, symbol_name)
            docstring = getattr(symbol, '__doc__', "")

            if not docstring: 
                print 'No documentation'
            elif len(docstring) < 50:
                print chomp(docstring)
            else:
                print chomp(docstring[:40]), "..."
    # a little trailing whitespace
    print ""
    print ""


import gof
import tensor

def isOp(thing):
    return hasattr(thing, 'perform')

for module in [tensor]:
    title = 'Ops in module: `%s`' % module.__name__
    print title
    print '-' * len(title)

    for symbol_name in dir(module):

        symbol = getattr(module, symbol_name)

        if isOp(symbol):
            print ""
            print "- `%s.%s`" % (module.__name__, symbol_name)
            docstring = getattr(symbol, '__doc__', "")

            if not docstring: 
                print 'No documentation'
            elif len(docstring) < 50:
                print docstring
            else:
                print docstring[:40], "..."



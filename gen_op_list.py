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



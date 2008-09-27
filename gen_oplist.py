"""script to generate doc/oplist.txt, which compiles to :doc:`oplist`. """
__docformat__ = "restructuredtext en"
import sys
import gof

def print_title(title_string, under_char):
    print title_string
    print under_char * len(title_string)
    print ""


class Entry:
    """Structure for generating the oplist file"""
    symbol = None
    name = None
    module = None
    docstring = None
    tags = []

    def __init__(self, symbol, name, current_module):
        self.symbol = symbol
        self.name = name
        self.module = symbol.__module__ #current_module.__name__ # symbol.__module__
        self.docstring = symbol.__doc__
        self.tags = ['module:%s' % current_module.__name__] + getattr(symbol, '__oplist_tags', [])

    def mini_desc(self, maxlen=50):
        """Return a short description of the op"""
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
                if c in '\t*`': 
                    c = ' ';
                r.append(c)

            if debug: 
                print >> sys.stderr, r

            return "".join(r)

        minmax = 5
        assert maxlen >= minmax
        if not self.docstring: 
            return "" #+ '(no doc)'
        elif len(self.docstring) < maxlen:
            return chomp(self.docstring)
        else:
            return "%s ..."% chomp(self.docstring[:maxlen-minmax])

    apilink = property(lambda self: ":api:`%s.%s`"% (self.module, self.name))
    """Return the ReST link into the epydoc of this symbol"""

class EntryOp(Entry):
    def __init__(self, symbol, *args):
        has_perform = hasattr(symbol, 'perform')
        if symbol is gof.Op:
            raise TypeError('not an Op subclass')
        if not issubclass(symbol, gof.Op):
            raise TypeError('not an Op subclass')
        Entry.__init__(self, symbol, *args)

class EntryConstructor(Entry):
    def __init__(self, symbol, name, module):
        is_op = isinstance(symbol, gof.Op)
        is_ctor = symbol in getattr(module, '__oplist_constructor_list', [])
        if not (is_op or is_ctor):
            raise TypeError('not a constructor', symbol)
        Entry.__init__(self, symbol, name, module)


def search_entries(module_list):
    ops = []
    constructors = []

    for module in module_list:
        symbol_name_list = [s for s in dir(module) if not s[0] == '_']

	for symbol_name in symbol_name_list:
	    symbol = getattr(module, symbol_name)
            try:
                ops.append(EntryOp(symbol, symbol_name, module))
            except TypeError:
                try:
                    constructors.append(EntryConstructor(symbol, symbol_name, module))
                except TypeError:
                    pass

    return ops, constructors

def print_entries(ops, constructors):
    print_title("Theano Op List", "~")
    print ""
    print ".. contents:: "
    print ""

    tags = {}
    for o in ops + constructors:
        for t in o.tags:
            tags.setdefault(t, []).append(o)

    for t in tags:
        print_title(t, '=')

        tagged_ops = [op for op in tags[t] if isinstance(op, EntryOp)]
        if len(tagged_ops):
            print_title('Op Classes', '-')
            for op in tagged_ops:
                print "- %s" % op.apilink
                print "  %s" % op.mini_desc()
                print ""

        tagged_ops = [op for op in tags[t] if isinstance(op, EntryConstructor)]
        if len(tagged_ops):
            print_title('Op Constructors', '-')
            for op in tagged_ops:
                print "- %s" % op.apilink
                print "  %s" % op.mini_desc()
                print ""


if __name__ == "__main__":
    """Generate the op list"""
    import scalar, sparse, tensor

    ops, constructors = search_entries([scalar, sparse, tensor])
    print_entries(ops, constructors)

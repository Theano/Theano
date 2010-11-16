"""script to generate doc/oplist.txt, which compiles to :doc:`oplist`. """
__docformat__ = "restructuredtext en"
import sys, os
throot = "/".join(sys.path[0].split("/")[:-2])
sys.path[0:0] = [throot]

from theano import gof

def print_title(file, title_string, under_char, over_char=''):

    l = len(title_string)
    if over_char:
        print >>file, over_char * l

    print >>file, title_string

    if under_char:
        print >>file, under_char * l

    print >>file, ""

def print_hline(file):
    print >>file, '-' * 80

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
        self.module = symbol.__module__ # current_module.__name__ # symbol.__module__
        self.docstring = symbol.__doc__
        self.tags = ['%s' % current_module.__name__] + getattr(symbol, '__oplist_tags', [])

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

    apilink = property(lambda self: ":api:`%s <%s.%s>`"% (self.name, self.module, self.name))
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


def search_entries(module_list, ops = None, constructors = None, seen = None):
    if ops is None: ops = []
    if constructors is None: constructors = []
    if seen is None: seen = set()
    modules = []

    for module in module_list:
        symbol_name_list = [s for s in dir(module) if not s[0] == '_']
        
        for symbol_name in symbol_name_list:
            symbol = getattr(module, symbol_name)
            try:
                if symbol in seen:
                    continue
                seen.add(symbol)
            except TypeError:
                pass
            if type(symbol) == type(module): # module
                modules.append(symbol)
            try:
                ops.append(EntryOp(symbol, symbol_name, module))
            except TypeError:
                try:
                    constructors.append(EntryConstructor(symbol, symbol_name, module))
                except TypeError:
                    pass

    for symbol in modules:
        search_entries([symbol], ops, constructors, seen)

    return ops, constructors

def print_entries(file, ops, constructors):
    tags = {}
    for o in ops + constructors:
        for t in o.tags:
            tags.setdefault(t, []).append(o)

    for t in tags:
        print_title(file, t, '=')

        tagged_ops = [op for op in tags[t] if isinstance(op, EntryOp)]
        if len(tagged_ops):
            print_title(file, 'Op Classes', '-')
            for op in tagged_ops:
                print >>file, "- %s" % op.apilink
                print >>file, "  %s" % op.mini_desc()
                print >>file, ""

        tagged_ops = [op for op in tags[t] if isinstance(op, EntryConstructor)]
        if len(tagged_ops):
            print_title(file, 'Op Constructors', '-')
            for op in tagged_ops:
                print >>file, "- %s" % op.apilink
                print >>file, "  %s" % op.mini_desc()
                print >>file, ""


def print_file(file):

    print >>file, '.. _oplist:\n\n'

    print_title(file, "Op List", "~", "~")
    print >>file, """
This page lists the `Op Classes` and `constructors` that are provided by the Theano library.
`Op Classes` drive from :api:`Op`, whereas `constructors` are typically `Op Class` instances, but may be true Python functions.

In the future, this list may distinguish `constructors` that are Op instances from true Python functions.

"""
    print_hline(file)
    print >>file, ""
    print >>file, ".. contents:: "
    print >>file, ""

    ops, constructors = search_entries([theano])

    print_entries(file, ops, constructors)

    print >>file, ""
    

import theano

if __name__ == "__main__":
    """Generate the op list"""

    if len(sys.argv) >= 2:
        file = open(sys.argv[1], 'w')
    else:
        file = sys.stdout

    print_file(file)


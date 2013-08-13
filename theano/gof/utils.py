import re
import traceback

from theano import config


def add_tag_trace(thing):
    """Add tag.trace to an node or variable.

    The argument is returned after being affected (inplace).
    """
    limit = config.traceback.limit
    if limit == -1:
        limit = None
    thing.tag.trace = traceback.extract_stack(limit=limit)[:-1]
    return thing


def hashgen():
    hashgen.next += 1
    return  hashgen.next
hashgen.next = 0


def hashtype(self):
    t = type(self)
    return hash(t.__name__) ^ hash(t.__module__)


class MethodNotDefined(Exception):
    """
    To be raised by functions defined as part of an interface.

    When the user sees such an error, it is because an important interface
    function has been left out of an implementation class.
    """


class object2(object):
    __slots__ = []
    if 0:
        def __hash__(self):
            # this fixes silent-error-prone new-style class behavior
            if hasattr(self, '__eq__') or hasattr(self, '__cmp__'):
                raise TypeError("unhashable object: %s" % self)
            return id(self)

    def __ne__(self, other):
        return not self == other


class scratchpad:
    def clear(self):
        self.__dict__.clear()

    def __update__(self, other):
        self.__dict__.update(other.__dict__)
        return self

    def __str__(self):
        return "scratchpad" + str(self.__dict__)

    def __repr__(self):
        return "scratchpad" + str(self.__dict__)

    def info(self):
        print "<theano.gof.utils.scratchpad instance at %i>" % id(self)
        for k, v in self.__dict__.items():
            print "  %s: %s" % (k, v)


class D:
    def __init__(self, **d):
        self.__dict__.update(d)


def memoize(f):
    """Cache the return value for each tuple of arguments
    (which must be hashable) """
    cache = {}

    def rval(*args, **kwargs):
        kwtup = tuple(kwargs.items())
        key = (args, kwtup)
        if key not in cache:
            val = f(*args, **kwargs)
            cache[key] = val
        else:
            val = cache[key]
        return val

    return rval


def deprecated(filename, msg=''):
    """Decorator which will print a warning message on the first call.

    Use it like this:

    @deprecated('myfile', 'do something different...')
    def fn_name(...)
        ...

    And it will print

    WARNING myfile.fn_name deprecated. do something different...

    """
    def _deprecated(f):
        printme = [True]

        def g(*args, **kwargs):
            if printme[0]:
                print 'WARNING: %s.%s deprecated. %s'\
                        % (filename, f.__name__, msg)
                printme[0] = False
            return f(*args, **kwargs)
        return g

    return _deprecated


def uniq(seq):
    """
    Do not use set, this must always return the same value at the same index.
    If we just exchange other values, but keep the same pattern of duplication,
    we must keep the same order.
    """
    #TODO: consider building a set out of seq so that the if condition
    #is constant time -JB
    return [x for i, x in enumerate(seq) if seq.index(x) == i]


def difference(seq1, seq2):
    """
    Returns all elements in seq1 which are not in seq2: i.e seq1\seq2
    """
    try:
        # try to use O(const * len(seq1)) algo
        if len(seq2) < 4:  # I'm guessing this threshold -JB
            raise Exception('not worth it')
        set2 = set(seq2)
        return [x for x in seq1 if x not in set2]
    except Exception, e:
        # maybe a seq2 element is not hashable
        # maybe seq2 is too short
        # -> use O(len(seq1) * len(seq2)) algo
        return [x for x in seq1 if x not in seq2]


def partition(f, seq):
    seqt = []
    seqf = []
    for elem in seq:
        if f(elem):
            seqt.append(elem)
        else:
            seqf.append(elem)
    return seqt, seqf


def attr_checker(*attrs):
    def f(candidate):
        for attr in attrs:
            if not hasattr(candidate, attr):
                return False
        return True

    f.__doc__ = ("Checks that the candidate has the following attributes: %s"
                  % ", ".join(["'%s'" % attr for attr in attrs]))
    return f


def all_bases(cls, accept):
    rval = set([cls])
    for base in cls.__bases__:
        rval.update(all_bases(base, accept))
    return [cls for cls in rval if accept(cls)]


def all_bases_collect(cls, raw_name):
    rval = set()
    name = "__%s__" % raw_name
    if name in cls.__dict__:  # don't use hasattr
        rval.add(getattr(cls, name))
    cut = "__%s_override__" % raw_name
    if not cls.__dict__.get(cut, False):
        for base in cls.__bases__:
            rval.update(all_bases_collect(base, raw_name))
    return rval


def camelcase_to_separated(string, sep="_"):
    return re.sub('(.)([A-Z])', '\\1%s\\2' % sep, string).lower()


def to_return_values(values):
    if len(values) == 1:
        return values[0]
    else:
        return values


def from_return_values(values):
    if isinstance(values, (list, tuple)):
        return values
    else:
        return [values]


class ClsInit(type):
    """Class initializer for L{Op} subclasses"""
    def __init__(cls, name, bases, dct):
        """
        Validate and initialize the L{Op} subclass 'cls'

        This function:
          - changes class attributes input_names and output_names to be lists
            if they are single strings.
        """
        type.__init__(cls, name, bases, dct)

        cls.__clsinit__(cls, name, bases, dct)


def toposort(prereqs_d):
    """
    Sorts prereqs_d.keys() topologically.

    prereqs_d[x] contains all the elements that must come before x
    in the ordering.
    """

#     all1 = set(prereqs_d.keys())
#     all2 = set()
#     for x, y in prereqs_d.items():
#         all2.update(y)
#     print all1.difference(all2)

    seq = []
    done = set()
    postreqs_d = {}
    for x, prereqs in prereqs_d.items():
        for prereq in prereqs:
            postreqs_d.setdefault(prereq, set()).add(x)
    next = set([k for k in prereqs_d if not prereqs_d[k]])
    while next:
        bases = next
        next = set()
        for x in bases:
            done.add(x)
            seq.append(x)
        for x in bases:
            for postreq in postreqs_d.get(x, []):
                if not prereqs_d[postreq].difference(done):
                    next.add(postreq)
    if len(prereqs_d) != len(seq):
        raise Exception("Cannot sort topologically: there might be cycles, "
                        "prereqs_d does not have a key for each element or "
                        "some orderings contain invalid elements.")
    return seq


def print_for_dot(self):
    #TODO: popen2("dot -Tpng | display") and actually make the graph window
    #pop up
    print ("digraph unix { size = '6,6'; node [color = lightblue2;"
           "style = filled];")
    for op in self.order:
        for input in op.inputs:
            if input.owner:
                print ' '.join((
                    input.owner.__class__.__name__ + str(abs(id(input.owner))),
                    " -> ",
                    op.__class__.__name__ + str(abs(id(op))),
                    ";"))


class Keyword:

    def __init__(self, name, nonzero=True):
        self.name = name
        self.nonzero = nonzero

    def __nonzero__(self):
        return self.nonzero

    def __str__(self):
        return "<%s>" % self.name

    def __repr__(self):
        return "<%s>" % self.name

ABORT = Keyword("ABORT", False)
RETRY = Keyword("RETRY", False)
FAILURE = Keyword("FAILURE", False)


simple_types = (int, float, str, bool, None.__class__, Keyword)


ANY_TYPE = Keyword("ANY_TYPE")
FALL_THROUGH = Keyword("FALL_THROUGH")


def comm_guard(type1, type2):
    def wrap(f):
        old_f = f.func_globals[f.__name__]

        def new_f(arg1, arg2, *rest):
            if (type1 is ANY_TYPE or isinstance(arg1, type1)) \
                   and (type2 is ANY_TYPE or isinstance(arg2, type2)):
                pass
            elif (type1 is ANY_TYPE or isinstance(arg2, type1)) \
                     and (type2 is ANY_TYPE or isinstance(arg1, type2)):
                arg1, arg2 = arg2, arg1
            else:
                return old_f(arg1, arg2, *rest)

            variable = f(arg1, arg2, *rest)
            if variable is FALL_THROUGH:
                return old_f(arg1, arg2, *rest)
            else:
                return variable

        new_f.__name__ = f.__name__

        def typename(type):
            if isinstance(type, Keyword):
                return str(type)
            elif isinstance(type, (tuple, list)):
                return "(" + ", ".join([x.__name__ for x in type]) + ")"
            else:
                return type.__name__

        new_f.__doc__ = (str(old_f.__doc__) + "\n" +
                ", ".join([typename(type) for type in (type1, type2)]) +
                "\n" + str(f.__doc__ or ""))
        return new_f

    return wrap


def type_guard(type1):
    def wrap(f):
        old_f = f.func_globals[f.__name__]

        def new_f(arg1, *rest):
            if (type1 is ANY_TYPE or isinstance(arg1, type1)):
                variable = f(arg1, *rest)
                if variable is FALL_THROUGH:
                    return old_f(arg1, *rest)
                else:
                    return variable
            else:
                return old_f(arg1, *rest)

        new_f.__name__ = f.__name__

        def typename(type):
            if isinstance(type, Keyword):
                return str(type)
            elif isinstance(type, (tuple, list)):
                return "(" + ", ".join([x.__name__ for x in type]) + ")"
            else:
                return type.__name__

        new_f.__doc__ = (str(old_f.__doc__) + "\n" +
                ", ".join([typename(type) for type in (type1,)]) +
                "\n" + str(f.__doc__ or ""))
        return new_f

    return wrap


def flatten(a):
    if isinstance(a, (tuple, list, set)):
        l = []
        for item in a:
            l.extend(flatten(item))
        return l
    else:
        return [a]


def unique(x):
    return len(set(x)) == len(x)


def hist(coll):
    counts = {}
    for elem in coll:
        counts[elem] = counts.get(elem, 0) + 1
    return counts


def give_variables_names(variables):
    """ Gives unique names to an iterable of variables. Modifies input.

    This function is idempotent."""
    names = map(lambda var: var.name, variables)
    h = hist(names)
    bad_var = lambda var: not var.name or h[var.name] > 1

    for i, var in enumerate(filter(bad_var, variables)):
        var.name = (var.name or "") + "_%d" % i

    if not unique(map(str, variables)):
        raise ValueError("Not all variables have unique names."
                "Maybe you've named some of the variables identically")

    return variables


def remove(predicate, coll):
    """ Return those items of collection for which predicate(item) is true.

    >>> from itertoolz import remove
    >>> def even(x):
    ...     return x % 2 == 0
    >>> remove(even, [1, 2, 3, 4])
    [1, 3]
    """
    return filter(lambda x: not predicate(x), coll)

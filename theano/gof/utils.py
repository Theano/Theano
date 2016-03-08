from __future__ import print_function
import linecache
import sys

import numpy
from six import iteritems

from theano import config
from theano.compat import OrderedDict, PY3


def simple_extract_stack(f=None, limit=None, skips=[]):
    """This is traceback.extract_stack from python 2.7 with this change:

    - Comment the update of the cache.
    - Skip internal stack trace level.

    The update of the cache call os.stat to verify is the cache is up
    to date.  This take too much time on cluster.

    limit - The number of stack level we want to return. If None, mean
    all what we can.

    skips - partial path of stack level we don't want to keep and count.
        When we find one level that isn't skipped, we stop skipping.

    """
    if f is None:
        try:
            raise ZeroDivisionError
        except ZeroDivisionError:
            f = sys.exc_info()[2].tb_frame.f_back
    if limit is None:
        if hasattr(sys, 'tracebacklimit'):
            limit = sys.tracebacklimit
    trace = []
    n = 0
    while f is not None and (limit is None or n < limit):
        lineno = f.f_lineno
        co = f.f_code
        filename = co.co_filename
        name = co.co_name
#        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        if line:
            line = line.strip()
        else:
            line = None
        f = f.f_back

        # Just skip inner level
        if len(trace) == 0:
            rm = False
            for p in skips:
                # Julian: I added the 'tests' exception together with
                # Arnaud.  Otherwise, we'd lose the stack trace during
                # in our test cases (e.g. in test_opt.py). We're not
                # sure this is the right way to do it though.
                if p in filename and 'tests' not in filename:
                    rm = True
                    break
            if rm:
                continue
        trace.append((filename, lineno, name, line))
        n = n + 1
    trace.reverse()
    return trace


def add_tag_trace(thing, user_line=None):
    """
    Add tag.trace to an node or variable.

    The argument is returned after being affected (inplace).

    Parameters
    ----------
    thing
        The object where we add .tag.trace.
    user_line
        The max number of user line to keep.

    Notes
    -----
    We alse use config.traceback.limit for the maximum number of stack level
    we look.

    """
    if user_line is None:
        user_line = config.traceback.limit

    if user_line == -1:
        user_line = None
    skips = ["theano/tensor/", "theano\\tensor\\",
             "theano/compile/", "theano\\compile\\",
             "theano/gof/", "theano\\gof\\",
             "theano/scalar/basic.py", "theano\\scalar\\basic.py",
             "theano/sandbox/", "theano\\sandbox\\",
             "theano/scan_module/", "theano\\scan_module\\",
             "theano/sparse/", "theano\\sparse\\",
             "theano/typed_list/", "theano\\typed_list\\"]
    tr = simple_extract_stack(limit=user_line, skips=skips)
    # Different python version use different sementic for
    # limit. python 2.7 include the call to extrack_stack. The -1 get
    # rid of it.

    if tr:
        thing.tag.trace = [tr]
    else:
        thing.tag.trace = tr
    return thing


def hashtype(self):
    t = type(self)
    return hash(t.__name__) ^ hash(t.__module__)


# Object to mark that a parameter is undefined (useful in cases where
# None is a valid value with defined semantics)
undef = object()


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
        print("<theano.gof.utils.scratchpad instance at %i>" % id(self))
        for k, v in iteritems(self.__dict__):
            print("  %s: %s" % (k, v))


class D:
    def __init__(self, **d):
        self.__dict__.update(d)


def memoize(f):
    """
    Cache the return value for each tuple of arguments (which must be hashable).

    """
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
    """
    Decorator which will print a warning message on the first call.

    Use it like this::

      @deprecated('myfile', 'do something different...')
      def fn_name(...)
          ...

    And it will print::

      WARNING myfile.fn_name deprecated. do something different...

    """
    def _deprecated(f):
        printme = [True]

        def g(*args, **kwargs):
            if printme[0]:
                print('WARNING: %s.%s deprecated. %s' %
                      (filename, f.__name__, msg))
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
    # TODO: consider building a set out of seq so that the if condition
    # is constant time -JB
    return [x for i, x in enumerate(seq) if seq.index(x) == i]


def difference(seq1, seq2):
    """
    Returns all elements in seq1 which are not in seq2: i.e ``seq1\seq2``.

    """
    try:
        # try to use O(const * len(seq1)) algo
        if len(seq2) < 4:  # I'm guessing this threshold -JB
            raise Exception('not worth it')
        set2 = set(seq2)
        return [x for x in seq1 if x not in set2]
    except Exception:
        # maybe a seq2 element is not hashable
        # maybe seq2 is too short
        # -> use O(len(seq1) * len(seq2)) algo
        return [x for x in seq1 if x not in seq2]


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


def toposort(prereqs_d):
    """
    Sorts prereqs_d.keys() topologically.

    prereqs_d[x] contains all the elements that must come before x
    in the ordering.

    """

#     all1 = set(prereqs_d.keys())
#     all2 = set()
#     for x, y in iteritems(prereqs_d):
#         all2.update(y)
#     print all1.difference(all2)

    seq = []
    done = set()
    postreqs_d = {}
    for x, prereqs in iteritems(prereqs_d):
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


class Keyword:

    def __init__(self, name, nonzero=True):
        self.name = name
        self.nonzero = nonzero

    def __nonzero__(self):
        # Python 2.x
        return self.__bool__()

    def __bool__(self):
        # Python 3.x
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
        old_f = f.__globals__[f.__name__]

        def new_f(arg1, arg2, *rest):
            if ((type1 is ANY_TYPE or isinstance(arg1, type1)) and
                    (type2 is ANY_TYPE or isinstance(arg2, type2))):
                pass
            elif ((type1 is ANY_TYPE or isinstance(arg2, type1)) and
                  (type2 is ANY_TYPE or isinstance(arg1, type2))):
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
                         ", ".join([typename(type)
                                    for type in (type1, type2)]) +
                         "\n" + str(f.__doc__ or ""))
        return new_f

    return wrap


def type_guard(type1):
    def wrap(f):
        old_f = f.__globals__[f.__name__]

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
    """
    Recursively flatten tuple, list and set in a list.

    """
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
    """
    Gives unique names to an iterable of variables. Modifies input.

    This function is idempotent.

    """
    names = [var.name for var in variables]
    h = hist(names)

    def bad_var(var):
        return not var.name or h[var.name] > 1

    for i, var in enumerate(filter(bad_var, variables)):
        var.name = (var.name or "") + "_%d" % i

    if not unique([str(v) for v in variables]):
        raise ValueError("Not all variables have unique names. Maybe you've "
                         "named some of the variables identically")
    return variables


def remove(predicate, coll):
    """
    Return those items of collection for which predicate(item) is true.

    Examples
    --------
    >>> def even(x):
    ...     return x % 2 == 0
    >>> remove(even, [1, 2, 3, 4])
    [1, 3]

    """
    return [x for x in coll if not predicate(x)]


if PY3:
    import hashlib

    def hash_from_code(msg):
        # hashlib.md5() requires an object that supports buffer interface,
        # but Python 3 (unicode) strings don't.
        if isinstance(msg, str):
            msg = msg.encode()
        # Python 3 does not like module names that start with
        # a digit.
        return 'm' + hashlib.md5(msg).hexdigest()

else:
    import hashlib

    def hash_from_code(msg):
        try:
            return hashlib.md5(msg).hexdigest()
        except TypeError:
            assert isinstance(msg, numpy.ndarray)
            return hashlib.md5(numpy.getbuffer(msg)).hexdigest()


def hash_from_file(file_path):
    """
    Return the MD5 hash of a file.

    """
    return hash_from_code(open(file_path, 'rb').read())


def hash_from_dict(d):
    """
    Work around the fact that dict are not hashable in python.

    This request that all object have a sorted order that depend only
    on the key of the object. We support only integer/float/string keys.

    Also, we transform values that are list into tuple as list are not
    hashable.

    Notes
    -----
    Special case for OrderedDict, it use the order of the dict,
    so the key don't need to be sortable.

    """
    if isinstance(d, OrderedDict):
        items = list(iteritems(d))
    else:
        items = list(d.items())
        items.sort()
    first_part = [k for k, v in items]
    second_part = []
    for k, v in items:
        assert isinstance(k, (str, int, float))
        if isinstance(v, (tuple, list)):
            second_part += [tuple(v)]
        else:
            second_part += [v]
    tuple_items = tuple(first_part + second_part + [d.__class__])
    return hash(tuple_items)

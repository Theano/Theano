
# import op
# import result


class OmegaError(Exception): pass

class AbstractFunctionError(Exception): 
    """To be raised by functions defined as part of an interface.

    When the user sees such an error, it is because an important interface
    function has been left out of an implementation class.
    """


def all_bases(cls, accept):
    rval = set([cls])
    for base in cls.__bases__:
        rval.update(all_bases(base, accept))
    return [cls for cls in rval if accept(cls)]



def all_bases_collect(cls, raw_name):
    rval = set()
    name = "__%s__" % raw_name
    if name in cls.__dict__: # don't use hasattr
        rval.add(getattr(cls, name))
    cut = "__%s_override__" % raw_name
    if not cls.__dict__.get(cut, False):
        for base in cls.__bases__:
            rval.update(all_bases_collect(base, raw_name))
    return rval



def uniq_features(_features, *_rest):
    """Return a list such that no element is a subclass of another"""
    # used in Env.__init__ to 
    features = [x for x in _features]
    for other in _rest:
        features += [x for x in other]
    res = []
    while features:
        feature = features.pop()
        for feature2 in features:
            if issubclass(feature2, feature):
                break
        else:
            res.append(feature)
    return res



def partial(func, *args, **keywords):
    def newfunc(*fargs, **fkeywords):
        newkeywords = keywords.copy()
        newkeywords.update(fkeywords)
        return func(*(args + fargs), **newkeywords)
    newfunc.func = func
    newfunc.args = args
    newfunc.keywords = keywords
    return newfunc


class ClsInit(type):
    """Class initializer for Op subclasses"""
    def __init__(cls, name, bases, dct):
        """Validate and initialize the Op subclass 'cls'

        This function:
        - changes class attributes input_names and output_names to be lists if they are single strings.
        """
        type.__init__(cls, name, bases, dct)

        cls.__clsinit__(cls, name, bases, dct)


def toposort(prereqs_d):
    """
    Sorts prereqs_d.keys() topologically. prereqs_d[x] contains all the elements
    that must come before x in the ordering.
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
    next = set(k for k in prereqs_d if not prereqs_d[k])
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
        raise Exception("Cannot sort topologically: there might be cycles, " + \
                        "prereqs_d does not have a key for each element or " + \
                        "some orderings contain invalid elements.")
    return seq


# def schedule(**kwargs):
    
#     after = kwargs.get('after', [])
#     if not isinstance(after, (list, tuple)):
#         after = [after]
        
#     before = kwargs.get('before', [])
#     if not isinstance(before, (list, tuple)):
#         before = [before]

#     def decorate(fn):
#         name = fn.__name__
#         fn.prereqs_d = {}
#         for postreq in after:
#             prereqs_d[postreq] = name
#         for prereq in before:
#             prereqs_d[name] = prereq
#         return fn
    
#     return decorate


# def after(*others):
#     return schedule(after = others)


# def before(*others):
#     return schedule(before = others)


# class TopoList(list):

#     def add(self, item, **kwargs):
#         after = kwargs.get('after', [])
#         if not isinstance(after, (list, tuple)):
#             after = [after]
        
#         before = kwargs.get('before', [])
#         if not isinstance(before, (list, tuple)):
#             before = [before]
        




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
                try:
                    return old_f(arg1, arg2, *rest)
                except:
                    raise

            try:
                result = f(arg1, arg2, *rest)
            except:
                raise
            if result is FALL_THROUGH:
                try:
                    return old_f(arg1, arg2, *rest)
                except:
                    raise
            else:
                return result

        new_f.__name__ = f.__name__
        def typename(type):
            if isinstance(type, Keyword):
                return str(type)
            elif isinstance(type, (tuple, list)):
                return "(" + ", ".join([x.__name__ for x in type]) + ")"
            else:
                return type.__name__
        new_f.__doc__ = str(old_f.__doc__) + "\n" + ", ".join([typename(type) for type in (type1, type2)]) + "\n" + str(f.__doc__ or "")
        return new_f
    return wrap


def type_guard(type1):
    def wrap(f):
        old_f = f.func_globals[f.__name__]
        def new_f(arg1, *rest):
            if (type1 is ANY_TYPE or isinstance(arg1, type1)):
                result = f(arg1, *rest)
                if result is FALL_THROUGH:
                    return old_f(arg1, *rest)
                else:
                    return result
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
        new_f.__doc__ = str(old_f.__doc__) + "\n" + ", ".join([typename(type) for type in (type1,)]) + "\n" + str(f.__doc__ or "")
        return new_f
    return wrap


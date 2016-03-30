"""
If you have two expressions containing unification variables, these expressions
can be "unified" if there exists an assignment to all unification variables
such that the two expressions are equal.

For instance, [5, A, B] and [A, C, 9] can be unified if A=C=5 and B=9,
yielding [5, 5, 9]. 
[5, [A, B]] and [A, [1, 2]] cannot be unified because there is no value for A
that satisfies the constraints. That's useful for pattern matching.

"""
from __future__ import absolute_import, print_function, division
from copy import copy

from functools import partial
from theano.gof.utils import ANY_TYPE, comm_guard, FALL_THROUGH, iteritems


################################


class Variable:
    """
    Serves as a base class of variables for the purpose of unification.
    "Unification" here basically means matching two patterns, see the
    module-level docstring.

    Behavior for unifying various types of variables should be added as
    overloadings of the 'unify' function.

    Notes
    -----
    There are two Variable classes in theano and this is the more rarely used
    one.
    This class is used internally by the PatternSub optimization,
    and possibly other subroutines that have to perform graph queries.
    If that doesn't sound like what you're doing, the Variable class you
    want is probably theano.gof.graph.Variable.

    """
    def __init__(self, name="?"):
        self.name = name

    def __str__(self):
        return (self.__class__.__name__ + "(" +
                ", ".join("%s=%s" % (key, value)
                          for key, value in iteritems(self.__dict__)) + ")")

    def __repr__(self):
        return str(self)


class FreeVariable(Variable):
    """
    This Variable can take any value.

    """

    pass


class BoundVariable(Variable):
    """
    This Variable is bound to a value accessible via the value field.

    """

    def __init__(self, name, value):
        self.name = name
        self.value = value


class OrVariable(Variable):
    """
    This Variable could be any value from a finite list of values,
    accessible via the options field.

    """

    def __init__(self, name, options):
        self.name = name
        self.options = options


class NotVariable(Variable):
    """
    This Variable can take any value but a finite amount of forbidden
    values, accessible via the not_options field.

    """

    def __init__(self, name, not_options):
        self.name = name
        self.not_options = not_options


class VariableInList:  # not a subclass of Variable
    """
    This special kind of variable is matched against a list and unifies
    an inner Variable to an OrVariable of the values in the list.
    For example, if we unify VariableInList(FreeVariable('x')) to [1,2,3],
    the 'x' variable is unified to an OrVariable('?', [1,2,3]).

    """

    def __init__(self, variable):
        self.variable = variable


################################


_all = {}


def var_lookup(vartype, name, *args, **kwargs):
    sig = (vartype, name)
    if sig in _all:
        return _all[sig]
    else:
        v = vartype(name, *args)
        _all[sig] = v
        return v

Var = partial(var_lookup, FreeVariable)
V = Var
OrV = partial(var_lookup, OrVariable)
NV = partial(var_lookup, NotVariable)


################################


class Unification:
    """
    This class represents a possible unification of a group of variables
    with each other or with tangible values.

    
    Parameters
    ----------
    inplace : bool
        If inplace is False, the merge method will return a new Unification
        that is independent from the previous one (which allows backtracking).

    """

    def __init__(self, inplace=False):
        self.unif = {}
        self.inplace = inplace

    def merge(self, new_best, *vars):
        """
        Links all the specified vars to a Variable that represents their
        unification.

        """
        if self.inplace:
            U = self
        else:
            # Copy all the unification data.
            U = Unification(self.inplace)
            for var, (best, pool) in iteritems(self.unif):
                # The pool of a variable is the set of all the variables that
                # are unified to it (all the variables that must have the same
                # value). The best is the Variable that represents a set of
                # values common to all the variables in the pool.
                U.unif[var] = (best, pool)
        # We create a new pool for our new set of unified variables, initially
        # containing vars and new_best
        new_pool = set(vars)
        new_pool.add(new_best)
        for var in copy(new_pool):
            best, pool = U.unif.get(var, (var, set()))
            # We now extend the new pool to contain the pools of all the variables.
            new_pool.update(pool)
        # All variables get the new pool.
        for var in new_pool:
            U.unif[var] = (new_best, new_pool)
        return U

    def __getitem__(self, v):
        """
        For a variable v, returns a Variable that represents the tightest
        set of possible values it can take.

        """
        return self.unif.get(v, (v, None))[0]


################################


def unify_walk(a, b, U):
    """
    unify_walk(a, b, U) returns an Unification where a and b are unified,
    given the unification that already exists in the Unification U. If the
    unification fails, it returns False.

    There are two ways to expand the functionality of unify_walk. The first way
    is:
    @comm_guard(type_of_a, type_of_b)
    def unify_walk(a, b, U):
        ...
    A function defined as such will be executed whenever the types of a and b
    match the declaration. Note that comm_guard automatically guarantees that
    your function is commutative: it will try to match the types of a, b or
    b, a.
    It is recommended to define unify_walk in that fashion for new types of
    Variable because different types of Variable interact a lot with each other,
    e.g. when unifying an OrVariable with a NotVariable, etc. You can return
    the special marker FALL_THROUGH to indicate that you want to relay execution
    to the next match of the type signature. The definitions of unify_walk are
    tried in the reverse order of their declaration.

    Another way is to override __unify_walk__ in an user-defined class.

    Limitations: cannot embed a Variable in another (the functionality could
    be added if required)

    Here is a list of unification rules with their associated behavior:

    """
    if a.__class__ != b.__class__:
        return False
    elif a == b:
        return U
    else:
        return False


@comm_guard(FreeVariable, ANY_TYPE)
def unify_walk(fv, o, U):
    """
    FreeV is unified to BoundVariable(other_object).

    """
    v = BoundVariable("?", o)
    return U.merge(v, fv)


@comm_guard(BoundVariable, ANY_TYPE)
def unify_walk(bv, o, U):
    """
    The unification succeed iff BV.value == other_object.

    """
    if bv.value == o:
        return U
    else:
        return False


@comm_guard(OrVariable, ANY_TYPE)
def unify_walk(ov, o, U):
    """
    The unification succeeds iff other_object in OrV.options.

    """
    if o in ov.options:
        v = BoundVariable("?", o)
        return U.merge(v, ov)
    else:
        return False


@comm_guard(NotVariable, ANY_TYPE)
def unify_walk(nv, o, U):
    """
    The unification succeeds iff other_object not in NV.not_options.

    """
    if o in nv.not_options:
        return False
    else:
        v = BoundVariable("?", o)
        return U.merge(v, nv)


@comm_guard(FreeVariable, Variable)
def unify_walk(fv, v, U):
    """
    Both variables are unified.

    """
    v = U[v]
    return U.merge(v, fv)


@comm_guard(BoundVariable, Variable)
def unify_walk(bv, v, U):
    """
    V is unified to BV.value.

    """
    return unify_walk(v, bv.value, U)


@comm_guard(OrVariable, OrVariable)
def unify_walk(a, b, U):
    """
    OrV(list1) == OrV(list2) == OrV(intersection(list1, list2))

    """
    opt = intersection(a.options, b.options)
    if not opt:
        return False
    elif len(opt) == 1:
        v = BoundVariable("?", opt[0])
    else:
        v = OrVariable("?", opt)
    return U.merge(v, a, b)


@comm_guard(NotVariable, NotVariable)
def unify_walk(a, b, U):
    """
    NV(list1) == NV(list2) == NV(union(list1, list2))

    """
    opt = union(a.not_options, b.not_options)
    v = NotVariable("?", opt)
    return U.merge(v, a, b)


@comm_guard(OrVariable, NotVariable)
def unify_walk(o, n, U):
    """
    OrV(list1) == NV(list2) == OrV(list1 \ list2)

    """
    opt = [x for x in o.options if x not in n.not_options]
    if not opt:
        return False
    elif len(opt) == 1:
        v = BoundVariable("?", opt[0])
    else:
        v = OrVariable("?", opt)
    return U.merge(v, o, n)


@comm_guard(VariableInList, (list, tuple))
def unify_walk(vil, l, U):
    """
    Unifies VIL's inner Variable to OrV(list).

    """
    v = vil.variable
    ov = OrVariable("?", l)
    return unify_walk(v, ov, U)


@comm_guard((list, tuple), (list, tuple))
def unify_walk(l1, l2, U):
    """
    Tries to unify each corresponding pair of elements from l1 and l2.

    """
    if len(l1) != len(l2):
        return False
    for x1, x2 in zip(l1, l2):
        U = unify_walk(x1, x2, U)
        if U is False:
            return False
    return U


@comm_guard(dict, dict)
def unify_walk(d1, d2, U):
    """
    Tries to unify values of corresponding keys.

    """
    for (k1, v1) in iteritems(d1):
        if k1 in d2:
            U = unify_walk(v1, d2[k1], U)
            if U is False:
                return False
    return U


@comm_guard(ANY_TYPE, ANY_TYPE)
def unify_walk(a, b, U):
    """
    Checks for the existence of the __unify_walk__ method for one of
    the objects.

    """
    if (not isinstance(a, Variable) and
            not isinstance(b, Variable) and
            hasattr(a, "__unify_walk__")):
        return a.__unify_walk__(b, U)
    else:
        return FALL_THROUGH


@comm_guard(Variable, ANY_TYPE)
def unify_walk(v, o, U):
    """
    This simply checks if the Var has an unification in U and uses it
    instead of the Var. If the Var is already its tighest unification,
    falls through.

    """
    best_v = U[v]
    if v is not best_v:
        return unify_walk(o, best_v, U)  # reverse argument order so if o is a Variable this block of code is run again
    else:
        return FALL_THROUGH  # call the next version of unify_walk that matches the type signature


################################


class FVar:

    def __init__(self, fn, *args):
        self.fn = fn
        self.args = args

    def __call__(self, u):
        return self.fn(*[unify_build(arg, u) for arg in self.args])


################################


def unify_merge(a, b, U):
    return a


@comm_guard(Variable, ANY_TYPE)
def unify_merge(v, o, U):
    return v


@comm_guard(BoundVariable, ANY_TYPE)
def unify_merge(bv, o, U):
    return bv.value


@comm_guard(VariableInList, (list, tuple))
def unify_merge(vil, l, U):
    return [unify_merge(x, x, U) for x in l]


@comm_guard((list, tuple), (list, tuple))
def unify_merge(l1, l2, U):
    return [unify_merge(x1, x2, U) for x1, x2 in zip(l1, l2)]


@comm_guard(dict, dict)
def unify_merge(d1, d2, U):
    d = d1.__class__()
    for k1, v1 in iteritems(d1):
        if k1 in d2:
            d[k1] = unify_merge(v1, d2[k1], U)
        else:
            d[k1] = unify_merge(v1, v1, U)
    for k2, v2 in iteritems(d2):
        if k2 not in d1:
            d[k2] = unify_merge(v2, v2, U)
    return d


@comm_guard(FVar, ANY_TYPE)
def unify_merge(vs, o, U):
    return vs(U)


@comm_guard(ANY_TYPE, ANY_TYPE)
def unify_merge(a, b, U):
    if (not isinstance(a, Variable) and
            not isinstance(b, Variable) and
            hasattr(a, "__unify_merge__")):
        return a.__unify_merge__(b, U)
    else:
        return FALL_THROUGH


@comm_guard(Variable, ANY_TYPE)
def unify_merge(v, o, U):
    """
    This simply checks if the Var has an unification in U and uses it
    instead of the Var. If the Var is already its tighest unification,
    falls through.

    """
    best_v = U[v]
    if v is not best_v:
        return unify_merge(o, best_v, U)  # reverse argument order so if o is a Variable this block of code is run again
    else:
        return FALL_THROUGH  # call the next version of unify_walk that matches the type signature


################################


def unify_build(x, U):
    return unify_merge(x, x, U)


################################


def unify(a, b):
    U = unify_walk(a, b, Unification())
    if not U:
        return None, False
    else:
        return unify_merge(a, b, U), U


################################


if __name__ == "__main__":

    vx = NotVariable("x", ["big", "bones"])
    vy = OrVariable("y", ["hello", "big"])
    vz = V("z")
    va = V("a")
    vl = VariableInList(vz)

    pattern1 = dict(hey=vx, ulala=va, a=1)
    pattern2 = dict(hey=vy, ulala=10, b=2)

#    pattern1 = ["hello", "big", "bones"]
#    pattern2 = vl

#    pattern1 = [vx]#, "big", "bones"]
#    pattern2 = [vy]#, vy, vz]

    U = unify_walk(pattern1, pattern2, Unification())

    if U:
        print(U[va])
        print(U[vx])
        print(U[vy])
        print(U[vz])
        print(unify_merge(pattern1, pattern2, U))
    else:
        print("no match")

    U = unify_walk((1, 2), (va, va), Unification())
    print(U[va])

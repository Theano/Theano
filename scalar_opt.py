
from scalar import *
from gof import PatternOptimizer as Pattern
from gof import utils

C = constant

# x**2 -> x*x
pow2sqr_float = Pattern((Pow, 'x', C(2.0)), (Sqr, 'x'))
pow2sqr_int = Pattern((Pow, 'x', C(2)), (Sqr, 'x'))

# x**0 -> 1
pow2one_float = Pattern((Pow, 'x', C(0.0)), C(1.0))
pow2one_int = Pattern((Pow, 'x', C(0)), C(1))

# x**1 -> x
pow2x_float = Pattern((Pow, 'x', C(1.0)), 'x')
pow2x_int = Pattern((Pow, 'x', C(1)), 'x')

# log(x**y) -> y*log(x)
logpow = Pattern((Log, (Pow, 'x', 'y')),
                 (Mul, 'y', (Log, 'x')))


class Canonizer(gof.Optimizer):
    """
    Simplification tool.

    Usage: Canonizer(main, inverse, reciprocal, mainfn, invfn, recfn, transform)
    
    * main: a suitable Op class that is commutative, associative and takes
            one to an arbitrary number of inputs, e.g. Add or Mul
    * inverse: an Op class such that inverse(main(x, y), y) == x
               e.g. Sub or Div
    * reciprocal: a function such that main(x, reciprocal(y)) == inverse(x, y)
                  e.g. Neg or Inv

    * mainfn, invfn, recfn: functions that behave just like the previous three
                            Ops, but on true scalars (e.g. their impl)

    * transform: a function that maps (numerator, denominatur) where numerator
                 and denominator are lists of Result instances, to new lists
                 where further simplifications may have been applied.

    Examples:
      add_canonizer = Canonizer(Add, Sub, Neg, lambda *inputs: sum(inputs), ...)
      mul_canonizer = Canonizer(Mul, Div, Inv, lambda *inputs: product(inputs), ...)
    
    Examples of optimizations mul_canonizer can perform:
      x / x -> 1
      (x * y) / x -> y
      x / y / x -> 1 / y
      x / y / z -> x / (y * z)
      x / (y / z) -> (x * z) / y
      (a / b) * (b / c) * (c / d) -> a / d
      (2.0 * x) / (4.0 * y) -> (0.5 * x) / y
      2 * x / 2 -> x
    """

    def __init__(self, main, inverse, reciprocal, mainfn, invfn, recfn, transform = None):
        self.main = main
        self.inverse = inverse
        self.reciprocal = reciprocal
        self.mainfn = mainfn
        self.invfn = invfn
        self.recfn = recfn
        self.neutral = mainfn()
        self.transform = transform

    def apply(self, env):

        def canonize(r):
            
#             if r in env.inputs or r in env.orphans():
#                 return
            next = env.follow(r)
            if next is None:
                return
            
            def flatten(r, nclients_check = True):
                # Collapses a tree of main/inverse/reciprocal Ops (aka Mul/Div/Inv or Add/Sub/Neg)
                # into a list of numerators and a list of denominators
                # e.g. (x*(1/y))*(x/(z/a)) aka Mul(Mul(x, (Inv, y)), Div(x, Div(z, a))) -> [x, x, a], [z, y]

                if env.edge(r):
                    return [r], []
                op = r.owner
#                 if op is None or r in env.inputs or r in env.orphans():
#                     return [r], []
                
                results = [r2.dtype == r.dtype and flatten(r2) or ([r2], []) for r2 in op.inputs]
                if isinstance(op, self.main) and (not nclients_check or env.nclients(r) == 1):
                    nums = [x[0] for x in results]
                    denums = [x[1] for x in results]
                elif isinstance(op, self.inverse) and (not nclients_check or env.nclients(r) == 1):
                    # num, denum of the second argument are added to the denum, num respectively
                    nums = [results[0][0], results[1][1]]
                    denums = [results[0][1], results[1][0]]
                elif isinstance(op, self.reciprocal) and (not nclients_check or env.nclients(r) == 1):
                    # num, denum of the sole argument are added to the denum, num respectively
                    nums = [results[0][1]]
                    denums = [results[0][0]]
                else:
                    return [r], []

                return reduce(list.__add__, nums), reduce(list.__add__, denums)

            num, denum = flatten(r, False)

            if (num, denum) == ([r], []):
                for input in (env.follow(r) or []):
                    canonize(input)
                return
#                 if r.owner is None:
#                     return
#                 else:
#                     for input in r.owner.inputs:
#                         canonize(input)
#                     return

            # Terms that are both in the num and denum lists cancel each other
            for d in list(denum):
                if d in list(num):
                    # list.remove only removes the element once
                    num.remove(d)
                    denum.remove(d)

            # We identify the constants in num and denum
            numct, num = utils.partition(lambda factor: getattr(factor, 'constant', False) and factor.data is not None, num)
            denumct, denum = utils.partition(lambda factor: getattr(factor, 'constant', False) and factor.data is not None, denum)

            # All constants in num and denum are combined into a single constant which we add to num (unless it's a neutral constant)
            v = self.invfn(self.mainfn(*[x.data for x in numct]), self.mainfn(*[x.data for x in denumct]))
            if v != self.neutral:
                num.insert(0, C(v))

            # We optimize the num and denum lists further if requested
            if self.transform is not None:
                num, denum = self.transform(env, num, denum)

            def make(factors):
                # Combines the factors using self.main (aka Mul) depending
                # on the number of elements.
                n = len(factors)
                if n == 0:
                    return None
                elif n == 1:
                    return factors[0]
                else:
                    return self.main(*factors).out

            numr, denumr = make(num), make(denum)
            
            if numr is None:
                if denumr is None:
                    # Everything cancelled each other so we're left with
                    # the neutral element.
                    new_r = Scalar(dtype = r.dtype)
                    new_r.constant = True
                    new_r.data = self.neutral
                else:
                    # There's no numerator so we use reciprocal
                    new_r = self.reciprocal(denumr).out
            else:
                if denumr is None:
                    new_r = numr
                else:
                    new_r = self.inverse(numr, denumr).out

            # Hopefully this won't complain!
            env.replace(r, new_r)

            for factor in num + denum:
                canonize(factor)

        for output in env.outputs:
            canonize(output)


def group_powers(env, num, denum):
    """
    Plugin for Canonizer: use as Canonizer(..., transform = group_powers)

    Takes num, denum such that mul(*num) / mul(*denum) is in env
    and searches for instances of exp(x) or x**y in order to group
    together powers of the same variable. Returns num2, denum2 in
    which the grouping has been done.

    Note: this function does not modify env.

    Examples:
      group_powers([x, exp(x), exp(y)], [exp(z)]) -> [x, exp(x+y-z)], []
    """

    # maps a base to the list of powers it is raised to in the
    # numerator/denominator lists.
    num_powers = {}
    denum_powers = {}

    def populate(d, seq):
        # For each instance of exp or pow in seq, removes it from seq
        # and does d[base].append(power).
        for factor in list(seq):
            op = factor.owner
            if env.edge(factor):
                continue
            if isinstance(op, Exp):
                d.setdefault('e', []).append(op.inputs[0])
                seq.remove(factor)
            elif isinstance(op, Pow):
                d.setdefault(op.inputs[0], []).append(op.inputs[1])
                seq.remove(factor)

    populate(num_powers, num)
    populate(denum_powers, denum)
    
    for x in set(num_powers.keys() + denum_powers.keys()):
        # we append base ** (num_powers[base] - denum_powers[base])
        # to the num list
        
        try: num_ys = num_powers.pop(x)
        except KeyError: num_ys = []
        
        try: denum_ys = denum_powers.pop(x)
        except KeyError: denum_ys = []

        num_r = num_ys and add(*num_ys) or C(0)
        denum_r = denum_ys and add(*denum_ys) or C(0)
        
        if x == 'e':
            num.append(exp(num_r - denum_r))
        else:
            num.append(pow(x, num_r - denum_r))

    return num, denum


def simple_factorize(env, num, denum):

    # a*b + a*c -> a*(b+c)
    # a*b + a*c + b*c -> a*(b+c) + b*c
    #                 -> a*b + (a+b)*c
    #  => a: {b, c}, b: {a, c}, c: {a, b}
    # a*c + a*d + b*c + b*d
    #  => a: {c, d}, b: {c, d}, c: {a, b}, d: {a, b}
    # (a+b*x)*(c+d) --> a*c + a*d + b*x*c + b*x*d
    #  => a: {c, d}, b: {xc, xd}, c: {a, bx}, d: {a, bx}, x: {bc, bd}
    
    pass





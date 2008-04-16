
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
            if r in env.inputs or r in env.orphans():
                return
            
            def flatten(r, nclients_check = True):
                op = r.owner
                if op is None or r in env.inputs or r in env.orphans():
                    return [r], []
                
                results = [r2.dtype == r.dtype and flatten(r2) or ([r2], []) for r2 in op.inputs]
                if isinstance(op, self.main) and (not nclients_check or env.nclients(r) == 1):
                    nums = [x[0] for x in results]
                    denums = [x[1] for x in results]
                elif isinstance(op, self.inverse) and (not nclients_check or env.nclients(r) == 1):
                    nums = [results[0][0], results[1][1]]
                    denums = [results[0][1], results[1][0]]
                elif isinstance(op, self.reciprocal) and (not nclients_check or env.nclients(r) == 1):
                    nums = [results[0][1]]
                    denums = [results[0][0]]
                else:
                    return [r], []

                return reduce(list.__add__, nums), reduce(list.__add__, denums)

            num, denum = flatten(r, False)

            if (num, denum) == ([r], []):
                if r.owner is None:
                    return
                else:
                    for input in r.owner.inputs:
                        canonize(input)
                    return
            
            for d in list(denum):
                if d in list(num):
                    num.remove(d)
                    denum.remove(d)

            numct, num = utils.partition(lambda factor: getattr(factor, 'constant', False) and factor.data is not None, num)
            denumct, denum = utils.partition(lambda factor: getattr(factor, 'constant', False) and factor.data is not None, denum)
            
            v = self.invfn(self.mainfn(*[x.data for x in numct]), self.mainfn(*[x.data for x in denumct]))
            if v != self.neutral:
                num.insert(0, C(v))

            if self.transform is not None:
                num, denum = self.transform(env, num, denum)

            def make(factors):
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
                    new_r = Scalar(dtype = r.dtype)
                    new_r.constant = True
                    new_r.data = self.neutral
                else:
                    new_r = self.reciprocal(denumr).out
            else:
                if denumr is None:
                    new_r = numr
                else:
                    new_r = self.inverse(numr, denumr).out

            env.replace(r, new_r)

            for factor in num + denum:
                canonize(factor)

        for output in env.outputs:
            canonize(output)


def group_powers(env, num, denum):

    num_powers = {}
    denum_powers = {}

    def populate(d, seq):
        for factor in list(seq):
            op = factor.owner
            if op is None or factor in env.inputs or factor in env.orphans():
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





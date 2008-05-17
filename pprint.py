
import gof


class PrinterState(gof.utils.scratchpad):
    
    def __init__(self, props = {}, **more_props):
        if isinstance(props, gof.utils.scratchpad):
            self.__update__(props)
        else:
            self.__dict__.update(props)
        self.__dict__.update(more_props)

    def clone(self, props = {}, **more_props):
        return PrinterState(self, **dict(props, **more_props))


class OperatorPrinter:

    def __init__(self, operator, precedence, assoc = 'left'):
        self.operator = operator
        self.precedence = precedence
        self.assoc = assoc

    def process(self, output, pstate):
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            raise TypeError("operator %s cannot represent a result with no associated operation" % self.operator)
        outer_precedence = getattr(pstate, 'precedence', -999999)
        outer_assoc = getattr(pstate, 'assoc', 'none')
        if outer_precedence > self.precedence:
            parenthesize = True
        #elif outer_assoc != self.assoc:
        #    parenthesize = True
        else:
            parenthesize = False
        input_strings = []
        max_i = len(node.inputs) - 1
        for i, input in enumerate(node.inputs):
            if self.assoc == 'left' and i != 0 or self.assoc == 'right' and i != max_i:
                s = pprinter.process(input, pstate.clone(precedence = self.precedence + 1e-6))
            else:
                s = pprinter.process(input, pstate.clone(precedence = self.precedence))
            input_strings.append(s)
        if len(input_strings) == 1:
            s = self.operator + input_strings[0]
        else:
            s = (" %s " % self.operator).join(input_strings)
        if parenthesize: return "(%s)" % s
        else: return s

class FunctionPrinter:

    def __init__(self, *names):
        self.names = names

    def process(self, output, pstate):
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            raise TypeError("function %s cannot represent a result with no associated operation" % self.function)
        names = self.names
        idx = node.outputs.index(output)
        name = self.names[idx]
        return "%s(%s)" % (name, ", ".join([pprinter.process(input, pstate.clone(precedence = -1000))
                                            for input in node.inputs]))


class DimShufflePrinter:

    def __p(self, new_order, pstate, r):
        if new_order != () and  new_order[0] == 'x':
            return "[%s]" % self.__p(new_order[1:], pstate, r)
        if list(new_order) == range(r.type.ndim):
            return pstate.pprinter.process(r)
        if list(new_order) == list(reversed(range(r.type.ndim))):
            return "%s.T" % pstate.pprinter.process(r)
        return "DimShuffle{%s}(%s)" % (", ".join(map(str, new_order)), pstate.pprinter.process(r))

    def process(self, r, pstate):
        if r.owner is None:
            raise TypeError("Can only print DimShuffle.")
        elif isinstance(r.owner.op, ShuffleRule):
            #print r, r.owner.op
            new_r = r.owner.op.expand(r)
            #print new_r.owner, isinstance(new_r.owner.op, ShuffleRule)
            return self.process(new_r, pstate)
        elif isinstance(r.owner.op, DimShuffle):
            ord = r.owner.op.new_order
            return self.__p(ord, pstate, r.owner.inputs[0])            
        else:
            raise TypeError("Can only print DimShuffle.")


class DefaultPrinter:

    def __init__(self):
        pass

    def process(self, r, pstate):
        pprinter = pstate.pprinter
        node = r.owner
        if node is None:
            return LeafPrinter().process(r, pstate)
        return "%s(%s)" % (str(node.op), ", ".join([pprinter.process(input, pstate.clone(precedence = -1000))
                                                    for input in node.inputs]))

class LeafPrinter:
    def process(self, r, pstate):
        if r.name in greek:
            return greek[r.name]
        else:
            return str(r)


special = dict(middle_dot = u"\u00B7",
               big_sigma = u"\u03A3")

greek = dict(alpha    = u"\u03B1",
             beta     = u"\u03B2",
             gamma    = u"\u03B3",
             delta    = u"\u03B4",
             epsilon  = u"\u03B5")


ppow = OperatorPrinter('**', 0, 'right')
pmul = OperatorPrinter('*', -1, 'either')
pdiv = OperatorPrinter('/', -1, 'left')
padd = OperatorPrinter('+', -2, 'either')
psub = OperatorPrinter('-', -2, 'left')
pdot = OperatorPrinter(special['middle_dot'], -1, 'left')
psum = OperatorPrinter(special['big_sigma']+' ', -2, 'left')
plog = FunctionPrinter('log')


class PPrinter:

    def __init__(self):
        self.printers = []

    def assign(self, condition, printer):
        if isinstance(condition, gof.Op):
            op = condition
            condition = lambda pstate, r: r.owner is not None and r.owner.op == op
        self.printers.insert(0, (condition, printer))

    def process(self, r, pstate = None):
        if pstate is None:
            pstate = PrinterState(pprinter = self)
        for condition, printer in self.printers:
            if condition(pstate, r):
                return printer.process(r, pstate)

    def clone(self):
        cp = copy(self)
        cp.printers = list(self.printers)
        return cp

    def clone_assign(self, condition, printer):
        cp = self.clone()
        cp.assign(condition, printer)
        return cp


# class ExtendedPPrinter:

#     def __init__(self, pprinter, leaf_pprinter):
#         self.pprinter = pprinter
#         self.leaf_pprinter = pprinter

#     def process(self, r, pstate = None):
        



from tensor import *
from elemwise import Sum, ShuffleRule

x, y, z = matrices('xyz')

pp = PPrinter()
pp.assign(lambda pstate, r: True, DefaultPrinter())
pp.assign(add, padd)
pp.assign(mul, pmul)
pp.assign(sub, psub)
pp.assign(neg, psub)
pp.assign(div, pdiv)
pp.assign(pow, ppow)
pp.assign(dot, pdot)
pp.assign(Sum(), FunctionPrinter('sum'))
pp.assign(sgrad, FunctionPrinter('d'))
pp.assign(lambda pstate, r: r.owner and isinstance(r.owner.op, DimShuffle), DimShufflePrinter())
pp.assign(lambda pstate, r: r.owner and isinstance(r.owner.op, ShuffleRule), DimShufflePrinter())


print pp.process(x + y * z)
print pp.process((x + y) * z)
print pp.process(x * (y * z))
print pp.process(x / (y / z) / x)
print pp.process((x ** y) ** z)
print pp.process(-x+y)
print pp.process(-x*y)
print pp.process(sum(x))
print pp.process(sum(x * 10))

a = Tensor(broadcastable=(False,False,False), dtype='float64')('alpha')
print a.type
print pp.process(DimShuffle((False,)*2, [1, 0])(x) + a)

print pp.process(x / (y * z))

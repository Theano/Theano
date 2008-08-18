
from .. import tensor as T
from .. import scalar as S
from .. import gof
from copy import copy
import sys


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


class PatternPrinter:

    def __init__(self, *patterns):
        self.patterns = []
        for pattern in patterns:
            if isinstance(pattern, str):
                self.patterns.append((pattern, ()))
            else:
                self.patterns.append((pattern[0], pattern[1:]))

    def process(self, output, pstate):
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            raise TypeError("Patterns %s cannot represent a result with no associated operation" % self.patterns)
        idx = node.outputs.index(output)
        pattern, precedences = self.patterns[idx]
        precedences += (1000,) * len(node.inputs)
        return pattern % dict((str(i), x)
                              for i, x in enumerate(pprinter.process(input, pstate.clone(precedence = precedence))
                                                    for input, precedence in zip(node.inputs, precedences)))


class FunctionPrinter:

    def __init__(self, *names):
        self.names = names

    def process(self, output, pstate):
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            raise TypeError("function %s cannot represent a result with no associated operation" % self.names)
        idx = node.outputs.index(output)
        name = self.names[idx]
        return "%s(%s)" % (name, ", ".join([pprinter.process(input, pstate.clone(precedence = -1000))
                                            for input in node.inputs]))

class MemberPrinter:

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
        input = node.inputs[0]
        return "%s.%s" % (pprinter.process(input, pstate.clone(precedence = 1000)), name)


class IgnorePrinter:

    def process(self, output, pstate):
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            raise TypeError("function %s cannot represent a result with no associated operation" % self.function)
        input = node.inputs[0]
        return "%s" % pprinter.process(input, pstate)


class DimShufflePrinter:

    def __p(self, new_order, pstate, r):
        if new_order != () and  new_order[0] == 'x':
            return "%s" % self.__p(new_order[1:], pstate, r)
#            return "[%s]" % self.__p(new_order[1:], pstate, r)
        if list(new_order) == range(r.type.ndim):
            return pstate.pprinter.process(r)
        if list(new_order) == list(reversed(range(r.type.ndim))):
            return "%s.T" % pstate.pprinter.process(r)
        return "DimShuffle{%s}(%s)" % (", ".join(map(str, new_order)), pstate.pprinter.process(r))

    def process(self, r, pstate):
        if r.owner is None:
            raise TypeError("Can only print DimShuffle.")
        elif isinstance(r.owner.op, T.DimShuffle):
            ord = r.owner.op.new_order
            return self.__p(ord, pstate, r.owner.inputs[0])            
        else:
            raise TypeError("Can only print DimShuffle.")


class SubtensorPrinter:

    def process(self, r, pstate):
        if r.owner is None:
            raise TypeError("Can only print Subtensor.")
        elif isinstance(r.owner.op, T.Subtensor):
            idxs = r.owner.op.idx_list
            inputs = list(r.owner.inputs)
            input = inputs.pop()
            sidxs = []
            inbrack_pstate = pstate.clone(precedence = -1000)
            for entry in idxs:
                if isinstance(entry, int):
                    sidxs.append(str(entry))
                elif isinstance(entry, S.Scalar):
                    sidxs.append(inbrack_pstate.pprinter.process(inputs.pop()))
                elif isinstance(entry, slice):
                    sidxs.append("%s:%s%s" % ("" if entry.start is None or entry.start == 0 else entry.start,
                                              "" if entry.stop is None or entry.stop == sys.maxint else entry.stop,
                                              "" if entry.step is None else ":%s" % entry.step))
            return "%s[%s]" % (pstate.clone(precedence = 1000).pprinter.process(input),
                               ", ".join(sidxs))
        else:
            raise TypeError("Can only print Subtensor.")


class MakeVectorPrinter:

    def process(self, r, pstate):
        if r.owner is None:
            raise TypeError("Can only print make_vector.")
        elif isinstance(r.owner.op, T.MakeVector):
            return "[%s]" % ", ".join(pstate.clone(precedence = 1000).pprinter.process(input) for input in r.owner.inputs)
        else:
            raise TypeError("Can only print make_vector.")


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

    def process_graph(self, inputs, outputs):
        strings = ["inputs: " + ", ".join(map(str, inputs))]
        pprinter = self.clone_assign(lambda pstate, r: r.name is not None and r is not current,
                                     LeafPrinter())
        for node in gof.graph.io_toposort(inputs, outputs):
            for output in node.outputs:
                if output.name is not None or output in outputs:
                    name = 'outputs[%i]' % outputs.index(output) if output.name is None else output.name
                    current = output
                    strings.append("%s = %s" % (name, pprinter.process(output)))
        return "\n".join(strings)




special = dict(middle_dot = u"\u00B7",
               big_sigma = u"\u03A3")

greek = dict(alpha    = u"\u03B1",
             beta     = u"\u03B2",
             gamma    = u"\u03B3",
             delta    = u"\u03B4",
             epsilon  = u"\u03B5")


ppow = OperatorPrinter('**', 1, 'right')
pneg = OperatorPrinter('-',  0, 'either')
pmul = OperatorPrinter('*', -1, 'either')
pdiv = OperatorPrinter('/', -1, 'left')
padd = OperatorPrinter('+', -2, 'either')
psub = OperatorPrinter('-', -2, 'left')
pdot = OperatorPrinter(special['middle_dot'], -1, 'left')
psum = OperatorPrinter(special['big_sigma']+' ', -2, 'left')

def pprinter():
    pp = PPrinter()
    pp.assign(lambda pstate, r: True, DefaultPrinter())
    pp.assign(T.add, padd)
    pp.assign(T.mul, pmul)
    pp.assign(T.sub, psub)
    pp.assign(T.neg, pneg)
    pp.assign(T.div, pdiv)
    pp.assign(T.pow, ppow)
    pp.assign(T.dot, pdot)
    pp.assign(T.Sum(), FunctionPrinter('sum'))
    pp.assign(lambda pstate, r: r.owner and isinstance(r.owner.op, T.DimShuffle), DimShufflePrinter())
    pp.assign(T.tensor_copy, IgnorePrinter())
    pp.assign(T.log, FunctionPrinter('log'))
    pp.assign(T.tanh, FunctionPrinter('tanh'))
    pp.assign(T.transpose_inplace, MemberPrinter('T'))
    pp.assign(T._abs, PatternPrinter(('|%(0)s|', -1000)))
    pp.assign(T.sgn, FunctionPrinter('sgn'))
    pp.assign(lambda pstate, r: r.owner and isinstance(r.owner.op, T.Filler) and r.owner.op.value == 0, FunctionPrinter('seros'))
    pp.assign(lambda pstate, r: r.owner and isinstance(r.owner.op, T.Filler) and r.owner.op.value == 1, FunctionPrinter('ones'))
    pp.assign(lambda pstate, r: r.owner and isinstance(r.owner.op, T.Subtensor), SubtensorPrinter())
    pp.assign(T.shape, MemberPrinter('shape'))
    pp.assign(T.fill, FunctionPrinter('fill'))
    pp.assign(T.vertical_stack, FunctionPrinter('vstack'))
    pp.assign(lambda pstate, r: r.owner and isinstance(r.owner.op, T.MakeVector), MakeVectorPrinter())
    return pp

pp = pprinter()


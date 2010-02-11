"""Pretty-printing (pprint()), the 'Print' Op, debugprint() and pydotprint().
They all allow different way to print a graph or the result of an Op in a graph(Print Op)
"""
import gof
from copy import copy
import sys,os
from theano import config
from gof import Op, Apply
from theano.gof.python25 import any

#We import the debugprint here to have all printing of graph available from this module
from theano.compile.debugmode import debugprint

class Print(Op):
    """This identity-like Op has the side effect of printing a message followed by its inputs
    when it runs. Default behaviour is to print the __str__ representation. Optionally, one 
    can pass a list of the input member functions to execute, or attributes to print.
    
    @type message: String
    @param message: string to prepend to the output
    @type attrs: list of Strings
    @param attrs: list of input node attributes or member functions to print. Functions are
    identified through callable(), executed and their return value printed.
    """
    view_map={0:[0]}
    def __init__(self,message="", attrs=("__str__",)):
        self.message=message
        self.attrs=attrs

    def make_node(self,xin):
        xout = xin.type.make_variable()
        return Apply(op = self, inputs = [xin], outputs=[xout])

    def perform(self,node,inputs,output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin
        for attr in self.attrs:
            temp = getattr(xin, attr)
            if callable(temp):
              pmsg = temp()
            else:
              psmg = temp
            print self.message, attr,'=', pmsg
            #backport
            #print self.message, attr,'=', temp() if callable(temp) else temp

    def grad(self,input,output_gradients):
        return output_gradients

    def __eq__(self, other):
        return type(self)==type(other) and self.message==other.message and self.attrs==other.attrs

    def __hash__(self):
        return hash(self.message) ^ hash(self.attrs)


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
            raise TypeError("operator %s cannot represent a variable that is not the result of an operation" % self.operator)

        ## Precedence seems to be buggy, see #249
        ## So, in doubt, we parenthesize everything.
        #outer_precedence = getattr(pstate, 'precedence', -999999)
        #outer_assoc = getattr(pstate, 'assoc', 'none')
        #if outer_precedence > self.precedence:
        #    parenthesize = True
        #else:
        #    parenthesize = False
        parenthesize = True

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
            raise TypeError("Patterns %s cannot represent a variable that is not the result of an operation" % self.patterns)
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
            raise TypeError("function %s cannot represent a variable that is not the result of an operation" % self.names)
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
            raise TypeError("function %s cannot represent a variable that is not the result of an operation" % self.function)
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
            raise TypeError("function %s cannot represent a variable that is not the result of an operation" % self.function)
        input = node.inputs[0]
        return "%s" % pprinter.process(input, pstate)


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
        elif isinstance(pstate, dict):
            pstate = PrinterState(pprinter = self, **pstate)
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

    def process_graph(self, inputs, outputs, updates = {}, display_inputs = False):
        if not isinstance(inputs, (list, tuple)): inputs = [inputs]
        if not isinstance(outputs, (list, tuple)): outputs = [outputs]
        current = None
        if display_inputs:
            strings = [(0, "inputs: " + ", ".join(map(str, list(inputs) + updates.keys())))]
        else:
            strings = []
        pprinter = self.clone_assign(lambda pstate, r: r.name is not None and r is not current,
                                     LeafPrinter())
        inv_updates = dict((b, a) for (a, b) in updates.iteritems())
        i = 1
        for node in gof.graph.io_toposort(list(inputs) + updates.keys(),
                                          list(outputs) + updates.values()):
            for output in node.outputs:
                if output in inv_updates:
                    name = str(inv_updates[output])
                    strings.append((i + 1000, "%s <- %s" % (name, pprinter.process(output))))
                    i += 1
                if output.name is not None or output in outputs:
                    if output.name is None:
                      name = 'out[%i]' % outputs.index(output)
                    else:
                      name = output.name
                    #backport
                    #name = 'out[%i]' % outputs.index(output) if output.name is None else output.name
                    current = output
                    try:
                        idx = 2000 + outputs.index(output)
                    except ValueError:
                        idx = i
                    if len(outputs) == 1 and outputs[0] is output:
                        strings.append((idx, "return %s" % pprinter.process(output)))
                    else:
                        strings.append((idx, "%s = %s" % (name, pprinter.process(output))))
                    i += 1
        strings.sort()
        return "\n".join(s[1] for s in strings)

    def __call__(self, *args):
        if len(args) == 1:
            return self.process(*args)
        elif len(args) == 2 and isinstance(args[1], (PrinterState, dict)):
            return self.process(*args)
        elif len(args) > 2:
            return self.process_graph(*args)
        else:
            raise TypeError('Not enough arguments to call.')

use_ascii = True

if use_ascii:
    special = dict(middle_dot = "\dot",
                   big_sigma = "\Sigma")

    greek = dict(alpha    = "\alpha",
                 beta     = "\beta",
                 gamma    = "\gamma",
                 delta    = "\delta",
                 epsilon  = "\epsilon")
else:

    special = dict(middle_dot = u"\u00B7",
                   big_sigma = u"\u03A3")

    greek = dict(alpha    = u"\u03B1",
                 beta     = u"\u03B2",
                 gamma    = u"\u03B3",
                 delta    = u"\u03B4",
                 epsilon  = u"\u03B5")


pprint = PPrinter()
pprint.assign(lambda pstate, r: True, DefaultPrinter())
pprint.assign(lambda pstate, r: hasattr(pstate, 'target') and pstate.target is not r and r.name is not None,
              LeafPrinter())

pp = pprint


def pydotprint(fct, outfile=os.path.join(config.compiledir,'theano.pydotprint.png')):
    """
    print to a file in png format the graph of op of a compile theano fct.

    :param fct: the theano fct returned by theano.function.
    :param outfile: the output file where to put the graph.

    In the graph, box are an Apply Node(the execution of an op) and ellipse are variable.
    If variable have name they are used as the text(if multiple var have the same name, they will be merged in the graph).
    Otherwise, if the variable is constant, we print the value and finaly we print the type + an uniq number to don't have multiple var merged.
    We print the op of the apply in the Apply box with a number that represent the toposort order of application of those Apply.

    green ellipse are input to the graph and blue ellipse are output of the graph.
    """
    import pydot as pd

    g=pd.Dot()
    var_str={}
    def var_name(var):
        if var in var_str:
            return var_str[var]
        
        if var.name is not None:
            varstr = var.name
        elif isinstance(var,gof.Constant):
            varstr = str(var.data)
        elif var in input_update and input_update[var].variable.name is not None:
            varstr = input_update[var].variable.name
        else:
            #a var id is needed as otherwise var with the same type will be merged in the graph.
            varstr = str(var.type)
        varstr += ' ' + str(len(var_str))
        var_str[var]=varstr

        return varstr

    # Update the inputs that have an update function
    input_update={}
    outputs = list(fct.maker.env.outputs)
    for i in reversed(fct.maker.expanded_inputs):
        if i.update is not None:
            input_update[outputs.pop()] = i

    for node_idx,node in enumerate(fct.maker.env.toposort()):
        astr=str(node.op).replace(':','_')+'    '+str(node_idx)

        g.add_node(pd.Node(astr,shape='box'))

        for var in node.inputs:
            varstr=var_name(var)
            if var.owner is None:
                g.add_node(pd.Node(varstr,color='green'))
            g.add_edge(pd.Edge(varstr,astr))

        for var in node.outputs:
            varstr=var_name(var)

            g.add_edge(pd.Edge(astr,varstr))
            if any([x[0]=='output' for x in var.env.clients(var)]):
        	g.add_node(pd.Node(varstr,color='blue'))

    g.write_png(outfile, prog='dot')

    print 'The output file is available at',outfile
    #from matplotlib import pyplot
    #image=pyplot.imread(outfile)
    #pyplot.imshow(image)
    #import pdb;pdb.set_trace()


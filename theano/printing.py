"""Pretty-printing (pprint()), the 'Print' Op, debugprint() and pydotprint().
They all allow different way to print a graph or the result of an Op in a graph(Print Op)
"""
import sys, os, StringIO
from copy import copy

import gof
from theano import config
from gof import Op, Apply
from theano.gof.python25 import any
from theano.compile import Function, debugmode
from theano.compile.profilemode import ProfileMode

def debugprint(obj, depth=-1, print_type=False, file=None):
    """Print a computation graph to file

    :type obj: Variable, Apply, or Function instance
    :param obj: symbolic thing to print
    :type depth: integer
    :param depth: print graph to this depth (-1 for unlimited)
    :type print_type: boolean
    :param print_type: wether to print the type of printed objects
    :type file: None, 'str', or file-like object
    :param file: print to this file ('str' means to return a string)

    :returns: string if `file` == 'str', else file arg

    Each line printed represents a Variable in the graph.
    The indentation of each line corresponds to its depth in the symbolic graph.
    The first part of the text identifies whether it is an input (if a name or type is printed)
    or the output of some Apply (in which case the Op is printed).
    The second part of the text is the memory location of the Variable.
    If print_type is True, there is a third part, containing the type of the Variable

    If a Variable is encountered multiple times in the depth-first search, it is only printed
    recursively the first time.  Later, just the Variable and its memory location are printed.

    If an Apply has multiple outputs, then a '.N' suffix will be appended to the Apply's
    identifier, to indicate which output a line corresponds to.

    """
    if file == 'str':
        _file = StringIO.StringIO()
    elif file is None:
        _file = sys.stdout
    else:
        _file = file
    done = set()
    results_to_print = []
    if isinstance(obj, gof.Variable):
        results_to_print.append(obj)
    elif isinstance(obj, gof.Apply):
        results_to_print.extend(obj.outputs)
    elif isinstance(obj, Function):
        results_to_print.extend(obj.maker.env.outputs)
    elif isinstance(obj, (list, tuple)):
        results_to_print.extend(obj)
    else:
        raise TypeError("debugprint cannot print an object of this type", obj)
    for r in results_to_print:
        debugmode.debugprint(r, depth=depth, done=done, print_type=print_type, file=_file)
    if file is _file:
        return file
    elif file=='str':
        return _file.getvalue()
    else:
        _file.flush()


def _print_fn(op, xin):
    for attr in op.attrs:
        temp = getattr(xin, attr)
        if callable(temp):
            pmsg = temp()
        else:
            pmsg = temp
        print op.message, attr,'=', pmsg

class Print(Op):
    """This identity-like Op has the side effect of printing a message followed by its inputs
    when it runs. Default behaviour is to print the __str__ representation. Optionally, one
    can pass a list of the input member functions to execute, or attributes to print.

    @type message: String
    @param message: string to prepend to the output
    @type attrs: list of Strings
    @param attrs: list of input node attributes or member functions to print. Functions are
    identified through callable(), executed and their return value printed.

    :note: WARNING. This can disable some optimization(speed and stabilization)!
    """
    view_map={0:[0]}
    def __init__(self,message="", attrs=("__str__",), global_fn=_print_fn):
        self.message=message
        self.attrs=tuple(attrs) # attrs should be a hashable iterable
        self.global_fn=global_fn

    def make_node(self,xin):
        xout = xin.type.make_variable()
        return Apply(op = self, inputs = [xin], outputs=[xout])

    def perform(self,node,inputs,output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin
        self.global_fn(self, xin)

    def grad(self,input,output_gradients):
        return output_gradients

    def __eq__(self, other):
        return type(self)==type(other) and self.message==other.message and self.attrs==other.attrs

    def __hash__(self):
        return hash(self.message) ^ hash(self.attrs)

    def __setstate__(self, dct):
        dct.setdefault('global_fn', _print_fn)
        self.__dict__.update(dct)

    def c_code_cache_version(self):
        return (1,)

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
    special = dict(middle_dot = "\\dot",
                   big_sigma = "\\Sigma")

    greek = dict(alpha    = "\\alpha",
                 beta     = "\\beta",
                 gamma    = "\\gamma",
                 delta    = "\\delta",
                 epsilon  = "\\epsilon")
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


def pydotprint(fct, outfile=os.path.join(config.compiledir,'theano.pydotprint.png'),
        compact=True, mode=None, format='png', with_ids=False):
    """
    print to a file in png format the graph of op of a compile theano fct.

    :param fct: the theano fct returned by theano.function.
    :param outfile: the output file where to put the graph.
    :param compact: if True, will remove intermediate var that don't have name.
    :param mode: if a ProfileMode, add to each Apply label (s in apply,% in apply in total op time, % in fct time)
                         Otherwise ignore it
    :param format: the file format of the output.

    In the graph, box are an Apply Node(the execution of an op) and ellipse are variable.
    If variable have name they are used as the text(if multiple var have the same name, they will be merged in the graph).
    Otherwise, if the variable is constant, we print the value and finaly we print the type + an uniq number to don't have multiple var merged.
    We print the op of the apply in the Apply box with a number that represent the toposort order of application of those Apply.
    If an Apply have more then 1 input, print add a label to the edge that in the index of the inputs.

    green ellipses are inputs to the graph
    blue ellipses are outputs of the graph
    grey ellipses are var generated by the graph that are not output and are not used.
    red ellipses are transfer to/from the gpu.
        op with those name GpuFromHost, HostFromGpu
    """
    if not isinstance(mode,ProfileMode) or not mode.fct_call.has_key(fct):
        mode=None
    try:
        import pydot as pd
    except:
        print "failed to import pydot. Yous must install pydot for this function to work."
        return

    g=pd.Dot()
    var_str={}
    all_strings = set()
    def var_name(var):
        if var in var_str:
            return var_str[var]

        if var.name is not None:
            varstr = 'name='+var.name+" "+str(var.type)
        elif isinstance(var,gof.Constant):
            dstr = 'val='+str(var.data)
            if '\n' in dstr:
                dstr = dstr[:dstr.index('\n')]
            if len(dstr) > 30:
                dstr = dstr[:27]+'...'
            varstr = '%s [%s]'% (dstr, str(var.type))
        elif var in input_update and input_update[var].variable.name is not None:
            varstr = input_update[var].variable.name+" "+str(var.type)
        else:
            #a var id is needed as otherwise var with the same type will be merged in the graph.
            varstr = str(var.type)
        if (varstr in all_strings) or with_ids:
            varstr += ' id=' + str(len(var_str))
        var_str[var]=varstr
        all_strings.add(varstr)

        return varstr
    topo = fct.maker.env.toposort()
    apply_name_cache = {}
    def apply_name(node):
        if node in apply_name_cache:
            return apply_name_cache[node]
        prof_str=''
        if mode:
            time = mode.apply_time.get((topo.index(node),node),0)
            #second, % total time in profiler, %fct time in profiler
            if mode.local_time[0]==0:
                pt=0
            else: pt=time*100/mode.local_time[0]
            if mode.fct_call[fct]==0:
                pf=0
            else: pf = time*100/mode.fct_call_time[fct]
            prof_str='   (%.3fs,%.3f%%,%.3f%%)'%(time,pt,pf)
        applystr = str(node.op).replace(':','_')
        if (applystr in all_strings) or with_ids:
            applystr = applystr+'    id='+str(topo.index(node))+prof_str
        all_strings.add(applystr)
        apply_name_cache[node] = applystr
        return applystr

    # Update the inputs that have an update function
    input_update={}
    outputs = list(fct.maker.env.outputs)
    for i in reversed(fct.maker.expanded_inputs):
        if i.update is not None:
            input_update[outputs.pop()] = i

    apply_shape='ellipse'
    var_shape='box'
    for node_idx,node in enumerate(topo):
        astr=apply_name(node)

        if node.op.__class__.__name__ in ('GpuFromHost','HostFromGpu'):
            # highlight CPU-GPU transfers to simplify optimization
            g.add_node(pd.Node(astr,color='red',shape=apply_shape))
        else:
            g.add_node(pd.Node(astr,shape=apply_shape))

        for id,var in enumerate(node.inputs):
            varstr=var_name(var)
            label=''
            if len(node.inputs)>1:
                label=str(id)
            if var.owner is None:
                g.add_node(pd.Node(varstr,color='green',shape=var_shape))
                g.add_edge(pd.Edge(varstr,astr, label=label))
            elif var.name or not compact:
                g.add_edge(pd.Edge(varstr,astr, label=label))
            else:
                #no name, so we don't make a var ellipse
                g.add_edge(pd.Edge(apply_name(var.owner),astr, label=label))


        for id,var in enumerate(node.outputs):
            varstr=var_name(var)
            out = any([x[0]=='output' for x in var.clients])
            label=''
            if len(node.outputs)>1:
                label=str(id)
            if out:
                g.add_edge(pd.Edge(astr, varstr, label=label))
                g.add_node(pd.Node(varstr,color='blue',shape=var_shape))
            elif len(var.clients)==0:
                g.add_edge(pd.Edge(astr, varstr, label=label))
                g.add_node(pd.Node(varstr,color='grey',shape=var_shape))
            elif var.name or not compact:
                g.add_edge(pd.Edge(astr, varstr, label=label))
#            else:
            #don't add egde here as it is already added from the inputs.
    if not outfile.endswith('.'+format):
        outfile+='.'+format
    g.write(outfile, prog='dot', format=format)

    print 'The output file is available at',outfile




def pydot_var(vars, outfile=os.path.join(config.compiledir,'theano.pydotprint.png'), depth = -1):
    ''' Identical to pydotprint just that it starts from a variable instead
    of a compiled function. Could be useful ? '''
    try:
        import pydot as pd
    except:
        print "failed to import pydot. Yous must install pydot for this function to work."
        return
    g=pd.Dot()
    my_list = {}
    if type(vars) not in (list,tuple):
        vars = [vars]
    var_str = {}
    def var_name(var):
        if var in var_str:
            return var_str[var]

        if var.name is not None:
            varstr = 'name='+var.name
        elif isinstance(var,gof.Constant):
            dstr = 'val='+str(var.data)
            if '\n' in dstr:
                dstr = dstr[:dstr.index('\n')]
            if len(dstr) > 30:
                dstr = dstr[:27]+'...'
            varstr = '%s [%s]'% (dstr, str(var.type))
        else:
            #a var id is needed as otherwise var with the same type will be merged in the graph.
            varstr = str(var.type)
        varstr += ' ' + str(len(var_str))
        var_str[var]=varstr

        return varstr
    def apply_name(node):
        return str(node.op).replace(':','_')

    def plot_apply(app, d):
        if d == 0:
            return
        if app in my_list:
            return
        astr = apply_name(app) + '_' + str(len(my_list.keys()))
        my_list[app] = astr
        g.add_node(pd.Node(astr, shape='box'))
        for i,nd  in enumerate(app.inputs):
            if nd not in my_list:
                varastr = var_name(nd) + '_' + str(len(my_list.keys()))
                my_list[nd] = varastr
                g.add_node(pd.Node(varastr))
            else:
                varastr = my_list[nd]
            label = ''
            if len(app.inputs)>1:
                label = str(i)
            g.add_edge(pd.Edge(varastr, astr, label = label))

        for i,nd in enumerate(app.outputs):
            if nd not in my_list:
                varastr = var_name(nd) + '_' + str(len(my_list.keys()))
                my_list[nd] = varastr
                g.add_node(pd.Node(varastr))
            else:
                varastr = my_list[nd]
            label = ''
            if len(app.outputs) > 1:
                label = str(i)
            g.add_edge(pd.Edge(astr, varastr,label = label))
        for nd in app.inputs:
            if nd.owner:
                plot_apply(nd.owner, d-1)


    for nd in vars:
        if nd.owner:
            plot_apply(nd.owner, depth)

    g.write_png(outfile, prog='dot')

    print 'The output file is available at',outfile

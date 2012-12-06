"""Pretty-printing (pprint()), the 'Print' Op, debugprint() and pydotprint().

They all allow different way to print a graph or the result of an Op
in a graph(Print Op)
"""
from copy import copy
import logging
import os
import StringIO
import sys
# Not available on all platforms
hashlib = None

import numpy
np = numpy

try:
    import pydot as pd
    if pd.find_graphviz():
        pydot_imported = True
    else:
        pydot_imported = False
except ImportError:
    pydot_imported = False

import theano
import gof
from theano import config
from gof import Op, Apply
from theano.gof.python25 import any
from theano.compile import Function, debugmode
from theano.compile.profilemode import ProfileMode

_logger = logging.getLogger("theano.printing")


def debugprint(obj, depth=-1, print_type=False,
               file=None, ids='CHAR', stop_on_name=False):
    """Print a computation graph as text to stdout or a file.

    :type obj: Variable, Apply, or Function instance
    :param obj: symbolic thing to print
    :type depth: integer
    :param depth: print graph to this depth (-1 for unlimited)
    :type print_type: boolean
    :param print_type: whether to print the type of printed objects
    :type file: None, 'str', or file-like object
    :param file: print to this file ('str' means to return a string)
    :type ids: str
    :param ids: How do we print the identifier of the variable
                id - print the python id value
                int - print integer character
                CHAR - print capital character
                "" - don't print an identifier
    :param stop_on_name: When True, if a node in the graph has a name,
                         we don't print anything below it.

    :returns: string if `file` == 'str', else file arg

    Each line printed represents a Variable in the graph.
    The indentation of lines corresponds to its depth in the symbolic graph.
    The first part of the text identifies whether it is an input
    (if a name or type is printed) or the output of some Apply (in which case
    the Op is printed).
    The second part of the text is an identifier of the Variable.
    If print_type is True, we add a part containing the type of the Variable

    If a Variable is encountered multiple times in the depth-first search,
    it is only printed recursively the first time. Later, just the Variable
    identifier is printed.

    If an Apply has multiple outputs, then a '.N' suffix will be appended
    to the Apply's identifier, to indicate which output a line corresponds to.

    """
    if file == 'str':
        _file = StringIO.StringIO()
    elif file is None:
        _file = sys.stdout
    else:
        _file = file
    done = dict()
    results_to_print = []
    order = []
    if isinstance(obj, gof.Variable):
        results_to_print.append(obj)
    elif isinstance(obj, gof.Apply):
        results_to_print.extend(obj.outputs)
    elif isinstance(obj, Function):
        results_to_print.extend(obj.maker.fgraph.outputs)
        order = obj.maker.fgraph.toposort()
    elif isinstance(obj, (list, tuple)):
        results_to_print.extend(obj)
    elif isinstance(obj, gof.FunctionGraph):
        results_to_print.extend(obj.outputs)
        order = obj.toposort()
    else:
        raise TypeError("debugprint cannot print an object of this type", obj)
    for r in results_to_print:
        debugmode.debugprint(r, depth=depth, done=done, print_type=print_type,
                             file=_file, order=order, ids=ids,
                             stop_on_name=stop_on_name)
    if file is _file:
        return file
    elif file == 'str':
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
        print op.message, attr, '=', pmsg


class Print(Op):
    """ This identity-like Op print as a side effect.

    This identity-like Op has the side effect of printing a message
    followed by its inputs when it runs. Default behaviour is to print
    the __str__ representation. Optionally, one can pass a list of the
    input member functions to execute, or attributes to print.

    @type message: String
    @param message: string to prepend to the output
    @type attrs: list of Strings
    @param attrs: list of input node attributes or member functions to print.
                  Functions are identified through callable(), executed and
                  their return value printed.

    :note: WARNING. This can disable some optimizations!
                    (speed and/or stabilization)

            Detailed explanation:
            As of 2012-06-21 the Print op is not known by any optimization.
            Setting a Print op in the middle of a pattern that is usually
            optimized out will block the optimization. for example, log(1+x)
            optimizes to log1p(x) but log(1+Print(x)) is unaffected by
            optimizations.

    """
    view_map = {0: [0]}

    def __init__(self, message="", attrs=("__str__",), global_fn=_print_fn):
        self.message = message
        self.attrs = tuple(attrs)  # attrs should be a hashable iterable
        self.global_fn = global_fn

    def make_node(self, xin):
        xout = xin.type.make_variable()
        return Apply(op=self, inputs=[xin], outputs=[xout])

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin
        self.global_fn(self, xin)

    def grad(self, input, output_gradients):
        return output_gradients

    def R_op(self, inputs, eval_points):
        return [x for x in eval_points]

    def __eq__(self, other):
        return (type(self) == type(other) and self.message == other.message
                and self.attrs == other.attrs)

    def __hash__(self):
        return hash(self.message) ^ hash(self.attrs)

    def __setstate__(self, dct):
        dct.setdefault('global_fn', _print_fn)
        self.__dict__.update(dct)

    def c_code_cache_version(self):
        return (1,)


class PrinterState(gof.utils.scratchpad):

    def __init__(self, props=None, **more_props):
        if props is None:
            props = {}
        if isinstance(props, gof.utils.scratchpad):
            self.__update__(props)
        else:
            self.__dict__.update(props)
        self.__dict__.update(more_props)

    def clone(self, props=None, **more_props):
        if props is None:
            props = {}
        return PrinterState(self, **dict(props, **more_props))


class OperatorPrinter:

    def __init__(self, operator, precedence, assoc='left'):
        self.operator = operator
        self.precedence = precedence
        self.assoc = assoc

    def process(self, output, pstate):
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            raise TypeError("operator %s cannot represent a variable that is "
                            "not the result of an operation" % self.operator)

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
            if (self.assoc == 'left' and i != 0 or self.assoc == 'right'
                and i != max_i):
                s = pprinter.process(input, pstate.clone(
                        precedence=self.precedence + 1e-6))
            else:
                s = pprinter.process(input, pstate.clone(
                        precedence=self.precedence))
            input_strings.append(s)
        if len(input_strings) == 1:
            s = self.operator + input_strings[0]
        else:
            s = (" %s " % self.operator).join(input_strings)
        if parenthesize:
            return "(%s)" % s
        else:
            return s


class PatternPrinter:

    def __init__(self, *patterns):
        self.patterns = []
        for pattern in patterns:
            if isinstance(pattern, basestring):
                self.patterns.append((pattern, ()))
            else:
                self.patterns.append((pattern[0], pattern[1:]))

    def process(self, output, pstate):
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            raise TypeError("Patterns %s cannot represent a variable that is "
                            "not the result of an operation" % self.patterns)
        idx = node.outputs.index(output)
        pattern, precedences = self.patterns[idx]
        precedences += (1000,) * len(node.inputs)

        pp_process = lambda input, precedence: pprinter.process(
            input, pstate.clone(precedence=precedence))
        d = dict((str(i), x)
                 for i, x in enumerate(pp_process(input, precedence)
                                       for input, precedence in
                                       zip(node.inputs, precedences)))
        return pattern % d


class FunctionPrinter:

    def __init__(self, *names):
        self.names = names

    def process(self, output, pstate):
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            raise TypeError("function %s cannot represent a variable that is "
                            "not the result of an operation" % self.names)
        idx = node.outputs.index(output)
        name = self.names[idx]
        return "%s(%s)" % (name, ", ".join(
                [pprinter.process(input, pstate.clone(precedence=-1000))
                 for input in node.inputs]))


class MemberPrinter:

    def __init__(self, *names):
        self.names = names

    def process(self, output, pstate):
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            raise TypeError("function %s cannot represent a variable that is"
                            " not the result of an operation" % self.function)
        names = self.names
        idx = node.outputs.index(output)
        name = self.names[idx]
        input = node.inputs[0]
        return "%s.%s" % (pprinter.process(input,
                                           pstate.clone(precedence=1000)),
                          name)


class IgnorePrinter:

    def process(self, output, pstate):
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            raise TypeError("function %s cannot represent a variable that is"
                            " not the result of an operation" % self.function)
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
        return "%s(%s)" % (str(node.op), ", ".join(
                [pprinter.process(input, pstate.clone(precedence=-1000))
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
            condition = (lambda pstate, r: r.owner is not None
                         and r.owner.op == op)
        self.printers.insert(0, (condition, printer))

    def process(self, r, pstate=None):
        if pstate is None:
            pstate = PrinterState(pprinter=self)
        elif isinstance(pstate, dict):
            pstate = PrinterState(pprinter=self, **pstate)
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

    def process_graph(self, inputs, outputs, updates=None,
                      display_inputs=False):
        if updates is None:
            updates = {}
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        current = None
        if display_inputs:
            strings = [(0, "inputs: " + ", ".join(
                        map(str, list(inputs) + updates.keys())))]
        else:
            strings = []
        pprinter = self.clone_assign(lambda pstate, r: r.name is not None
                                     and r is not current,
                                     LeafPrinter())
        inv_updates = dict((b, a) for (a, b) in updates.iteritems())
        i = 1
        for node in gof.graph.io_toposort(list(inputs) + updates.keys(),
                                          list(outputs) + updates.values()):
            for output in node.outputs:
                if output in inv_updates:
                    name = str(inv_updates[output])
                    strings.append((i + 1000, "%s <- %s" % (
                                name, pprinter.process(output))))
                    i += 1
                if output.name is not None or output in outputs:
                    if output.name is None:
                        name = 'out[%i]' % outputs.index(output)
                    else:
                        name = output.name
                    #backport
                    #name = 'out[%i]' % outputs.index(output) if output.name
                    #  is None else output.name
                    current = output
                    try:
                        idx = 2000 + outputs.index(output)
                    except ValueError:
                        idx = i
                    if len(outputs) == 1 and outputs[0] is output:
                        strings.append((idx, "return %s" %
                                        pprinter.process(output)))
                    else:
                        strings.append((idx, "%s = %s" %
                                        (name, pprinter.process(output))))
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
    special = dict(middle_dot="\\dot",
                   big_sigma="\\Sigma")

    greek = dict(alpha="\\alpha",
                 beta="\\beta",
                 gamma="\\gamma",
                 delta="\\delta",
                 epsilon="\\epsilon")
else:

    special = dict(middle_dot=u"\u00B7",
                   big_sigma=u"\u03A3")

    greek = dict(alpha=u"\u03B1",
                 beta=u"\u03B2",
                 gamma=u"\u03B3",
                 delta=u"\u03B4",
                 epsilon=u"\u03B5")


pprint = PPrinter()
pprint.assign(lambda pstate, r: True, DefaultPrinter())
pprint.assign(lambda pstate, r: hasattr(pstate, 'target')
              and pstate.target is not r and r.name is not None,
              LeafPrinter())

pp = pprint
"""
Print to the terminal a math-like expression.
"""

# colors not used: orange, amber#FFBF00, purple, pink,
# used by default: green, blue, grey, red
default_colorCodes = {'GpuFromHost': 'red',
              'HostFromGpu': 'red',
              'Scan': 'yellow',
              'Shape': 'cyan',
              'IfElse': 'magenta',
              'Elemwise': '#FFAABB',
              'Subtensor': '#FFAAFF'}


def pydotprint(fct, outfile=None,
               compact=True, format='png', with_ids=False,
               high_contrast=True, cond_highlight=None, colorCodes=None,
               max_label_size=50, scan_graphs=False,
               var_with_name_simple=False,
               print_output_file=True,
               assert_nb_all_strings=-1
               ):
    """
    Print to a file (png format) the graph of a compiled theano function's ops.

    :param fct: the theano fct returned by theano.function.
    :param outfile: the output file where to put the graph.
    :param compact: if True, will remove intermediate var that don't have name.
    :param format: the file format of the output.
    :param with_ids: Print the toposort index of the node in the node name.
                     and an index number in the variable ellipse.
    :param high_contrast: if true, the color that describes the respective
            node is filled with its corresponding color, instead of coloring
            the border
    :param colorCodes: dictionary with names of ops as keys and colors as
            values
    :param cond_highlight: Highlights a lazy if by sorrounding each of the 3
                possible categories of ops with a border. The categories
                are: ops that are on the left branch, ops that are on the
                right branch, ops that are on both branches
                As an alternative you can provide the node that represents
                the lazy if
    :param scan_graphs: if true it will plot the inner graph of each scan op
                in files with the same name as the name given for the main
                file to which the name of the scan op is concatenated and
                the index in the toposort of the scan.
                This index can be printed with the option with_ids.
    :param var_with_name_simple: If true and a variable have a name,
                we will print only the variable name.
                Otherwise, we concatenate the type to the var name.
    :param assert_nb_all_strings: Used for tests. If non-negative, assert that
                the number of unique string nodes in the dot graph is equal to
                this number. This is used in tests to verify that dot won't
                merge Theano nodes.

    In the graph, ellipses are Apply Nodes (the execution of an op)
    and boxes are variables.  If variables have names they are used as
    text (if multiple vars have the same name, they will be merged in
    the graph).  Otherwise, if the variable is constant, we print its
    value and finally we print the type + a unique number to prevent
    multiple vars from being merged.  We print the op of the apply in
    the Apply box with a number that represents the toposort order of
    application of those Apply.  If an Apply has more than 1 input, we
    label each edge between an input and the Apply node with the
    input's index.

    Green boxes are inputs variables to the graph,
    blue boxes are outputs variables of the graph,
    grey boxes are variables that are not outputs and are not used,
    red ellipses are transfers from/to the gpu (ops with names GpuFromHost,
    HostFromGpu).

    """
    if colorCodes is None:
        colorCodes = default_colorCodes

    if outfile is None:
        outfile = os.path.join(config.compiledir, 'theano.pydotprint.' +
                               config.device + '.' + format)

    if isinstance(fct, Function):
        mode = fct.maker.mode
        profile = getattr(fct, "profile", None)
        if (not isinstance(mode, ProfileMode)
            or not fct in mode.profile_stats):
            mode = None
        fct_fgraph = fct.maker.fgraph
    elif isinstance(fct, gof.FunctionGraph):
        mode = None
        profile = None
        fct_fgraph = fct
    else:
        raise ValueError(('pydotprint expects as input a theano.function or '
                         'the FunctionGraph of a function!'), fct)

    if not pydot_imported:
        raise RuntimeError("Failed to import pydot. You must install pydot"
                            " for `pydotprint` to work.")
        return

    g = pd.Dot()
    if cond_highlight is not None:
        c1 = pd.Cluster('Left')
        c2 = pd.Cluster('Right')
        c3 = pd.Cluster('Middle')
        cond = None
        for node in fct_fgraph.toposort():
            if (node.op.__class__.__name__ == 'IfElse'
                and node.op.name == cond_highlight):
                cond = node
        if cond is None:
            _logger.warn("pydotprint: cond_highlight is set but there is no"
                         " IfElse node in the graph")
            cond_highlight = None

    if cond_highlight is not None:
        def recursive_pass(x, ls):
            if not x.owner:
                return ls
            else:
                ls += [x.owner]
                for inp in x.inputs:
                    ls += recursive_pass(inp, ls)
                return ls

        left = set(recursive_pass(cond.inputs[1], []))
        right = set(recursive_pass(cond.inputs[2], []))
        middle = left.intersection(right)
        left = left.difference(middle)
        right = right.difference(middle)
        middle = list(middle)
        left = list(left)
        right = list(right)

    var_str = {}
    all_strings = set()

    def var_name(var):
        if var in var_str:
            return var_str[var]

        if var.name is not None:
            if var_with_name_simple:
                varstr = var.name
            else:
                varstr = 'name=' + var.name + " " + str(var.type)
        elif isinstance(var, gof.Constant):
            dstr = 'val=' + str(numpy.asarray(var.data))
            if '\n' in dstr:
                dstr = dstr[:dstr.index('\n')]
            varstr = '%s %s' % (dstr, str(var.type))
        elif (var in input_update
              and input_update[var].variable.name is not None):
            if var_with_name_simple:
                varstr = input_update[var].variable.name + " UPDATE"
            else:
                varstr = (input_update[var].variable.name + " UPDATE "
                          + str(var.type))
        else:
            #a var id is needed as otherwise var with the same type will be
            #merged in the graph.
            varstr = str(var.type)
        if (varstr in all_strings) or with_ids:
            idx = ' id=' + str(len(var_str))
            if len(varstr) + len(idx) > max_label_size:
                varstr = varstr[:max_label_size - 3 - len(idx)] + idx + '...'
            else:
                varstr = varstr + idx
        elif len(varstr) > max_label_size:
            varstr = varstr[:max_label_size - 3] + '...'
            idx = 1
            while varstr in all_strings:
                idx += 1
                suffix = ' id=' + str(idx)
                varstr = (varstr[:max_label_size - 3 - len(suffix)] +
                          '...' +
                          suffix)
        var_str[var] = varstr
        all_strings.add(varstr)

        return varstr
    topo = fct_fgraph.toposort()
    apply_name_cache = {}

    def apply_name(node):
        if node in apply_name_cache:
            return apply_name_cache[node]
        prof_str = ''
        if mode:
            time = mode.profile_stats[fct].apply_time.get(node, 0)
            #second, % total time in profiler, %fct time in profiler
            if mode.local_time == 0:
                pt = 0
            else:
                pt = time * 100 / mode.local_time
            if mode.profile_stats[fct].fct_callcount == 0:
                pf = 0
            else:
                pf = time * 100 / mode.profile_stats[fct].fct_call_time
            prof_str = '   (%.3fs,%.3f%%,%.3f%%)' % (time, pt, pf)
        elif profile:
            time = profile.apply_time.get(node, 0)
            #second, %fct time in profiler
            if profile.fct_callcount == 0:
                pf = 0
            else:
                pf = time * 100 / profile.fct_call_time
            prof_str = '   (%.3fs,%.3f%%)' % (time, pf)
        applystr = str(node.op).replace(':', '_')
        applystr += prof_str
        if (applystr in all_strings) or with_ids:
            idx = ' id=' + str(topo.index(node))
            if len(applystr) + len(idx) > max_label_size:
                applystr = (applystr[:max_label_size - 3 - len(idx)] + idx
                            + '...')
            else:
                applystr = applystr + idx
        elif len(applystr) > max_label_size:
            applystr = applystr[:max_label_size - 3] + '...'
            idx = 1
            while applystr in all_strings:
                idx += 1
                suffix = ' id=' + str(idx)
                applystr = (applystr[:max_label_size - 3 - len(suffix)] +
                            '...' +
                            suffix)

        all_strings.add(applystr)
        apply_name_cache[node] = applystr
        return applystr

    # Update the inputs that have an update function
    input_update = {}
    outputs = list(fct_fgraph.outputs)
    if isinstance(fct, Function):
        for i in reversed(fct.maker.expanded_inputs):
            if i.update is not None:
                input_update[outputs.pop()] = i

    apply_shape = 'ellipse'
    var_shape = 'box'
    for node_idx, node in enumerate(topo):
        astr = apply_name(node)

        use_color = None
        for opName, color in colorCodes.items():
            if opName in node.op.__class__.__name__:
                use_color = color

        if use_color is None:
            nw_node = pd.Node(astr, shape=apply_shape)
        elif high_contrast:
            nw_node = pd.Node(astr, style='filled', fillcolor=use_color,
                               shape=apply_shape)
        else:
            nw_node = pd.Node(astr, color=use_color, shape=apply_shape)
        g.add_node(nw_node)
        if cond_highlight:
            if node in middle:
                c3.add_node(nw_node)
            elif node in left:
                c1.add_node(nw_node)
            elif node in right:
                c2.add_node(nw_node)

        for id, var in enumerate(node.inputs):
            varstr = var_name(var)
            label = str(var.type)
            if len(node.inputs) > 1:
                label = str(id) + ' ' + label
            if len(label) > max_label_size:
                label = label[:max_label_size - 3] + '...'
            if var.owner is None:
                if high_contrast:
                    g.add_node(pd.Node(varstr,
                                       style='filled',
                                       fillcolor='green',
                                       shape=var_shape))
                else:
                    g.add_node(pd.Node(varstr, color='green', shape=var_shape))
                g.add_edge(pd.Edge(varstr, astr, label=label))
            elif var.name or not compact:
                g.add_edge(pd.Edge(varstr, astr, label=label))
            else:
                #no name, so we don't make a var ellipse
                g.add_edge(pd.Edge(apply_name(var.owner), astr, label=label))

        for id, var in enumerate(node.outputs):
            varstr = var_name(var)
            out = any([x[0] == 'output' for x in var.clients])
            label = str(var.type)
            if len(node.outputs) > 1:
                label = str(id) + ' ' + label
            if len(label) > max_label_size:
                label = label[:max_label_size - 3] + '...'
            if out:
                g.add_edge(pd.Edge(astr, varstr, label=label))
                if high_contrast:
                    g.add_node(pd.Node(varstr, style='filled',
                                       fillcolor='blue', shape=var_shape))
                else:
                    g.add_node(pd.Node(varstr, color='blue', shape=var_shape))
            elif len(var.clients) == 0:
                g.add_edge(pd.Edge(astr, varstr, label=label))
                if high_contrast:
                    g.add_node(pd.Node(varstr, style='filled',
                                       fillcolor='grey', shape=var_shape))
                else:
                    g.add_node(pd.Node(varstr, color='grey', shape=var_shape))
            elif var.name or not compact:
                g.add_edge(pd.Edge(astr, varstr, label=label))
#            else:
            #don't add egde here as it is already added from the inputs.

    if cond_highlight:
        g.add_subgraph(c1)
        g.add_subgraph(c2)
        g.add_subgraph(c3)

    if not outfile.endswith('.' + format):
        outfile += '.' + format

    g.write(outfile, prog='dot', format=format)
    if print_output_file:
        print 'The output file is available at', outfile

    if assert_nb_all_strings != -1:
        assert len(all_strings) == assert_nb_all_strings

    if scan_graphs:
        scan_ops = [(idx, x) for idx, x in enumerate(fct_fgraph.toposort())
                    if isinstance(x.op, theano.scan_module.scan_op.Scan)]
        path, fn = os.path.split(outfile)
        basename = '.'.join(fn.split('.')[:-1])
        # Safe way of doing things .. a file name may contain multiple .
        ext = fn[len(basename):]

        for idx, scan_op in scan_ops:
            # is there a chance that name is not defined?
            if hasattr(scan_op.op, 'name'):
                new_name = basename + '_' + scan_op.op.name + '_' + str(idx)
            else:
                new_name = basename + '_' + str(idx)
            new_name = os.path.join(path, new_name + ext)
            pydotprint(scan_op.op.fn, new_name, compact, format, with_ids,
                       high_contrast, cond_highlight, colorCodes,
                       max_label_size, scan_graphs)


def pydotprint_variables(vars,
                         outfile=None,
                         format='png',
                         depth=-1,
                         high_contrast=True, colorCodes=None,
                         max_label_size=50,
                         var_with_name_simple=False):
    ''' Identical to pydotprint just that it starts from a variable instead
    of a compiled function. Could be useful ? '''

    if colorCodes is None:
        colorCodes = default_colorCodes
    if outfile is None:
        outfile = os.path.join(config.compiledir, 'theano.pydotprint.' +
                               config.device + '.' + format)
    try:
        import pydot as pd
    except ImportError:
        print ("Failed to import pydot. You must install pydot for "
               "`pydotprint_variables` to work.")
        return
    g = pd.Dot()
    my_list = {}
    orphanes = []
    if type(vars) not in (list, tuple):
        vars = [vars]
    var_str = {}

    def var_name(var):
        if var in var_str:
            return var_str[var]

        if var.name is not None:
            if var_with_name_simple:
                varstr = var.name
            else:
                varstr = 'name=' + var.name + " " + str(var.type)
        elif isinstance(var, gof.Constant):
            dstr = 'val=' + str(var.data)
            if '\n' in dstr:
                dstr = dstr[:dstr.index('\n')]
            varstr = '%s %s' % (dstr, str(var.type))
        else:
            #a var id is needed as otherwise var with the same type will be
            #merged in the graph.
            varstr = str(var.type)

        varstr += ' ' + str(len(var_str))
        if len(varstr) > max_label_size:
            varstr = varstr[:max_label_size - 3] + '...'
        var_str[var] = varstr
        return varstr

    def apply_name(node):
        name = str(node.op).replace(':', '_')
        if len(name) > max_label_size:
            name = name[:max_label_size - 3] + '...'
        return name

    def plot_apply(app, d):
        if d == 0:
            return
        if app in my_list:
            return
        astr = apply_name(app) + '_' + str(len(my_list.keys()))
        if len(astr) > max_label_size:
            astr = astr[:max_label_size - 3] + '...'
        my_list[app] = astr

        use_color = None
        for opName, color in colorCodes.items():
            if opName in app.op.__class__.__name__:
                use_color = color

        if use_color is None:
            g.add_node(pd.Node(astr, shape='box'))
        elif high_contrast:
            g.add_node(pd.Node(astr, style='filled', fillcolor=use_color,
                               shape='box'))
        else:
            g.add_node(pd.Nonde(astr, color=use_color, shape='box'))

        for i, nd in enumerate(app.inputs):
            if nd not in my_list:
                varastr = var_name(nd) + '_' + str(len(my_list.keys()))
                if len(varastr) > max_label_size:
                    varastr = varastr[:max_label_size - 3] + '...'
                my_list[nd] = varastr
                if nd.owner is not None:
                    g.add_node(pd.Node(varastr))
                elif high_contrast:
                    g.add_node(pd.Node(varastr, style='filled',
                                        fillcolor='green'))
                else:
                    g.add_node(pd.Node(varastr, color='green'))
            else:
                varastr = my_list[nd]
            label = ''
            if len(app.inputs) > 1:
                label = str(i)
            g.add_edge(pd.Edge(varastr, astr, label=label))

        for i, nd in enumerate(app.outputs):
            if nd not in my_list:
                varastr = var_name(nd) + '_' + str(len(my_list.keys()))
                if len(varastr) > max_label_size:
                    varastr = varastr[:max_label_size - 3] + '...'
                my_list[nd] = varastr
                color = None
                if nd in vars:
                    color = 'blue'
                elif nd in orphanes:
                    color = 'gray'
                if color is None:
                    g.add_node(pd.Node(varastr))
                elif high_contrast:
                    g.add_node(pd.Node(varastr, style='filled',
                                        fillcolor=color))
                else:
                    g.add_node(pd.Node(varastr, color=color))
            else:
                varastr = my_list[nd]
            label = ''
            if len(app.outputs) > 1:
                label = str(i)
            g.add_edge(pd.Edge(astr, varastr, label=label))
        for nd in app.inputs:
            if nd.owner:
                plot_apply(nd.owner, d - 1)

    for nd in vars:
        if nd.owner:
            for k in nd.owner.outputs:
                if k not in vars:
                    orphanes.append(k)

    for nd in vars:
        if nd.owner:
            plot_apply(nd.owner, depth)

    g.write_png(outfile, prog='dot')

    print 'The output file is available at', outfile


class _TagGenerator:
    """ Class for giving abbreviated tags like to objects.
        Only really intended for internal use in order to
        implement min_informative_st """
    def __init__(self):
        self.cur_tag_number = 0

    def get_tag(self):
        rval = debugmode.char_from_number(self.cur_tag_number)

        self.cur_tag_number += 1

        return rval


def min_informative_str(obj, indent_level=0,
                        _prev_obs=None, _tag_generator=None):
    """
    Returns a string specifying to the user what obj is
    The string will print out as much of the graph as is needed
    for the whole thing to be specified in terms only of constants
    or named variables.


    Parameters
    ----------
    obj: the name to convert to a string
    indent_level: the number of tabs the tree should start printing at
                  (nested levels of the tree will get more tabs)
    _prev_obs: should only be used by min_informative_str
                    a dictionary mapping previously converted
                    objects to short tags


    Basic design philosophy
    -----------------------

    The idea behind this function is that it can be used as parts of
    command line tools for debugging or for error messages. The
    information displayed is intended to be concise and easily read by
    a human. In particular, it is intended to be informative when
    working with large graphs composed of subgraphs from several
    different people's code, as in pylearn2.

    Stopping expanding subtrees when named variables are encountered
    makes it easier to understand what is happening when a graph
    formed by composing several different graphs made by code written
    by different authors has a bug.

    An example output is:

    A. Elemwise{add_no_inplace}
        B. log_likelihood_v_given_h
        C. log_likelihood_h


    If the user is told they have a problem computing this value, it's
    obvious that either log_likelihood_h or log_likelihood_v_given_h
    has the wrong dimensionality. The variable's str object would only
    tell you that there was a problem with an
    Elemwise{add_no_inplace}. Since there are many such ops in a
    typical graph, such an error message is considerably less
    informative. Error messages based on this function should convey
    much more information about the location in the graph of the error
    while remaining succint.

    One final note: the use of capital letters to uniquely identify
    nodes within the graph is motivated by legibility. I do not use
    numbers or lower case letters since these are pretty common as
    parts of names of ops, etc. I also don't use the object's id like
    in debugprint because it gives such a long string that takes time
    to visually diff.

    """

    if _prev_obs is None:
        _prev_obs = {}

    indent = ' ' * indent_level

    if id(obj) in _prev_obs:
        tag = _prev_obs[id(obj)]

        return indent + '<' + tag + '>'

    if _tag_generator is None:
        _tag_generator = _TagGenerator()

    cur_tag = _tag_generator.get_tag()

    _prev_obs[id(obj)] = cur_tag

    if hasattr(obj, '__array__'):
        name = '<ndarray>'
    elif hasattr(obj, 'name') and obj.name is not None:
        name = obj.name
    elif hasattr(obj, 'owner') and obj.owner is not None:
        name = str(obj.owner.op)
        for ipt in obj.owner.inputs:
            name += '\n' + min_informative_str(ipt,
                    indent_level=indent_level + 1,
                    _prev_obs=_prev_obs, _tag_generator=_tag_generator)
    else:
        name = str(obj)

    prefix = cur_tag + '. '

    rval = indent + prefix + name

    return rval




def var_descriptor(obj, _prev_obs=None, _tag_generator=None):
    """
    Returns a string, with no endlines, fully specifying
    how a variable is computed. Does not include any memory
    location dependent information such as the id of a node.
    """

    global hashlib
    if hashlib is None:
        try:
            import hashlib
        except ImportError:
            raise RuntimeError("Can't run var_descriptor because hashlib is not available.")

    if _prev_obs is None:
        _prev_obs = {}

    if id(obj) in _prev_obs:
        tag = _prev_obs[id(obj)]

        return '<' + tag + '>'

    if _tag_generator is None:
        _tag_generator = _TagGenerator()

    cur_tag = _tag_generator.get_tag()

    _prev_obs[id(obj)] = cur_tag

    if hasattr(obj, '__array__'):
        # hashlib hashes only the contents of the buffer, but
        # it can have different semantics depending on the strides
        # of the ndarray
        name = '<ndarray:'
        name += 'strides=['+','.join(str(stride) for stride in obj.strides)+']'
        name += ',digest='+hashlib.md5(obj).hexdigest()+'>'
    elif hasattr(obj, 'owner') and obj.owner is not None:
        name = str(obj.owner.op) + '('
        name += ','.join(var_descriptor(ipt,
                    _prev_obs=_prev_obs, _tag_generator=_tag_generator) for ipt
                    in obj.owner.inputs)
        name += ')'
    elif hasattr(obj, 'name') and obj.name is not None:
        # Only print the name if there is no owner.
        # This way adding a name to an intermediate node can't make
        # a deeper graph get the same descriptor as a shallower one
        name = obj.name
    else:
        name = str(obj)
        if ' at 0x' in name:
            # The __str__ method is encoding the object's id in its str
            name = position_independent_str(obj)
            if ' at 0x' in name:
                print name
                assert False

    prefix = cur_tag + '='

    rval = prefix + name

    return rval

def position_independent_str(obj):
    if isinstance(obj, theano.gof.graph.Variable):
        rval = 'theano_var'
        rval += '{type='+str(obj.type)+'}'
    else:
        raise NotImplementedError()

    return rval


def hex_digest(x):
    """
    Returns a short, mostly hexadecimal hash of a numpy ndarray
    """

    global hashlib
    if hashlib is None:
        try:
            import hashlib
        except ImportError:
            raise RuntimeError("Can't run hex_digest because hashlib is not available.")
    assert isinstance(x, np.ndarray)
    rval = hashlib.md5(x.tostring()).hexdigest()
    # hex digest must be annotated with strides to avoid collisions
    # because the buffer interface only exposes the raw data, not
    # any info about the semantics of how that data should be arranged
    # into a tensor
    rval = rval + '|strides=[' + ','.join(str(stride) for stride in x.strides) + ']'
    rval = rval + '|shape=[' + ','.join(str(s) for s in x.shape) + ']'
    return rval

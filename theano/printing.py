"""Pretty-printing (pprint()), the 'Print' Op, debugprint() and pydotprint().

They all allow different way to print a graph or the result of an Op
in a graph(Print Op)
"""
from __future__ import absolute_import, print_function, division
from copy import copy
import logging
import os
import sys
import hashlib

import numpy as np
from six import string_types, integer_types, iteritems
from six.moves import StringIO, reduce

import theano
from theano import gof
from theano import config
from theano.gof import Op, Apply
from theano.compile import Function, debugmode, SharedVariable

pydot_imported = False
pydot_imported_msg = ""
try:
    # pydot-ng is a fork of pydot that is better maintained
    import pydot_ng as pd
    if pd.find_graphviz():
        pydot_imported = True
    else:
        pydot_imported_msg = "pydot-ng can't find graphviz. Install graphviz."
except ImportError:
    try:
        # fall back on pydot if necessary
        import pydot as pd
        if hasattr(pd, 'find_graphviz'):
            if pd.find_graphviz():
                pydot_imported = True
            else:
                pydot_imported_msg = "pydot can't find graphviz"
        else:
            pd.Dot.create(pd.Dot())
            pydot_imported = True
    except ImportError:
        # tests should not fail on optional dependency
        pydot_imported_msg = ("Install the python package pydot or pydot-ng."
                              " Install graphviz.")
    except Exception as e:
        pydot_imported_msg = "An error happened while importing/trying pydot: "
        pydot_imported_msg += str(e.args)


_logger = logging.getLogger("theano.printing")
VALID_ASSOC = set(['left', 'right', 'either'])


def debugprint(obj, depth=-1, print_type=False,
               file=None, ids='CHAR', stop_on_name=False,
               done=None, print_storage=False, print_clients=False,
               used_ids=None):
    """Print a computation graph as text to stdout or a file.

    :type obj: :class:`~theano.gof.Variable`, Apply, or Function instance
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
    :type done: None or dict
    :param done: A dict where we store the ids of printed node.
        Useful to have multiple call to debugprint share the same ids.
    :type print_storage: bool
    :param print_storage: If True, this will print the storage map
        for Theano functions. Combined with allow_gc=False, after the
        execution of a Theano function, we see the intermediate result.
    :type print_clients: bool
    :param print_clients: If True, this will print for Apply node that
         have more then 1 clients its clients. This help find who use
         an Apply node.
    :type used_ids: dict or None
    :param used_ids: the id to use for some object, but maybe we only
         refered to it yet.

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
    if not isinstance(depth, integer_types):
        raise Exception("depth parameter must be an int")
    if file == 'str':
        _file = StringIO()
    elif file is None:
        _file = sys.stdout
    else:
        _file = file
    if done is None:
        done = dict()
    if used_ids is None:
        used_ids = dict()
    used_ids = dict()
    results_to_print = []
    profile_list = []
    order = []  # Toposort
    smap = []  # storage_map
    if isinstance(obj, (list, tuple, set)):
        lobj = obj
    else:
        lobj = [obj]
    for obj in lobj:
        if isinstance(obj, gof.Variable):
            results_to_print.append(obj)
            profile_list.append(None)
            smap.append(None)
            order.append(None)
        elif isinstance(obj, gof.Apply):
            results_to_print.extend(obj.outputs)
            profile_list.extend([None for item in obj.outputs])
            smap.extend([None for item in obj.outputs])
            order.extend([None for item in obj.outputs])
        elif isinstance(obj, Function):
            results_to_print.extend(obj.maker.fgraph.outputs)
            profile_list.extend(
                [obj.profile for item in obj.maker.fgraph.outputs])
            if print_storage:
                smap.extend(
                    [obj.fn.storage_map for item in obj.maker.fgraph.outputs])
            else:
                smap.extend(
                    [None for item in obj.maker.fgraph.outputs])
            topo = obj.maker.fgraph.toposort()
            order.extend(
                [topo for item in obj.maker.fgraph.outputs])
        elif isinstance(obj, gof.FunctionGraph):
            results_to_print.extend(obj.outputs)
            profile_list.extend([getattr(obj, 'profile', None)
                                 for item in obj.outputs])
            smap.extend([getattr(obj, 'storage_map', None)
                         for item in obj.outputs])
            topo = obj.toposort()
            order.extend([topo for item in obj.outputs])
        elif isinstance(obj, (integer_types, float, np.ndarray)):
            print(obj)
        elif isinstance(obj, (theano.In, theano.Out)):
            results_to_print.append(obj.variable)
            profile_list.append(None)
            smap.append(None)
            order.append(None)
        else:
            raise TypeError("debugprint cannot print an object of this type",
                            obj)

    scan_ops = []
    if any([p for p in profile_list if p is not None and p.fct_callcount > 0]):
        print("""
Timing Info
-----------
--> <time> <% time> - <total time> <% total time>'

<time>         computation time for this node
<% time>       fraction of total computation time for this node
<total time>   time for this node + total times for this node's ancestors
<% total time> total time for this node over total computation time

N.B.:
* Times include the node time and the function overhead.
* <total time> and <% total time> may over-count computation times
  if inputs to a node share a common ancestor and should be viewed as a
  loose upper bound. Their intended use is to help rule out potential nodes
  to remove when optimizing a graph because their <total time> is very low.
""", file=_file)

    for r, p, s, o in zip(results_to_print, profile_list, smap, order):
        # Add the parent scan op to the list as well
        if (hasattr(r.owner, 'op') and
                isinstance(r.owner.op, theano.scan_module.scan_op.Scan)):
                    scan_ops.append(r)

        debugmode.debugprint(r, depth=depth, done=done, print_type=print_type,
                             file=_file, order=o, ids=ids,
                             scan_ops=scan_ops, stop_on_name=stop_on_name,
                             profile=p, smap=s, used_ids=used_ids,
                             print_clients=print_clients)

    if len(scan_ops) > 0:
        print("", file=_file)
        new_prefix = ' >'
        new_prefix_child = ' >'
        print("Inner graphs of the scan ops:", file=_file)

        for s in scan_ops:
            # prepare a dict which maps the scan op's inner inputs
            # to its outer inputs.
            if hasattr(s.owner.op, 'fn'):
                # If the op was compiled, print the optimized version.
                inner_inputs = s.owner.op.fn.maker.fgraph.inputs
            else:
                inner_inputs = s.owner.op.inputs
            outer_inputs = s.owner.inputs
            inner_to_outer_inputs = \
                dict([(inner_inputs[i], outer_inputs[o])
                      for i, o in
                      s.owner.op.var_mappings['outer_inp_from_inner_inp']
                      .items()])

            print("", file=_file)
            debugmode.debugprint(
                s, depth=depth, done=done,
                print_type=print_type,
                file=_file, ids=ids,
                scan_ops=scan_ops,
                stop_on_name=stop_on_name,
                scan_inner_to_outer_inputs=inner_to_outer_inputs,
                print_clients=print_clients, used_ids=used_ids)
            if hasattr(s.owner.op, 'fn'):
                # If the op was compiled, print the optimized version.
                outputs = s.owner.op.fn.maker.fgraph.outputs
            else:
                outputs = s.owner.op.outputs
            for idx, i in enumerate(outputs):

                if hasattr(i, 'owner') and hasattr(i.owner, 'op'):
                    if isinstance(i.owner.op, theano.scan_module.scan_op.Scan):
                        scan_ops.append(i)

                debugmode.debugprint(
                    r=i, prefix=new_prefix,
                    depth=depth, done=done,
                    print_type=print_type, file=_file,
                    ids=ids, stop_on_name=stop_on_name,
                    prefix_child=new_prefix_child,
                    scan_ops=scan_ops,
                    scan_inner_to_outer_inputs=inner_to_outer_inputs,
                    print_clients=print_clients, used_ids=used_ids)

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
        print(op.message, attr, '=', pmsg)


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

    __props__ = ('message', 'attrs', 'global_fn')

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

    def __setstate__(self, dct):
        dct.setdefault('global_fn', _print_fn)
        self.__dict__.update(dct)

    def c_code_cache_version(self):
        return (1,)


class PrinterState(gof.utils.scratchpad):

    def __init__(self, props=None, **more_props):
        if props is None:
            props = {}
        elif isinstance(props, gof.utils.scratchpad):
            self.__update__(props)
        else:
            self.__dict__.update(props)
        self.__dict__.update(more_props)
        # A dict from the object to print to its string
        # representation. If it is a dag and not a tree, it allow to
        # parse each node of the graph only once. They will still be
        # printed many times
        self.memo = {}


class OperatorPrinter:

    def __init__(self, operator, precedence, assoc='left'):
        self.operator = operator
        self.precedence = precedence
        self.assoc = assoc
        assert self.assoc in VALID_ASSOC

    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            raise TypeError("operator %s cannot represent a variable that is "
                            "not the result of an operation" % self.operator)

        # Precedence seems to be buggy, see #249
        # So, in doubt, we parenthesize everything.
        # outer_precedence = getattr(pstate, 'precedence', -999999)
        # outer_assoc = getattr(pstate, 'assoc', 'none')
        # if outer_precedence > self.precedence:
        #    parenthesize = True
        # else:
        #    parenthesize = False
        parenthesize = True

        input_strings = []
        max_i = len(node.inputs) - 1
        for i, input in enumerate(node.inputs):
            new_precedence = self.precedence
            if (self.assoc == 'left' and i != 0 or self.assoc == 'right' and
                    i != max_i):
                new_precedence += 1e-6
            try:
                old_precedence = getattr(pstate, 'precedence', None)
                pstate.precedence = new_precedence
                s = pprinter.process(input, pstate)
            finally:
                pstate.precedence = old_precedence
            input_strings.append(s)
        if len(input_strings) == 1:
            s = self.operator + input_strings[0]
        else:
            s = (" %s " % self.operator).join(input_strings)
        if parenthesize:
            r = "(%s)" % s
        else:
            r = s
        pstate.memo[output] = r
        return r


class PatternPrinter:

    def __init__(self, *patterns):
        self.patterns = []
        for pattern in patterns:
            if isinstance(pattern, string_types):
                self.patterns.append((pattern, ()))
            else:
                self.patterns.append((pattern[0], pattern[1:]))

    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            raise TypeError("Patterns %s cannot represent a variable that is "
                            "not the result of an operation" % self.patterns)
        idx = node.outputs.index(output)
        pattern, precedences = self.patterns[idx]
        precedences += (1000,) * len(node.inputs)

        def pp_process(input, new_precedence):
            try:
                old_precedence = getattr(pstate, 'precedence', None)
                pstate.precedence = new_precedence
                r = pprinter.process(input, pstate)
            finally:
                pstate.precedence = old_precedence

            return r

        d = dict((str(i), x)
                 for i, x in enumerate(pp_process(input, precedence)
                                       for input, precedence in
                                       zip(node.inputs, precedences)))
        r = pattern % d
        pstate.memo[output] = r
        return r


class FunctionPrinter:

    def __init__(self, *names):
        self.names = names

    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            raise TypeError("function %s cannot represent a variable that is "
                            "not the result of an operation" % self.names)
        idx = node.outputs.index(output)
        name = self.names[idx]
        new_precedence = -1000
        try:
            old_precedence = getattr(pstate, 'precedence', None)
            pstate.precedence = new_precedence
            r = "%s(%s)" % (name, ", ".join(
                [pprinter.process(input, pstate) for input in node.inputs]))
        finally:
            pstate.precedence = old_precedence

        pstate.memo[output] = r
        return r


class IgnorePrinter:

    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            raise TypeError("function %s cannot represent a variable that is"
                            " not the result of an operation" % self.function)
        input = node.inputs[0]
        r = "%s" % pprinter.process(input, pstate)
        pstate.memo[output] = r
        return r


class LeafPrinter:
    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        if output.name in greek:
            r = greek[output.name]
        else:
            r = str(output)
        pstate.memo[output] = r
        return r
leaf_printer = LeafPrinter()


class DefaultPrinter:
    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]
        pprinter = pstate.pprinter
        node = output.owner
        if node is None:
            return leaf_printer.process(output, pstate)
        new_precedence = -1000
        try:
            old_precedence = getattr(pstate, 'precedence', None)
            pstate.precedence = new_precedence
            r = "%s(%s)" % (str(node.op), ", ".join(
                [pprinter.process(input, pstate)
                 for input in node.inputs]))
        finally:
            pstate.precedence = old_precedence

        pstate.memo[output] = r
        return r
default_printer = DefaultPrinter()


class PPrinter:
    def __init__(self):
        self.printers = []
        self.printers_dict = {}

    def assign(self, condition, printer):
        # condition can be a class or an instance of an Op.
        if isinstance(condition, (gof.Op, type)):
            self.printers_dict[condition] = printer
            return
        self.printers.insert(0, (condition, printer))

    def process(self, r, pstate=None):
        if pstate is None:
            pstate = PrinterState(pprinter=self)
        elif isinstance(pstate, dict):
            pstate = PrinterState(pprinter=self, **pstate)
        if getattr(r, 'owner', None) is not None:
            if r.owner.op in self.printers_dict:
                return self.printers_dict[r.owner.op].process(r, pstate)
            if type(r.owner.op) in self.printers_dict:
                return self.printers_dict[type(r.owner.op)].process(r, pstate)
        for condition, printer in self.printers:
            if condition(pstate, r):
                return printer.process(r, pstate)

    def clone(self):
        cp = copy(self)
        cp.printers = list(self.printers)
        cp.printers_dict = dict(self.printers_dict)
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
        pprinter = self.clone_assign(lambda pstate, r: r.name is not None and
                                     r is not current, leaf_printer)
        inv_updates = dict((b, a) for (a, b) in iteritems(updates))
        i = 1
        for node in gof.graph.io_toposort(list(inputs) + updates.keys(),
                                          list(outputs) +
                                          updates.values()):
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
                    # backport
                    # name = 'out[%i]' % outputs.index(output) if output.name
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
pprint.assign(lambda pstate, r: True, default_printer)

pp = pprint
"""
Print to the terminal a math-like expression.
"""

# colors not used: orange, amber#FFBF00, purple, pink,
# used by default: green, blue, grey, red
default_colorCodes = {'GpuFromHost': 'red',
                      'HostFromGpu': 'red',
                      'Scan': 'yellow',
                      'Shape': 'brown',
                      'IfElse': 'magenta',
                      'Elemwise': '#FFAABB',  # dark pink
                      'Subtensor': '#FFAAFF',  # purple
                      'Alloc': '#FFAA22',  # orange
                      'Output': 'blue'}


def pydotprint(fct, outfile=None,
               compact=True, format='png', with_ids=False,
               high_contrast=True, cond_highlight=None, colorCodes=None,
               max_label_size=70, scan_graphs=False,
               var_with_name_simple=False,
               print_output_file=True,
               return_image=False,
               ):
    """Print to a file the graph of a compiled theano function's ops. Supports
    all pydot output formats, including png and svg.

    :param fct: a compiled Theano function, a Variable, an Apply or
                a list of Variable.
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
    :param return_image: If True, it will create the image and return it.
        Useful to display the image in ipython notebook.

        .. code-block:: python

            import theano
            v = theano.tensor.vector()
            from IPython.display import SVG
            SVG(theano.printing.pydotprint(v*2, return_image=True,
                                           format='svg'))

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

    Variable color code::
        - Cyan boxes are SharedVariable, inputs and/or outputs) of the graph,
        - Green boxes are inputs variables to the graph,
        - Blue boxes are outputs variables of the graph,
        - Grey boxes are variables that are not outputs and are not used,

    Default apply node code::
        - Red ellipses are transfers from/to the gpu
        - Yellow are scan node
        - Brown are shape node
        - Magenta are IfElse node
        - Dark pink are elemwise node
        - Purple are subtensor
        - Orange are alloc node

    For edges, they are black by default. If a node returns a view
    of an input, we put the corresponding input edge in blue. If it
    returns a destroyed input, we put the corresponding edge in red.

    .. note::

        Since October 20th, 2014, this print the inner function of all
        scan separately after the top level debugprint output.

    """
    if colorCodes is None:
        colorCodes = default_colorCodes

    if outfile is None:
        outfile = os.path.join(config.compiledir, 'theano.pydotprint.' +
                               config.device + '.' + format)

    if isinstance(fct, Function):
        profile = getattr(fct, "profile", None)
        outputs = fct.maker.fgraph.outputs
        topo = fct.maker.fgraph.toposort()
    elif isinstance(fct, gof.FunctionGraph):
        profile = None
        outputs = fct.outputs
        topo = fct.toposort()
    else:
        if isinstance(fct, gof.Variable):
            fct = [fct]
        elif isinstance(fct, gof.Apply):
            fct = fct.outputs
        assert isinstance(fct, (list, tuple))
        assert all(isinstance(v, gof.Variable) for v in fct)
        fct = gof.FunctionGraph(inputs=gof.graph.inputs(fct),
                                outputs=fct)
        profile = None
        outputs = fct.outputs
        topo = fct.toposort()
    if not pydot_imported:
        raise RuntimeError("Failed to import pydot. You must install graphviz"
                           " and either pydot or pydot-ng for "
                           "`pydotprint` to work.",
                           pydot_imported_msg)

    g = pd.Dot()

    if cond_highlight is not None:
        c1 = pd.Cluster('Left')
        c2 = pd.Cluster('Right')
        c3 = pd.Cluster('Middle')
        cond = None
        for node in topo:
            if (node.op.__class__.__name__ == 'IfElse' and
                    node.op.name == cond_highlight):
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
    var_id = {}
    all_strings = set()

    def var_name(var):
        if var in var_str:
            return var_str[var], var_id[var]

        if var.name is not None:
            if var_with_name_simple:
                varstr = var.name
            else:
                varstr = 'name=' + var.name + " " + str(var.type)
        elif isinstance(var, gof.Constant):
            dstr = 'val=' + str(np.asarray(var.data))
            if '\n' in dstr:
                dstr = dstr[:dstr.index('\n')]
            varstr = '%s %s' % (dstr, str(var.type))
        elif (var in input_update and
              input_update[var].name is not None):
            varstr = input_update[var].name
            if not var_with_name_simple:
                varstr += str(var.type)
        else:
            # a var id is needed as otherwise var with the same type will be
            # merged in the graph.
            varstr = str(var.type)
        if len(varstr) > max_label_size:
            varstr = varstr[:max_label_size - 3] + '...'
        var_str[var] = varstr
        var_id[var] = str(id(var))

        all_strings.add(varstr)

        return varstr, var_id[var]

    apply_name_cache = {}
    apply_name_id = {}

    def apply_name(node):
        if node in apply_name_cache:
            return apply_name_cache[node], apply_name_id[node]
        prof_str = ''
        if profile:
            time = profile.apply_time.get(node, 0)
            # second, %fct time in profiler
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
                applystr = (applystr[:max_label_size - 3 - len(idx)] + idx +
                            '...')
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
        apply_name_id[node] = str(id(node))

        return applystr, apply_name_id[node]

    # Update the inputs that have an update function
    input_update = {}
    reverse_input_update = {}
    # Here outputs can be the original list, as we should not change
    # it, we must copy it.
    outputs = list(outputs)
    if isinstance(fct, Function):
        function_inputs = zip(fct.maker.expanded_inputs, fct.maker.fgraph.inputs)
        for i, fg_ii in reversed(list(function_inputs)):
            if i.update is not None:
                k = outputs.pop()
                # Use the fgaph.inputs as it isn't the same as maker.inputs
                input_update[k] = fg_ii
                reverse_input_update[fg_ii] = k

    apply_shape = 'ellipse'
    var_shape = 'box'
    for node_idx, node in enumerate(topo):
        astr, aid = apply_name(node)

        use_color = None
        for opName, color in iteritems(colorCodes):
            if opName in node.op.__class__.__name__:
                use_color = color

        if use_color is None:
            nw_node = pd.Node(aid, label=astr, shape=apply_shape)
        elif high_contrast:
            nw_node = pd.Node(aid, label=astr, style='filled',
                              fillcolor=use_color,
                              shape=apply_shape)
        else:
            nw_node = pd.Node(aid, label=astr,
                              color=use_color, shape=apply_shape)
        g.add_node(nw_node)
        if cond_highlight:
            if node in middle:
                c3.add_node(nw_node)
            elif node in left:
                c1.add_node(nw_node)
            elif node in right:
                c2.add_node(nw_node)

        for idx, var in enumerate(node.inputs):
            varstr, varid = var_name(var)
            label = ""
            if len(node.inputs) > 1:
                label = str(idx)
            param = {}
            if label:
                param['label'] = label
            if hasattr(node.op, 'view_map') and idx in reduce(
                    list.__add__, node.op.view_map.values(), []):
                    param['color'] = colorCodes['Output']
            elif hasattr(node.op, 'destroy_map') and idx in reduce(
                    list.__add__, node.op.destroy_map.values(), []):
                        param['color'] = 'red'
            if var.owner is None:
                color = 'green'
                if isinstance(var, SharedVariable):
                    # Input are green, output blue
                    # Mixing blue and green give cyan! (input and output var)
                    color = "cyan"
                if high_contrast:
                    g.add_node(pd.Node(varid,
                                       style='filled',
                                       fillcolor=color,
                                       label=varstr,
                                       shape=var_shape))
                else:
                    g.add_node(pd.Node(varid,
                                       color=color,
                                       label=varstr,
                                       shape=var_shape))
                g.add_edge(pd.Edge(varid, aid, **param))
            elif var.name or not compact or var in outputs:
                g.add_edge(pd.Edge(varid, aid, **param))
            else:
                # no name, so we don't make a var ellipse
                if label:
                    label += " "
                label += str(var.type)
                if len(label) > max_label_size:
                    label = label[:max_label_size - 3] + '...'
                param['label'] = label
                g.add_edge(pd.Edge(apply_name(var.owner)[1], aid, **param))

        for idx, var in enumerate(node.outputs):
            varstr, varid = var_name(var)
            out = var in outputs
            label = ""
            if len(node.outputs) > 1:
                label = str(idx)
            if len(label) > max_label_size:
                label = label[:max_label_size - 3] + '...'
            param = {}
            if label:
                param['label'] = label
            if out or var in input_update:
                g.add_edge(pd.Edge(aid, varid, **param))
                if high_contrast:
                    g.add_node(pd.Node(varid, style='filled',
                                       label=varstr,
                                       fillcolor=colorCodes['Output'], shape=var_shape))
                else:
                    g.add_node(pd.Node(varid, color=colorCodes['Output'],
                                       label=varstr,
                                       shape=var_shape))
            elif len(var.clients) == 0:
                g.add_edge(pd.Edge(aid, varid, **param))
                # grey mean that output var isn't used
                if high_contrast:
                    g.add_node(pd.Node(varid, style='filled',
                                       label=varstr,
                                       fillcolor='grey', shape=var_shape))
                else:
                    g.add_node(pd.Node(varid, label=varstr,
                                       color='grey', shape=var_shape))
            elif var.name or not compact:
                if not(not compact):
                    if label:
                        label += " "
                    label += str(var.type)
                    if len(label) > max_label_size:
                        label = label[:max_label_size - 3] + '...'
                    param['label'] = label
                g.add_edge(pd.Edge(aid, varid, **param))
                g.add_node(pd.Node(varid, shape=var_shape, label=varstr))
#            else:
            # don't add egde here as it is already added from the inputs.

    # The var that represent updates, must be linked to the input var.
    for sha, up in input_update.items():
        _, shaid = var_name(sha)
        _, upid = var_name(up)
        g.add_edge(pd.Edge(shaid, upid, label="UPDATE", color=colorCodes['Output']))

    if cond_highlight:
        g.add_subgraph(c1)
        g.add_subgraph(c2)
        g.add_subgraph(c3)

    if not outfile.endswith('.' + format):
        outfile += '.' + format

    if scan_graphs:
        scan_ops = [(idx, x) for idx, x in enumerate(topo)
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
            if hasattr(scan_op.op, 'fn'):
                to_print = scan_op.op.fn
            else:
                to_print = scan_op.op.outputs
            pydotprint(to_print, new_name, compact, format, with_ids,
                       high_contrast, cond_highlight, colorCodes,
                       max_label_size, scan_graphs)

    if return_image:
        return g.create(prog='dot', format=format)
    else:
        try:
            g.write(outfile, prog='dot', format=format)
        except pd.InvocationException:
            # based on https://github.com/Theano/Theano/issues/2988
            version = getattr(pd, '__version__', "")
            if version and [int(n) for n in version.split(".")] < [1, 0, 28]:
                raise Exception("Old version of pydot detected, which can "
                                "cause issues with pydot printing. Try "
                                "upgrading pydot version to a newer one")
            raise

        if print_output_file:
            print('The output file is available at', outfile)


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
            name += '\n'
            name += min_informative_str(ipt,
                                        indent_level=indent_level + 1,
                                        _prev_obs=_prev_obs,
                                        _tag_generator=_tag_generator)
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
        name += 'strides=[' + ','.join(str(stride)
                                       for stride in obj.strides) + ']'
        name += ',digest=' + hashlib.md5(obj).hexdigest() + '>'
    elif hasattr(obj, 'owner') and obj.owner is not None:
        name = str(obj.owner.op) + '('
        name += ','.join(var_descriptor(ipt,
                                        _prev_obs=_prev_obs,
                                        _tag_generator=_tag_generator)
                         for ipt in obj.owner.inputs)
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
                print(name)
                assert False

    prefix = cur_tag + '='

    rval = prefix + name

    return rval


def position_independent_str(obj):
    if isinstance(obj, theano.gof.graph.Variable):
        rval = 'theano_var'
        rval += '{type=' + str(obj.type) + '}'
    else:
        raise NotImplementedError()

    return rval


def hex_digest(x):
    """
    Returns a short, mostly hexadecimal hash of a numpy ndarray
    """
    assert isinstance(x, np.ndarray)
    rval = hashlib.md5(x.tostring()).hexdigest()
    # hex digest must be annotated with strides to avoid collisions
    # because the buffer interface only exposes the raw data, not
    # any info about the semantics of how that data should be arranged
    # into a tensor
    rval = rval + '|strides=[' + ','.join(str(stride)
                                          for stride in x.strides) + ']'
    rval = rval + '|shape=[' + ','.join(str(s) for s in x.shape) + ']'
    return rval

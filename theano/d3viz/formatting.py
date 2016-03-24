"""Functions for formatting Theano compute graphs.

Author: Christof Angermueller <cangermueller@gmail.com>
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import os
from functools import reduce
from six import iteritems, itervalues

import theano
from theano import gof
from theano.compile.profilemode import ProfileMode
from theano.compile import Function
from theano.compile import builders

pydot_imported = False
try:
    # pydot-ng is a fork of pydot that is better maintained
    import pydot_ng as pd
    if pd.find_graphviz():
        pydot_imported = True
except ImportError:
    try:
        # fall back on pydot if necessary
        import pydot as pd
        if pd.find_graphviz():
            pydot_imported = True
    except ImportError:
        pass  # tests should not fail on optional dependency


class PyDotFormatter(object):
    """Create `pydot` graph object from Theano function.

    Parameters
    ----------
    compact : bool
        if True, will remove intermediate variables without name.

    Attributes
    ----------
    node_colors : dict
        Color table of node types.
    apply_colors : dict
        Color table of apply nodes.
    shapes : dict
        Shape table of node types.
    """

    def __init__(self, compact=True):
        """Construct PyDotFormatter object."""
        if not pydot_imported:
            raise ImportError('Failed to import pydot. You must install pydot'
                              ' and graphviz for `PyDotFormatter` to work.')

        self.compact = compact
        self.node_colors = {'input': 'limegreen',
                            'constant_input': 'SpringGreen',
                            'shared_input': 'YellowGreen',
                            'output': 'dodgerblue',
                            'unused': 'lightgrey'}
        self.apply_colors = {'GpuFromHost': 'red',
                             'HostFromGpu': 'red',
                             'Scan': 'yellow',
                             'Shape': 'cyan',
                             'IfElse': 'magenta',
                             'Elemwise': '#FFAABB',  # dark pink
                             'Subtensor': '#FFAAFF',  # purple
                             'Alloc': '#FFAA22'}  # orange
        self.shapes = {'input': 'box',
                       'output': 'box',
                       'apply': 'ellipse'}
        self.__node_prefix = 'n'

    def __add_node(self, node):
        """Add new node to node list and return unique id.

        Parameters
        ----------
        node : Theano graph node
            Apply node, tensor variable, or shared variable in compute graph.

        Returns
        -------
        str
            Unique node id.
        """
        assert node not in self.__nodes
        _id = '%s%d' % (self.__node_prefix, len(self.__nodes) + 1)
        self.__nodes[node] = _id
        return _id

    def __node_id(self, node):
        """Return unique node id.

        Parameters
        ----------
        node : Theano graph node
            Apply node, tensor variable, or shared variable in compute graph.

        Returns
        -------
        str
            Unique node id.
        """
        if node in self.__nodes:
            return self.__nodes[node]
        else:
            return self.__add_node(node)

    def __call__(self, fct, graph=None):
        """Create pydot graph from function.

        Parameters
        ----------
        fct : theano.compile.function_module.Function
            A compiled Theano function, variable, apply or a list of variables.
        graph: pydot.Dot
            `pydot` graph to which nodes are added. Creates new one if
            undefined.

        Returns
        -------
        pydot.Dot
            Pydot graph of `fct`
        """
        if graph is None:
            graph = pd.Dot()

        self.__nodes = {}

        profile = None
        if isinstance(fct, Function):
            mode = fct.maker.mode
            if (not isinstance(mode, ProfileMode) or
                    fct not in mode.profile_stats):
                mode = None
            if mode:
                profile = mode.profile_stats[fct]
            else:
                profile = getattr(fct, "profile", None)
            outputs = fct.maker.fgraph.outputs
            topo = fct.maker.fgraph.toposort()
        elif isinstance(fct, gof.FunctionGraph):
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
            outputs = fct.outputs
            topo = fct.toposort()
        outputs = list(outputs)

        # Loop over apply nodes
        for node in topo:
            nparams = {}
            __node_id = self.__node_id(node)
            nparams['name'] = __node_id
            nparams['label'] = apply_label(node)
            nparams['profile'] = apply_profile(node, profile)
            nparams['node_type'] = 'apply'
            nparams['apply_op'] = nparams['label']
            nparams['shape'] = self.shapes['apply']

            use_color = None
            for opName, color in iteritems(self.apply_colors):
                if opName in node.op.__class__.__name__:
                    use_color = color
            if use_color:
                nparams['style'] = 'filled'
                nparams['fillcolor'] = use_color
                nparams['type'] = 'colored'

            pd_node = dict_to_pdnode(nparams)
            graph.add_node(pd_node)

            # Loop over input nodes
            for id, var in enumerate(node.inputs):
                var_id = self.__node_id(var.owner if var.owner else var)
                if var.owner is None:
                    vparams = {'name': var_id,
                               'label': var_label(var),
                               'node_type': 'input'}
                    if isinstance(var, gof.Constant):
                        vparams['node_type'] = 'constant_input'
                    elif isinstance(var, theano.tensor.sharedvar.
                                    TensorSharedVariable):
                        vparams['node_type'] = 'shared_input'
                    vparams['dtype'] = type_to_str(var.type)
                    vparams['tag'] = var_tag(var)
                    vparams['style'] = 'filled'
                    vparams['fillcolor'] = self.node_colors[
                        vparams['node_type']]
                    vparams['shape'] = self.shapes['input']
                    pd_var = dict_to_pdnode(vparams)
                    graph.add_node(pd_var)

                edge_params = {}
                if hasattr(node.op, 'view_map') and \
                        id in reduce(list.__add__,
                                     itervalues(node.op.view_map), []):
                    edge_params['color'] = self.node_colors['output']
                elif hasattr(node.op, 'destroy_map') and \
                        id in reduce(list.__add__,
                                     itervalues(node.op.destroy_map), []):
                    edge_params['color'] = 'red'

                edge_label = vparams['dtype']
                if len(node.inputs) > 1:
                    edge_label = str(id) + ' ' + edge_label
                pdedge = pd.Edge(var_id, __node_id, label=edge_label,
                                 **edge_params)
                graph.add_edge(pdedge)

            # Loop over output nodes
            for id, var in enumerate(node.outputs):
                var_id = self.__node_id(var)

                if var in outputs or len(var.clients) == 0:
                    vparams = {'name': var_id,
                               'label': var_label(var),
                               'node_type': 'output',
                               'dtype': type_to_str(var.type),
                               'tag': var_tag(var),
                               'style': 'filled'}
                    if len(var.clients) == 0:
                        vparams['fillcolor'] = self.node_colors['unused']
                    else:
                        vparams['fillcolor'] = self.node_colors['output']
                    vparams['shape'] = self.shapes['output']
                    pd_var = dict_to_pdnode(vparams)
                    graph.add_node(pd_var)

                    graph.add_edge(pd.Edge(__node_id, var_id,
                                           label=vparams['dtype']))
                elif var.name or not self.compact:
                    graph.add_edge(pd.Edge(__node_id, var_id,
                                           label=vparams['dtype']))

            # Create sub-graph for OpFromGraph nodes
            if isinstance(node.op, builders.OpFromGraph):
                subgraph = pd.Cluster(__node_id)
                gf = PyDotFormatter()
                # Use different node prefix for sub-graphs
                gf.__node_prefix = __node_id
                gf(node.op.fn, subgraph)
                graph.add_subgraph(subgraph)
                pd_node.get_attributes()['subg'] = subgraph.get_name()

                def format_map(m):
                    return str([list(x) for x in m])

                # Inputs mapping
                ext_inputs = [self.__node_id(x) for x in node.inputs]
                int_inputs = [gf.__node_id(x)
                              for x in node.op.fn.maker.fgraph.inputs]
                assert len(ext_inputs) == len(int_inputs)
                h = format_map(zip(ext_inputs, int_inputs))
                pd_node.get_attributes()['subg_map_inputs'] = h

                # Outputs mapping
                ext_outputs = []
                for n in topo:
                    for i in n.inputs:
                        h = i.owner if i.owner else i
                        if h is node:
                            ext_outputs.append(self.__node_id(n))
                int_outputs = node.op.fn.maker.fgraph.outputs
                int_outputs = [gf.__node_id(x) for x in int_outputs]
                assert len(ext_outputs) == len(int_outputs)
                h = format_map(zip(int_outputs, ext_outputs))
                pd_node.get_attributes()['subg_map_outputs'] = h

        return graph


def var_label(var, precision=3):
    """Return label of variable node."""
    if var.name is not None:
        return var.name
    elif isinstance(var, gof.Constant):
        h = np.asarray(var.data)
        is_const = False
        if h.ndim == 0:
            is_const = True
            h = np.array([h])
        dstr = np.array2string(h, precision=precision)
        if '\n' in dstr:
            dstr = dstr[:dstr.index('\n')]
        if is_const:
            dstr = dstr.replace('[', '').replace(']', '')
        return dstr
    else:
        return type_to_str(var.type)


def var_tag(var):
    """Parse tag attribute of variable node."""
    tag = var.tag
    if hasattr(tag, 'trace') and len(tag.trace) and len(tag.trace[0]) == 4:
        path, line, _, src = tag.trace[0]
        path = os.path.basename(path)
        path = path.replace('<', '')
        path = path.replace('>', '')
        src = src.encode()
        return [path, line, src]
    else:
        return None


def apply_label(node):
    """Return label of apply node."""
    return node.op.__class__.__name__


def apply_profile(node, profile):
    """Return apply profiling informaton."""
    if not profile or profile.fct_call_time == 0:
        return None
    time = profile.apply_time.get(node, 0)
    call_time = profile.fct_call_time
    return [time, call_time]


def broadcastable_to_str(b):
    """Return string representation of broadcastable."""
    named_broadcastable = {(): 'scalar',
                           (False,): 'vector',
                           (False, True): 'col',
                           (True, False): 'row',
                           (False, False): 'matrix'}
    if b in named_broadcastable:
        bcast = named_broadcastable[b]
    else:
        bcast = ''
    return bcast


def dtype_to_char(dtype):
    """Return character that represents data type."""
    dtype_char = {
        'complex64': 'c',
        'complex128': 'z',
        'float32': 'f',
        'float64': 'd',
        'int8': 'b',
        'int16': 'w',
        'int32': 'i',
        'int64': 'l'}
    if dtype in dtype_char:
        return dtype_char[dtype]
    else:
        return 'X'


def type_to_str(t):
    """Return str of variable type."""
    if not hasattr(t, 'broadcastable'):
        return str(t)
    s = broadcastable_to_str(t.broadcastable)
    if s == '':
        s = str(t.dtype)
    else:
        s = dtype_to_char(t.dtype) + s
    return s


def dict_to_pdnode(d):
    """Create pydot node from dict."""
    e = dict()
    for k, v in iteritems(d):
        if v is not None:
            if isinstance(v, list):
                v = '\t'.join([str(x) for x in v])
            else:
                v = str(v)
            v = str(v)
            v = v.replace('"', '\'')
            e[k] = v
    pynode = pd.Node(**e)
    return pynode

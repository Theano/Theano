import numpy as np
import logging
import re
import os.path as pt

try:
    import pydot as pd
    if pd.find_graphviz():
        pydot_imported = True
    else:
        pydot_imported = False
except ImportError:
    pydot_imported = False

import theano
from theano import gof
from theano.compile.profilemode import ProfileMode
from theano.compile import Function
from theano.compile import builders

_logger = logging.getLogger("theano.printing")



class GraphFormatter(object):

    def __init__(self):
        """Formatter class

        :param compact: if True, will remove intermediate var that don't have name.
        :param with_ids: Print the toposort index of the node in the node name.
            and an index number in the variable ellipse.
        :param scan_graphs: if true it will plot the inner graph of each scan op
            in files with the same name as the name given for the main
            file to which the name of the scan op is concatenated and
            the index in the toposort of the scan.
            This index can be printed with the option with_ids.
        :param var_with_name_simple: If true and a variable have a name,
            we will print only the variable name.
            Otherwise, we concatenate the type to the var name.
        """
        self.compact = True
        self.with_ids = False
        self.scan_graphs = True
        self.var_with_name_simple = False
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
        self.max_label_size = 70
        self.node_prefix = 'n'

    def add_node(self, node):
        assert not node in self.nodes
        _id = '%s%d' % (self.node_prefix, len(self.nodes) + 1)
        self.nodes[node] = _id
        return _id

    def node_id(self, node):
        if node in self.nodes:
            return self.nodes[node]
        else:
            return self.add_node(node)

    def to_pydot(self, fct, graph=None):
        """Create pydot graph from function.

        :param fct: a compiled Theano function, a Variable, an Apply or
                    a list of Variable.
        """
        if graph is None:
            graph = pd.Dot()

        self.nodes = {}
        self.var_str = {}
        self.all_strings = set()
        self.apply_name_cache = {}

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
        if not pydot_imported:
            raise RuntimeError("Failed to import pydot. You must install pydot"
                               " for `pydotprint` to work.")


        # Update the inputs that have an update function
        self.input_update = {}
        # Here outputs can be the original list, as we should not change
        # it, we must copy it.
        outputs = list(outputs)
        if isinstance(fct, Function):
            for i in reversed(fct.maker.expanded_inputs):
                if i.update is not None:
                    self.input_update[outputs.pop()] = i

        for node in topo:
            nparams = {}
            node_id = self.node_id(node)
            nparams['name'] = node_id
            nparams['label'] = apply_label(node)
            nparams['profile'] = self.apply_profile(node, profile)
            nparams['node_type'] = 'apply'
            nparams['apply_op'] = apply_op(node)

            use_color = None
            for opName, color in self.apply_colors.items():
                if opName in node.op.__class__.__name__:
                    use_color = color
            if use_color:
                nparams['style'] = 'filled'
                nparams['fillcolor'] = use_color
                nparams['type'] = 'colored'

            pd_node = self.dict_to_pdnode(nparams)
            graph.add_node(pd_node)

            for id, var in enumerate(node.inputs):
                var_id = self.node_id(var.owner if var.owner else var)
                if var.owner is None:
                    vparams = {}
                    vparams['name'] = var_id
                    vparams['label'] = self.var_label(var)
                    vparams['node_type'] = 'input'
                    if isinstance(var, gof.Constant):
                        vparams['node_type'] = 'constant_input'
                    elif isinstance(var, theano.tensor.sharedvar.TensorSharedVariable):
                        vparams['node_type'] = 'shared_input'
                    vparams['dtype'] = type_to_str(var.type)
                    vparams['tag'] = self.var_tag(var)
                    vparams['style'] = 'filled'
                    vparams['fillcolor'] = self.node_colors[vparams['node_type']]
                    vparams['shape'] = 'ellipse'
                    pd_var = self.dict_to_pdnode(vparams)
                    graph.add_node(pd_var)

                edge_params = {}
                if hasattr(node.op, 'view_map') and id in reduce(list.__add__, node.op.view_map.values(), []):
                    edge_params['color'] = self.node_colors['output']
                elif hasattr(node.op, 'destroy_map') and id in reduce(
                        list.__add__, node.op.destroy_map.values(), []):
                            edge_params['color'] = 'red'

                edge_label = vparams['dtype']
                if len(node.inputs) > 1:
                    edge_label = str(id) + ' ' + edge_label
                pdedge = pd.Edge(var_id, node_id, label=edge_label,
                                 **edge_params)
                graph.add_edge(pdedge)

            for id, var in enumerate(node.outputs):
                var_id = self.node_id(var)

                if var in outputs or len(var.clients) == 0:
                    vparams = {}
                    vparams['name'] = var_id
                    vparams['label'] = self.var_label(var)
                    vparams['node_type'] = 'output'
                    vparams['dtype'] = type_to_str(var.type)
                    vparams['tag'] = self.var_tag(var)
                    vparams['style'] = 'filled'
                    if len(var.clients) == 0:
                        vparams['fillcolor'] = self.node_colors['unused']
                    else:
                        vparams['fillcolor'] = self.node_colors['output']
                    vparams['shape'] = 'box'
                    pd_var = self.dict_to_pdnode(vparams)
                    graph.add_node(pd_var)

                    graph.add_edge(pd.Edge(node_id, var_id, label=vparams['dtype']))
                elif var.name or not self.compact:
                    graph.add_edge(pd.Edge(node_id, var_id, label=vparams['dtype']))

            if isinstance(node.op, builders.OpFromGraph):
                subgraph = pd.Cluster(node_id)
                gf = GraphFormatter()
                gf.node_prefix = node_id
                gf.to_pydot(node.op.fn, subgraph)
                graph.add_subgraph(subgraph)
                pd_node.get_attributes()['subg'] = subgraph.get_name()

                def format_map(m):
                    return str([list(x) for x in m])

                # Inputs mapping
                ext_inputs = [self.node_id(x) for x in node.inputs]
                int_inputs = [gf.node_id(x) for x in node.op.fn.maker.fgraph.inputs]
                assert len(ext_inputs) == len(int_inputs)
                h = format_map(zip(ext_inputs, int_inputs))
                pd_node.get_attributes()['subg_map_inputs'] = h

                # Outputs mapping
                ext_outputs = []
                for n in topo:
                    for i in n.inputs:
                        h = i.owner if i.owner else i
                        if h is node:
                            ext_outputs.append(self.node_id(n))
                int_outputs = node.op.fn.maker.fgraph.outputs
                int_outputs = [gf.node_id(x) for x in int_outputs]
                assert len(ext_outputs) == len(int_outputs)
                h = format_map(zip(int_outputs, ext_outputs))
                pd_node.get_attributes()['subg_map_outputs'] = h

        return graph

    def dict_to_pdnode(self, d):
        e = dict()
        for k, v in d.items():
            if v is not None:
                v = str(v)
                v = v.replace('"', '')
                e[k] = v
        pynode = pd.Node(**e)
        return pynode

    def var_label(self, var, precision=3):
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
        elif (var in self.input_update and
            self.input_update[var].variable.name is not None):
            return self.input_update[var].variable.name + " UPDATE"
        else:
            return type_to_str(var.type)

    def var_tag(self, var):
        tag = var.tag
        if hasattr(tag, 'trace') and len(tag.trace) and len(tag.trace[0]) == 4:
            path, line, _, src = tag.trace[0]
            path = pt.basename(path)
            src = src.encode()
            return (path, line, src)
        else:
            return None

    def apply_profile(self, node, profile):
        if not profile or profile.fct_call_time == 0:
            return None
        time = profile.apply_time.get(node, 0)
        call_time = profile.fct_call_time
        return [time, call_time]


def broadcastable_to_str(b):
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
    if not hasattr(t, 'broadcastable'):
        return str(t)
    s = broadcastable_to_str(t.broadcastable)
    if s == '':
        s = str(t.dtype)
    else:
        s = dtype_to_char(t.dtype) + s
    return s


def apply_label(node):
    return node.op.__class__.__name__


def apply_op(node):
    name = str(node.op).replace(':', '_')
    name = re.sub('^<', '', name)
    name = re.sub('>$', '', name)
    return name

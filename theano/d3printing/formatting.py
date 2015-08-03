import numpy as np
import logging
import re

try:
    import pydot as pd
    if pd.find_graphviz():
        pydot_imported = True
    else:
        pydot_imported = False
except ImportError:
    pydot_imported = False

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
        :param colorCodes: dictionary with names of ops as keys and colors as
            values
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
        self.colorCodes = {'GpuFromHost': 'red',
                            'HostFromGpu': 'red',
                            'Scan': 'yellow',
                            'Shape': 'cyan',
                            'IfElse': 'magenta',
                            'Elemwise': '#FFAABB',  # dark pink
                            'Subtensor': '#FFAAFF',  # purple
                            'Alloc': '#FFAA22'}  # orange
        self.node_colors = {'input': 'limegreen',
                            'output': 'dodgerblue',
                            'unused': 'lightgrey'
                            }
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

        if isinstance(fct, Function):
            mode = fct.maker.mode
            profile = getattr(fct, "profile", None)
            if (not isinstance(mode, ProfileMode) or
                    fct not in mode.profile_stats):
                    mode = None
            outputs = fct.maker.fgraph.outputs
            topo = fct.maker.fgraph.toposort()
        elif isinstance(fct, gof.FunctionGraph):
            mode = None
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
            mode = None
            profile = None
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

        apply_shape = 'ellipse'
        var_shape = 'box'
        for node in topo:
            node_id = self.node_id(node)
            astr, aprof = self.apply_name(node, fct, topo, mode, profile)

            use_color = None
            for opName, color in self.colorCodes.items():
                if opName in node.op.__class__.__name__:
                    use_color = color

            if use_color is None:
                pd_node = pd.Node(node_id, label=astr, shape=apply_shape, profile=aprof)
            else:
                pd_node = pd.Node(node_id, label=astr, style='filled', fillcolor=use_color,
                                shape=apply_shape, type='colored', profile=aprof)
            graph.add_node(pd_node)

            def make_node(_id, label, **kwargs):
                t = {k:v for k,v in kwargs.items() if v is not None}
                return pd.Node(_id, label=label, **t)

            for id, var in enumerate(node.inputs):
                edge_params = {}
                if hasattr(node.op, 'view_map') and id in reduce(list.__add__, node.op.view_map.values(), []):
                    edge_params['color'] = self.node_colors['output']
                elif hasattr(node.op, 'destroy_map') and id in reduce(
                        list.__add__, node.op.destroy_map.values(), []):
                            edge_params['color'] = 'red'

                edge_label = str(var.type)
                if len(node.inputs) > 1:
                    edge_label = str(id) + ' ' + edge_label

                var_id = self.node_id(var.owner if var.owner else var)

                if var.owner is None:
                    var_name = self.var_name(var)
                    n = make_node(var_id, var_name, style='filled',
                                  fillcolor=self.node_colors['input'],
                                  shape=var_shape, profile=aprof)
                    graph.add_node(n)

                graph.add_edge(pd.Edge(var_id, node_id, label=edge_label, **edge_params))

            for id, var in enumerate(node.outputs):
                var_id = self.node_id(var)
                var_name = self.var_name(var)
                edge_label = str(var.type)

                if var in outputs:
                    n = make_node(var_id, var_name, style='filled',
                                  fillcolor=self.node_colors['output'],
                                  shape=var_shape, profile=aprof)
                    graph.add_node(n)
                    graph.add_edge(pd.Edge(node_id, var_id, label=edge_label))
                elif len(var.clients) == 0:
                    n = make_node(var_id, var_name, style='filled',
                                    fillcolor=self.node_colors['unused'],
                                    shape=var_shape, profile=aprof)
                    graph.add_node(n)
                    graph.add_edge(pd.Edge(node_id, var_id, label=edge_label))
                elif var.name or not self.compact:
                    graph.add_edge(pd.Edge(node_id, var_id, label=edge_label))

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



    def var_name(self, var):
        if var in self.var_str:
            return self.var_str[var]

        if var.name is not None:
            if self.var_with_name_simple:
                varstr = var.name
            else:
                varstr = 'name=' + var.name + " " + str(var.type)
        elif isinstance(var, gof.Constant):
            dstr = 'val=' + str(np.asarray(var.data))
            if '\n' in dstr:
                dstr = dstr[:dstr.index('\n')]
            varstr = '%s %s' % (dstr, str(var.type))
        elif (var in self.input_update and
            self.input_update[var].variable.name is not None):
            if self.var_with_name_simple:
                varstr = self.input_update[var].variable.name + " UPDATE"
            else:
                varstr = (self.input_update[var].variable.name + " UPDATE " +
                        str(var.type))
        else:
            # a var id is needed as otherwise var with the same type will be
            # merged in the graph.
            varstr = str(var.type)
        if (varstr in self.all_strings) or self.with_ids:
            idx = ' id=' + str(len(self.var_str))
            if len(varstr) + len(idx) > self.max_label_size:
                varstr = varstr[:self.max_label_size - 3 - len(idx)] + idx + '...'
            else:
                varstr = varstr + idx
        elif len(varstr) > self.max_label_size:
            varstr = varstr[:self.max_label_size - 3] + '...'
            idx = 1
            while varstr in self.all_strings:
                idx += 1
                suffix = ' id=' + str(idx)
                varstr = (varstr[:self.max_label_size - 3 - len(suffix)] +
                        '...' +
                        suffix)
        self.var_str[var] = varstr
        self.all_strings.add(varstr)

        return varstr

    def apply_name(self, node, fct, topo, mode=None, profile=None):
        if node in self.apply_name_cache:
            return self.apply_name_cache[node]

        prof = ''
        if mode:
            profile = mode.profile_stats[fct]
        if profile:
            time = profile.apply_time.get(node, 0)
            call_time = profile.fct_call_time
            prof = str([time, call_time])

        applystr = str(node.op).replace(':', '_')
        applystr = re.sub('^<', '', applystr)
        applystr = re.sub('>$', '', applystr)
        if (applystr in self.all_strings) or self.with_ids:
            idx = ' id=' + str(topo.index(node))
            if len(applystr) + len(idx) > self.max_label_size:
                applystr = (applystr[:self.max_label_size - 3 - len(idx)] + idx +
                            '...')
            else:
                applystr = applystr + idx
        elif len(applystr) > self.max_label_size:
            applystr = applystr[:self.max_label_size - 3] + '...'
            idx = 1
            while applystr in self.all_string:
                idx += 1
                suffix = ' id=' + str(idx)
                applystr = (applystr[:self.max_label_size - 3 - len(suffix)] +
                            '...' +
                            suffix)

        self.all_strings.add(applystr)
        rv = (applystr, prof)
        self.apply_name_cache[node] = rv
        return rv

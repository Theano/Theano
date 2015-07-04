import numpy as np
import logging

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

_logger = logging.getLogger("theano.printing")


class GraphFormatter(object):

    def __init__(self):
        """Formatter class

        :param compact: if True, will remove intermediate var that don't have name.
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
        """
        self.compact = True
        self.with_ids = False
        self.high_contrast = True
        self.cond_highlight = None
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
        self.node_colors = {'input': 'green',
                            'output': 'blue',
                            'unused': 'grey'
                            }
        self.max_label_size = 70


    def to_pydot(self, fct):
        """Create pydot graph from function.

        :param fct: a compiled Theano function, a Variable, an Apply or
                    a list of Variable.
        """

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

        g = pd.Dot()

        cond_highlight = self.cond_highlight
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

        self.var_str = {}
        self.all_strings = set()
        self.apply_name_cache = {}


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
        for node_idx, node in enumerate(topo):
            astr = self.apply_name(node, fct, topo, mode, profile)

            use_color = None
            for opName, color in self.colorCodes.items():
                if opName in node.op.__class__.__name__:
                    use_color = color

            if use_color is None:
                nw_node = pd.Node(astr, shape=apply_shape)
            elif self.high_contrast:
                nw_node = pd.Node(astr, style='filled', fillcolor=use_color,
                                shape=apply_shape)
            else:
                nw_node = pd.Node(astr, color=use_color, shape=apply_shape)
            g.add_node(nw_node)
            if self.cond_highlight:
                if node in middle:
                    c3.add_node(nw_node)
                elif node in left:
                    c1.add_node(nw_node)
                elif node in right:
                    c2.add_node(nw_node)

            for id, var in enumerate(node.inputs):
                varstr = self.var_name(var)
                label = str(var.type)
                if len(node.inputs) > 1:
                    label = str(id) + ' ' + label
                if len(label) >self.max_label_size:
                    label = label[:self.max_label_size - 3] + '...'
                param = {}
                if hasattr(node.op, 'view_map') and id in reduce(
                        list.__add__, node.op.view_map.values(), []):
                        param['color'] = self.node_colors['output']
                elif hasattr(node.op, 'destroy_map') and id in reduce(
                        list.__add__, node.op.destroy_map.values(), []):
                            param['color'] = 'red'
                if var.owner is None:
                    if self.high_contrast:
                        g.add_node(pd.Node(varstr,
                                        style='filled',
                                        fillcolor=self.node_colors['input'],
                                        shape=var_shape))
                    else:
                        g.add_node(pd.Node(varstr, color=self.node_colors['input'],
                                        shape=var_shape))
                    g.add_edge(pd.Edge(varstr, astr, label=label, **param))
                elif var.name or not self.compact:
                    g.add_edge(pd.Edge(varstr, astr, label=label, **param))
                else:
                    # no name, so we don't make a var ellipse
                    name = self.apply_name(var.owner, fct, topo, mode, profile)
                    g.add_edge(pd.Edge(name, astr,
                                       label=label, **param))

            for id, var in enumerate(node.outputs):
                varstr = self.var_name(var)
                out = var in outputs
                label = str(var.type)
                if len(node.outputs) > 1:
                    label = str(id) + ' ' + label
                if len(label) >self.max_label_size:
                    label = label[:self.max_label_size - 3] + '...'
                if out:
                    g.add_edge(pd.Edge(astr, varstr, label=label))
                    if self.high_contrast:
                        g.add_node(pd.Node(varstr, style='filled',
                                        fillcolor=self.node_colors['output'],
                                        shape=var_shape))
                    else:
                        g.add_node(pd.Node(varstr, color=self.node_colors['output'],
                                           shape=var_shape))
                elif len(var.clients) == 0:
                    g.add_edge(pd.Edge(astr, varstr, label=label))
                    if self.high_contrast:
                        g.add_node(pd.Node(varstr, style='filled',
                                        fillcolor=self.node_colors['unused'],
                                        shape=var_shape))
                    else:
                        g.add_node(pd.Node(varstr, color=self.node_colors['unused'],
                                        shape=var_shape))
                elif var.name or not self.compact:
                    g.add_edge(pd.Edge(astr, varstr, label=label))


        if self.cond_highlight:
            g.add_subgraph(c1)
            g.add_subgraph(c2)
            g.add_subgraph(c3)

        return g



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

        prof_str = ''
        if mode:
            time = mode.profile_stats[fct].apply_time.get(node, 0)
            # second, % total time in profiler, %fct time in profiler
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
            # second, %fct time in profiler
            if profile.fct_callcount == 0:
                pf = 0
            else:
                pf = time * 100 / profile.fct_call_time
            prof_str = '   (%.3fs,%.3f%%)' % (time, pf)
        applystr = str(node.op).replace(':', '_')
        applystr += prof_str
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
        self.apply_name_cache[node] = applystr
        return applystr

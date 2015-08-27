"""Dynamic visualization of Theano graphs.

Author: Christof Angermueller <cangermueller@gmail.com>
"""

import os
import os.path as pt
import shutil
import re

from formatting import PyDotFormatter

__path__ = pt.dirname(pt.realpath(__file__))


def replace_patterns(x, replace):
    """Replace patterns `replace` in x."""
    for from_, to in replace.items():
        x = x.replace(str(from_), str(to))
    return x


def escape_quotes(s):
    """Escape quotes in string."""
    s = re.sub(r'''(['"])''', r'\\\1', s)
    return s


def d3viz(fct, outfile, copy_deps=True, *args, **kwargs):
    """Create HTML file with dynamic visualizing of a Theano function graph.

    :param fct: A compiled Theano function, variable, apply or a list of
                variables.
    :param outfile: The output HTML file.
    :param copy_deps: Copy javascript and CSS dependencies to output directory.
    :param *args, **kwargs: Arguments passed to PyDotFormatter.

    In the HTML file, the whole graph or single nodes can be moved by drag and
    drop. Zooming is possible via the mouse wheel. Detailed information about
    nodes and edges are displayed via mouse-over events. Node labels can be
    edited by selecting Edit from the context menu.

    Input nodes are colored in green, output nodes in blue. Apply nodes are
    ellipses, and colored depending on the type of operation they perform. Red
    ellipses are transfers from/to the GPU (ops with names GpuFromHost,
    HostFromGpu).

    Edges are black by default. If a node returns a view of an
    input, the input edge will be blue. If it returns a destroyed input, the
    edge will be red.
    """

    # Create DOT graph
    formatter = PyDotFormatter(*args, **kwargs)
    graph = formatter(fct)
    dot_graph = escape_quotes(graph.create_dot()).replace('\n', '')

    # Create output directory if not existing
    outdir = pt.dirname(outfile)
    if not pt.exists(outdir):
        os.makedirs(outdir)

    # Read template HTML file
    template_file = pt.join(__path__, 'html', 'template.html')
    f = open(template_file)
    template = f.read()
    f.close()

    # Copy dependencies to output directory
    src_deps = __path__
    if copy_deps:
        dst_deps = 'd3viz'
        for d in ['js', 'css']:
            dep = pt.join(outdir, dst_deps, d)
            if not pt.exists(dep):
                shutil.copytree(pt.join(src_deps, d), dep)
    else:
        dst_deps = src_deps

    # Replace patterns in template
    replace = {
        '%% JS_DIR %%': pt.join(dst_deps, 'js'),
        '%% CSS_DIR %%': pt.join(dst_deps, 'css'),
        '%% DOT_GRAPH %%': pt.basename(dot_graph),
    }
    html = replace_patterns(template, replace)

    # Write HTML file
    if outfile is not None:
        f = open(outfile, 'w')
        f.write(html)
        f.close()


def d3write(fct, path, *args, **kwargs):
    """Convert Theano graph to pydot graph and write to file."""
    formatter = PyDotFormatter(*args, **kwargs)
    graph = formatter(fct)
    graph.write_dot(path)

"""Dynamic visualization of Theano graphs."""

# Authors: Christof Angermueller <cangermueller@gmail.com>

import os
import os.path as pt
from glob import glob
import shutil

from formatting import GraphFormatter

__path__ = pt.dirname(pt.realpath(__file__))


def replace_patterns(x, replace):
    """ Replaces patterns defined by `replace` in x."""
    for from_, to in replace.items():
        x = x.replace(str(from_), str(to))
    return x


def d3write(fct, path, *args, **kwargs):
    # Convert theano graph to pydot graph and write to file
    gf = GraphFormatter(*args, **kwargs)
    g = gf.to_pydot(fct)
    g.write_dot(path)


def d3print(fct, outfile,  *args, **kwargs):
    """Creates dynamic graph visualization using d3.js javascript library.

    :param fct: A compiled Theano function, variable, apply or a list of
                variables
    :param outfile: The output file
    """

    outdir = pt.dirname(outfile)
    if not pt.exists(outdir):
        os.makedirs(outdir)

    # Create dot file
    dot_file = pt.splitext(outfile)[0] + '.dot'
    d3write(fct, dot_file, *args, **kwargs)

    # Read template HTML file
    template_file = pt.join(__path__, 'html', 'template.html')
    f = open(template_file)
    template = f.read()
    f.close()

    # Replace patterns in template
    replace = {
        '%% JS_DIR %%': pt.join(__path__, 'js'),
        '%% CSS_DIR %%': pt.join(__path__, 'css'),
        '%% DOT_FILE %%': pt.basename(dot_file),
    }
    html = replace_patterns(template, replace)

    # Write HTML file
    if outfile is not None:
        f = open(outfile, 'w')
        f.write(html)
        f.close()

from __future__ import absolute_import, print_function, division

import os
import sys

import versioneer

# Ensure Theano's path is correctly added
theano_path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(os.path.abspath(theano_path))

import theano  # Import theano to resolve the undefined variable issue

# Import necessary modules for extensions
try:
    import sphinx_rtd_theme
except ImportError:
    sphinx_rtd_theme = None

# General configuration
# ---------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
    'sphinx.ext.linkcode'
]

todo_include_todos = True
napoleon_google_docstring = False
napoleon_include_special_with_doc = False

# Add imgmath or pngmath if available
try:
    from sphinx.ext import imgmath
    extensions.append('sphinx.ext.imgmath')
except ImportError:
    try:
        from sphinx.ext import pngmath
        extensions.append('sphinx.ext.pngmath')
    except ImportError:
        pass

# Paths and configurations
templates_path = ['.templates']
source_suffix = '.txt'
master_doc = 'index'
project = 'Theano'
copyright = '2008--2020, LISA lab'

# Version information
_curpath = os.getcwd()
os.chdir(os.path.dirname(os.path.dirname(__file__)))
release = versioneer.get_version()
version = '.'.join(release.split('.')[:2])
os.chdir(_curpath)
del _curpath

today_fmt = '%B %d, %Y'
exclude_dirs = ['images', 'scripts', 'sandbox']
pygments_style = 'sphinx'

# Options for HTML output
# -----------------------

if os.environ.get('READTHEDOCS') != 'True' and sphinx_rtd_theme:
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
    html_theme = 'sphinx_rtd_theme'

def setup(app):
    app.add_css_file("fix_rtd.css")

html_logo = 'images/theano_logo_allwhite_210x70.png'
html_static_path = ['.static', 'images', 'library/d3viz/examples']
html_last_updated_fmt = '%b %d, %Y'
html_use_smartypants = True
htmlhelp_basename = 'theanodoc'

# Options for linkcode extension
# ------------------------------
def linkcode_resolve(domain, info):
    def find_source():
        obj = sys.modules[info['module']]
        for part in info['fullname'].split('.'):
            obj = getattr(obj, part)
        import inspect
        import os
        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(theano.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != 'py' or not info['module']:
        return None
    try:
        filename = 'theano/%s#L%d-L%d' % find_source()
    except Exception:
        filename = info['module'].replace('.', '/') + '.py'
    import subprocess
    tag = subprocess.Popen(['git', 'rev-parse', 'HEAD'],
                           stdout=subprocess.PIPE,
                           universal_newlines=True).communicate()[0].strip()
    return "https://github.com/Theano/theano/blob/%s/%s" % (tag, filename)

# Options for LaTeX output
# ------------------------

latex_elements = {
    'pointsize': '11pt',
}

latex_documents = [
    ('index', 'theano.tex', 'Theano Documentation',
     'LISA lab, University of Montreal', 'manual'),
]

latex_logo = 'images/theano_logo_allblue_200x46.png'


from docutils import nodes
import epydoc.docwriter.xlink as xlink

def role_fn(name, rawtext, text, lineno, inliner,
            options={}, content=[]):
    node = nodes.reference(rawtext, text, refuri = "http://pylearn.org/theano/wiki/%s" % text)
    return [node], []

def setup(app):
#    print dir(xlink)
    app.add_role("wiki", role_fn)
